"""
STEP 1: dense 기준 모델 학습 (ERM, ResNet-50)

목표는 '좋은 모델'이 아니라 '배경 shortcut 에 의존하는 모델'이다.
  통과 기준: 평균 정확도는 높고(대략 90% 이상), WGA 는 그보다 한참 낮을 것.
  WGA 가 평균과 비슷하면 shortcut 이 약한 것이므로 --no_augment 등으로 다시 학습.

모델 선택은 validation '평균' 정확도 기준으로 한다.
WGA 기준으로 고르면 shortcut 이 약한 체크포인트가 뽑혀 실험 전제가 무너진다.

사용법
  python train_erm.py --wb_root /path/to/waterbird_complete95_forest2water2 \
      --out_dir runs/erm_seed0 --seed 0
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from waterbirds import (WaterbirdsDataset, build_transform, evaluate_groups,
                        format_eval)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(device, pretrained=True):
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
    except AttributeError:                      # 구버전 torchvision
        model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wb_root", required=True)
    ap.add_argument("--out_dir", default="runs/erm")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--no_augment", action="store_true",
                    help="학습 증강을 끈다 (shortcut 의존을 강화하고 싶을 때)")
    ap.add_argument("--amp", action="store_true", help="mixed precision 사용")
    ap.add_argument("--subset", type=int, default=0,
                    help="파이프라인 점검용. >0 이면 각 split 을 이 장수로 제한")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_tf = build_transform(train=True, augment=not args.no_augment)
    eval_tf = build_transform(train=False)

    train_set = WaterbirdsDataset(args.wb_root, "train", transform=train_tf)
    val_set = WaterbirdsDataset(args.wb_root, "val", transform=eval_tf)
    test_set = WaterbirdsDataset(args.wb_root, "test", transform=eval_tf)
    if args.subset > 0:
        from torch.utils.data import Subset
        def head(ds, n):
            sub = Subset(ds, list(range(min(n, len(ds)))))
            sub.group_counts = ds.group_counts
            return sub
        train_set = head(train_set, args.subset)
        val_set = head(val_set, args.subset)
        test_set = head(test_set, args.subset)
        print(f"[subset] 각 split 을 {args.subset} 장으로 제한 (점검 모드)")

    train_counts = train_set.group_counts()

    print(f"train {len(train_set)} / val {len(val_set)} / test {len(test_set)}")
    print(f"train group counts: {train_counts.tolist()}")

    common = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, **common)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **common)

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    try:                                    # torch >= 2.4
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    except (AttributeError, TypeError):     # 구버전
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_avg,val_adj,val_wga,test_avg,test_adj,test_wga\n")

    best_val_avg = -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = n_correct = n_total = 0

        for x, y, _g, _i in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * y.size(0)
            n_correct += (out.argmax(1) == y).sum().item()
            n_total += y.size(0)

        train_loss = loss_sum / n_total
        train_acc = n_correct / n_total

        is_eval_epoch = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if not is_eval_epoch:
            print(f"[{epoch:3d}] loss {train_loss:.4f}  acc {train_acc*100:.2f}")
            continue

        val = evaluate_groups(model, val_loader, device, train_counts)
        test = evaluate_groups(model, test_loader, device, train_counts)

        print(f"[{epoch:3d}] loss {train_loss:.4f}  acc {train_acc*100:.2f}  "
              f"({time.time()-t0:.0f}s)")
        print("  val : " + format_eval(val))
        print("  test: " + format_eval(test))

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_acc:.4f},"
                    f"{val['avg_acc']:.4f},{val['adjusted_acc']:.4f},{val['worst_group_acc']:.4f},"
                    f"{test['avg_acc']:.4f},{test['adjusted_acc']:.4f},{test['worst_group_acc']:.4f}\n")

        # 모델 선택: validation 평균 정확도 기준 (ERM 표준 프로토콜)
        if val["avg_acc"] > best_val_avg:
            best_val_avg = val["avg_acc"]
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in val.items()},
                        "args": vars(args)},
                       os.path.join(args.out_dir, "best.pt"))

    torch.save({"epoch": args.epochs, "state_dict": model.state_dict(),
                "args": vars(args)}, os.path.join(args.out_dir, "last.pt"))

    # ---- 최종 점검 -------------------------------------------------------
    try:                                    # torch >= 2.6 은 weights_only 기본값이 True
        ckpt = torch.load(os.path.join(args.out_dir, "best.pt"),
                          map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    final = evaluate_groups(model, test_loader, device, train_counts)

    print("\n" + "=" * 62)
    print(f"best checkpoint (epoch {ckpt['epoch']}) 의 테스트 성능")
    print("=" * 62)
    print(format_eval(final))

    gap = final["avg_acc"] - final["worst_group_acc"]
    print(f"\n평균 - WGA 격차: {gap*100:.1f} pt")
    if final["avg_acc"] >= 0.90 and gap >= 0.15:
        print("[통과] 평균은 높고 WGA 는 낮음 -> shortcut 에 의존하는 모델 확보")
    elif final["avg_acc"] < 0.90:
        print("[확인 필요] 평균 정확도가 낮음. epoch/lr 을 조정할 것")
    else:
        print("[확인 필요] 격차가 작아 shortcut 이 약함.")
        print("            --no_augment 로 재학습하거나 상관 100% 변형 데이터를 고려")

    with open(os.path.join(args.out_dir, "final_test.json"), "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in final.items()}, f, indent=2)


if __name__ == "__main__":
    main()
