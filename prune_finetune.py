"""
STEP 3. 전역 magnitude pruning + fine-tuning.

dense ERM 체크포인트에서 Conv/Linear 가중치 전체를 통틀어 절댓값이 작은 순으로
sparsity 비율만큼 0 으로 만든 뒤(one-shot), 마스크를 고정한 채 fine-tuning 한다.
BatchNorm 과 bias 는 자르지 않는다.

fine-tuning 은 ERM 과 같은 설정(SGD, 증강, 평균 정확도 기준 모델 선택)을 쓴다.
sparsity 0 도 같은 횟수만큼 fine-tuning 해 "fine-tuning 만의 효과" 대조군으로 쓴다.

출력
  best.pt          : validation 평균 정확도 기준 최고 체크포인트 (마스크를 반영한 일반 state_dict)
  final_test.json  : best.pt 의 테스트 성능 + 실제 sparsity + fine-tuning 전 성능(oneshot)
  log.csv, args.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, Subset

from train_erm import build_model, set_seed
from waterbirds import WaterbirdsDataset, build_transform, evaluate_groups, format_eval


def prunable(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def measured_sparsity(model):
    zeros = total = 0
    for m, name in prunable(model):
        w = getattr(m, name)
        zeros += int((w == 0).sum())
        total += w.numel()
    return zeros / total


def plain_state_dict(model):
    """prune 의 weight_orig, weight_mask 를 곱해 일반 weight 로 바꾼 state_dict."""
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for k in [k for k in sd if k.endswith("_orig")]:
        base = k[: -len("_orig")]
        sd[base] = sd.pop(k) * sd.pop(base + "_mask")
    return sd


def to_json(res):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="dense ERM 의 best.pt")
    ap.add_argument("--wb_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sparsity", type=float, required=True, help="0~1")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--subset", type=int, default=0, help="점검용. 각 split 을 이 장수로 제한")
    args = ap.parse_args()
    if not 0 <= args.sparsity < 1:
        raise SystemExit("--sparsity 는 0 이상 1 미만")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    train_set = WaterbirdsDataset(args.wb_root, "train", transform=build_transform(train=True))
    val_set = WaterbirdsDataset(args.wb_root, "val", transform=build_transform(train=False))
    test_set = WaterbirdsDataset(args.wb_root, "test", transform=build_transform(train=False))
    train_counts = train_set.group_counts()
    if args.subset > 0:
        train_set, val_set, test_set = (Subset(ds, list(range(min(args.subset, len(ds)))))
                                        for ds in (train_set, val_set, test_set))
        print(f"[subset] 각 split 을 {args.subset} 장으로 제한 (점검 모드)")

    common = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **common)

    model = build_model(device, pretrained=False)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    if args.sparsity > 0:
        prune.global_unstructured(prunable(model), pruning_method=prune.L1Unstructured,
                                  amount=args.sparsity)
    print(f"sparsity 목표 {args.sparsity:.2f} / 실제 {measured_sparsity(model):.4f}")

    oneshot_val = evaluate_groups(model, val_loader, device, train_counts)
    oneshot_test = evaluate_groups(model, test_loader, device, train_counts)
    print("[  0] one-shot (fine-tuning 전)")
    print("  val : " + format_eval(oneshot_val))
    print("  test: " + format_eval(oneshot_test))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_path = os.path.join(args.out_dir, "log.csv")
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_acc,val_avg,val_adj,val_wga,test_avg,test_adj,test_wga\n")
        f.write(f"0,,,{oneshot_val['avg_acc']:.4f},{oneshot_val['adjusted_acc']:.4f},"
                f"{oneshot_val['worst_group_acc']:.4f},{oneshot_test['avg_acc']:.4f},"
                f"{oneshot_test['adjusted_acc']:.4f},{oneshot_test['worst_group_acc']:.4f}\n")

    best_val_avg, best_epoch = -1.0, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = n_correct = n_total = 0
        for x, y, _g, _i in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.item() * y.size(0)
            n_correct += (out.argmax(1) == y).sum().item()
            n_total += y.size(0)

        if epoch % args.eval_every and epoch != args.epochs:
            continue
        val = evaluate_groups(model, val_loader, device, train_counts)
        test = evaluate_groups(model, test_loader, device, train_counts)
        print(f"[{epoch:3d}] loss {loss_sum/n_total:.4f}  acc {n_correct/n_total*100:.2f}  "
              f"({time.time()-t0:.0f}s)")
        print("  val : " + format_eval(val))
        print("  test: " + format_eval(test))
        with open(log_path, "a") as f:
            f.write(f"{epoch},{loss_sum/n_total:.4f},{n_correct/n_total:.4f},"
                    f"{val['avg_acc']:.4f},{val['adjusted_acc']:.4f},{val['worst_group_acc']:.4f},"
                    f"{test['avg_acc']:.4f},{test['adjusted_acc']:.4f},{test['worst_group_acc']:.4f}\n")

        # 모델 선택: ERM 과 같은 validation 평균 정확도 기준. fine-tuning 한 epoch 중에서만 고른다.
        if val["avg_acc"] > best_val_avg:
            best_val_avg, best_epoch = val["avg_acc"], epoch
            torch.save({"epoch": epoch, "state_dict": plain_state_dict(model),
                        "sparsity": args.sparsity, "args": vars(args)},
                       os.path.join(args.out_dir, "best.pt"))

    best = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device,
                      weights_only=False)
    final_model = build_model(device, pretrained=False)
    final_model.load_state_dict(best["state_dict"])
    final = evaluate_groups(final_model, test_loader, device, train_counts)
    actual = measured_sparsity(final_model)

    print("\n" + "=" * 62)
    print(f"best checkpoint (epoch {best_epoch}) 의 테스트 성능, 실제 sparsity {actual:.4f}")
    print("=" * 62)
    print(format_eval(final))
    if abs(actual - args.sparsity) > 0.005:
        raise SystemExit(f"[실패] 저장된 모델의 sparsity {actual:.4f} 가 목표와 다르다.")

    out = to_json(final)
    out.update({"best_epoch": best_epoch, "sparsity_target": args.sparsity,
                "sparsity_actual": actual, "oneshot": to_json(oneshot_test)})
    with open(os.path.join(args.out_dir, "final_test.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
