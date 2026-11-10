"""
STEP 4. 이미지별 attribution 과 Attr_bg 계산 (CLAUDE.md §4 "2026-11-10 결정 사항" 6~9).

설명 대상: 원본 사진의 예측 클래스 margin  z_pred − z_other.  RISE 는 예측 클래스 softmax 확률.

  gradcam : model.layer4 출력 A 와 d margin / dA. 채널 가중치 = 공간 평균 gradient,
            cam = ReLU(sum_c w_c A_c). 224 bilinear 업샘플 맵(주)과 7×7 맵(마스크를 면적 평균으로 축소) 둘 다.
  ig      : baseline = 검정 이미지(픽셀 0 을 정규화한 값), 중점 Riemann 합 50 step.
            픽셀 맵 = 채널별 |attr| 합. completeness 상대오차 |Σattr − (F(x)−F(x'))| / |F(x)−F(x')| 를 기록하고,
            중앙값이 5% 를 넘으면 100 step 으로 다시 계산한다.
  rise    : 7×7 격자, p=0.5, 마스크 2,000개 (bilinear 업샘플 후 랜덤 시프트, 원 논문 방식).
            마스크는 고정 시드로 만들어 모든 모델이 같은 것을 쓴다. 픽셀 공간에서 곱한다(가려진 곳 = 검정).

Attr_bg (CLAUDE.md §2)
  마스크 = 224 해상도 이진(>0.5). Attr_bg = Σ_{배경}|s| / Σ|s|.
  Attr_bg_band = 경계 ±3px 띠를 분자·분모에서 모두 뺀 값.  C = (1 − Attr_bg) / area.

출력: <out>.csv.gz (이미지별), <out>.json (전체·그룹별 평균, 처리 시간)

예)
  python compute_attribution.py --ckpt runs/erm_seed0/best.pt --wb_root $WB_ROOT --cub_root $CUB_ROOT \\
      --method gradcam --out runs/erm_seed0/attr_gradcam
  python compute_attribution.py --make_rise_list --wb_root $WB_ROOT        # results/rise_images.csv
  python compute_attribution.py --sanity 200 --ckpt runs/erm_seed0/best.pt ...   # 가중치 무작위화 점검
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Subset

from train_erm import build_model
from waterbirds import GROUP_NAMES, IMAGENET_MEAN, IMAGENET_STD, SPLIT_ID, WaterbirdsDataset

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

RES = 224
BAND = 3
RISE_LIST = os.path.join("results", "rise_images.csv")
MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


# ---------------------------------------------------------------- 설명 방법

def margin_of(logits, pred):
    """예측 클래스 margin z_pred − z_other (이진 분류)."""
    z = logits.gather(1, pred[:, None])[:, 0]
    return 2 * z - logits.sum(1)


def gradcam(model, x):
    """반환: pred, cam224 (B,224,224), cam7 (B,7,7)"""
    feats = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: feats.__setitem__("A", o))
    try:
        logits = model(x)
    finally:
        h.remove()
    pred = logits.argmax(1)
    A = feats["A"]
    g, = torch.autograd.grad(margin_of(logits, pred).sum(), A)
    cam7 = F.relu((g.mean((2, 3), keepdim=True) * A).sum(1))
    cam224 = F.interpolate(cam7[:, None], size=(RES, RES), mode="bilinear", align_corners=False)[:, 0]
    return pred.detach(), cam224.detach(), cam7.detach()


def ig_baseline(x, kind):
    """black: 픽셀 0 을 정규화한 값. blur: 원본을 가우시안 σ=10px 로 흐린 이미지 (P4 ST-C).
    정규화는 채널별 affine 이고 커널 합이 1, reflect 패딩이라 정규화 공간에서 흐려도 같다."""
    if kind == "blur":
        return TF.gaussian_blur(x, kernel_size=61, sigma=10.0)
    return ((torch.zeros_like(x[:1]) - MEAN.to(x)) / STD.to(x)).expand_as(x)


def integrated_gradients(model, x, steps, chunk=64, baseline="black"):
    """반환: pred, map (B,224,224), completeness 상대오차 (B,)"""
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1)
        base = ig_baseline(x, baseline)
        f_x = margin_of(logits, pred)
        f_b = margin_of(model(base), pred)
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps
    B = x.shape[0]
    grad_sum = torch.zeros_like(x)
    diff = x - base
    # (B × steps) 보간 입력을 chunk 단위로 forward/backward
    pairs = [(b, a) for b in range(B) for a in range(steps)]
    for s in range(0, len(pairs), chunk):
        bi = torch.tensor([p[0] for p in pairs[s:s + chunk]], device=x.device)
        ai = alphas[[p[1] for p in pairs[s:s + chunk]]]
        inp = (base[bi] + ai.view(-1, 1, 1, 1) * diff[bi]).requires_grad_(True)
        out = margin_of(model(inp), pred[bi])
        g, = torch.autograd.grad(out.sum(), inp)
        grad_sum.index_add_(0, bi, g)
    attr = diff * grad_sum / steps
    total = attr.sum((1, 2, 3))
    rel_err = (total - (f_x - f_b)).abs() / (f_x - f_b).abs().clamp_min(1e-8)
    return pred, attr.abs().sum(1).detach(), rel_err.detach()


def make_rise_masks(n, s=7, p=0.5, seed=0, device="cpu"):
    """원 논문(Petsiuk et al. 2018) 방식: 이진 격자 → bilinear 업샘플 → 랜덤 시프트 크롭."""
    g = torch.Generator().manual_seed(seed)
    cell = int(np.ceil(RES / s))
    up = (s + 1) * cell
    grid = (torch.rand(n, 1, s, s, generator=g) < p).float()
    big = F.interpolate(grid, size=(up, up), mode="bilinear", align_corners=False)
    sx = torch.randint(0, cell, (n,), generator=g)
    sy = torch.randint(0, cell, (n,), generator=g)
    masks = torch.stack([big[i, 0, sx[i]:sx[i] + RES, sy[i]:sy[i] + RES] for i in range(n)])
    return masks.to(device)                                       # (n,224,224)


@torch.no_grad()
def rise(model, x, masks, p=0.5, chunk=500):
    """반환: pred, map (B,224,224), 가린 입력들의 평균 확률 p̄ (B,). 가려진 픽셀은 검정(픽셀 0).

    RISE 맵 = Σ p_i M_i / (N p) 에는 p̄ × (마스크 합 패턴) 이 깔린다. 이 패턴은 모든 모델이 같은 마스크를 쓰므로
    모델과 무관하게 공유되고, Attr_bg 를 1 − area 쪽으로 끈다. 보조로 이 패턴을 정확히 뺀
    centered 맵 Σ (p_i − p̄) M_i / (N p) 의 Attr_bg (attr_bg_centered) 도 저장한다.
    """
    logits = model(x)
    pred = logits.argmax(1)
    mean, std = MEAN.to(x), STD.to(x)
    pix = x * std + mean
    out = torch.zeros(x.shape[0], RES, RES, device=x.device)
    pbar = torch.zeros(x.shape[0], device=x.device)
    for b in range(x.shape[0]):
        for s in range(0, masks.shape[0], chunk):
            M = masks[s:s + chunk]
            inp = (pix[b:b + 1] * M[:, None] - mean) / std
            with torch.autocast("cuda", dtype=torch.float16, enabled=x.is_cuda):
                prob = model(inp).float().softmax(1)[:, pred[b]]
            out[b] += (prob[:, None, None] * M).sum(0)
            pbar[b] += prob.sum()
    return pred, out / (masks.shape[0] * p), pbar / masks.shape[0]


def rise_centered(smap, pbar, masks, p=0.5):
    """Σ (p_i − p̄) M_i / (N p) = s − p̄ · Σ M_i / (N p). 모델과 무관한 마스크 합 패턴을 뺀다."""
    pattern = masks.mean(0) / p                                   # (224,224)
    return smap - pbar[:, None, None] * pattern[None]


# ---------------------------------------------------------------- Attr_bg

def band_of(mask):
    """경계 ±BAND px 띠 (팽창 − 침식)."""
    k = 2 * BAND + 1
    dil = F.max_pool2d(mask[:, None], k, 1, BAND)[:, 0]
    ero = 1 - F.max_pool2d(1 - mask[:, None], k, 1, BAND)[:, 0]
    return dil - ero


def bg_share(s, bg_weight, keep=None):
    s = s.abs()
    if keep is not None:
        s = s * keep
    tot = s.sum((1, 2))
    return torch.where(tot > 0, (s * bg_weight).sum((1, 2)) / tot.clamp_min(1e-30),
                       torch.full_like(tot, float("nan")))


def scores(smap, mask, cam7=None):
    area = mask.mean((1, 2))
    bg = 1 - mask
    out = {"area": area,
           "attr_bg": bg_share(smap, bg),
           "attr_bg_band": bg_share(smap, bg, keep=1 - band_of(mask))}
    out["C"] = (1 - out["attr_bg"]) / area.clamp_min(1e-12)
    out["C"][area == 0] = float("nan")
    if cam7 is not None:
        m7 = F.adaptive_avg_pool2d(mask[:, None], 7)[:, 0]     # 칸별 새 면적 비율
        out["attr_bg_7x7"] = bg_share(cam7, 1 - m7)
    return out


# ---------------------------------------------------------------- 실행

def load_model(ckpt, device):
    model = build_model(device, pretrained=False)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    for p in model.parameters():
        p.requires_grad_(True)
    return model.eval()


def dataset(args):
    ds = WaterbirdsDataset(args.wb_root, split="test", return_mask=True, cub_root=args.cub_root)
    ids = ds.df["img_id"].to_numpy()
    if args.images == "rise1000":
        keep = set(pd.read_csv(RISE_LIST)["img_id"])
        idx = [i for i, v in enumerate(ids) if v in keep]
        assert len(idx) == len(keep), "rise_images.csv 의 img_id 가 테스트셋과 맞지 않다"
    else:
        idx = list(range(len(ds)))
    if args.subset > 0:
        rng = np.random.default_rng(args.seed)
        idx = sorted(rng.choice(idx, size=min(args.subset, len(idx)), replace=False).tolist())
    return ds, ids, Subset(ds, idx)


def run_method(model, method, x, masks_rise, ig_steps, ig_base="black"):
    if method == "gradcam":
        pred, m, cam7 = gradcam(model, x)
        return pred, m, {"cam7": cam7}
    if method == "ig":
        pred, m, err = integrated_gradients(model, x, ig_steps, baseline=ig_base)
        return pred, m, {"ig_rel_err": err}
    pred, m, pbar = rise(model, x, masks_rise)
    return pred, m, {"rise_pbar": pbar}


def compute(model, method, loader, ids, device, ig_steps, masks_rise, png_dir=None, n_png=0, ig_base="black"):
    rows, saved = [], 0
    for x, y, g, i, mask in loader:
        x, mask = x.to(device), mask.to(device)
        pred, smap, extra = run_method(model, method, x, masks_rise, ig_steps, ig_base)
        sc = scores(smap, mask, extra.get("cam7"))
        rec = {"img_id": ids[i.numpy()], "group": g.numpy(), "y": y.numpy(), "pred": pred.cpu().numpy()}
        rec.update({k: v.cpu().numpy() for k, v in sc.items()})
        if "ig_rel_err" in extra:
            rec["ig_rel_err"] = extra["ig_rel_err"].cpu().numpy()
        if "rise_pbar" in extra:
            pb = extra["rise_pbar"]
            rec["rise_pbar"] = pb.cpu().numpy()
            rec["attr_bg_centered"] = bg_share(rise_centered(smap, pb, masks_rise), 1 - mask).cpu().numpy()
        rows.append(pd.DataFrame(rec))
        if png_dir and saved < n_png:
            saved += save_png(png_dir, method, x, mask, smap, rec, n_png - saved)
    df = pd.concat(rows, ignore_index=True)
    df["correct"] = (df["pred"] == df["y"]).astype(int)
    return df


def save_png(out_dir, method, x, mask, smap, rec, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    pix = (x.cpu() * STD + MEAN).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    k = min(n, x.shape[0])
    for b in range(k):
        s = smap[b].abs().cpu().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        ax[0].imshow(pix[b]); ax[0].contour(mask[b].cpu().numpy(), [0.5], colors="lime", linewidths=0.8)
        ax[1].imshow(pix[b]); ax[1].imshow(s / (s.max() + 1e-12), cmap="jet", alpha=0.5)
        ax[1].contour(mask[b].cpu().numpy(), [0.5], colors="white", linewidths=0.8)
        g = int(rec["group"][b])
        fig.suptitle(f"{method} id={rec['img_id'][b]} {GROUP_NAMES[g]} pred={rec['pred'][b]} "
                     f"Attr_bg={rec['attr_bg'][b]:.2f} area={rec['area'][b]:.2f}", fontsize=7)
        for a in ax:
            a.axis("off")
        fig.savefig(os.path.join(out_dir, f"{method}_{rec['img_id'][b]}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)
    return k


def summary(df, method, extra):
    cols = [c for c in ("attr_bg", "attr_bg_band", "attr_bg_7x7", "attr_bg_centered", "C", "area",
                        "ig_rel_err") if c in df]
    out = {"method": method, "n_images": int(len(df)), "acc": float(df["correct"].mean()),
           "n_nan_attr_bg": int(df["attr_bg"].isna().sum()), "n_area0": int((df["area"] == 0).sum()),
           "overall": {c: float(df[c].mean()) for c in cols},
           "groups": {GROUP_NAMES[g]: {c: float(s[c].mean()) for c in cols} for g, s in df.groupby("group")}}
    if "ig_rel_err" in df:
        out["ig_rel_err_median"] = float(df["ig_rel_err"].median())
    out.update(extra)
    return out


def make_rise_list(args):
    meta = pd.read_csv(os.path.join(args.wb_root, "metadata.csv"))
    meta = meta[meta["split"] == SPLIT_ID["test"]].copy()
    meta["group"] = meta["y"] * 2 + meta["place"]
    pick = meta.groupby("group").sample(n=250, random_state=args.seed)
    pick = pick.sort_values("img_id")[["img_id", "img_filename", "y", "place", "group"]]
    os.makedirs(os.path.dirname(RISE_LIST), exist_ok=True)
    pick.to_csv(RISE_LIST, index=False)
    print(f"저장: {RISE_LIST} ({len(pick)}장, 그룹별 {pick['group'].value_counts().sort_index().tolist()})")


def spearman_rows(a, b):
    """행별 Spearman (B, P) -> (B,). 동순위(Grad-CAM 의 0 영역 등)는 평균 순위로 처리한다."""
    from scipy.stats import spearmanr
    a, b = a.cpu().numpy(), b.cpu().numpy()
    out = []
    for u, v in zip(a, b):
        out.append(np.nan if u.std() == 0 or v.std() == 0 else spearmanr(u, v)[0])
    return np.array(out)


def sanity(args, device, loader, masks_rise):
    """가중치 무작위화 점검 (Adebayo et al. 2018): 원래 모델과 무작위 모델의 맵 순위 상관."""
    torch.manual_seed(args.seed)
    rand = build_model(device, pretrained=False).eval()       # 모든 가중치 새로 초기화
    model = load_model(args.ckpt, device)
    out = {}
    if os.path.exists(args.out + ".json"):          # 방법별로 나눠 돌릴 때 앞 결과를 보존
        with open(args.out + ".json") as f:
            out = json.load(f).get("spearman_orig_vs_random", {})

    def record(name, r):
        out[name] = {"n": int(len(r)), "median": float(np.nanmedian(r)),
                     "q25": float(np.nanpercentile(r, 25)), "q75": float(np.nanpercentile(r, 75)),
                     "n_nan": int(np.isnan(r).sum())}
        print(f"[sanity] {name}: Spearman 중앙값 {out[name]['median']:.3f} "
              f"(IQR {out[name]['q25']:.3f}~{out[name]['q75']:.3f}, n={len(r)})", flush=True)

    for method in args.sanity_methods.split(","):
        rhos, rhos_c = [], []
        for x, *_ in loader:
            x = x.to(device)
            _, m1, e1 = run_method(model, method, x, masks_rise, args.ig_steps)
            _, m2, e2 = run_method(rand, method, x, masks_rise, args.ig_steps)
            rhos.append(spearman_rows(m1.abs().flatten(1), m2.abs().flatten(1)))
            if method == "rise":
                c1 = rise_centered(m1, e1["rise_pbar"], masks_rise)
                c2 = rise_centered(m2, e2["rise_pbar"], masks_rise)
                rhos_c.append(spearman_rows(c1.abs().flatten(1), c2.abs().flatten(1)))
        record(method, np.concatenate(rhos))
        if rhos_c:
            record("rise_centered", np.concatenate(rhos_c))
    with open(args.out + ".json", "w") as f:
        json.dump({"ckpt": args.ckpt, "n_images": args.sanity, "spearman_orig_vs_random": out}, f, indent=2)
    print(f"저장: {args.out}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt")
    ap.add_argument("--wb_root")
    ap.add_argument("--cub_root")
    ap.add_argument("--method", choices=["gradcam", "ig", "rise"])
    ap.add_argument("--out", help="출력 경로 접두어 (.csv.gz / .json 이 붙는다)")
    ap.add_argument("--images", choices=["all", "rise1000"], default=None,
                    help="기본: rise 는 rise1000, 나머지는 all")
    ap.add_argument("--subset", type=int, default=0, help="점검용으로 N장만 (고정 시드 무작위)")
    ap.add_argument("--seed", type=int, default=0, help="subset·RISE 목록·무작위화 시드")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--ig_steps", type=int, default=50)
    ap.add_argument("--ig_baseline", choices=["black", "blur"], default="black",
                    help="blur = 가우시안 σ=10px 로 흐린 원본 (P4 ST-C)")
    ap.add_argument("--rise_masks", type=int, default=2000)
    ap.add_argument("--rise_seed", type=int, default=0, help="RISE 마스크 시드 (모든 모델 공통)")
    ap.add_argument("--expect_acc", type=float, default=None,
                    help="final_test.json 의 avg_acc. 전체 이미지일 때 예측 정확도와 대조한다")
    ap.add_argument("--ref_dp", default=None,
                    help="shortcut_k5_dp.csv.gz. 이미지별 원본 예측(p_w_orig > 0.5)과 대조한다")
    ap.add_argument("--max_pred_mismatch", type=int, default=5,
                    help="원본 예측이 이 장수보다 많이 다르면 멈춘다")
    ap.add_argument("--png_dir", default=None)
    ap.add_argument("--n_png", type=int, default=0)
    ap.add_argument("--make_rise_list", action="store_true")
    ap.add_argument("--sanity", type=int, default=0, help="가중치 무작위화 점검 이미지 수")
    ap.add_argument("--sanity_methods", default="gradcam,ig,rise")
    args = ap.parse_args()

    if args.make_rise_list:
        return make_rise_list(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    masks_rise = (make_rise_masks(args.rise_masks, seed=args.rise_seed, device=device)
                  if args.sanity or args.method == "rise" else None)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.sanity:
        args.images, args.subset = "all", args.sanity
        _, _, sub = dataset(args)
        return sanity(args, device, DataLoader(sub, batch_size=8, num_workers=args.num_workers), masks_rise)

    args.images = args.images or ("rise1000" if args.method == "rise" else "all")
    bs = args.batch_size or {"gradcam": 64, "ig": 4, "rise": 4}[args.method]
    ds, ids, sub = dataset(args)
    loader = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = load_model(args.ckpt, device)
    print(f"{args.method}: {args.ckpt}  이미지 {len(sub)}장", flush=True)

    t0 = time.time()
    steps = args.ig_steps
    df = compute(model, args.method, loader, ids, device, steps, masks_rise, args.png_dir, args.n_png,
                 ig_base=args.ig_baseline)
    extra = {}
    if args.method == "ig":
        med = float(df["ig_rel_err"].median())
        print(f"IG {steps} step completeness 상대오차 중앙값 {med:.4f}", flush=True)
        if med > 0.05 and steps < 100:
            extra["ig_rel_err_median_first"] = med
            steps = 100
            df = compute(model, args.method, loader, ids, device, steps, masks_rise, ig_base=args.ig_baseline)
            print(f"IG 100 step 재계산: 중앙값 {df['ig_rel_err'].median():.4f}", flush=True)
        extra["ig_steps"] = steps
        extra["ig_baseline"] = args.ig_baseline
    if args.method == "rise":
        extra.update({"rise_masks": args.rise_masks, "rise_seed": args.rise_seed, "rise_grid": 7, "rise_p": 0.5})
    extra.update({"ckpt": args.ckpt, "images": args.images, "subset": args.subset,
                  "seconds": round(time.time() - t0, 1)})

    # 예측 대조. 배치 크기가 달라 결정 경계에 걸린 이미지 1~2장이 바뀔 수 있다 (2026-11-10 확인: 1장).
    # 그래서 정확히 같음 대신 measure_shortcut 의 원본 예측과 이미지별로 맞춰 보고, 어긋난 장수를 기록한다.
    acc = float(df["correct"].mean())
    if args.ref_dp and os.path.exists(args.ref_dp):
        ref = pd.read_csv(args.ref_dp)[["img_id", "p_w_orig"]]
        m = df[["img_id", "pred"]].merge(ref, on="img_id", how="left")
        assert m["p_w_orig"].notna().all(), "ref_dp 에 없는 이미지가 있다"
        bad = m[(m["p_w_orig"] > 0.5).astype(int) != m["pred"]]
        extra["n_pred_mismatch_vs_measure"] = int(len(bad))
        extra["pred_mismatch_p_w_orig"] = [round(float(v), 4) for v in bad["p_w_orig"]][:20]
        print(f"measure_shortcut 원본 예측과 다른 이미지 {len(bad)}장 (p_w_orig {extra['pred_mismatch_p_w_orig']})")
        if len(bad) > args.max_pred_mismatch:
            raise SystemExit(f"[실패] 원본 예측이 {len(bad)}장 다르다 (허용 {args.max_pred_mismatch}장)")
    if args.expect_acc is not None and args.images == "all" and not args.subset:
        n_diff = abs(acc - args.expect_acc) * len(df)
        extra["acc_diff_images_vs_final_test"] = round(n_diff, 2)
        if n_diff > args.max_pred_mismatch + 1e-6:
            raise SystemExit(f"[실패] 예측 정확도 {acc} 가 final_test.json 의 {args.expect_acc} 와 "
                             f"{n_diff:.1f}장만큼 다르다")

    df.to_csv(args.out + ".csv.gz", index=False, float_format="%.6g")
    res = summary(df, args.method, extra)
    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"정확도 {acc*100:.2f}%  Attr_bg {res['overall']['attr_bg']:.4f}  band {res['overall']['attr_bg_band']:.4f}"
          + (f"  7x7 {res['overall']['attr_bg_7x7']:.4f}" if "attr_bg_7x7" in res["overall"] else "")
          + f"  area {res['overall']['area']:.4f}  ({extra['seconds']}s)")
    for name, g in res["groups"].items():
        print(f"  {name:16s} Attr_bg {g['attr_bg']:.4f}  C {g['C']:.3f}")
    print(f"저장: {args.out}.csv.gz, {args.out}.json")


if __name__ == "__main__":
    main()
