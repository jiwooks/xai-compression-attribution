"""
Waterbirds 데이터셋 로더 + 그룹별 평가 유틸.

group = y * 2 + place
  0: landbird  on land   (다수)
  1: landbird  on water  (소수)
  2: waterbird on land   (소수)
  3: waterbird on water  (다수)

전처리는 Sagawa et al. (2020) 의 공식 구현을 따름:
  eval  : Resize((256,256)) -> CenterCrop(224)
  train : RandomResizedCrop(224, scale=(0.7,1.0)) -> RandomHorizontalFlip
"""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

N_GROUPS = 4
GROUP_NAMES = [
    "landbird/land",
    "landbird/water",
    "waterbird/land",
    "waterbird/water",
]
SPLIT_ID = {"train": 0, "val": 1, "test": 2}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(train: bool, augment: bool = True, resolution: int = 224):
    """augment=False 로 두면 학습에도 eval 과 동일한 결정적 전처리를 사용한다.

    증강을 끄면 모델이 배경 shortcut 에 더 강하게 의존하는 경향이 있어,
    H1 검증에 필요한 변화 폭을 확보하는 용도로 쓸 수 있다.
    """
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                resolution, scale=(0.7, 1.0), ratio=(0.75, 4.0 / 3.0)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            norm,
        ])
    scaled = int(resolution * 256 / 224)
    return transforms.Compose([
        transforms.Resize((scaled, scaled)),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        norm,
    ])


class WaterbirdsDataset(Dataset):
    """반환: (image, y, group, index)

    return_mask=True 이면 (image, y, group, index, mask) 를 반환한다.
    mask 는 이미지와 같은 해상도의 float tensor (1=새, 0=배경) 이며,
    Attr_bg 계산에 사용한다. cub_root 가 필요하다.
    """

    def __init__(self, wb_root, split="train", transform=None,
                 return_mask=False, cub_root=None, resolution=224):
        self.wb_root = wb_root
        self.cub_root = cub_root
        self.return_mask = return_mask
        self.resolution = resolution

        if return_mask and cub_root is None:
            raise ValueError("return_mask=True 이면 cub_root 가 필요함")

        df = pd.read_csv(os.path.join(wb_root, "metadata.csv"))
        df["group"] = df["y"] * 2 + df["place"]
        if split is not None:
            df = df[df["split"] == SPLIT_ID[split]].reset_index(drop=True)
        self.df = df

        self.filenames = df["img_filename"].values
        self.y = df["y"].values.astype(np.int64)
        self.group = df["group"].values.astype(np.int64)

        self.transform = transform if transform is not None else build_transform(
            train=(split == "train"), resolution=resolution
        )
        # 마스크는 이미지와 동일한 기하 변환을 받아야 하므로 별도로 처리
        scaled = int(resolution * 256 / 224)
        self.mask_transform = transforms.Compose([
            transforms.Resize((scaled, scaled), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(resolution),
        ])

    def __len__(self):
        return len(self.df)

    def group_counts(self):
        return np.bincount(self.group, minlength=N_GROUPS)

    def __getitem__(self, i):
        rel = self.filenames[i]
        img = Image.open(os.path.join(self.wb_root, rel)).convert("RGB")
        x = self.transform(img)

        if not self.return_mask:
            return x, self.y[i], self.group[i], i

        seg_path = os.path.join(self.cub_root, "segmentations",
                                rel.replace(".jpg", ".png"))
        seg = Image.open(seg_path).convert("L")
        if seg.size != img.size:
            raise RuntimeError(
                f"마스크 크기 불일치: {rel} img={img.size} seg={seg.size}. "
                "check_data.py 로 먼저 점검할 것."
            )
        m = self.mask_transform(seg)
        mask = (torch.from_numpy(np.asarray(m)).float() > 127).float()
        return x, self.y[i], self.group[i], i, mask


@torch.no_grad()
def evaluate_groups(model, loader, device, train_group_counts=None):
    """그룹별 정확도 / 평균 / worst-group 정확도를 계산한다.

    반환 dict:
      group_acc      : 그룹별 정확도 (길이 4)
      group_n        : 그룹별 샘플 수
      avg_acc        : 테스트셋 전체 평균 (모든 이미지 동일 가중)
      adjusted_acc   : 학습셋 그룹 비율로 가중한 평균 (Sagawa et al. 방식)
      worst_group_acc: 최저 그룹 정확도 (WGA)
      worst_group    : 최저 그룹 번호
    """
    model.eval()
    correct = np.zeros(N_GROUPS)
    total = np.zeros(N_GROUPS)

    for batch in loader:
        x, y, g = batch[0], batch[1], batch[2]
        x = x.to(device, non_blocking=True)
        pred = model(x).argmax(1).cpu()
        hit = (pred == y).numpy()
        g = g.numpy()
        for k in range(N_GROUPS):
            sel = g == k
            correct[k] += hit[sel].sum()
            total[k] += sel.sum()

    group_acc = np.divide(correct, total, out=np.full(N_GROUPS, np.nan), where=total > 0)
    seen = total > 0

    out = {
        "group_acc": group_acc,
        "group_n": total.astype(int),
        "avg_acc": float(correct.sum() / max(total.sum(), 1)),
        "worst_group_acc": float(np.nanmin(group_acc[seen])) if seen.any() else float("nan"),
        "worst_group": int(np.nanargmin(np.where(seen, group_acc, np.inf))),
    }

    if train_group_counts is not None:
        w = np.asarray(train_group_counts, dtype=float)
        w = w / w.sum()
        valid = seen & ~np.isnan(group_acc)
        out["adjusted_acc"] = float((group_acc[valid] * w[valid]).sum() / w[valid].sum())
    else:
        out["adjusted_acc"] = float("nan")

    return out


def format_eval(res, prefix=""):
    parts = [f"{prefix}avg {res['avg_acc']*100:.2f}"]
    if not np.isnan(res["adjusted_acc"]):
        parts.append(f"adj {res['adjusted_acc']*100:.2f}")
    parts.append(f"WGA {res['worst_group_acc']*100:.2f} (g{res['worst_group']})")
    per = " | ".join(
        f"{GROUP_NAMES[k]} {res['group_acc'][k]*100:.1f}" for k in range(N_GROUPS)
    )
    return "  ".join(parts) + "\n    " + per
