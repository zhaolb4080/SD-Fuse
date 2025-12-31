#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

# 你的工程文件
from network1 import SD_Fuse_Net, ResT, build_structure_extractor


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(p: str) -> bool:
    return os.path.splitext(p.lower())[1] in IMG_EXTS


def _last_int_in_name(fname: str) -> Optional[int]:
    """
    从文件名中提取“最后一段数字”，例如:
      "001.png" -> 1
      "img_12_far.jpg" -> 12
    """
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.search(r"(\d+)(?!.*\d)", base)
    if m is None:
        return None
    return int(m.group(1))


def build_pairs_from_odd_even(input_dir: str) -> List[Tuple[int, str, str]]:
    """
    根据“奇数编号=近景，偶数编号=远景”，构建 (id, near_path, far_path)
    规则：near=odd k, far=even k+1
    """
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if _is_image_file(f)]
    num2path: Dict[int, str] = {}

    for p in files:
        n = _last_int_in_name(p)
        if n is None:
            continue
        num2path[n] = p

    pairs: List[Tuple[int, str, str]] = []
    keys = sorted(num2path.keys())
    for n in keys:
        if n % 2 == 1 and (n + 1) in num2path:
            pairs.append((n, num2path[n], num2path[n + 1]))

    # 若无法按数字规则配对，则退化为“排序后两两配对”
    if len(pairs) == 0:
        files_sorted = sorted(files)
        for i in range(0, len(files_sorted) - 1, 2):
            pairs.append((i // 2 + 1, files_sorted[i], files_sorted[i + 1]))

    return pairs

def build_pairs_from_two_dirs(near_dir: str, far_dir: str) -> List[Tuple[int, str, str]]:
    """
    近景/远景分别在两个文件夹中，文件名含相同数字ID（例如 1.png 与 1.png 或 img_0003.jpg 与 0003.png）。
    返回: (id, near_path, far_path)
    """
    def scan_dir(d: str) -> Dict[int, str]:
        files = [os.path.join(d, f) for f in os.listdir(d) if _is_image_file(f)]
        num2path: Dict[int, str] = {}
        for p in sorted(files):
            n = _last_int_in_name(p)
            if n is None:
                continue
            # 若出现相同数字重复文件，保留第一个并提示
            if n in num2path:
                # 你也可以改成覆盖或抛错，这里选择保守提示
                # print(f"[WARN] duplicate id={n} in {d}: keep {num2path[n]}, ignore {p}")
                continue
            num2path[n] = p
        return num2path

    if not os.path.isdir(near_dir):
        raise FileNotFoundError(f"near_dir not found: {near_dir}")
    if not os.path.isdir(far_dir):
        raise FileNotFoundError(f"far_dir not found: {far_dir}")

    near_map = scan_dir(near_dir)
    far_map = scan_dir(far_dir)

    common_ids = sorted(set(near_map.keys()) & set(far_map.keys()))
    pairs: List[Tuple[int, str, str]] = [(i, near_map[i], far_map[i]) for i in common_ids]

    if len(pairs) == 0:
        raise RuntimeError(
            "No matched pairs found between near_dir and far_dir by numeric ID. "
            "Ensure filenames contain a numeric ID (e.g., 1.png / 1.png)."
        )

    return pairs



def pil_to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)
    return t


def tensor_to_pil_rgb(x: torch.Tensor) -> Image.Image:
    """
    x: (3,H,W) in [0,1]
    """
    x = x.detach().clamp(0, 1).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def tensor_to_pil_gray(x: torch.Tensor) -> Image.Image:
    """
    x: (1,H,W) or (H,W), in [0,1]
    """
    if x.dim() == 3:
        x = x[0]
    x = x.detach().clamp(0, 1).cpu()
    arr = (x.numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def pad_to_multiple(x: torch.Tensor, mult: int = 16) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """
    x: (B,C,H,W)
    pad 到 H,W 为 mult 的倍数，使用 reflect padding。
    返回 padded_x 和 pad=(pl, pr, pt, pb)
    """
    _, _, h, w = x.shape
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    pt, pb = 0, pad_h
    pl, pr = 0, pad_w
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0, 0, 0)
    x = F.pad(x, (pl, pr, pt, pb), mode="reflect")
    return x, (pl, pr, pt, pb)


def unpad(x: torch.Tensor, pad: Tuple[int, int, int, int]) -> torch.Tensor:
    """
    x: (B,C,H,W) padded
    pad=(pl, pr, pt, pb)
    """
    pl, pr, pt, pb = pad
    if (pl, pr, pt, pb) == (0, 0, 0, 0):
        return x
    _, _, h, w = x.shape
    return x[:, :, pt: h - pb, pl: w - pr]


def load_model(ckpt_path: str, device: torch.device,
               extractor_name: Optional[str] = None,
               out_stage: Optional[int] = None,
               r: Optional[int] = None,
               lam: Optional[float] = None,
               strict: bool = True) -> SD_Fuse_Net:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    extractor_name = extractor_name or ckpt_args.get("extractor", "sobel")
    out_stage = int(out_stage if out_stage is not None else ckpt_args.get("out_stage", 2))
    r = int(r if r is not None else ckpt_args.get("r", 5))
    lam = float(lam if lam is not None else ckpt_args.get("lam", 0.5))

    extractor = build_structure_extractor(extractor_name)
    transformer = ResT(in_chans=1, out_stage=out_stage)

    model = SD_Fuse_Net(
        extractor=extractor,
        transformer=transformer,
        r=r,
        lam=lam,
        num_marm=6,
        guidance_from="mean",
        transformer_in_channels=1,
    )

    state = None
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
    if state is None:

        state = ckpt

    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        if strict:
            raise

        missing, unexpected = model.load_state_dict(state, strict=False)
        print("[WARN] load_state_dict(strict=False)")
        print("  missing keys:", missing)
        print("  unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model

def infer_dm(model: torch.nn.Module, near: torch.Tensor, far: torch.Tensor,
             pad_mult: int = 16) -> torch.Tensor:

    #near/far: (1,3,H,W) in [0,1]
    #return: dm_prob (1,1,H,W) in [0,1]

    near_p, pad = pad_to_multiple(near, mult=pad_mult)
    far_p, _ = pad_to_multiple(far, mult=pad_mult)

    out = model(near_p, far_p)  # 可能是 prob 或 logits
    # 统一成 (B,1,H,W)
    if out.dim() == 3:
        out = out.unsqueeze(1)
    if out.size(1) != 1:
        out = out.mean(dim=1, keepdim=True)

    # 若看起来是 logits，则 sigmoid
    out_min = float(out.min())
    out_max = float(out.max())
    if out_min < -1e-3 or out_max > 1.0 + 1e-3:
        dm = torch.sigmoid(out)
    else:
        dm = out.clamp(0.0, 1.0)

    dm = unpad(dm, pad)
    return dm


def fuse(near: torch.Tensor, far: torch.Tensor, dm: torch.Tensor) -> torch.Tensor:
    """
    near/far: (1,3,H,W)
    dm: (1,1,H,W) in [0,1]
    return fused: (1,3,H,W)
    """
    return dm * near + (1.0 - dm) * far


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, default=None,
                    help="single folder mode: contains test images, odd=near even=far")
    ap.add_argument("--near_dir", type=str, default=None,
                    help="two-folder mode: near/foreground images directory")
    ap.add_argument("--far_dir", type=str, default=None,
                    help="two-folder mode: far/background images directory")

    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt from training")
    ap.add_argument("--out_dm_dir", type=str, default="test_out", help="output folder")
    ap.add_argument("--out_fuse_dir", type=str, default="test_out", help="output folder")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")

    ap.add_argument("--extractor", type=str, default=None, help="override extractor (default from ckpt)")
    ap.add_argument("--out_stage", type=int, default=None, help="override ResT out_stage (default from ckpt)")
    ap.add_argument("--r", type=int, default=None, help="override SGF radius r (default from ckpt)")
    ap.add_argument("--lam", type=float, default=None, help="override SGF lambda lam (default from ckpt)")
    ap.add_argument("--pad_mult", type=int, default=16, help="pad H/W to multiple of this")
    ap.add_argument("--max_pairs", type=int, default=0, help="0 means all pairs")

    args = ap.parse_args()

    os.makedirs(args.out_fuse_dir, exist_ok=True)
    os.makedirs(args.out_dm_dir, exist_ok=True)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("device:", device)

    model = load_model(
        ckpt_path=args.ckpt,
        device=device,
        extractor_name=args.extractor,
        out_stage=args.out_stage,
        r=args.r,
        lam=args.lam,
        strict=True,
    )

    # choose pairing mode
    if args.near_dir is not None or args.far_dir is not None:
        if not (args.near_dir and args.far_dir):
            raise ValueError("two-folder mode requires both --near_dir and --far_dir")
        pairs = build_pairs_from_two_dirs(args.near_dir, args.far_dir)
    else:
        if not args.input_dir:
            raise ValueError("please provide either --input_dir (single folder) or --near_dir/--far_dir (two folders)")
        pairs = build_pairs_from_odd_even(args.input_dir)

    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    if len(pairs) == 0:
        raise RuntimeError("No image pairs found. Check input_dir and file naming.")

    print(f"found pairs: {len(pairs)}")

    for pid, p_near, p_far in pairs:
        img_near = Image.open(p_near).convert("RGB")
        img_far = Image.open(p_far).convert("RGB")

        if img_near.size != img_far.size:
            print(f"[SKIP] size mismatch pid={pid}: near={img_near.size} far={img_far.size}")
            continue

        near_t = pil_to_tensor_rgb(img_near).unsqueeze(0).to(device)  # (1,3,H,W)
        far_t = pil_to_tensor_rgb(img_far).unsqueeze(0).to(device)

        dm = infer_dm(model, near_t, far_t, pad_mult=args.pad_mult)   # (1,1,H,W)

        fused_t = fuse(near_t, far_t, dm)  # (1,3,H,W)

        # save
        near_stem = os.path.splitext(os.path.basename(p_near))[0]
        out_fused = os.path.join(args.out_fuse_dir, f"{near_stem}_fused.png")

        tensor_to_pil_rgb(fused_t[0]).save(out_fused)

        print(f"[OK] pid={pid} -> {os.path.basename(out_fused)}")

    print("done.")


if __name__ == "__main__":
    main()
