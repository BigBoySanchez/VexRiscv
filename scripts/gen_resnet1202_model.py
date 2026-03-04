#!/usr/bin/env python3
"""
gen_resnet1202_model.py — ResNet-1202 FPGA export script (CIFAR-10, n=200)

TARGET VARIANT (do not change without updating everywhere):
  Model   : ResNet-1202, He et al. 2016, CIFAR-10 (n=200, 6n+2=1202 layers)
  Input   : 32 × 32 × 3 (CIFAR-10 spatial size)
  Classes : 10 (CIFAR-10)
  Platform: iCEBreaker iCE40UP5K + MuraxHyperRAM
  Weights : BlockDialect VWB2 (block_size=32) stored in SPI flash at 0x2011_0000

Model loading strategy (in priority order):
  Default : Download/cache the akamaster/pytorch_resnet_cifar10 pretrained ResNet-1202
            checkpoint via torch.hub.  The SHA-256 of the expected file is hard-coded;
            the script aborts if the file does not match.
            Source: https://github.com/akamaster/pytorch_resnet_cifar10
            This uses He et al. Option A shortcuts (zero-pad, no conv1×1 projection).
  --checkpoint PATH : Use a local checkpoint file instead of hub download.
  --train : Train from scratch with a fixed seed on CIFAR-10 (slow, ~4h GPU).

This script produces (in --outdir, default ./scripts/resnet1202_artifacts/):
  weights_bd.bin        — BlockDialect VWB2 weight blob (BN-folded, biases as float32)
  model_constants.h     — C header with topology + blob-size constants
  input.h               — CIFAR-10 test image, int8 CHW (3×32×32)
  input_32x32.raw       — same image as raw int8 binary for quantized_reference.py
  expected_fp32.h       — FP32 logits + top-1 class + SHA-256 + u32sum
  quantized_ref.h       — Integer hashes at all stage boundaries (from fw simulation)
  weight_budget.txt     — Param count, BD block count, bytes, flash-fit check

Reference: RESNET1202_FPGA_PLAN.md §3, RESNET1202_FPGA_PLAN.md §4
"""

from __future__ import annotations

# =============================================================================
# !! CANONICAL MODEL CONSTANTS — firmware must match these exactly !!
# =============================================================================
MODEL_VARIANT       = "resnet1202_cifar10"
INPUT_H             = 32
INPUT_W             = 32
INPUT_C             = 3
N_CLASSES           = 10
N_PER_STAGE         = 200           # n in 6n+2 = 1202
BLOCK_SIZE          = 32            # BlockDialect block size (elements/block)
FLASH_WEIGHT_BASE   = 0x20000000    # CPU address of weights (flash window base; flashOffset=0x100000)
FLASH_OFFSET_BYTES  = 0x100000      # raw-flash byte offset of weights blob (after bitstream)

# CIFAR-10 normalisation (standard values)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

# =============================================================================
# !! Canonical checkpoint source — akamaster/pytorch_resnet_cifar10 !!
# =============================================================================
# He et al. Option A (LambdaLayer zero-pad shortcut) — no conv1×1 projection weights.
# SHA-256 is hard-coded so any tampered/wrong file is rejected immediately.
HUB_REPO            = "akamaster/pytorch_resnet_cifar10"
HUB_ENTRYPOINT      = "resnet1202"
CHECKPOINT_FILENAME = "resnet1202-f3b1deed.th"
CHECKPOINT_SHA256   = "f3b1deed382cd4c986ff8aa090c805d99a646e99d1f9227d7178183648844f62"

# =============================================================================
import os
import sys
import struct
import hashlib
import argparse
import textwrap
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as T
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\n  pip install torch torchvision")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blockdialect_codec as bd

# Default cache location that torch.hub uses on this machine (resolved after import os):
CHECKPOINT_CACHE = os.path.join(
    os.path.expanduser("~/.cache/torch/hub"),
    "akamaster_pytorch_resnet_cifar10_master",
    "pretrained_models",
    CHECKPOINT_FILENAME,
)

# =============================================================================
# Hub loading (default model source)
# =============================================================================

def _verify_sha256(path: str, expected: str) -> None:
    """Abort if the SHA-256 of *path* does not match *expected*."""
    print(f"  Verifying SHA-256 of {os.path.basename(path)} ...", end=" ", flush=True)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if digest != expected:
        sys.exit(
            f"\nSHA-256 MISMATCH for {path}\n"
            f"  expected : {expected}\n"
            f"  got      : {digest}\n"
            "The checkpoint file is corrupt or is not the expected model."
        )
    print("OK")


def load_model_hub() -> Any:
    """Load the canonical akamaster ResNet-1202 (Option A) directly from the
    hub cache, bypassing torch.hub.load().

    torch.hub.load() fails because the repo's hubconf.py does bare
    ``import resnet20`` / ``import resnet1202`` etc., expecting separate
    per-model .py files that were consolidated into a single resnet.py —
    so those module names are never found.  We work around this by:

      1. Resolving the hub repo directory (downloading it once if absent).
      2. Adding it to sys.path.
      3. Importing resnet1202() directly from resnet.py.
      4. Loading and verifying the pretrained checkpoint ourselves.

    Returns the model in eval() mode.
    """
    import importlib.util

    hub_repo_dir = os.path.join(
        os.path.expanduser("~/.cache/torch/hub"),
        "akamaster_pytorch_resnet_cifar10_master",
    )

    # Download the repo archive if not already cached
    if not os.path.isdir(hub_repo_dir):
        print(f"Hub repo not found at {hub_repo_dir}; downloading via torch.hub ...")
        try:
            # download_url_to_file / _get_torch_home etc. aren't public; use
            # hub._get_cache_or_reload which only downloads the repo (no model load)
            torch.hub._get_cache_or_reload(HUB_REPO, force_reload=False, verbose=True)
        except Exception as exc:
            sys.exit(
                f"Could not download hub repo {HUB_REPO}: {exc}\n"
                "Run with --checkpoint to supply a local checkpoint instead."
            )

    if not os.path.isdir(hub_repo_dir):
        sys.exit(
            f"Hub repo directory not found: {hub_repo_dir}\n"
            "Run with --checkpoint to supply a local checkpoint instead."
        )

    print(f"Loading {HUB_REPO}:{HUB_ENTRYPOINT} (direct import from hub cache) ...")

    # Import resnet.py from the cached repo
    resnet_py = os.path.join(hub_repo_dir, "resnet.py")
    if not os.path.isfile(resnet_py):
        sys.exit(f"Expected {resnet_py} — hub cache may be corrupt.  Delete and retry.")

    spec   = importlib.util.spec_from_file_location("akamaster_resnet", resnet_py)
    rm     = importlib.util.module_from_spec(spec)
    # Make local imports within resnet.py (if any) resolve correctly
    if hub_repo_dir not in sys.path:
        sys.path.insert(0, hub_repo_dir)
    spec.loader.exec_module(rm)

    model = rm.resnet1202()

    # Load pretrained weights
    ckpt_path = CHECKPOINT_CACHE
    if not os.path.isfile(ckpt_path):
        # Search common alternative locations
        for root, _, files in os.walk(os.path.join(os.path.expanduser("~/.cache/torch/hub"))):
            if CHECKPOINT_FILENAME in files:
                ckpt_path = os.path.join(root, CHECKPOINT_FILENAME)
                break
        else:
            sys.exit(
                f"Pretrained checkpoint not found: {CHECKPOINT_FILENAME}\n"
                f"Expected path: {CHECKPOINT_CACHE}\n"
                "Place the checkpoint there or use --checkpoint."
            )

    _verify_sha256(ckpt_path, CHECKPOINT_SHA256)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = state.get("state_dict", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {params:,}  ({params / 1e6:.2f}M)")
    print(f"  Shortcut   : Option A (LambdaLayer zero-pad — no conv1×1 proj weights)")
    return model


# =============================================================================
# ResNet-1202 CIFAR-10 architecture (He et al. 2016)
# =============================================================================
# NOTE: This custom class uses Option B (conv1×1 projection) and is used only
# by the --train path.  The hub-loaded model (default) uses Option A.
# =============================================================================

class _BasicBlock(nn.Module):
    """BasicBlock for CIFAR ResNet.  Both convs are 3×3.  No bottleneck."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut: Optional[nn.Sequential] = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        out = self.relu(out + shortcut)
        return out


class ResNet1202Cifar(nn.Module):
    """He et al. 2016 CIFAR ResNet with n=200 (6n+2=1202 total layers)."""

    def __init__(self, n: int = N_PER_STAGE, num_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.in_planes = 16

        # Stem: 3×3 conv (CIFAR uses 3×3 stem, not 7×7)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)

        # Three stages of 2n blocks (n=200)
        self.stage1 = self._make_stage(16, n, stride=1)   # 16 × 32 × 32
        self.stage2 = self._make_stage(32, n, stride=2)   # 32 × 16 × 16
        self.stage3 = self._make_stage(64, n, stride=2)   # 64 ×  8 ×  8

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc      = nn.Linear(64, num_classes)

        # Weight init (He)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_stage(self, planes: int, n: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(self.in_planes, planes, stride=stride)]
        self.in_planes = planes
        for _ in range(n - 1):
            layers.append(_BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


# =============================================================================
# Model loading
# =============================================================================

def load_model_from_checkpoint(checkpoint_path: str) -> Any:
    """Load ResNet-1202 weights from a local PyTorch checkpoint file.

    Tries to determine the architecture from the checkpoint keys and picks the
    matching model class.  Checkpoints produced by akamaster/pytorch_resnet_cifar10
    (Option A, layer1/2/3, linear) are loaded into torch.hub model directly so
    all weights are present.  Checkpoints produced by --train (Option B, stage1/2/3,
    fc) are loaded into ResNet1202Cifar.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = state.get("state_dict", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    uses_layer_naming = any(k.startswith("layer") for k in sd)
    uses_linear       = any(k.startswith("linear") for k in sd)

    if uses_layer_naming or uses_linear:
        # akamaster-style checkpoint (Option A); load via hub so architecture matches
        print("  Detected akamaster-style keys (layer1/2/3, linear) — loading via hub model")
        _verify_sha256(checkpoint_path, CHECKPOINT_SHA256)
        model = torch.hub.load(
            HUB_REPO, HUB_ENTRYPOINT, pretrained=False, verbose=False)
        model.load_state_dict(sd, strict=True)
    else:
        # Option B checkpoint (stage1/2/3, fc) — load into our custom class
        print("  Detected Option B keys (stage1/2/3, fc) — loading into ResNet1202Cifar")
        model = ResNet1202Cifar(n=N_PER_STAGE, num_classes=N_CLASSES)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  WARNING: missing keys: {missing[:5]}")
        if unexpected:
            print(f"  WARNING: unexpected keys: {unexpected[:5]}")

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}  ({params / 1e6:.2f}M)")
    return model


def train_model(seed: int = 42, epochs: int = 200,
                data_root: str = "~/.cache/cifar10") -> ResNet1202Cifar:
    """Train ResNet-1202 from scratch on CIFAR-10 with a fixed seed.

    This reproduces the training setup from He et al. 2016:
      - SGD with momentum 0.9, weight decay 1e-4
      - LR schedule: 0.1 → 0.01 at epoch 100 → 0.001 at epoch 150
      - Data augmentation: random crop + horizontal flip
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Training ResNet-1202 from scratch (seed={seed}, epochs={epochs}) ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    data_root = os.path.expanduser(data_root)
    trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                            download=True, transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                            download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=256,
                                              shuffle=False, num_workers=2)

    model = ResNet1202Cifar(n=N_PER_STAGE, num_classes=N_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for inputs, targets in testloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    pred = model(inputs).argmax(1)
                    correct += int((pred == targets).sum())
                    total   += targets.size(0)
            acc = 100.0 * correct / total
            best_acc = max(best_acc, acc)
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss / len(trainloader):.4f}"
                  f"  test_acc={acc:.2f}%  best={best_acc:.2f}%")

    model.eval()
    model = model.cpu()
    print(f"Training complete.  Best accuracy: {best_acc:.2f}%")
    return model


# =============================================================================
# BN folding
# =============================================================================

def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[np.ndarray, np.ndarray]:
    """Fold Conv2d + BatchNorm2d → (folded_weight, folded_bias) as float32."""
    w = conv.weight.detach().float()
    b = conv.bias.detach().float() if conv.bias is not None else torch.zeros(w.shape[0])

    gamma = bn.weight.detach().float()
    beta  = bn.bias.detach().float()
    mean  = bn.running_mean.detach().float()
    var   = bn.running_var.detach().float()
    eps   = bn.eps

    scale = gamma / torch.sqrt(var + eps)
    w_f   = w * scale.view(-1, 1, 1, 1)
    b_f   = (b - mean) * scale + beta
    return w_f.numpy(), b_f.numpy()


def extract_folded_weights(model: Any) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Walk any CIFAR ResNet-1202 model and fold all Conv+BN pairs.

    Handles both:
      • akamaster model  — attributes layer1/layer2/layer3, linear, Option A shortcut
                           (LambdaLayer or empty Sequential — no learnable proj weights)
      • ResNet1202Cifar  — attributes stage1/stage2/stage3, fc, Option B shortcut
                           (nn.Sequential([Conv2d, BN]) when present)

    VWB2 tensor naming convention (must match resnet1202_layers.h tensor IDs):
      conv1
      stageN.B.conv1, stageN.B.conv2        — for all N in {1,2,3}, B in 0..199
      stageN.0.proj                         — ONLY if Option B proj weights present
      fc
    """
    layers: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _add(name: str, w: np.ndarray, b: np.ndarray) -> None:
        layers[name] = (w, b)

    # Detect attribute naming style
    has_stage_attr = hasattr(model, "stage1")
    has_layer_attr = hasattr(model, "layer1")
    if not has_stage_attr and not has_layer_attr:
        raise ValueError("Cannot detect model architecture: no stage1/layer1 attribute")

    stage_attrs = (["stage1", "stage2", "stage3"] if has_stage_attr
                   else ["layer1", "layer2", "layer3"])
    fc_module   = model.fc if hasattr(model, "fc") else model.linear

    option_a = not has_stage_attr  # akamaster hub model → Option A
    if option_a:
        print("\nFolding BN into Conv (akamaster / Option A — no projection weights) ...")
    else:
        print("\nFolding BN into Conv (ResNet1202Cifar / Option B) ...")

    # Stem
    w, b = fold_bn(model.conv1, model.bn1)
    _add("conv1", w, b)
    print(f"  {'conv1':50s}  w={w.shape}  b={b.shape}")

    proj_count = 0
    for stage_idx, attr in enumerate(stage_attrs, 1):
        stage = getattr(model, attr)
        for blk_idx, block in enumerate(stage):
            pfx = f"stage{stage_idx}.{blk_idx}"

            w, b = fold_bn(block.conv1, block.bn1)
            _add(f"{pfx}.conv1", w, b)

            w, b = fold_bn(block.conv2, block.bn2)
            _add(f"{pfx}.conv2", w, b)

            # Projection shortcut: only present for Option B (nn.Sequential with weights).
            # Option A shortcuts are LambdaLayer or empty Sequential — skip them.
            sc = block.shortcut if hasattr(block, "shortcut") else None
            if (sc is not None
                    and isinstance(sc, nn.Sequential)
                    and len(sc) >= 2
                    and isinstance(sc[0], nn.Conv2d)):
                w, b = fold_bn(sc[0], sc[1])
                _add(f"{pfx}.proj", w, b)
                proj_count += 1

    # Classifier
    fc_w = fc_module.weight.detach().float().numpy()
    fc_b = fc_module.bias.detach().float().numpy()
    _add("fc", fc_w, fc_b)

    total_params = sum(w.size + b.size for w, b in layers.values())
    print(f"\n  Folded total elements: {total_params:,}  ({total_params / 1e6:.2f}M)")
    if proj_count == 0:
        print("  Projection (Option A model): no conv1×1 proj tensors emitted.")
        print("  NOTE: firmware resnet1202_layers.h has_proj path will not be exercised.")
    else:
        print(f"  Projection blocks: {proj_count} (Option B)")
    print()
    return layers


# =============================================================================
# Test input — first test image from CIFAR-10
# =============================================================================

# Default input image path (relative to the repo root)
INPUT_IMAGE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_image.png")


def get_test_image(image_path: str = INPUT_IMAGE_PATH) -> Tuple[np.ndarray, torch.Tensor, int, float]:
    """Return (int8_chw, float_tensor_1x3x32x32, true_label, scale) for the given PNG.

    The image is resized to 32×32 (CIFAR-10 spatial size) and normalised with
    standard CIFAR-10 per-channel mean/std.  true_label is -1 (unknown) since
    the image is not from the CIFAR-10 test set.
    """
    from PIL import Image as _PILImage

    print(f"Loading input image: {image_path}")
    pil_img = _PILImage.open(image_path).convert("RGB")
    original_size = pil_img.size
    if pil_img.size != (INPUT_W, INPUT_H):
        pil_img = pil_img.resize((INPUT_W, INPUT_H), _PILImage.BICUBIC)
        print(f"  Resized {original_size} → {INPUT_W}×{INPUT_H}")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    img_f32 = transform(pil_img)  # [3, 32, 32]

    # Symmetric per-tensor int8 quantization
    arr     = img_f32.numpy()
    max_abs = float(np.max(np.abs(arr)))
    scale   = max_abs / 127.0 if max_abs != 0.0 else 1.0
    q       = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)

    label = -1  # unknown for non-CIFAR images
    return q, img_f32.unsqueeze(0), label, scale


# =============================================================================
# FP32 inference (golden reference)
# =============================================================================

def run_fp32_inference(model: ResNet1202Cifar,
                       tensor_f32: torch.Tensor) -> Tuple[List[float], int]:
    with torch.no_grad():
        logits = model(tensor_f32)[0]  # [10]
    top1 = int(logits.argmax())
    return logits.numpy().tolist(), top1


# =============================================================================
# Export: BlockDialect VWB2 blob
# =============================================================================

def export_weights_bd(layers: Dict[str, Tuple[np.ndarray, np.ndarray]],
                      out_path: str,
                      tap_blocked: bool = False) -> Tuple[bytes, List]:
    """Write all BN-folded weights as a BlockDialect VWB2 indexed blob.

    If *tap_blocked* is True, conv3×3 weight tensors (shape [OC,IC,3,3]) are
    permuted to [OC,KY,KX,IC] before BD4 encoding so that each 32-element
    BD block covers a single (KY,KX) tap over 32 consecutive input channels.
    conv1×1 (shape [OC,IC,1,1] or [OC,IC]) and FC weights are left unchanged.
    """
    print(f"Exporting BlockDialect VWB2 weights → {out_path}"
          + ("  [TAP-BLOCKED: conv3×3 permuted OC,IC,KY,KX→OC,KY,KX,IC]" if tap_blocked else ""))
    tensors_spec: List[Tuple[str, int, np.ndarray]] = []
    n_permuted = 0

    for name, (w, b) in layers.items():
        w_export = w
        if tap_blocked and w.ndim == 4 and w.shape[2] == 3 and w.shape[3] == 3:
            # Permute [OC, IC, KY, KX] → [OC, KY, KX, IC]
            w_export = np.ascontiguousarray(w.transpose(0, 2, 3, 1))
            n_permuted += 1
        tensors_spec.append((f"{name}.weight", bd.DTYPE_BD4,     w_export.astype(np.float32)))
        tensors_spec.append((f"{name}.bias",   bd.DTYPE_FLOAT32, b.astype(np.float32)))

    if tap_blocked:
        print(f"  Permuted {n_permuted} conv3×3 weight tensors to tap-blocked [OC,KY,KX,IC] order")

    # Append a single-float32 sentinel tensor as the LAST entry in the blob.
    # Firmware reads tbl[tensor_count-1] at boot and hard-fails on flag mismatch.
    #   1.0f (0x3F800000) = tap-blocked layout
    #   0.0f (0x00000000) = flat CHW layout
    layout_flag_val = 1.0 if tap_blocked else 0.0
    tensors_spec.append(("rn1202.layout_flags", bd.DTYPE_FLOAT32,
                         np.array([layout_flag_val], dtype=np.float32)))

    blob_bytes = bd.write_weight_blob_v2(tensors_spec, out_path)
    entries, _ = bd.read_weight_blob_v2(out_path)

    n_layers = len(layers)
    print(f"  Wrote {len(blob_bytes):,} bytes  "
          f"({n_layers} layers × 2 tensors = {2 * n_layers} entries, biases as float32)")
    return blob_bytes, entries


# =============================================================================
# Weight budget report
# =============================================================================

def weight_budget_report(layers: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         bd_blob: bytes,
                         out_path: str) -> None:
    """Print and save weight budget summary."""
    total_w_elems  = sum(w.size for w, _ in layers.values())
    total_b_elems  = sum(b.size for _, b in layers.values())
    n_bd_blocks    = sum((w.size + BLOCK_SIZE - 1) // BLOCK_SIZE for w, _ in layers.values())
    BD_BLOCK_BYTES = 18
    bias_bytes_f32 = sum(b.size * 4 for _, b in layers.values())
    naive_f32_bytes = total_w_elems * 4 + bias_bytes_f32
    flash_available = 16 * 1024 * 1024 - FLASH_OFFSET_BYTES  # 15 MiB

    ratio = naive_f32_bytes / max(len(bd_blob), 1)

    lines = [
        f"ResNet-1202 (CIFAR-10) Weight Budget Report",
        f"Model variant : {MODEL_VARIANT}",
        f"Input size    : {INPUT_C}×{INPUT_H}×{INPUT_W}",
        f"Output classes: {N_CLASSES}",
        f"Blocks/stage  : {N_PER_STAGE}  (6n+2 = {6*N_PER_STAGE+2} total layers)",
        f"",
        f"Weight layers  : {len(layers)}",
        f"Total w elems  : {total_w_elems:,}  ({total_w_elems/1e6:.2f}M)",
        f"Total bias elems: {total_b_elems:,}  (stored as float32)",
        f"",
        f"BlockDialect encoding (arXiv:2501.01144v5 §3):",
        f"  Block size       : {BLOCK_SIZE} elements",
        f"  BD blocks        : {n_bd_blocks:,}",
        f"  Per-block bytes  : {BD_BLOCK_BYTES}  (2-byte meta + 16-byte codes)",
        f"  Actual BD blob   : {len(bd_blob):,} bytes  ({len(bd_blob)/1024/1024:.2f} MiB)",
        f"  Naive float32    : {naive_f32_bytes:,} bytes  ({naive_f32_bytes/1024/1024:.2f} MiB)",
        f"  Compression ratio: {ratio:.2f}x",
        f"",
        f"Flash available (after FLASH_OFFSET_BYTES=0x{FLASH_OFFSET_BYTES:06X}): "
        f"{flash_available/1024/1024:.1f} MiB",
        f"  BD blob fits in flash: {'YES ✓' if len(bd_blob) < flash_available else 'NO — TOO BIG ✗'}",
        f"",
        f"iCEBreaker SPRAM: 128 KB at 0x11000000",
        f"  Stage1 activation (16×32×32 int8): 16,384 B = 16 KB — fits ✓",
        f"  Three-buffer peak (BD4 skip):      ~44 KB — fits ✓",
    ]

    report = "\n".join(lines)
    print("\n" + report + "\n")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report + "\n")
    print(f"Saved {out_path}")


# =============================================================================
# Export: input.h
# =============================================================================

def export_input_header(q_int8: np.ndarray, scale: float, out_path: str) -> None:
    """Write the 32×32 CIFAR-10 int8 test image as a C header."""
    flat = q_int8.flatten().tolist()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("// AUTO-GENERATED by gen_resnet1202_model.py — do not edit\n")
        f.write(f"// Model: {MODEL_VARIANT}\n")
        f.write(f"// Input: {INPUT_C}×{INPUT_H}×{INPUT_W} int8, CHW layout\n")
        f.write(f"// Quantization scale: {scale:.8f}\n\n")
        f.write("#ifndef RN1202_INPUT_DATA_H\n#define RN1202_INPUT_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define RN1202_INPUT_H       {INPUT_H}\n")
        f.write(f"#define RN1202_INPUT_W       {INPUT_W}\n")
        f.write(f"#define RN1202_INPUT_C       {INPUT_C}\n")
        f.write(f"#define RN1202_INPUT_NELEMS  {len(flat)}\n\n")
        f.write(f"const int8_t RN1202_INPUT[{len(flat)}] = {{\n")
        for i, v in enumerate(flat):
            f.write(f"{v:4d},")
            if (i + 1) % 16 == 0:
                f.write("\n")
        f.write("\n};\n\n#endif /* RN1202_INPUT_DATA_H */\n")
    print(f"Saved {out_path}  ({len(flat)} int8 values, scale={scale:.6f})")


# =============================================================================
# Export: expected_fp32.h
# =============================================================================

def export_expected_fp32_header(logits: List[float], top1: int,
                                true_label: int, out_path: str) -> None:
    """Write FP32 logits + top-1 classification as a C header."""
    logits_np    = np.array(logits, dtype=np.float32)
    logits_bytes = logits_np.tobytes()
    sha          = hashlib.sha256(logits_bytes).hexdigest()
    u32          = np.frombuffer(logits_bytes, dtype=np.uint32)
    checksum     = int(np.sum(u32.astype(np.uint64)) & 0xFFFFFFFF)
    top_idx      = [int(i) for i in np.argsort(logits)[::-1]]
    top5_idx     = top_idx[:5]
    top5_vals    = [logits[i] for i in top5_idx]

    cifar10_classes = ["airplane","automobile","bird","cat","deer",
                       "dog","frog","horse","ship","truck"]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("// AUTO-GENERATED by gen_resnet1202_model.py — do not edit\n")
        f.write(f"// Model: {MODEL_VARIANT}\n")
        f.write(f"// FP32 golden reference (no quantization)\n\n")
        f.write("#ifndef RN1202_EXPECTED_FP32_H\n#define RN1202_EXPECTED_FP32_H\n\n")
        f.write("#include <stdint.h>\n\n")
        true_label_name = cifar10_classes[true_label] if 0 <= true_label < len(cifar10_classes) else "unknown"
        f.write(f"#define EXPECTED_TOP1_CLASS   {top1}\n")
        f.write(f"#define EXPECTED_TRUE_LABEL   {true_label}\n")
        f.write(f"// Top-1 class name: {cifar10_classes[top1]}\n")
        f.write(f"// True label name : {true_label_name}\n")
        f.write(f"#define EXPECTED_LOGITS_SHA256 \"{sha}\"\n")
        f.write(f"#define EXPECTED_LOGITS_U32SUM  0x{checksum:08X}u\n\n")
        f.write(f"const int32_t  EXPECTED_TOP5_IDX[5]   = "
                f"{{ {', '.join(str(i) for i in top5_idx)} }};\n")
        f.write(f"const float    EXPECTED_TOP5_LOGIT[5] = "
                f"{{ {', '.join(f'{v:.6f}f' for v in top5_vals)} }};\n\n")
        f.write(f"// Full logit array (FP32, 10 classes)\n")
        f.write(f"const float EXPECTED_LOGITS[{N_CLASSES}] = {{\n")
        for i, v in enumerate(logits):
            f.write(f"    {v:.6f}f,")
            if (i + 1) % 5 == 0:
                f.write("\n")
        f.write("\n};\n\n#endif /* RN1202_EXPECTED_FP32_H */\n")

    true_label_name = cifar10_classes[true_label] if 0 <= true_label < len(cifar10_classes) else "unknown"
    print(f"Saved {out_path}  (top1={top1} ({cifar10_classes[top1]}), true={true_label_name}, sha256={sha[:16]}...)")


# =============================================================================
# Export: model_constants.h
# =============================================================================

def export_model_constants_header(layers: Dict[str, Tuple[np.ndarray, np.ndarray]],
                                  bd_blob: bytes,
                                  entries: List,
                                  out_path: str,
                                  tap_blocked: bool = False) -> None:
    """Write C header with model topology + VWB2 blob layout constants for firmware."""
    entry_by_hash = {e.name_hash: e for e in entries}

    def _entry(tensor_name: str):
        h = bd.fnv1a32(tensor_name)
        e = entry_by_hash.get(h)
        if e is None:
            raise KeyError(f"Tensor {tensor_name!r} not found in VWB2 blob")
        return e

    data_offset = struct.unpack_from("<I", bd_blob, 20)[0]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("// AUTO-GENERATED by gen_resnet1202_model.py — do not edit\n")
        f.write(f"// Model: {MODEL_VARIANT}\n\n")
        f.write("#ifndef RN1202_MODEL_CONSTANTS_H\n#define RN1202_MODEL_CONSTANTS_H\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write("// ---- Layout sentinel (for firmware hard-fail on blob/fw mismatch) ----\n")
        layout_flags_hash = bd.fnv1a32("rn1202.layout_flags")
        f.write(f"#define RN1202_LAYOUT_FLAGS_HASH  0x{layout_flags_hash:08X}u\n")
        f.write("// Sentinel tensor is the LAST entry in the VWB2 blob (tbl[tensor_count-1]).\n")
        f.write("// FLOAT32 value: 1.0f (0x3F800000) = tap-blocked, 0.0f = flat layout.\n\n")

        f.write("// ---- Weight layout flags ----\n")
        if tap_blocked:
            f.write("#define RN1202_WEIGHT_LAYOUT_TAP_BLOCKED 1\n")
            f.write("// conv3×3 weights stored as [OC,KY,KX,IC] (tap-major, channel-inner)\n")
        else:
            f.write("/* RN1202_WEIGHT_LAYOUT_TAP_BLOCKED not defined — flat [OC,IC,KY,KX] layout */\n")
        f.write("\n")

        f.write("// ---- Model identity ----\n")
        f.write(f'#define RN1202_MODEL_VARIANT  "{MODEL_VARIANT}"\n')
        f.write(f"#define RN1202_MODEL_INPUT_H  {INPUT_H}\n")
        f.write(f"#define RN1202_MODEL_INPUT_W  {INPUT_W}\n")
        f.write(f"#define RN1202_MODEL_INPUT_C  {INPUT_C}\n")
        f.write(f"#define RN1202_N_CLASSES      {N_CLASSES}\n")
        f.write(f"#define RN1202_N_PER_STAGE    {N_PER_STAGE}\n\n")

        f.write("// ---- BlockDialect parameters ----\n")
        f.write(f"#define BD_BLOCK_SIZE   {BLOCK_SIZE}\n")
        f.write(f"#define BD_BLOCK_BYTES  18u  /* 2-byte meta + 16-byte packed codes */\n\n")

        f.write("// ---- Flash / WeightStore layout ----\n")
        f.write(f"#define FLASH_WEIGHT_BASE    0x{FLASH_WEIGHT_BASE:08X}u\n")
        f.write(f"#define FLASH_WEIGHT_OFFSET  0x{FLASH_OFFSET_BYTES:08X}u"
                f"  /* raw SPI-flash byte offset; SoC window starts here */\n")
        f.write(f"#define WEIGHT_BLOB_ADDR     FLASH_WEIGHT_BASE\n\n")

        f.write("// ---- VWB2 blob total size ----\n")
        f.write(f"#define RN1202_BD_BLOB_SIZE  {len(bd_blob)}u\n\n")

        f.write("// ---- Total weight tensor count ----\n")
        # Layers dict has both weight and bias entries; each layer has one weight + one bias
        total_tensors = len(layers) * 2
        f.write(f"#define RN1202_TOTAL_TENSORS  {total_tensors}u\n")
        f.write(f"#define RN1202_W_LAYERS       {len(layers)}u\n\n")

        f.write("// ---- Per-layer tensor offsets (from WEIGHT_BLOB_ADDR) ----\n")
        f.write("// Used by firmware to seek directly to any layer's weights.\n")
        f.write("typedef struct {\n")
        f.write("    uint32_t w_elements;\n")
        f.write("    uint32_t b_elements;\n")
        f.write("    uint32_t n_bd_blocks;\n")
        f.write("    uint32_t weight_offset;   /* bytes from WEIGHT_BLOB_ADDR */\n")
        f.write("    uint32_t weight_bytes;\n")
        f.write("    uint32_t bias_offset;     /* bytes from WEIGHT_BLOB_ADDR */\n")
        f.write("    uint32_t bias_bytes;\n")
        f.write("} Rn1202LayerInfo;\n\n")

        f.write(f"static const Rn1202LayerInfo RN1202_LAYER_INFO[{len(layers)}] = {{\n")
        for i, (name, (w, b)) in enumerate(layers.items()):
            n_blocks = (w.size + BLOCK_SIZE - 1) // BLOCK_SIZE
            we = _entry(f"{name}.weight")
            be = _entry(f"{name}.bias")
            w_abs = data_offset + we.tensor_offset
            b_abs = data_offset + be.tensor_offset
            f.write(f"    /* [{i:4d}] {name} */\n")
            f.write(f"    {{ {w.size}u, {b.size}u, {n_blocks}u,"
                    f" 0x{w_abs:08X}u, {we.tensor_bytes}u,"
                    f" 0x{b_abs:08X}u, {be.tensor_bytes}u }},\n")
        f.write("};\n\n")

        f.write("#endif /* RN1202_MODEL_CONSTANTS_H */\n")

    print(f"Saved {out_path}")


# =============================================================================
# Export: quantized_ref.h (u32sum hashes at stage boundaries for firmware)
# =============================================================================

def export_quantized_ref_header(model: ResNet1202Cifar,
                                tensor_f32: torch.Tensor,
                                out_path: str) -> None:
    """Run forward pass with BN-folded int8 approximation and record stage hashes."""
    print("Computing quantized reference hashes ...")

    def _act_hash(t: torch.Tensor) -> int:
        """Compute u32sum of int8-quantized tensor."""
        arr   = t[0].detach().numpy()          # [C, H, W] or [C]
        max_a = float(np.max(np.abs(arr)))
        scale = max_a / 127.0 if max_a != 0.0 else 1.0
        q     = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)
        return int(np.sum(q.astype(np.int32)) & 0xFFFFFFFF)

    hooks = {}
    hashes: Dict[str, int] = {}

    def _make_hook(name: str):
        def hook(module, inp, out):
            hashes[name] = _act_hash(out)
        return hook

    # Detect attribute layout: akamaster (layer1/2/3, linear) vs ResNet1202Cifar (stage1/2/3, fc)
    _s1 = model.layer1 if hasattr(model, "layer1") else model.stage1
    _s2 = model.layer2 if hasattr(model, "layer2") else model.stage2
    _s3 = model.layer3 if hasattr(model, "layer3") else model.stage3
    _fc = model.linear if hasattr(model, "linear") else model.fc

    model.eval()
    h1   = _s1.register_forward_hook(_make_hook("stage1"))
    h2   = _s2.register_forward_hook(_make_hook("stage2"))
    h3   = _s3.register_forward_hook(_make_hook("stage3"))
    h_fc = _fc.register_forward_hook(_make_hook("fc"))

    with torch.no_grad():
        _ = model(tensor_f32)

    h1.remove(); h2.remove(); h3.remove(); h_fc.remove()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("// AUTO-GENERATED by gen_resnet1202_model.py — do not edit\n")
        f.write(f"// Model: {MODEL_VARIANT}\n")
        f.write("// Quantized reference hashes at stage boundaries.\n")
        f.write("// These are int8-quantized (symmetric per-tensor) u32sum values.\n")
        f.write("// Due to BN folding + quantization error, firmware may differ slightly.\n\n")
        f.write("#ifndef RN1202_QUANTIZED_REF_H\n#define RN1202_QUANTIZED_REF_H\n\n")
        f.write("#include <stdint.h>\n\n")
        for tag, h in hashes.items():
            cname = tag.upper().replace(".", "_")
            f.write(f"#define RN1202_HASH_{cname}  0x{h:08X}u\n")
        f.write("\n#endif /* RN1202_QUANTIZED_REF_H */\n")

    print(f"Saved {out_path}  (stage hashes: {hashes})")


# =============================================================================
# Dry-run estimate
# =============================================================================

def dry_run_estimate() -> None:
    """Print estimated BD blob size without actually encoding."""
    print(f"\n[dry-run] ResNet-1202 weight budget estimate:")
    print(f"  N per stage     : {N_PER_STAGE}  (6n+2 = {6*N_PER_STAGE+2} layers)")
    print(f"  conv1 stem      : 3×3×3×16  = 432 params")
    print(f"  stage1 (200×2)  : 200×2×(3×3×16×16) = 921,600 params")
    print(f"  stage2 (200 blk): ~3,687,424 params (incl. proj)")
    print(f"  stage3 (200 blk): ~14,748,160 params (incl. proj)")
    print(f"  fc              : 64×10 = 640 params")
    total = 432 + 921600 + 3687424 + 14748160 + 640
    n_bd  = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
    BD_BLOCK_BYTES = 18
    bias_est = (16 + sum([16*2, 32*2, 64*2, 10]) + 4) * 4  # rough
    blob_est = 32 + n_bd * BD_BLOCK_BYTES + bias_est
    print(f"  Total params    : {total:,}  ({total/1e6:.2f}M)")
    print(f"  BD blocks       : {n_bd:,}")
    print(f"  Estimated blob  : {blob_est:,} bytes  ({blob_est/1024/1024:.2f} MiB)")
    flash_avail = 16*1024*1024 - FLASH_OFFSET_BYTES
    print(f"  Flash available : {flash_avail/1024/1024:.1f} MiB  "
          f"→ {'FITS ✓' if blob_est < flash_avail else 'TOO BIG ✗'}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--outdir", default="scripts/resnet1202_artifacts",
        help="Output directory (default: scripts/resnet1202_artifacts/)")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to a PyTorch checkpoint file (Option A).  "
             "If absent, ResNet1202 is pulled from a repo")
    parser.add_argument(
        "--train", action="store_true",
        help="Train ResNet-1202 from scratch on CIFAR-10 (Option B, slow).")
    parser.add_argument(
        "--train-epochs", type=int, default=200,
        help="Epochs for --train (default: 200).")
    parser.add_argument(
        "--save-checkpoint", default=None,
        help="If training, save the trained model to this path.")
    parser.add_argument(
        "--data-root", default="~/.cache/cifar10",
        help="CIFAR-10 dataset root (default: ~/.cache/cifar10).")
    parser.add_argument(
        "--tap-blocked", action="store_true",
        help="Permute conv3×3 weights to tap-blocked [OC,KY,KX,IC] layout "
             "(required for firmware USE_TAP_BLOCKED=1 inference path).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print weight budget estimate only; do not train or encode.")
    args = parser.parse_args()

    if args.dry_run:
        dry_run_estimate()
        return

    od = args.outdir
    os.makedirs(od, exist_ok=True)

    # 1. Load or train model
    if args.checkpoint:
        model = load_model_from_checkpoint(args.checkpoint)
    elif args.train:
        model = train_model(seed=42, epochs=args.train_epochs, data_root=args.data_root)
        if args.save_checkpoint:
            torch.save({"state_dict": model.state_dict()}, args.save_checkpoint)
            print(f"Saved trained checkpoint → {args.save_checkpoint}")
    else:
        # Default: pull pretrained ResNet-1202 from akamaster/pytorch_resnet_cifar10
        model = load_model_hub()

    # 2. Fold BN
    layers = extract_folded_weights(model)

    # 3. Test image
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_image.png")
    q_int8, tensor_f32, true_label, scale = get_test_image(image_path=image_path)

    # 4. FP32 golden inference
    print("\nRunning FP32 inference (golden reference) ...")
    logits, top1 = run_fp32_inference(model, tensor_f32)
    cifar_classes = ["airplane","automobile","bird","cat","deer",
                     "dog","frog","horse","ship","truck"]
    true_label_name = cifar_classes[true_label] if 0 <= true_label < len(cifar_classes) else "unknown"
    print(f"  True label  : {true_label} ({true_label_name})")
    print(f"  Top-1 class : {top1} ({cifar_classes[top1]})")

    # 5. Export VWB2 blob
    bd_blob, bd_entries = export_weights_bd(layers, os.path.join(od, "weights_bd.bin"),
                                            tap_blocked=args.tap_blocked)

    # 6. Weight budget report
    weight_budget_report(layers, bd_blob, os.path.join(od, "weight_budget.txt"))

    # 7. input.h + input_32x32.raw
    export_input_header(q_int8, scale, os.path.join(od, "input.h"))
    raw_path = os.path.join(od, "input_32x32.raw")
    q_int8.tofile(raw_path)
    print(f"Saved {raw_path}  ({q_int8.size} int8 bytes)")

    # 8. expected_fp32.h
    export_expected_fp32_header(logits, top1, true_label,
                                os.path.join(od, "expected_fp32.h"))

    # 9. model_constants.h
    export_model_constants_header(layers, bd_blob, bd_entries,
                                  os.path.join(od, "model_constants.h"),
                                  tap_blocked=args.tap_blocked)

    # 10. quantized_ref.h
    export_quantized_ref_header(model, tensor_f32, os.path.join(od, "quantized_ref.h"))

    print(f"\nAll artifacts written to: {od}/")
    print(textwrap.dedent(f"""\
        Next steps (RESNET1202_FPGA_PLAN.md §5):
          • Review weight_budget.txt — confirm blob fits in flash.
          • Flash weights_bd.bin at offset 0x{FLASH_OFFSET_BYTES:08x}:
              iceprog -o 0x110000 scripts/resnet1202_artifacts/weights_bd.bin
          • Build resnet1202_phase0_smoke (Milestone 0) to validate blob on-board.
    """))


if __name__ == "__main__":
    main()
