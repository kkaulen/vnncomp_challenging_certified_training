#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
import urllib.request
import zipfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from xml.etree import ElementTree

import numpy as np
import onnx
import requests
import torch
import torch.nn as nn
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_DATA_ROOT = SCRIPT_DIR / "data"
DEFAULT_ONNX_ZIP = SCRIPT_DIR / "onnx_models.zip"
TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
SCIEBO_ONNX_SHARE_URL = "https://rwth-aachen.sciebo.de/s/zr2GXGNWwjyWrBX"
SCIEBO_ONNX_SHARE_PASSWORD = "cw8PkR3JgL"

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], dtype=torch.float32).view(3, 1, 1)
CIFAR_STD = torch.tensor([0.2023, 0.1994, 0.2010], dtype=torch.float32).view(3, 1, 1)
TINY_MEAN = torch.tensor([0.4802, 0.4481, 0.3975], dtype=torch.float32).view(3, 1, 1)
TINY_STD = torch.tensor([0.2302, 0.2265, 0.2262], dtype=torch.float32).view(3, 1, 1)

BIN_COUNTS = OrderedDict(
    [
        ("0_10", 2),
        ("10_100", 2),
        ("100_1000", 3),
        ("timeout", 3),
    ]
)
BIN_TIMEOUTS = {
    "0_10": 30,
    "10_100": 120,
    "100_1000": 550,
    "timeout": 550,
}
EXPECTED_TOTAL_TIMEOUT = 21_600
METADATA_PATH = SCRIPT_DIR / "metadata" / "sampled_instances.json"
EXPECTED_ONNX_FILES = [
    "cifar10_eps2_cnn7.onnx",
    "cifar10_eps2_wide_cnn7.onnx",
    "cifar10_eps8_cnn7.onnx",
    "cifar10_eps8_wide_cnn7.onnx",
    "tinyimagenet_eps1_cnn7.onnx",
    "tinyimagenet_eps1_wide_cnn7.onnx",
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    dataset: str
    network: str
    width: int
    input_shape: tuple[int, int, int]
    n_classes: int
    eps: float
    checkpoint: str
    results_json: str


@dataclass(frozen=True)
class SampledInstance:
    model_key: str
    dataset: str
    network: str
    image_index: int
    bin: str
    abcrown_result: str
    abcrown_running_time: float
    timeout: int
    onnx_path: str
    vnnlib_path: str
    source_results_json: str
    source_checkpoint: str
    replacement_sampling: bool


MODEL_SPECS = [
    ModelSpec(
        key="cifar10_eps2_cnn7",
        dataset="cifar10",
        network="cnn7",
        width=64,
        input_shape=(3, 32, 32),
        n_classes=10,
        eps=2 / 255,
        checkpoint="results/diffusion_scaling/dataset=cifar/network=cnn7/eps=2_over_255/activation=relu/optimizer=adam/spectral_norm=False/scheduler=multi_step/data=x100/compute=x10/ratio=r0p3/seed=42/mtl_ibp_inc_cifar10_2_255_42.pt",
        results_json="verification_results/cifar10_eps2_cnn7.json",
    ),
    ModelSpec(
        key="cifar10_eps2_wide_cnn7",
        dataset="cifar10",
        network="wide_cnn7",
        width=128,
        input_shape=(3, 32, 32),
        n_classes=10,
        eps=2 / 255,
        checkpoint="results/diffusion_scaling/dataset=cifar/network=wide_cnn7/eps=2_over_255/activation=relu/optimizer=adam/spectral_norm=False/scheduler=multi_step/data=x100/compute=x10/ratio=r0p3/seed=42/mtl_ibp_inc_cifar10_2_255_42.pt",
        results_json="verification_results/cifar10_eps2_wide_cnn7.json",
    ),
    ModelSpec(
        key="cifar10_eps8_cnn7",
        dataset="cifar10",
        network="cnn7",
        width=64,
        input_shape=(3, 32, 32),
        n_classes=10,
        eps=8 / 255,
        checkpoint="results/diffusion_scaling/dataset=cifar/network=cnn7/eps=8_over_255/activation=relu/optimizer=adam/spectral_norm=False/scheduler=multi_step/data=x50/compute=x10/ratio=r0p3/seed=42/mtl_ibp_inc_cifar10_2_255_42.pt",
        results_json="verification_results/cifar10_eps8_cnn7.json",
    ),
    ModelSpec(
        key="cifar10_eps8_wide_cnn7",
        dataset="cifar10",
        network="wide_cnn7",
        width=128,
        input_shape=(3, 32, 32),
        n_classes=10,
        eps=8 / 255,
        checkpoint="results/diffusion_scaling/dataset=cifar/network=wide_cnn7/eps=8_over_255/activation=relu/optimizer=adam/spectral_norm=False/scheduler=multi_step/data=x50/compute=x10/ratio=r0p3/seed=42/mtl_ibp_inc_cifar10_2_255_42.pt",
        results_json="verification_results/cifar10_eps8_wide_cnn7.json",
    ),
    ModelSpec(
        key="tinyimagenet_eps1_cnn7",
        dataset="tinyimagenet",
        network="cnn7_tinyimagenet",
        width=64,
        input_shape=(3, 64, 64),
        n_classes=200,
        eps=1 / 255,
        checkpoint="results/diffusion_scaling/dataset=tiny_imagenet/network=cnn7_tinyimagenet/eps=1_over_255/activation=relu/optimizer=adam/spectral_norm=False/scheduler=multi_step/data=x10/compute=x5/ratio=r0p6/seed=42/mtl_ibp_inc_cifar10_2_255_42.pt",
        results_json="verification_results/tinyimagenet_eps1_cnn7.json",
    ),
    ModelSpec(
        key="tinyimagenet_eps1_wide_cnn7",
        dataset="tinyimagenet",
        network="wide_cnn7_tinyimagenet",
        width=128,
        input_shape=(3, 64, 64),
        n_classes=200,
        eps=1 / 255,
        checkpoint="results/diffusion_scaling/dataset=tiny_imagenet/network=wide_cnn7_tinyimagenet/eps=1_over_255/activation=relu/optimizer=adam/spectral_norm=False/scheduler=multi_step/data=x10/compute=x5/ratio=r0p6/seed=42/mtl_ibp_inc_cifar10_2_255_42.pt",
        results_json="verification_results/tinyimagenet_eps1_wide_cnn7.json",
    ),
]


class CNN7(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], width: int, n_classes: int) -> None:
        super().__init__()
        in_channels, in_dim, _ = input_shape
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((in_dim // 2) * (in_dim // 2) * 2 * width, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CNN7TinyImageNet(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], width: int, n_classes: int) -> None:
        super().__init__()
        in_channels, in_dim, _ = input_shape
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.Conv2d(width, 2 * width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Conv2d(2 * width, 2 * width, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(2 * width),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((in_dim // 4) * (in_dim // 4) * 2 * width, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Cifar10TestSet:
    def __init__(self, data_root: Path) -> None:
        path = data_root / "cifar-10-batches-py" / "test_batch"
        if not path.exists():
            raise FileNotFoundError(f"CIFAR-10 test batch not found: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        self.data = payload["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.labels = payload.get("labels") or payload.get("fine_labels")
        if self.labels is None:
            raise KeyError(f"No labels found in CIFAR-10 test batch: {path}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.data[index])
        x = (x - CIFAR_MEAN) / CIFAR_STD
        return x, int(self.labels[index])


class TinyImageNetValSet:
    def __init__(self, data_root: Path) -> None:
        root = data_root / "tiny-imagenet-200" / "val" / "images"
        if not root.exists():
            raise FileNotFoundError(f"TinyImageNet validation images not found: {root}")
        class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
        self.class_to_idx = {path.name: idx for idx, path in enumerate(class_dirs)}
        self.samples: list[tuple[Path, int]] = []
        for class_dir in class_dirs:
            label = self.class_to_idx[class_dir.name]
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    self.samples.append((image_path, label))
        if not self.samples:
            raise FileNotFoundError(f"No TinyImageNet validation images found under: {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        x = (x - TINY_MEAN) / TINY_STD
        return x, label


def build_model(spec: ModelSpec) -> nn.Module:
    if spec.dataset == "tinyimagenet":
        return CNN7TinyImageNet(spec.input_shape, spec.width, spec.n_classes)
    return CNN7(spec.input_shape, spec.width, spec.n_classes)


def load_bounded_module_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    raw_state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(raw_state, OrderedDict):
        if isinstance(raw_state, dict):
            raw_state = raw_state.get("state_dict") or raw_state.get("model_state_dict") or raw_state
        if not isinstance(raw_state, OrderedDict):
            raw_state = OrderedDict(raw_state)

    model_state = model.state_dict()
    # CTRAIN stores ModelWrapper.state_dict(), which delegates to auto_LiRPA's
    # BoundedModule.state_dict(). Its keys are graph-node names such as
    # "/1.param" and "/5.buffer", and it omits BatchNorm num_batches_tracked
    # counters. The saved tensor order matches the underlying PyTorch module
    # parameters and persistent buffers.
    source_items = [(key, value) for key, value in raw_state.items() if torch.is_tensor(value)]
    target_items = [(key, value) for key, value in model_state.items() if not key.endswith("num_batches_tracked")]
    if len(source_items) != len(target_items):
        raise ValueError(
            f"Checkpoint tensor count mismatch for {checkpoint_path}: "
            f"{len(source_items)} source tensors, {len(target_items)} model tensors"
        )

    converted = OrderedDict(model_state)
    for (source_key, source_value), (target_key, target_value) in zip(source_items, target_items):
        if tuple(source_value.shape) != tuple(target_value.shape):
            raise ValueError(
                f"Shape mismatch for {checkpoint_path}: {source_key} {tuple(source_value.shape)} "
                f"does not match {target_key} {tuple(target_value.shape)}"
            )
        converted[target_key] = source_value.to(dtype=target_value.dtype)
    model.load_state_dict(converted, strict=True)


def remove_training_mode_attr(onnx_path: Path) -> None:
    model = onnx.load(onnx_path)
    changed = False
    for node in model.graph.node:
        if node.op_type in {"BatchNormalization", "Dropout"}:
            cleaned_attrs = [attr for attr in node.attribute if attr.name != "training_mode"]
            if len(cleaned_attrs) != len(node.attribute):
                del node.attribute[:]
                node.attribute.extend(cleaned_attrs)
                changed = True
    if changed:
        onnx.save(model, onnx_path)


def expected_onnx_paths() -> list[Path]:
    return [SCRIPT_DIR / "onnx" / filename for filename in EXPECTED_ONNX_FILES]


def sciebo_token(share_url: str) -> str:
    parsed = urlparse(share_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2 or parts[-2] != "s":
        raise ValueError(f"Expected a Sciebo share URL ending in /s/<token>, got: {share_url}")
    return parts[-1]


def sciebo_webdav_root(share_url: str) -> str:
    parsed = urlparse(share_url)
    return f"{parsed.scheme}://{parsed.netloc}/public.php/webdav/"


def download_sciebo_share(share_url: str, password: str, output_path: Path) -> None:
    token = sciebo_token(share_url)
    root_url = sciebo_webdav_root(share_url)
    auth = (token, password)
    response = requests.request("PROPFIND", root_url, auth=auth, headers={"Depth": "1"}, timeout=60)
    if response.status_code == 401:
        raise PermissionError("Sciebo rejected the share password while listing ONNX files.")
    response.raise_for_status()

    candidates: list[str] = []
    xml_root = ElementTree.fromstring(response.content)
    for href in xml_root.findall(".//{DAV:}href"):
        text = href.text or ""
        filename = Path(unquote(text).rstrip("/")).name
        if filename.endswith(".zip"):
            candidates.append(text)
    if candidates:
        href = candidates[0]
        download_url = href if href.startswith("http") else f"{urlparse(share_url).scheme}://{urlparse(share_url).netloc}{href}"
        label = Path(unquote(href)).name
    else:
        # Password-protected Sciebo shares may point directly to a single file.
        # In that case PROPFIND returns only the root resource, and GET on the
        # WebDAV root streams the shared file itself.
        download_url = root_url
        label = "shared root file"
    print(f"Downloading ONNX models from Sciebo {label}")
    with requests.get(download_url, auth=auth, stream=True, timeout=60) as stream:
        if stream.status_code == 401:
            raise PermissionError("Sciebo rejected the share password while downloading ONNX models.")
        stream.raise_for_status()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            for chunk in stream.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_onnx_archive(url: str, output_path: Path) -> None:
    if "sciebo.de/s/" in url:
        password = os.environ.get("VNNCOMP_SCIEBO_PASSWORD", SCIEBO_ONNX_SHARE_PASSWORD)
        download_sciebo_share(url, password, output_path)
        return
    print(f"Downloading ONNX models from {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)


def ensure_onnx_models(onnx_zip_url: str | None) -> None:
    if all(path.exists() for path in expected_onnx_paths()):
        return

    archive_path = DEFAULT_ONNX_ZIP
    if not archive_path.exists():
        url = onnx_zip_url or os.environ.get("VNNCOMP_ONNX_ZIP_URL") or SCIEBO_ONNX_SHARE_URL
        download_onnx_archive(url, archive_path)

    extract_onnx_archive(archive_path)
    missing = [path.as_posix() for path in expected_onnx_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"ONNX archive did not provide all expected models: {missing}")


def extract_onnx_archive(archive_path: Path) -> None:
    output_dir = SCRIPT_DIR / "onnx"
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.namelist() if member.endswith(".onnx")]
        by_name = {Path(member).name: member for member in members}
        for filename in EXPECTED_ONNX_FILES:
            member = by_name.get(filename)
            if member is None:
                continue
            target = output_dir / filename
            with archive.open(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
            remove_training_mode_attr(target)


def validate_onnx_models() -> None:
    ensure_onnx_models(None)
    for path in expected_onnx_paths():
        model = onnx.load(path)
        onnx.checker.check_model(model)
        for node in model.graph.node:
            if node.op_type in {"BatchNormalization", "Dropout"}:
                if any(attr.name == "training_mode" for attr in node.attribute):
                    raise AssertionError(f"{path} still has a {node.op_type} training_mode attribute")


def export_onnx(spec: ModelSpec, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(spec)
    load_bounded_module_checkpoint(model, REPO_ROOT / spec.checkpoint)
    model.eval()
    dummy = torch.randn(1, *spec.input_shape, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
            verbose=False,
        )
    remove_training_mode_attr(output_path)


def classify_bin(item: dict[str, Any]) -> str | None:
    result = item.get("result")
    if result not in {"sat", "unsat", "timeout"}:
        return None
    if result == "timeout":
        return "timeout"
    running_time = item.get("running_time")
    if running_time is None:
        return None
    running_time = float(running_time)
    if 0 <= running_time < 10:
        return "0_10"
    if 10 <= running_time < 100:
        return "10_100"
    if 100 <= running_time < 1000:
        return "100_1000"
    return None


def load_binned_results(spec: ModelSpec) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    path = SCRIPT_DIR / spec.results_json
    if not path.exists():
        raise FileNotFoundError(f"abCROWN results not found for {spec.key}: {path}")
    payload = json.loads(path.read_text())
    bins: dict[str, list[tuple[int, dict[str, Any]]]] = {name: [] for name in BIN_COUNTS}
    for key, item in payload.items():
        if not isinstance(item, dict):
            continue
        bin_name = classify_bin(item)
        if bin_name is None:
            continue
        bins[bin_name].append((int(key), item))
    return bins


def sample_results(spec: ModelSpec, rng: np.random.Generator) -> list[tuple[int, str, dict[str, Any], bool]]:
    bins = load_binned_results(spec)
    selected: list[tuple[int, str, dict[str, Any], bool]] = []
    for bin_name, count in BIN_COUNTS.items():
        entries = bins[bin_name]
        if not entries:
            raise ValueError(f"No entries available for bin {bin_name} in {spec.results_json}")
        replace = len(entries) < count
        choices = rng.choice(len(entries), size=count, replace=replace)
        for choice in choices:
            index, item = entries[int(choice)]
            selected.append((index, bin_name, item, replace))
    return selected


def ensure_cifar10(data_root: Path) -> None:
    if (data_root / "cifar-10-batches-py" / "test_batch").exists():
        return
    from torchvision import datasets

    datasets.CIFAR10(root=str(data_root), train=False, download=True)


def create_tinyimagenet_val_folders(dataset_dir: Path) -> None:
    val_dir = dataset_dir / "val"
    image_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"
    if not annotations.exists():
        return
    for line in annotations.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        image_name, class_name = parts[0], parts[1]
        source = image_dir / image_name
        target_dir = image_dir / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / image_name
        if source.exists() and not target.exists():
            source.rename(target)


def ensure_tinyimagenet(data_root: Path) -> None:
    dataset_dir = data_root / "tiny-imagenet-200"
    image_dir = dataset_dir / "val" / "images"
    if image_dir.exists() and any(path.is_dir() for path in image_dir.iterdir()):
        return

    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / "tiny-imagenet-200.zip"
    if not archive_path.exists():
        print(f"Downloading TinyImageNet from {TINYIMAGENET_URL}")
        urllib.request.urlretrieve(TINYIMAGENET_URL, archive_path)
    print("Extracting TinyImageNet")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(data_root)
    create_tinyimagenet_val_folders(dataset_dir)


def ensure_datasets(data_root: Path, use_ctrain: bool) -> None:
    data_root.mkdir(parents=True, exist_ok=True)
    if use_ctrain:
        try:
            from CTRAIN.data_loaders.data_loaders import load_cifar10, load_tinyimagenet

            load_cifar10(batch_size=1, normalise=True, val_split=False, data_root=str(data_root))
            load_tinyimagenet(batch_size=1, normalise=True, val_split=False, data_root=str(data_root))
            return
        except Exception as exc:
            print(f"CTRAIN dataset download unavailable, falling back to local download helpers: {exc}")
    ensure_cifar10(data_root)
    ensure_tinyimagenet(data_root)


def dataset_for(spec: ModelSpec, cache: dict[str, Any], data_root: Path) -> Any:
    if spec.dataset not in cache:
        cache[spec.dataset] = TinyImageNetValSet(data_root) if spec.dataset == "tinyimagenet" else Cifar10TestSet(data_root)
    return cache[spec.dataset]


def bounds_for(x: torch.Tensor, spec: ModelSpec) -> torch.Tensor:
    std = TINY_STD if spec.dataset == "tinyimagenet" else CIFAR_STD
    mean = TINY_MEAN if spec.dataset == "tinyimagenet" else CIFAR_MEAN
    eps = torch.full_like(std, float(spec.eps)) / std
    data_min = (torch.zeros_like(mean) - mean) / std
    data_max = (torch.ones_like(mean) - mean) / std
    lower = (x - eps).clamp(data_min, data_max)
    upper = (x + eps).clamp(data_min, data_max)
    return torch.stack([lower, upper], dim=-1)


def format_float(value: float) -> str:
    return format(float(value), ".10g")


def write_vnnlib(path: Path, spec: ModelSpec, image_index: int, label: int, bounds: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_bounds = bounds.reshape(-1, 2)
    with path.open("w") as handle:
        handle.write(f"; VNN-COMP challenging certified training benchmark\n")
        handle.write(f"; model: {spec.key}\n")
        handle.write(f"; dataset index: {image_index}, label: {label}, L_inf epsilon: {spec.eps}\n\n")

        for i in range(flat_bounds.shape[0]):
            handle.write(f"(declare-const X_{i} Real)\n")
        handle.write("\n")
        for i in range(spec.n_classes):
            handle.write(f"(declare-const Y_{i} Real)\n")
        handle.write("\n; Input constraints\n")
        for i, (lower, upper) in enumerate(flat_bounds.tolist()):
            handle.write(f"(assert (>= X_{i} {format_float(lower)}))\n")
            handle.write(f"(assert (<= X_{i} {format_float(upper)}))\n")
        handle.write("\n; Output constraints: counterexample to robustness\n")
        handle.write("(assert (or\n")
        for i in range(spec.n_classes):
            if i != label:
                handle.write(f"  (and (<= Y_{label} Y_{i}))\n")
        handle.write("))\n")


def prepare_output_dirs() -> None:
    for dirname in ("onnx", "vnnlib", "metadata"):
        path = SCRIPT_DIR / dirname
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    instances = SCRIPT_DIR / "instances.csv"
    if instances.exists():
        instances.unlink()


def prepare_instance_dirs() -> None:
    for dirname in ("vnnlib", "metadata"):
        path = SCRIPT_DIR / dirname
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    instances = SCRIPT_DIR / "instances.csv"
    if instances.exists():
        instances.unlink()


def validate_instances(sampled: list[SampledInstance | dict[str, Any]]) -> None:
    if len(sampled) != 60:
        raise AssertionError(f"Expected 60 sampled instances, got {len(sampled)}")
    total_timeout = sum(get_item(item, "timeout") for item in sampled)
    if total_timeout != EXPECTED_TOTAL_TIMEOUT:
        raise AssertionError(f"Expected timeout sum {EXPECTED_TOTAL_TIMEOUT}, got {total_timeout}")
    for spec in MODEL_SPECS:
        rows = [item for item in sampled if get_item(item, "model_key") == spec.key]
        if len(rows) != 10:
            raise AssertionError(f"{spec.key} expected 10 instances, got {len(rows)}")
        for bin_name, count in BIN_COUNTS.items():
            found = sum(1 for item in rows if get_item(item, "bin") == bin_name)
            if found != count:
                raise AssertionError(f"{spec.key} bin {bin_name} expected {count}, got {found}")
    for item in sampled:
        if not get_item(item, "abcrown_result") or get_item(item, "abcrown_running_time") is None:
            raise AssertionError(f"Missing abCROWN result metadata: {item}")
        if not (SCRIPT_DIR / get_item(item, "onnx_path")).exists():
            raise AssertionError(f"Missing ONNX path: {get_item(item, 'onnx_path')}")
        if not (SCRIPT_DIR / get_item(item, "vnnlib_path")).exists():
            raise AssertionError(f"Missing VNN-LIB path: {get_item(item, 'vnnlib_path')}")


def get_item(item: SampledInstance | dict[str, Any], key: str) -> Any:
    if isinstance(item, dict):
        return item[key]
    return getattr(item, key)


def write_instances_csv(rows: list[SampledInstance | dict[str, Any]]) -> None:
    with (SCRIPT_DIR / "instances.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        for item in rows:
            writer.writerow(
                [
                    get_item(item, "onnx_path"),
                    get_item(item, "vnnlib_path"),
                    get_item(item, "timeout"),
                ]
            )


def generate_instances(seed: int, data_root: Path) -> list[SampledInstance]:
    rng = np.random.default_rng(seed)
    sampled_metadata: list[SampledInstance] = []
    dataset_cache: dict[str, Any] = {}

    for spec in MODEL_SPECS:
        onnx_rel = Path("onnx") / f"{spec.key}.onnx"
        dataset = dataset_for(spec, dataset_cache, data_root)
        selected = sample_results(spec, rng)

        for ordinal, (image_index, bin_name, item, replacement) in enumerate(selected):
            x, label = dataset[image_index]
            if int(label) < 0 or int(label) >= spec.n_classes:
                raise ValueError(f"Label {label} out of range for {spec.key} instance {image_index}")
            vnnlib_rel = Path("vnnlib") / spec.key / f"{spec.key}_idx{image_index}_sample{ordinal}.vnnlib"
            write_vnnlib(SCRIPT_DIR / vnnlib_rel, spec, image_index, int(label), bounds_for(x, spec))
            timeout = BIN_TIMEOUTS[bin_name]
            sampled_metadata.append(
                SampledInstance(
                    model_key=spec.key,
                    dataset=spec.dataset,
                    network=spec.network,
                    image_index=image_index,
                    bin=bin_name,
                    abcrown_result=str(item["result"]),
                    abcrown_running_time=float(item["running_time"]),
                    timeout=timeout,
                    onnx_path=onnx_rel.as_posix(),
                    vnnlib_path=vnnlib_rel.as_posix(),
                    source_results_json=spec.results_json,
                    source_checkpoint=spec.checkpoint,
                    replacement_sampling=replacement,
                )
            )

    return sampled_metadata


def write_metadata(seed: int, sampled_metadata: list[SampledInstance], mode: str) -> None:
    validate_instances(sampled_metadata)
    write_instances_csv(sampled_metadata)
    metadata = {
        "seed": seed,
        "mode": mode,
        "total_instances": len(sampled_metadata),
        "total_timeout": sum(item.timeout for item in sampled_metadata),
        "bin_counts_per_model": dict(BIN_COUNTS),
        "bin_timeouts": BIN_TIMEOUTS,
        "models": [asdict(spec) for spec in MODEL_SPECS],
        "instances": [asdict(item) for item in sampled_metadata],
    }
    (SCRIPT_DIR / "metadata" / "sampled_instances.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Generated {len(sampled_metadata)} instances in {SCRIPT_DIR}")
    print(f"Total CSV timeout: {metadata['total_timeout']} seconds")


def run_benchmark_generation(seed: int, onnx_zip_url: str | None, data_root: Path, use_ctrain: bool) -> None:
    ensure_onnx_models(onnx_zip_url)
    validate_onnx_models()
    ensure_datasets(data_root, use_ctrain)
    prepare_instance_dirs()
    sampled_metadata = generate_instances(seed, data_root)
    write_metadata(seed, sampled_metadata, "portable")


def rebuild_from_sources(seed: int, data_root: Path, use_ctrain: bool) -> None:
    prepare_output_dirs()
    ensure_datasets(data_root, use_ctrain)
    for spec in MODEL_SPECS:
        onnx_rel = Path("onnx") / f"{spec.key}.onnx"
        export_onnx(spec, SCRIPT_DIR / onnx_rel)
    sampled_metadata = generate_instances(seed, data_root)
    write_metadata(seed, sampled_metadata, "source_rebuild")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the VNN-COMP challenging certified training benchmark.")
    parser.add_argument("seed", type=int, help="Random seed used to randomize benchmark generation.")
    parser.add_argument(
        "--onnx-zip-url",
        default=None,
        help=(
            "URL used to download onnx_models.zip when bundled ONNX files are absent. "
            "Can also be provided with VNNCOMP_ONNX_ZIP_URL."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory used for downloaded CIFAR-10 and TinyImageNet test data.",
    )
    parser.add_argument(
        "--no-ctrain-download",
        action="store_true",
        help="Use local torchvision/urllib dataset download helpers instead of trying CTRAIN loaders first.",
    )
    parser.add_argument(
        "--rebuild-from-sources",
        action="store_true",
        help=(
            "Recreate ONNX/VNN-LIB artifacts from the original training workspace. "
            "The default portable mode downloads/extracts ONNX models and regenerates VNN-LIB specs from bundled results."
        ),
    )
    return parser.parse_args(argv[1:])


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    use_ctrain = not args.no_ctrain_download
    if args.rebuild_from_sources:
        rebuild_from_sources(args.seed, data_root, use_ctrain)
    else:
        run_benchmark_generation(args.seed, args.onnx_zip_url, data_root, use_ctrain)


if __name__ == "__main__":
    main(sys.argv)
