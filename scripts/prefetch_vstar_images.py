#!/usr/bin/env python3

import os
import urllib.request
from pathlib import Path

from datasets import load_dataset

from hal.benchmarks._benchmark_utils import ASSET_CACHE_DIR


def _iter_image_paths():
    dataset = None
    errors = []
    for split_name in ("test", "validation", "train"):
        try:
            dataset = load_dataset("craigwu/vstar_bench", split=split_name)
            break
        except Exception as exc:  # pragma: no cover - operational fallback
            errors.append(f"{split_name}: {exc}")
    if dataset is None:
        raise RuntimeError("Failed to load craigwu/vstar_bench: " + " | ".join(errors))

    seen = set()
    for row in dataset:
        asset = row.get("image") or row.get("img")
        if not isinstance(asset, str):
            continue
        normalized = asset.lstrip("/")
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


def main() -> None:
    hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    asset_root = Path(os.getenv("HAL_BENCHMARK_ASSET_CACHE", ASSET_CACHE_DIR))
    downloaded = 0
    skipped = 0

    for asset_path in _iter_image_paths():
        output_path = asset_root / asset_path
        if output_path.exists():
            skipped += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        asset_url = (
            f"{hf_endpoint}/datasets/craigwu/vstar_bench/resolve/main/{asset_path}"
        )
        urllib.request.urlretrieve(asset_url, output_path)
        downloaded += 1
        print(f"downloaded {asset_path} -> {output_path}", flush=True)

    print(
        f"done: downloaded={downloaded} skipped={skipped} asset_root={asset_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
