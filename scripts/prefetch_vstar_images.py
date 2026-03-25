#!/usr/bin/env python3

import os
import socket
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError, URLError

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


def _download_with_retries(asset_url: str, output_path: Path) -> None:
    timeout = float(os.getenv("VSTAR_PREFETCH_TIMEOUT_SECONDS", "60"))
    retries = int(os.getenv("VSTAR_PREFETCH_RETRIES", "5"))
    backoff_seconds = float(os.getenv("VSTAR_PREFETCH_BACKOFF_SECONDS", "2"))

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(asset_url, timeout=timeout) as response:
                output_path.write_bytes(response.read())
            return
        except (HTTPError, URLError, TimeoutError, socket.timeout, OSError) as exc:
            last_error = exc
            if output_path.exists():
                output_path.unlink()
            if attempt == retries:
                break
            sleep_seconds = backoff_seconds * attempt
            print(
                (
                    f"retrying {asset_url} attempt={attempt}/{retries} "
                    f"error={type(exc).__name__}: {exc} sleep={sleep_seconds}s"
                ),
                flush=True,
            )
            time.sleep(sleep_seconds)

    raise RuntimeError(
        f"failed to download {asset_url} after {retries} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def main() -> None:
    hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    asset_root = Path(os.getenv("HAL_BENCHMARK_ASSET_CACHE", ASSET_CACHE_DIR))
    downloaded = 0
    skipped = 0
    failed = []

    for asset_path in _iter_image_paths():
        output_path = asset_root / asset_path
        if output_path.exists():
            skipped += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        asset_url = (
            f"{hf_endpoint}/datasets/craigwu/vstar_bench/resolve/main/{asset_path}"
        )
        try:
            _download_with_retries(asset_url, output_path)
            downloaded += 1
            print(f"downloaded {asset_path} -> {output_path}", flush=True)
        except Exception as exc:  # pragma: no cover - operational behavior
            failed.append((asset_path, str(exc)))
            print(f"failed {asset_path}: {exc}", flush=True)

    print(
        (
            f"done: downloaded={downloaded} skipped={skipped} failed={len(failed)} "
            f"asset_root={asset_root}"
        ),
        flush=True,
    )
    if failed:
        print("failed_assets:", flush=True)
        for asset_path, error in failed:
            print(f"- {asset_path}: {error}", flush=True)


if __name__ == "__main__":
    main()
