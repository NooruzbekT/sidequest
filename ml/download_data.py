"""Скачивает датасет Kaggle в data/raw и пишет manifest.json с версией и sha256 файлов."""

import hashlib
import json
import shutil
from pathlib import Path

import kagglehub

DATASET = "antonkozyriev/game-recommendations-on-steam"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    src = Path(kagglehub.dataset_download(DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict = {"dataset": DATASET, "kaggle_version": src.name, "files": {}}
    for f in sorted(src.iterdir()):
        dst = RAW_DIR / f.name
        if not dst.exists() or dst.stat().st_size != f.stat().st_size:
            shutil.copy2(f, dst)
        manifest["files"][f.name] = {"bytes": dst.stat().st_size, "sha256": sha256_of(dst)}
        print(f"{f.name}: {manifest['files'][f.name]['bytes']:,} bytes")

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
