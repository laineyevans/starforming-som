import os
import shutil
from pathlib import Path

SRC_ROOT = Path("/raid/scratch/mevans/spectral_cube_test")
DST_ROOT = Path("/raid/scratch/mevans/spectral_line_data")
DST_ROOT.mkdir(parents=True, exist_ok=True)

REQUIRED_COUNT = 3  # require at least this many FITS files per source
REQUIRED_TAGS = ["H2CO", "SiO", "DCN"]  # expected line identifiers (case-insensitive)


def has_required_tags(fits_files):
    """Check that all REQUIRED_TAGS are present in the filenames (case-insensitive)."""
    names = [f.name.lower() for f in fits_files]
    return all(any(tag.lower() in name for name in names) for tag in REQUIRED_TAGS)


def collect_sources():
    copied = 0
    skipped_empty = 0
    skipped_missing = 0

    for entry in sorted(SRC_ROOT.iterdir()):
        if not entry.is_dir():
            continue

        fits_files = [p for p in entry.iterdir() if p.is_file() and p.suffix.lower() in {".fits", ".fit", ".fts"}]
        if not fits_files:
            skipped_empty += 1
            continue

        # Require the three line maps to exist
        if len(fits_files) < REQUIRED_COUNT or not has_required_tags(fits_files):
            skipped_missing += 1
            continue

        dest_dir = DST_ROOT / entry.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for f in fits_files:
            shutil.copy2(f, dest_dir / f.name)
        copied += 1

    print(f"Copied {copied} sources to {DST_ROOT}")
    print(f"Skipped {skipped_empty} empty source dirs and {skipped_missing} missing-required-line dirs")


def main():
    collect_sources()


if __name__ == "__main__":
    main()
