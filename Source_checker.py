import os, sys
from spectral_cube import SpectralCube

# Define the root directory for sources
sources_root = '/almagal1/ALMA/ALMAGAL/ALMAGAL/data/2019.1.00195.L/sources'

# Ensure the sources root exists and contains directories
if not os.path.isdir(sources_root):
    print(f"Sources root not found: {sources_root}")
    sys.exit(1)

source_dirs = [d for d in sorted(os.listdir(sources_root), reverse=True)
               if os.path.isdir(os.path.join(sources_root, d))]

if not source_dirs:
    print(f"No source directories found in {sources_root}")
    sys.exit(0)
source_ids = source_dirs 
fits_list = []
empty_files = []
count_normal = 0
count_fallback = 0
count_missing = 0
# Lists for failure reasons
sources_without_specific_file = []
sources_without_any_fits = []
missing_sources = []
for source_id in source_ids:
    almagal_dir = os.path.join(sources_root, source_id, 'images/combined/7MTM2TM1/almagal')
    fallback_dir = os.path.join(sources_root, source_id, 'images/combined/7MTM2TM1')

    search_dirs = []
    if os.path.isdir(almagal_dir):
        search_dirs.append(almagal_dir)
    if os.path.isdir(fallback_dir):
        search_dirs.append(fallback_dir)

    if not search_dirs:
        count_missing += 1
        sources_without_any_fits.append(source_id)
        missing_sources.append(source_id)
        print(
            f"Skipping source {source_id}: missing both FITS dirs "
            f"{almagal_dir} and {fallback_dir}"
        )
        continue

    selected_fits = None
    selected_dir = None
    for d in search_dirs:
        # Only consider non-empty files as valid candidates
        candidates = [f for f in os.listdir(d)
                      if f.endswith('spw0_7MTM2TM1_jointdeconv.image.fits') and
                      os.path.getsize(os.path.join(d, f)) > 0]
        if candidates:
            selected_dir = d
            selected_fits = os.path.join(d, sorted(candidates)[0])
            break

    if selected_fits is None:
        count_missing += 1
        # Decide whether the source has other FITS files (so it's missing the specific file)
        # Check for any non-empty .fits files; record empty ones
        has_any_fits = False
        for d in search_dirs:
            for fname in os.listdir(d):
                if not fname.lower().endswith('.fits'):
                    continue
                path = os.path.join(d, fname)
                try:
                    if os.path.getsize(path) > 0:
                        has_any_fits = True
                        break
                    else:
                        empty_files.append(path)
                except OSError:
                    # Can't stat the file; treat as missing/skip
                    continue
            if has_any_fits:
                break

        if has_any_fits:
            sources_without_specific_file.append(source_id)
        else:
            sources_without_any_fits.append(source_id)

        missing_sources.append(source_id)
        print(
            f"Skipping source {source_id}: no spw0 jointdeconv cube in any of {search_dirs}"
        )
        continue

    if selected_dir == almagal_dir:
        count_normal += 1
    else:
        count_fallback += 1
        print(f"Using fallback FITS dir for source {source_id}: {selected_dir}")

    fits_list.append((source_id, selected_fits))

print(f"Normal directory: {count_normal} sources")
print(f"Fallback directory: {count_fallback} sources")
print(f"Missing file: {count_missing} sources")

print("\nSources missing the specific spw0_7MTM2TM1_jointdeconv.image.fits file:")
if sources_without_specific_file:
    print("  " + "\n  ".join(sources_without_specific_file))
else:
    print("  (none)")

print("\nSources without any FITS files (no almagal or fallback FITS):")
if sources_without_any_fits:
    print("  " + "\n  ".join(sources_without_any_fits))
else:
    print("  (none)")

if missing_sources:
    print("\nTrying failed sources until 3 with FITS files are found in fallback directories...")
    found_with_fits = 0

    for test_source in missing_sources:
        if found_with_fits >= 3:
            break

        fallback_dir = os.path.join(sources_root, test_source, 'images/combined/7MTM2TM1')

        if not os.path.isdir(fallback_dir):
            print(f"Skipping {test_source}: fallback directory not found: {fallback_dir}")
            continue

        # Only include non-empty FITS files for inspection
        all_fits = []
        for fname in os.listdir(fallback_dir):
            if not fname.lower().endswith('.fits'):
                continue
            path = os.path.join(fallback_dir, fname)
            try:
                if os.path.getsize(path) > 0:
                    all_fits.append(path)
                else:
                    print(f"Found empty FITS file (skipping): {path}")
                    empty_files.append(path)
            except OSError:
                print(f"Could not stat file (skipping): {path}")

        if not all_fits:
            print(f"Skipping {test_source}: no FITS files found in fallback directory: {fallback_dir}")
            continue

        found_with_fits += 1
        print(f"\n[{found_with_fits}/3] Inspecting source {test_source}")
        for fits_path in sorted(all_fits):
            try:
                cube = SpectralCube.read(fits_path)
                freq_axis = cube.spectral_axis
                fmin = freq_axis.min()
                fmax = freq_axis.max()
                print(f"OK: {fits_path}")
                print(f"    Frequency range: {fmin} to {fmax}")
            except Exception as e:
                print(f"FAILED: {fits_path}")
                print(f"    Could not open with SpectralCube: {e}")

    if found_with_fits < 3:
        print(f"\nOnly found {found_with_fits} failed sources with FITS files in fallback directories.")
else:
    print("\nNo missing sources found, so no per-source FITS inspection was run.")

if empty_files:
    print("\nEmpty FITS files encountered (skipped):")
    for p in empty_files:
        print(f"  {p}")