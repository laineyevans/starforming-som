import os
from astropy.io import fits


KEYS_TO_SHOW = [
    'OBJECT', 'DATE-OBS', 'BUNIT', 'BMAJ', 'BMIN', 'BPA',
    'CDELT1', 'NAXIS1', 'NAXIS2',
    'AGSTDRES', 'DVC_KMS', 'DV_KMS',
    'RMS', 'NOISE', 'SIGMA', 'RMS_MOM', 'RMS_CHAN', 'NCHANINT', 'SIG_ORIG', 'SIG_SMO'
]


def print_fits_headers_for_source(source_id: str, search_root: str = "/raid/scratch/mevans/spectral_cube_test") -> None:
    """Find FITS files for `source_id` under `search_root` and print primary headers."""
    src_dir = os.path.join(search_root, str(source_id))
    if not os.path.isdir(src_dir):
        print(f"Source directory not found: {src_dir}")
        return

    fits_files = [f for f in sorted(os.listdir(src_dir)) if f.lower().endswith('.fits')]
    if not fits_files:
        print(f"No FITS files found in {src_dir}")
        return

    print(f"Found {len(fits_files)} FITS files for source {source_id} in {src_dir}:\n")
    for fname in fits_files:
        path = os.path.join(src_dir, fname)
        try:
            with fits.open(path) as hd:
                hdr = hd[0].header
                print('---', fname, '---')
                for key in KEYS_TO_SHOW:
                    print(f"  {key}: {hdr.get(key, 'MISSING')}")
                print('')
        except Exception as e:
            print(f"Error reading {path}: {e}")


if __name__ == '__main__':
    # Concise single-file print (selected keys only)
    path = "/raid/scratch/mevans/moment0_with_rms2/730135/730135_DCN_moment0.fits"
    try:
        with fits.open(path) as hd:
            hdr = hd[0].header
            print(f"Headers for file: {path}\n")
            for key in KEYS_TO_SHOW:
                print(f"{key}: {hdr.get(key, 'MISSING')}")
    except Exception as e:
        print(f"Error reading {path}: {e}")