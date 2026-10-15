from astropy.io import fits
from astropy.convolution import convolve_fft, Gaussian2DKernel
try:
    from reproject import reproject_interp, reproject_exact
except Exception:
    reproject_interp = None
    reproject_exact = None
import numpy as np
from astropy.wcs import WCS
import os
import re
from scipy import ndimage

def reproject_to_target(data_arr, src_hdr, tgt_hdr, shape_out):
    if reproject_exact is None:
        raise RuntimeError(
            "The 'reproject' package is not installed. Install it with:"
            "\n  conda install -c conda-forge reproject\n  or\n  pip install reproject"
        )
    arr, footprint = reproject_exact((data_arr, src_hdr), tgt_hdr, shape_out=shape_out)
    return arr, footprint

def extract_distance(source_id, id_info_table):
    source_id = str(source_id).strip()
    with open(id_info_table, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:     #skip any lines with wrong number of columns
                continue
            if parts[3] == source_id:
                try:
                    return float(parts[8]) #heliocentric distance pc
                except ValueError:
                    return None
    return None
 
 

# Process multiple file pairs: find matching IDs in the two directories and
# process the first 5 pairs. The original single-file logic is refactored
# into `process_pair()` below.

tm2tm1_dir = '/almagal1/ALMA/ALMAGAL/ALMAGAL/data/2019.1.00195.L/sources'
tm2_dir = '/raid/scratch/hannahb/Data7mtm2'
savedir = '/raid/scratch/mevans'

def process_pair(file_tm2tm1, file_tm2, savedir_base):
    """Process one pair of TM2TM1 and TM2 FITS files (smoothing + noise removal).

    This is a mostly verbatim refactor of the previous inline script logic,
    made local so we can call it for multiple pairs.
    """
    try:
        hdul = fits.open(file_tm2tm1)
        hdu = hdul[0]
        hdr = hdu.header
        wcs = WCS(hdr)
        data = hdu.data
        source = hdr.get("OBJECT", "").strip() or os.path.basename(file_tm2tm1).split('_')[0]
        sigma0 = hdr.get("AGSTDRES")

        dist = extract_distance(source, "/raid/scratch/hannahb/imageproperties.dat")
        cdelt = hdr.get("CDELT1")
        deg_per_pixel_tm2tm1 = abs(cdelt)

        bmaj_tm2tm1 = hdr["BMAJ"]
        bmin_tm2tm1 = hdr["BMIN"]
        bpa = hdr["BPA"]

        hdul2 = fits.open(file_tm2)
        hdr2 = hdul2[0].header
        data_tm2 = hdul2[0].data
        deg_per_pixel_tm2 = abs(hdr2.get("CDELT1"))
        bmaj_tm2 = hdr2["BMAJ"]
        bmin_tm2 = hdr2["BMIN"]
        bpa_tm2 = hdr2["BPA"]

        # beam difference (degrees) - guard against small negative values from rounding
        bmaj_tm2_sq = max(bmaj_tm2**2 - bmaj_tm2tm1**2, 0.0)
        bmin_tm2_sq = max(bmin_tm2**2 - bmin_tm2tm1**2, 0.0)

        smooth_bmaj_deg = np.sqrt(bmaj_tm2_sq)
        smooth_bmin_deg = np.sqrt(bmin_tm2_sq)

        # convert to pixels using the pixel scale of the image being convolved
        smooth_bmaj_pix = (smooth_bmaj_deg / deg_per_pixel_tm2tm1) / 2.355
        smooth_bmin_pix = (smooth_bmin_deg / deg_per_pixel_tm2tm1) / 2.355

        # Kernel FWHM in arcsec and pixels
        fwhm_major_arcsec = smooth_bmaj_deg * 3600.0
        fwhm_minor_arcsec = smooth_bmin_deg * 3600.0
        fwhm_major_pix = smooth_bmaj_pix * 2.355
        fwhm_minor_pix = smooth_bmin_pix * 2.355
        print(f"Kernel FWHM: major={fwhm_major_arcsec:.3f} arcsec ({fwhm_major_pix:.3f} pix), minor={fwhm_minor_arcsec:.3f} arcsec ({fwhm_minor_pix:.3f} pix)")

        os.makedirs(savedir_base, exist_ok=True)

        # Use the PA difference for the smoothing kernel (convert degrees -> radians)
        theta_smooth = np.deg2rad(bpa_tm2 - bpa)
        kernel = Gaussian2DKernel(x_stddev=smooth_bmaj_pix, y_stddev=smooth_bmin_pix, theta=theta_smooth)

        # Reprojection
        footprint = None
        data_tm2tm1_og = data

        mask = np.isfinite(data).astype(float)
        data_filled = np.where(np.isfinite(data), data, 0.0)
        target_shape = (hdr2.get('NAXIS2'), hdr2.get('NAXIS1'))

        # Normalised convolution: convolve(data*mask) / convolve(mask) on source grid
        num = convolve_fft(data_filled * mask, kernel, normalize_kernel=True)
        den = convolve_fft(mask, kernel, normalize_kernel=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            smooth_data = np.where(den > 0, num / den, np.nan)

        # Mask the smoothed data to avoid edge artifacts
        smooth_data = np.where(mask > 0, smooth_data, np.nan)

        # Reproject smoothed data to target
        smooth_data, footprint = reproject_to_target(smooth_data, hdr, hdr2, shape_out=target_shape)
        print(f"TM2 data after reproject: shape={smooth_data.shape}, range=[{np.nanmin(smooth_data):.6f}, {np.nanmax(smooth_data):.6f}]")

        # If reproject provided a footprint, mask invalid pixels
        if 'footprint' in locals() and footprint is not None:
            smooth_data = np.where(footprint > 0, smooth_data, np.nan)

        def normalize(data):
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            return (data - data_min) / (data_max - data_min) if data_max > data_min else data

        # Noise propagation: for beam-convolved radio images (Jy/beam), pixels
        # within one beam are correlated, so the per-pixel formula (sum of
        # squared kernel weights) over-counts independent samples and gives an
        # unrealistically small smoothed noise.  The correct scaling uses the
        # beam solid-angle ratio: smoothing to a larger beam averages over
        # Omega_new/Omega_orig independent beams, reducing noise by
        # sqrt(Omega_orig / Omega_new).
        sigma_smooth = sigma0 * np.sqrt(
            (bmaj_tm2tm1 * bmin_tm2tm1) / (bmaj_tm2 * bmin_tm2)
        )

        # print the noise values and the maximum and minimum flux values for the original and smoothed data
        print(f"Original noise (sigma0): {sigma0:.6f}")
        print(f"Smoothed noise (sigma_smooth): {sigma_smooth:.6f}")
        print(f"Original data range: [{np.nanmin(data):.6f}, {np.nanmax(data):.6f}]")
        print(f"Smoothed data range: [{np.nanmin(smooth_data):.6f}, {np.nanmax(smooth_data):.6f}]") 

        # Noise removal at 1-sigma and 2-sigma thresholds, saving FITS to separate directories
        for n_sigma in [0]:
            noise_dir = os.path.join(savedir_base, f"smoothed_7mtm1tm2_noiseremoved{n_sigma}")
            os.makedirs(noise_dir, exist_ok=True)

            finite_smooth = np.isfinite(smooth_data)
            smooth_data_clip = np.full_like(smooth_data, np.nan)
            smooth_data_clip[finite_smooth] = np.where(
                smooth_data[finite_smooth] >= n_sigma * sigma_smooth,
                smooth_data[finite_smooth],
                0.0
            )

            # Save noise-removed FITS file
            # Update header metadata to record the smoothing and noise propagation
            new_hdr = hdr2.copy()
            try:
                new_hdr['HISTORY'] = 'Smoothed to target TM2 beam and reprojected'
            except Exception:
                pass
            new_hdr['SMOOTHBM'] = (fwhm_major_arcsec, 'smoothing FWHM major (arcsec)')
            new_hdr['SMOOTHBN'] = (fwhm_minor_arcsec, 'smoothing FWHM minor (arcsec)')
            new_hdr['SMOOTHPX'] = (f"{fwhm_major_pix:.4f},{fwhm_minor_pix:.4f}", 'FWHM pixels (major,minor)')
            new_hdr['SMOOTHPA'] = (theta_smooth, 'smoothing rotation (radians)')
            # original reported noise (AGSTDRES) and propagated RMS after smoothing
            try:
                new_hdr['AGSTDRES_ORIG'] = (float(sigma0) if sigma0 is not None else np.nan, 'Original AGSTDRES')
            except Exception:
                new_hdr['AGSTDRES_ORIG'] = (np.nan, 'Original AGSTDRES')
            new_hdr['AGSTDRES_SMO'] = (float(sigma_smooth), 'Estimated RMS after smoothing')

            # Count of valid pixels after reprojection (if footprint present)
            valid_pixels = None
            if 'footprint' in locals() and footprint is not None:
                try:
                    valid_pixels = int(np.count_nonzero(footprint > 0))
                except Exception:
                    valid_pixels = None
            if valid_pixels is not None:
                new_hdr['NVALIDFP'] = (valid_pixels, 'Valid pixels after reprojection')

            # Write out FITS file (overwrite existing) with updated header
            out_hdu = fits.PrimaryHDU(data=smooth_data_clip, header=new_hdr)
            out_path = os.path.join(noise_dir, f"{source}_smoothed_noiseremoved{n_sigma}sigma.fits")
            out_hdu.writeto(out_path, overwrite=True)
            print(f"Saved {n_sigma}-sigma noise-removed FITS to {out_path}")

    finally:
        try:
            hdul.close()
        except Exception:
            pass


# Build one continuum pair per source.  Each source directory also contains
# four spectral-window products (spw0-spw3), which are not separate sources.
tm2tm1_files = []
for sourcename in os.listdir(tm2tm1_dir):
    continuum_dir = os.path.join(
        tm2tm1_dir, sourcename, 'images', 'combined', '7MTM2TM1', 'almagal'
    )
    if os.path.isdir(continuum_dir):
        tm2tm1_files.extend(
            os.path.join(continuum_dir, fname)
            for fname in os.listdir(continuum_dir)
            if fname.endswith('7MTM2TM1_jointdeconv.image.fits')
            and '_cont_' in fname
        )
tm2tm1_files.sort()
pairs = []
for path_tm2tm1 in tm2tm1_files:
    fname = os.path.basename(path_tm2tm1)
    source = fname.split('_')[0]
    # find a matching tm2 file in tm2_dir that starts with the same source id
    candidates = [f for f in os.listdir(tm2_dir) if f.startswith(source) and '7MTM2_jointdeconv.image.fits' in f]
    if candidates:
        path_tm2 = os.path.join(tm2_dir, sorted(candidates)[0])
        pairs.append((path_tm2tm1, path_tm2))
    else:
        print(f"Skipping {source}: matching TM2 file not found in {tm2_dir}")

if not pairs:
    print("No matching file pairs found in the dataset directories.")
else:
    total_pairs = len(pairs)
    processed_count = 0
    for file_a, file_b in pairs:
        remaining = total_pairs - processed_count
        print(f"\nSources left to process: {remaining}")
        print(f"\nProcessing pair:\n  {file_a}\n  {file_b}")
        try:
            process_pair(file_a, file_b, savedir)
            processed_count += 1
        except Exception as e:
            print(f"Error processing pair {file_a}, {file_b}: {e}")
            continue

    print(
        f"\nFinished. Successfully processed {processed_count} out of "
        f"{total_pairs} pairs. Sources left to process: "
        f"{total_pairs - processed_count}."
    )
