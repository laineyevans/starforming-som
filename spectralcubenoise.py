import os
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
from multiprocessing import Pool, cpu_count

outdir = "/raid/scratch/mevans/moment0_with_rms2"
os.makedirs(outdir, exist_ok=True)

VERBOSE = False
TEST_ONE_SOURCE = False
TEST_SOURCE_ID = "730135"  # set to a specific source for one-source test
FORCE_REPROCESS_TEST_SOURCE = True

def diag(message):
    if VERBOSE:
        print(message)

def get_source_outdir(source_id):
    return os.path.join(outdir, str(source_id))

def source_has_outputs(source_id):
    candidate = get_source_outdir(source_id)
    if not os.path.isdir(candidate):
        return False
    with os.scandir(candidate) as entries:
        return any(True for _ in entries)

def find_nearest(array, value):
    diff = np.abs(array - value)
    index = diff.argmin()
    nearest = array[index]
    return nearest, index

info_table = 'imageproperties.dat'

c = 299792 #km/s

sources_root = '/almagal1/ALMA/ALMAGAL/ALMAGAL/data/2019.1.00195.L/sources'

def extract_velocity(source_id, id_info_table):
    source_id = str(source_id).strip()
    with open(id_info_table, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 9:
                continue
            if parts[3] == source_id:
                try:
                    return float(parts[7])
                except ValueError:
                    return None
    return None

lines = [
    ("H2CO", 218.222192 * 10**9),
    ("SiO", 217.104980 * 10**9),
    ("DCN", 217.238538 * 10**9),
]

LINE_SEPARATION_TOLERANCE_HZ = 2e6
STRICT_LINE_SEPARATION_VALIDATION = True
FALLBACK_INTEGRATION_WINDOW_HZ = 1e6
EMISSION_FREE_GUARD_HZ = 8e6

# Fixed fallback value used only if channel spacing cannot be measured
FIXED_REFERENCE_FREQ_HZ = 2.176e11
FIXED_DNU_HZ = 488e3

# Keep concurrency low to avoid temporary-file pressure
MAX_CONCURRENT_SOURCES = 5

def Gauss(x, A, mu, sigma, C):
    return A * np.exp(-0.5 * ((x - mu) / sigma)**2) + C

def velocity_from_fit(fitted_freq, rest_freq, c=299792.0):
    """Back-calculate the LSR velocity (km/s) from a fitted line centre."""
    return c * (1.0 - fitted_freq / rest_freq)


def validate_fit(fit_mu, fit_sigma, expected_freq, flux, fit_y,
                 max_sigma=1.2e8, max_offset_hz=1.5e9, min_snr=2.0):
    """Return True if the Gaussian fit is physically reasonable."""
    if abs(fit_sigma) > max_sigma:
        return False
    if abs(fit_mu - expected_freq) > max_offset_hz:
        return False
    residuals = flux - fit_y
    rms = np.sqrt(np.nanmean(residuals**2))
    if rms == 0:
        return True
    peak = np.nanmax(fit_y) - np.nanmin(fit_y)
    if peak / rms < min_snr:
        return False
    return True


def validate_line_spacing(fit_mu_a, fit_mu_b, nu0_a, nu0_b, v_lsr,
                          c=299792.0, tolerance_hz=LINE_SEPARATION_TOLERANCE_HZ):
    """Validate that two fitted lines have the expected separation."""
    expected_sep = abs(nu0_a - nu0_b) * (1 - v_lsr / c)
    measured_sep = abs(fit_mu_a - fit_mu_b)
    sep_error = abs(measured_sep - expected_sep)
    is_valid = sep_error <= tolerance_hz
    return is_valid, expected_sep, measured_sep, sep_error


def fit_gaussian(flux_in, freq_in):
    A0 = np.nanmax(flux_in) - np.nanmin(flux_in)
    mu0 = freq_in[np.nanargmax(flux_in)]
    sigma0 = 1e6
    C0 = np.nanmin(flux_in)

    p0 = [A0, mu0, sigma0, C0]

    parameters, _ = curve_fit(Gauss, freq_in, flux_in, p0=p0)
    fit_A, fit_mu, fit_sigma, fit_C = parameters
    fit_y = Gauss(freq_in, fit_A, fit_mu, fit_sigma, fit_C)
    return fit_y, fit_mu, fit_sigma


def estimate_source_rms_from_emission_free_channel(cube_data, freq_axis_hz, v_lsr,
                                                   lines, c=299792.0,
                                                   guard_hz=EMISSION_FREE_GUARD_HZ,
                                                   fixed_freq_hz=FIXED_REFERENCE_FREQ_HZ):
    """Estimate per-source channel RMS using a fixed reference frequency.

    Returns
    -------
    rms_chan : float
        RMS noise from the selected channel map.
    dnu_hz : float
        Frequency width of one channel [Hz].
    ch_idx : int
        Selected channel index.
    """
    avg_spectrum = np.nanmean(cube_data, axis=(1, 2))
    finite_spec = np.isfinite(avg_spectrum) & np.isfinite(freq_axis_hz)
    if finite_spec.sum() < 3:
        raise RuntimeError("Not enough finite channels to estimate RMS")

    # Doppler-shift the fixed reference frequency by v_lsr for this source
    fixed_freq_shifted = fixed_freq_hz * (1 - v_lsr / c)

    # Select the channel nearest to the shifted reference frequency
    _, ch_idx = find_nearest(freq_axis_hz, fixed_freq_shifted)

    # If the chosen channel is non-finite, try to fall back to the nearest finite channel
    if not finite_spec[ch_idx]:
        finite_indices = np.where(finite_spec)[0]
        if len(finite_indices) == 0:
            raise RuntimeError("No finite channels to estimate RMS")
        # pick finite index whose frequency is closest to fixed_freq_hz
        nearest_freqs = np.abs(freq_axis_hz[finite_indices] - fixed_freq_shifted)
        ch_idx = int(finite_indices[np.argmin(nearest_freqs)])

    ch_map = cube_data[ch_idx, :, :]
    rms_chan = float(np.nanstd(ch_map))
    if (not np.isfinite(rms_chan)) or rms_chan <= 0:
        raise RuntimeError("Could not estimate finite non-zero channel RMS")

    # Channel width in frequency units [Hz] to match moment-0 BUNIT (Hz Jy beam-1)
    dnu_hz = np.nan
    finite_freq = np.isfinite(freq_axis_hz)
    freq_good = np.asarray(freq_axis_hz)[finite_freq]
    if freq_good.size >= 2:
        df = np.abs(np.diff(freq_good))
        df = df[np.isfinite(df) & (df > 0)]
        if df.size > 0:
            dnu_hz = float(np.nanmedian(df))
    if (not np.isfinite(dnu_hz)) or dnu_hz <= 0:
        dnu_hz = float(FIXED_DNU_HZ)

    return rms_chan, dnu_hz, ch_idx

source_dirs = [d for d in sorted(os.listdir(sources_root), reverse=True)
               if os.path.isdir(os.path.join(sources_root, d))]
source_ids = source_dirs 
fits_list = []
for source_id in source_ids:
    almagal_dir = os.path.join(sources_root, source_id, 'images/combined/7MTM2TM1/almagal')
    fallback_dir = os.path.join(sources_root, source_id, 'images/combined/7MTM2TM1')

    search_dirs = []
    if os.path.isdir(almagal_dir):
        search_dirs.append(almagal_dir)
    if os.path.isdir(fallback_dir):
        search_dirs.append(fallback_dir)

    if not search_dirs:
        print(
            f"Skipping source {source_id}: missing both FITS dirs "
            f"{almagal_dir} and {fallback_dir}"
        )
        continue

    selected_fits = None
    selected_dir = None
    for d in search_dirs:
        candidates = [f for f in os.listdir(d)
                      if f.endswith('spw0_7MTM2TM1_jointdeconv.image.fits')]
        if candidates:
            selected_dir = d
            selected_fits = os.path.join(d, sorted(candidates)[0])
            break

    if selected_fits is None:
        print(
            f"Skipping source {source_id}: no spw0 jointdeconv cube in any of {search_dirs}"
        )
        continue

    if selected_dir != almagal_dir:
        print(f"Using fallback FITS dir for source {source_id}: {selected_dir}")

    fits_list.append((source_id, selected_fits))

def process_line(name, nu0_line, v_lsr, cube0, x, y, avg_spectrum, image_id, outdir,
                 c=299792.0, ref_name=None, ref_nu0=None, ref_fit_mu=None,
                 separation_tolerance_hz=LINE_SEPARATION_TOLERANCE_HZ,
                 rms_chan=None, dnu_hz=None):
    """Fit one spectral line and produce moment-0 products."""
    diag(f"  Processing line: {name}")
    nu = nu0_line * (1 - v_lsr / c)
    arr = np.array(cube0.spectral_axis)
    _, ind = find_nearest(arr, nu)

    sub_cube = cube0[(ind - 100):(ind + 200), :, :]

    spec = sub_cube[:, x, y]
    avg_flux = avg_spectrum[(ind - 100):(ind + 200)]

    freq = spec.spectral_axis.value

    finite = np.isfinite(avg_flux) & np.isfinite(freq)
    if finite.sum() < 5:
        diag(f"  Skipping Gaussian fit for {image_id}, {name}: not enough finite points")
        return None, None, False

    use_fallback = False
    try:
        fit_y, fit_mu, fit_sigma = fit_gaussian(avg_flux[finite], freq[finite])
    except Exception as e:
        diag(f"  DIAGNOSTIC: Gaussian fit failed for {name} ({e}); using fixed 1 MHz integration window")
        fit_y, fit_mu, fit_sigma = None, None, None
        use_fallback = True

    fit_ok = False
    if not use_fallback:
        fit_ok = validate_fit(fit_mu, fit_sigma, nu, avg_flux[finite], fit_y)
    if not fit_ok and not use_fallback:
           diag(f"  WARNING: fit for {image_id} {name} failed validation (mu={fit_mu:.1f}, "
               f"sigma={fit_sigma:.1f}, expected~{nu:.1f})")

    if fit_ok and ref_name is not None and ref_nu0 is not None and ref_fit_mu is not None:
        spacing_ok, expected_sep, measured_sep, sep_error = validate_line_spacing(
            fit_mu, ref_fit_mu, nu0_line, ref_nu0, v_lsr, c=c,
            tolerance_hz=separation_tolerance_hz
        )
        if not spacing_ok:
            diag(
                f"  DIAGNOSTIC: line spacing check failed for {name} vs {ref_name} "
                f"(expected Δν={expected_sep:.1f} Hz, measured Δν={measured_sep:.1f} Hz, "
                f"|Δ|={sep_error:.1f} Hz > {separation_tolerance_hz:.1f} Hz)"
            )
            if STRICT_LINE_SEPARATION_VALIDATION:
                fit_ok = False
        else:
            diag(
                f"  Spacing check passed for {name} vs {ref_name}: "
                f"expected Δν={expected_sep:.1f} Hz, measured Δν={measured_sep:.1f} Hz"
            )

    if not fit_ok:
        use_fallback = True
        diag(f"  DIAGNOSTIC: using fixed 1 MHz integration window centered on predicted frequency for {name}")

    plt.figure()
    plt.plot(freq, avg_flux, label='Data', color='k', linestyle='-')
    if fit_y is not None:
        plt.plot(freq[finite], fit_y, label='Fit', color='r', linestyle='-')
    plt.axvline(nu, color='blue', linestyle='--', alpha=0.5, label='Predicted freq')
    plt.legend()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Intensity")
    plt.title(f"{image_id} - {name} Gaussian Fit  (v_LSR={v_lsr:.2f} km/s)")
    plt.savefig(os.path.join(outdir, f"{image_id}_cubetest1_{name}.png"))
    plt.close()

    if use_fallback:

        half_window = 0.5 * FALLBACK_INTEGRATION_WINDOW_HZ
        lower_lim = nu - half_window
        upper_lim = nu + half_window
    else:
        sigma_abs = abs(fit_sigma)
        fwhm = 2.355 * sigma_abs
        upper_lim = fit_mu + 2 * fwhm
        lower_lim = fit_mu - 2 * fwhm

    # --- Ensure integration window is within spectral boundaries ---
    spectral_axis = np.array(sub_cube.spectral_axis)
    min_freq = spectral_axis.min()
    max_freq = spectral_axis.max()
    orig_lower_lim, orig_upper_lim = lower_lim, upper_lim
    # If window is out of bounds, shift it to fit within the spectral axis
    window_width = upper_lim - lower_lim
    if lower_lim < min_freq:
        lower_lim = min_freq
        upper_lim = lower_lim + window_width
    if upper_lim > max_freq:
        upper_lim = max_freq
        lower_lim = upper_lim - window_width
    # If window is still out of bounds (e.g., window wider than axis), clamp to axis
    lower_lim = max(lower_lim, min_freq)
    upper_lim = min(upper_lim, max_freq)
    # If after all this, window is still invalid, fallback to full sub_cube
    if lower_lim >= upper_lim:
        diag(f"    WARNING: Integration window invalid after boundary check. Using full sub_cube.")
        i0, i1 = 0, sub_cube.shape[0]
    else:
        _, upper_lim_idx = find_nearest(spectral_axis, upper_lim)
        _, lower_lim_idx = find_nearest(spectral_axis, lower_lim)
        i0, i1 = sorted((upper_lim_idx, lower_lim_idx))
        # Ensure at least one channel is selected
        if i0 == i1:
            diag(f"    WARNING: Integration window selects zero channels. Expanding to one channel.")
            i1 = min(i0+1, sub_cube.shape[0])
    sub_sub_cube = sub_cube[i0:i1, :, :]

    moment_0 = sub_sub_cube.moment(order=0)

    # AGSTDRES for moment-0 in matching units (Hz Jy beam-1):
    # sigma_M0 = sigma_chan * sqrt(dnu_hz * dv_hz)
    dv_hz = abs(float(upper_lim - lower_lim))
    dv_kms = float(c * dv_hz / abs(nu)) if nu != 0 else np.nan
    dvc_kms = float(c * dnu_hz / abs(nu)) if (dnu_hz is not None and np.isfinite(dnu_hz) and dnu_hz > 0 and nu != 0) else np.nan

    agstdres = np.nan
    if rms_chan is not None and dnu_hz is not None:
        if np.isfinite(dv_hz) and dv_hz > 0 and np.isfinite(rms_chan) and np.isfinite(dnu_hz) and dnu_hz > 0:
            agstdres = float(rms_chan * np.sqrt(dnu_hz * dv_hz))
        elif np.isfinite(rms_chan):
            # Fallback: use channel RMS directly when scaling is invalid
            agstdres = float(rms_chan)
            if (not np.isfinite(dv_hz)) or dv_hz <= 0:
                dv_hz = float(dnu_hz) if np.isfinite(dnu_hz) else np.nan
            if (not np.isfinite(dv_kms)) or dv_kms <= 0:
                dv_kms = float(dvc_kms) if np.isfinite(dvc_kms) else np.nan

    out_fits = os.path.join(outdir, f'{image_id}_{name}_moment0.fits')
    hdu = moment_0.hdu
    hdu.header['AGSTDRES'] = (float(agstdres) if np.isfinite(agstdres) else np.nan,
                              'Moment-0 RMS estimate [Hz Jy beam-1]')
    hdu.header['DNU_HZ'] = (float(dnu_hz) if dnu_hz is not None and np.isfinite(dnu_hz) else np.nan,
                            'Frequency width of one channel [Hz]')
    hdu.header['DV_HZ'] = (float(dv_hz) if np.isfinite(dv_hz) else np.nan,
                           'Frequency range used for moment-0 [Hz]')
    hdu.header['DVC_KMS'] = (float(dvc_kms) if dvc_kms is not None and np.isfinite(dvc_kms) else np.nan,
                             'Velocity width of one channel [km/s]')
    hdu.header['DV_KMS'] = (float(dv_kms) if np.isfinite(dv_kms) else np.nan,
                            'Velocity range used for moment-0 [km/s]')
    hdu.writeto(out_fits, overwrite=True)

    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(111, projection=moment_0.wcs)
    im = ax.imshow(moment_0.hdu.data, cmap="plasma")
    ax.set_xlabel("Galactic Longitude (degrees)", fontsize=16)
    ax.set_ylabel("Galactic Latitude (degrees)", fontsize=16)
    ax.set_title(f"{image_id} - {name} Moment-0 Map", fontsize=16)
    cbar = plt.colorbar(im, pad=0.07)
    cbar.set_label("", size=16)
    plt.savefig(os.path.join(outdir, f'{image_id}_{name}_moment.png'))
    plt.close()

    return fit_mu, fit_sigma, fit_ok



def process_source(args):
    image_id, fits_path = args

    pid = os.getpid()
    source_outdir = get_source_outdir(image_id)
    print(f"[PID {pid}] Starting {image_id}")
    os.makedirs(source_outdir, exist_ok=True)

    v_original = extract_velocity(image_id, info_table)
    if v_original is None:
        v = 30.0
        diag(f"  DIAGNOSTIC: catalogue v_LSR missing for {image_id}; using fallback v_LSR = {v} km/s")
    else:
        v = v_original

    try:
        cube0 = SpectralCube.read(fits_path)
    except Exception as e:
        diag(f"  DIAGNOSTIC: unreadable spectral cube, skipping source ({e})")
        return

    find_peak_data = cube0.filled_data[:].value
    ind_peak = np.unravel_index(np.nanargmax(find_peak_data), find_peak_data.shape)
    _, y, x = ind_peak

    spec_og = cube0[:, x, y]
    freq_og = spec_og.spectral_axis.value

    try:
        rms_chan, dnu_hz, rms_channel_idx = estimate_source_rms_from_emission_free_channel(
            find_peak_data, freq_og, v, lines, c=c
        )
        diag(
            f"  RMS channel idx={rms_channel_idx}, rms={rms_chan:.4e}, dnu={dnu_hz:.4e} Hz"
        )
    except Exception as e:
        rms_chan, dnu_hz = np.nan, np.nan
        diag(f"  WARNING: RMS estimate failed for {image_id}: {e}")

    sigma_map = np.nanstd(find_peak_data, axis=0)
    mask = np.nanmax(find_peak_data, axis=0) > 4 * sigma_map
    if not np.any(mask):
        mask_finite = np.isfinite(find_peak_data).any(axis=0)
        if np.any(mask_finite):
            selected_spectra = find_peak_data[:, mask_finite]
        else:
            selected_spectra = find_peak_data
    else:
        selected_spectra = find_peak_data[:, mask]
    avg_spectrum = np.nanmean(selected_spectra, axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(freq_og, avg_spectrum, lw=1)
    plt.xlabel("Frequency")
    plt.ylabel("Intensity")
    plt.title(f"Average spectrum >4σ - {image_id}")
    plt.grid(True)
    plt.savefig(os.path.join(source_outdir, f'spectrumtest_{image_id}.png'), dpi=200)
    plt.close()

    h2co_name, h2co_nu0 = lines[0]
    assert h2co_name == "H2CO", "First entry in `lines` must be H2CO"

    fit_mu_h2co, fit_sigma_h2co, h2co_valid = process_line(
        h2co_name, h2co_nu0, v, cube0, x, y, avg_spectrum, image_id, source_outdir,
        rms_chan=rms_chan, dnu_hz=dnu_hz
    )

    fit_results = {
        "H2CO": {
            "nu0": h2co_nu0,
            "fit_mu": fit_mu_h2co,
            "valid": h2co_valid,
        }
    }

    if h2co_valid and fit_mu_h2co is not None:
        v_new = velocity_from_fit(fit_mu_h2co, h2co_nu0)
        delta_v = v_new - v
        if 0.5 < abs(delta_v) < 200.0:
            diag(f"[PID {pid}] {image_id}: updating v_LSR {v:.2f} -> {v_new:.2f} km/s (Δv={delta_v:.2f})")
            v = v_new
        elif abs(delta_v) <= 0.5:
            diag(f"[PID {pid}] {image_id}: H2CO agrees with catalogue v_LSR")
        else:
            diag(f"[PID {pid}] {image_id}: |Δv| too large ({delta_v:.1f} km/s); keeping catalogue v_LSR")
    else:
        diag(f"[PID {pid}] {image_id}: H2CO fit invalid/not detected; keeping catalogue v_LSR")

    for name, nu0_loop in lines[1:]:
        ref_name = "H2CO" if fit_results["H2CO"]["valid"] else None
        ref_nu0 = fit_results["H2CO"]["nu0"] if fit_results["H2CO"]["valid"] else None
        ref_fit_mu = fit_results["H2CO"]["fit_mu"] if fit_results["H2CO"]["valid"] else None

        fit_mu_line, _, fit_valid_line = process_line(
            name, nu0_loop, v, cube0, x, y, avg_spectrum, image_id, source_outdir,
            ref_name=ref_name, ref_nu0=ref_nu0, ref_fit_mu=ref_fit_mu,
            rms_chan=rms_chan, dnu_hz=dnu_hz
        )
        fit_results[name] = {
            "nu0": nu0_loop,
            "fit_mu": fit_mu_line,
            "valid": fit_valid_line,
        }

    for line_a, line_b in [("SiO", "H2CO"), ("DCN", "H2CO"), ("SiO", "DCN")]:
        a = fit_results.get(line_a, {})
        b = fit_results.get(line_b, {})
        if a.get("valid") and b.get("valid"):
            spacing_ok, expected_sep, measured_sep, sep_error = validate_line_spacing(
                a["fit_mu"], b["fit_mu"],
                a["nu0"], b["nu0"],
                v, c=c, tolerance_hz=LINE_SEPARATION_TOLERANCE_HZ
            )
            if spacing_ok:
                diag(
                    f"[PID {pid}] {image_id}: {line_a}/{line_b} spacing ok (exp {expected_sep:.1f} Hz, obs {measured_sep:.1f} Hz)"
                )
            else:
                diag(
                    f"[PID {pid}] {image_id}: {line_a}/{line_b} spacing off by {sep_error:.1f} Hz (> {LINE_SEPARATION_TOLERANCE_HZ:.1f})"
                )
        else:
            diag(f"[PID {pid}] {image_id}: skipped {line_a}/{line_b} spacing check")

    print(f"[PID {pid}] Finished {image_id}")
    return image_id


if __name__ == "__main__":
    if TEST_ONE_SOURCE and TEST_SOURCE_ID is not None:
        test_matches = [(sid, fpath) for sid, fpath in fits_list if str(sid) == str(TEST_SOURCE_ID)]
        if not test_matches:
            print(f"TEST_SOURCE_ID={TEST_SOURCE_ID} not found in available sources.")
            pending_sources = []
        else:
            pending_sources = [test_matches[0]]
            if source_has_outputs(TEST_SOURCE_ID) and FORCE_REPROCESS_TEST_SOURCE:
                print(f"Test mode: reprocessing {TEST_SOURCE_ID} even though outputs already exist.")
            elif source_has_outputs(TEST_SOURCE_ID) and not FORCE_REPROCESS_TEST_SOURCE:
                print(f"Test mode: {TEST_SOURCE_ID} already has outputs. Set FORCE_REPROCESS_TEST_SOURCE=True to rerun.")
                pending_sources = []
            else:
                print(f"Test mode enabled: processing one source ({TEST_SOURCE_ID}).")
    else:
        pending_sources = []

    if not TEST_ONE_SOURCE:
        for src_id, fits_path in fits_list:
            if source_has_outputs(src_id):
                print(f"Skipping {src_id}: output already exists at {get_source_outdir(src_id)}")
            else:
                pending_sources.append((src_id, fits_path))
    elif TEST_ONE_SOURCE and TEST_SOURCE_ID is None:
        for src_id, fits_path in fits_list:
            if source_has_outputs(src_id):
                print(f"Skipping {src_id}: output already exists at {get_source_outdir(src_id)}")
            else:
                pending_sources.append((src_id, fits_path))
        if pending_sources:
            pending_sources = [pending_sources[0]]
            print(f"Test mode enabled: processing one source ({pending_sources[0][0]}).")

    pending_total = len(pending_sources)
    if pending_total == 0:
        print("No pending sources to process.")
    else:
        n_available = cpu_count()
        n_cores = max(1, n_available // 2)
        n_workers = min(n_cores, pending_total, MAX_CONCURRENT_SOURCES)

        if pending_total == 1 or n_workers == 1:
            print(f"Processing {pending_total} pending source sequentially.")
            completed = 0
            for src in pending_sources:
                process_source(src)
                completed += 1
                print(f"Progress: {completed}/{pending_total} sources")
                if completed % 5 == 0:
                    remaining = pending_total - completed
                    print(f"Batch update: {completed} done, {remaining} left")
        else:
            print(f"Processing {pending_total} pending sources on {n_workers} cores (capped at {MAX_CONCURRENT_SOURCES})...")
            completed = 0
            with Pool(n_workers) as p:
                for _ in p.imap_unordered(process_source, pending_sources, chunksize=1):
                    completed += 1
                    print(f"Progress: {completed}/{pending_total} sources")
                    if completed % 5 == 0:
                        remaining = pending_total - completed
                        print(f"Batch update: {completed} done, {remaining} left")