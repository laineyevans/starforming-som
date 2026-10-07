import os
import matplotlib.pyplot as plt
from spectral_cube import SpectralCube
import numpy as np
from scipy.optimize import curve_fit

outdir = "/raid/scratch/mevans/spectral_cube_test"
os.makedirs(outdir, exist_ok=True)

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
                 separation_tolerance_hz=LINE_SEPARATION_TOLERANCE_HZ):
    """Fit one spectral line and produce moment-0 products."""
    print(f"  Processing line: {name}")
    nu = nu0_line * (1 - v_lsr / c)
    arr = np.array(cube0.spectral_axis)
    _, ind = find_nearest(arr, nu)

    sub_cube = cube0[(ind - 100):(ind + 200), :, :]

    spec = sub_cube[:, x, y]
    avg_flux = avg_spectrum[(ind - 100):(ind + 200)]

    freq = spec.spectral_axis.value

    finite = np.isfinite(avg_flux) & np.isfinite(freq)
    if finite.sum() < 5:
        print(f"  Skipping Gaussian fit for {image_id}, {name}: not enough finite points")
        return None, None, False

    use_fallback = False
    try:
        fit_y, fit_mu, fit_sigma = fit_gaussian(avg_flux[finite], freq[finite])
    except Exception as e:
        print(f"  DIAGNOSTIC: Gaussian fit failed for {name} ({e}); using fixed 1 MHz integration window")
        fit_y, fit_mu, fit_sigma = None, None, None
        use_fallback = True

    fit_ok = False
    if not use_fallback:
        fit_ok = validate_fit(fit_mu, fit_sigma, nu, avg_flux[finite], fit_y)
    if not fit_ok and not use_fallback:
        print(f"  WARNING: fit for {image_id} {name} failed validation (mu={fit_mu:.1f}, "
              f"sigma={fit_sigma:.1f}, expected~{nu:.1f})")

    if fit_ok and ref_name is not None and ref_nu0 is not None and ref_fit_mu is not None:
        spacing_ok, expected_sep, measured_sep, sep_error = validate_line_spacing(
            fit_mu, ref_fit_mu, nu0_line, ref_nu0, v_lsr, c=c,
            tolerance_hz=separation_tolerance_hz
        )
        if not spacing_ok:
            print(
                f"  DIAGNOSTIC: line spacing check failed for {name} vs {ref_name} "
                f"(expected Δν={expected_sep:.1f} Hz, measured Δν={measured_sep:.1f} Hz, "
                f"|Δ|={sep_error:.1f} Hz > {separation_tolerance_hz:.1f} Hz)"
            )
            if STRICT_LINE_SEPARATION_VALIDATION:
                fit_ok = False
        else:
            print(
                f"  Spacing check passed for {name} vs {ref_name}: "
                f"expected Δν={expected_sep:.1f} Hz, measured Δν={measured_sep:.1f} Hz"
            )

    if not fit_ok:
        use_fallback = True
        print(f"  DIAGNOSTIC: using fixed 1 MHz integration window centered on predicted frequency for {name}")

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

    _, upper_lim_idx = find_nearest(np.array(sub_cube.spectral_axis), upper_lim)
    _, lower_lim_idx = find_nearest(np.array(sub_cube.spectral_axis), lower_lim)
    i0, i1 = sorted((upper_lim_idx, lower_lim_idx))
    sub_sub_cube = sub_cube[i0:i1, :, :]

    moment_0 = sub_sub_cube.moment(order=0)
    moment_0.write(os.path.join(outdir, f'{image_id}_{name}_moment0.fits'), overwrite=True)

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


for image_id, fits_path in fits_list:

    print(f"\nProcessing source: {image_id}")
    source_outdir = os.path.join(outdir, str(image_id))
    os.makedirs(source_outdir, exist_ok=True)

    # --- DIAGNOSTIC: Check FITS file and cube properties ---
    try:
        cube0 = SpectralCube.read(fits_path)
    except Exception as e:
        print(f"  DIAGNOSTIC: unreadable spectral cube, skipping source ({e})")
        continue

    print(f"  DIAGNOSTIC: cube shape: {cube0.shape}")
    print(f"  DIAGNOSTIC: cube WCS: {cube0.wcs}")
    print(f"  DIAGNOSTIC: cube spectral axis: {cube0.spectral_axis}")
    print(f"  DIAGNOSTIC: cube min/max: {cube0.filled_data[:].value.min()} / {cube0.filled_data[:].value.max()}")

    # Check for all-NaN slices along the spectral axis
    find_peak_data = cube0.filled_data[:].value
    nan_slices = np.all(np.isnan(find_peak_data), axis=(1,2))
    print(f"  DIAGNOSTIC: Number of all-NaN frequency slices: {np.sum(nan_slices)} / {find_peak_data.shape[0]}")
    if np.any(nan_slices):
        print(f"  DIAGNOSTIC: Indices of all-NaN frequency slices: {np.where(nan_slices)[0]}")
        # Plot a diagnostic image showing which frequency slices are all-NaN
        plt.figure(figsize=(10,2))
        plt.plot(nan_slices.astype(int), drawstyle='steps-mid')
        plt.xlabel('Frequency channel index')
        plt.ylabel('All-NaN (1=True)')
        plt.title(f'All-NaN frequency slices: {image_id}')
        plt.savefig(os.path.join(source_outdir, f'all_nan_slices_{image_id}.png'), dpi=150)
        plt.close()

    # Plot a histogram of finite values in the cube
    finite_vals = find_peak_data[np.isfinite(find_peak_data)]
    plt.figure(figsize=(8,4))
    plt.hist(finite_vals, bins=100, log=True)
    plt.xlabel('Cube value')
    plt.ylabel('Log count')
    plt.title(f'Histogram of finite cube values: {image_id}')
    plt.savefig(os.path.join(source_outdir, f'cube_value_hist_{image_id}.png'), dpi=150)
    plt.close()

    v_original = extract_velocity(image_id, info_table)
    if v_original is None:
        v = 30.0
        print(f"  DIAGNOSTIC: catalogue v_LSR missing for {image_id}; using fallback v_LSR = {v} km/s")
    else:
        v = v_original

    try:
        cube0 = SpectralCube.read(fits_path)
    except Exception as e:
        print(f"  DIAGNOSTIC: unreadable spectral cube, skipping source ({e})")
        continue

    find_peak_data = cube0.filled_data[:].value
    ind_peak = np.unravel_index(np.nanargmax(find_peak_data), find_peak_data.shape)
    _, y, x = ind_peak

    spec_og = cube0[:, x, y]
    freq_og = spec_og.spectral_axis.value

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
        h2co_name, h2co_nu0, v, cube0, x, y, avg_spectrum, image_id, source_outdir
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

        if abs(delta_v) > 0.5 and abs(delta_v) < 200.0:
            print(f"  DIAGNOSTIC: updating v_LSR from {v:.4f} to {v_new:.4f} km/s (Δv={delta_v:.4f})")
            v = v_new
        elif abs(delta_v) <= 0.5:
            print("  DIAGNOSTIC: H2CO agrees with catalogue v_LSR (no update)")
        else:
            print(f"  DIAGNOSTIC: |Δv| too large ({delta_v:.1f} km/s), keeping catalogue v_LSR")
    else:
        print("  DIAGNOSTIC: H2CO fit invalid/not detected, keeping catalogue v_LSR")

    for name, nu0_loop in lines[1:]:
        ref_name = "H2CO" if fit_results["H2CO"]["valid"] else None
        ref_nu0 = fit_results["H2CO"]["nu0"] if fit_results["H2CO"]["valid"] else None
        ref_fit_mu = fit_results["H2CO"]["fit_mu"] if fit_results["H2CO"]["valid"] else None

        fit_mu_line, _, fit_valid_line = process_line(
            name, nu0_loop, v, cube0, x, y, avg_spectrum, image_id, source_outdir,
            ref_name=ref_name, ref_nu0=ref_nu0, ref_fit_mu=ref_fit_mu
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
                print(
                    f"  {line_a}/{line_b} spacing check passed: expected Δν={expected_sep:.1f} Hz, "
                    f"measured Δν={measured_sep:.1f} Hz"
                )
            else:
                print(
                    f"  WARNING: {line_a}/{line_b} spacing check failed "
                    f"(expected Δν={expected_sep:.1f} Hz, measured Δν={measured_sep:.1f} Hz, "
                    f"|Δ|={sep_error:.1f} Hz > {LINE_SEPARATION_TOLERANCE_HZ:.1f} Hz)"
                )
        else:
            print(f"  DIAGNOSTIC: skipped {line_a}/{line_b} spacing check (invalid or missing fit)")