import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation, binary_closing
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

def _add_delays(signal: np.ndarray, n_delay: int) -> np.ndarray:
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]
    n_samples, n_features = signal.shape
    slices = [signal[shift: shift + (n_samples - n_delay), :] for shift in range(n_delay + 1)]
    return np.hstack(slices)

def _cluster_weighting_binarized(xpos: np.ndarray, ypos: np.ndarray, n_delay: int = 5,
                                 window_time: float = 0.2, step_time: float = 0.02, freq: float = 1000.0) -> np.ndarray:
    stack_data = np.stack([xpos, ypos], axis=1)
    data = _add_delays(stack_data, n_delay=n_delay)

    num_samples = int(window_time * freq)
    step_size = max(1, int(step_time * freq))

    total_weights = np.zeros(len(xpos))

    for i in range(0, len(data) - num_samples + 1, step_size):
        idx = range(i, i + num_samples)

        orig_idx = range(i, i + num_samples + n_delay)
        if np.isnan(xpos[orig_idx]).any() or np.isnan(ypos[orig_idx]).any():
            continue

        window_data = data[idx]
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=5, max_iter=30)
        labels = kmeans.fit_predict(window_data)

        switches = np.abs(np.diff(labels))

        if np.sum(switches) == 1:
            fi = np.where(switches == 1)[0][0]
            global_idx = i + fi + (n_delay // 2)
            total_weights[global_idx] += 1

    peaks_mask = total_weights > 0

    block_size = int(0.04 * freq)
    saccade_mask = binary_dilation(peaks_mask, structure=np.ones(block_size))

    return saccade_mask.astype(float)

def preprocess_blinks(x: np.ndarray, y: np.ndarray, freq: float = 1000.0,
                      missing_value=np.nan, pad_ms: float = 50.0, max_interp_ms: float = 80.0):
    x_clean = np.array(x, dtype=float)
    y_clean = np.array(y, dtype=float)

    if missing_value is not None and not np.isnan(missing_value):
        x_clean[x_clean == missing_value] = np.nan
        y_clean[y_clean == missing_value] = np.nan

    mask = np.isnan(x_clean) | np.isnan(y_clean)

    pad_samples = int(np.round((pad_ms / 1000.0) * freq))
    if pad_samples > 0:
        mask = binary_dilation(mask, iterations=pad_samples)

    x_clean[mask] = np.nan
    y_clean[mask] = np.nan

    max_interp_samples = int(np.round((max_interp_ms / 1000.0) * freq))

    mask_int = np.pad(mask.astype(int), (1, 1), mode='constant', constant_values=0)
    diff = np.diff(mask_int)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        gap_length = e - s
        if gap_length <= max_interp_samples:

            left_anchor = max(0, s - 5)
            right_anchor = min(len(x_clean), e + 5)

            valid_mask = ~mask[left_anchor:right_anchor]
            valid_idx = np.arange(left_anchor, right_anchor)[valid_mask]

            if len(valid_idx) >= 4:
                gap_idx = np.arange(s, e)
                x_clean[gap_idx] = PchipInterpolator(valid_idx, x_clean[valid_idx])(gap_idx)
                y_clean[gap_idx] = PchipInterpolator(valid_idx, y_clean[valid_idx])(gap_idx)
                mask[gap_idx] = False  

    return x_clean, y_clean, mask

def subspace_pca_ecma(xpos: np.ndarray, ypos: np.ndarray, freq: float = 1000.0,
                      n_delay: int = 3, window_time: float = 0.2, step_time: float = 0.02):
    stack_data = np.stack([xpos, ypos], axis=1)
    data = _add_delays(stack_data, n_delay=n_delay)

    num_samples = int(window_time * freq)
    step_samples = max(1, int(step_time * freq))
    D = 2 * (n_delay + 1)

    total_weights = np.zeros(len(xpos))
    i = 0
    while i <= len(data) - num_samples:
        idx = range(i, i + num_samples)
        orig_idx = range(i, i + num_samples + n_delay)

        if np.isnan(xpos[orig_idx]).any() or np.isnan(ypos[orig_idx]).any():
            i += step_samples
            continue

        window_data = data[idx]
        centered_data = window_data - np.mean(window_data, axis=0)

        cov_matrix = np.cov(centered_data, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        total_variance = np.sum(eigvals)
        if total_variance < 1e-10:
            i += step_samples
            continue

        l1_ratio = eigvals[-1] / total_variance
        pc1 = eigvecs[:, -1]
        proj_1d = np.dot(centered_data, pc1).reshape(-1, 1)

        init_centers = np.array([[proj_1d[0, 0]], [proj_1d[-1, 0]]])
        if np.allclose(init_centers[0], init_centers[1]):
            init_centers[1] += 1e-5

        kmeans = KMeans(n_clusters=2, init=init_centers, n_init=1, max_iter=20)
        labels = kmeans.fit_predict(proj_1d)

        switches = np.abs(np.diff(labels))
        if np.sum(switches) == 1:
            fi = np.where(switches == 1)[0][0]
            global_idx = i + fi + (n_delay // 2)
            total_weights[global_idx] += l1_ratio

        i += step_samples

    max_possible_votes = num_samples / step_samples
    normalized_weights = total_weights / max_possible_votes
    structural_noise_floor = 1.0 / D

    valid_transitions = normalized_weights > structural_noise_floor
    nyquist_closure_size = int(step_samples * 2)
    saccade_mask = binary_closing(valid_transitions, structure=np.ones(nyquist_closure_size))

    return saccade_mask.astype(float)

def f(t):
    with np.errstate(over='ignore'):
        result = np.where(t >= 0, t + 0.25 * np.exp(-2 * t), 0.25 * np.exp(2 * t))
    return result

def saccadic_waveform(t, eta, c, tau):
    return c * f(eta * t / c) - c * f(eta * (t - tau) / c)

def waveform_xy(t, dur, c, amp, t0=None):
    eta = c * np.log(0.5 / 0.01 * (np.exp(amp / c) + 1)) / dur
    tau = amp / eta
    if t0 is None: t0 = -tau / 2
    return saccadic_waveform(t - t0, eta, c, tau)

def saccade_model_xy(t, dur, c, amp, alpha, k, t0=None, s0x=0, s0y=0):
    eta = c * np.log(0.5 / 0.01 * (np.exp(amp / c) + 1)) / dur
    tau = amp / eta
    if t0 is None: t0 = -tau / 2
    waveform = saccadic_waveform(t - t0, eta, c, tau)
    sacc_x = waveform * np.cos(alpha) + s0x
    sacc_y = k * waveform * np.sin(alpha) + s0y
    return np.concatenate((sacc_x, sacc_y))

def xy_to_dist(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def saccade_fit_xy(saccade_x, saccade_y, sampling_rate):
    duration = len(saccade_x) / sampling_rate
    t = np.linspace(-duration / 2, duration / 2, len(saccade_x))
    data_xy = np.concatenate((saccade_x, saccade_y))

    s0x_min, s0x_max = saccade_x.min(), saccade_x.max()
    s0y_min, s0y_max = saccade_y.min(), saccade_y.max()

    bounds = (
        [0.01, 200, 0.1, -np.pi, 0.9, t[0], s0x_min, s0y_min],
        [0.1, 1000, data_xy.max() - data_xy.min(), np.pi, 1.1, t[-1], s0x_max, s0y_max]
    )

    try:
        popt, _ = curve_fit(saccade_model_xy, t, data_xy, method='trf', bounds=bounds)
    except Exception:
        return None

    optimized_data = saccade_model_xy(t, *popt)
    sac_x, sac_y = optimized_data[:len(t)], optimized_data[len(t):]
    waveform = waveform_xy(t, popt[0], popt[1], popt[2], popt[5])

    vel = np.diff(waveform) * sampling_rate
    vel = np.abs(np.insert(vel, 0, 0))

    threshold = np.max(vel) * 0.05
    left_index = np.argmax(vel > threshold)
    right_index = len(vel) - 1 - np.argmax(vel[::-1] > threshold)

    info = {
        'sac_x': sac_x,
        'sac_y': sac_y,
        'dur': t[right_index] - t[left_index],
        'amp': np.abs(waveform[0] - waveform[-1]),
        'begin_peak': left_index,
        'end_peak': right_index,
        'dist': xy_to_dist(sac_x[left_index], sac_y[left_index], sac_x[right_index], sac_y[right_index])
    }
    return info

def _win_approx(saccade_mask, xpos, ypos, freq=1000.0, offset_time=0.1):
    padded_mask = np.insert(np.append(saccade_mask, 0), 0, 0)
    diff = np.diff(padded_mask)
    start_idx = np.where(diff == 1)[0]
    end_idx = np.where(diff == -1)[0] - 1

    offset = int(offset_time * freq)

    begin_peaks, end_peaks, durations, dist, peaks = [], [], [], [], []
    r2_x, r2_y = [], []
    curves_x, curves_y, curves_idx = [], [], []  

    for i in range(len(start_idx)):
        peak = (start_idx[i] + end_idx[i]) // 2

        left_border = max(0, peak - offset) if i == 0 else max(peak - offset, (end_idx[i - 1] + start_idx[i]) // 2)
        right_border = min(peak + offset, len(xpos)) if i == len(start_idx) - 1 else min(peak + offset, (start_idx[i + 1] + end_idx[i]) // 2)

        idx = range(left_border, right_border)
        win_x, win_y = xpos[idx], ypos[idx]

        fit_data = saccade_fit_xy(win_x, win_y, freq)

        if fit_data is not None and fit_data['dur'] >= 0.01 and fit_data['amp'] > 2:
            global_begin = idx[0] + fit_data['begin_peak']
            global_end = idx[0] + fit_data['end_peak']

            begin_peaks.append(global_begin)
            end_peaks.append(global_end)
            peaks.append((global_end + global_begin) // 2)
            durations.append(fit_data['dur'])
            dist.append(fit_data['dist'])
            r2_x.append(r2_score(win_x, fit_data['sac_x']))
            r2_y.append(r2_score(win_y, fit_data['sac_y']))

            curves_x.append(fit_data['sac_x'])
            curves_y.append(fit_data['sac_y'])
            curves_idx.append(list(idx)) 

    return {
        'dur': durations, 'peaks': peaks, 'start': begin_peaks, 'end': end_peaks,
        'dist': dist, 'r2_x': r2_x, 'r2_y': r2_y,
        'curves_x': curves_x, 'curves_y': curves_y, 'curves_idx': curves_idx
    }


def acma(xpos, ypos, freq=1000.0, n_delay=3, r2_th=0.6, method='pca', missing_value=np.nan, window_time=0.2, step_time=0.02, approx_offset=0.05):
    x_clean, y_clean, blink_mask = preprocess_blinks(xpos, ypos, freq=freq, missing_value=missing_value)

    if method == 'pca':
        saccade_mask = subspace_pca_ecma(x_clean, y_clean, freq=freq, n_delay=n_delay, window_time=window_time, step_time=step_time)
    else:
        saccade_mask = _cluster_weighting_binarized(x_clean, y_clean, freq=freq, n_delay=n_delay, window_time=window_time, step_time=step_time)

    sac_info = _win_approx(saccade_mask, x_clean, y_clean, freq=freq, offset_time=approx_offset)

    r2_max = np.maximum(np.array(sac_info['r2_x']), np.array(sac_info['r2_y']))
    indices = np.where(r2_max > r2_th)[0]

    sac_filtered = {key: [value[i] for i in indices] for key, value in sac_info.items()}
    sac_filtered['raw_mask'] = saccade_mask
    labels = np.full(len(xpos), 'fix', dtype=object)
    labels[np.isnan(xpos) | np.isnan(ypos)] = 'blink'
    for s, e in zip(sac_filtered['start'], sac_filtered['end']):
        labels[s:e+1] = 'sac'
    sac_filtered['label'] = labels
    return sac_filtered