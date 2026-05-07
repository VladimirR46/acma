import numpy as np
from scipy.optimize import linear_sum_assignment

def unified_peyes_label(peyes_labels):
    """
    Converts EventLabelEnum objects from peyes to the CSV string format.
    """
    label_map = {
        'UNDEFINED': 'undef',
        'FIXATION': 'fix',
        'SACCADE': 'sac',
        'PSO': 'sac',
        'SMOOTH_PURSUIT': 'fix',
        'BLINK': 'blink'
    }
    string_labels = [lbl.name for lbl in peyes_labels]
    unified_labels = np.array([label_map.get(lbl, 'undef') for lbl in string_labels])
    return unified_labels


def get_events(labels, target_class='sac'):
    """
    Converts a per-sample label array into a list of events.
    Returns a list of (start_idx, end_idx) tuples for the target class.
    """
    is_target = (labels == target_class)

    padded = np.pad(is_target, (1, 1), mode='constant', constant_values=0)

    starts = np.where(padded[1:] & ~padded[:-1])[0]
    ends = np.where(~padded[1:] & padded[:-1])[0]

    events = list(zip(starts, ends))
    return events


def evaluate_event_level_1to1_2(y_true, y_pred, target_class='sac',
                              overlap_threshold=0.25, t_ms=None,
                              verbose=True):
    """
    One-to-one matching of target_class events (Hungarian algorithm on the
    overlap matrix). If t_ms is provided, additionally computes |Δonset| and
    |Δoffset| in ms over the same pairs that were counted as TP.

    Returns
    -------
    true_events, pred_events, metrics
        metrics : dict with keys precision, recall, f1, tp, fp, fn,
                  onset_ms, offset_ms (the last two are NaN if t_ms=None or TP=0).
    """
    true_events = get_events(y_true, target_class)
    pred_events = get_events(y_pred, target_class)

    n_true, n_pred = len(true_events), len(pred_events)

    metrics = {
        'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
        'tp': 0, 'fp': n_pred, 'fn': n_true,
        'onset_ms': np.nan, 'offset_ms': np.nan,
    }

    if n_true == 0 or n_pred == 0:
        if verbose:
            print(f"No events: ground_truth={n_true}, predicted={n_pred}")
        return true_events, pred_events, metrics

    overlap = np.zeros((n_true, n_pred))
    for i, (ts, te) in enumerate(true_events):
        for j, (ps, pe) in enumerate(pred_events):
            inter = max(0, min(te, pe) - max(ts, ps))
            union = max(te, pe) - min(ts, ps)
            overlap[i, j] = inter / union if union > 0 else 0.0

    cost = overlap.copy()
    cost[cost < overlap_threshold] = 0
    row_idx, col_idx = linear_sum_assignment(-cost)

    tp = 0
    on_errs, off_errs = [], []
    n_t = len(t_ms) if t_ms is not None else 0

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] < overlap_threshold:
            continue
        tp += 1
        if t_ms is None:
            continue
        ts, te = true_events[r]
        ps, pe = pred_events[c]
        t_off = min(max(te - 1, 0), n_t - 1)
        p_off = min(max(pe - 1, 0), n_t - 1)
        on_errs.append(abs(t_ms[ps] - t_ms[ts]))
        off_errs.append(abs(t_ms[p_off] - t_ms[t_off]))

    fn = n_true - tp
    fp = n_pred - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    metrics.update({
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'onset_ms':  float(np.mean(on_errs))  if on_errs  else np.nan,
        'offset_ms': float(np.mean(off_errs)) if off_errs else np.nan,
    })

    if verbose:
        print("=== Event-level (One-to-One) ===")
        print(f"Class: '{target_class}', IoU threshold: {overlap_threshold:.0%}")
        print(f"Ground truth: {n_true} | Predicted: {n_pred}")
        print(f"TP={tp}  FN={fn}  FP={fp}")
        print(f"Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")
        if t_ms is not None and tp > 0:
            print(f"Δ Onset  = {metrics['onset_ms']:.2f} ms")
            print(f"Δ Offset = {metrics['offset_ms']:.2f} ms")
        print("================================\n")

    return true_events, pred_events, metrics