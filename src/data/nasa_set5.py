import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat


def _safe_get(d: Dict[str, Any], key: str, default=None):
    return d[key] if key in d else default


def _to_scalar(x):
    try:
        if x is None:
            return np.nan
        arr = np.asarray(x)
        if arr.size == 0:
            return np.nan
        if arr.size == 1:
            v = arr.ravel()[0]
            try:
                return float(v)
            except Exception:
                return np.nan
        # vector -> use nanmedian as representative scalar
        return float(np.nanmedian(arr.astype(float)))
    except Exception:
        return np.nan


def _extract_cycle_rows(cycle_entry: Any) -> pd.DataFrame:
    # Cycle entries in NASA .mat are MATLAB structs (mat_struct). They expose fields as attributes.
    # Some variants may be dict-like. We support both.
    # Prefer a nested 'data' struct if present; otherwise use the entry itself as data container.
    if hasattr(cycle_entry, 'data'):
        data = getattr(cycle_entry, 'data')
    elif isinstance(cycle_entry, dict) and 'data' in cycle_entry:
        data = cycle_entry['data']
    else:
        data = cycle_entry

    # In case data is a 0-dim ndarray wrapping an object
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data.item()

    def _get_field(obj: Any, name: str):
        # Try attribute access first (mat_struct), then dict access.
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        return None

    def get_arr(name: str):
        arr = _get_field(data, name)
        if arr is None:
            return None
        arr = np.asarray(arr).squeeze()
        if arr.size == 0:
            return None
        try:
            return arr.astype(float)
        except Exception:
            # if object dtype or non-numeric, skip
            return None

    # Common fields in NASA battery set
    def first_non_none(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    t = first_non_none(get_arr('Time'), get_arr('time'))
    v_meas = first_non_none(get_arr('Voltage_measured'), get_arr('voltage_measured'), get_arr('Voltage'))
    i_meas = first_non_none(get_arr('Current_measured'), get_arr('current_measured'), get_arr('Current'))
    temp = first_non_none(get_arr('Temperature_measured'), get_arr('temperature_measured'), get_arr('Temperature'))
    ir = first_non_none(get_arr('IR'), get_arr('ir'), get_arr('Impedance'))
    cap = first_non_none(get_arr('Capacity'), get_arr('capacity'))

    n = 0
    for arr in [t, v_meas, i_meas, temp]:
        if arr is not None:
            n = max(n, len(arr))
    if n == 0:
        return pd.DataFrame()

    def pad_or_trim(arr, n):
        if arr is None:
            return np.full(n, np.nan)
        if len(arr) == n:
            return arr
        if len(arr) > n:
            return arr[:n]
        out = np.full(n, np.nan)
        out[: len(arr)] = arr
        return out

    df = pd.DataFrame({
        't': pad_or_trim(t, n),
        'V': pad_or_trim(v_meas, n),
        'I': pad_or_trim(i_meas, n),
        'T': pad_or_trim(temp, n),
    })

    # Attach per-cycle scalars (IR, Capacity) as columns
    # Prefer dedicated impedance fields in impedance cycles
    ir_scalar = np.nan
    # Try explicit IR vector/array first
    if ir is not None:
        try:
            arr = np.asarray(ir).astype(float).ravel()
            ir_scalar = float(np.nanmedian(arr)) if arr.size > 0 else np.nan
        except Exception:
            ir_scalar = _to_scalar(ir)
    # If not available, look for impedance-specific fields
    if np.isnan(ir_scalar):
        # data may have attributes like 'Re', 'Battery_impedance', 'Rectified_Impedance'
        re_field = _get_field(data, 'Re')
        if re_field is not None:
            ir_scalar = _to_scalar(re_field)
        else:
            imp = _get_field(data, 'Battery_impedance')
            if imp is None:
                imp = _get_field(data, 'Rectified_Impedance')
            if imp is not None:
                try:
                    arr = np.asarray(imp).ravel()
                    # Use real part median as ohmic resistance proxy
                    arr_real = np.real(arr.astype(np.complex128))
                    if arr_real.size > 0:
                        ir_scalar = float(np.nanmedian(arr_real))
                except Exception:
                    pass
    cap_scalar = _to_scalar(cap)

    df['IR_cycle'] = ir_scalar
    df['Capacity_cycle'] = cap_scalar if cap_scalar is not None else np.nan

    return df


def parse_mat_cell(mat_path: str) -> Tuple[str, List[pd.DataFrame], Dict[str, Any]]:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    # The key is typically like 'B0005'
    top_keys = [k for k in mat.keys() if k.startswith('B')]
    if not top_keys:
        # fallback: take any struct-like top-level key
        top_keys = [k for k, v in mat.items() if hasattr(v, '__dict__')]
    if not top_keys:
        raise ValueError(f"No battery struct found in {mat_path}")

    k = top_keys[0]
    root = mat[k]

    meta = {}
    for attr in ['description', 'ambient_temperature', 'type']:
        if hasattr(root, attr):
            meta[attr] = getattr(root, attr)

    cycles = []
    if hasattr(root, 'cycle'):
        cyc = root.cycle
        if isinstance(cyc, np.ndarray):
            iterable = list(cyc)
        else:
            iterable = [cyc]
        for idx, c in enumerate(iterable):
            if hasattr(c, 'type'):
                ctype = str(getattr(c, 'type'))
            else:
                ctype = 'unknown'
            df = _extract_cycle_rows(c)
            if df.empty:
                continue
            df['cycle_index'] = idx
            df['cycle_type'] = ctype
            cycles.append(df)
    else:
        # Some variants have 'data' directly
        if hasattr(root, 'data'):
            df = _extract_cycle_rows({'data': root.data})
            df['cycle_index'] = 0
            df['cycle_type'] = 'unknown'
            cycles = [df]
        else:
            raise ValueError(f"No cycle/data field in {mat_path}")

    cell_id = os.path.splitext(os.path.basename(mat_path))[0]
    return cell_id, cycles, meta


def make_cycle_table(cycles: List[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for df in cycles:
        ci = int(df['cycle_index'].iloc[0])
        ir = _to_scalar(df['IR_cycle'].iloc[0]) if 'IR_cycle' in df.columns else np.nan
        cap = _to_scalar(df['Capacity_cycle'].iloc[0]) if 'Capacity_cycle' in df.columns else np.nan
        ctype = str(df['cycle_type'].iloc[0])
        # Per-cycle temperature and discharge-current stats
        try:
            I = df['I'].values.astype(float)
            T = df['T'].values.astype(float)
            dis_mask = np.isfinite(I) & (I < 0)
            if dis_mask.any():
                temp_med = float(np.nanmedian(T[dis_mask]))
                iabs_med_dis = float(np.nanmedian(np.abs(I[dis_mask])))
            else:
                temp_med = float(np.nanmedian(T)) if np.isfinite(T).any() else np.nan
                iabs_med_dis = np.nan
        except Exception:
            temp_med = np.nan
            iabs_med_dis = np.nan
        rows.append({'cycle_index': ci,
                     'IR': ir,
                     'Capacity': cap,
                     'cycle_type': ctype,
                     'Temp_med': temp_med,
                     'Iabs_med_dis': iabs_med_dis})
    cyc_df = pd.DataFrame(rows).sort_values('cycle_index').reset_index(drop=True)
    # Forward-fill IR across cycles for visualization and downstream use (no backfill to avoid leakage)
    cyc_df['IR'] = cyc_df['IR'].replace([np.inf, -np.inf], np.nan).ffill()
    return cyc_df


def compute_labels(cyc_df: pd.DataFrame, eol_ir_mult: float = 1.7, eol_cap_frac: float = 0.8) -> pd.DataFrame:
    cyc_df = cyc_df.copy()
    # IR-based SOH and EoL (if available)
    finite_ir = cyc_df['IR'].replace([np.inf, -np.inf], np.nan)
    if finite_ir.notna().sum() > 0:
        ir_non_nan = finite_ir.dropna()
        r0 = np.median(ir_non_nan.iloc[: max(3, min(5, len(ir_non_nan)))])
        cyc_df['SOH_R'] = r0 / cyc_df['IR']
        eol_ir_threshold = eol_ir_mult * r0
        eol_idx_ir = cyc_df.index[cyc_df['IR'] >= eol_ir_threshold].tolist()
    else:
        cyc_df['SOH_R'] = np.nan
        eol_idx_ir = []

    # Capacity-based SOH and EoL (fallback when IR missing)
    finite_cap = cyc_df['Capacity'].replace([np.inf, -np.inf], np.nan)
    if finite_cap.notna().sum() > 0:
        cap_non_nan = finite_cap.dropna()
        q0 = np.median(cap_non_nan.iloc[: max(3, min(5, len(cap_non_nan)))])
        cyc_df['SOH_Q'] = cyc_df['Capacity'] / q0
        eol_cap_threshold = eol_cap_frac * q0
        eol_idx_cap = cyc_df.index[cyc_df['Capacity'] <= eol_cap_threshold].tolist()
    else:
        cyc_df['SOH_Q'] = np.nan
        eol_idx_cap = []

    # Determine earliest EoL by either criterion
    eol_candidates = sorted(set(eol_idx_ir + eol_idx_cap))
    if len(eol_candidates) > 0:
        eol_pos = int(eol_candidates[0])
    else:
        eol_pos = len(cyc_df) - 1

    cyc_df['EoL'] = False
    if 0 <= eol_pos < len(cyc_df):
        cyc_df.loc[eol_pos:, 'EoL'] = True

    cyc_df['RUL_cycles'] = np.maximum(0, eol_pos - cyc_df.index.values)
    return cyc_df


def load_all_cells(raw_dir: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.lower().endswith('.mat'):
            continue
        path = os.path.join(raw_dir, fname)
        try:
            cell_id, cycles, meta = parse_mat_cell(path)
            cyc_df = make_cycle_table(cycles)
            labels = compute_labels(cyc_df)
            out[cell_id] = {
                'cycles': cycles,
                'cycle_table': labels,
                'meta': meta,
            }
        except Exception as e:
            print(f"[WARN] Failed to parse {fname}: {e}")
    return out


def save_cycle_tables(dataset: Dict[str, Dict[str, Any]], out_csv: str):
    frames = []
    for cell_id, d in dataset.items():
        df = d['cycle_table'].copy()
        df['cell_id'] = cell_id
        frames.append(df)
    if frames:
        all_df = pd.concat(frames, axis=0, ignore_index=True)
        all_df.to_csv(out_csv, index=False)
        return all_df
    return pd.DataFrame()


if __name__ == "__main__":
    raw_dir = os.environ.get('NASA_SET5_RAW', 'data/nasa_set5/raw')
    out_csv = os.environ.get('NASA_SET5_SUMMARY', 'data/nasa_set5/summary.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    ds = load_all_cells(raw_dir)
    summary = save_cycle_tables(ds, out_csv)
    print(f"Parsed cells: {list(ds.keys())}")
    print(f"Saved summary to {out_csv} with shape {summary.shape}")
