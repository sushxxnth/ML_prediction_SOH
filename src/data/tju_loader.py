"""
TJU Battery Dataset Loader

Loads the TJU NCM/NCA battery dataset stored as a NumPy dict of DataFrames.
This dataset contains cycle-level summary features and capacity measurements.

File: Dataset_3_NCM_NCA_battery_1C.npy
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.base_loader import BaseBatteryLoader, CellData, CycleData
from src.context.extended_context import (
    ExtendedBatteryContext,
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext
)


class TJULoader(BaseBatteryLoader):
    """Loader for TJU NCM/NCA dataset."""

    @property
    def dataset_name(self) -> str:
        return "tju"

    @property
    def default_chemistry(self) -> str:
        return "NMC"

    @property
    def default_temperature(self) -> float:
        return 25.0

    def _parse_raw_data(self) -> Dict[str, CellData]:
        cells: Dict[str, CellData] = {}

        npy_path = self.data_dir / "Dataset_3_NCM_NCA_battery_1C.npy"
        if not npy_path.exists():
            print(f"[WARN] TJU data not found at {npy_path}")
            return cells

        data = np.load(npy_path, allow_pickle=True).item()

        for cell_name, df in data.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            # Capacity series
            capacity = df["Capacity"].astype(float).values
            if len(capacity) == 0:
                continue

            initial_capacity = float(np.median(capacity[:5])) if len(capacity) >= 5 else float(capacity[0])
            if not np.isfinite(initial_capacity) or initial_capacity <= 0:
                initial_capacity = float(capacity[0]) if capacity[0] > 0 else 1.0

            cell = CellData(
                cell_id=cell_name,
                source_dataset="tju",
                chemistry="NMC",
                nominal_capacity=initial_capacity,
                nominal_voltage=3.7,
                form_factor="18650",
                test_temperature=25.0,
                charge_rate=1.0,
                discharge_rate=1.0,
                usage_profile="constant_current",
            )

            for idx, row in df.iterrows():
                cycle_idx = row.get("cycle index", row.get("Cycle", idx))
                try:
                    cycle_index = int(cycle_idx)
                except Exception:
                    cycle_index = int(idx)

                cap = float(row.get("Capacity", np.nan))

                # Voltage statistics
                v_mean = float(row.get("voltage mean", np.nan))
                v_std = float(row.get("voltage std", np.nan))
                if np.isfinite(v_mean) and np.isfinite(v_std):
                    voltage_min = v_mean - 2.0 * abs(v_std)
                    voltage_max = v_mean + 2.0 * abs(v_std)
                else:
                    voltage_min = np.nan
                    voltage_max = np.nan

                # Current statistics
                i_mean = float(row.get("current mean", np.nan))
                i_std = float(row.get("current std", np.nan))
                current_max = i_mean + abs(i_std) if np.isfinite(i_mean) and np.isfinite(i_std) else np.nan

                # Charge/discharge time (approximate)
                cc_time = float(row.get("CC charge time", 0.0))
                cv_time = float(row.get("CV charge time", 0.0))
                charge_time = max(0.0, cc_time + cv_time)
                discharge_time = charge_time  # best-effort fallback

                cycle = CycleData(
                    cell_id=cell_name,
                    cycle_index=cycle_index,
                    capacity=cap,
                    internal_resistance=np.nan,
                    soh_capacity=1.0,
                    soh_resistance=np.nan,
                    rul_cycles=0,
                    temperature_mean=25.0,
                    temperature_max=25.0,
                    temperature_min=25.0,
                    current_mean=i_mean if np.isfinite(i_mean) else 0.0,
                    current_max=current_max if np.isfinite(current_max) else 0.0,
                    voltage_min=voltage_min if np.isfinite(voltage_min) else 0.0,
                    voltage_max=voltage_max if np.isfinite(voltage_max) else 0.0,
                    charge_time=charge_time,
                    discharge_time=discharge_time,
                )

                cell.cycles.append(cycle)

            if cell.cycles:
                cells[cell.cell_id] = cell

        return cells

    def _create_context(self, cell: CellData) -> ExtendedBatteryContext:
        return ExtendedBatteryContext(
            temperature=TemperatureContext.ROOM,
            chemistry=ChemistryContext.NMC,
            usage_profile=UsageProfileContext.CONSTANT_CURRENT,
            c_rate=CRateContext.NORMAL,
            temperature_celsius=25.0,
            c_rate_value=1.0,
            soc_pct=50.0,
            source_dataset="tju",
            cell_id=cell.cell_id,
            additional_info={
                "charge_rate": cell.charge_rate,
                "discharge_rate": cell.discharge_rate,
            },
        )
