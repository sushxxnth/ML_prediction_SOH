"""
Randomized Battery Usage Dataset Loader (NASA Random Walk)

Parses the NASA Random Walk Li-ion dataset (RW9–RW12) containing mixed
charge/discharge steps with recorded temperature, current, voltage, and
periodic reference cycles for capacity benchmarking.

We extract reference discharge cycles to derive capacity/SOH and basic
cycle-level statistics to integrate with the unified pipeline.

Author: Battery ML Research
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy.io import loadmat

from .base_loader import BaseBatteryLoader, CellData, CycleData

# Context helpers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.context.extended_context import (
    ExtendedBatteryContext,
    TemperatureContext,
    ChemistryContext,
    UsageProfileContext,
    CRateContext,
    create_nasa_context
)


class RandomizedBatteryLoader(BaseBatteryLoader):
    """
    Loader for the NASA Random Walk Battery Usage dataset (RW9–RW12).
    """

    @property
    def dataset_name(self) -> str:
        return "randomized_battery_usage"

    @property
    def default_chemistry(self) -> str:
        # Li-ion 18650 cells, treat as LCO for chemistry id
        return "LCO"

    @property
    def default_temperature(self) -> float:
        # Room temp for this subset; other zips (40C) can be parsed similarly
        return 25.0

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        nominal_capacity: float = 2.0  # Ah (typical for 18650 in this set)
    ):
        super().__init__(data_dir, cache_dir, use_cache)
        self.nominal_capacity = nominal_capacity
        # Infer temperature from path (40C folders)
        if "40C" in str(self.data_dir):
            self.default_temperature = 40.0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _integrate_capacity(self, current: np.ndarray, time_s: np.ndarray) -> float:
        """Integrate current over time to get Ah (discharge positive)."""
        if len(current) == 0 or len(time_s) == 0:
            return np.nan
        cap_as = np.trapz(current, time_s)  # A*s
        return cap_as / 3600.0

    def _parse_mat_file(self, file_path: Path) -> CellData:
        """Parse a single MAT file and return CellData."""
        mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
        data = mat.get("data", None)
        if data is None or not hasattr(data, "step"):
            raise ValueError(f"No 'data.step' in {file_path}")

        steps = data.step
        if not isinstance(steps, np.ndarray):
            steps = np.array([steps])

        ref_caps = []
        ref_cycles: List[CycleData] = []

        # Identify reference discharge steps by comment
        for idx, st in enumerate(steps):
            comment = getattr(st, "comment", "")
            comment = comment if isinstance(comment, str) else str(comment)
            current = np.asarray(getattr(st, "current", np.array([]))).flatten()
            rel_time = np.asarray(getattr(st, "relativeTime", np.array([]))).flatten()

            if "reference discharge" in comment.lower():
                cap = self._integrate_capacity(current, rel_time)
                ref_caps.append(cap)

        if len(ref_caps) == 0:
            raise ValueError(f"No reference discharge steps found in {file_path}")

        cap0 = ref_caps[0] if ref_caps[0] > 0 else np.nanmax(ref_caps)
        # Build cycles from reference discharges only (as capacity benchmarks)
        cycle_idx = 0
        for idx, st in enumerate(steps):
            comment = getattr(st, "comment", "")
            comment = comment if isinstance(comment, str) else str(comment)
            current = np.asarray(getattr(st, "current", np.array([]))).flatten()
            voltage = np.asarray(getattr(st, "voltage", np.array([]))).flatten()
            temp = np.asarray(getattr(st, "temperature", np.array([]))).flatten()
            rel_time = np.asarray(getattr(st, "relativeTime", np.array([]))).flatten()
            time_s = rel_time

            if "reference discharge" not in comment.lower():
                continue

            cap = self._integrate_capacity(current, time_s)
            soh_cap = cap / cap0 if cap0 and cap0 > 0 else np.nan

            temp_mean = float(np.nanmean(temp)) if len(temp) else self.default_temperature
            temp_max = float(np.nanmax(temp)) if len(temp) else temp_mean
            temp_min = float(np.nanmin(temp)) if len(temp) else temp_mean
            current_mean = float(np.nanmean(current)) if len(current) else 0.0
            current_max = float(np.nanmax(np.abs(current))) if len(current) else 0.0
            voltage_min = float(np.nanmin(voltage)) if len(voltage) else 3.2
            voltage_max = float(np.nanmax(voltage)) if len(voltage) else 4.2
            charge_time = float(np.sum(time_s[current < 0])) if len(time_s) else 0.0
            discharge_time = float(np.sum(time_s[current > 0])) if len(time_s) else 0.0

            context = ExtendedBatteryContext()
            context.temperature = TemperatureContext.from_celsius(temp_mean)
            context.chemistry = ChemistryContext.LCO
            context.usage_profile = UsageProfileContext.MIXED_DRIVING
            context.c_rate = CRateContext.from_c_rate(current_max / max(self.nominal_capacity, 0.1))
            context.source_dataset = self.dataset_name

            cyc = CycleData(
                cell_id=file_path.stem,
                cycle_index=cycle_idx,
                capacity=float(cap),
                internal_resistance=np.nan,
                soh_capacity=float(soh_cap),
                soh_resistance=np.nan,
                rul_cycles=0,  # will be filled downstream
                temperature_mean=temp_mean,
                temperature_max=temp_max,
                temperature_min=temp_min,
                current_mean=current_mean,
                current_max=current_max,
                voltage_min=voltage_min,
                voltage_max=voltage_max,
                charge_time=charge_time,
                discharge_time=discharge_time,
                context=context,
                time_series=None
            )
            ref_cycles.append(cyc)
            cycle_idx += 1

        # Build CellData
        cell = CellData(
            cell_id=file_path.stem,
            source_dataset=self.dataset_name,
            chemistry=self.default_chemistry,
            nominal_capacity=self.nominal_capacity,
            nominal_voltage=3.7,
            form_factor="cylindrical",
            test_temperature=self.default_temperature,
            charge_rate=1.0,
            discharge_rate=1.0,
            usage_profile="random_walk",
            cycles=ref_cycles
        )
        return cell

    # ------------------------------------------------------------------ #
    # Required abstract implementations
    # ------------------------------------------------------------------ #
    def _parse_raw_data(self) -> Dict[str, CellData]:
        """Parse all MAT files under the data_dir."""
        cells: Dict[str, CellData] = {}
        mat_files = list(Path(self.data_dir).rglob("*.mat"))
        if not mat_files:
            print(f"[WARN] No MAT files found in {self.data_dir}")
            return cells

        for mf in mat_files:
            if "__MACOSX" in str(mf):
                continue
            try:
                cell = self._parse_mat_file(mf)
                cells[cell.cell_id] = cell
                print(f"Loaded {cell.cell_id}: {len(cell.cycles)} reference cycles")
            except Exception as e:
                print(f"[WARN] Failed to parse {mf}: {e}")
                continue

        return cells

    def _create_context(self, cell: CellData) -> ExtendedBatteryContext:
        """Create a base context for the cell (used if per-cycle missing)."""
        ctx = ExtendedBatteryContext()
        ctx.temperature = TemperatureContext.from_celsius(getattr(cell, 'test_temperature', self.default_temperature))
        ctx.chemistry = ChemistryContext.LCO
        ctx.usage_profile = UsageProfileContext.MIXED_DRIVING
        ctx.c_rate = CRateContext.NORMAL
        ctx.source_dataset = self.dataset_name
        return ctx

