"""
Battery Dataset Registry

Central registry of open-source battery datasets with:
- Download URLs and documentation
- Data format specifications
- Context metadata (temperature, chemistry, etc.)
- Preprocessing requirements

Supported Datasets:
1. NASA Battery Dataset (baseline) - Room temp, LCO
2. Sandia National Labs - Multi-temperature (5°C-45°C), Multi-chemistry
3. CALCE (UMD) - Multi-chemistry, Dynamic driving profiles
4. Oxford Battery Degradation - Dynamic urban profiles, LCO
5. TBSI Sunwoda - Fast charging, EV conditions
6. McMaster University - Multi-temp, fast charging
7. BatteryLife (Meta-dataset) - 80 chemistries, 12 temperatures

Author: Battery ML Research
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class DataFormat(Enum):
    """Supported data formats"""
    MAT = "mat"           # MATLAB .mat files (NASA, CALCE)
    CSV = "csv"           # CSV files
    HDF5 = "hdf5"         # HDF5 files
    PARQUET = "parquet"   # Parquet files
    JSON = "json"         # JSON files
    PICKLE = "pickle"     # Python pickle
    EXCEL = "xlsx"        # Excel files


@dataclass
class DatasetInfo:
    """
    Comprehensive information about a battery dataset.
    
    Attributes:
        name: Short identifier
        full_name: Full descriptive name
        description: Dataset description
        url: Primary download URL
        documentation_url: Link to documentation/paper
        
        # Data characteristics
        format: Primary data format
        num_cells: Number of battery cells
        chemistries: List of battery chemistries
        temperatures: List of test temperatures (°C)
        c_rates: List of C-rates tested
        profiles: List of usage profiles
        
        # Metadata
        citation: How to cite this dataset
        license: Data license
        size_mb: Approximate size in MB
        
        # Processing
        requires_extraction: Whether ZIP extraction needed
        preprocessing_notes: Special preprocessing requirements
    """
    
    name: str
    full_name: str
    description: str
    url: str
    documentation_url: str = ""
    
    # Data characteristics
    format: DataFormat = DataFormat.CSV
    num_cells: int = 0
    chemistries: List[str] = field(default_factory=list)
    temperatures: List[float] = field(default_factory=list)
    c_rates: List[float] = field(default_factory=list)
    profiles: List[str] = field(default_factory=list)
    
    # Metadata
    citation: str = ""
    license: str = ""
    size_mb: int = 0
    
    # Processing
    requires_extraction: bool = False
    preprocessing_notes: str = ""
    
    def __repr__(self) -> str:
        return (f"DatasetInfo({self.name}: {self.num_cells} cells, "
                f"chemistries={self.chemistries}, temps={self.temperatures})")


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_REGISTRY: Dict[str, DatasetInfo] = {
    
    # -------------------------------------------------------------------------
    # NASA Battery Dataset (Baseline)
    # -------------------------------------------------------------------------
    "nasa": DatasetInfo(
        name="nasa",
        full_name="NASA Prognostics Center Battery Dataset",
        description="""
        NASA's battery aging dataset from the Prognostics Center of Excellence.
        18650 Li-ion cells aged under various charge/discharge cycles at room temperature.
        Includes capacity fade and impedance growth measurements.
        """,
        url="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/",
        documentation_url="https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/",
        format=DataFormat.MAT,
        num_cells=34,
        chemistries=["LCO"],
        temperatures=[24.0],  # Room temperature only
        c_rates=[1.0, 2.0],
        profiles=["constant_current"],
        citation="B. Saha and K. Goebel (2007). NASA Ames Prognostics Data Repository",
        license="Public Domain",
        size_mb=500,
        requires_extraction=True,
        preprocessing_notes="MATLAB .mat files with nested structures. Use scipy.io.loadmat."
    ),
    
    # -------------------------------------------------------------------------
    # Sandia National Labs (Temperature King)
    # -------------------------------------------------------------------------
    "sandia": DatasetInfo(
        name="sandia",
        full_name="Sandia National Laboratories Battery Archive",
        description="""
        Extensive battery cycling data from Sandia National Labs.
        Multiple cell chemistries (LCO, LFP, NCA, NMC) tested at various temperatures.
        Key dataset for understanding temperature effects on degradation.
        
        Temperature range: 5°C to 45°C
        Includes short-term cycling and long-term degradation data.
        """,
        url="https://www.batteryarchive.org/study_summaries.html",
        documentation_url="https://www.batteryarchive.org/",
        format=DataFormat.CSV,
        num_cells=200,
        chemistries=["LCO", "LFP", "NCA", "NMC"],
        temperatures=[5.0, 15.0, 25.0, 35.0, 45.0],
        c_rates=[0.5, 1.0, 2.0, 3.0],
        profiles=["constant_current"],
        citation="Sandia National Laboratories Battery Archive",
        license="Public Domain (US Government)",
        size_mb=2000,
        requires_extraction=True,
        preprocessing_notes="""
        Data available via BatteryArchive.org API.
        Download specific studies: SNL_18650_LCO, SNL_18650_LFP, etc.
        CSV format with standard columns.
        """
    ),
    
    # -------------------------------------------------------------------------
    # CALCE (Chemistry + Driving Profiles)
    # -------------------------------------------------------------------------
    "calce": DatasetInfo(
        name="calce",
        full_name="CALCE Battery Research Group (University of Maryland)",
        description="""
        Comprehensive battery data from the Center for Advanced Life Cycle Engineering.
        Multiple chemistries and dynamic driving profiles.
        
        Key profiles:
        - DST (Dynamic Stress Test)
        - FUDS (Federal Urban Driving Schedule)
        - US06 (Highway Driving Schedule)
        - BJDST (Beijing Dynamic Stress Test)
        
        Temperature testing at 0°C, 25°C, and 45°C.
        """,
        url="https://calce.umd.edu/battery-data",
        documentation_url="https://calce.umd.edu/battery-data",
        format=DataFormat.MAT,
        num_cells=50,
        chemistries=["LCO", "NMC", "LFP"],
        temperatures=[0.0, 25.0, 45.0],
        c_rates=[0.5, 1.0, 2.0],
        profiles=["DST", "FUDS", "US06", "BJDST", "constant_current"],
        citation="""
        CALCE Battery Research Group, University of Maryland.
        https://calce.umd.edu/battery-data
        """,
        license="Academic Research Use",
        size_mb=1500,
        requires_extraction=True,
        preprocessing_notes="""
        Multiple sub-datasets:
        - CS2: Prismatic cells (LCO)
        - CX2: Cylindrical cells
        - PL: Pouch cells (NMC)
        MATLAB .mat format similar to NASA.
        """
    ),
    
    # -------------------------------------------------------------------------
    # Oxford Battery Degradation (Dynamic Urban Profiles)
    # -------------------------------------------------------------------------
    "oxford": DatasetInfo(
        name="oxford",
        full_name="Oxford Battery Degradation Dataset",
        description="""
        Small LCO pouch cells tested using dynamic urban driving profiles.
        Derived from real-world driving data collected in Oxford, UK.
        
        Unique value: Real-world applicability testing with actual
        street driving profiles (stop-start, acceleration patterns).
        """,
        url="https://www.batteryarchive.org/study_summaries.html",
        documentation_url="https://howey.eng.ox.ac.uk/data-and-code/",
        format=DataFormat.CSV,
        num_cells=8,
        chemistries=["LCO"],
        temperatures=[25.0],
        c_rates=[1.0, 2.0],
        profiles=["urban_driving", "mixed_driving"],
        citation="""
        Birkl et al. "Degradation diagnostics for lithium ion cells"
        Journal of Power Sources, 2017
        """,
        license="CC BY 4.0",
        size_mb=200,
        requires_extraction=False,
        preprocessing_notes="CSV format with time-series voltage/current data."
    ),
    
    # -------------------------------------------------------------------------
    # TBSI Sunwoda (Fast Charging, EV Conditions)
    # -------------------------------------------------------------------------
    "tbsi_sunwoda": DatasetInfo(
        name="tbsi_sunwoda",
        full_name="TBSI Sunwoda Battery Dataset",
        description="""
        Industrial-grade battery data from Sunwoda Electronic Co., Ltd
        and Tsinghua Berkeley Shenzhen Institute.
        
        Key features:
        - Real EV charging conditions
        - Wide temperature range
        - Fast charging protocols
        - Initial manufacturing variability data
        """,
        url="https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset",
        documentation_url="https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset",
        format=DataFormat.CSV,
        num_cells=100,
        chemistries=["NMC"],
        temperatures=[10.0, 25.0, 35.0, 45.0],
        c_rates=[0.5, 1.0, 2.0, 3.0, 4.0],
        profiles=["EV_charging", "fast_charging"],
        citation="TBSI-Sunwoda Battery Dataset, GitHub",
        license="MIT",
        size_mb=500,
        requires_extraction=True,
        preprocessing_notes="""
        GitHub repository with Python scripts for loading.
        CSV format organized by cell ID and test condition.
        """
    ),

    # -------------------------------------------------------------------------
    # NASA Randomized Battery Usage (Random Walk) - RW9–RW12
    # -------------------------------------------------------------------------
    "randomized_battery_usage": DatasetInfo(
        name="randomized_battery_usage",
        full_name="NASA Randomized Battery Usage (Random Walk) RW9–RW12",
        description="""
        NASA Ames Random Walk Li-ion 18650 cells (RW9–RW12). Mixed random-walk
        charge/discharge currents with periodic reference cycles for capacity
        benchmarking. Contains temperature, current, voltage, and step-level
        metadata.
        """,
        url="local",
        documentation_url="local (see README_RW_ChargeDischarge_RT)",
        format=DataFormat.MAT,
        num_cells=4,
        chemistries=["LCO"],
        temperatures=[25.0, 40.0],
        c_rates=[0.5, 1.0, 2.0, 4.5],
        profiles=["random_walk"],
        citation="Bole, Kulkarni, Daigle (PHM 2014), NASA Random Walk",
        license="Public Domain (NASA)",
        size_mb=1000,
        requires_extraction=True,
        preprocessing_notes="MAT files with 'data.step' structure; reference discharges used for capacity."
    ),
    
    # -------------------------------------------------------------------------
    # McMaster University (Multi-temp + Fast Charging)
    # -------------------------------------------------------------------------
    "mcmaster": DatasetInfo(
        name="mcmaster",
        full_name="McMaster University Battery Research Group Datasets",
        description="""
        High-quality battery datasets from McMaster's battery research group.
        
        Key datasets:
        - Multi-temperature characterization
        - Drive cycle data
        - Fast charging protocols
        - EIS measurements
        """,
        url="https://battery.mcmaster.ca/research/datasets-and-algorithms/",
        documentation_url="https://battery.mcmaster.ca/research/datasets-and-algorithms/",
        format=DataFormat.CSV,
        num_cells=50,
        chemistries=["NMC", "LFP"],
        temperatures=[-10.0, 0.0, 10.0, 25.0, 35.0, 45.0],
        c_rates=[0.5, 1.0, 2.0, 3.0],
        profiles=["constant_current", "drive_cycle", "fast_charging"],
        citation="McMaster Battery Research Group",
        license="Academic Research Use",
        size_mb=800,
        requires_extraction=True,
        preprocessing_notes="Multiple sub-datasets with different formats."
    ),
    
    # -------------------------------------------------------------------------
    # BatteryLife Meta-Dataset (80 chemistries, 12 temperatures)
    # -------------------------------------------------------------------------
    "batterylife": DatasetInfo(
        name="batterylife",
        full_name="BatteryLife: Comprehensive Battery Life Prediction Dataset",
        description="""
        Meta-dataset integrating 16 existing datasets, standardized format.
        
        Covers:
        - 998 batteries
        - 8 form factors
        - 80 chemical systems
        - 12 operating temperatures
        - 646 charge/discharge protocols
        
        Includes Li-ion, Zn-ion, and Na-ion batteries.
        """,
        url="https://github.com/Ruifeng-Tan/BatteryLife",
        documentation_url="https://arxiv.org/abs/2502.18807",
        format=DataFormat.PARQUET,
        num_cells=998,
        chemistries=[
            "LCO", "NMC", "NCA", "LFP", "LMO",
            "NMC111", "NMC523", "NMC622", "NMC811",
            "Zn-ion", "Na-ion"
        ],
        temperatures=[-20.0, -10.0, 0.0, 10.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
        c_rates=[0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        profiles=[
            "constant_current", "DST", "FUDS", "US06",
            "urban_driving", "highway_driving"
        ],
        citation="""
        Tan et al. "BatteryLife: A Comprehensive Dataset and Benchmark
        for Battery Life Prediction" arXiv:2502.18807
        """,
        license="Various (see individual datasets)",
        size_mb=5000,
        requires_extraction=True,
        preprocessing_notes="""
        Pre-standardized format. Python package available.
        pip install batterylife
        """
    ),
    
    # -------------------------------------------------------------------------
    # Battery Data Alliance (LF Energy)
    # -------------------------------------------------------------------------
    "bda": DatasetInfo(
        name="bda",
        full_name="Battery Data Alliance (LF Energy)",
        description="""
        High-quality cycling data from Battery Data Alliance.
        
        Features:
        - 199 coin cell batteries
        - NMC//graphite and LFP//graphite chemistries
        - 1,000 cycles per cell
        - Automated, precisely controlled workflows
        """,
        url="https://batterydataalliance.energy/",
        documentation_url="https://batterydataalliance.energy/",
        format=DataFormat.PARQUET,  # Battery Data Format (BDF)
        num_cells=199,
        chemistries=["NMC", "LFP"],
        temperatures=[25.0],
        c_rates=[1.0],
        profiles=["constant_current"],
        citation="LF Energy Battery Data Alliance",
        license="Open Source",
        size_mb=1000,
        requires_extraction=True,
        preprocessing_notes="Uses Battery Data Format (BDF). Python SDK available."
    ),
    
    # -------------------------------------------------------------------------
    # PulseBat (Second-Life Batteries)
    # -------------------------------------------------------------------------
    "pulsebat": DatasetInfo(
        name="pulsebat",
        full_name="PulseBat Second-Life Battery Dataset",
        description="""
        Tests on 464 retired lithium-ion batteries.
        
        Covers:
        - 3 cathode material types
        - 6 historical usage patterns
        - 3 physical formats
        - 6 capacity designs
        
        Pulse test experiments with voltage response and temperature signals.
        Valuable for second-life battery diagnostics.
        """,
        url="https://arxiv.org/abs/2502.16848",
        documentation_url="https://arxiv.org/abs/2502.16848",
        format=DataFormat.CSV,
        num_cells=464,
        chemistries=["NMC", "LFP", "NCA"],
        temperatures=[25.0],
        c_rates=[1.0, 2.0],
        profiles=["pulse", "constant_current"],
        citation="PulseBat Dataset, arXiv:2502.16848",
        license="Research Use",
        size_mb=300,
        requires_extraction=True,
        preprocessing_notes="Pulse response data requires special processing."
    ),
}


# =============================================================================
# Registry Functions
# =============================================================================

def get_dataset_info(name: str) -> DatasetInfo:
    """Get information about a specific dataset."""
    name = name.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]


def list_datasets() -> List[str]:
    """List all available datasets."""
    return list(DATASET_REGISTRY.keys())


def list_datasets_by_feature(
    has_temperature: bool = False,
    has_chemistry: bool = False,
    has_profiles: bool = False,
    has_fast_charging: bool = False
) -> List[str]:
    """Filter datasets by features."""
    results = []
    
    for name, info in DATASET_REGISTRY.items():
        include = True
        
        if has_temperature and len(info.temperatures) <= 1:
            include = False
        if has_chemistry and len(info.chemistries) <= 1:
            include = False
        if has_profiles and len(info.profiles) <= 1:
            include = False
        if has_fast_charging and not any(c > 2.0 for c in info.c_rates):
            include = False
        
        if include:
            results.append(name)
    
    return results


def get_all_chemistries() -> List[str]:
    """Get all unique chemistries across all datasets."""
    chemistries = set()
    for info in DATASET_REGISTRY.values():
        chemistries.update(info.chemistries)
    return sorted(chemistries)


def get_all_temperatures() -> List[float]:
    """Get all unique temperatures across all datasets."""
    temps = set()
    for info in DATASET_REGISTRY.values():
        temps.update(info.temperatures)
    return sorted(temps)


def get_all_profiles() -> List[str]:
    """Get all unique usage profiles across all datasets."""
    profiles = set()
    for info in DATASET_REGISTRY.values():
        profiles.update(info.profiles)
    return sorted(profiles)


def print_registry_summary():
    """Print a summary of all datasets in the registry."""
    print("="*80)
    print("BATTERY DATASET REGISTRY SUMMARY")
    print("="*80)
    
    total_cells = 0
    for name, info in DATASET_REGISTRY.items():
        total_cells += info.num_cells
        print(f"\n{name.upper()} - {info.full_name}")
        print(f"  Cells: {info.num_cells}")
        print(f"  Chemistries: {info.chemistries}")
        print(f"  Temperatures: {info.temperatures}")
        print(f"  C-rates: {info.c_rates}")
        print(f"  Profiles: {info.profiles}")
        print(f"  URL: {info.url}")
    
    print("\n" + "="*80)
    print(f"TOTAL: {len(DATASET_REGISTRY)} datasets, {total_cells} cells")
    print(f"Unique chemistries: {get_all_chemistries()}")
    print(f"Temperature range: {min(get_all_temperatures())}°C to {max(get_all_temperatures())}°C")
    print("="*80)


# =============================================================================
# Download Helpers
# =============================================================================

def get_download_instructions(name: str) -> str:
    """Get download instructions for a dataset."""
    info = get_dataset_info(name)
    
    instructions = f"""
{'='*60}
DOWNLOAD INSTRUCTIONS: {info.full_name}
{'='*60}

URL: {info.url}
Documentation: {info.documentation_url}

Format: {info.format.value}
Size: ~{info.size_mb} MB
Requires Extraction: {info.requires_extraction}

PREPROCESSING NOTES:
{info.preprocessing_notes}

CITATION:
{info.citation}

LICENSE:
{info.license}
{'='*60}
"""
    return instructions


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print_registry_summary()
    
    print("\n\nDATASETS WITH TEMPERATURE VARIATION:")
    temp_datasets = list_datasets_by_feature(has_temperature=True)
    for name in temp_datasets:
        info = get_dataset_info(name)
        print(f"  {name}: {info.temperatures}")
    
    print("\n\nDATASETS WITH MULTIPLE CHEMISTRIES:")
    chem_datasets = list_datasets_by_feature(has_chemistry=True)
    for name in chem_datasets:
        info = get_dataset_info(name)
        print(f"  {name}: {info.chemistries}")
    
    print("\n\nDATASETS WITH FAST CHARGING (>2C):")
    fc_datasets = list_datasets_by_feature(has_fast_charging=True)
    for name in fc_datasets:
        info = get_dataset_info(name)
        print(f"  {name}: C-rates = {info.c_rates}")
    
    print("\n\nDOWNLOAD INSTRUCTIONS FOR SANDIA:")
    print(get_download_instructions("sandia"))

