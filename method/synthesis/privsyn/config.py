# path related constant
import os
import tempfile

_BASE_TEMP_DIR = os.environ.get(
    "PRIVSYN_DATA_ROOT",
    os.path.join(tempfile.gettempdir(), "privsyn_temp_data"),
)

RAW_DATA_PATH = os.environ.get("PRIVSYN_RAW_ROOT", "data")
PROCESSED_DATA_PATH = os.path.join(_BASE_TEMP_DIR, "processed_data")
SYNTHESIZED_RECORDS_PATH = os.path.join(_BASE_TEMP_DIR, "synthesized_records")
MARGINAL_PATH = os.path.join(_BASE_TEMP_DIR, "marginal")
DEPENDENCY_PATH = os.path.join(_BASE_TEMP_DIR, "dependency")
EXPERIMENT_BASE_PATH = os.environ.get(
    "PRIVSYN_EXP_ROOT",
    os.path.join(_BASE_TEMP_DIR, "exp"),
)

ALL_PATH = [
    PROCESSED_DATA_PATH,
    SYNTHESIZED_RECORDS_PATH,
    MARGINAL_PATH,
    DEPENDENCY_PATH,
]

# config file path
TYPE_CONIFG_PATH = os.path.join(_BASE_TEMP_DIR, "fields")
