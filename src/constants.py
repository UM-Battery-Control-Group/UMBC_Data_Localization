import os
from datetime import timedelta

DATE_FORMAT = '%Y-%m-%d_%H-%M-%S'
ROOT_PATH = os.path.join(os.path.dirname(os.getcwd()), 'voltaiq_data')
JSON_FILE_PATH = os.path.join(ROOT_PATH, 'directory_structure.json')
TIME_TOLERANCE = timedelta(hours=2)
CYCLE_ID_LIMS= {
    'RPT': {'V_max_cycle':4.1, 'V_min_cycle':3.8, 'dt_min': 600, 'dAh_min':0.1},
    'CYC': {'V_max_cycle':3.8, 'V_min_cycle':3.8, 'dt_min': 600, 'dAh_min':0.1},
    'Test11': {'V_max_cycle':3.6, 'V_min_cycle':3.6, 'dt_min': 600, 'dAh_min':0.1},
    'EIS': {'V_max_cycle':4.1, 'V_min_cycle':3.8, 'dt_min': 600, 'dAh_min':0.5}, # same as RPT, but says EIS in the filenames for some GM cells
    'CAL': {'V_max_cycle':3.8, 'V_min_cycle':3.8, 'dt_min': 600, 'dAh_min':0.5},
    '_F': {'V_max_cycle':3.8, 'V_min_cycle':3.8, 'dt_min': 3600, 'dAh_min':0.5} # Formation files handled via peak finding
}