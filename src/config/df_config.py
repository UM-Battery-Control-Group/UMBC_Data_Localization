CYCLE_ID_LIMS= {
    'RPT': {'V_max_cycle':4.1, 'V_min_cycle':3.8, 'dt_min': 600, 'dAh_min':0.1},
    'CYC': {'V_max_cycle':3.8, 'V_min_cycle':3.8, 'dt_min': 600, 'dAh_min':0.1},
    'Test11': {'V_max_cycle':3.6, 'V_min_cycle':3.6, 'dt_min': 600, 'dAh_min':0.1},
    'EIS': {'V_max_cycle':4.1, 'V_min_cycle':3.8, 'dt_min': 600, 'dAh_min':0.5}, # same as RPT, but says EIS in the filenames for some GM cells
    'CAL': {'V_max_cycle':3.8, 'V_min_cycle':3.8, 'dt_min': 600, 'dAh_min':0.5},
    '_F': {'V_max_cycle':3.8, 'V_min_cycle':3.8, 'dt_min': 3600, 'dAh_min':0.5} # Formation files handled via peak finding
    
}
DEFAULT_TRACE_KEYS = ['h_datapoint_time', 'h_test_time', 'h_current', 'h_potential', 'c_cumulative_capacity', 'h_charge_capacity','h_discharge_capacity','h_step_ord',
                    'aux_neware_xls_t1_none_0', 'h_step_index','h_cycle']
DEFAULT_DF_LABELS = ['Time [ms]', 'Test Time [ms]', 'Current [A]', 'Voltage [V]', 'Ah throughput [A.h]', 'Charge Ah throughput [A.h]','Discharge Ah throughput [A.h]','Step ord',
                    'Temperature [degC]', 'Step index','Cycle index']
TIME_COLUMNS = ['aux_vdf_timestamp_datetime_0', 'aux_vdf_timestamp_epoch_0', 'h_datapoint_time']