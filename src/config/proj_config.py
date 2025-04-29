# # MAX_PULSES = 11
# I_threshold for identifying C/20 dis/charge for eSOH processing in _load_V_data
import pandas as pd

GMJULY2022_PULSE_CURRENTS = [3.0, 1.5, -3.0, -1.5, -0.5]
GMFEB23_PULSE_CURRENTS = [2.0, 1.0, 0.5, -2.0, -1.0, -0.5]
UMBL2022FEB_PULSE_CURRENTS = [2.0, 1.0, 0.5, -2.0, -1.0, -0.5]
DEFAULT_PULSE_CURRENTS = [2.0, 1.0, -2.0, -1.0, -0.5]
Qmax=3.8

# read in Un and Up data for interpolation. Downsampling doesn't save execution time
fn = r'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\pybamm\input\parameters\lithium_ion\negative_electrodes\graphite_UMBL_Tran2025\graphite_ocp_Siegel.csv' 
graphite_data = pd.read_csv(fn, comment='#', header = None, names = ["x", "U"])#.iloc[0::10]
fn = r'C:\Users\Vivian\Dropbox (University of Michigan)\from_box\Research\PyBaMM\PyBaMM\pybamm\input\parameters\lithium_ion\positive_electrodes\NMC_UMBL_Tran2025\NMC_ocp_Siegel.csv' 
NMC_data = pd.read_csv(fn, comment='#', header = None, names = ["y", "U"])#.iloc[0::10]


PROJECT = {
    'DEFAULT':{
        'pulse_currents':DEFAULT_PULSE_CURRENTS,
        'nominal_capacity':3.5, #A.h
        'Qmax': 3.8,
        'Qmin': 0.2,
        'I_C20': 0.177,
        'I_threshold':0.005,
        'x_peaks_nom': [0.17753721, 0.51569198],
        'lb_esoh': [1, 0, 1, 0,0,0], #[Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        'ub_esoh': [5, 1, 6.5, 1, 0.5, 0], 
        'W': [1,1,1], # weight for _loss_func with loss = W[0]*RMSE_V_norm + W[1]*dVdQ_loss_norm
        'Un_data': graphite_data,
        'Up_data': NMC_data,
        'min_peak_prominence': 0.02,
        'V_min': 3.0,
        'V_max': 4.2,
    },
    'GMJULY2022':{
        'pulse_currents':GMJULY2022_PULSE_CURRENTS,
        'nominal_capacity':3.5, #A.h
        'Qmax': 3.8,
        'Qmin': 0.2,
        'I_C20': 0.177,
        'I_C10': 0.177,
        'I_threshold':0.005,
        'x_peaks_nom': [0.17753721, 0.51569198],
        'lb_esoh': [1, 0, 1, 0,0,0], #[Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        'ub_esoh': [5, 1, 6.5, 1, 0.5, 0], 
        'W': [1,1,1], # weight for _loss_func with loss = W[0]*RMSE_V_norm + W[1]*dVdQ_loss_norm
        'Un_data': graphite_data,
        'Up_data': NMC_data,
        'min_peak_prominence': 0.02,
        'V_min': 3.0,
        'V_max': 4.2,
    },
    'UNKNOWN_PROJECT':{
        'pulse_currents':GMJULY2022_PULSE_CURRENTS,
        'nominal_capacity':3.5, #A.h
        'Qmax': 3.8,
        'Qmin': 0.2,
        'I_C20': 0.177,
        'I_threshold':0.005,
        'x_peaks_nom': [0.17753721, 0.51569198],
        'lb_esoh': [1, 0, 1, 0,0,0], #[Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        'ub_esoh': [5, 1, 6.5, 1, 0.5, 0], 
        'W': [1,1,1], # weight for _loss_func with loss = W[0]*RMSE_V_norm + W[1]*dVdQ_loss_norm
        'Un_data': graphite_data,
        'Up_data': NMC_data,
        'min_peak_prominence': 0.02,
        'V_min': 3.0,
        'V_max': 4.2,
    },
    'GMFEB23':{
        'pulse_currents':GMFEB23_PULSE_CURRENTS,
        'nominal_capacity':3.5, #A.h
        'Qmax': 3.8,
        'Qmin': 0.2,
        'I_C20': 0.177,
        'I_threshold':0.005,
        'x_peaks_nom': [0.17753721, 0.51569198],
        'lb_esoh': [1, 0, 1, 0,0,0], #[Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        'ub_esoh': [5, 1, 6.5, 1, 0.5, 0], 
        'W': [1,1,1], # weight for _loss_func with loss = W[0]*RMSE_V_norm + W[1]*dVdQ_loss_norm
        'Un_data': graphite_data,
        'Up_data': NMC_data,
        'min_peak_prominence': 0.02,
        'V_min': 3.0,
        'V_max': 4.2,
    },
    'UMBL2022FEB':{
        'pulse_currents':UMBL2022FEB_PULSE_CURRENTS,
        'nominal_capacity':2.5, #A.h
        'Qmax': 3.05, #first charge during formation ~2.8 A.h
        'Qmin': 0.2,
        'I_C20': 0.125,
        'I_C10': 0.25, # formation
        'I_threshold':0.026,
        'x_peaks_nom': [0.17753721, 0.51569198], # x location for low SOC and high SOC peaks of interest based on 3E Un data to init Cn and x0 for eSOH
        'lb_esoh': [1, 0, 1, 0,0,0], #[Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        'ub_esoh': [5, 1, 5, 1, 1, 0], 
        'W': [1,1,0], # weight for _loss_func with loss = W[0]*RMSE_V_norm + W[1]*dVdQ_loss_norm + W[2]*loss_peak
        'Un_data': graphite_data,
        'Up_data': NMC_data,
        'min_peak_prominence': 0.02,
        'default_fit_params': [3,0,4, 1, 0.01, 0],  #[Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        'V_min': 2.8,
        'V_max': 4.2,
        'dt_pulse_max':35, # max hppc pulse duration expected, use 35s to include formation, <13s for others   
    }
}