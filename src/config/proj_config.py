# # MAX_PULSES = 11
# I_threshold for identifying C/20 dis/charge for eSOH processing in _load_V_data

GMJULY2022_PULSE_CURRENTS = [3.0, 1.5, -3.0, -1.5, -0.5]
GMFEB23_PULSE_CURRENTS = [2.0, 1.0, 0.5, -2.0, -1.0, -0.5]
UMBL2022FEB_PULSE_CURRENTS = [2.0, 1.0, 0.5, -2.0, -1.0, -0.5]
DEFAULT_PULSE_CURRENTS = [2.0, 1.0, -2.0, -1.0, -0.5]
Qmax=3.8

PROJECT = {
    'DEFAULT':{
        'pulse_currents':DEFAULT_PULSE_CURRENTS,
        'nominal_capacity':3.5, #A.h
        'Qmax': 3.8,
        'I_C20': 0.177,
        'I_threshold':0.005,
    },
    'GMJULY2022':{
        'pulse_currents':GMJULY2022_PULSE_CURRENTS,
        'nominal_capacity':3.5, #A.h
        'Qmax': 3.8,
        'I_C20': 0.177,
        'I_threshold':0.005,
    },
    'UNKNOWN_PROJECT':{
        'pulse_currents':GMJULY2022_PULSE_CURRENTS,
        'nominal_capacity':3.5, #A.h
        'Qmax': 3.8,
        'I_C20': 0.177,
        'I_threshold':0.005,
    },
    'GMFEB23':{
        'pulse_currents':GMFEB23_PULSE_CURRENTS,
        'nominal_capacity':3.5, #A.h
        'Qmax': 3.8,
        'I_C20': 0.177,
        'I_threshold':0.005,
    },
    'UMBL2022FEB':{
        'pulse_currents':UMBL2022FEB_PULSE_CURRENTS,
        'nominal_capacity':2.5, #A.h
        'Qmax': 3.0, #first charge during formation ~2.8 A.h
        'I_C20': 0.125,
        'I_C10': 0.25, # formation
        'I_threshold':0.026,
    }
}