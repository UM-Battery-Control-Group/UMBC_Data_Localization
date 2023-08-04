from src.model.DataManager import DataManager

from src.logger_config import setup_logger

class Presenter:
    def __init__(self, dataManager: DataManager):
        self.dataManager = dataManager
        self.logger = setup_logger()

    def get_measured_data_time(self, cell_name, plot_cycles = True):
        """
        Get measured data from the local disk

        Parameters
        ----------
        cell_name: str
            The cell name of the data to be found
        plot_cycles: bool, optional
            Whether to plot the cycles
        
        Returns
        -------
        dict
            The dictionary of measured data
        """
        #TODO: modify process_cell
        cell_data, cell_data_vdf, cell_cycle_metrics = self.dataManager.process_cell(cell_name)
        # setup timeseries data
        t = cell_data['Time [s]']
        I = cell_data['Current [A]']
        V = cell_data['Voltage [V]'] 
        T = cell_data['Temperature [degC]'] 
        AhT = cell_data['Ah throughput [A.h]']
        t_vdf = cell_data_vdf['Time [s]']
        exp_vdf = cell_data_vdf['Expansion [-]']
        T_vdf = cell_data_vdf['Temperature [degC]']
        # T_amb = cell_data_vdf['Amb Temp [degC]']    
        # setup cycle metrics
        t_cycle = cell_cycle_metrics['Time [s]'] 
        Q_c = cell_cycle_metrics['Charge capacity [A.h]'] 
        Q_d = cell_cycle_metrics['Discharge capacity [A.h]'] 

        cycle_idx = []
        capacity_check_idx = []
        cycle_idx_vdf = []
        capacity_check_in_cycle_idx = []
        charge_idx = []

        if plot_cycles:
            cycle_idx = cell_data.cycle_indicator[cell_data.cycle_indicator].index     # indices in cell_data to check cycle alignment
            capacity_check_idx = cell_data.capacity_check_indicator[cell_data.capacity_check_indicator].index 
            cycle_idx_vdf = cell_data_vdf.cycle_indicator[cell_data_vdf.cycle_indicator].index
            capacity_check_in_cycle_idx = cell_cycle_metrics[cell_cycle_metrics.capacity_check_indicator].index
            charge_idx = cell_data.charge_cycle_indicator[cell_data.charge_cycle_indicator].index
            step_idx = cell_data['Step index']

        return {
            't': t,
            'I': I,
            'V': V,
            'T': T,
            'AhT': AhT,
            't_vdf': t_vdf,
            'exp_vdf': exp_vdf,
            'T_vdf': T_vdf,
            't_cycle': t_cycle,
            'Q_c': Q_c,
            'Q_d': Q_d,
            'cycle_idx': cycle_idx,
            'capacity_check_idx': capacity_check_idx,
            'cycle_idx_vdf': cycle_idx_vdf,
            'capacity_check_in_cycle_idx': capacity_check_in_cycle_idx,
            'charge_idx': charge_idx,
        }
    