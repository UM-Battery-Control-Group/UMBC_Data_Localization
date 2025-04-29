import pandas as pd 
import numpy as np
import time
from scipy import integrate, interpolate
from scipy.signal import find_peaks, medfilt, savgol_filter, butter, sosfiltfilt,peak_widths
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from scipy.stats import norm
from itertools import compress
import ruptures as rpt
import matplotlib.pyplot as plt
import rfcnt
from src.model.DirStructure import DirStructure
from src.model.DataFilter import DataFilter
from src.utils.Logger import setup_logger
from src.utils.DateConverter import DateConverter
from src.config.df_config import CYCLE_ID_LIMS, DEFAULT_TRACE_KEYS, DEFAULT_DF_LABELS
from src.config.calibration_config import X1, X2, C
# from src.config.esoh_config import W1, W2, W3, UN_VAR1, UN_VAR2, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10
from src.config.proj_config import PROJECT
# from pymoo.algorithms.soo.nonconvex.pattern import PatternSearch
# from pymoo.optimize import minimize as pmminimize
# from pymoo.core.problem import Problem

class DataProcessor:
    """
    The class to process the data

    Attributes
    ----------
    dataIO: DataIO object
        The object to save and load data
    dataFilter: DataFilter object
        The object to filter data from the local disk
    dirStructure: DirStructure object
        The object to manage the directory structure for the local data
    logger: logger object
        The object to log information
    
    
    Methods
    -------
    process_cell(records_cycler, records_vdf, cell_cycle_metrics=None, cell_data=None, cell_data_vdf=None, numFiles=1000, cycle_id_lims=CYCLE_ID_LIMS)
        Using the test records to process and update the cell data, cycle metrics, and expansion data
    sort_records(records, start_time=None, end_time=None)
        Sort the records by start time from low to high
    summarize_rpt_data(cell_data, cell_data_vdf, cell_cycle_metrics, project_name)
        Get the summary data for each RPT file
    """
    def __init__(self, dataFilter: DataFilter, dirStructure: DirStructure, dateConverter: DateConverter):
        self.dataFilter = dataFilter
        self.dirStructure = dirStructure
        self.dateConverter = dateConverter
        self.last_AHT=0
        self.logger = setup_logger()


    def process_cell(self, project_name, cell_cycle_metrics=None, cell_data=None, cell_data_vdf=None, calibration_parameters=None, numFiles=1000, cycle_id_lims=CYCLE_ID_LIMS):
        """
        Using the test records to process and update the cell data, cycle metrics, and expansion data

        Parameters 
        ----------
        records_cycler: list of dict
            The list of records of the cycler data
        records_vdf: list of dict
            The list of records of the expansion data
        project_name: str
            The project name of the cell            
        cell_cycle_metrics: DataFrame, optional
            The old dataframe of the cell cycle metrics
        cell_data: DataFrame, optional
            The old dataframe of the cell data
        cell_data_vdf: DataFrame, optional
            The old dataframe of the cell data vdf
        numFiles: int, optional
            The max number of files to be processed
        cycle_id_lims: list of int, optional
            The cycle id limits for the cycle metrics

        Returns
        -------
        DataFrame
            The updated dataframe of the cell cycle metrics
        DataFrame
            The updated dataframe of the cell data
        DataFrame
            The updated dataframe of the cell data vdf
        bool
            Whether the data is updated during the process
        """
        # Process the cycling data
        cell_data, cell_cycle_metrics = self._process_cycler_data(cell_data = cell_data, cycle_id_lims=cycle_id_lims, project_name= project_name)
        
        # else: # if pickle file doesn't exist or load_pickle is False, (re)process all expansion data
        self.logger.info(f"Processing vdf data")
        # records_new_data_vdf = records_vdf.copy()
        cell_data_vdf, cell_cycle_metrics = self._process_cycler_expansion(cell_data_vdf, cell_cycle_metrics, calibration_parameters, numFiles = numFiles)    

        # self.logger.info(f"Finished processing {len(records_new_data)} new cycler files and {len(records_new_data_vdf)} new vdf files")
        # rearrange columns of cell_cycle_metrics for easy reading with data on left and others on right
        cols = cell_cycle_metrics.columns.to_list()
        move_idx = [c for c in cols if '[' in c] + [c for c in cols if '[' not in c] # Columns with data include '[' in the key
        cell_cycle_metrics = cell_cycle_metrics[move_idx]

        # if there is new data, save it to pickle files
        update = True #len(records_new_data)>0 or len(records_new_data_vdf)>0
        return cell_cycle_metrics, cell_data, cell_data_vdf, update
    
    def sort_records(self, records, start_time=None, end_time=None):
        """
        Sort the records by start time from low to high

        Parameters
        ----------
        records: list of dict
            The list of records to be sorted
        start_time: str, optional
            The start time of the records in the format '%Y-%m-%d_%H-%M-%S'
        end_time: str, optional
            The end time of the records in the format '%Y-%m-%d_%H-%M-%S'
        
        Returns
        -------
        list of dict
            The list of records sorted by start time from low to high
        """
        
        # Filter and sort based on the string representation
        filtered_sorted_records = sorted(
            [record for record in records if 
                (start_time is None or record['start_time'] >= start_time) and 
                (end_time is None or record['start_time'] <= end_time)
            ], 
            key=lambda x: x['start_time']
        )
        return filtered_sorted_records
    
    def _filter_records_new_data(self, cell_cycle_metrics, records, last_cycle_time=None):
        """
        Get the list of test records that have not been processed

        Parameters
        ----------
        cell_cycle_metrics: DataFrame
            The dataframe of the cell cycle metrics
        records: list of dict
            The list of records to be filtered
        last_cycle_time: float, optional
            The timestamp of the last cycle

        Returns
        -------
        list of TestRecord objects
            The list of test records that have not been processed
        """
        recorded_cycle_times = cell_cycle_metrics['Time [ms]']
        last_recorded_cycle_time = recorded_cycle_times.iloc[-1] if not recorded_cycle_times.empty else 0
        records_new_data = []
        # for each file, check that cell_cycle_metrics has timestamps in this range
        for record in records:
            cycle_end_times = self.dataFilter.filter_cycle_end_times(record)
            last_cycle_time_in_file = cycle_end_times.iloc[-1] if not last_cycle_time else last_cycle_time
            # if last_cycle_time_in_file is not int, replace it with last_recorded_cycle_time. It could be np.nan
            if not isinstance(last_cycle_time_in_file, (int, np.integer)):
                self.logger.warning(f"last_cycle_time_in_file is not int, replace it with last_recorded_cycle_time: {last_recorded_cycle_time}")
                last_cycle_time_in_file = last_recorded_cycle_time
            record_start_time = self.dateConverter._str_to_timestamp(record['start_time'])
            if len(cycle_end_times) > 1:
                timestamps_in_range_count = sum(1 for t in recorded_cycle_times if record_start_time <= t <= last_cycle_time_in_file)
                if timestamps_in_range_count == 0:
                    records_new_data.append(record)     
        return records_new_data
    
    def _update_dataframe(self, df, df_new, file_start_time, file_end_time, update_AhT=True):
        """
        Update the dataframe with the new test data, and update the Ah throughput.

        Parameters
        ----------
        df: DataFrame
            The dataframe to be updated
        df_new: DataFrame
            The dataframe of the new test data
        file_start_time: float
            The start time of the new test data, get from df['Time [ms]'].iloc[0]
        file_end_time: float
            The end time of the new test data, get from df['Time [ms]'].iloc[-1]
        update_AhT: bool, optional
            Whether to update the Ah throughput

        Returns
        -------
        DataFrame
            The updated dataframe
        """

        # Find overlapping data
        file_drop_idx = df[(df['Time [ms]'] >= file_start_time) & (df['Time [ms]'] <= file_end_time)].index

        # Remove overlapping data
        df = df.drop(file_drop_idx)

        # Split old dataframe into before and after sections based on new data
        if len(file_drop_idx) > 0:
            df_before_test = df[df['Time [ms]'] < file_start_time]
            df_after_test = df[df['Time [ms]'] > file_end_time]

            # If Ah throughput update is needed and the field exists in both dataframes
            if update_AhT and 'Ah throughput [A.h]' in df.columns and 'Ah throughput [A.h]' in df_new.columns:
                last_AhT_before_test = df_before_test['Ah throughput [A.h]'].iloc[-1] if not df_before_test.empty else 0
                df_new.loc[:, 'Ah throughput [A.h]'] += last_AhT_before_test
                last_AhT_from_test = df_new['Ah throughput [A.h]'].iloc[-1] if not df_new.empty else 0
                df_after_test.loc[:, 'Ah throughput [A.h]'] += last_AhT_from_test

            df = pd.concat([df_before_test, df_new, df_after_test])

        # If no overlap, simply append the data (This could be modified based on exact use case)
        else:
            if update_AhT and 'Ah throughput [A.h]' in df.columns and 'Ah throughput [A.h]' in df_new.columns:
                last_AhT_before_test = df['Ah throughput [A.h]'].iloc[-1]
                df_new['Ah throughput [A.h]'] += last_AhT_before_test
            df = pd.concat([df, df_new])

        df.reset_index(drop=True, inplace=True)

        return df    

    
    def summarize_rpt_data_2(self, cell_cycle_metrics, project_name, cell_rpt_data = None, cell_data = None, cell_data_vdf = None):
        """
        Identify files with RPTs, process RPT metrics, and save the summary data for each RPT file

        Parameters
        ----------
        cell_data: DataFrame
            The dataframe of the cell data
        cell_data_vdf: DataFrame
            The dataframe of the cell data vdf
        cell_cycle_metrics: DataFrame
            The dataframe of the cell cycle metrics
        project_name: str
            The project name for the cell
        
        Returns
        -------
        cell_rpt_data: DataFrame
            The dataframe of the summary data for each RPT file
        """
        # get rpt file names. maintain chronological order
        rpt_idx = [i for (i, test) in zip(cell_cycle_metrics.index,cell_cycle_metrics['Test type']) if test in ['RPT','_F','_Cy100','_Cby100']]
        # rpt_filenames0 = cell_cycle_metrics.loc[rpt_idx, 'Test name']         # (cell_cycle_metrics['Test type'] == 'RPT') | (cell_cycle_metrics['Test type'] == '_F')| (cell_cycle_metrics['Test type'] == '_Cy100')| (cell_cycle_metrics['Test type'] == '_Cby100')
        # rpt_filenames = [list(set(rpt_filenames0))[i] for i, x in sorted(enumerate(set(rpt_filenames0)), key=lambda x: x[1])]
       
        rpt_protocols = ['HPPC','C/20 Charge', 'C/20 Discharge'] # process HPPC first to have Rs for IR correction
        # RPT number based on number of HPPCs
        cell_cycle_metrics['RPT #'] = list(np.cumsum(cell_cycle_metrics['Protocol']=='HPPC')) # change to C/20 charge to include formation?

        # create new df if cell_rpt_data doesn't exist
        if cell_rpt_data is None: 
            cell_rpt_data = cell_cycle_metrics[cell_cycle_metrics['Protocol'].isin(rpt_protocols)].copy()# & ([ p in rpt_protocols for p in cell_cycle_metrics['Protocol']])]
            cell_rpt_data['RPT #'] = cell_cycle_metrics.loc[cell_cycle_metrics['Protocol'].isin(rpt_protocols), 'RPT #']
        else: 
            cell_rpt_data = cell_rpt_data[cell_rpt_data['Protocol'].isin(rpt_protocols)].copy()
            cell_rpt_data['RPT #'] = list(np.cumsum(cell_rpt_data['Protocol']=='HPPC')) # previously processed data lists filenames instead
        
        # Determine the pulse currents based on project name
        if project_name in PROJECT.keys(): 
            project_settings = PROJECT[project_name]
        else:
            project_settings = PROJECT['DEFAULT']
        Q_nom = project_settings['nominal_capacity']
        pulse_currents = project_settings['pulse_currents']
        dt_pulse_max = project_settings['dt_pulse_max']
        I_C20 = project_settings['I_C20']
        I_C10 = project_settings['I_C10']
        V_min_data = project_settings['V_min']
        V_max_data = project_settings['V_max']
        dI_threshold = project_settings['I_threshold']
        x_peaks_nom = project_settings['x_peaks_nom']
        Un_data = project_settings['Un_data'] #savgol_filter(project_settings['Un_data'],window_length,3,0) 
        Up_data = project_settings['Up_data']
        min_peak_prominence = project_settings['min_peak_prominence']

        # predefine models and df columns
        model_sig_dv = {
            'name': '',
            'lb': project_settings['lb_esoh'],
            'ub': project_settings['ub_esoh'],
            'W': project_settings['W'],
            'x_esoh_fit': np.array(project_settings['default_fit_params']),#[Cn_hat, x0_hat, Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat],
            'dz_n': 0.1, 
            'dz_p': 1,
            'RMSE_V':1,
            'capacity_error': 5, 
                    }                    
        # fit with sigma, no dVdQ weight
        model_sig = {
            'name': '_sig',
            'lb': project_settings['lb_esoh'],
            'ub': project_settings['ub_esoh'],
            'W': [1,0] + project_settings['W'][-1::],
            'x_esoh_fit': np.array(project_settings['default_fit_params']),# [Cn_hat, x0_hat, Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat], 
            'dz_n': 0.1, 
            'dz_p': 1,
            'RMSE_V':1, 
            'capacity_error': 5,
                    }
        # fit with dVdQ weight, no sigma 
        model_dv = {
            'name': '_dv',
            'lb': project_settings['lb_esoh'][0:-2] + [0,0],
            'ub': project_settings['ub_esoh'][0:-2] + [0,0],
            'W': project_settings['W'],
            'x_esoh_fit': np.array(project_settings['default_fit_params']),#[Cn_hat, x0_hat, Cp_hat, y0_hat, 0,0],
            'dz_n': 1, 
            'dz_p': 1,
            'RMSE_V':1, 
            'capacity_error': 5,
                    }
        # fit with voltage fitting 
        model_vf = {
            'name': '_vf',
            'lb': project_settings['lb_esoh'][0:-2] + [0,0],
            'ub': project_settings['ub_esoh'][0:-2] + [0,0],
            'W': [1,0] + project_settings['W'][-1::],
            'x_esoh_fit': np.array(project_settings['default_fit_params']),#[Cn_hat, x0_hat, Cp_hat, y0_hat, 0,0],
            'dz_n': 1, 
            'dz_p': 1,
            'RMSE_V':1, 
            'capacity_error': 5,
                    }
        model_sig_dv_noIR = {
            'name': '_sig_dv_noIR',
            'lb': project_settings['lb_esoh'],
            'ub': project_settings['ub_esoh'],
            'W': project_settings['W'],
            'x_esoh_fit': np.array(project_settings['default_fit_params']),#[Cn_hat, x0_hat, Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat],
            'dz_n': 0.1, 
            'dz_p': 1,
            'RMSE_V':1, 
            'capacity_error': 5,
                    }   
        model_sig_dv_constantIR = {
            'name': '_sig_dv_constantIR',
            'lb': project_settings['lb_esoh'],
            'ub': project_settings['ub_esoh'],
            'W': project_settings['W'],
            'x_esoh_fit': np.array(project_settings['default_fit_params']),#[Cn_hat, x0_hat, Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat],
            'dz_n': 0.1, 
            'dz_p': 1,
            'RMSE_V':1, 
            'capacity_error': 5,
                    }      
        model_sig_dv_avgV = {
            'name': '_avgV',
            'lb': project_settings['lb_esoh'],
            'ub': project_settings['ub_esoh'],
            'W': project_settings['W'],
            'x_esoh_fit': np.array(project_settings['default_fit_params']),#[Cn_hat, x0_hat, Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat],
            'dz_n': 0.1, 
            'dz_p': 1,
            'RMSE_V':1, 
            'capacity_error': 5,
                    }       
        models = [model_sig_dv, model_dv, model_vf, model_sig_dv_noIR] #model_sig_dv_noIR, model_sig_dv_avgV] model_sig_dv_constantIR

        # make empty columns for each model 
        new_columns = ['RPT #', "C_voltage_win"] 
        for model in models:
            for col in ["C", "Cn","x0","x100", "sigma_Cn", "Cp", "y0", "y100", "sigma_Cp", "RMSE_V", "RMSE_dVdQ", "RMSE_V_full", "RMSE_dVdQ_full", "p1_err", "p2_err", 'n_Li', 'LLI', 'LAM_NE', 'LAM_PE', 'capacity_error']:
                new_columns.append(''.join(col + model['name']))
        # for df in [cell_cycle_metrics, cell_rpt_data]:
        cell_cycle_metrics = pd.concat([cell_cycle_metrics, pd.DataFrame(columns=[col for col in new_columns if col not in cell_cycle_metrics.keys()], dtype='object')])
        cell_rpt_data = pd.concat([cell_rpt_data, pd.DataFrame(columns=[col for col in new_columns if col not in cell_rpt_data.keys()], dtype='object')])
            # if ''.join(col + model['name']) not in df.columns:
                # df[''.join(col + model['name'])] = pd.Series(dtype='object')


        # make empty columns
        hppc_columns = ["Q_pulse","dt_pulse","I_pulse","Rs","Rlong"]
        cell_cycle_metrics = pd.concat([cell_cycle_metrics, pd.DataFrame(columns=[col for col in hppc_columns if col not in cell_cycle_metrics.keys()], dtype='object')])
        cell_rpt_data = pd.concat([cell_rpt_data, pd.DataFrame(columns=[col for col in hppc_columns if col not in cell_rpt_data.keys()], dtype='object')])
        new_columns = new_columns + hppc_columns
        # for df in [cell_cycle_metrics, cell_rpt_data]:
            # make columns 
            # for col in ["Q_pulse","dt_pulse","I_pulse","Rs","Rlong"]:
            #     if col not in df.columns:
            #         # df[col] = pd.Series(dtype='object')
            #         df = pd.concat([df, pd.DataFrame(columns=[col], dtype='object')])

        # For each RPT file, copy relevant CCM rows and process each subcycle according to the protocol 
        if cell_rpt_data['RPT #'].isnull().all(): 
            
            existing_rpt_nums = list(set(cell_cycle_metrics['RPT #']))
        else: 
            existing_rpt_nums = set(cell_cycle_metrics['RPT #']).intersection(cell_rpt_data['RPT #']) # for some cells, ccm has 1 more than cell_rpt_data

        for rpt_file in existing_rpt_nums : #list(set(cell_cycle_metrics['RPT #'])):
            # PREP ccm_rpt_file
            # copy ccm rows with same test name and initialize columns for data dfs
            ccm_rpt_file = cell_cycle_metrics[(cell_cycle_metrics['RPT #'] == rpt_file) & cell_cycle_metrics['Protocol'].isin(rpt_protocols)].copy()# & ([ p in rpt_protocols for p in cell_cycle_metrics['Protocol']])]
            if 'Test name' not in cell_rpt_data.keys():
                cell_rpt_data['Test name'] = cell_cycle_metrics['Test name']

            # PROCESS SUBCYCLE DATA BY PROTOCOL: Identify and clean timeseries data, then extract metrics to save
            # For each protocol...
            # set timeseries data for cycles if they exist. Else, find it in cell_data and cell_data_vdf
            if ('Data' in cell_rpt_data.keys()) and ('Data vdf' in cell_rpt_data.keys()) and (cell_data is None) and (cell_data_vdf is None): # used 'Data VDF'  for original/Ziyi-processed cell_rpt_data?
                ccm_rpt_file['Data'] = cell_rpt_data.loc[(cell_rpt_data['RPT #'] == rpt_file), 'Data'].values
                ccm_rpt_file['Data vdf'] = cell_rpt_data.loc[(cell_rpt_data['RPT #'] == rpt_file), 'Data vdf'].values
            else:
                ccm_rpt_file['Data'] = pd.Series(dtype='object')
                ccm_rpt_file['Data vdf'] = pd.Series(dtype='object')
                for protocol in  rpt_protocols: 
                    # find indices in ccm of subcycles with the protocol 
                    protocol_idx = ccm_rpt_file[ccm_rpt_file['Protocol']==protocol].index
                    if len(protocol_idx) > 1: # warn if multiple of any protocol is found in one RPT file 
                        self.logger.warning("Multiple " + protocol + " protocols detected.")  
                        # protocol_idx = protocol_idx.iloc[-1]         
                    # For each subcycle of the protocol subcycle, identify data from cell_data and cell_data_vdf and write to ccm. Process HPPC while it's loaded
                    if (cell_data is not None) and (cell_data_vdf is not None): 
                        for p in protocol_idx:
                            # identify subcycle start and end timestamps and get data from cell_data
                            t_start_protocol = ccm_rpt_file.at[p,'Time [ms]']-30
                            try: # end of partial cycle = next time listed
                                t_end_protocol = cell_cycle_metrics.at[p+1,'Time [ms]']+30
                            except: # end of partial cycle = end of all known data
                                t_end_protocol = cell_data['Time [ms]'].iloc[-1]+30
                            if 'Temperature [degC]' in cell_data.keys(): #neware
                                subcycle_data = cell_data[['Time [ms]', 'Current [A]', 'Voltage [V]', 'Ah throughput [A.h]', 'Temperature [degC]', 'Step index']][(cell_data['Time [ms]']>t_start_protocol) & (cell_data['Time [ms]']<t_end_protocol)]
                            else: #arbin
                                subcycle_data = cell_data[['Time [ms]', 'Current [A]', 'Voltage [V]', 'Ah throughput [A.h]', 'Step index']][(cell_data['Time [ms]']>t_start_protocol) & (cell_data['Time [ms]']<t_end_protocol)]
                            ccm_rpt_file.at[p,'Data'] = subcycle_data #pd.concat([ccm_rpt_file.at[p,'Data'],subcycle_data])
                            # if len(subcycle_data)==0: 
                            #     self.logger.warning(f"NO DATA IN THIS CYCLE")
                            
                            # add corresponding vdf data to dictionary if exists
                            t_vdf = cell_data_vdf['Time [ms]']
                            if len(t_vdf)>1: #ignore for constrained cells
                                subcycle_vdf_data = cell_data_vdf[(t_vdf>t_start_protocol) & (t_vdf<t_end_protocol)] 
                                ccm_rpt_file.at[p,'Data vdf'] = subcycle_vdf_data #pd.concat([ccm_rpt_file.at[p,'Data vdf'],subcycle_vdf_data])                 
                # save to cell_rpt_data
                cell_rpt_data.loc[cell_rpt_data['RPT #'] == rpt_file, 'Data']= ccm_rpt_file['Data'].values
                cell_rpt_data.loc[cell_rpt_data['RPT #'] == rpt_file, 'Data vdf']= ccm_rpt_file['Data vdf'].values


            # For HPPC data, calculate resistances
            for p, hppc in ccm_rpt_file[ccm_rpt_file['Protocol']=='HPPC'].iterrows():
                # calculate resistances and add to ccm
                self.update_cycle_metrics_hppc(ccm_rpt_file.loc[p], [cell_cycle_metrics, ccm_rpt_file,], p, pulse_currents)
        
            # After all data is assigned, calculate eSOH parameters from C/20 charge
            # START ESOH UPDATE -----------------
            # Fit esoh params for each model
            for c, charge in ccm_rpt_file[ccm_rpt_file['Protocol']=='C/20 Charge'].iterrows():
                data = charge['Data']
                #check if subcycle time is nonmonotonic
                if (not all(x<=y for x, y in zip(data['Time [ms]'][0:-1], data['Time [ms]'][1::]))):
                    # check if subcycle was restarted
                    if (sum(data['Time [ms]']==data['Time [ms]'].iloc[0])>1): # check if initial cycle time repeats
                        data = data.loc[data[data['Time [ms]']==data['Time [ms]'].iloc[0]].index[-1]::].copy() # only keep last restart
                        # ccm_rpt_file.loc[c, 'Data'] = data #re-save
                        self.logger.warning(f"Cycle was restarted in {test_name}")
                        for df in [cell_cycle_metrics,ccm_rpt_file]:
                            df.loc[c, 'Min cycle voltage [V]'] = min(data['Voltage [V]'])
                            df.loc[c, 'Max cycle voltage [V]'] = max(data['Voltage [V]'])

                    else: # Force t monotonically increasing (Arbin)
                        data.drop(data[~(data.Q >= data.Q.cummax())].index, inplace = True) 
                        data = data.reset_index(drop=True)
                        self.logger.warning(f"Forcing AhT to be monotonic for esoh processing in {test_name}")
                # Use I_slow closest to I_C10 (formation) or I_C20 (RPT)
                median_charge_current = np.nanmedian(data['Current [A]'][data['Current [A]']>0])
                if  median_charge_current >  np.mean([I_C20, I_C10]):
                    I_slow = I_C10
                else: 
                    I_slow = I_C20
                
                test_name = charge['Test name']
                # try:
                self.logger.info(f"Processing eSOH for {test_name}")
                # Isolate data of interest
                data = self._isolate_data(data, I_slow = I_slow, dI_threshold = dI_threshold, V_min = V_min_data, V_max = V_max_data)

                # Pre-process data before filtering 
                t = (data['Time [ms]']-data['Time [ms]'].iloc[0])/1000
                Q = data['Ah throughput [A.h]']-data['Ah throughput [A.h]'].iloc[0]
                Q_c = max(Q) #charge['Charge capacity [A.h]']
                ccm_rpt_file.at[c,"C_voltage_win"] = Q_c
                z = 1 - (Q.iloc[-1]-Q)/Q_c #Q/rpt.Q_c set 100% at top of charge
                
                # IR CORRECTION
                # for formation, use first RPT
                if rpt_file == 0: # use first rpt 
                    IR_correction = np.zeros(len(data))
                    # if sum(cell_cycle_metrics['Protocol']=='HPPC')>0: # not fresh cell
                    #     hppc = cell_cycle_metrics[(cell_cycle_metrics['Protocol']=='HPPC') & (cell_cycle_metrics['RPT #']==charge['RPT #']+1)].iloc[0]
                else: 
                    # SOC-dependent IR correction to voltage using Rs from closest HPPC
                    hppc = ccm_rpt_file[(ccm_rpt_file['Protocol']=='HPPC') & (ccm_rpt_file['RPT #']==charge['RPT #'])].iloc[0]
                    # interpolate Rs(z)
                    try: # largest charge pulse (I>0 for charge)
                        idx_pulse = (np.array(hppc['I_pulse'])>0) & (np.array(hppc['dt_pulse'])<=dt_pulse_max) # np.array(hppc['I_pulse'])>=max(pulse_currents) #
                        # Rs = np.array(hppc['Rs'])[idx_pulse]
                        Rs = np.array(hppc['Rlong'])[idx_pulse]
                        # SOC = (np.array(hppc['Q_pulse'])[np.array(hppc['I_pulse'])>=max(pulse_currents)]-hppc['Ah throughput [A.h]'])/Q_c
                        Q_hppc = np.array(hppc['Q_pulse'])[idx_pulse]-hppc['Ah throughput [A.h]']
                        AhT_hppc = np.nanmax([hppc['Discharge capacity [A.h]'], hppc['Charge capacity [A.h]']])
                        if ~np.isnan(hppc['Charge capacity [A.h]']):# hppc on charge (formation)
                            SOC = 1-(AhT_hppc-Q_hppc)/Q_c # np.nanmax([hppc['Discharge capacity [A.h]'], hppc['Charge capacity [A.h]']]). SOC=1 at top of charge
                            # SOC = np.append(SOC, 1)  
                            # Rs = np.append(Rs, Rs[-1])  
                        else: # hppc on discharge (most rpt)
                            SOC = np.flipud(1-Q_hppc/Q_c)
                            Rs = np.flipud(Rs)
                            # SOC = np.insert(SOC, 0, 1)  # append value at z=1 for interp
                            # Rs = np.insert(Rs, 0, Rs[0])    
                            # coefficients = np.polyfit(SOC, Rs, min(4, len(SOC)))
                            # R_correction = np.polyval(coefficients, z) #max(min(np.polyval(coefficients, z), max(Rs)), min(Rs))
                        # else: 
                        #     # R_correction = np.interp(z, SOC, Rs)
                        #     coefficients = np.polyfit(SOC, Rs, min(2, len(SOC)))
                        #     R_correction = np.polyval(coefficients, z) #max(min(np.polyval(coefficients, z), max(Rs)), min(Rs))
                        # cs = interpolate.CubicSpline(SOC, Rs)
                        # if model['name'] == '_sig_dv_constantIR': 
                        R_correction = np.interp(0.5, SOC, Rs) # np.nanmean(Rs) # try constant Rs 
                        # else: 
                            # R_correction = np.interp(z, SOC, Rs)
                    except: # largest discharge current 
                        # Rs = np.array(hppc['Rs'])[np.array(hppc['I_pulse'])<=min(pulse_currents)]
                        Rs = np.array(hppc['Rlong'])[np.array(hppc['I_pulse'])<=min(pulse_currents)]
                        SOC = (np.array(hppc['Q_pulse'])[np.array(hppc['I_pulse'])<=min(pulse_currents)]-hppc['Ah throughput [A.h]'])/Q_c
                        self.logger.warning(f"Using discharge pulse for IR correction.")
                        # if model['name'] == '_sig_dv_constantIR': 
                        R_correction = np.interp(0.5, SOC, Rs) # np.nanmean(Rs) # try constant Rs 
                        # else: 
                            # R_correction = np.interp(z, SOC, Rs)
                        # coefficients = np.polyfit(SOC, Rs, min(2, len(SOC)))
                        # R_correction = max(min(np.polyval(coefficients, z), max(Rs)), min(Rs))
                    IR_correction = R_correction*data['Current [A]'].to_numpy() 

                # fit esoh params for each model 
                for model in models: 
                    if model['name'] == '_sig_dv_noIR':
                        V = data['Voltage [V]']
                    elif model['name'] == '_avgV': 
                        discharge = ccm_rpt_file[(ccm_rpt_file['Protocol']=='C/20 Discharge') & (ccm_rpt_file['RPT #']==charge['RPT #'])].iloc[0]
                        discharge_data = discharge['Data']
                        #check if subcycle time is nonmonotonic
                        if (not all(x<=y for x, y in zip(discharge_data['Time [ms]'][0:-1], discharge_data['Time [ms]'][1::]))):
                            # check if subcycle was restarted
                            if (sum(discharge_data['Time [ms]']==discharge_data['Time [ms]'].iloc[0])>1): 
                                discharge_data = discharge_data.loc[discharge_data[discharge_data['Time [ms]']==discharge_data['Time [ms]'].iloc[0]].index[-1]::].copy() # only keep last restart
                                # ccm_rpt_file.loc[c, 'Data'] = data #re-save
                                self.logger.warning(f"Cycle was restarted in {test_name}")
                            else: # Force t monotonically increasing (Arbin)
                                discharge_data.drop(discharge_data[~(discharge_data.Q >= discharge_data.Q.cummax())].index, inplace = True) 
                                discharge_data = discharge_data.reset_index(drop=True)
                                self.logger.warning(f"Forcing AhT to be monotonic for esoh processing in {test_name}")
                        # Use I_slow closest to I_C10 (formation) or I_C20 (RPT)
                        median_discharge_current = np.nanmedian(discharge_data['Current [A]'][discharge_data['Current [A]']<0])
                        if  -median_discharge_current >  np.mean([I_C20, I_C10]):
                            I_slow = -I_C10
                        else: 
                            I_slow = -I_C20
                        
                        test_name = charge['Test name']
                        # try:
                        self.logger.info(f"Processing eSOH for {test_name}")
                        # Isolate data of interest
                        discharge_data = self._isolate_data(discharge_data, I_slow = I_slow, dI_threshold = dI_threshold)
                        Q_discharge = discharge_data['Ah throughput [A.h]']-discharge_data['Ah throughput [A.h]'].iloc[0]
                        z_discharge = 1-Q_discharge/Q_c #Q/rpt.Q_c set 100% at top of charge
                        V = (data['Voltage [V]'] + np.interp(z, z_discharge.iloc[::-1], discharge_data['Voltage [V]'].iloc[::-1]))/2
                    else:
                        V = data['Voltage [V]'] - IR_correction

                    # downsample and filter data and calculate derivatives
                    downsample = min(max(int(round(2.5e-4/np.mean(np.diff(z)))),1),10) # based on UMBL2022FEB Neware data
                    downsample_idx = list(range(0, len(Q),downsample))
                    t_sim = t.iloc[downsample_idx]
                    z_sim = z.iloc[downsample_idx]
                    Q_sim = Q.iloc[downsample_idx]
                    window_length = int(min(3000/np.median(np.diff(t_sim)), int((0.065/np.average(np.diff(z_sim)) + 1)), 3000) + 1) # int((0.065/np.average(np.diff(z)) + 1)) # 
                    filtered_data = self._filter_data(t_sim, z_sim, Q_sim, V.iloc[downsample_idx], window_length=window_length) #rpt.charge.dv_ic["data"] #
                    # data = self._filter_data(t, z,Q, V, downsample = downsample, window_length= window_length)
                    V_min = min(filtered_data.V)
                    V_max = max(filtered_data.V)

                    # find peaks to estimate Cn and x0
                    x1 = x_peaks_nom[0] # low SOC peak
                    x2 = x_peaks_nom[1] # high SOC peak
                    try: 
                        idx_peaks_dv, _ = self._find_dV_peaks(filtered_data.dVdz, filtered_data.z, min_prominence=min_peak_prominence, z_upper_peak_est=-0.1/0.3*(Q_c/Q_nom-1)+0.7) # min _prominence originally
                        # calculate peak properties
                        _, _, left_ips, right_ips = peak_widths(filtered_data.dVdz, idx_peaks_dv, rel_height=1, wlen = len(filtered_data)*0.2)
                        left_ips = [int(i) for i in left_ips]
                        right_ips = [int(i) for i in right_ips]
                        # create W_peak function 
                        W_peaks = np.zeros(len(filtered_data.z))
                        for left_ip, right_ip in zip(left_ips, right_ips):
                            W_peaks[left_ip:right_ip] = 1
                        
                        # init Cn and x0
                        Q1 = filtered_data.Q[idx_peaks_dv[0]]
                        Q2 = filtered_data.Q[idx_peaks_dv[1]]
                        Cn_hat = abs(Q1-Q2)/abs(x1-x2)
                        x0_hat = x1 - Q1/Cn_hat
                        if rpt_file !=0: # replace if incorrect peaks found for non-formation cycles
                            if (Cn_hat-model['x_esoh_fit'][0]>0.2) | (Cn_hat-model['x_esoh_fit'][0]<-0.5): 
                                Cn_hat = model['x_esoh_fit'][0]
                            if abs(x0_hat - model['x_esoh_fit'][1])>0.05:
                                x0_hat = model['x_esoh_fit'][1]
                        # # init y0_hat from full-cell voltage at beginning of charge
                        # Un0 = self._U_aged(Un_data, Up_data, filtered_data.Q, stoic_0_hat = x0_hat, C_hat = Cn_hat, sigma_C = 0, electrode = 'negative', dz = 0.1, z_range = [0,0])[0]
                        # Up0 = V.iloc[0] + Un0
                        # y0_hat = np.interp(Up0, np.flipud(Up_data.U), np.flipud(Up_data.y))

                        # # init Cp_hat from full-cell voltage at end of charge
                        # Up100 = V.iloc[-1] + self._U_aged(Un_data, Up_data, filtered_data.Q, stoic_0_hat = x0_hat, C_hat = Cn_hat, sigma_C = 0, electrode = 'negative', dz = 0.1, z_range = [0,0])[-1]
                        # y100_hat = np.interp(Up100, np.flipud(Up_data.U), np.flipud(Up_data.y))
                        # Cp_hat = Q_c/(y0_hat-y100_hat)
                        
                        # if rpt_file !=0: # replace if incorrect peaks found for non-formation cycles
                        #     if (Cp_hat-model['x_esoh_fit'][2]>0.2) | (Cp_hat-model['x_esoh_fit'][2]<-0.5): 
                        #         Cp_hat = model['x_esoh_fit'][2]
                        #     if abs(y0_hat - model['x_esoh_fit'][3])>0.05:
                        #         y0_hat = model['x_esoh_fit'][3]

 
                        # min_peak_prominence = min(min(peak_prominences(data.dVdz, idx_peaks_dv)[0])*0.8,project_settings['min_peak_prominence'] ) # update to continue identifying correct peak as cell ages     
                    except: # if can't find peaks...

                        W_peaks = np.zeros(len(filtered_data.z))
                        if all(model['x_esoh_fit'] == project_settings['default_fit_params']): # and there's no previous values
                            Cn_hat, x0_hat = project_settings['default_fit_params'][0],project_settings['default_fit_params'][1]
                            # Cp_hat, y0_hat = project_settings['default_fit_params'][2],project_settings['default_fit_params'][3]
                        else: # use previous values
                            Cn_hat, x0_hat = model['x_esoh_fit'][0], model['x_esoh_fit'][1]
                            # Cp_hat, y0_hat = model['x_esoh_fit'][2], model['x_esoh_fit'][3]
                        pass
                    
                    # try: # not sure if this try except is needed... isn't needed for 152051
                        # find min soc to include in dVdz loss based on next highest max 
                    idx_z_dv_loss_min = np.argmin(abs(filtered_data.dVdz[0:int(idx_peaks_dv[0]*0.75)] - max(filtered_data.dVdz[idx_peaks_dv[-1]::]))) # ignore upper 25% of points close to peak in case that's closer than tail
                    z_dv_loss_min = filtered_data.z[idx_z_dv_loss_min] 

                    # find max soc to include in dVdz loss based on lowest trough. if there's a steep drop off at high soc, cut it off at next lowest trough. usually no tail till near EOL
                    idx_troughs_dv = np.argmin(filtered_data.dVdz[(filtered_data.z<0.8)])
                    idx_z_dv_loss_max = np.argmin(abs(filtered_data.dVdz[idx_peaks_dv[-1]::]-filtered_data.dVdz[idx_troughs_dv]))
                    z_dv_loss_max =  max(filtered_data.z[idx_peaks_dv[-1]::].iloc[idx_z_dv_loss_max],0.95) #1 # usually no tail till near EOL  
                    # except:
                    #     z_dv_loss_min = 0.05
                    #     z_dv_loss_max = 0.95
                    z_dv_loss = [z_dv_loss_min, z_dv_loss_max]
                
                # fit esoh params for each model 
                # for model in models: 
                    try: 
                        # init other params with previous rpt's fitted esoh params to speed up solver
                        if any(model['x_esoh_fit'] != project_settings['default_fit_params']): 
                            Cp_hat,y0_hat,sigma_Cn_hat,sigma_Cp_hat = model['x_esoh_fit'][2], model['x_esoh_fit'][3], model['x_esoh_fit'][4], model['x_esoh_fit'][5]
                            # sigma_Cn_hat,sigma_Cp_hat = model['x_esoh_fit'][4], model['x_esoh_fit'][5]

                        else: # use default [Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
                            # Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat = project_settings['default_fit_params'][0],project_settings['default_fit_params'][1],project_settings['default_fit_params'][2],project_settings['default_fit_params'][3]
                            Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat = project_settings['default_fit_params'][2],project_settings['default_fit_params'][3],project_settings['default_fit_params'][4],project_settings['default_fit_params'][5]
                            # sigma_Cn_hat, sigma_Cp_hat = project_settings['default_fit_params'][4],project_settings['default_fit_params'][5]

                        x_esoh_0 = [Cn_hat, x0_hat, Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat]

                        # update bounds based on last rpt results. make sure they don't exceed "global" limits
                        if model['RMSE_V']>0.025:
                            lb = project_settings['lb_esoh']
                            ub = project_settings['ub_esoh']
                        else:
                            # lb0 = [Cn_hat-1, x0_hat-0.1, Cp_hat-1.5, y0_hat-0.1, sigma_Cn_hat*0.5, sigma_Cp_hat*0.5]
                            # ub0 = [Cn_hat+0.3, x0_hat+0.05, Cp_hat+0.3, y0_hat+0.05, project_settings['ub_esoh'][-2], project_settings['ub_esoh'][-1]]
                            lb0 = [Cn_hat-1, project_settings['lb_esoh'][1], project_settings['lb_esoh'][2], project_settings['lb_esoh'][3], sigma_Cn_hat*0.5, project_settings['lb_esoh'][5]]
                            ub0 = [Cn_hat+0.4, project_settings['ub_esoh'][1], project_settings['ub_esoh'][2], project_settings['ub_esoh'][3], project_settings['ub_esoh'][4], project_settings['ub_esoh'][5]]
                            lb = [max(l0,l) for (l0, l) in zip(lb0, project_settings['lb_esoh'])] 
                            ub = [min(u0,u) for (u0, u, l) in zip(ub0, project_settings['ub_esoh'], project_settings['lb_esoh'])] 
                            # lb = project_settings['lb_esoh']
                            # ub = project_settings['ub_esoh']
                        # if sigmas are not a fitting parameter, zero them out
                        if ('sig' not in model['name']) and (model['name'] != ''): 
                            x_esoh_0 =x_esoh_0[0:-2] + [0,0] 
                            lb = lb[0:-2] + [0,0] 
                            ub = ub[0:-2] + [0,0] 
                        
                        # fit params and save values if low RMSE_V. if error is high, ignore fitted theta unless formation
                        if any(model['x_esoh_fit']!=project_settings['default_fit_params']): 
                            theta, capacity, x_esoh_fit, RMSE_V, RMSE_dVdq, RMSE_V_full, RMSE_dVdq_full, dQ_low_peak,dQ_high_peak, capacity_error = self._esoh_est(filtered_data, capacity = Q_c, x_esoh_0 = x_esoh_0, x_esoh_prev = model['x_esoh_fit'], Un = Un_data, Up = Up_data, lb = lb, ub =ub, W = model['W'], W_peaks = W_peaks, dz_n = model['dz_n'], dz_p = model['dz_p'], z_dv_loss=z_dv_loss, min_peak_prominence=min_peak_prominence, Q_nom=Q_nom) # rmse_V is a mag smaller than rmse_dvdq, 1/(40*max(model.x_esoh_0[4],0.1))
                        else: 
                            model['W'][2]=0
                            theta, capacity, x_esoh_fit, RMSE_V, RMSE_dVdq, RMSE_V_full, RMSE_dVdq_full, dQ_low_peak,dQ_high_peak,capacity_error = self._esoh_est(filtered_data, capacity = Q_c, x_esoh_0 = x_esoh_0, x_esoh_prev = x_esoh_0, Un = Un_data, Up = Up_data, lb = lb, ub =ub, W = model['W'], W_peaks = W_peaks, dz_n = model['dz_n'], dz_p = model['dz_p'], z_dv_loss=z_dv_loss, min_peak_prominence=min_peak_prominence, Q_nom=Q_nom) # rmse_V is a mag smaller than rmse_dvdq, 1/(40*max(model.x_esoh_0[4],0.1))

                        if ((RMSE_V_full > 0.02) or (RMSE_dVdq>0.1)): # check for high error
                            self.logger.warning(f"Error in V estimation is too high for model {model['name']}: {RMSE_V}. For {test_name}")
                            if rpt_file !=0: # don't nan if it's a formation cycle
                                theta[:] = np.full(len(theta), np.NaN)
                        if (RMSE_V_full > 0.015) and model['name'] == "": # check for high error
                            self.logger.warning(f"cell 152051 high error ")
                        if rpt_file ==0: # reset initial conditions after formation cycles
                            x_esoh_fit = x_esoh_0 # don't init with fitted bc of high error. Use previous working one

                        else: 
                            model.update({
                                'theta': theta,
                                'capacity': capacity,
                                'x_esoh_fit': x_esoh_fit,
                                'RMSE_V': RMSE_V,
                                'RMSE_dVdq': RMSE_dVdq,
                                'RMSE_V_full':RMSE_V_full, 
                                'RMSE_dVdq_full':RMSE_dVdq_full, 
                                'dQ_low_peak': dQ_low_peak, 
                                'dQ_high_peak': dQ_high_peak,
                                'RMSE_V':RMSE_V, 
                                'capacity_error':capacity_error,
                                })
                    except: # tried but couldn't fit or has high error
                        model.update({
                            'theta': [np.nan]*8,
                            'capacity': np.nan,
                            'x_esoh_fit': [np.nan]*5, 
                            'RMSE_V': np.nan,
                            'RMSE_dVdq': np.nan,
                            'RMSE_V_full':RMSE_V_full, 
                            'RMSE_dVdq_full':RMSE_dVdq_full, 
                            'dQ_low_peak': np.nan, 
                            'dQ_high_peak': np.nan,
                            'capacity_error': np.nan
                            })
                        # continue

                    # Update the cell_cycle_metrics with eSOH data
                    for df in [cell_cycle_metrics, ccm_rpt_file]:
                        df.at[c, ''.join('C' + model['name'])] = capacity
                        df.at[c, ''.join('Cn' + model['name'])] = theta[0]
                        df.at[c, ''.join('x0' + model['name'])] = theta[1]
                        df.at[c, ''.join('x100' + model['name'])] = theta[2]
                        df.at[c, ''.join('sigma_Cn' + model['name'])] = theta[3]
                        df.at[c, ''.join('Cp' + model['name'])] = theta[4]
                        df.at[c, ''.join('y0' + model['name'])] = theta[5]
                        df.at[c, ''.join('y100' + model['name'])] = theta[6]
                        df.at[c, ''.join('sigma_Cp' + model['name'])] = theta[7]
                        df.at[c, ''.join('RMSE_V' + model['name'])] = RMSE_V
                        df.at[c, ''.join('RMSE_dVdQ' + model['name'])] = RMSE_dVdq
                        df.at[c, ''.join('RMSE_V_full' + model['name'])] = RMSE_V
                        df.at[c, ''.join('RMSE_dVdQ_full' + model['name'])] = RMSE_dVdq
                        df.at[c, ''.join('p1_err' + model['name'])] = dQ_low_peak
                        df.at[c, ''.join('p2_err' + model['name'])] = dQ_high_peak
                        df.at[c, ''.join('capacity_error' + model['name'])] = capacity_error
            # update rows corresponding to this rpt/rpt_file in cell_rpt_data         
            for new_column in new_columns:  
                cell_rpt_data.loc[cell_rpt_data['RPT #'] == rpt_file, new_column]= ccm_rpt_file[new_column].values #pd.concat([cell_rpt_data, ccm_rpt_file])
        # END ESOH UPDATE -----------------
        # after all rpts have been fitted, calculate and update LLI and LAM and save
        for model in models: 
            for df in [cell_cycle_metrics, cell_rpt_data]:
                n_Li, LLI, LAM_NE, LAM_PE = self._calculate_LLI_LAM(df[''.join('Cn' + model['name'])],df[''.join('x0' + model['name'])],df['x100' + model['name']],df['Cp' + model['name']],df['y0' + model['name']],df['y100' + model['name']], df['RPT #'])
                df[''.join('n_Li' + model['name'])] = n_Li
                df[''.join('LLI' + model['name'])] = LLI
                df[''.join('LAM_NE' + model['name'])] = LAM_NE
                df[''.join('LAM_PE' + model['name'])] = LAM_PE

        # format df
        cell_rpt_data = cell_rpt_data.sort_values(by=['Ah throughput [A.h]']) # no 'Time [ms]' or 'timestamp'?
        cell_rpt_data = cell_rpt_data[[p in rpt_protocols for p in cell_rpt_data['Protocol']]] # only keep hppc and c/20 dis/charge
        # cell_rpt_data = cell_rpt_data[cell_rpt_data['Protocol'].notna()] # nans pass through previous filter so remove here
        cell_rpt_data.reset_index(drop=True, inplace=True)

        return cell_rpt_data, cell_cycle_metrics
    
    def _calculate_LLI_LAM(self, Cn,x0,x100,Cp,y0,y100, rpt_num): 
        try: 
            first_idx = rpt_num[(rpt_num==1) & (Cn.notna())].index[0]
            n_Li = 3600/96485*(x0*Cn+y0*Cp)
            LLI = (1-n_Li/n_Li.loc[first_idx])
            LAM_NE = (1-Cn/Cn.loc[first_idx])
            LAM_PE = (1-Cp/Cp.loc[first_idx])
        except:
            n_Li = [np.nan]*len(Cn)
            LLI = [np.nan]*len(Cn)
            LAM_NE = [np.nan]*len(Cn)
            LAM_PE = [np.nan]*len(Cn)     
        return n_Li, LLI, LAM_NE, LAM_PE

    def _isolate_data(self, data, I_slow, dI_threshold, V_min = 3.0, V_max = 4.2): 
        """
        Isolate data where current is within dI_threshold of I_slow. 
        Used for pre-processing data for eSOH fitting
        Parameters
        ----------
        data: Dataframe 
        Timeseries data of the subcycle
        I_slow: float 
        Current of interest (e.g. C/20 or C/10)
        dI_threshold: float
        Largest allowable current deviation 
        Returns
        -------
        data: 
        """
        # start_idx = data[abs(data['Current [A]']-(I_slow))<dI_threshold].index[0]
        # end_idx = data[abs(data['Current [A]']-(I_slow))<dI_threshold].index[-1]
        # if data['Current [A]'].value_counts().get(0)!=None: # if data jumps in and out of the acceptable current range, check 
        #     idx_interrupt_start = data[data['Current [A]']==0].index[0]
        #     idx_interrupt_end = data[data['Current [A]']==0].index[-1]
        #     idx_endpoint_start = np.argmin(abs(np.array([start_idx, end_idx]) - idx_interrupt_start)) 
        #     idx_endpoint_end = np.argmin(abs(np.array([start_idx, end_idx]) - idx_interrupt_end)) 
        #     # if data jumps to 0 and back for a SHORT data interruption (rare arbin cell) so that start and finish of interruption are close to the same end of data, use as cut-off from closest end (ex:152060, 152061)
        #     if idx_endpoint_start == idx_endpoint_end: 
        #         if idx_endpoint_start ==0: # if interruption is at the beginning of the data
        #             start_idx = idx_interrupt_end # only include data after interruption
        #         else: 
        #             end_idx = idx_interrupt_start # only include data before interruption
        # # best estimate of set current
        # # I_desire = np.median(data.loc[start_idx:end_idx, 'Current [A]'])

        # if end_idx != len(data)-1: 
        #     data.drop(data.loc[0:start_idx-1].index, inplace=True)  
        #     data.drop(data.loc[end_idx::].index, inplace=True)     

        # when current is within expected bounds of I_slow
        valid_data = (abs(data['Current [A]']-I_slow)<dI_threshold)# & (data['Voltage [V]']>=V_min) & (data['Voltage [V]']<=V_max) #& data.index>np.where(np.array(list(np.diff(data[valid_data].index))+[1])>1)
        # data = data.loc[valid_data].copy()  
        data.drop(data.loc[~valid_data].index, inplace=True)

        # check for large gaps in the data, and reset time.
        I_subcycle = data['Current [A]']
        t_subcycle = data['Time [ms]']
        t_start = t_subcycle.iloc[0]
        while True:
            dt = np.diff(t_subcycle)
            dt=np.append(dt,0)
            gap_index= np.argwhere(abs(dt)>1e5)#look for gaps greater than 100 s
            # remove dt gaps larger than dAh_max and reintegrate
            if (gap_index.size >0):
                # if gap_index[0][0]>0:
                for gap in gap_index:
                    g = gap[0]
                    dt[g] = min(dt[g-1], dt[g+1])
                t_subcycle = t_start + np.cumsum(dt)
            else:
                break
        data['Time [ms]'] = t_subcycle
        Q_subcycle = integrate.cumtrapz(I_subcycle, (t_subcycle-t_start)/1000)/3600
        Q_subcycle = np.append(Q_subcycle,Q_subcycle[-1])
        data['Ah throughput [A.h]']=Q_subcycle
        return data
    
    def _filter_data(self, t, z, Q, V, window_length = 300, polyorder=3): 
        """
        Downsample and filter data and calculate derivatives for eSOH fitting
        Parameters
        ----------
        t: Series of time data in [s]
        z: Series of SOC data calculated such that 100% is top of charge
        Q: Series of subcycle AhT data in [A.h]
        V: Series of voltage data in [V]
        downsample: Int. Reduces length of filtered_data to speed up optimization. Default is no downsampling.
        
        Returns
        -------
        filtered_data: DataFrame
        
        """
        # filtered signal
        tf_d = savgol_filter(t,window_length,polyorder,0) 
        zf_d = savgol_filter(z,window_length,polyorder,0) 
        Vf_d = savgol_filter(V,window_length, polyorder,0)
        Qf_d = savgol_filter(Q,window_length, polyorder,0)
        # filtered difference
        dz_d = savgol_filter(z,window_length,polyorder,1)
        dQ_d = savgol_filter(Q,window_length,polyorder,1)
        dV_d = savgol_filter(V,window_length,polyorder,1) 
        # calculate derivative
        dVdQ_d = dV_d/dQ_d
        # dQdV_d = dQ_d/dV_d
        # dzdV_d = dz_d/dV_d 
        dVdz_d = dV_d/dz_d

        # save data to dataframe
        filtered_data = pd.DataFrame.from_dict({
                "t": tf_d,
                "Q": Qf_d,
                "z": zf_d,
                "V": Vf_d,
                "dQ": dQ_d,
                "dz": dz_d,
                "dV": dV_d,
                "dVdQ": dVdQ_d, 
                "dVdz": dVdz_d,
                # "dQdV": dQdV_d,
                # "dzdV": dzdV_d,
            }, dtype='float') # remove warning about poor performance?
        return filtered_data
    
    def _find_dV_peaks(self, dVdz, z, min_prominence = 0.002, z_low_peak_min = 0.1, z_low_peak_max = 0.5, z_upper_peak_est = 0.7,):
        """
        Find indices in given data where peaks occur based on SOC peak locations
        Parameters
        ----------
        dVdz: Series of filtered data
        z: Corresponding series of filtered data. Should have same length as dVdz
        min_prominence: Float for corresponding param in find_peaks()
        z_low_peak_min: Float. Lowest expected peak location in z for the low SOC peak
        z_low_peak_max: Float. Highest expected peak location in z for the low SOC peak
        
        Returns
        -------
        idx_peaks_dv: List of indices in z and dVdz for the peaks of interet
        idx_peaks0_dv: List of ALL indices in z and dVdz for identified peaks before filtering
        """
        # find DV peaks
        idx_peaks0_dv,_ = find_peaks(abs(dVdz),prominence=(min_prominence,None)) # , height=(0.2, None), distance tuned to remove the smallest peak in the set of 3 (~Q=0.6 on fresh)
        idx_peaks_dv = [[p for p in idx_peaks0_dv if (z_low_peak_max<z[p])][np.argmin(abs(z.loc[[p for p in idx_peaks0_dv if (z_low_peak_max<z[p])]]-z_upper_peak_est))]] # upper peak (and (z[p]<z_upper_peak_max)))

        idx_low_peak_idx0 = sorted([p for p in idx_peaks0_dv if (z_low_peak_min<z[p]) and (z[p]<z_low_peak_max)])
        if len(idx_low_peak_idx0)>=3:
                idx_peaks_dv.append(idx_low_peak_idx0[1])
        else:
            try:
                idx_peaks_dv.append(idx_low_peak_idx0[0])
                
            except: # very aged discharge 
                pass
        idx_peaks_dv.sort() # sort so in order of low, then high SOC peak
        return idx_peaks_dv, idx_peaks0_dv

    # class eSOH(Problem):
    #     def __init__(self, **kwargs):
    #         super().__init__(n_var=6, n_obj=1, n_ieq_constr=4, xl=np.array([1, 0, 1, 0,0,0]), xu=np.array([5, 1, 5, 1,0.5,0.5]),**kwargs) # xl = lb on x, xu = ub on x
    #         self.Q = kwargs['Q']
    #         self.data = kwargs['data']
    #         self.W = kwargs['W']
    #         self.obj_func = kwargs['obj_func']
    #         self.obj_func_args = kwargs['obj_func_args']

    #     def _evaluate(self, x, out):
    #         capacity = self.Q
    #         out["F"] = np.array([self.obj_func(x0, *self.obj_func_args) for x0 in x]) #np.sum((x - 0.5) ** 2, axis=1) #
    #         x100 = capacity/x[:,0]+x[:,1]
    #         y100 = -capacity/x[:,2]+x[:,3]
    #         out["G"] = np.array([
    #                     (x100-self.xu[1], self.xl[1]-x100,y100-self.xu[3], self.xl[3]-y100),
    #                     ])
            
    def _esoh_est(self,filtered_data, Un, Up, capacity, x_esoh_0 = [4.2,0.85,5.5,0.3,0,0], x_esoh_prev = [4.2,0.85,5.5,0.3,0,0], lb = [1, 0, 1, 0,0,0], ub = [5, 1, 6.5, 1, 0.3, 0.3], W = [1,1,1], W_peaks = np.zeros(1),z_dv_loss = [0.05,0.95], dz_n = 1, dz_p = 1, min_peak_prominence=0.002, Q_nom = 2.5):
        """
        Run optimization to find eSOH parameters and calculate errors
        Parameters
        ----------
        filtered_data: DataFrame of filtered data with columns from _filter_data()
        Un: DataFrame with columns [x,U] for interpolating aged Un
        Up: DataFrame with columns [y,U] for interpolating aged Up
        capacity: Float. Current RPT capacity, typically from C/20 charge
        x0: List of floats. Initial guess of x for minimize(), where x =[Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        lb: List of floats. Lower bounds of corresponding parameters in x0.
        ub: List of floats. Upper bounds of corresponding parameters in x0.
        z_dv_loss: List with [min, max] of SOC range to include for dVdQ error calculation

        Return
        -------
        theta: List of fitted and derived eSOH parameters [C_n,x_0,x_100,sigma_Cn,C_p,y_0,y_100, sigma_Cp]
        capacity: Float. Pass through calculated capacity from cycler data.
        x_esoh_fit: List of fitted eSOH parameters [Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        RMSE_V, RMSE_dVdq,dQ_low_peak,dQ_high_peak: Floats 
        """

        # setup fitting optimization 
        soh=capacity/Q_nom
        bounds = Bounds(lb, ub)
        # V_max,V_min = max(filtered_data.V), min(filtered_data.V)
        nleq1 = lambda x: capacity/x[0]+x[1] 
        nlcon1 = NonlinearConstraint(nleq1, 0, 1) # ineq constraint 0 < (x0+C/Cn = x100) < 1
        nleq2 = lambda x:  -capacity/x[2]+x[3]
        nlcon2 = NonlinearConstraint(nleq2, 0, 1) # ineq constraint 0 < (y0-C/Cp = y100) < 1

        # Set up the constraints dictionary
        # eq_constraint = lambda x: np.interp(max(filtered_data.Q), filtered_data.Q, self._calc_ocv(x,filtered_data.Q, Un, Up, dz_p = dz_p, dz_n = dz_n)) # min voltage
        # constraints = NonlinearConstraint(eq_constraint, V_max, V_max) # 
        x_fit = minimize(self._loss_func, x_esoh_0,
                        args=(x_esoh_prev, filtered_data, Un, Up, W, W_peaks, z_dv_loss[0], z_dv_loss[1], dz_n, dz_p, soh, min_peak_prominence),
                        bounds = bounds,
                        constraints=[nlcon1, nlcon2],
                        tol=1e-6,
                        #  options={'disp': True},
                        #  method='SLSQP', 
                        ).x

        # pymoo: pattern search
        # problem = self.eSOH(Q = capacity, W = W, data = filtered_data, 
        #                     obj_func=self._loss_func,
        #                     obj_func_args=(filtered_data, Un, Up, W, z_dv_loss[0], z_dv_loss[1], dz_n, dz_p)
        #                     )
        # algorithm = PatternSearch(x0 = np.array(x_esoh_0))
        # res = pmminimize(problem,
        #             algorithm,
        #             verbose=False,
        #             seed=1, 
        #             )
        # x_fit = res.X # [Cn, x0, Cp, y0]

        # assign results to theta: [Cn, x0, Cp, y0, sigma_Cn, sigma_Cp] -> [C_n,x_0,x_100,sigma_Cn,C_p,y_0,y_100, sigma_Cp]
        C_n, x_0, sigma_Cn  = x_fit[0], x_fit[1], x_fit[4]
        C_p, y_0, sigma_Cp = x_fit[2], x_fit[3], x_fit[5]
        # V_min_error = lambda C: abs(np.interp(0, C-(max(filtered_data.Q)-filtered_data.Q), self._calc_ocv(x_fit,C-(max(filtered_data.Q)-filtered_data.Q), Un, Up, dz_p = dz_p, dz_n = dz_n))-V_min)**2
        # V_max_error = lambda C: abs(np.interp(C, C-(max(filtered_data.Q)-filtered_data.Q), self._calc_ocv(x_fit,C-(max(filtered_data.Q)-filtered_data.Q), Un, Up, dz_p = dz_p, dz_n = dz_n))-V_max)**2
        # capacity_hat = minimize(V_min_error, max(filtered_data.Q),
        #         bounds = Bounds([0], [ub[0]]),
        #         constraints=[NonlinearConstraint(V_max_error, V_max-0.01, V_max+0.01)],
        #         tol=1e-6,
        #         #  options={'disp': True},
        #         #  method='SLSQP', 
        #         ).x
        
        x_100 = capacity/C_n + x_0
        y_100 = -capacity/C_p + y_0
        capacity_hat = C_n*(x_100 - x_0)

        # x_100 = capacity_hat/C_n + x_0
        # y_100 = -capacity_hat/C_p + y_0
        theta = [C_n,x_0,x_100,sigma_Cn,C_p,y_0,y_100, sigma_Cp]
        # theta = [round(tt,4) for tt in theta]

        # sim OCV from fitted eSOH
        Q_sim = filtered_data.Q
        t_sim = filtered_data.t
        z_sim = filtered_data.z
        V_sim_fit = self._calc_ocv(x_fit, Q_sim, Un, Up, dz_n = dz_n, dz_p = dz_p) 
        V_sim_fit_prev = self._calc_ocv(x_esoh_prev, Q_sim, Un, Up, dz_n = dz_n, dz_p = dz_p)
        # dVdq_sim = np.diff(V_sim_fit)/np.diff(Q_sim)
        # dVdq_sim = np.append(dVdq_sim,dVdq_sim[-1])
        sim = self._filter_data(t_sim, z_sim, Q_sim, V_sim_fit, window_length= int(min(3000/np.median(np.diff(t_sim)), int((0.065/np.average(np.diff(z_sim)) + 1)), 3000) + 1)) #int((40/np.average(np.diff(z_sim))/1000 + 1))) #int(3000/np.average(np.diff(t_sim)) + 1) int((4/np.average(np.diff(z_sim))/1000 + 1))
        sim_prev = self._filter_data(t_sim, z_sim, Q_sim, V_sim_fit_prev, window_length= int(min(3000/np.median(np.diff(t_sim)), int((0.065/np.average(np.diff(z_sim)) + 1)), 3000) + 1)) #int((40/np.average(np.diff(z_sim))/1000 + 1))) #int(3000/np.average(np.diff(t_sim)) + 1) int((4/np.average(np.diff(z_sim))/1000 + 1))

        # calculate errors
        valid_data  = (z_sim>z_dv_loss[0]) & (z_sim<z_dv_loss[1])
        RMSE_V = np.sqrt(np.mean(np.square(sim.V-filtered_data.V)))
        RMSE_dVdq = np.sqrt(np.mean(np.square(sim.dVdQ[valid_data]-filtered_data.dVdQ[valid_data])))
        RMSE_V_full = np.sqrt(np.mean(np.square(sim.V-filtered_data.V)))
        RMSE_dVdq_full = np.sqrt(np.mean(np.square(sim.dVdQ-filtered_data.dVdQ)))
        RMSE_reg_V_full = np.sqrt(np.mean(np.square(sim.V-sim_prev.V)))
        # RMSE_peaks = np.sqrt(np.mean(np.square((sim.dVdQ-filtered_data.dVdQ)*W_peaks)))
        
        if (RMSE_V_full > 0.010) or (RMSE_dVdq > 0.06):
                self.logger.warning(f"High RMSE_V error({RMSE_V_full}) and/or high dVdq error {RMSE_dVdq_full}")
                if (RMSE_V_full>0.0190) & (RMSE_V_full<0.022):
                    print('High RMSE_V_full CELL152085: can filter out if min V <2.4V')
        try: 
            idx_peaks_dv_data, _ = self._find_dV_peaks(filtered_data.dVdz, filtered_data.z, z_upper_peak_est=-0.1/0.3*(capacity/Q_nom-1)+0.7) # recalculate with filtered data. can also bring in original
            Q_peak_data = filtered_data.Q[idx_peaks_dv_data]
            idx_peaks_dv, _ = self._find_dV_peaks(sim.dVdz, sim.z, min_prominence=min_peak_prominence, z_upper_peak_est=-0.1/0.3*(capacity/Q_nom-1)+0.7)
            peak_errors = sim.Q[idx_peaks_dv].to_numpy()-Q_peak_data.to_numpy()
            dQ_low_peak = peak_errors[0]
            dQ_high_peak = peak_errors[1]
        except: # can't find peaks (e.g. formation C/10)
            dQ_low_peak = np.nan
            dQ_high_peak = np.nan
        capacity_error = capacity_hat-capacity
        return theta, capacity_hat, x_fit, RMSE_V, RMSE_dVdq, RMSE_V_full, RMSE_dVdq_full, dQ_low_peak,dQ_high_peak, capacity_error 
    
    def _loss_func(self,x_esoh,x_esoh_prev, filtered_data, Un, Up, W = [1, 1, 1],W_peaks=np.zeros(1), z_loss_min = 0.05, z_loss_max = 0.95, dz_n = 1, dz_p = 1, soh = 0.7, min_peak_prominence=0.002):
        """
        Calculate loss function based on model fit error 
        INPUTS
        x: Array of eSOH parameters [Cn, x0, Cp, y0]
        Q: pd.Series of AhT data for C/20 charge
        filtered_data: pd.DataFrame with columns of filtered data from "_filter_qv_data"
        W: List of weights on V error, DV error, and IC error
        z_loss_min: Float. Min SOC of data range for calculating dVdQ error  
        z_loss_max: Float. Max SOC of data range for calculating dVdQ error  
        OUTPUT
        loss: Calculated loss based on summing weighted errors (1 value)
        """
        # peaks_dv_idx, peaks_ic_idx = _find_peaks(filtered_data)
        t_data = filtered_data.t.to_numpy()
        z_data = filtered_data.z.to_numpy()
        Q_data = filtered_data.Q.to_numpy()
        V_data = filtered_data.V.to_numpy()
        dVdQ_data = filtered_data.dVdQ.to_numpy()
        # dQdV_data = filtered_data.dQdV.to_numpy()
        # x_esoh[4] = 0.2
        V_sim = self._calc_ocv(x_esoh, Q_data, Un, Up, dz_n = dz_n, dz_p = dz_p)
        # V_sim_prev = self._calc_ocv(x_esoh_prev, Q_data, Un, Up, dz_n = dz_n, dz_p = dz_p)
        # filter window length depends on average delta_z
        # dV_sim = np.diff(V_sim)
        # dV_sim = np.append(dV_sim, dV_sim[-1])
        # dVdQ_sim = dV_sim/filtered_data.dQ
        # # dQ_sim = np.diff(Q_data)
        # # dQ_sim = np.append(dQ_sim, dQ_sim[-1])
        # # dVdQ_sim = dV_sim/ dQ_sim #filtered_data.dQ (not sure why np.mean(np.diff(Q_data)) is a mag larger than np.mean(filtered_data.dQ)
        # # dQdV_sim = 1/dVdQ_sim
        sim = self._filter_data(t_data, z_data, Q_data, V_sim, window_length= int(min(3000/np.median(np.diff(t_data)), int((0.065/np.average(np.diff(z_data)) + 1)), 3000) + 1))
        #sim_prev = self._filter_data(t_data, z_data, Q_data, V_sim_prev, window_length=int(min(3000/np.average(np.diff(t_data)), 3000) + 1))
        dVdQ_sim = sim.dVdQ.to_numpy()
        
        # calculate errors within valid ranges of Q and V
        valid_z_data = (z_loss_min <z_data) & (z_data<z_loss_max) # ignore data outside [0.1,0.9] SOC
        V_error = V_sim-V_data #V_sim[valid_z_data]-V_data[valid_z_data]
        dVdQ_error = (dVdQ_sim[valid_z_data]-dVdQ_data[valid_z_data])#*dVdQ_data[valid_z_data] #<---NOT RMSE use dVdQ_data as a weight
        #reg_error = V_sim-V_sim_prev
        # dQdV_error = dQdV_sim[valid_z_data]-dQdV_data[valid_z_data]
        # try: 
        #     idx_peaks_dv_data, _ = self._find_dV_peaks(filtered_data.dVdz, filtered_data.z, z_upper_peak_est=-0.1/0.3*(soh-1)+0.7) # recalculate with filtered data. can also bring in original
        #     Q_peak_data = filtered_data.Q[idx_peaks_dv_data]
            # idx_peaks_dv, _ = self._find_dV_peaks(sim.dVdz, sim.z, min_prominence=min_peak_prominence, z_upper_peak_est=-0.1/0.3*(soh-1)+0.7)
            # peak_errors = sim.Q[idx_peaks_dv].to_numpy()-Q_peak_data.to_numpy()
            # dQ_low_peak = peak_errors[0]
            # dQ_high_peak = peak_errors[1]
        # except: # can't find peaks (e.g. formation C/10)
        #     dQ_low_peak = np.nan
        #     dQ_high_peak = np.nan

        # calculate loss based on error, normalized by max value within valid range of V and Q
        error = lambda error: np.sqrt(np.nanmean(error**2)) # rmse
        RMSE_V = error(V_error)#/np.mean(V_data[valid_z_data])
        RMSE_dV = error(dVdQ_error)#/np.mean(dVdQ_data[valid_z_data])
        # RMSE_reg = error(reg_error)
        RMSE_peaks = error((dVdQ_sim-dVdQ_data)*W_peaks)
        V_loss = W[0]*RMSE_V
        dVdQ_loss = W[1]*RMSE_dV
        peak_loss = W[2]*RMSE_peaks
        # reg_loss = W[2]*RMSE_reg
        # if np.isnan(dQ_high_peak):
        #     high_peak_loss = 0
        # else: 
        #     high_peak_loss = abs(dQ_high_peak)*W[2]
        # dQdV_loss = W[2]*error(dQdV_error)/np.mean(dQdV_data[valid_z_data]) 
        loss = V_loss + dVdQ_loss + peak_loss #+ reg_loss#+ high_peak_loss#+ dQdV_loss
        return loss #, (RMSE_V, RMSE_dV), dVdQ_error #, dQdV_loss

    def _calc_ocv(self,x,Q, Un0, Up0, dz_p = 1, dz_n = 1): #self._calc_ocv(x_esoh, Q_data, Un, Up)
        """
        Calculate ocv curve given eSOH parameters in x
        x: List of subset of eSOH params [Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        Q: np.array of Q data
        """
        Up = self._U_aged(Un0, Up0, Q, stoic_0_hat = x[3], C_hat = x[2], sigma_C = x[5], electrode = 'positive', dz = dz_p, z_range = [0,0]) # z_range = [0,0] for no distribution in Cp
        Un = self._U_aged(Un0, Up0, Q, stoic_0_hat = x[1], C_hat = x[0], sigma_C = x[4], electrode = 'negative', dz = dz_n, z_range = [-3,3]) # increased # of disc. pts with increased sigma_Cn to avoid dUn/dQ peak separation
        ocv = Up - Un
        return ocv
    
    def _U_aged(self, Un, Up, Q, stoic_0_hat, C_hat, electrode = 'negative', sigma_C = 0.10, dz = 0.25, z_range = [-3,3]): 
        # create C distribution 
        Cs, p_values = self._C_dist(C_hat, sigma_C, dz = dz, z_range = z_range)

        # calculate Uns 
        Us = []
        for C in Cs:
            if electrode.lower() =='negative': 
                stoic = self._q2x (Q,C,stoic_0_hat)
                U = np.interp(stoic, Un.x,Un.U)
            elif electrode.lower() =='positive': 
                stoic = self._q2y (Q,C,stoic_0_hat)
                U = np.interp(stoic, Up.y,Up.U)
            Us.append(U)

        # weighted average by probability and average
        U_avg = np.average(np.vstack(Us), axis=0, weights=p_values) 
        U_avg = savgol_filter(U_avg,window_length=300,polyorder=3,deriv=0)
        return U_avg
    
    def _C_dist(self, C_hat, sigma_C, z_range = [-3,3], dz = 0.25): 
        z_scores = np.arange(z_range[0],z_range[1] + dz,dz)
        p_values =  norm.pdf(z_scores)#norm.cdf(-abs(z_scores)) 
        p_values = p_values/sum(p_values)
        C = C_hat + sigma_C*z_scores
        return C, p_values
    
    def _q2x (self,q,Cn,x0):
        x = x0+q/Cn
        return x 
    def _q2y (self,q,Cp,y0):
        y = y0-q/Cp
        return y 

    def update_cycle_metrics_hppc(self, rpt_subcycle, cycle_metrics, i, pulse_currents, hppc_pts = 4):
        """
        Update cell cycle metrics based on the RPT subcycle data.

        Parameters:
        -----------
        rpt_subcycle: dict
            The dictionary of the RPT subcycle data
        cycle_metrics: list of DataFrame
            The list of dataframes with cycle metrics to update
        i: int
            The index of the cell cycle metrics
        pulse_currents: list of float
            The list of pulse currents
        hppc_pts: int
            Number of pts to take before and after the pulse

        Returns:
        --------
        None
        """
        if rpt_subcycle['Protocol'] == 'HPPC':
            # Extract necessary data for get_Rs_SOC function
            time_s = rpt_subcycle['Data']['Time [ms]'] / 1000.0
            current_a = rpt_subcycle['Data']['Current [A]']
            voltage_v = rpt_subcycle['Data']['Voltage [V]']
            ah_throughput = rpt_subcycle['Data']['Ah throughput [A.h]']
            # Call the get_Rs_SOC function with PULSE_CURRENTS from config
            hppc_data = self.get_Rs_SOC(time_s, current_a, voltage_v, ah_throughput, pts = hppc_pts)
            # Dynamically generate metrics_mapping based on PULSE_CURRENTS
            
            for df in cycle_metrics:
                for col in ["Q_pulse","dt_pulse","I_pulse","Rs","Rlong"]:
                    if col not in df.columns:
                        df.loc[:, col] = [np.nan] #[np.nan]*len(df)
                    df[col] = df[col].astype(object)
                if not hppc_data['Q'].empty:
                # Update the cell_cycle_metrics with the new data
                    df.at[i, "Q_pulse"] = hppc_data['Q'].tolist()# if hppc_data['Q'].empty else np.nan
                    df.at[i, "dt_pulse"] = hppc_data['pulse_duration'].tolist()#[0] if hppc_data['pulse_duration'].empty  else np.nan
                    df.at[i, "I_pulse"] = hppc_data['pulse_current'].tolist()#[0] if hppc_data['pulse_current'].empty  else np.nan
                    df.at[i, "Rs"] = hppc_data['R_s'].tolist()#[0] if hppc_data['R_s'].empty  else np.nan
                    df.at[i, "Rlong"] = hppc_data['R_l'].tolist()#[0] if hppc_data['R_l'].empty else np.nan


    def get_Rs_SOC(self, t, I, V, Q, pts = 4, max_dt_pulse=10000, delta_I = 0.1):
        """ 
        Processes HPPC data to get DC Resistance for given pulse currents at different Qs. 
        Assumes that this is a discharge HPPC i.e. the initial Q is 0, corresponding to 100% SOC
        """
        # # find candidate pulse start and end indices
        # charge_pulse_start_idx0, discharge_pulse_start_idx0 = self._find_cycle_idx(t, I, V, Q, dAh_max = 0.5)
        # charge_pulse_end_idx0, discharge_pulse_end_idx0 = self._find_cycle_idx(t.iloc[::-1], I.iloc[::-1], V.iloc[::-1], Q.iloc[::-1], dAh_max = 0.5)
        # charge_pulse_end_idx0 = np.sort(len(t)-charge_pulse_end_idx0)
        # discharge_pulse_end_idx0 = np.sort(len(t)-discharge_pulse_end_idx0)
        
        # # keep pulses shorter than max_dt_pulse. assume last one might get cutoff
        # num_charge_pulses = min(len(charge_pulse_start_idx0),len(charge_pulse_end_idx0))
        # num_discharge_pulses = min(len(discharge_pulse_start_idx0),len(discharge_pulse_end_idx0))
        # charge_pulse_start_idx = charge_pulse_start_idx0[0:num_charge_pulses][(charge_pulse_end_idx0-charge_pulse_start_idx0[0:num_charge_pulses])<max_dt_pulse]
        # discharge_pulse_start_idx = discharge_pulse_start_idx0[0:num_discharge_pulses][(discharge_pulse_end_idx0-discharge_pulse_start_idx0[0:num_discharge_pulses])<max_dt_pulse]
        # charge_pulse_end_idx = charge_pulse_end_idx0[(charge_pulse_end_idx0-charge_pulse_start_idx0[0:num_charge_pulses])<max_dt_pulse]
        # discharge_pulse_end_idx = discharge_pulse_end_idx0[(discharge_pulse_end_idx0-discharge_pulse_start_idx0[0:num_discharge_pulses])<max_dt_pulse]

        idxi1 = np.where((np.diff(I)>delta_I) & (I[1:]>delta_I))[0] # increasing current and positive (start of charge pulse)
        idxi1 = np.delete(idxi1, np.where(abs(I.iloc[idxi1])>delta_I)) # should be near zero
        idxi2 = np.where((np.diff(I)<-delta_I) & (I[1:]<-delta_I))[0] # decreasing current and negative (start of discharge pulse)
        # idxi1 = charge_pulse_start_idx # increasing current and positive (start of charge pulse)
        # idxi2 = discharge_pulse_start_idx # decreasing current and negative (start of discharge pulse)
        idxi = np.concatenate([idxi1,idxi2])
        idxi = idxi + 1
        idxk1 = np.where((np.diff(I)<-delta_I) & (I[:-1]>delta_I))[0] # decreasing current and positive (end of charge pulse)
        idxk2 = np.where((np.diff(I)>delta_I) & (I[:-1]<-delta_I))[0] # increasing current and negative (end of discharge pulse)
        # idxk1 = charge_pulse_end_idx # decreasing current and positive (end of charge pulse)
        # idxk2 = discharge_pulse_end_idx # increasing current and negative (end of discharge pulse)
        idxk = np.concatenate([idxk1,idxk2])
        no_pulses = min(len(idxi),len(idxk))

        r1, r2, qr, pcur, pdur = [], [], [], [], []
        for pno in range(no_pulses):
            t1, V1, I1 = t[idxi[pno]-1-pts:idxi[pno]-1], V[idxi[pno]-1-pts:idxi[pno]-1], I[idxi[pno]-1-pts:idxi[pno]-1] # pts right efore the pulse starts
            t2, V2, I2 = t[idxi[pno]:idxi[pno]+pts], V[idxi[pno]:idxi[pno]+pts], I[idxi[pno]:idxi[pno]+pts] # pts right after the pulse starts (for Rs)
            t3, V3, I3 = t[idxk[pno]+1-pts:idxk[pno]+1], V[idxk[pno]+1-pts:idxk[pno]+1], I[idxk[pno]+1-pts:idxk[pno]+1] # pts rigth before the pulse ends
            if len(t1) < 4 or len(t2) < 4 or len(t3) < 4:
                continue
            r_p1 = abs((np.average(V2) - np.average(V1)) / (np.average(I2) - np.average(I1)))
            r_p2 = abs((np.average(V3) - np.average(V1)) / (np.average(I3) - np.average(I1)))
            delta_q=np.average(Q[idxk[pno]+1-pts:idxk[pno]+1])-np.average(Q[idxi[pno]-1-pts:idxi[pno]-1])
            q_val = np.average(Q[idxi[pno]-1-pts:idxi[pno]-1])
            r1.append(round(r_p1, 4))
            r2.append(round(r_p2, 4))
            qr.append(q_val)
            pcur.append(round(np.average(I2),3))
            pdur.append(round(np.average(t3)-np.average(t1),3))
        df =  pd.DataFrame({'pulse_current': pcur, 'pulse_duration': pdur, 'Q': qr, 'R_s': r1, 'R_l': r2})
        return df
   
    def _process_cycler_expansion(self, cell_data_vdf, cell_cycle_metrics, calibration_parameters, numFiles = 1000, t_match_threshold=60000):
        # Combine vdf data into a single df
        # cell_data_vdf = self._combine_cycler_expansion(records_vdf, calibration_parameters, numFiles)
        
        # Find matching cycle timestamps from cycler data
        t_vdf = cell_data_vdf['Time [ms]']
        exp_vdf = cell_data_vdf['Expansion [-]']
        exp_vdf_um = cell_data_vdf['Expansion [um]']
        cycle_timestamps = cell_cycle_metrics['Time [ms]'][cell_cycle_metrics.cycle_indicator==True]
        t_cycle_vdf, cycle_idx_vdf, matched_timestamp_indices = self._find_matching_timestamp(cycle_timestamps, t_vdf, t_match_threshold=10000)  

        # add cycle indicator. These should align with cycles timestamps previously defined by cycler data
        cell_data_vdf['cycle_indicator'] = pd.array([False]*len(cell_data_vdf))
        cell_data_vdf['cycle_indicator'].iloc[cycle_idx_vdf] = True

        # filter by rate of expansion 
        # x,y1 = reject_outliers(x0,y0, m=reject_outliers_std)
        # nonoutlier_idx = np.where(abs(np.diff(y0) - np.mean(np.diff(y0))) < reject_outliers_std * np.std(np.diff(y0)))[0]
        # x = np.array(x0)[nonoutlier_idx]
        # y1 = np.array(np.diff(y0))[nonoutlier_idx]
        # y = [y-min(y1) for y in y1]

        # find min/max expansion
        cycle_idx_vdf_minmax = [i for i in cycle_idx_vdf if i is not np.nan]
        cycle_idx_vdf_minmax.append(len(t_vdf)-1) #append end
        exp_max, exp_min = self._max_min_cycle_data(exp_vdf, cycle_idx_vdf_minmax)
        exp_rev = np.subtract(exp_max,exp_min)
        exp_max_um, exp_min_um = self._max_min_cycle_data(exp_vdf_um, cycle_idx_vdf_minmax)
        exp_rev_um = np.subtract(exp_max_um,exp_min_um)

        # save data to dataframe: initialize with nan and fill in timestamp-matched values
        discharge_cycle_idx = list(np.where(cell_cycle_metrics.cycle_indicator==True)[0])
        cell_cycle_metrics['Time vdf [s]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Min cycle expansion [-]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Max cycle expansion [-]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Reversible cycle expansion [-]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Min cycle expansion [um]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Max cycle expansion [um]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Reversible cycle expansion [um]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Drive Current [-]']= [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Expansion STDDEV [cnt]']= [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Ref STDDEV [cnt]']= [np.nan]*len(cell_cycle_metrics)


        for i,j in enumerate(matched_timestamp_indices):
            cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Time vdf [s]'] = t_cycle_vdf[i]
            cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Min cycle expansion [-]'] = exp_min[i]
            cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Max cycle expansion [-]'] = exp_max[i]
            cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Reversible cycle expansion [-]'] = exp_rev[i]
            cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Min cycle expansion [um]'] = exp_min_um[i]
            cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Max cycle expansion [um]'] = exp_max_um[i]
            cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Reversible cycle expansion [um]'] = exp_rev_um[i]
            # try:
            #     tmpa=cell_data_vdf['Drive Current [-]']
            #     cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Drive Current [-]'] = tmpa[i]
            # except:
            #     cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Drive Current [-]'] = 0
            try:    
                tmpb=cell_data_vdf['Expansion STDDEV [cnt]']
                cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Expansion STDDEV [cnt]'] = tmpb[i]
            except:
                cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Expansion STDDEV [cnt]'] = 0

            try:
                tmpc=cell_data_vdf['Ref STDDEV [cnt]']
                cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Ref STDDEV [cnt]'] = tmpc[i]
            except:
                cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Ref STDDEV [cnt]'] = 0
            
        # also add timestamps for charge cycles
        charge_cycle_idx = list(np.where(cell_cycle_metrics.charge_cycle_indicator==True)[0])
        charge_cycle_timestamps = cell_cycle_metrics['Time [ms]'][cell_cycle_metrics.charge_cycle_indicator==True]
        t_charge_cycle_vdf, charge_cycle_idx_vdf, matched_charge_timestamp_indices = self._find_matching_timestamp(charge_cycle_timestamps, t_vdf, t_match_threshold=10000)
        for i,j in enumerate(matched_charge_timestamp_indices):
            cell_cycle_metrics.loc[charge_cycle_idx[j], 'Time vdf [s]'] = t_charge_cycle_vdf[i]

        return cell_data_vdf, cell_cycle_metrics

    def _get_calibration_parameters(self, df_vdf, dev_name, calibration_parameters):
        """
        Get the calibration parameters for the device

        Parameters
        ----------
        df_vdf: DataFrame
            The dataframe of the vdf data
        dev_name: str
            The device name
        calibration_parameters: dict
            The dictionary of the calibration parameters
        
        Returns
        -------
        df_vdf: DataFrame
            The dataframe of the vdf data with the calibration parameters

            
        """
        df_vdf['x1'] = X1  # Default values
        df_vdf['x2'] = X2
        df_vdf['c'] = C
        
        if dev_name in calibration_parameters:
            for start_date, removal_date, x1, x2, c in calibration_parameters[dev_name]:
                if not start_date:
                    start_date = "01/01/2000"    
                if not removal_date:
                    removal_date = "01/01/2100"
                start_date = self.dateConverter._str_to_timestamp(self.dateConverter._format_date_str(start_date))
                removal_date = self.dateConverter._str_to_timestamp(self.dateConverter._format_date_str(removal_date))
                mask = (df_vdf['Time [ms]'] >= start_date) & (df_vdf['Time [ms]'] <= removal_date)
                df_vdf.loc[mask, ['x1', 'x2', 'c']] = x1, x2, c
                
        return df_vdf

    def _combine_cycler_expansion(self, records_vdf, calibration_parameters, numFiles = 1000):
        """
        PROCESS NEWARE VDF DATA
        Reads in data from the last "numFiles" files in the "records_vdf" and concatenates them into a long dataframe. Then looks for corresponding cycle start/end timestamps in vdf time. 
        Finally, it'll calculate the min, max, and reversible expansion for each cycle.  
        """
        self.logger.debug(f"Processing {len(records_vdf)} vdf files")
        # concatenate vdf data frames for last numFiles files
        frames_vdf =[]
        # For each vdf file...
        for record_vdf in records_vdf[0:min(len(records_vdf), numFiles)]:
            try:
                # Read in timeseries data from test and formating into dataframe. Remove rows with expansion value outliers.
                self.logger.debug(f"Now Processing {record_vdf['tr_name']}")
                # df_vdf = test2df(test_vdf, test_trace_keys = ['aux_vdf_timestamp_datetime_0','aux_vdf_ldcsensor_none_0', 'aux_vdf_ldcref_none_0', 'aux_vdf_ambienttemperature_celsius_0', 'aux_vdf_temperature_celsius_0'], df_labels =['Time [ms]','Expansion [-]', 'Expansion ref [-]', 'Amb Temp [degC]', 'Temperature [degC]'])
                df_vdf = self._record_to_df(record_vdf, test_trace_keys = ['aux_vdf_timestamp_epoch_0','aux_vdf_ldcsensor_none_0', 'aux_vdf_ldcref_none_0', 'aux_vdf_ambienttemperature_celsius_0','aux_vdf_ldcstd_none_0','aux_vdf_refstd_none_0', 'aux_vdf_drivecurrent_none_0'], df_labels =['Time [ms]','Raw expansion [-]', 'Expansion ref [-]','Temperature [degC]','Expansion STDDEV [cnt]','Ref STDDEV [cnt]','Drive Current [-]'])
                
                # Filter data to reduce sudden spikes in data
                df_vdf = df_vdf[(df_vdf['Raw expansion [-]'] >1e5) & (df_vdf['Raw expansion [-]'] <1e7)].copy() #keep good signals.
                fs = 1/5 # sampling frequency
                fc = 1/(60*60) # cutoff frequency
                sos = butter(4, fc/(fs/2), 'low', output='sos') #N,Wn
                df_vdf['Expansion [-]'] = sosfiltfilt(sos, df_vdf['Raw expansion [-]'])

                # Add LDC sensor calibration to df_vdf
                df_vdf = self._get_calibration_parameters(df_vdf, record_vdf['dev_name'], calibration_parameters)
                self.logger.info(f"Using calibration parameters for the entire dataframe.")
                df_vdf['Raw expansion [um]'] = 1000 * (30.6 - (df_vdf['x2'] * (df_vdf['Raw expansion [-]'] / 10**6)**2 + df_vdf['x1'] * (df_vdf['Raw expansion [-]'] / 10**6) + df_vdf['c']))
                df_vdf['Expansion [um]'] = 1000 * (30.6 - (df_vdf['x2'] * (df_vdf['Expansion [-]'] / 10**6)**2 + df_vdf['x1'] * (df_vdf['Expansion [-]'] / 10**6) + df_vdf['c']))
                df_vdf['Temperature [degC]'] = np.where((df_vdf['Temperature [degC]'] >= 200) & (df_vdf['Temperature [degC]'] <250), np.nan, df_vdf['Temperature [degC]']) 
                # df_vdf['Amb Temp [degC]'] = np.where((df_vdf['Amb Temp [degC]'] >= 200) & (df_vdf['Amb Temp [degC]'] <250), np.nan, df_vdf['Amb Temp [degC]']) 
                frames_vdf.append(df_vdf)
                self.logger.debug(f"Finished processing with {len(frames_vdf)} data points")
            except Exception as e:
                self.logger.error(f"Error processing {record_vdf['tr_name']}: {e}")
                continue
          #  time.sleep(0.1) 
        
        if (len(frames_vdf) == 0):
            self.logger.debug(f"No vdf data found")
            cell_data_vdf = self._create_default_cell_data_vdf()
            return cell_data_vdf
        # Combine vdf data into a single df and reset the index 
        cell_data_vdf = pd.concat(frames_vdf).sort_values(by=['Time [ms]'])
        cell_data_vdf.reset_index(drop=True, inplace=True)
        return cell_data_vdf
    
    def _create_default_cell_data(self):
        return pd.DataFrame(columns=['Time [ms]','Current [A]', 'Voltage [V]', 'Ah throughput [A.h]', 'Temperature [degC]','cycle_indicator', 'discharge_cycle_indicator', 'charge_cycle_indicator', 'capacity_check_indicator'])
    def _create_default_cell_cycle_metrics(self):
        return pd.DataFrame(columns=['Time [ms]','Ah throughput [A.h]', 'Test type','Protocol','discharge_cycle_indicator','cycle_indicator','charge_cycle_indicator','capacity_check_indicator', 'Test name','Drive Current [-]','Expansion STDDEV [cnt]','Ref STDDEV [cnt]'])
    def _create_default_cell_data_vdf(self):
        return pd.DataFrame(columns=['Time [ms]','Raw expansion [-]', 'Expansion [-]', 'Expansion ref [-]', 'Temperature [degC]','cycle_indicator','Drive Current [-]','Expansion STDDEV [cnt]','Ref STDDEV [cnt]'])

    def _process_cycler_data(self, cell_data, cycle_id_lims, project_name):
        """
        Process cycler data from a list of test records

        Parameters
        ----------
        records: list of dict
            The list of test records
        cycle_id_lims: list of ints
            The cycle number limits for charge, discharge, and total cycles
        project_name: str
            The project name for the cell 
        numFiles: int, optional
            The max number of files to process

        Returns
        -------
        DataFrame
            The processed data
        DataFrame
            The processed cycle metrics
        """
        # calculate capacities 
        if project_name in PROJECT.keys(): 
            Qmax = PROJECT[project_name]['Qmax']
        else:
            Qmax = PROJECT['DEFAULT']['Qmax']

        # combine data for all files 
        self.logger.info(f"Processing cycler data")
        cell_data, cell_cycle_metrics = self._combine_cycler_data(cell_data, cycle_id_lims, project_name = project_name)

        charge_t_idx = list(cell_data[cell_data.charge_cycle_indicator ==True].index)
        discharge_t_idx = list(cell_data[cell_data.discharge_cycle_indicator ==True].index)
        Q_c, Q_d = self._calc_capacities(cell_data['Time [ms]'].values, cell_data['Current [A]'].values,cell_data['Voltage [V]'].values, cell_data['Ah throughput [A.h]'].values, charge_t_idx, discharge_t_idx, Qmax)
        # find average current
        I_avg_c,I_avg_d = self._avg_cycle_data_x(cell_data['Time [ms]'], cell_data['Current [A]'], charge_t_idx, discharge_t_idx)
        # Find min/max metrics
        cycle_idx_minmax = list(cell_data[cell_data.cycle_indicator ==True].index)
        cycle_idx_minmax.append(len(cell_data)-1)
        V_max, V_min = self._max_min_cycle_data(cell_data['Voltage [V]'], cycle_idx_minmax)
        # If cycler data doesn't have temperature measurements
        if 'Temperature [degC]' in cell_data.keys():
            T_max, T_min = self._max_min_cycle_data(cell_data['Temperature [degC]'], cycle_idx_minmax)
        else:
            T_max, T_min = [np.nan]*len(cell_cycle_metrics), [np.nan]*len(cell_cycle_metrics)

        cell_cycle_metrics['Charge capacity [A.h]'] = [np.nan]*len(cell_cycle_metrics) # init capacity columns in cell_cycle_metrics
        cell_cycle_metrics['Discharge capacity [A.h]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Min cycle voltage [V]'] = [np.nan]*len(cell_cycle_metrics) # init capacity columns in cell_cycle_metrics
        cell_cycle_metrics['Max cycle voltage [V]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Min cycle temperature [degC]'] = [np.nan]*len(cell_cycle_metrics) # init capacity columns in cell_cycle_metrics
        cell_cycle_metrics['Max cycle temperature [degC]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Avg Charge cycle current [A]'] = [np.nan]*len(cell_cycle_metrics)
        cell_cycle_metrics['Avg Dis-Charge cycle current [A]'] = [np.nan]*len(cell_cycle_metrics) 
       
        # Add to dataframe
        charge_cycle_number = list(cell_cycle_metrics[cell_cycle_metrics.charge_cycle_indicator ==True].index) # aligns with charge start
        discharge_cycle_number = list(cell_cycle_metrics[cell_cycle_metrics.discharge_cycle_indicator ==True].index) # aligns with discharge start
        cycle_number = list(cell_cycle_metrics[cell_cycle_metrics.cycle_indicator ==True].index) # align with charge start
        for i,j in enumerate(charge_cycle_number): 
            cell_cycle_metrics.loc[j, 'Charge capacity [A.h]'] = Q_c[i]
            cell_cycle_metrics.loc[j, 'Avg Charge cycle current [A]'] = I_avg_c[i]  
        for i,j in enumerate(discharge_cycle_number): 
            cell_cycle_metrics.loc[j, 'Discharge capacity [A.h]'] = Q_d[i]
            cell_cycle_metrics.loc[j, 'Avg Dis-Charge cycle current [A]'] = I_avg_d[i]  
        for i,j in enumerate(cycle_number): 
            cell_cycle_metrics.loc[j, 'Min cycle voltage [V]'] = V_min[i] 
            cell_cycle_metrics.loc[j, 'Max cycle voltage [V]'] = V_max[i] 
            cell_cycle_metrics.loc[j, 'Min cycle temperature [degC]'] = T_min[i] 
            cell_cycle_metrics.loc[j, 'Max cycle temperature [degC]'] = T_max[i] 
        return cell_data, cell_cycle_metrics

    def _avg_cycle_data_x(self,t, data, charge_idx, discharge_idx):
        # calculate avg data for each cycle (e.g. voltage, temperature, or expansion)
        # Modified Min/Max by CES
        cycle_idx = charge_idx + discharge_idx
        cycle_idx.append(len(t)-1) # add last data point
        cycle_idx.sort() # should alternate charge and discharge start indices

        y_avg_c  = []
        y_avg_d  = []
        # for each cycle...
        # Calculate capacity 
        for i in range(len(cycle_idx)-1):
            # Calculate capacity based on AhT.
            Dt_total= t[cycle_idx[i+1]]-t[cycle_idx[i]]
            if(Dt_total>0):
                #Dt_elements=t[(cycle_idx[i]:cycle_idx[i+1])+1]-t[cycle_idx[i]:cycle_idx[i+1]]
                y_avg =np.trapz(data[cycle_idx[i]:cycle_idx[i+1]],x=t[cycle_idx[i]:cycle_idx[i+1]])/Dt_total #np.sum( Dt_elements*(data[cycle_idx[i]:cycle_idx[i+1]]+data[cycle_idx[i]-1:cycle_idx[i+1]-1])/2)/Dt_total
            else:
                y_avg=0  
                
            if cycle_idx[i] in charge_idx:

                y_avg_c.append(y_avg) 
            else:
                #y_avg = 0
                y_avg_d.append(y_avg) 
        return np.array(y_avg_c), np.array(y_avg_d)
        
    def _max_min_cycle_data(self, data, cycle_idx_minmax):
        """
        Get the max and min data for each cycle

        Parameters
        ----------
        data: list of floats
            The data to be processed
        cycle_idx_minmax: list of ints
            The list of cycle indices

        Returns
        -------
        list of floats
            The max data for each cycle
        list of floats
            The min data for each cycle
        """

        # calculate min and max data for each cycle (e.g. voltage, temperature, or expansion)
        y_max  = []
        y_min  = []
        # for each cycle...
        for i in range(len(cycle_idx_minmax)-1):
            # if there's cycle data between two consecutive points...
            if len(data[cycle_idx_minmax[i]:cycle_idx_minmax[i+1]])>0:
                y_max.append(max(data[cycle_idx_minmax[i]:cycle_idx_minmax[i+1]]))
                y_min.append(min(data[cycle_idx_minmax[i]:cycle_idx_minmax[i+1]]))
            # handling edge cases
            else: 
                y_max.append(data[cycle_idx_minmax[i]])   
                y_min.append(data[cycle_idx_minmax[i]])  
        return y_max, y_min

    def _calc_capacities(self, t, I, V, AhT, charge_idx, discharge_idx, Qmax):
        """
        Calculate the charge and discharge capacities

        Parameters
        ----------
        t: list of floats
            The time data
        I: list of floats
            The current data
        AhT: list of floats
            The Ah throughput data
        charge_idx: list of ints
            The list of charge indices
        discharge_idx: list of ints
            The list of discharge indices

        Returns
        -------
        list of floats
            The charge capacities
        """
        # TODO: I is not used, maybe we should use it to calculate the capacity?
        # combine charge and discharge idx into a list. assumes there are the same length, and alternate charge-discharge (or vice versa)
        cycle_idx = charge_idx + discharge_idx
        Q_c = []
        Q_d = []

        if len(cycle_idx)>0:
            if max(cycle_idx)<len(t)-1:
                cycle_idx.append(len(t)-1) # add last data point
            cycle_idx.sort() # should alternate charge and discharge start indices
            
            # Calculate capacity 
            for i in range(len(cycle_idx)-1):
                # Calculate capacity based on AhT.
                Q = AhT[cycle_idx[i+1]]-AhT[cycle_idx[i]]
                t_cycle = t[cycle_idx[i]:cycle_idx[i+1]]
                I_cycle = I[cycle_idx[i]:cycle_idx[i+1]]
                V_cycle = V[cycle_idx[i]:cycle_idx[i+1]] # debugging
                # check for large gaps in the data (dt>100s) and reset time.
                # if(Q>Qmax):
                if (np.argwhere(abs(np.diff(t_cycle))>1e5).size >0):
                    self.logger.warning(f'Fixing large dt gap and recalculating capacity for cycle {i}.')
                    while True:
                        dt = np.diff(t_cycle)
                        dt=np.append(dt,0)
                        gap_index= np.argwhere(abs(dt)>1e5)#look for gaps greater than 100 s
                        # remove dt gaps and reintegrate
                        if (gap_index.size >0):
                            # if gap_index[0][0]>0:
                            for gap in gap_index:
                                g = gap[0]
                                dt[g] = min(dt[g-1], dt[g+1])
                            t_cycle = t_cycle[0] + np.cumsum(dt)
                        else:
                            break
                    
                    Q=abs(integrate.trapz(I_cycle, (t_cycle-t_cycle[0])/1000)/3600)
                    if(Q>Qmax): 
                        Q = np.nan
                        self.logger.warning(f"Invalid Capacity for cycle {i} with coulomb counting.")
                if cycle_idx[i] in charge_idx:
                    Q_c.append(Q) 
                else:
                    Q_d.append(Q) 
        return np.array(Q_c), np.array(Q_d)

    def _combine_cycler_data(self, cell_data, cycle_id_lims, last_AhT = 0, Qmax=3.8, project_name='DEFAULT', min_V_max_C20 = 4.19):
        """
        Combine cycler data from multiple files into a single dataframe.
        PROCESS CYCLER DATA.
        Concatenate neware data frames, identify cycles based on rests (I=0), then calculate min and max voltage and temperature for each cycle.
        If you only want the latest test, set numFiles = 1. Assumes test files in records_cycler are sorted with records_cycler[0] being the latest.

        Parameters
        ----------
        records_cycler: list of dict
            The list of test records
        cycle_id_lims: dict
            Dictionary of cycle identification thresholds for different test types.
        numFiles: int, optional
            Number of files to process. Default is 1000.
        last_AhT: float, optional   
            Ah throughput from last file. Default is 0.
        
        Returns
        -------
        cell_data: dataframe
            Dataframe of all cycler data from the specified files.
        cell_cycle_metrics: dataframe
            Dataframe of cycle metrics from the specified files.
        """

        # Project settings to filter out bad cycle ID
        if project_name in PROJECT.keys(): 
            Q_max = PROJECT[project_name]['Qmax']
            Q_min = PROJECT[project_name]['Qmin']
        else:
            Q_max = PROJECT['DEFAULT']['Qmax']
            Q_min = PROJECT[project_name]['Qmin']

        # Appends data from each data file to a dataframe
        # frames =[]
        test_types = list(cycle_id_lims.keys())

        # For each data file...
        t = cell_data['Time [ms]']
        t_test = cell_data['Test Time [ms]']
        I = cell_data['Current [A]']
        V = cell_data['Voltage [V]']
        # T = cell_data['Temperature [degC]']
        step_idx = cell_data['Step index']

        # Recalculate AhT: remove dAh gaps larger than dAh_max and reintegrate
        AhT = cell_data['Ah throughput [A.h]'].reset_index(drop=True)
        # AhT=integrate.cumtrapz(abs(I), t,initial=0)/3600/1000 # ms to hours 
        # AhT=AhT-AhT.min()
        dAh = np.diff(AhT)
        dAh=np.append(dAh,0)
        gap_index= np.argwhere(abs(dAh)>Q_max)
        if (gap_index.size >0):
            for gap in gap_index:
                g = gap[0]
                dAh[g] = dAh[g + (np.where(abs(dAh[g::])<Q_max)[0][0]+1)] #dAh[g+1]
        AhT = np.cumsum(dAh)
        cell_data['Ah throughput [A.h]'] = AhT
        Ah_Discharge = cell_data['Discharge Ah throughput [A.h]'].reset_index(drop=True)
        Ah_Charge = cell_data['Charge Ah throughput [A.h]'].reset_index(drop=True)


        # check that indices are consistent    #update sidegeljb 12/20/2023  to use new signals (TODO)
        # indices_to_check = [t, I, V, AhT, step_idx]
        # for i in range(len(indices_to_check)-1):
            # assert indices_to_check[i].index.equals(indices_to_check[i+1].index), f"Indices are not consistent between columns in the data from {record['tr_name']}"
        # lengths_to_check = [len(t), len(I), len(V), len(AhT), len(step_idx)]
        # assert len(set(lengths_to_check)) == 1, f"Inconsistent data lengths in the data from {record['tr_name']}"

        # 4. Change cycle filtering thresholds by test type and include the idx at the end of the file in case cell is still cycling.
        # Search for test type in test name. If there's no match, use the default settings 
        lims={}
        lims = cycle_id_lims['CYC']
        test_protocol = 'CYC'
        V_max_cycle = lims['V_max_cycle']
        V_min_cycle = lims['V_min_cycle']
        dAh_min = lims['dAh_min']
        dt_min = lims['dt_min']

        # 5. Find indices for cycles in file
        if False:#isFormation and 'arbin' in record['tags']: # find peaks in voltage where I==0, ignore min during hppc
            peak_prominence = 0.1
            trough_prominence = 0.1
            discharge_start_idx_file, _ = find_peaks(medfilt(V[I==0], kernel_size = 101),prominence = peak_prominence)
            discharge_start_idx_file = cell_data[I==0].iloc[discharge_start_idx_file].index.to_list()
            charge_start_idx_file,_ = find_peaks(-medfilt(V[I==0], kernel_size = 101),prominence = trough_prominence, height = (None, -2.7)) # height to ignore min during hppc
            charge_start_idx_file = cell_data[I==0].iloc[charge_start_idx_file].index.to_list()
            charge_start_idx_file.insert(0, 0)
            charge_start_idx_file.insert(len(charge_start_idx_file), len(V)-1)
            charge_start_idx_file, discharge_start_idx_file = self._match_charge_discharge(np.array(charge_start_idx_file), np.array(discharge_start_idx_file))

        # if  'neware_xls_4000' in record['tags']:
        #     discharge_start_idx_file=np.where(np.diff(cycle_idx).astype(bool))

        # if  'arbin' in record['tags']:
        #     discharge_start_idx_file=np.where(np.diff(cycle_idx).astype(bool))

        else: # find I==0 and filter out irrelevant points
            charge_start_idx_file0, discharge_start_idx_file0 = self._find_cycle_idx(t, I, V, AhT, dAh_max=Q_max) #Ah_Discharge,Ah_Charge, step_idx, test_protocol, V_max_cycle = V_max_cycle, V_min_cycle = V_min_cycle, dt_min = dt_min, dAh_min= dAh_min,
            cycle_idx0 = np.sort(np.concatenate((charge_start_idx_file0, discharge_start_idx_file0)))
            charge_start_idx_file0, discharge_start_idx_file0 = self._filter_cycle_idx(cycle_idx0, t/1000, I, V, AhT, V_max_cycle=4, V_min_cycle=4, dt_min = 600, dAh_min=1) # doesn't work for 2C hot 152041
            try: # won't work for half cycles (files with only charge or only discharge)
                charge_start_idx_file, discharge_start_idx_file = self._match_charge_discharge(charge_start_idx_file0, discharge_start_idx_file0)
            except:
                # self.logger.error(f"Error processing {record['tr_name']}: failed to _match_charge_discharge")
                pass

        # 6. Add aux cycle indicators to df. Column of True if start of a cycle, otherwise False. Set default cycle indicator = charge start 
        # file_with_capacity_check =  False #isRPT or isFormation 
        cell_data['discharge_cycle_indicator'] = [False]*len(cell_data)
        cell_data['charge_cycle_indicator'] = [False]*len(cell_data)
        cell_data['capacity_check_indicator'] = [False]*len(cell_data)

        cell_data.loc[discharge_start_idx_file, 'discharge_cycle_indicator'] = True
        cell_data.loc[charge_start_idx_file, 'charge_cycle_indicator'] = True
        cell_data['cycle_indicator'] = cell_data['charge_cycle_indicator'] # default cycle = charge start 
        # if file_with_capacity_check:
        #     cell_data.loc[charge_start_idx_file, 'capacity_check_indicator'] = True

        # 6a. Add test type and test name to cell_data
        # cell_data['Test type'] = [' ']*len(cell_data)
        # cell_data['Test name'] = [' ']*len(cell_data)
        cell_data.loc[np.concatenate((discharge_start_idx_file,charge_start_idx_file)), 'Test type'] = test_protocol
        # cell_data.loc[np.concatenate((discharge_start_idx_file,charge_start_idx_file)), 'Test name'] = record['tr_name']

        # 6b. identify subcycle type. For extracting HPPC and C/20 dis/charge data later. 
        cell_data['Protocol'] = [np.nan]*len(cell_data)
        file_cell_cycle_metrics = cell_data[(cell_data.charge_cycle_indicator==True) | (cell_data.discharge_cycle_indicator==True)]

        for i in range(0,len(file_cell_cycle_metrics)):
            t_start = file_cell_cycle_metrics['Time [ms]'].iloc[i]
            if i == len(file_cell_cycle_metrics)-1: # if last subcycle, end of subcycle = end of file 
                t_end = cell_data['Time [ms]'].iloc[-1]
            else: # end of subcycle = start of next subcycle
                t_end = file_cell_cycle_metrics['Time [ms]'].iloc[i+1]
            # find valid indices. Some arbin cycles have backwards time jumps mid-cycler file (e.g.152062,152067)
            subcycle_idx = (t>=t_start) & (t<=t_end)
            if sum(subcycle_idx)>0:
                I_subcycle = cell_data['Current [A]'][subcycle_idx]
                V_subcycle = cell_data['Voltage [V]'][subcycle_idx]
                Ah_subcycle = cell_data['Ah throughput [A.h]'][subcycle_idx]
                t_subcycle = cell_data['Time [ms]'][subcycle_idx].values
                # check for large gaps in the data, and reset time.
                while True:
                    dt = np.diff(t_subcycle)
                    dt=np.append(dt,0)
                    gap_index= np.argwhere(abs(dt)>1e5)#look for gaps greater than 100 s
                    # remove dt gaps larger than dAh_max and reintegrate
                    if (gap_index.size >0):
                        # if gap_index[0][0]>0:
                        for gap in gap_index:
                            g = gap[0]
                            dt[g] = min(dt[g-1], dt[g+1])
                        t_subcycle = t_start + np.cumsum(dt)
                        t_end = t_subcycle[-1]
                    else:
                        break
                Q_subcycle = abs(integrate.trapz(I_subcycle, (t_subcycle-t_subcycle[0])/1000)/3600) # time not in ms anymore?
                dV = np.diff(V_subcycle) # used to check for C/20 charge restart
                dV = np.append(dV, 0)

                data_idx = file_cell_cycle_metrics.index.tolist()[i]
                # if file_with_capacity_check and Q_subcycle>Q_min and Q_subcycle<Q_max:
                #     if len(np.where(np.diff(np.sign(I_subcycle)))[0])>10: # hppc: ID by # of types of current sign changes (threshold is arbitrary)
                #         cell_data.loc[data_idx,'Protocol'] = 'HPPC'
                #     elif (t_end-t_start)/3600/1000 >8 and  np.mean(I_subcycle) > 0: #and  np.mean(I_subcycle) < Qmax / 18: # C/20 charge: longer than 8 hrs and mean(I)>0. Will ID C/10 during formation as C/20...
                #         cell_data.loc[data_idx,'Protocol'] = 'C/20 Charge'
                #     elif (t_end-t_start)/3600/1000 > 8 and  np.mean(I_subcycle) < 0:# and  np.mean(I_subcycle) > - Qmax / 18 : # C/20 discharge: longer than 8 hrs and mean(I)<0.Will ID C/10 during formation as C/20...
                #             cell_data.loc[data_idx,'Protocol'] = 'C/20 Discharge'
                # id arbin rpt cycles
                if Q_subcycle>Q_min and Q_subcycle<Q_max:
                    if (len(np.where(np.diff(np.sign(I_subcycle)))[0])>10) and (min(V_subcycle)>1): #or (len(np.where(abs(np.diff(I_subcycle))>2*2)[0])>10)?? hppc: ID by # of types of current sign changes (threshold is arbitrary)
                        cell_data.loc[data_idx,'Protocol'] = 'HPPC'
                    elif (t_end-t_start)/1000/3600.0 >8 and  (np.mean(I_subcycle) > 0) and (np.median(I_subcycle) > 0) and ~(((dV> 0.04) & (V_subcycle>3.5)).any()) and max(V_subcycle)>min_V_max_C20: #and (t_end-t_start): # and  np.mean(I_subcycle) < Qmax / 8: # C/20 charge: longer than 8 hrs and mean(I)>0. Will ID C/10 during formation as C/20... (C/18->c/8). Filter out c/20 charge restarts with (((dV> 0.04) & (V_subcycle>3.5)).any())
                        cell_data.loc[data_idx,'Protocol'] = 'C/20 Charge'
                    elif (t_end-t_start)/1000/3600.0 > 8 and  (np.mean(I_subcycle) < 0) and (np.median(I_subcycle) < 0): # and  np.mean(I_subcycle) > - Qmax / 8: # C/20 discharge: longer than 8 hrs and mean(I)<0.Will ID C/10 during formation as C/20...
                        cell_data.loc[data_idx,'Protocol'] = 'C/20 Discharge'
                    else:
                        pass
        cell_data.loc[((cell_data['Protocol'] == 'HPPC')| (cell_data['Protocol'] =='C/20 Charge') |( cell_data['Protocol'] =='C/20 Discharge')) & (cell_data['Test type'] != '_F'),'Test type'] = 'RPT'
        cell_data.loc[((cell_data['Protocol'] == 'HPPC')| (cell_data['Protocol'] =='C/20 Charge') |( cell_data['Protocol'] =='C/20 Discharge')), 'capacity_check_indicator'] = True #& (cell_data['Test type'] != '_F')

        # 7. Add to list of dfs where each element is the resulting df from each file.
        # self.logger.debug(record['tr_name'] + '   Cycles: ' + str(len(charge_start_idx_file)) + '   AhT: ' + str(round(AhT.iloc[-1],2)))
        self.logger.debug(f"cell_data: {cell_data}")
        
        # Combine cycling data into a single df and reset the index
        # self.logger.info(f"Combining {len(frames)} dataframes")
        if len(cell_data) == 0:
            cell_data, cell_cycle_metrics = self._create_default_cell_data(), self._create_default_cell_cycle_metrics()
            return cell_data, cell_cycle_metrics
        
        # cell_data = pd.concat(frames)
        cell_data.reset_index(drop=True, inplace=True)
        # Get cycle indices from combined df originally identified from individual tests (with lims based on test type) 
        discharge_start_idx_0 = np.array(list(compress(range(len(cell_data['discharge_cycle_indicator'])), cell_data['discharge_cycle_indicator'])))
        charge_start_idx_0 = np.array(list(compress(range(len(cell_data['charge_cycle_indicator'])), cell_data['charge_cycle_indicator'])))
        capacity_check_idx_0 = np.array(list(compress(range(len(cell_data['capacity_check_indicator'])), cell_data['capacity_check_indicator'])))       
        # Filter cycle indices again to match every discharge and charge index. Set default cycle index to charge start
        if len((discharge_start_idx_0)>1) and (len(charge_start_idx_0)>1):
            charge_start_idx, discharge_start_idx = self._match_charge_discharge(charge_start_idx_0, discharge_start_idx_0) 
            cycle_idx = charge_start_idx
            # Remove cycle indices that were filtered out
            removed_charge_cycle_idx = list(set(charge_start_idx_0).symmetric_difference(set(charge_start_idx)))
            cell_data.loc[removed_charge_cycle_idx,'charge_cycle_indicator'] = False
            removed_discharge_cycle_idx = list(set(discharge_start_idx_0).symmetric_difference(set(discharge_start_idx)))
            cell_data.loc[removed_discharge_cycle_idx,'discharge_cycle_indicator'] = False
            removed_capacity_check_idx = list(set(capacity_check_idx_0).symmetric_difference(set(charge_start_idx)))
            cell_data.loc[removed_capacity_check_idx,'capacity_check_indicator'] = False
        cell_data['cycle_indicator'] = cell_data.charge_cycle_indicator #default cycle indicator on charge

        # save cycle metrics to separate dataframe and sort. only keep columns where charge and discharge cycles start. Label the type of protocol
        cycle_metrics_columns = ['Time [ms]','Ah throughput [A.h]', 'Test type','Protocol','discharge_cycle_indicator','cycle_indicator','charge_cycle_indicator','capacity_check_indicator', 'Test name']
        cell_cycle_metrics = cell_data[cycle_metrics_columns][(cell_data.discharge_cycle_indicator==True) | (cell_data.charge_cycle_indicator==True)].copy()
        self.logger.info(f"Found {len(cell_data)} cell data")
        self.logger.info(f"Found {len(cell_cycle_metrics)} cycles")
        # cell_cycle_metrics.sort_values(by=['Time [ms]'])
        cell_cycle_metrics.reset_index(drop=True, inplace=True)
        return cell_data, cell_cycle_metrics

    def _record_to_df(self, record, test_trace_keys = DEFAULT_TRACE_KEYS, df_labels = DEFAULT_DF_LABELS, ms = False):
        """
        Filter and format data from a TestRecord object into a dataframe

        Parameters
        ----------
        record: dict
            The test record
        test_trace_keys: list of str, optional
            The list of test trace keys to be extracted
        df_labels: list of str, optional
            The list of labels for the return dataframe keys

        Returns
        -------
        DataFrame
            The processed dataframes
        """
        
        # Read in timeseries data from test and formating into dataframe
        df_raw = self.dataFilter.filter_df_by_record(record, trace_keys = test_trace_keys)
        if df_raw is None:
            self.logger.warning(f"Cannot find data for {record['tr_name']} with trace keys {test_trace_keys}")
            return None
        # preserve listed trace key order and rename columns for easy calling
        df = df_raw.set_axis(df_labels, axis=1)
        return df
    
    def _find_cycle_idx(self, t, I, V, AhT, dAh_max = 5): #Ah_Discharge,Ah_Charge,step_idx, test_protocol,V_max_cycle=3, V_min_cycle=4, dt_min = 600, dAh_min=1,
        """
        Find the cycle charge and discharge indices by calling filter_cycle_idx

        Parameters
        ----------
        t: list of floats
            The time data
        I: list of floats
            The current data
        V: list of floats
            The voltage data
        AhT: list of floats
            The Ah throughput data
        step_idx: list of ints
            The step indices
        V_max_cycle: float, optional
            The maximum voltage for a cycle
        V_min_cycle: float, optional
            The minimum voltage for a cycle
        dt_min: float, optional
            The minimum time for a cycle
        dAh_min: float, optional
            The minimum Ah throughput for a cycle

        Returns
        -------
        list of ints
            The list of charge indices
        list of ints
            The list of discharge indices
        """
        Ic=(I.values>1e-5).astype(int)
        Id=(I.values<-1e-5).astype(int)
        potential_charge_start_idx= np.where(np.diff(Ic)>0.2)[0]
        potential_discharge_start_idx=np.where(np.diff(Id)>0.2)[0]
        dt=np.diff(t)
        #Cumah=Ah_Charge-Ah_Discharge
        Cumah=integrate.cumtrapz(I, t,initial=0)/3600/1000 # ms to hours 
        # calculate the average discharge current and average time until the next charge step
        Cumah=Cumah-Cumah.min()
        # check for large gaps in the data, and reset the cumah counter.
        dAh = np.diff(Cumah)
        dAh=np.append(dAh,0)
        gap_index= np.argwhere(abs(dAh)>dAh_max)#np.argwhere(dt>1e6)# look for gaps greater than 1000 s

        # remove dAh gaps larger than dAh_max and reintegrate
        if (gap_index.size >0):
            # if gap_index[0][0]>0:
            for gap in gap_index:
                g = gap[0]
                dAh[g] = dAh[g+1]
                # Cumah[(gap[0]+1):]=Cumah[gap[0]]+Cumah[(gap[0]+1):]-Cumah[gap[0]+1]
        Cumah = np.cumsum(dAh)
                

        class_count = 10 # basically needs to change by more than 10% of the full range. (Increase to include more pts)
        class_range = Cumah.ptp()
        class_width = class_range / (class_count - 1)
        class_offset = Cumah.min() - class_width / 2

        try:
            res=rfcnt.rfc(
                Cumah,
                class_count=class_count,
                class_offset=class_offset,
                class_width=class_width,
                hysteresis=class_width,
                spread_damage=rfcnt.SDMethod.FULL_P2,           # assign damage for closed cycles to 2nd turning point
                residual_method=rfcnt.ResidualMethod._NO_FINALIZE,  # don't consider residues and leave internal sequence open
                wl={"sd": 1e3, "nd": 1e7, "k": 5})

            turning_points=res["tp"][:, 0].astype(int)-1
            cum_ah_at_turn=res["tp"][:, 1]
            #if(test_protocol == 'RPT'):
                #start with first charge assume its not the HPPC 

    #        find first turning point after the charge start index.
            last_index=0
            last_tp=0

            if len(turning_points)>2:
                if(cum_ah_at_turn[1]>cum_ah_at_turn[0]):
                    #2nd turn point is start of discharge.
                    charge_start_idx=np.array([min(potential_charge_start_idx, key=lambda x:abs(x-turning_points[0]))])                   
                    discharge_start_idx=np.array([min(potential_discharge_start_idx, key=lambda x:abs(x-turning_points[1]))])
                    last_tp=1
                    
                elif(cum_ah_at_turn[0]-class_offset>class_range/2) : # the first turning point is likely a start of discharge 
                    #charge_start_idx=np.array([potential_charge_start_idx[0]])
                    # Case of a partial cycle. so set the charge start to the start of the file....
                    charge_start_idx=np.array([0])
                    if( turning_points[0]>charge_start_idx[0]-10 ): # check that is comes after the first charge
                        discharge_start_idx=np.array([min(potential_discharge_start_idx, key=lambda x:abs(x-turning_points[0]))])
                        last_tp=0
                    else:
                        discharge_start_idx=np.array([min(potential_discharge_start_idx, key=lambda x:abs(x-turning_points[1]))])
                        last_tp=1
                        self.logger.info(f"choosing next turning point caveat empor.") 

                #elif(cum_ah_at_turn[1]-class_offset>class_range/2) : # the second turning point a start of discharge
                else:
                    charge_start_idx=np.array([potential_charge_start_idx[0]])
                    if (turning_points[1]>charge_start_idx[0]-100):
                        discharge_start_idx=np.array([min(potential_discharge_start_idx, key=lambda x:abs(x-turning_points[1]))])
                        last_tp=1

                # need to add the else case here in case we dont start with a charge cyccle.

                for ii in range(last_tp+1,len(turning_points)-1,2):
                #for pci in potential_charge_start_idx: #range(len(charge_start_idx))
                    charge_start_idx=np.append(charge_start_idx,min(potential_charge_start_idx, key=lambda x:abs(x-turning_points[ii])))
                    discharge_start_idx=np.append(discharge_start_idx, min(potential_discharge_start_idx, key=lambda x:abs(x-turning_points[ii+1])))
            elif len(turning_points)==1: # test: for calendar aging cells? -> picks up last hppc pulse before c/20 charge during rpt
                charge_start_idx=potential_charge_start_idx
                discharge_start_idx=potential_discharge_start_idx
            else:
                # no turning points in the data, just take the extents? this will probably breaksomething else...
                charge_start_idx=[]#np.array([0])
                discharge_start_idx=[]#np.array([len(t)-1])
                # find the next discharge
                #discharge_start_idx=np.array([potential_charge_start_idx[0]])
        except Exception as e:
            print(e)
            self.logger.info(f"No cycles detected (using the whole test).")    
            charge_start_idx=[]#np.array([0])
            discharge_start_idx=[]#np.array([len(t)-1])
        return charge_start_idx, discharge_start_idx

    def _filter_cycle_idx(self, cycle_idx0, t, I, V, AhT, V_max_cycle=3, V_min_cycle=4, dt_min = 600, dAh_min=1):
        """
        Filter the cycle indices based on the specified thresholds
        
        Parameters
        ----------
        cycle_idx0: list of ints
            The list of cycle indices
        t: list of floats
            The time data
        I: list of floats
            The current data
        V: list of floats
            The voltage data
        AhT: list of floats
            The Ah throughput data
        V_max_cycle: float, optional
            The maximum voltage for a cycle
        V_min_cycle: float, optional
            The minimum voltage for a cycle
        dt_min: float, optional
            The minimum time for a cycle
        dAh_min: float, optional
            The minimum Ah throughput for a cycle
        
        Returns
        -------
        list of ints
            The list of charge indices
        list of ints
            The list of discharge indices
        """
        # dt_check_1 = [i for i,dt in enumerate(np.diff(t[cycle_idx0])) if dt > dt_min]
        # dt_check_1.append(len(cycle_idx0)-1) #add end of file
        # dt_check = np.array(list(map(lambda x: x in dt_check_1, range(len(cycle_idx0)))))

        dt_check_0_forward = [i for i,dt in enumerate(np.diff(t[cycle_idx0])) if dt > dt_min]
        dt_check_0_backward = [len(cycle_idx0)-i for i,dt in enumerate(-np.diff(t[cycle_idx0[::-1]])) if dt > dt_min]
        dt_check_0_forward.append(len(cycle_idx0)-1)  #add end of file
        dt_check_0_backward.insert(0,0) #add beginning of file
        dt_check = np.array(list(map(lambda x: (x in dt_check_0_backward) or (x in dt_check_0_forward), range(len(cycle_idx0)))))

        dAh_check_0_forward = [i for i,dAh in enumerate(np.diff(AhT[cycle_idx0])) if dAh > dAh_min]
        dAh_check_0_backward = [len(cycle_idx0)-i for i,dAh in enumerate(-np.diff(AhT[cycle_idx0[::-1]])) if dAh > dAh_min]
        dAh_check_0_forward.append(len(cycle_idx0)-1)  #add end of file
        dAh_check_0_backward.insert(0,0) #add beginning of file
        dAh_check = np.array(list(map(lambda x: (x in dAh_check_0_backward) or (x in dAh_check_0_forward), range(len(cycle_idx0)))))

        # check that cycle start voltages are outside V(charge_start)<V_min and V(discharge_start)>V_max
        V_min_check = (V[cycle_idx0]<V_min_cycle).to_numpy()
        V_max_check = (V[cycle_idx0]>V_max_cycle).to_numpy()
        
        # combine checks 
        charge_start_idx = cycle_idx0[np.where(dt_check & dAh_check & V_min_check)[0]]
        discharge_start_idx = cycle_idx0[np.where(dt_check & dAh_check & V_max_check)[0]]

        return charge_start_idx, discharge_start_idx

    def _match_charge_discharge(self, charge_start_idx_0, discharge_start_idx_0):
        """
        Get the charge and discharge indices that match

        Parameters
        ----------
        charge_start_idx_0: list of ints
            The list of charge indices
        discharge_start_idx_0: list of ints
            The list of discharge indices
        
        Returns
        -------
        list of ints
            The list of charge indices that match
        list of ints
            The list of discharge indices that match
        """
        discharge_start_idx = []
        charge_start_idx = []
        if len(charge_start_idx_0) == 0 or len(discharge_start_idx_0) == 0:
            return charge_start_idx, discharge_start_idx
        # Filter out unnecessary cycle indices created when identifying cycles per filer. Length of charge and discharge start indices should be the same afterwards. 
        if discharge_start_idx_0[0]<charge_start_idx_0[0]: # if cycling starts on a discharge
            for i in range(len(charge_start_idx_0)):
                middle = charge_start_idx_0[i]
                left = discharge_start_idx_0[np.where(discharge_start_idx_0 < middle)[0][-1]]
                # right = discharge_start_idx_0[np.where(discharge_start_idx_0 > middle)[0][0]]
                discharge_start_idx.append(left)
                charge_start_idx.append(middle)
        else: # if cycling starts on a charge
            for i in range(len(discharge_start_idx_0)):
                middle = discharge_start_idx_0[i]
                left = charge_start_idx_0[np.where(charge_start_idx_0 < middle)[0][-1]]
                # right = charge_start_idx_0[np.where(charge_start_idx_0 > middle)[0][0]]
                charge_start_idx.append(left)
                discharge_start_idx.append(middle)
        return charge_start_idx, discharge_start_idx

    def _find_matching_timestamp(self, desired_timestamps, t, t_match_threshold=60, nan_pad = False):
        """
        Find the matching timestamps

        Parameters
        ----------
        desired_timestamps: list of floats
            The list of desired timestamps
        t: floats
            The time data
        t_match_threshold: float, optional
            The threshold for matching timestamps in seconds
        nan_pad: bool, optional
            Whether to pad with nan
        
        Returns
        -------
        list of floats
            The list of matching timestamps
        list of ints
            The list of mapped indices
        list of ints
            The list of matched timestamp indices
        """
        # find matched_timestamps by use timestamps as series index to use "get_indexer"
        t_test = t.drop_duplicates()
        t_test = pd.Series(t_test, index = t_test)
        mapped_indices_uniq = t_test.index.get_indexer(desired_timestamps, method="nearest", tolerance = t_match_threshold*1000)
        matched_timestamp_indices = np.argwhere(mapped_indices_uniq!=-1)[:,0] #indices of mapped_indices_uniq that are not -1 = desired_timestamps with valid matches
        mapped_indices_uniq = mapped_indices_uniq[mapped_indices_uniq!=-1] # matching timestamp indices in t_test
        matched_timestamps = t_test.iloc[mapped_indices_uniq].index

        # use the matching vdf timestamps to find the index in the vdf data 
        t_test2 = pd.Series(t, index = t)
        mapped_indices = t_test2.index.get_indexer_for(matched_timestamps)

        return matched_timestamps, mapped_indices, matched_timestamp_indices