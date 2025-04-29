import pandas as pd 
import numpy as np
import time
from scipy import integrate, interpolate
from scipy.signal import find_peaks, medfilt, savgol_filter, butter, sosfiltfilt
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


    def process_cell(self, records_cycler, records_vdf, project_name, cell_cycle_metrics=None, cell_data=None, cell_data_vdf=None, calibration_parameters=None, numFiles=1000, cycle_id_lims=CYCLE_ID_LIMS):
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
        if cell_cycle_metrics is not None and cell_data is not None:
            # Make list of data files with new data to process
            records_new_data = self._filter_records_new_data(cell_cycle_metrics, records_cycler)
            self.logger.info(f"Found {len(records_new_data)} new data files to process")
            # For each new file, load the data and add it to the existing dfs
            for record in records_new_data: 
                self.logger.debug(f"Processing cycler data: {record['tr_name']}")
                # process test file
                cell_data_new, cell_cycle_metrics_new = self._process_cycler_data([record], cycle_id_lims=cycle_id_lims, project_name= project_name, numFiles = numFiles)
                # load test data to df and get start and end times
                df_test = self._record_to_df(record, test_trace_keys = ['aux_vdf_timestamp_epoch_0'], df_labels =['Time [ms]'])
                if df_test is None:
                    continue
                file_start_time, file_end_time = df_test['Time [ms]'].iloc[0], df_test['Time [ms]'].iloc[-1] 
                # Update cell_data and cell_cycle_metrics and Ah throughput
                cell_data = self._update_dataframe(cell_data, cell_data_new, file_start_time, file_end_time)
                cell_cycle_metrics = self._update_dataframe(cell_cycle_metrics, cell_cycle_metrics_new, file_start_time, file_end_time)
                cell_cycle_metrics['Ah throughput [A.h]'] = cell_data['Ah throughput [A.h]'][(cell_data.discharge_cycle_indicator==True) | (cell_data.charge_cycle_indicator==True)]
        else:
            records_new_data = records_cycler.copy()
            cell_data, cell_cycle_metrics = self._process_cycler_data(records_new_data, cycle_id_lims=cycle_id_lims, project_name= project_name, numFiles = numFiles)
        
        # Process the expansion data
        if len(records_vdf)==0: 
            self.logger.info("No vdf data for this cell")
           
            cell_data_vdf = pd.DataFrame(columns=['Time [ms]','Raw expansion [-]','Raw expansion [um]', 'Expansion [-]','Expansion [um]','Expansion ref [-]', 'Temperature [degC]','cycle_indicator','Expansion STDEV [cnt]','Ref STDEV [cnt]','Drive Current [-]'])
            cell_cycle_metrics['Max cycle expansion [-]'] = np.nan
            cell_cycle_metrics['Min cycle expansion [-]'] = np.nan
            cell_cycle_metrics['Reversible cycle expansion [-]'] = np.nan
            cell_cycle_metrics['Max cycle expansion [um]'] = np.nan
            cell_cycle_metrics['Min cycle expansion [um]'] = np.nan
            cell_cycle_metrics['Reversible cycle expansion [um]'] = np.nan
            cell_cycle_metrics['Drive current [-]']=np.nan
            cell_cycle_metrics['Expansion STDDEV [cnt]']=np.nan
            cell_cycle_metrics['Ref STDDEV [cnt]']=np.nan
            
            records_new_data_vdf=cell_data_vdf
        elif cell_data_vdf is not None:
            self.logger.info(f"Process cell_data_vdf")
            # Make list of data files with new data to process
            records_new_data_vdf = self._filter_records_new_data(cell_cycle_metrics, records_vdf)
            if len(records_new_data_vdf)>0:
                for record in records_new_data_vdf:
                    self.logger.info(f"Processing new vdf data: {record['tr_name']}")
                    # process test file
                    cell_data_vdf_new, cell_cycle_metrics_new = self._process_cycler_expansion([record], cell_cycle_metrics, calibration_parameters, numFiles = numFiles)
                    # load test data to df and get start and end times
                    df_test = self._record_to_df(record, test_trace_keys = ['h_datapoint_time'], df_labels =['Time [ms]'])
                    file_start_time, file_end_time = df_test['Time [ms]'].iloc[0], df_test['Time [ms]'].iloc[-1] 
                    # Update cell_data_vdf and cell_cycle_metrics
                    cell_data_vdf = self._update_dataframe(cell_data_vdf, cell_data_vdf_new, file_start_time, file_end_time, update_AhT = False)
                    cell_cycle_metrics = self._update_dataframe(cell_data, cell_cycle_metrics_new, file_start_time, file_end_time, update_AhT = False)

        else: # if pickle file doesn't exist or load_pickle is False, (re)process all expansion data
            self.logger.info(f"Process all vdf data")
            records_new_data_vdf = records_vdf.copy()
            cell_data_vdf, cell_cycle_metrics = self._process_cycler_expansion(records_new_data_vdf, cell_cycle_metrics, calibration_parameters, numFiles = numFiles)    

        self.logger.info(f"Finished processing {len(records_new_data)} new cycler files and {len(records_new_data_vdf)} new vdf files")
        # rearrange columns of cell_cycle_metrics for easy reading with data on left and others on right
        cols = cell_cycle_metrics.columns.to_list()
        move_idx = [c for c in cols if '[' in c] + [c for c in cols if '[' not in c] # Columns with data include '[' in the key
        cell_cycle_metrics = cell_cycle_metrics[move_idx]

        # if there is new data, save it to pickle files
        update = len(records_new_data)>0 or len(records_new_data_vdf)>0
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

      
    # def summarize_rpt_data(self, cell_data, cell_data_vdf, cell_cycle_metrics, project_name):
    #     """
    #     Identify files with RPTs, process RPT metrics, and save the summary data for each RPT file

    #     Parameters
    #     ----------
    #     cell_data: DataFrame
    #         The dataframe of the cell data
    #     cell_data_vdf: DataFrame
    #         The dataframe of the cell data vdf
    #     cell_cycle_metrics: DataFrame
    #         The dataframe of the cell cycle metrics
    #     project_name: str
    #         The project name for the cell
        
    #     Returns
    #     -------
    #     DataFrame
    #         The dataframe of the summary data for each RPT file
    #     """
    #     rpt_filenames = list(set(cell_cycle_metrics['Test name'][(cell_cycle_metrics['Test type'] == 'RPT') | (cell_cycle_metrics['Test type'] == '_F')| (cell_cycle_metrics['Test type'] == '_Cy100')| (cell_cycle_metrics['Test type'] == '_Cby100')]))
    #     cycle_summary_cols = [c for c in cell_cycle_metrics.columns.to_list() if '[' in c] + ['Test name', 'Protocol']
    #     cell_rpt_data = pd.DataFrame() 
    #     # Determine the pulse currents based on project name
    #     # pulse_currents = DEFAULT_PULSE_CURRENTS
    #     if project_name in PROJECT.keys(): 
    #         project_settings = PROJECT[project_name]
    #     else:
    #         project_settings = PROJECT['DEFAULT']
    #     pulse_currents = project_settings['pulse_currents']
    #     I_C20 = project_settings['I_C20']
    #     I_C10 = project_settings['I_C10']
    #     I_threshold = project_settings['I_threshold']

    #     # for each RPT file (not sure what it'll do if there are multiple RPT files for 1 RPT...)
    #     for j,rpt_file in enumerate(rpt_filenames):
    #         rpt_idx = cell_cycle_metrics[(cell_cycle_metrics['Test name'] == rpt_file)].index #& (cell_cycle_metrics['Test type'] == 'RPT') #include formation and other files?
    #         pre_rpt = pd.DataFrame()
    #         esoh_record_line = -1

    #         if '_F' in cell_cycle_metrics['Test name']:
    #             I_slow = I_C10 # Formation
    #         else: 
    #             I_slow = I_C20 # RPT

    #         # for each section of the RPT...
    #         for i in rpt_idx:
    #             rpt_subcycle = pd.DataFrame()
    #             # find timestamps for partial cycle
    #             t_start = cell_cycle_metrics['Time [ms]'].loc[i]-30
    #             try: # end of partial cycle = next time listed
    #                 t_end = cell_cycle_metrics['Time [ms]'].loc[i+1]+30
    #             except: # end of partial cycle = end of file
    #                 t_end = cell_data['Time [ms]'].iloc[-1]+30

    #             # log summary stats for this partial cycle in dictionary
    #             rpt_subcycle['RPT #'] = j
    #             rpt_subcycle = cell_cycle_metrics[cycle_summary_cols].loc[i].to_dict()

    #             t = cell_data['Time [ms]']
    #             rpt_subcycle['Data'] = [cell_data[['Time [ms]', 'Current [A]', 'Voltage [V]', 'Ah throughput [A.h]', 'Temperature [degC]', 'Step index']][(t>t_start) & (t<t_end)]]
                
    #             self.update_cycle_metrics_hppc(rpt_subcycle, cell_cycle_metrics, i, pulse_currents)
    #             index_code = self.update_cycle_metrics_esoh(rpt_subcycle, cell_cycle_metrics, i, pre_rpt, esoh_record_line, I_slow = I_slow, I_threshold = I_threshold)
    #             if index_code >= 0:
    #                 # Update the previous subcycle line and data
    #                 pre_rpt = rpt_subcycle
    #                 esoh_record_line = i
    #             elif index_code == -1:
    #                 # Reset the previous subcycle line and data
    #                 pre_rpt = pd.DataFrame()
    #                 esoh_record_line = -1
            
    #             # add vdf data to dictionary
    #             t_vdf = cell_data_vdf['Time [ms]']
    #             if len(t_vdf)>1: #ignore for constrained cells
    #                 rpt_subcycle['Data vdf'] = [cell_data_vdf[(t_vdf>t_start) & (t_vdf<t_end)]]

    #             # convert and add dictionary to dataframe
    #             cell_rpt_data= pd.concat([cell_rpt_data, pd.DataFrame.from_dict(rpt_subcycle)])
    #     # format df: put protocol in front and reindex
    #     cell_rpt_data.reset_index(drop=True, inplace=True)
    #     cols = cell_rpt_data.columns.to_list()
    #     if cols != []:
    #         cell_rpt_data = cell_rpt_data[[cols[len(cols)-1]] + cols[0:-1]] 
    #     # Creating a temporary column 'temp_sort' with the sorting values
            
    #     try:
    #         cell_rpt_data['temp_sort'] = cell_rpt_data['Data'].apply(lambda x: x['Time [ms]'].iloc[0] if not x.empty else float('inf'))
    #         cell_rpt_data = cell_rpt_data.sort_values(by='temp_sort')
    #         cell_rpt_data.drop('temp_sort', axis=1, inplace=True)

    #     except: 
    #         self.logger.error(f"Error while creating temp_sort column for {cell_cycle_metrics['Test name']}")

        
    #     return cell_rpt_data
    
    def summarize_rpt_data_2(self, cell_data, cell_data_vdf, cell_cycle_metrics, project_name):
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
        rpt_filenames0 = cell_cycle_metrics.loc[rpt_idx, 'Test name']         # (cell_cycle_metrics['Test type'] == 'RPT') | (cell_cycle_metrics['Test type'] == '_F')| (cell_cycle_metrics['Test type'] == '_Cy100')| (cell_cycle_metrics['Test type'] == '_Cby100')
        rpt_filenames = [list(set(rpt_filenames0))[i] for i, x in sorted(enumerate(set(rpt_filenames0)), key=lambda x: x[1])]
        cell_rpt_data = pd.DataFrame() 

        # Determine the pulse currents based on project name
        if project_name in PROJECT.keys(): 
            project_settings = PROJECT[project_name]
        else:
            project_settings = PROJECT['DEFAULT']
        Q_nom = project_settings['nominal_capacity']
        pulse_currents = project_settings['pulse_currents']
        I_C20 = project_settings['I_C20']
        I_C10 = project_settings['I_C10']
        dI_threshold = project_settings['I_threshold']
        x_peaks_nom = project_settings['x_peaks_nom']
        Un_data = project_settings['Un_data']
        Up_data = project_settings['Up_data']

        # For each RPT file, copy relevant CCM rows and process each subcycle according to the protocol 
        x_esoh_fit = [] # track previous fit params to init next
        for rpt_file in rpt_filenames:
            # PREP ccm_rpt_file
            # copy ccm rows with same test name and initialize columns for data dfs
            ccm_rpt_file = cell_cycle_metrics[(cell_cycle_metrics['Test name'] == rpt_file)].copy()# & ([ p in rpt_protocols for p in cell_cycle_metrics['Protocol']])]
            ccm_rpt_file['Data'] = [pd.DataFrame()]*len(ccm_rpt_file)
            ccm_rpt_file['Data vdf'] = [pd.DataFrame()]*len(ccm_rpt_file)

            # calculate RPT number based on number of HPPCs
            rpt_num = list(np.cumsum(cell_cycle_metrics['Protocol']=='HPPC')) # change to C/20 charge to include formation?
            ccm_rpt_file['RPT #'] = [rpt_num[i] for i in ccm_rpt_file.index]
            
            # PROCESS SUBCYCLE DATA BY PROTOCOL: Identify and clean timeseries data, then extract metrics to save
            # For each protocol...
            rpt_protocols = ['HPPC','C/20 charge', 'C/20 discharge'] # process HPPC first to have Rs for IR correction
            for protocol in  rpt_protocols: 
                # find indices in ccm of subcycles with the protocol 
                protocol_idx = ccm_rpt_file[ccm_rpt_file['Protocol']==protocol].index
                if len(protocol_idx) > 1: # warn if multiple of any protocol is found in one RPT file 
                    self.logger.warning("Multiple " + protocol + " protocols detected.")
                
                # For each subcycle of the protocol subcycle, identify data and write to ccm. Process HPPC while it's loaded
                for p in protocol_idx:
                    # identify subcycle start and end timestamps and get data from cell_data
                    t_start_protocol = ccm_rpt_file.at[p,'Time [ms]']-30
                    try: # end of partial cycle = next time listed
                        t_end_protocol = cell_cycle_metrics.at[p+1,'Time [ms]']+30
                    except: # end of partial cycle = end of all known data
                        t_end_protocol = cell_data['Time [ms]'].iloc[-1]+30
                    subcycle_data = cell_data[['Time [ms]', 'Current [A]', 'Voltage [V]', 'Ah throughput [A.h]', 'Temperature [degC]', 'Step index']][(cell_data['Time [ms]']>t_start_protocol) & (cell_data['Time [ms]']<t_end_protocol)]
                    ccm_rpt_file.at[p,'Data'] = pd.concat([ccm_rpt_file.at[p,'Data'],subcycle_data])
                    
                    # add corresponding vdf data to dictionary if exists
                    t_vdf = cell_data_vdf['Time [ms]']
                    if len(t_vdf)>1: #ignore for constrained cells
                        subcycle_vdf_data = cell_data_vdf[(t_vdf>t_start_protocol) & (t_vdf<t_end_protocol)] 
                        ccm_rpt_file.at[p,'Data vdf'] = pd.concat([ccm_rpt_file.at[p,'Data vdf'],subcycle_vdf_data])                 
                    
                    # For HPPC data, calculate resistances
                    if protocol == 'HPPC':
                        # calculate resistances and add to ccm
                        self.update_cycle_metrics_hppc(ccm_rpt_file.loc[p], [cell_cycle_metrics, ccm_rpt_file,], p, pulse_currents)

            # After all data is assigned, calculate eSOH parameters from C/20 charge
            # START ESOH UPDATE -----------------
            for c, charge in ccm_rpt_file[ccm_rpt_file['Protocol']=='C/20 charge'].iterrows():
                data = charge['Data']
                # Use I_slow closest to I_C10 (formation) or I_C20 (RPT)
                mean_charge_current = np.nanmean(data['Current [A]'][data['Current [A]']>0])
                if  mean_charge_current >  np.mean([I_C20, I_C10]):
                    I_slow = I_C10
                else: 
                    I_slow = I_C20
                
                test_name = charge['Test name']
                # try:
                self.logger.info(f"Processing eSOH for {test_name}")
                # Isolate data of interest
                data = self._isolate_data(data, I_slow = I_slow, dI_threshold = dI_threshold)

                # Pre-process data before filtering 
                Q_c = charge['Charge capacity [A.h]']
                t = (data['Time [ms]']-data['Time [ms]'].iloc[0])/1000
                Q = data['Ah throughput [A.h]']-data['Ah throughput [A.h]'].iloc[0]
                z = 1 - (Q.iloc[-1]-Q)/Q_c #Q/rpt.Q_c set 100% at top of charge
                
                # SOC-dependent IR correction to voltage using Rs from closest HPPC
                hppc = ccm_rpt_file[ccm_rpt_file['Protocol']=='HPPC'].iloc[0]
                try: # largest discharge current
                    Rs = np.array(hppc['Rs'])[np.array(hppc['I_pulse'])<=min(pulse_currents)]
                    SOC = (np.array(hppc['Q_pulse'])[np.array(hppc['I_pulse'])<=min(pulse_currents)]-hppc['Ah throughput [A.h]'])/Q_c
                    R_correction = np.interp(z, SOC, Rs)
                except: # largest discharge current
                    Rs = np.array(hppc['Rs'])[np.array(hppc['I_pulse'])>=max(pulse_currents)]
                    SOC = (np.array(hppc['Q_pulse'])[np.array(hppc['I_pulse'])>=max(pulse_currents)]-hppc['Ah throughput [A.h]'])/Q_c
                    R_correction = np.interp(z, SOC, Rs)
                IR_correction = R_correction*data['Current [A]'].to_numpy()  
                V = data['Voltage [V]'] - IR_correction

                # downsample and filter data and calculate derivatives
                downsample = max(int(round(2.5e-4/np.mean(np.diff(z)))),1) # based on UMBL2022FEB Neware data
                downsample_idx = list(range(0, len(Q),downsample))
                t_sim = t.iloc[downsample_idx]
                z_sim = z.iloc[downsample_idx]
                Q_sim = Q.iloc[downsample_idx]
                window_length = int(3000/np.average(np.diff(t_sim)) + 1) #int((0.07/np.average(np.diff(z)) + 1)) #
                data = self._filter_data(t_sim, z_sim, Q_sim, V.iloc[downsample_idx], window_length=window_length) #rpt.charge.dv_ic["data"] #
                # data = self._filter_data(t, z,Q, V, downsample = downsample, window_length= window_length)

                # find peaks to estimate Cn and x0
                x1 = x_peaks_nom[0] # low SOC peak
                x2 = x_peaks_nom[1] # high SOC peak
                idx_peaks_dv, _ = self._find_dV_peaks(data.dVdz, data.z)
                Q1 = data.Q[idx_peaks_dv[0]]
                Q2 = data.Q[idx_peaks_dv[1]]
                Cn_hat = abs(Q1-Q2)/abs(x1-x2)
                x0_hat = x1 - Q1/Cn_hat

                # init other params with previous rpt's fitted esoh params to speed up solver
                if len(x_esoh_fit) != 0: 
                    Cp_hat,y0_hat,sigma_Cn_hat,sigma_Cp_hat = x_esoh_fit[2], x_esoh_fit[3], x_esoh_fit[4], x_esoh_fit[5]
                else: 
                    Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat = 4, 0.9, 0, 0
                x_esoh_0 = [Cn_hat, x0_hat, Cp_hat, y0_hat, sigma_Cn_hat, sigma_Cp_hat]
                
                # fit eSOH 
                lb, ub, W = project_settings['lb_esoh'], project_settings['ub_esoh'], project_settings['W']
                theta, capacity, x_esoh_fit, RMSE_V, RMSE_dVdq,dQ_low_peak,dQ_high_peak = self._esoh_est(data, capacity = Q_c, x0 = x_esoh_0, Un = Un_data, Up = Up_data, lb = lb, ub = ub, W = W) # rmse_V is a mag smaller than rmse_dvdq
                if (RMSE_V > 0.025):
                    self.logger.warning(f"Error in V estimation is too high: {RMSE_V}. For {test_name}")
                    theta[:] = np.full(len(theta), np.NaN)
                    x_esoh_fit = [] # don't init with fitted bc of high error
                    
                # Update the cell_cycle_metrics with eSOH data
                for df in [cell_cycle_metrics, ccm_rpt_file]:
                    # make columns 
                    for col in ["C", "Cn","x0","x100", "sigma_Cn", "Cp", "y0", "y100", "sigma_Cp", "RMSE_V", "RMSE_dVdQ", "p1_err", "p2_err"]:
                        if col not in df.columns:
                            df[col] = np.nan
                        df[col] = df[col].astype(object)

                    # Update the cell_cycle_metrics with the new data
                        df.at[c, 'C'] = capacity
                        df.at[c, 'Cn'] = theta[0]
                        df.at[c, 'x0'] = theta[1]
                        df.at[c, 'x100'] = theta[2]
                        df.at[c, 'sigma_Cn'] = theta[3]
                        df.at[c, 'Cp'] = theta[4]
                        df.at[c, 'y0'] = theta[5]
                        df.at[c, 'y100'] = theta[6]
                        df.at[c, 'sigma_Cp'] = theta[7]
                        df.at[c, 'RMSE_V'] = RMSE_V
                        df.at[c, 'RMSE_dVdQ'] = RMSE_dVdq
                        df.at[c, 'p1_err'] = dQ_low_peak
                        df.at[c, 'p2_err'] = dQ_high_peak

            # END ESOH UPDATE -----------------
            cell_rpt_data = pd.concat([cell_rpt_data, ccm_rpt_file])
        
        # Calculate LLI and LAM 
        n_Li, LLI, LAM_NE, LAM_PE = self._calculate_LLI_LAM(cell_rpt_data['Cn'],cell_rpt_data['x0'],cell_rpt_data['x100'],cell_rpt_data['Cp'],cell_rpt_data['y0'],cell_rpt_data['y100'])
        for df in [cell_cycle_metrics, cell_rpt_data]:
            df['n_Li'] = n_Li
            df['LLI'] = LLI
            df['LAM_NE'] = LAM_NE
            df['LAM_PE'] = LAM_PE
        
        # copy RPT number from cell_rpt_data to ccm
        cell_cycle_metrics['RPT #'] = cell_rpt_data['RPT #'].copy()

        # format df
        cell_rpt_data = cell_rpt_data.sort_values(by=['Time [ms]'])
        cell_rpt_data = cell_rpt_data[[p in rpt_protocols for p in cell_rpt_data['Protocol']]] # only keep hppc and c/20 dis/charge
        # cell_rpt_data = cell_rpt_data[cell_rpt_data['Protocol'].notna()] # nans pass through previous filter so remove here
        cell_rpt_data.reset_index(drop=True, inplace=True)

        return cell_rpt_data, cell_cycle_metrics
    
    def _calculate_LLI_LAM(self, Cn,x0,x100,Cp,y0,y100): 
        try:
            n_Li = 3600/96485*(x100*Cn-y100*Cp)
            LLI = (1-n_Li/n_Li.loc[n_Li.first_valid_index()])
            LAM_NE = (1-Cn/Cn.loc[Cn.first_valid_index()])
            LAM_PE = (1-Cp/Cp.loc[Cp.first_valid_index()])
        except:
            n_Li = [np.nan]*len(Cn)
            LLI = [np.nan]*len(Cn)
            LAM_NE = [np.nan]*len(Cn)
            LAM_PE = [np.nan]*len(Cn)     
        return n_Li, LLI, LAM_NE, LAM_PE
    # def update_cycle_metrics_esoh(self, rpt_subcycle, cell_cycle_metrics, index, pre_subcycle: pd.DataFrame, record_line_index, I_slow, I_threshold):
    #     """
    #     Method used for eSOH calculation

    #     Parameters
    #     ----------
    #     rpt_subcycle: DataFrame
    #         The dataframe of the subcycle
    #     cell_cycle_metrics: DataFrame
    #         The dataframe of the cell cycle metrics
    #     index: int
    #         The index of the current subcycle in CC
    #     ch_subcycle: DataFrame
    #         The dataframe of the charge subcycle
    #     record_line_index: int
    #         The index for the record line
    #     I_slow: float
    #         The RPT dis/charge current magnitude (e.g. C/20) [A]
    #     """
    #     # Skip the Formation data
    #     # if '_F_' in cell_cycle_metrics.loc[index, 'Test name']:
    #     #     return -2

    #     if record_line_index > 0 and rpt_subcycle['Protocol'] == 'HPPC':
    #         td = (cell_cycle_metrics["Time [ms]"].iloc[index] - cell_cycle_metrics["Time [ms]"].iloc[record_line_index])/1e3/3600
    #         if td < 10:
    #             return -1
        
    #     if rpt_subcycle['Protocol'] == 'C/20 discharge' or  rpt_subcycle['Protocol'] == 'C/20 charge':
    #         # If there is a previous subcycle, process eSOH
    #         if isinstance(pre_subcycle, pd.DataFrame) and pre_subcycle.empty: 
    #             return index
    #         if isinstance(pre_subcycle, dict) and not bool(pre_subcycle):
    #             return index
    #         if pre_subcycle['Protocol'] == 'C/20 charge':
    #             # ch_subcycle = rpt_subcycle
    #             # dh_subcycle = pre_subcycle
    #             dh_subcycle = rpt_subcycle
    #             ch_subcycle = pre_subcycle
    #         else:
    #             # ch_subcycle = pre_subcycle
    #             # dh_subcycle = rpt_subcycle
    #             dh_subcycle = pre_subcycle
    #             ch_subcycle = rpt_subcycle
    #         try:
    #             self.logger.info(f"Processing eSOH for {rpt_subcycle['Test name']}")
    #             q_data, v_data, dVdQ_data, q_full = self._load_V_data(ch_subcycle, dh_subcycle, I_slow = I_slow, I_threshold = I_threshold)
    #             theta, cap, err_v, err_dVdQ, p1_err, p2_err, p12_err = self.esoh_est(q_data, v_data, dVdQ_data, q_full, w1=W1, w2=W2, w3=W3, dVdQ_bool=False)
    #             if err_v > 20:
    #                 self.logger.warning(f"Error in V estimation is too high: {err_v}. For {rpt_subcycle['Test name']}")
    #                 # theta[:] = np.full(6, np.NaN)

    #         except Exception as e:
    #             if 'theta' not in locals(): # assign nans if theta doesn't exist, even if V error is high
    #                 theta, cap, err_v, err_dVdQ, p1_err, p2_err, p12_err = np.full(6, np.NaN), np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    #             self.logger.error(f"Error while processing eSOH for {rpt_subcycle['Test name']}: {e}") # ERROR: can only assign an iterable

    #         # Update the cell_cycle_metrics with eSOH data
    #         cell_cycle_metrics.at[record_line_index, 'C'] = cap
    #         cell_cycle_metrics.at[record_line_index, 'Cn'] = theta[0]
    #         cell_cycle_metrics.at[record_line_index, 'X0'] = theta[1]
    #         cell_cycle_metrics.at[record_line_index, 'X100'] = theta[2]
    #         cell_cycle_metrics.at[record_line_index, 'Cp'] = theta[3]
    #         cell_cycle_metrics.at[record_line_index, 'y0'] = theta[4]
    #         cell_cycle_metrics.at[record_line_index, 'y100'] = theta[5]
    #         cell_cycle_metrics.at[record_line_index, 'RMSE_V'] = err_v
    #         cell_cycle_metrics.at[record_line_index, 'RMSE_dVdQ'] = err_dVdQ
    #         cell_cycle_metrics.at[record_line_index, 'p1_err'] = p1_err
    #         cell_cycle_metrics.at[record_line_index, 'p2_err'] = p2_err
    #         cell_cycle_metrics.at[record_line_index, 'p12_err'] = p12_err
    #         return -1
    #     return -2
    
    # def update_cycle_metrics_esoh_2(self, ch_subcycle, dh_subcycle, cycle_metrics, ccm_idx, I_slow, I_threshold):
    #     """
    #     Method used for eSOH calculation

    #     Parameters
    #     ----------
    #     rpt_subcycle: DataFrame
    #         The dataframe of the subcycle
    #     cell_cycle_metrics: List of DataFrames 
    #         List of dataframes to update including cell cycle metrics and RPT cycle metrics
    #     index: int
    #         The index of the current subcycle
    #     ch_subcycle: DataFrame
    #         The dataframe of the charge subcycle
    #     record_line_index: int
    #         The index for the record line
    #     I_slow: float
    #         The RPT dis/charge current magnitude (e.g. C/20) [A]
    #     """

    #     test_name = ch_subcycle['Test name']
    #     try:
    #         self.logger.info(f"Processing eSOH for {test_name}")
    #         q_data, v_data, dVdQ_data, q_full = self._load_V_data(ch_subcycle, dh_subcycle, I_slow = I_slow, I_threshold = I_threshold)
    #         theta, cap, err_v, err_dVdQ, p1_err, p2_err, p12_err = self.esoh_est(q_data, v_data, dVdQ_data, q_full, w1=W1, w2=W2, w3=W3, dVdQ_bool=False)
    #         if (err_v > 25) & (ch_subcycle['Test type'] != "_F"):
    #             self.logger.warning(f"Error in V estimation is too high: {err_v}. For {test_name}")
    #             theta[:] = np.full(6, np.NaN)

    #     except Exception as e:
    #         # if 'theta' not in locals(): # assign nans if theta doesn't exist, even if V error is high
    #         theta, cap, err_v, err_dVdQ, p1_err, p2_err, p12_err = np.full(6, np.NaN), np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
    #         self.logger.error(f"Error while processing eSOH for {test_name}: {e}") # ERROR: can only assign an iterable

    #     # Update the cell_cycle_metrics with eSOH data
    #     for df in cycle_metrics:
    #         df.at[ccm_idx, 'C'] = cap
    #         df.at[ccm_idx, 'Cn'] = theta[0]
    #         df.at[ccm_idx, 'x0'] = theta[1]
    #         df.at[ccm_idx, 'x100'] = theta[2]
    #         df.at[ccm_idx, 'Cp'] = theta[3]
    #         df.at[ccm_idx, 'y0'] = theta[4]
    #         df.at[ccm_idx, 'y100'] = theta[5]
    #         df.at[ccm_idx, 'RMSE_V'] = err_v
    #         df.at[ccm_idx, 'RMSE_dVdQ'] = err_dVdQ
    #         df.at[ccm_idx, 'p1_err'] = p1_err
    #         df.at[ccm_idx, 'p2_err'] = p2_err
    #         df.at[ccm_idx, 'p12_err'] = p12_err

    def _isolate_data(self, data, I_slow, dI_threshold): 
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
        start_idx = data[abs(data['Current [A]']-(I_slow))<dI_threshold].index[0]
        end_idx = data[abs(data['Current [A]']-(I_slow))<dI_threshold].index[-1]
        if end_idx != len(data)-1: 
            data.drop(data.loc[0:start_idx-1].index, inplace=True)  
            data.drop(data.loc[end_idx+1::].index, inplace=True)        
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
            })
        return filtered_data
    
    def _find_dV_peaks(self, dVdz, z, min_prominence = 0.01, z_low_peak_min = 0.1, z_low_peak_max = 0.5, z_upper_peak_max = 0.85):
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
        idx_peaks_dv = [[p for p in idx_peaks0_dv if (z_low_peak_max<z[p])][0]] # upper peak (and (z[p]<z_upper_peak_max)))
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

    def _esoh_est(self,filtered_data, Un, Up, capacity, x0 = [4.2,0.85,5.5,0.3,0,0], lb = [1, 0, 1, 0,0,0], ub = [5, 1, 6.5, 1, 0.3, 0.3], W = [1,1], z_dv_loss = [0.1,0.9]):
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

        # Find z location for lowest and highest trough in dVdQ data to bound data included in RMSE loss func calc 
        # data_peak_idx0, _ = find_peaks(-filtered_data.dVdQ, prominence=(0.005,1e3))
        # z_loss_min = max(filtered_data.z[data_peak_idx0[0]],0.1)
        # z_loss_max = 0.95 

        # setup fitting optimization 
        bounds = Bounds(lb, ub)
        nleq1 = lambda x: capacity/x[0]+x[1] 
        nlcon1 = NonlinearConstraint(nleq1, 0, 1) # ineq constraint 0 < (x0+C/Cn = x100) < 1
        nleq2 = lambda x:  -capacity/x[2]+x[3]
        nlcon2 = NonlinearConstraint(nleq2, 0, 1) # ineq constraint 0 < (y0-C/Cp = y100) < 1
        x_fit = minimize(self._loss_func, x0, 
                        args=(filtered_data, Un, Up, W, z_dv_loss[0], z_dv_loss[1]),
                        bounds = bounds ,
                        constraints=[nlcon1,nlcon2],
                        tol=1e-6,
                        #  options={'disp': True},
                        #  method='SLSQP', 
                        ).x

        # assign results to theta: [Cn, x0, Cp, y0, sigma_Cn, sigma_Cp] -> [C_n,x_0,x_100,sigma_Cn,C_p,y_0,y_100, sigma_Cp]
        C_n, x_0, sigma_Cn  = x_fit[0], x_fit[1], x_fit[4]
        C_p, y_0, sigma_Cp = x_fit[2], x_fit[3], x_fit[5]
        x_100 = capacity/C_n + x_0
        y_100 = -capacity/C_p + y_0
        theta = [C_n,x_0,x_100,sigma_Cn,C_p,y_0,y_100, sigma_Cp]
        # theta = [round(tt,4) for tt in theta]

        # sim OCV from fitted eSOH
        Q_sim = filtered_data.Q
        t_sim = filtered_data.t
        z_sim = filtered_data.z
        V_sim_fit = self._calc_ocv(x_fit, Q_sim, Un, Up) 
        sim = self._filter_data(t_sim, z_sim, Q_sim, V_sim_fit, window_length=int(3000/np.average(np.diff(t_sim)) + 1))
        
        # calculate errors
        RMSE_V = np.sqrt(np.mean(np.square(sim.V-filtered_data.V)))
        valid_data  = (z_sim>z_dv_loss[0]) & (z_sim<z_dv_loss[1])
        RMSE_dVdq = np.sqrt(np.mean(np.square(sim.dVdQ[valid_data]-filtered_data.dVdQ[valid_data])))
        idx_peaks_dv_data, _ = self._find_dV_peaks(filtered_data.dVdz, filtered_data.z) # recalculate with filtered data. can also bring in original
        Q_peak_data = filtered_data.Q[idx_peaks_dv_data]
        try: 
            idx_peaks_dv, _ = self._find_dV_peaks(sim.dVdz, sim.z)
            peak_errors = sim.Q[idx_peaks_dv].to_numpy()-Q_peak_data.to_numpy()
            dQ_low_peak = peak_errors[0]
            dQ_high_peak = peak_errors[1]
        except: # can't find peaks (e.g. formation C/10)
            dQ_low_peak = np.nan
            dQ_high_peak = np.nan
        return theta, capacity, x_fit, RMSE_V, RMSE_dVdq,dQ_low_peak,dQ_high_peak 
    
    def _loss_func(self,x_esoh,filtered_data, Un, Up, W = [1, 1], z_loss_min = 0.1, z_loss_max = 0.9):
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
        z_data = filtered_data.z.to_numpy()
        Q_data = filtered_data.Q.to_numpy()
        V_data = filtered_data.V.to_numpy()
        dVdQ_data = filtered_data.dVdQ.to_numpy()
        # dQdV_data = filtered_data.dQdV.to_numpy()
        # x_esoh[4] = 0.2
        V_sim = self._calc_ocv(x_esoh, Q_data, Un, Up)

        # filter window length depends on average delta_z
        dV_sim = np.diff(V_sim)
        dV_sim = np.append(dV_sim, dV_sim[-1])
        dVdQ_sim = dV_sim/filtered_data.dQ
        # dQ_sim = np.diff(Q_data)
        # dQ_sim = np.append(dQ_sim, dQ_sim[-1])
        # dVdQ_sim = dV_sim/ dQ_sim #filtered_data.dQ (not sure why np.mean(np.diff(Q_data)) is a mag larger than np.mean(filtered_data.dQ)
        # dQdV_sim = 1/dVdQ_sim

        # calculate errors within valid ranges of Q and V
        V_error = V_sim-V_data
        valid_z_data = (z_loss_min <z_data) & (z_data<z_loss_max) # ignore data outside [0.1,0.9] SOC
        dVdQ_error = (dVdQ_sim[valid_z_data]-dVdQ_data[valid_z_data])#*dVdQ_data[valid_z_data] #<---NOT RMSE use dVdQ_data as a weight
        # dQdV_error = dQdV_sim[valid_z_data]-dQdV_data[valid_z_data]

        # calculate loss based on error, normalized by max value within valid range of V and Q
        error = lambda error: np.sqrt(np.nanmean(error**2)) # rmse
        RMSE_V_norm = error(V_error)/np.mean(V_data)
        RMSE_dV_norm = error(dVdQ_error)/np.mean(dVdQ_data[valid_z_data])
        V_loss = W[0]*RMSE_V_norm
        dVdQ_loss = W[1]*RMSE_dV_norm
        # dQdV_loss = W[2]*error(dQdV_error)/np.mean(dQdV_data[valid_z_data]) 
        loss = V_loss + dVdQ_loss #+ dQdV_loss
        return loss #, (RMSE_V, RMSE_dV), dVdQ_error #, dQdV_loss

    def _calc_ocv(self,x,Q, Un0, Up0): #self._calc_ocv(x_esoh, Q_data, Un, Up)
        """
        Calculate ocv curve given eSOH parameters in x
        x: List of subset of eSOH params [Cn, x0, Cp, y0, sigma_Cn, sigma_Cp]
        Q: np.array of Q data
        """
        Up = self._U_aged(Un0, Up0, Q, stoic_0_hat = x[3], C_hat = x[2], sigma_C = x[5], electrode = 'positive', dz = 1, z_range = [0,0]) # z_range = [0,0] for no distribution in Cp
        Un = self._U_aged(Un0, Up0, Q, stoic_0_hat = x[1], C_hat = x[0], sigma_C = x[4], electrode = 'negative', dz = 1/(20*max(x[4],0.1)), z_range = [-3,3]) # increased # of disc. pts with increased sigma_Cn to avoid dUn/dQ peak separation
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
        p_values =  norm.cdf(-abs(z_scores)) 
        C = C_hat + sigma_C*z_scores
        return C, p_values
    
    def _q2x (self,q,Cn,x0):
        x = x0+q/Cn
        return x 
    def _q2y (self,q,Cp,y0):
        y = y0-q/Cp
        return y 
    # def _load_V_data(self, ch_rpt, dh_rpt, I_slow, I_threshold=0.005, d_int=0.01): # , I_slow = 0.177,I_threshold=0.003default for whatever cell sravan set these currents for...
    #     """
    #     Method used for eSOH calculation
    #     """
    #     d_ch = ch_rpt["Data"].copy()
    #     d_dh = dh_rpt["Data"].copy()
    #     d_dh = d_dh.reset_index(drop=True)
    #     d_ch = d_ch.reset_index(drop=True)
    #     d_dh["Time [ms]"] = d_dh["Time [ms]"]-d_dh["Time [ms]"].iloc[0]
    #     d_ch["Time [ms]"] = d_ch["Time [ms]"]-d_ch["Time [ms]"].iloc[0]
    #     d_ch=d_ch[d_ch["Time [ms]"]<=1e8]
    #     d_dh=d_dh[d_dh["Time [ms]"]<=1e8]
    #     d_ch1=d_ch[(d_ch["Current [A]"]>0)] # charge current is + 
    #     d_dh1=d_dh[(d_dh["Current [A]"]<0)] # discharge current is - 
    #     d_ch=d_ch[(d_ch["Current [A]"]>(I_slow-I_threshold)) & (d_ch["Current [A]"]<(I_slow+I_threshold))] # look for C/20 current
    #     d_dh=d_dh[(d_dh["Current [A]"]<-(I_slow-I_threshold)) & (d_dh["Current [A]"]>-(I_slow+I_threshold))]
    #     q_cv = max(d_ch1["Ah throughput [A.h]"])-max(d_ch["Ah throughput [A.h]"])
    #     # Filter data points with only V>2.7 V and V< 4.2V
    #     d_ch=d_ch[(d_ch["Voltage [V]"]>=2.7) & (d_ch["Voltage [V]"]<=4.2)]
    #     d_dh=d_dh[(d_dh["Voltage [V]"]>=2.7) & (d_dh["Voltage [V]"]<=4.2)]
    #     d_ch["Ah throughput [A.h]"] = d_ch["Ah throughput [A.h]"]-d_ch["Ah throughput [A.h]"].iloc[0]
    #     d_dh["Ah throughput [A.h]"] = d_dh["Ah throughput [A.h]"]-d_dh["Ah throughput [A.h]"].iloc[0]

    #     t_d = d_dh["Time [ms]"].to_numpy()
    #     t_d = t_d - t_d[0]
    #     t_d = t_d/1e3 
    #     I_d = d_dh["Current [A]"].to_numpy()
    #     V_d = d_dh["Voltage [V]"].to_numpy()
    #     Ah_d = d_dh["Ah throughput [A.h]"].to_numpy()
    #     Q_d = integrate.cumtrapz(abs(I_d), t_d/3600)
    #     Q_d = np.append(Q_d,Q_d[-1])
    #     t_c = d_ch["Time [ms]"].to_numpy()
    #     t_c = t_c - t_c[0]
    #     t_c = t_c/1e3
    #     I_c = d_ch["Current [A]"].to_numpy()
    #     V_c = d_ch["Voltage [V]"].to_numpy()
    #     Ah_c = d_ch["Ah throughput [A.h]"].to_numpy()
    #     Q_c = integrate.cumtrapz(abs(I_c), t_c/3600)
    #     Q_c = np.append(Q_c,Q_c[-1])
    #     # Normalizing from SOC=100
    #     Ah_c = Ah_c[-1]-Ah_c + q_cv
    #     Q_c = Q_c[-1]-Q_c + q_cv
    #     # Interpolate for Averaging
    #     Q_d,idx_d = np.unique(Q_d,return_index=True)
    #     V_d = V_d[idx_d]
    #     Q_c,idx_c = np.unique(Q_c,return_index=True)
    #     V_c = V_c[idx_c]

    #     dt = np.average(np.diff(t_d))
    #     dt = round(dt,1)

    #     # filter charge and discharge data
    #     window = int(3000/dt + 1)
    #     Qf_d,Vf_d,dVdQ_d = self._filter_qv_data(Q_d,V_d,window_length=window,polyorder=3)
    #     Qf_c,Vf_c,dVdQ_c = self._filter_qv_data(Q_c,V_c,window_length=window,polyorder=3)

    #     # Interpolate filtered data for averaging
    #     Qf_d,idx_d = np.unique(Qf_d,return_index=True)
    #     Vf_d = Vf_d[idx_d]
    #     Qf_c,idx_c = np.unique(Qf_c,return_index=True)
    #     Vf_c = Vf_c[idx_c]
    #     int_V_d = interpolate.CubicSpline(Qf_d,Vf_d,extrapolate=True)
    #     int_dVdQ_d = interpolate.CubicSpline(Qf_d,dVdQ_d,extrapolate=True)
    #     int_V_c = interpolate.CubicSpline(Qf_c,Vf_c,extrapolate=True)
    #     int_dVdQ_c = interpolate.CubicSpline(Qf_c,dVdQ_c,extrapolate=True)

    #     Qmax = np.min([np.max(Qf_d),np.max(Qf_c)])
    #     Qmin = np.max([np.min(Qf_d),np.min(Qf_c)])
    #     Qfull = np.max([np.max(Qf_d),np.max(Qf_c)])

    #     Qin = np.arange(Qmin,Qmax,d_int)
    #     V_d_int = int_V_d(Qin)
    #     V_c_int = int_V_c(Qin)
    #     dVdQ_d_int = int_dVdQ_d(Qin)
    #     dVdQ_c_int = int_dVdQ_c(Qin)
    #     V_avg = (V_d_int+V_c_int)/2
    #     dVdQ_avg = (dVdQ_d_int+dVdQ_c_int)/2

    #     return Qin, V_avg, dVdQ_avg, Qfull
    

    # def _filter_qv_data(self,Q, V, z, window_length=3001, polyorder=3):
    #     """
    #     Method used for eSOH calculation
    #     """
    #     qf = savgol_filter(Q,window_length,polyorder,0)
    #     dQ = -savgol_filter(Q,window_length,polyorder,1)
    #     vf = savgol_filter(V,window_length,polyorder,0)
    #     dV = savgol_filter(V,window_length,polyorder,1)
    #     zf = savgol_filter(z,window_length,polyorder,0)
    #     dz = savgol_filter(z,window_length,polyorder,1)
    #     dVdQ = dV/dQ
    #     dVdz = dV/dz
    #     return qf,vf,zf, dVdQ, dVdz
    
    # def _get_peaks(self, q, dVdQ):
    #     """
    #     Method used for eSOH calculation
    #     """
    #     q_max = max(q)
    #     pks_,_ = find_peaks(dVdQ,prominence=0.01)
    #     pks = [pk for pk in pks_ if q[pk]>=0.1*q_max and q[pk]<=0.9*q_max ]
    #     q_peaks = q[pks]
    #     if len(q_peaks) >0:
    #         q1 = q_peaks[0]
    #         q2 = q_peaks[-1]
    #     else:
    #         q1 =  q[0]
    #         q2 =  q[-1]
    #     return q1,q2
    
    # def _calc_un(self, sto):
    #     """
    #     Method used for eSOH calculation
    #     """
    #     sto = sto#+0.002 #<-----EDIT
    #     p_eq = UN_VAR1[0:8]
    #     a_eq = UN_VAR1[8:15]
    #     b_eq = UN_VAR1[15:]
    #     u_eq2=p_eq[-1]+p_eq[-2]*np.exp((sto-a_eq[-1])/b_eq[-1])
    #     for i in range(6):
    #         u_eq2 += p_eq[i]*np.tanh((sto-a_eq[i])/b_eq[i])

    #     p_eq = UN_VAR2[0:8]
    #     a_eq = UN_VAR2[8:15]
    #     b_eq = UN_VAR2[15:]
    #     u_eq1=p_eq[-1]+p_eq[-2]*np.exp((sto-a_eq[-1])/b_eq[-1])
    #     for i in range(6):
    #         u_eq1 += p_eq[i]*np.tanh((sto-a_eq[i])/b_eq[i])
    #     u_eq = u_eq1*0.5 + u_eq2*0.5 #(u_eq1 + u_eq2)/2#-0.01 #<-----EDIT
    #     return u_eq

    # def _calc_up(self, sto):
    #     """
    #     Method used for eSOH calculation
    #     """
    #     # u_eq1 = P1*(sto**9) + P2*(sto**8) + P3*(sto**7) + P4*(sto**6) + P5*(sto**5) + P6*(sto**4) + P7*(sto**3) + P8*(sto**2) + P9*sto + P10
    #     u_eq = (
    #         -2708.586858 * (sto ** 9) +13443.432624 * (sto ** 8) 
    #         -28538.487067 * (sto ** 7) +33871.076514 * (sto ** 6)
    #         -24665.620934 * (sto ** 5) +11381.032457 * (sto ** 4) 
    #         -3313.239062 * (sto ** 3) +586.285653 * (sto ** 2) 
    #         -59.774094 * (sto ** 1) +7.408439 
    #         )
    #     return u_eq #(u_eq1+u_eq2)/2
    
    # def _calc_opc(self, x, q):
    #     """
    #     Method used for eSOH calculation
    #     """
    #     ocp = self._calc_up(x[3]+q/x[2]) - self._calc_un(x[1]-q/x[0])
    #     return ocp

    # def _fitfunc(self, x , q_data, v_data, dVdQdata, w1, w2, w3, dVdQ_bool):
    #     """
    #     Method used for eSOH calculation
    #     """
    #     model = np.concatenate([
    #             [self._calc_opc(x,q)]
    #             for q in q_data
    #         ]
    #     )
    #     qa, qb = self._get_peaks(q_data,dVdQdata)
    #     q_max = np.max(q_data)
    #     # Q1 = 0*0.1*Qmax
    #     q1 = 0.1*q_max
    #     q2 = 0.9*q_max

    #     # constrain qs to +/-5% of max value?? used to calculate bounds for error weights
    #     if qa>0 and qa<0.5*q_max  and dVdQ_bool: #dVdQ_bool = False by default
    #         q3 = qa - 0.05*q_max
    #         q4 = qa + 0.05*q_max
    #     else:
    #         q3 = 0.25*q_max
    #         q4 = 0.45*q_max

    #     #     
    #     if qb< q_max and qb > 0.5*q_max and dVdQ_bool:
    #         q5 = qb - 0.05*q_max
    #         q6 = qb + 0.05*q_max
    #     else:
    #         q5 = 0.7*q_max
    #         q6 = 0.9*q_max
        
    #     # calculated weighted error based on q 
    #     wvec = np.ones(len(q_data))
    #     for i,q in enumerate(q_data):
    #         if q<q1 or q>q2: # z outside of [0.1,0.9] 
    #             wvec[i]=w1
    #         if q>=q1 and q<=q2: # z within [0.1,0.9] 
    #             wvec[i]=w2
    #         if q>=q3 and q<=q4: # z within [0.25,0.45] for low SOC peak
    #             wvec[i]=w3
    #         if q>=q5 and q<=q6: # z within [0.7,0.9] for high SOC peak
    #             wvec[i]=w3    
    #     vd = np.multiply(v_data,wvec)
    #     vs = np.multiply(model,wvec)
    #     error = vd-vs

    #     # not sure what kind of loss function this is? 
    #     out = np.linalg.norm(error)/np.sqrt(len(vd))
    #     return out
        
    # def esoh_est(self, q_data, v_data, dVdQ_data, q_full, w1=1,w2=1,w3=1, dVdQ_bool=True, x0 = [4.2,0.85,5.5,0.3], lb = [1, 0, 1, 0], ub = [5, 1, 6.5, 1], win=5):
    #     """
    #     Method used for eSOH calculation
    #     """
    #     cap = q_full
    #     bounds = Bounds(lb, ub)
    #     nleq1 = lambda x: -cap/x[0]+x[1]
    #     nlcon1 = NonlinearConstraint(nleq1, 0, 1)
    #     nleq2 = lambda x:  cap/x[2]+x[3]
    #     nlcon2 = NonlinearConstraint(nleq2, 0, 1)
    #     result = minimize(self._fitfunc, x0, args=(q_data,v_data,dVdQ_data,w1,w2,w3,dVdQ_bool), bounds=bounds,constraints=[nlcon1,nlcon2])
    #     res = result.x
    #     cn = res[0]
    #     cp = res[2]
    #     x100 = res[1]
    #     y100 = res[3]
    #     x0 = res[1]-cap/res[0]
    #     y0 = res[3]+cap/res[2]
    #     theta = [cn,x0,x100,cp,y0,y100]
    #     theta = [round(tt,4) for tt in theta]
    #     v_fit = np.concatenate([
    #             [self._calc_opc(res,Q)]
    #             for Q in q_data
    #         ]
    #     )
    #     err_V = 1e3*np.linalg.norm(v_data-v_fit)/np.sqrt(len(v_data))
    #     err_V = round(err_V,1)
    #     cap = round(cap,3)
    #     _,_,dVdQ_fit = self._filter_qv_data(q_data,v_fit,window_length=win,polyorder=3)
    #     q1_data,q2_data = self._get_peaks(q_data,dVdQ_data)
    #     q1_fit,q2_fit = self._get_peaks(q_data,dVdQ_fit)
    #     p1_err = q1_data-q1_fit
    #     p2_err = q2_data-q2_fit
    #     p12_data = q2_data-q1_data
    #     p12_fit = q2_fit-q1_fit
    #     p12_err = p12_data-p12_fit
    #     q_bool = (q_data>0.1*cap) & (q_data<0.9*cap)
    #     dVdQ_data_f = dVdQ_data[q_bool]
    #     dVdQ_fit_f = dVdQ_fit[q_bool]
    #     q_f = q_data[q_bool]
        
    #     err_dVdQ = 1e3*np.linalg.norm(dVdQ_data_f-dVdQ_fit_f)/np.sqrt(len(v_data))
    #     err_dVdQ = round(err_dVdQ,1)

    #     return theta, cap, err_V, err_dVdQ, p1_err, p2_err, p12_err

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
            time_ms = rpt_subcycle['Data']['Time [ms]'] / 1000.0
            current_a = rpt_subcycle['Data']['Current [A]']
            voltage_v = rpt_subcycle['Data']['Voltage [V]']
            ah_throughput = rpt_subcycle['Data']['Ah throughput [A.h]']
            # Call the get_Rs_SOC function with PULSE_CURRENTS from config
            hppc_data = self.get_Rs_SOC(time_ms, current_a, voltage_v, ah_throughput, pts = hppc_pts)
            # Dynamically generate metrics_mapping based on PULSE_CURRENTS
            
            for df in cycle_metrics:
                for col in ["Q_pulse","dt_pulse","I_pulse","Rs","Rlong"]:
                    if col not in df.columns:
                        df[col] = np.nan
                    df[col] = df[col].astype(object)
                if not hppc_data['Q'].empty:
                # Update the cell_cycle_metrics with the new data
                    df.at[i, "Q_pulse"] = hppc_data['Q'].tolist()# if hppc_data['Q'].empty else np.nan
                    df.at[i, "dt_pulse"] = hppc_data['pulse_duration'].tolist()#[0] if hppc_data['pulse_duration'].empty  else np.nan
                    df.at[i, "I_pulse"] = hppc_data['pulse_current'].tolist()#[0] if hppc_data['pulse_current'].empty  else np.nan
                    df.at[i, "Rs"] = hppc_data['R_s'].tolist()#[0] if hppc_data['R_s'].empty  else np.nan
                    df.at[i, "Rlong"] = hppc_data['R_l'].tolist()#[0] if hppc_data['R_l'].empty else np.nan


    def get_Rs_SOC(self, t, I, V, Q, pts = 4):
        """ 
        Processes HPPC data to get DC Resistance for given pulse currents at different Qs. 
        Assumes that this is a discharge HPPC i.e. the initial Q is 0, corresponding to 100% SOC
        """
        idxi1 = np.where((np.diff(I)>0.1) & (I[1:]>0.1))[0]
        idxi2 = np.where((np.diff(I)<-0.1) & (I[1:]<-0.1))[0]
        idxi = np.concatenate([idxi1,idxi2])
        idxi = idxi + 1
        idxk1 = np.where((np.diff(I)<-0.1) & (I[:-1]>0.1))[0]
        idxk2 = np.where((np.diff(I)>0.1) & (I[:-1]<-0.1))[0]
        idxk = np.concatenate([idxk1,idxk2])
        no_pulses = min(len(idxi),len(idxk))

        r1, r2, qr, pcur, pdur = [], [], [], [], []
        for pno in range(no_pulses):
            t1, V1, I1 = t[idxi[pno]-1-pts:idxi[pno]-1], V[idxi[pno]-1-pts:idxi[pno]-1], I[idxi[pno]-1-pts:idxi[pno]-1]
            t2, V2, I2 = t[idxi[pno]:idxi[pno]+pts], V[idxi[pno]:idxi[pno]+pts], I[idxi[pno]:idxi[pno]+pts]
            t3, V3, I3 = t[idxk[pno]+1-pts:idxk[pno]+1], V[idxk[pno]+1-pts:idxk[pno]+1], I[idxk[pno]+1-pts:idxk[pno]+1]
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
   
    def _process_cycler_expansion(self, records_vdf, cell_cycle_metrics, calibration_parameters, numFiles = 1000, t_match_threshold=60000):
        # Combine vdf data into a single df
        cell_data_vdf = self._combine_cycler_expansion(records_vdf, calibration_parameters, numFiles)
        
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
            try:
                tmpa=cell_data_vdf['Drive Current [-]']
                cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Drive Current [-]'] = tmpa[i]
            except:
                cell_cycle_metrics.loc[discharge_cycle_idx[j], 'Drive Current [-]'] = 0
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

    def _process_cycler_data(self, records, cycle_id_lims, project_name, numFiles=1000):
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
        cell_data, cell_cycle_metrics = self._combine_cycler_data(records, cycle_id_lims, numFiles = numFiles, project_name = project_name)

        charge_t_idx = list(cell_data[cell_data.charge_cycle_indicator ==True].index)
        discharge_t_idx = list(cell_data[cell_data.discharge_cycle_indicator ==True].index)
        Q_c, Q_d = self._calc_capacities(cell_data['Time [ms]'], cell_data['Current [A]'], cell_data['Ah throughput [A.h]'], charge_t_idx, discharge_t_idx, Qmax)
        # find average current
        I_avg_c,I_avg_d = self._avg_cycle_data_x(cell_data['Time [ms]'], cell_data['Current [A]'], charge_t_idx, discharge_t_idx)
        # Find min/max metrics
        cycle_idx_minmax = list(cell_data[cell_data.cycle_indicator ==True].index)
        cycle_idx_minmax.append(len(cell_data)-1)
        V_max, V_min = self._max_min_cycle_data(cell_data['Voltage [V]'], cycle_idx_minmax)
        T_max, T_min = self._max_min_cycle_data(cell_data['Temperature [degC]'], cycle_idx_minmax)

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

    def _calc_capacities(self, t, I, AhT, charge_idx, discharge_idx, Qmax):
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
                if(Q>Qmax):
                    print(Q)
                    Q=np.nan
                    self.logger.warning(f"Invalid Capacity for cycle {i}")
                if cycle_idx[i] in charge_idx:
                    Q_c.append(Q) 
                else:
                    Q_d.append(Q) 
        return np.array(Q_c), np.array(Q_d)

    def _combine_cycler_data(self, records_cycler, cycle_id_lims, numFiles=1000, last_AhT = 0, Qmax=3.8, project_name='DEFAULT'):
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
        frames =[]
        test_types = list(cycle_id_lims.keys())

        # For each data file...
        for record in records_cycler[0:min(len(records_cycler), numFiles)]:

            # 1. Load data from each data file to a dataframe. Update AhT and ignore unplugged thermocouple values. For RPTs, convert t with ms.
            isRPT =  ('RPT').lower() in record['tr_name'].lower() or ('EIS').lower() in record['tr_name'].lower() 
            isFormation = ('_F').lower() in record['tr_name'].lower() and not ('_FORMTAP').lower() in record['tr_name'].lower() 

            # 1a. for arbin and biologic files
            test_data = pd.DataFrame()
            if ('arbin' in record['tags']) or ('biologic' in record['tags']): 
                test_trace_keys_arbin = ['h_datapoint_time','h_test_time','h_current', 'h_potential', 'c_cumulative_capacity', 'h_step_index','h_cycle','h_charge_capacity','h_discharge_capacity','h_step_ord',]
                df_labels_arbin = ['Time [ms]','Test Time [ms]', 'Current [A]', 'Voltage [V]', 'Ah throughput [A.h]', 'Step index','Cycle index', 'Charge Ah throughput [A.h]','Discharge Ah throughput [A.h]','Step ord']
                test_data = self._record_to_df(record, test_trace_keys_arbin, df_labels_arbin)
                
                if(test_data is None):
                    self.logger.error(f"test_data is None from {record['tr_name']}")
                else:
                    test_data['Temperature [degC]'] = [np.nan]*len(test_data) # make arbin tables with same columns as neware files
                if ('biologic' in record['tags']):
                    if(max(abs(test_data['Current [A]']))>20): # current data is ma vs A divide by 1000.
                        test_data['Current [A]']=test_data['Current [A]']/1000
                        test_data['Ah throughput [A.h]']=test_data['Ah throughput [A.h]']/1000
            # 1b. for neware files
            elif 'neware_xls_4000' in record['tags']: 
                test_data = self._record_to_df(record)
                if test_data['Temperature [degC]'] is not None:
                    test_data['Temperature [degC]'] = np.where((test_data['Temperature [degC]'] >= 200) & (test_data['Temperature [degC]'] <250), np.nan, test_data['Temperature [degC]']) 
                else:
                    self.logger.info(f"Missing Temperature [degC] data from {record['tr_name']}")

            else:
                raise ValueError(f"Unsupported test tag found in {record['tags']}")
            test_data.reset_index(drop=True, inplace=True)
            self.logger.info(f"Get {len(test_data)} rows of data from {record['tr_name']}")
            # 2. Reassign to variables
            t = test_data['Time [ms]'].reset_index(drop=True)
            t_test = test_data['Test Time [ms]'].reset_index(drop=True)
            I = test_data['Current [A]'].reset_index(drop=True)
            V = test_data['Voltage [V]'].reset_index(drop=True)
            T = test_data['Temperature [degC]'].reset_index(drop=True)
            step_idx = test_data['Step index'].reset_index(drop=True)
            cycle_idx=test_data['Cycle index'].reset_index(drop=True)
            Ah_Discharge=test_data['Discharge Ah throughput [A.h]'].reset_index(drop=True)
            Ah_Charge=test_data['Charge Ah throughput [A.h]'].reset_index(drop=True)
            step_ord=test_data['Step ord'].reset_index(drop=True)
            # 3. Calculate AhT 
            if 'neware_xls_4000' in record['tags'] and isFormation:  
                # 3a. From integrating current.... some formation files had wrong units
                AhT_calculated = integrate.cumtrapz(abs(I), (t-t[0])/1000)/3600 + last_AhT
                AhT_calculated = np.append(AhT_calculated,AhT_calculated[-1]) # repeat last value to make AhT the same length as t
                test_data['Ah throughput [A.h]'] = AhT_calculated
                # test_data['Ah throughput [A.h]'] = test_data['Ah throughput [A.h]']/1e6 + last_AhT # add last AhT value (if using scaled cycler cummulative capacity. Doesn't solve all neware formation AhT issues...)
            else:
                # 3b. From cycler cumulative capacity...
                test_data['Ah throughput [A.h]'] = test_data['Ah throughput [A.h]'] + last_AhT # add last AhT value (if using cycler cummulative capacity)
                AhT = test_data['Ah throughput [A.h]']
                
            # 3c. update AhT from last file
            AhT = test_data['Ah throughput [A.h]'].reset_index(drop=True)
            last_AhT = AhT.iloc[-1] #update last AhT value for next file

            # check that indices are consistent    #update sidegeljb 12/20/2023  to use new signals (TODO)
            indices_to_check = [t, I, V, AhT, step_idx]
            for i in range(len(indices_to_check)-1):
                assert indices_to_check[i].index.equals(indices_to_check[i+1].index), f"Indices are not consistent between columns in the data from {record['tr_name']}"
            lengths_to_check = [len(t), len(I), len(V), len(AhT), len(step_idx)]
            assert len(set(lengths_to_check)) == 1, f"Inconsistent data lengths in the data from {record['tr_name']}"

            # 4. Change cycle filtering thresholds by test type and include the idx at the end of the file in case cell is still cycling.
            # Search for test type in test name. If there's no match, use the default settings 
            lims={}
            for test_type in test_types: # check for test types with different filters (e.g. RPT, F, EIS)
                if (test_type).lower() in record['tr_name'].lower(): 
                    lims = cycle_id_lims[test_type]
                    if isRPT:
                        test_protocol = 'RPT' #EIS -> RPT
                    else:
                        test_protocol = test_type
                if len(lims) == 0: #default
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
                discharge_start_idx_file = test_data[I==0].iloc[discharge_start_idx_file].index.to_list()
                charge_start_idx_file,_ = find_peaks(-medfilt(V[I==0], kernel_size = 101),prominence = trough_prominence, height = (None, -2.7)) # height to ignore min during hppc
                charge_start_idx_file = test_data[I==0].iloc[charge_start_idx_file].index.to_list()
                charge_start_idx_file.insert(0, 0)
                charge_start_idx_file.insert(len(charge_start_idx_file), len(V)-1)
                charge_start_idx_file, discharge_start_idx_file = self._match_charge_discharge(np.array(charge_start_idx_file), np.array(discharge_start_idx_file))

            # if  'neware_xls_4000' in record['tags']:
            #     discharge_start_idx_file=np.where(np.diff(cycle_idx).astype(bool))

            # if  'arbin' in record['tags']:
            #     discharge_start_idx_file=np.where(np.diff(cycle_idx).astype(bool))

            else: # find I==0 and filter out irrelevant points

                charge_start_idx_file, discharge_start_idx_file = self._find_cycle_idx(t_test, I, V, AhT,Ah_Discharge,Ah_Charge,step_ord, step_idx, cycle_idx,test_protocol, V_max_cycle = V_max_cycle, V_min_cycle = V_min_cycle, dt_min = dt_min, dAh_min= dAh_min)
                
                try: # won't work for half cycles (files with only charge or only discharge)
                    charge_start_idx_file, discharge_start_idx_file = self._match_charge_discharge(charge_start_idx_file, discharge_start_idx_file)
                except:
                    self.logger.error(f"Error processing {record['tr_name']}: failed to _match_charge_discharge")
                    pass

            # 6. Add aux cycle indicators to df. Column of True if start of a cycle, otherwise False. Set default cycle indicator = charge start 
            file_with_capacity_check = isRPT or isFormation 
            test_data['discharge_cycle_indicator'] = [False]*len(test_data)
            test_data['charge_cycle_indicator'] = [False]*len(test_data)
            test_data['capacity_check_indicator'] = [False]*len(test_data)

            test_data.loc[discharge_start_idx_file, 'discharge_cycle_indicator'] = True
            test_data.loc[charge_start_idx_file, 'charge_cycle_indicator'] = True
            test_data['cycle_indicator'] = test_data['charge_cycle_indicator'] # default cycle = charge start 
            if file_with_capacity_check:
                test_data.loc[charge_start_idx_file, 'capacity_check_indicator'] = True

            # 6a. Add test type and test name to test_data
            test_data['Test type'] = [' ']*len(test_data)
            test_data['Test name'] = [' ']*len(test_data)
            test_data.loc[np.concatenate((discharge_start_idx_file,charge_start_idx_file)), 'Test type'] = test_protocol
            test_data.loc[np.concatenate((discharge_start_idx_file,charge_start_idx_file)), 'Test name'] = record['tr_name']

            # 6b. identify subcycle type. For extracting HPPC and C/20 dis/charge data later. 
            test_data['Protocol'] = [np.nan]*len(test_data)
            file_cell_cycle_metrics = test_data[(test_data.charge_cycle_indicator==True) | (test_data.discharge_cycle_indicator==True)]

            for i in range(0,len(file_cell_cycle_metrics)):
                t_start = file_cell_cycle_metrics['Time [ms]'].iloc[i]
                if i == len(file_cell_cycle_metrics)-1: # if last subcycle, end of subcycle = end of file 
                    t_end = test_data['Time [ms]'].iloc[-1]
                else: # end of subcycle = start of next subcycle
                    t_end = file_cell_cycle_metrics['Time [ms]'].iloc[i+1]
                # find valid indices. Some arbin cycles have backwards time jumps mid-cycler file (e.g.152062,152067)
                test_idx = (t>t_start) & (t<t_end)
                if sum(test_idx)>0:
                    t_subcycle = test_data['Time [ms]'][(t>t_start) & (t<t_end)]
                    I_subcycle = test_data['Current [A]'][(t>t_start) & (t<t_end)]
                    Ah_subcycle = test_data['Ah throughput [A.h]'][(t>t_start) & (t<t_end)]
                    Q_subcycle = Ah_subcycle.iloc[-1]-Ah_subcycle.iloc[0]
                    data_idx = file_cell_cycle_metrics.index.tolist()[i]
                    if file_with_capacity_check and Q_subcycle>Q_min and Q_subcycle<Q_max:
                        if len(np.where(np.diff(np.sign(I_subcycle)))[0])>10: # hppc: ID by # of types of current sign changes (threshold is arbitrary)
                            test_data.loc[data_idx,'Protocol'] = 'HPPC'
                        elif (t_end-t_start)/3600000.0 >8 and  np.mean(I_subcycle) > 0: #and  np.mean(I_subcycle) < Qmax / 18: # C/20 charge: longer than 8 hrs and mean(I)>0. Will ID C/10 during formation as C/20...
                            test_data.loc[data_idx,'Protocol'] = 'C/20 charge'
                        elif (t_end-t_start)/3600000.0 > 8 and  np.mean(I_subcycle) < 0:# and  np.mean(I_subcycle) > - Qmax / 18 : # C/20 discharge: longer than 8 hrs and mean(I)<0.Will ID C/10 during formation as C/20...
                                test_data.loc[data_idx,'Protocol'] = 'C/20 discharge'
                    

            # 7. Add to list of dfs where each element is the resulting df from each file.
            self.logger.debug(record['tr_name'] + '   Cycles: ' + str(len(charge_start_idx_file)) + '   AhT: ' + str(round(AhT.iloc[-1],2)))
            self.logger.debug(f"test_data: {test_data}")
            
            # 8. Special case for handling UMBL Arbin cycling + rpt files with continuous cycling
            if ('arbin' in record['tags']) and ('UMBL2022FEB' in record['tr_name'].upper()) and ('P25C' in record['tr_name'].upper()):
                # if 'CYC_P3CP3C' in record['tr_name'].upper():
                # find indices of RPT cycles
                # check_min_charge_duration = np.diff(t_test[charge_start_idx_file])/3600>18
                # check_min_discharge_duration = np.diff(t_test[discharge_start_idx_file])/3600>18
                # check_max_charge_duration = np.diff(t_test[charge_start_idx_file])/3600<100 # filter out cal aging
                # check_max_discharge_duration = np.diff(t_test[discharge_start_idx_file])/3600<100
                # rpt_idx_charge_start_idx_file = np.where(check_min_charge_duration & check_max_charge_duration)[0] # still works for formation bc dt beginning of charge ~20 hrs
                # rpt_idx_discharge_start_idx_file = np.where(check_min_discharge_duration & check_max_discharge_duration)[0]
                # charge_start_idx_file_rpt = [charge_start_idx_file[r] for r in rpt_idx_charge_start_idx_file]
                # discharge_start_idx_file_rpt = [discharge_start_idx_file[r] for r in rpt_idx_discharge_start_idx_file]

                # # indicate rpt dis/charge cycles in test_data
                # test_data.loc[charge_start_idx_file_rpt, 'capacity_check_indicator'] = True
                # test_data.loc[np.concatenate((charge_start_idx_file_rpt, discharge_start_idx_file_rpt)), 'Test type'] = 'RPT'
                
                # rerun rpt subcyle identification with test time 
                for i in range(0,len(file_cell_cycle_metrics)):
                    t_start = file_cell_cycle_metrics['Test Time [ms]'].iloc[i]
                    if i == len(file_cell_cycle_metrics)-1: # if last subcycle, end of subcycle = end of file 
                        t_end = test_data['Test Time [ms]'].iloc[-1]
                    else: # end of subcycle = start of next subcycle
                        t_end = file_cell_cycle_metrics['Test Time [ms]'].iloc[i+1]
                    I_subcycle = test_data['Current [A]'][(t_test>t_start) & (t_test<t_end)]
                    Ah_subcycle = test_data['Ah throughput [A.h]'][(t_test>t_start) & (t_test<t_end)]
                    Q_subcycle = Ah_subcycle.iloc[-1]-Ah_subcycle.iloc[0]
                    data_idx = file_cell_cycle_metrics.index.tolist()[i]
                    
                    if Q_subcycle>Q_min and Q_subcycle<Q_max:
                        if len(np.where(np.diff(np.sign(I_subcycle)))[0])>10: # hppc: ID by # of types of current sign changes (threshold is arbitrary)
                            test_data.loc[data_idx,'Protocol'] = 'HPPC'
                        elif (t_end-t_start)/3600.0 >18 and  np.mean(I_subcycle) > 0 and (t_end-t_start)/3600.0 <48: #and (t_end-t_start): # and  np.mean(I_subcycle) < Qmax / 8: # C/20 charge: longer than 8 hrs and mean(I)>0. Will ID C/10 during formation as C/20... (C/18->c/8)
                            test_data.loc[data_idx,'Protocol'] = 'C/20 charge'
                        elif (t_end-t_start)/3600.0 > 18 and  np.mean(I_subcycle) < 0 and (t_end-t_start)/3600.0 <48: # and  np.mean(I_subcycle) > - Qmax / 8: # C/20 discharge: longer than 8 hrs and mean(I)<0.Will ID C/10 during formation as C/20...
                            test_data.loc[data_idx,'Protocol'] = 'C/20 discharge'
                        else:
                            pass
                    test_data.loc[((test_data['Protocol'] == 'HPPC')| (test_data['Protocol'] =='C/20 charge') |( test_data['Protocol'] =='C/20 discharge')) & (test_data['Test type'] != '_F'),'Test type'] = 'RPT'
                    test_data.loc[((test_data['Protocol'] == 'HPPC')| (test_data['Protocol'] =='C/20 charge') |( test_data['Protocol'] =='C/20 discharge'))& (test_data['Test type'] != '_F'), 'capacity_check_indicator'] = True
            frames.append(test_data)    
    
         #   time.sleep(0.1) 

        # Combine cycling data into a single df and reset the index
        self.logger.info(f"Combining {len(frames)} dataframes")
        if len(frames) == 0:
            cell_data, cell_cycle_metrics = self._create_default_cell_data(), self._create_default_cell_cycle_metrics()
            return cell_data, cell_cycle_metrics
        cell_data = pd.concat(frames)
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
    
    def _find_cycle_idx(self, t, I, V, AhT, Ah_Discharge,Ah_Charge,step_ord,step_idx,cycle_idx, test_protocol,V_max_cycle=3, V_min_cycle=4, dt_min = 600, dAh_min=1):
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
        #neware cycler increases the index at the start of the discharge. 
        # discharge_start_idx_file_cycle=np.where(np.diff(cycle_idx).astype(bool))
        # new_step_idx_file_cycle=np.where(np.diff(step_idx).astype(bool))
        # reset_step_idx_file_cycle=np.where(np.abs(np.diff(step_idx))>1) # mark points where the step index changes by more than one, could be a sign that a loop was broken.
        # find indexs with positive current

        # change point detection
        # model = "l2"  # "l1", "rbf", "linear", "normal", "ar",...
        # algo = rpt.Window(width=400, model=model).fit(I.values.reshape(-1, 1))

        # model = "l2"  # "l2", "rbf"
        # algo = rpt.Pelt(model=model, min_size=20000, jump=5000).fit(Ah_Charge.values.reshape(-1, 1)) # may need to tune these numbers...
        # my_bkps_charge = algo.predict(pen=3)

        # model = "l2"  # "l2", "rbf"
        # algo = rpt.Pelt(model=model, min_size=20000, jump=5000).fit(Ah_Discharge.values.reshape(-1, 1)) # may need to tune these numbers...
        # my_bkps_discharge = algo.predict(pen=3)
#         if(test_protocol == 'RPT'):
#             model = "l1"  # "l2", "rbf"
#             #algo = rpt.Pelt(model=model, min_size=20000, jump=5000).fit(I.values.reshape(-1, 1)) # may need to tune these numbers...
#             algo = rpt.KernelCPD(kernel="linear", min_size=20000).fit(I.values.reshape(-1, 1))
#             my_bkps = algo.predict(pen=3)
#         else:
#             model = "l1"  # "l2", "rbf"
#             #algo = rpt.Pelt(model=model, min_size=2000, jump=500).fit(I.values.reshape(-1, 1)) # may need to tune these numbers...
#             algo = rpt.KernelCPD(kernel="linear", min_size=2000).fit(I.values.reshape(-1, 1))
#             my_bkps = algo.predict(pen=3)

#         fig, ax_arr = rpt.display(I.values.reshape(-1, 1), my_bkps, figsize=(10, 6))
#         plt.show()
# #        print(my_bkps_charge)
#         print(my_bkps)
        # show results



        Ic=(I.values>1e-5).astype(int)
        Id=(I.values<-1e-5).astype(int)
        potential_charge_start_idx= np.where(np.diff(Ic)>0.5)[0]
        potential_discharge_start_idx=np.where(np.diff(Id)>0.5)[0]
        dt=np.diff(t)
        #Cumah=Ah_Charge-Ah_Discharge
        Cumah=integrate.cumtrapz(I, t,initial=0)/3600/1000 # ms to hours 
        # calculate the average discharge current and average time until the next charge step
        Cumah=Cumah-Cumah.min()
        # check for large gaps in the data, and reset the cumah counter.
        gap_index=np.argwhere(dt>1e6)# look for gaps greater than 1000 s

        if (gap_index.size >0):
            if gap_index[0][0]>0:
                for gap in gap_index[0]:
                    Cumah[(gap+1):]=Cumah[gap]+Cumah[(gap+1):]-Cumah[gap+1]

        class_count = 10 # basically needs to change by more than 10% of the full range.
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
        #discharge_start_idx=np.array([np.searchsorted(potential_discharge_start_idx,charge_start_idx[0],side='right')])
        # if my_bkps[-1] >= len(t)-1:
        #     my_bkps[-1]=len(t)-1
        #     current_sign_change_idx= np.concatenate(([0],my_bkps))
        # else:
        #     current_sign_change_idx= np.concatenate(([0],my_bkps,[len(t)-1]))
 

        # charge_start_idx=new_step_idx_file_cycle[0][ I.values[new_step_idx_file_cycle]> 1e-2 ]
        # discharge_start_idx=new_step_idx_file_cycle[0][ I.values[new_step_idx_file_cycle]< 1e-1 ]

        # if (np.abs(I[0])<1e-2):
        #     #first step is a rest.
        #     a=0
        # elif ( I[0]>=1e-2 ):    
        #     # first step is a charge
        #     charge_start_idx.append(0)
        # else:
        #     discharge_start_idx.append(0)
        
        # # Find indices of sign changes and there's a change in step index and filter based on dt, dAh, and V
        # #current_sign_change_idx = np.where(np.diff(np.sign(I)).astype(bool) & (np.diff(step_idx) !=0))[0]
        # current_sign_change_idx = np.where(np.diff(np.sign(I)).astype(bool) & (np.diff(step_idx) !=0))[0]
        # current_sign_change_idx = np.sort(np.append(current_sign_change_idx,[AhT.first_valid_index(),len(t)-1])) # add the start and end for the diff checks

        # Filter to identify cycles based on threshold inputs
        #charge_start_idx, discharge_start_idx = self._filter_cycle_idx(current_sign_change_idx, t, I, V, AhT, V_max_cycle=V_max_cycle, V_min_cycle=V_min_cycle, dt_min = dt_min, dAh_min = dAh_min)
        # fig, (ax1,ax2,ax3) = plt.subplots(3,1)
        # ax1.plot(t,Cumah)
        # ax1.plot(t[charge_start_idx],Cumah[charge_start_idx],'rx')
        # ax1.plot(t[discharge_start_idx],Cumah[discharge_start_idx],'bo')
        # ax2.plot(t,Cumah)
        # ax2.plot(t[turning_points],Cumah[turning_points],'rx')
        # ax3.plot(t,I)
        # # ax3.plot(t[charge_start_idx],Cumah[charge_start_idx],'rx')
        # # ax3.plot(t[discharge_start_idx],Cumah[discharge_start_idx],'bo')
        # plt.show()
            

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
        dt_check_1 = [i for i,dt in enumerate(np.diff(t[cycle_idx0])) if dt > dt_min]
        dt_check_1.append(len(cycle_idx0)-1) #add end of file
        dt_check = np.array(list(map(lambda x: x in dt_check_1, range(len(cycle_idx0)))))

        dAh_check_0 = [i for i,dAh in enumerate(np.diff(AhT[cycle_idx0])) if dAh > dAh_min]
        dAh_check_0.append(len(cycle_idx0)-1)  #add end of file
        dAh_check = np.array([True if i in dAh_check_0 else False for i in range(len(cycle_idx0))])
            
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