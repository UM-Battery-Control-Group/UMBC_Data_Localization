import pandas as pd 
import numpy as np
import time
from scipy import integrate
from scipy.signal import find_peaks, medfilt
from itertools import compress

from src.model.DirStructure import DirStructure
from src.model.DataFilter import DataFilter
from src.utils.Logger import setup_logger
from src.utils.DateConverter import DateConverter
from src.config.df_config import CYCLE_ID_LIMS, DEFAULT_TRACE_KEYS, DEFAULT_DF_LABELS
from src.config.calibration_config import X1, X2, C
from src.config.pulse_config import GMJULY2022_PULSE_CURRENTS, GMFEB23_PULSE_CURRENTS, MAX_PULSES, DEFAULT_PULSE_CURRENTS


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
    summarize_rpt_data(cell_data, cell_data_vdf, cell_cycle_metrics)
        Get the summary data for each RPT file
    """
    def __init__(self, dataFilter: DataFilter, dirStructure: DirStructure, dateConverter: DateConverter):
        self.dataFilter = dataFilter
        self.dirStructure = dirStructure
        self.dateConverter = dateConverter
        self.last_AHT=0
        self.logger = setup_logger()


    def process_cell(self, records_cycler, records_vdf, cell_cycle_metrics=None, cell_data=None, cell_data_vdf=None, calibration_parameters=None, numFiles=1000, cycle_id_lims=CYCLE_ID_LIMS):
        """
        Using the test records to process and update the cell data, cycle metrics, and expansion data

        Parameters
        ----------
        records_cycler: list of dict
            The list of records of the cycler data
        records_vdf: list of dict
            The list of records of the expansion data
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
                cell_data_new, cell_cycle_metrics_new = self._procesfs_cycler_data([record], cycle_id_lims=cycle_id_lims, numFiles = numFiles)
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
            cell_data, cell_cycle_metrics = self._process_cycler_data(records_new_data, cycle_id_lims=cycle_id_lims, numFiles = numFiles)
        
        # Process the expansion data
        if len(records_vdf)==0: 
            self.logger.info("No vdf data for this cell")
           
            cell_data_vdf = pd.DataFrame(columns=['Time [ms]','Expansion [-]','Expansion [um]', 'Expansion ref [-]', 'Temperature [degC]','cycle_indicator','Expansion STDEV [cnt]','Ref STDEV [cnt]','Drive Current [-]'])
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

    def summarize_rpt_data(self, cell_data, cell_data_vdf, cell_cycle_metrics, project_name):
        """
        Get the summary data for each RPT file

        Parameters
        ----------
        cell_data: DataFrame
            The dataframe of the cell data
        cell_data_vdf: DataFrame
            The dataframe of the cell data vdf
        cell_cycle_metrics: DataFrame
            The dataframe of the cell cycle metrics
        project_name: str
            The project name
        
        Returns
        -------
        DataFrame
            The dataframe of the summary data for each RPT file
        """
        rpt_filenames = list(set(cell_cycle_metrics['Test name'][(cell_cycle_metrics['Test type'] == 'RPT') | (cell_cycle_metrics['Test type'] == '_F')]))
        cycle_summary_cols = [c for c in cell_cycle_metrics.columns.to_list() if '[' in c] + ['Test name', 'Protocol']
        cell_rpt_data = pd.DataFrame() 
        # Determine the pulse currents based on project name
        pulse_currents = DEFAULT_PULSE_CURRENTS
        if project_name == 'GMJuly2022':
            pulse_currents = GMJULY2022_PULSE_CURRENTS
        elif project_name == 'GMFeb23':
            pulse_currents = GMFEB23_PULSE_CURRENTS    
        # for each RPT file (not sure what it'll do if there are multiple RPT files for 1 RPT...)
        for j,rpt_file in enumerate(rpt_filenames):
            rpt_idx = cell_cycle_metrics[cell_cycle_metrics['Test name'] == rpt_file].index

            # for each section of the RPT...
            for i in rpt_idx:
                rpt_subcycle = pd.DataFrame()
                #find timestamps for partial cycle
                t_start = cell_cycle_metrics['Time [ms]'].loc[i]-30
                try: # end of partial cycle = next time listed
                    t_end = cell_cycle_metrics['Time [ms]'].loc[i+1]+30
                except: # end of partial cycle = end of file
                    t_end = cell_data['Time [ms]'].iloc[-1]+30

                # log summary stats for this partial cycle in dictionary
                rpt_subcycle['RPT #'] = j
                rpt_subcycle = cell_cycle_metrics[cycle_summary_cols].loc[i].to_dict()

                t = cell_data['Time [ms]']
                rpt_subcycle['Data'] = [cell_data[['Time [ms]', 'Current [A]', 'Voltage [V]', 'Ah throughput [A.h]', 'Temperature [degC]', 'Step index']][(t>t_start) & (t<t_end)]]
                
                self.update_cycle_metrics_hppc(rpt_subcycle, cell_cycle_metrics, i, pulse_currents)

                # add vdf data to dictionary
                t_vdf = cell_data_vdf['Time [ms]']
                if len(t_vdf)>1: #ignore for constrained cells
                    rpt_subcycle['Data vdf'] = [cell_data_vdf[(t_vdf>t_start) & (t_vdf<t_end)]]

                # convert and add dictionary to dataframe
                cell_rpt_data= pd.concat([cell_rpt_data, pd.DataFrame.from_dict(rpt_subcycle)])
        # format df: put protocol in front and reindex
        cell_rpt_data.reset_index(drop=True, inplace=True)
        cols = cell_rpt_data.columns.to_list()
        if cols != []:
            cell_rpt_data = cell_rpt_data[[cols[len(cols)-1]] + cols[0:-1]] 
        # Creating a temporary column 'temp_sort' with the sorting values
        cell_rpt_data['temp_sort'] = cell_rpt_data['Data'].apply(lambda x: x['Time [ms]'].iloc[0] if not x.empty else float('inf'))
        cell_rpt_data = cell_rpt_data.sort_values(by='temp_sort')
        cell_rpt_data.drop('temp_sort', axis=1, inplace=True)
        



        return cell_rpt_data
    
    def update_cycle_metrics_hppc(self, rpt_subcycle, cell_cycle_metrics, i, pulse_currents):
        """
        Update cell cycle metrics based on the RPT subcycle data.

        Parameters:
        -----------
        rpt_subcycle: dict
            The dictionary of the RPT subcycle data
        cell_cycle_metrics: DataFrame
            The dataframe of the cell cycle metrics
        i: int
            The index of the cell cycle metrics
        pulse_currents: list of float
            The list of pulse currents

        Returns:
        --------
        None
        """
        if rpt_subcycle['Protocol'] == 'HPPC':
            # Extract necessary data for get_Rs_SOC function
            time_ms = rpt_subcycle['Data'][0]['Time [ms]'] / 1000.0
            current_a = rpt_subcycle['Data'][0]['Current [A]']
            voltage_v = rpt_subcycle['Data'][0]['Voltage [V]']
            ah_throughput = rpt_subcycle['Data'][0]['Ah throughput [A.h]']
            # Call the get_Rs_SOC function with PULSE_CURRENTS from config
            hppc_data = self.get_Rs_SOC(time_ms, current_a, voltage_v, ah_throughput, pulse_currents, MAX_PULSES)
            # Dynamically generate metrics_mapping based on PULSE_CURRENTS
            for col in ["Q_ch1", "R_ch1_s", "R_ch1_l",
                        "Q_ch2", "R_ch2_s", "R_ch2_l",
                        "Q_dh1", "R_dh1_s", "R_dh1_l", 
                        "Q_dh2", "R_dh2_s", "R_dh2_l"]:
                if col not in cell_cycle_metrics.columns:
                    cell_cycle_metrics[col] = np.nan
                cell_cycle_metrics[col] = cell_cycle_metrics[col].astype(object)

            # Update the cell_cycle_metrics with the new data
            cell_cycle_metrics.at[i, "Q_ch1"] = hppc_data['Q'][0] if hppc_data['Q'].empty else np.nan
            cell_cycle_metrics.at[i, "Q_ch2"] = hppc_data['Q'][1] if len(hppc_data['Q']) > 1 else np.nan
            cell_cycle_metrics.at[i, "Q_dh1"] = hppc_data['Q'][2] if len(hppc_data['Q']) > 2 else np.nan
            cell_cycle_metrics.at[i, "Q_dh2"] = hppc_data['Q'][3] if len(hppc_data['Q']) > 3 else np.nan
            cell_cycle_metrics.at[i, "R_ch1_s"] = hppc_data['R_s'][0] if hppc_data['R_s'].empty else np.nan
            cell_cycle_metrics.at[i, "R_ch1_l"] = hppc_data['R_l'][0] if hppc_data['R_l'].empty else np.nan
            cell_cycle_metrics.at[i, "R_ch2_s"] = hppc_data['R_s'][1] if len(hppc_data['R_s']) > 1 else np.nan
            cell_cycle_metrics.at[i, "R_ch2_l"] = hppc_data['R_l'][1] if len(hppc_data['R_l']) > 1 else np.nan
            cell_cycle_metrics.at[i, "R_dh1_s"] = hppc_data['R_s'][2] if len(hppc_data['R_s']) > 2 else np.nan
            cell_cycle_metrics.at[i, "R_dh1_l"] = hppc_data['R_l'][2] if len(hppc_data['R_l']) > 2 else np.nan
            cell_cycle_metrics.at[i, "R_dh2_s"] = hppc_data['R_s'][3] if len(hppc_data['R_s']) > 3 else np.nan
            cell_cycle_metrics.at[i, "R_dh2_l"] = hppc_data['R_l'][3] if len(hppc_data['R_l']) > 3 else np.nan

    def get_Rs_SOC(self, t, I, V, Q, pulse_currents, max_pulses=11):
        """ 
        Processes HPPC data to get DC Resistance for given pulse currents at different Qs. 
        Assumes that this is a discharge HPPC i.e. the initial Q is 1. 
        """
        results = []
        pts = 4
        for pulse_current in pulse_currents:
            if pulse_current<0:
                idxi = np.where((np.diff(I)<-0.1) & (I[1:]>pulse_current-0.1)& (I[1:]<pulse_current+0.1))[0]
            else:
                idxi = np.where((np.diff(I)>0.1) & (I[1:]>pulse_current-0.1)& (I[1:]<pulse_current+0.1))[0]
            idxi = idxi+1
            if pulse_current>0:
                idxk = np.where((np.diff(I)<-0.1) & (I[:-1]>pulse_current-0.1)& (I[:-1]<pulse_current+0.1))[0]
            else:
                idxk = np.where((np.diff(I)>0.1) & (I[:-1]>pulse_current-0.1)& (I[:-1]<pulse_current+0.1))[0]
            idxk = idxk
            no_pulses = min(max_pulses,min(len(idxi),len(idxk))) #robustneess hack to drop last data can revisit. Siegeljb 12/8/2023

            r1, r2, qr = [], [], []
            for pno in range(no_pulses):
                t1, V1, I1 = t[idxi[pno]-1-pts:idxi[pno]-1], V[idxi[pno]-1-pts:idxi[pno]-1], I[idxi[pno]-1-pts:idxi[pno]-1]
                t2, V2, I2 = t[idxi[pno]:idxi[pno]+pts], V[idxi[pno]:idxi[pno]+pts], I[idxi[pno]:idxi[pno]+pts]
                t3, V3, I3 = t[idxk[pno]+1-pts:idxk[pno]+1], V[idxk[pno]+1-pts:idxk[pno]+1], I[idxk[pno]+1-pts:idxk[pno]+1]
                r_p1 = abs((np.average(V2) - np.average(V1)) / (np.average(I2) - np.average(I1)))
                r_p2 = abs((np.average(V3) - np.average(V1)) / (np.average(I3) - np.average(I1)))
                q_val = np.average(Q[idxi[pno]-1-pts:idxi[pno]-1])
                r1.append(round(r_p1, 4))
                r2.append(round(r_p2, 4))
                qr.append(q_val)
            results.append({'pulse_current': pulse_current, 'Q': qr, 'R_s': r1, 'R_l': r2})
        return pd.DataFrame(results)

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
        cell_data_vdf['cycle_indicator'] = list(map(lambda x: x in cycle_idx_vdf, range(len(cell_data_vdf))))
        
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
                df_vdf = self._record_to_df(record_vdf, test_trace_keys = ['aux_vdf_timestamp_epoch_0','aux_vdf_ldcsensor_none_0', 'aux_vdf_ldcref_none_0', 'aux_vdf_ambienttemperature_celsius_0','aux_vdf_ldcstd_none_0','aux_vdf_refstd_none_0', 'aux_vdf_drivecurrent_none_0'], df_labels =['Time [ms]','Expansion [-]', 'Expansion ref [-]','Temperature [degC]','Expansion STDDEV [cnt]','Ref STDDEV [cnt]','Drive Current [-]'])
                df_vdf = df_vdf[(df_vdf['Expansion [-]'] >1e1) & (df_vdf['Expansion [-]'] <1e7)] #keep good signals 
                # Add LDC sensor calibration to df_vdf
                df_vdf = self._get_calibration_parameters(df_vdf, record_vdf['dev_name'], calibration_parameters)
                self.logger.info(f"Using calibration parameters for the entire dataframe.")
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
        return pd.DataFrame(columns=['Time [ms]','Expansion [-]', 'Expansion ref [-]', 'Temperature [degC]','cycle_indicator','Drive Current [-]','Expansion STDDEV [cnt]','Ref STDDEV [cnt]'])

    def _process_cycler_data(self, records_neware, cycle_id_lims, numFiles=1000):
        """
        Process cycler data from a list of test records

        Parameters
        ----------
        records_neware: list of dict
            The list of test records
        cycle_id_lims: list of ints
            The cycle number limits for charge, discharge, and total cycles
        numFiles: int, optional
            The max number of files to process

        Returns
        -------
        DataFrame
            The processed data
        DataFrame
            The processed cycle metrics
        """

        # combine data for all files 
        cell_data, cell_cycle_metrics = self._combine_cycler_data(records_neware, cycle_id_lims, numFiles = numFiles)
        
        # calculate capacities 
        charge_t_idx = list(cell_data[cell_data.charge_cycle_indicator ==True].index)
        discharge_t_idx = list(cell_data[cell_data.discharge_cycle_indicator ==True].index)
        Q_c, Q_d = self._calc_capacities(cell_data['Time [ms]'], cell_data['Current [A]'], cell_data['Ah throughput [A.h]'], charge_t_idx, discharge_t_idx)
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
                #Dt_elements=t[(cycle_idx[i]:cycle_idx[i+1])+1]-t[cycle_idx[i]:cycle_idx[i+1]]
            y_avg =np.trapz(data[cycle_idx[i]:cycle_idx[i+1]],x=t[cycle_idx[i]:cycle_idx[i+1]])/Dt_total #np.sum( Dt_elements*(data[cycle_idx[i]:cycle_idx[i+1]]+data[cycle_idx[i]-1:cycle_idx[i+1]-1])/2)/Dt_total
      
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

    def _calc_capacities(self, t, I, AhT, charge_idx, discharge_idx):
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
        cycle_idx.append(len(t)-1) # add last data point
        cycle_idx.sort() # should alternate charge and discharge start indices
        Q_c = []
        Q_d = []
        
        # Calculate capacity 
        for i in range(len(cycle_idx)-1):
            # Calculate capacity based on AhT.
            Q = AhT[cycle_idx[i+1]]-AhT[cycle_idx[i]]
            if cycle_idx[i] in charge_idx:
                Q_c.append(Q) 
            else:
                Q_d.append(Q) 
        return np.array(Q_c), np.array(Q_d)

    def _combine_cycler_data(self, records_cycler, cycle_id_lims, numFiles=1000, last_AhT = 0):
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
                test_trace_keys_arbin = ['h_datapoint_time','h_test_time','h_current', 'h_potential', 'c_cumulative_capacity', 'h_step_index','h_cycle']
                df_labels_arbin = ['Time [ms]','Test Time [ms]', 'Current [A]', 'Voltage [V]', 'Ah throughput [A.h]', 'Step index','Cycle index']
                test_data = self._record_to_df(record, test_trace_keys_arbin, df_labels_arbin, ms = isRPT)
                test_data['Temperature [degC]'] = [np.nan]*len(test_data) # make arbin tables with same columns as neware files
                if ('biologic' in record['tags']):
                    if(max(abs(test_data['Current [A]']))>20): # current data is ma vs A divide by 1000.
                        test_data['Current [A]']=test_data['Current [A]']/1000
                        test_data['Ah throughput [A.h]']=test_data['Ah throughput [A.h]']/1000
            # 1b. for neware files
            elif 'neware_xls_4000' in record['tags']: 
                test_data = self._record_to_df(record, ms = isRPT)
                test_data['Temperature [degC]'] = np.where((test_data['Temperature [degC]'] >= 200) & (test_data['Temperature [degC]'] <250), np.nan, test_data['Temperature [degC]']) 
            else:
                raise ValueError(f"Unsupported test tag found in {record['tags']}")
            test_data.reset_index(drop=True, inplace=True)
            self.logger.info(f"Get {len(test_data)} rows of data from {record['tr_name']}")
            # 2. Reassign to variables
            # assert not test_data.isnull().any().any(), f"Null values found in the data from {record['tr_name']}"
            t = test_data['Time [ms]'].reset_index(drop=True)
            I = test_data['Current [A]'].reset_index(drop=True)
            V = test_data['Voltage [V]'].reset_index(drop=True)
            T = test_data['Temperature [degC]'].reset_index(drop=True)
            step_idx = test_data['Step index'].reset_index(drop=True)
            cycle_idx=test_data['Cycle index'].reset_index(drop=True)
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

            # check that indices are consistent
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
            if isFormation and 'arbin' in record['tags']: # find peaks in voltage where I==0, ignore min during hppc
                peak_prominence = 0.1
                trough_prominence = 0.1
                discharge_start_idx_file, _ = find_peaks(medfilt(V[I==0], kernel_size = 101),prominence = peak_prominence)
                discharge_start_idx_file = test_data[I==0].iloc[discharge_start_idx_file].index.to_list()
                charge_start_idx_file,_ = find_peaks(-medfilt(V[I==0], kernel_size = 101),prominence = trough_prominence, height = (None, -2.7)) # height to ignore min during hppc
                charge_start_idx_file = test_data[I==0].iloc[charge_start_idx_file].index.to_list()
                charge_start_idx_file.insert(0, 0)
                charge_start_idx_file.insert(len(charge_start_idx_file), len(V)-1)
                charge_start_idx_file, discharge_start_idx_file = self._match_charge_discharge(np.array(charge_start_idx_file), np.array(discharge_start_idx_file))
            else: # find I==0 and filter out irrelevant points
                charge_start_idx_file, discharge_start_idx_file = self._find_cycle_idx(t, I, V, AhT, step_idx, V_max_cycle = V_max_cycle, V_min_cycle = V_min_cycle, dt_min = dt_min, dAh_min= dAh_min)
                try: # won't work for half cycles (files with only charge or only discharge)
                    charge_start_idx_file, discharge_start_idx_file = self._match_charge_discharge(charge_start_idx_file, discharge_start_idx_file)
                except:
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
                t_subcycle = test_data['Time [ms]'][(t>t_start) & (t<t_end)]
                I_subcycle = test_data['Current [A]'][(t>t_start) & (t<t_end)]
                data_idx = file_cell_cycle_metrics.index.tolist()[i]
                if file_with_capacity_check:
                    if len(np.where(np.diff(np.sign(I_subcycle)))[0])>10: # hppc: ID by # of types of current sign changes (threshold is arbitrary)
                        test_data.loc[data_idx,'Protocol'] = 'HPPC'
                    elif (t_end-t_start)/3600.0 >8 and  np.mean(I_subcycle) > 0: # C/20 charge: longer than 8 hrs and mean(I)>0. Will ID C/10 during formation as C/20...
                        test_data.loc[data_idx,'Protocol'] = 'C/20 charge'
                    elif (t_end-t_start)/3600.0 > 8 and  np.mean(I_subcycle) < 0: # C/20 discharge: longer than 8 hrs and mean(I)<0.Will ID C/10 during formation as C/20...
                        test_data.loc[data_idx,'Protocol'] = 'C/20 discharge'
            
            # 7. Add to list of dfs where each element is the resulting df from each file.
            self.logger.debug(record['tr_name'] + '   Cycles: ' + str(len(charge_start_idx_file)) + '   AhT: ' + str(round(AhT.iloc[-1],2)))
            self.logger.debug(f"test_data: {test_data}")
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
    
    def _find_cycle_idx(self, t, I, V, AhT, step_idx, V_max_cycle=3, V_min_cycle=4, dt_min = 600, dAh_min=1):
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
        # Find indices of sign changes and there's a change in step index and filter based on dt, dAh, and V
        current_sign_change_idx = np.where(np.diff(np.sign(I)).astype(bool) & (np.diff(step_idx) !=0))[0]
        current_sign_change_idx = np.sort(np.append(current_sign_change_idx,[AhT.first_valid_index(),len(t)-1])) # add the start and end for the diff checks
        # Filter to identify cycles based on threshold inputs
        charge_start_idx, discharge_start_idx = self._filter_cycle_idx(current_sign_change_idx, t, I, V, AhT, V_max_cycle=V_max_cycle, V_min_cycle=V_min_cycle, dt_min = dt_min, dAh_min = dAh_min)
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
            The threshold for matching timestamps
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

        # find indices for "desired_timestamps" in an array of timestamps "t" within "t_match_threshold" seconds  
        mapped_indices = [] #indexes t
        matched_timestamp_indices =[] # indexes desired_timestamps
        matched_timestamps = [] # value of t closest to desired_timestamp. includes nan if can't find matching timestamp.
        # for each timestamp... 
        for k, desired_timestamp in enumerate(desired_timestamps):
            # if smallest dt < t_match_threshold
            time_diff_seconds = (t - desired_timestamp)
            min_time_diff = np.min(abs(time_diff_seconds))

            if min_time_diff < t_match_threshold:
                matched_idx = np.argmin(abs(time_diff_seconds))
                mapped_indices.append(matched_idx)
                matched_timestamps.append(t.iloc[matched_idx])
                matched_timestamp_indices.append(k)
            elif nan_pad: # else if pad with nan if requested
                matched_timestamps.append(np.nan)

        return matched_timestamps, mapped_indices, matched_timestamp_indices