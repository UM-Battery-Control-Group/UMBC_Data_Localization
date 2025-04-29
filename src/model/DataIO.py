import os
import pickle
import pandas as pd
import numpy as np
import gzip
import shutil
import hashlib
import matplotlib.pyplot as plt
import csv
import time
from src.model.DirStructure import DirStructure
from src.model.DataDeleter import DataDeleter
from src.config.time_config import DATE_FORMAT
from src.config.df_config import TIME_COLUMNS
from src.config.path_config import ROOT_PATH, SANITY_CHECK_CSV_PATH, WRONG_TR_NAME_PATH,ROOT_DATA_PATH
from src.config.calibration_config import X1, X2, C
from src.utils.Logger import setup_logger
from src.utils.RedisClient import RedisClient

class DataIO:
    """
    The class to save and load test data

    Attributes
    ----------
    rootPath: str
        The root path of the local data
    dirStructure: DirStructure object
        The object to manage the directory structure for the local data
    dataDeleter: DataDeleter object
        The object to delete data from the local disk
    redisClient: RedisClient object
        The object to interact with Redis cache
    logger: logger object
        The object to log information
        
    Methods
    -------
    create_dev_dic(devs)
        Create the path for each device and return a dictionary of device id and device folder path and a dictionary of device name and project name
    save_test_data_update_dict(trs, dfs, devices_id_to_name)
        Save test data to local disk and update the directory structure information
    save_df(df, df_path)
        Save the dataframe to a pickle file
    extract_project_name(tags)
        Extract the project name from the tags
    load_df(test_folder=None, df_path=None, trace_keys=None)
        Load the dataframe from the pickle file with the specified trace keys
    load_trs(test_folders)
        Load the test records based on the specified test folders
    load_dfs(test_folders)
        Load the dataframes based on the specified test folders
    load_processed_data(cell_name)
        Load the processed data from the processed folder
    save_processed_data(cell_name, cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt)
        Save the processed data to the processed folder
    load_ccm_csv(cell_name)
        Load the cycle metrics csv from the processed folder
    save_figs(figs, cell_name)
        Save the figures to the processed folder
    merge_folders(src, dest)
        Merge the source folder into the destination folder
    """
    def __init__(self, dirStructure: DirStructure, dataDeleter: DataDeleter, use_redis=False):
        self.rootPath = ROOT_PATH
        self.dirStructure = dirStructure
        self.dataDeleter = dataDeleter
        self.redisClient = RedisClient() if use_redis else None
        self.logger = setup_logger()

    def create_dev_dic(self, devs):
        """
        Create a dictionary of device id and device folder path

        Parameters
        ----------
        devs: list of Device objects
            The list of devices to be saved

        Returns
        -------
        list of str
            The list of device id
        list of str
            The list of device name
        list of str
            The list of project name
        """
        devices_id = []
        devices_name = []
        projects_name = []
        for dev in devs:
            project_name = self.extract_project_name(dev.tags)
            if not project_name:
                self.logger.warning(f"The device {dev.name} does not have a project name. Put it in the UNKNOWN_PROJECT.")
                project_name = "UNKNOWN_PROJECT"
            # Use the lazy creation of the directory
            # self._create_directory(device_folder)
            devices_id.append(dev.id)
            devices_name.append(dev.name.upper())
            projects_name.append(project_name)
        return devices_id, devices_name, projects_name
    
    def save_test_data_update_dict(self, trs, dfs, cycle_stats, devices_id, devices_name, projects_name):
        """
        Save test data to local disk and update the directory structure information
        
        Parameters
        ----------
        trs: list of TestRecord objects
            The list of test records to be saved
        dfs: list of pandas Dataframe
            The list of dataframes to be saved
        cycle_stats: list of pandas Dataframe
            The list of cycle status dataframes to be saved
        devices_id_to_name: dict
            The dictionary of device id and device name
        device_name_to_project_name: dict
            The dictionary of device name and project name

        Returns
        -------
        None
        """
        for tr, df, cycle_stat in zip(trs, dfs, cycle_stats):
            i = devices_id.index(tr.device_id)
            dev_name = devices_name[i]
            project_name = projects_name[i]
            self._handle_single_record(tr, df, cycle_stat, dev_name, project_name)
    
    def save_df(self, df, df_path):
        """
        Save the dataframe to a pickle file

        Parameters
        ----------
        df: pandas Dataframe
            The dataframe to be saved
        df_path: str
            The path of the pickle file

        Returns
        -------
        None
        """
        self._save_to_pickle(df, df_path)

    def extract_project_name(self, tags):
        prefix = "Project Name:"
        project_name = "UNKNOWN_PROJECT"
        for tag in tags:
            if tag.startswith(prefix):
                # Use UPPERCASE for the project name
                project_name = tag.split(prefix)[1].strip().upper()
        return project_name
    
    def _handle_single_record(self, tr, df, cycle_stat, dev_name, project_name):
        device_folder = os.path.join(self.rootPath, project_name if project_name else '', dev_name)
        if not project_name:
            self.logger.warning(f"The device {dev_name} does not have a project name.")
            project_name = "UNKNOWN_PROJECT"
        # device_folder = os.path.join(self.rootPath, dev_name)
        if device_folder is None:
            self.logger.error(f'Device folder not found for device id {tr.device_id}')
            return None

        start_time_str = tr.start_time.strftime(DATE_FORMAT)
        # last_modified_time_str = UNIX_timestamp_to_datetime(tr.last_dp_timestamp).strftime(DATE_FORMAT)

        test_folder = os.path.join(device_folder, start_time_str)
        if not os.path.exists(test_folder):
            self._create_directory(test_folder)

        # Save the test data to a pickle file
        tr_path = self.dirStructure.get_tr_path(test_folder)
        # Save the time series data to a pickle file
        df_path = self.dirStructure.get_df_path(test_folder)
        # Save the cycle status data to a pickle file
        cycle_stats_path = self.dirStructure.get_cycle_stats_path(test_folder)
        if tr is None or df is None:
            self.logger.error(f'Test record or dataframe is None')
            return None
        #TODO: If VDF and Other data start at same time, this one file could overwrite to another. 
        #If this really happen, we should change the df.pkl.gz to vdf.pkl.gz and arbin.pkl.gz something like this.
        if os.path.exists(tr_path) and os.path.exists(df_path):
            self.logger.error(f'File already exists: {tr_path} and {df_path}, skipping...')
            return None
        try:
            # Guarantee the transactional integrity
            self._save_to_pickle(tr, tr_path)
            self._save_to_pickle(df, df_path)
            if cycle_stat is not None:
                self._save_to_pickle(cycle_stat, cycle_stats_path)
            # Append the directory structure information to the list
            self.dirStructure.append_record(tr, dev_name, project_name)
        except Exception as e:
            self.logger.error(f"Transaction failed: {e}")
            # Remove any possibly corrupted files
            for path in [tr_path, df_path]:
                if os.path.exists(path):
                    self.dataDeleter.delete_file(path)
            return None
    
    def _create_directory(self, directory_path):
        try:
            os.makedirs(directory_path, exist_ok=True)
        except Exception as err:
            self.logger.error(f'Error occurred while creating directory {directory_path}: {err}')

    def _save_to_pickle(self, data, file_path):
        temp_path = file_path + ".tmp"
        try:
            self._create_directory(os.path.dirname(file_path))
            with gzip.open(temp_path, 'wb') as f:
                pickle.dump(data, f)
            shutil.move(temp_path, file_path)
            self.logger.info(f'Saved pickle file to {file_path}')
        except Exception as err:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self.logger.error(f'Error occurred while writing file {file_path}: {err}')

    def save_df_to_csv(self, data, file_path):
        """
        Save the dataframe to a csv file
        
        Parameters
        ----------
        data: pandas Dataframe
            The dataframe to be saved
        file_path: str
            The path of the csv file
        
        Returns
        -------
        None
        """
        try:
            self._create_directory(os.path.dirname(file_path))
            data.to_csv(file_path, index=False, sep=',')
            self.logger.info(f'Saved csv file to {file_path}')
        except Exception as err:
            self.logger.error(f'Error occurred while writing file {file_path}: {err}')

    def _check_time_column_in_trace_keys(self, df, trace_keys):
        # Find invalid time columns in trace_keys
        error_time_keys = set(trace_keys) & (set(TIME_COLUMNS) - set(df.columns))   
        if error_time_keys:
            self.logger.warning(f'Trace keys contain invalid time columns: {list(error_time_keys)}')    
            # Find an available time column from df.columns to replace the error time column
            replace_time_keys = set(TIME_COLUMNS) & set(df.columns)
            if replace_time_keys:
                replace_time_key = list(replace_time_keys)[0]
                #TODO: Fix this warning. This is because the specified time column is not in the trace keys for vdf or some other dataframes
                self.logger.warning(f'Replacing with time column: {replace_time_key}')
                trace_keys = [replace_time_key if key in error_time_keys else key for key in trace_keys]
            else:
                self.logger.error("No valid time column found in dataframe.")          
        return trace_keys
    
    def _check_exp_std_column_in_trace_keys(self, df, trace_keys):
        # Find invalid time columns in trace_keys
        if ("aux_vdf_ldcstd_none_0" in trace_keys) and ("aux_vdf_ldcstd_none_0" not in df.columns):
            trace_keys = ["aux_vdf_ldcn_none_0" if (key=="aux_vdf_ldcstd_none_0") and (key not in df.columns) else key for key in trace_keys]
            self.logger.warning("Expansion stdev key was replaced with aux_vdf_ldcn_none_0.")          
        return trace_keys

    
    def load_df(self, test_folder=None, df_path=None, trace_keys=None):
        """
        Load the dataframe from the pickle file with the specified trace keys

        Parameters
        ----------
        test_folder: str
            The path of the test folder
        df_path: str, optional
            The path of the pickle file
        trace_keys: list of str, optional
            The list of keys of the traces to be loaded

        Returns
        -------
        Dataframe
            The dataframe loaded from the pickle file
        """ 
        if df_path is None:
            if test_folder is None:
                self.logger.error('Either test_folder or df_path must be specified')
                return None
            df_path = self.dirStructure.get_df_path(test_folder)
        df = self._load_pickle(df_path)
        if df is None:
            self.logger.error(f"DataFrame is None when attempting to load from {df_path}")
            return None
        if trace_keys is not None:
            try:
                trace_keys = self._check_time_column_in_trace_keys(df, trace_keys)
                # trace_keys = self._check_exp_std_column_in_trace_keys(df, trace_keys)

                keys_short=[]
                for key in trace_keys:
                    if key in df:
                        keys_short.append(key)
                    else:
                        df[key]=[np.nan]*len(df[trace_keys[1]])
                #df = df[keys_short]
                
                df = df[trace_keys]
            except KeyError as err:
                self.logger.error(f'Error occurred while loading dataframe: {err} with trace keys {trace_keys}')

                return None
            except TypeError:
                self.logger.error(f"DataFrame is None when attempting to filter by trace keys: {trace_keys}")
                return None
        return df
    
    def load_tr(self, tr_path):
        """
        Load the test record from the pickle file

        Parameters
        ----------
        tr_path: str
            The path of the pickle file

        Returns
        -------
        TestRecord
            The test record loaded from the pickle file
        """
        return self._load_pickle(tr_path)

    def load_trs(self, test_folders):
        """
        Load the test records based on the specified test folders

        Parameters
        ----------
        test_folders: list of str
            The list of paths of the test folders
        
        Returns
        -------
        list of TestRecord objects
            The list of test records loaded from the pickle files
        """
        tr_paths = [self.dirStructure.get_tr_path(test_folder) for test_folder in test_folders]
        return self._load_pickles(tr_paths)
    
    def load_dfs(self, test_folders):
        """
        Load the dataframes based on the specified test folders

        Parameters
        ----------
        test_folders: list of str
            The list of paths of the test folders
        
        Returns
        ------- 
        list of Dataframe    
            The list of dataframes loaded from the pickle files
        """
        df_paths = [self.dirStructure.get_df_path(test_folder) for test_folder in test_folders]
        return self._load_pickles(df_paths)
    
    def load_cycle_stats(self, test_folder):
        """
        Load the cycle status data from the pickle file

        Parameters
        ----------
        test_folder: str
            The path of the test folder
        
        Returns
        -------
        Dataframe
            The dataframe loaded from the pickle file
        """
        cycle_stats_path = self.dirStructure.get_cycle_stats_path(test_folder)
        return self._load_pickle(cycle_stats_path)

    def load_csv(self, file_path):
        """
        Load the csv file

        Parameters
        ----------
        file_path: str
            The path of the csv file
        
        Returns
        -------
        csv string
            The csv string of the csv file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                csv_string = f.read()
            self.logger.info(f"Loaded csv file from {file_path} successfully")
            return csv_string
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return None

    def _read_csv(self, file_path):
        """
        Read the csv file line by line

        Parameters
        ----------
        file_path: str
            The path of the csv file
        
        Returns
        -------
        iterator
            An iterator of the rows in the csv file
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                # Use csv.reader to read the file and convert it to a list of rows
                rows = list(csv.reader(f))
            self.logger.info(f"Loaded csv file from {file_path} successfully")
            return iter(rows)
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return None

    def read_sanity_check_csv(self):
        """
        Read the sanity check csv file

        Returns
        -------
        iterator
            An iterator of the rows in the csv file
        """
        return self._read_csv(SANITY_CHECK_CSV_PATH)
    
    def save_wrong_trs_name(self, wrong_tr_name):
        """
        Save the wrong test record name to the json file

        Parameters
        ----------
        wrong_tr_name: dict
            The dictionary of wrong test record name
        
        Returns
        -------
        None
        """
        self.dirStructure._save(WRONG_TR_NAME_PATH, wrong_tr_name)

    def load_processed_data(self, cell_name):
        """
        load the processed data from the processed folder

        Parameters
        ----------
        cell_name: str
            The name of the cell
        
        Returns
        -------
        cell_cycle_metrics: Dataframe
            The dataframe of cell cycle metrics
        cell_data: Dataframe
            The dataframe of cell data
        cell_data_vdf: Dataframe
            The dataframe of cell data vdf
        cell_data_rpt: Dataframe
            The dataframe of cell data rpt
        """
        cell_path = self.dirStructure.load_processed_dev_folder(cell_name)
        if cell_path is None:
            self.logger.warning(f"No test record for the {cell_name} in our network drive. Please check if the cell name is correct.")
            return None, None, None, None
        # Filepaths for cycle metrics, cell data, cell data vdf and rpt
        filepath_ccm = os.path.join(cell_path, 'CCM.pkl.gz')
        filepath_cell_data = os.path.join(cell_path, 'CD.pkl.gz')
        filepath_cell_data_vdf = os.path.join(cell_path, 'CDvdf.pkl.gz')
        filepath_rpt = os.path.join(cell_path, 'RPT.pkl.gz')
        # Load dataframes for cycle metrics, cell data, cell data vdf
        cell_cycle_metrics = self.load_df(df_path=filepath_ccm)
        cell_data = self.load_df(df_path=filepath_cell_data)
        cell_data_vdf = self.load_df(df_path=filepath_cell_data_vdf)
        cell_data_rpt = self.load_df(df_path=filepath_rpt)
        return cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt
    
    def save_processed_data(self, cell_name, cell_cycle_metrics,cell_data_rpt, cell_data = None, cell_data_vdf = None): # cell_data, cell_data_vdf, 
        """
        Save the processed data to the processed folder

        Parameters
        ----------
        cell_name: str
            The name of the cell
        cell_cycle_metrics: Dataframe
            The dataframe of cell cycle metrics
        cell_data: Dataframe
            The dataframe of cell data
        cell_data_vdf: Dataframe
            The dataframe of cell data vdf
        cell_data_rpt: Dataframe
            The dataframe of cell data rpt
        
        Returns
        -------
        None
        """
        # cell_path = self.dirStructure.load_processed_dev_folder(cell_name)
        # if cell_path is None:
        #     self.logger.warning(f"No test record for the {cell_name} in our network drive. Please check if the cell name is correct.")
        #     return None
        project_name = cell_name.split('_')[0]
        cell_path = os.path.join(ROOT_PATH, project_name, cell_name)
        # Filepaths for cycle metrics, cell data, cell data vdf and rpt
        filepath_ccm = os.path.join(cell_path, 'CCM.pkl.gz')
        filepath_cell_data = os.path.join(cell_path, 'CD.pkl.gz')
        filepath_cell_data_vdf = os.path.join(cell_path, 'CDvdf.pkl.gz')
        filepath_rpt = os.path.join(cell_path, 'RPT.pkl.gz')
        filepath_csv = os.path.join(cell_path, f'{cell_name}_CCM.csv')
        # Filepaths for duplicate cycle metrics
        ccm_path = self.dirStructure.load_ccm_folder()
        filepath_ccm_pkl = os.path.join(ccm_path, 'pkl', f'{cell_name}_CCM.pkl.gz')
        filepath_ccm_csv = os.path.join(ccm_path, 'csv', f'{cell_name}_CCM.csv')

        # Save dataframes and rename old if it already exists
        dfs = [cell_cycle_metrics]*4 + [cell_data_rpt, cell_data, cell_data_vdf]
        filepaths = [filepath_ccm, filepath_csv,filepath_ccm_pkl, filepath_ccm_csv, filepath_rpt,filepath_cell_data, filepath_cell_data_vdf]
        for i,(df, filepath) in enumerate(zip(dfs, filepaths)):
            if df is not None:
                #rename old as a copy with (#) appended
                # if os.path.exists(filepath):  
                    # path_parts = os.path.split(filepath)
                    # count = self.count_files_with_substring(path_parts[0], path_parts[1])
                    # last_modified_date = time.strftime(r"%Y%m%d_%H%M%S",time.strptime(time.ctime(os.path.getmtime(filepath))))
                    # tail_parts = path_parts[1].split(".")
                    # tail_parts[0] = tail_parts[0] + f'({last_modified_date})'
                    # filepath2 = os.path.join(cell_path, '.'.join(tail_parts)) 
                    # os.rename(filepath, filepath2)
                if '.csv' in filepath: 
                    self.save_df_to_csv(df, filepath)
                else:
                    self.save_df(df, filepath)

    def count_files_with_substring(self,directory, substring):
        count = 0
        for filename in os.listdir(directory):
            if substring in filename:
                count += 1
        return count

    def load_ccm_csv(self, cell_name):
        """
        Load the cycle metrics csv from the processed folder

        Parameters
        ----------
        cell_name: str
            The name of the cell

        Returns
        -------
        csv_string: str
            The csv string of the cycle metrics
        """
        cell_path = self.dirStructure.load_processed_dev_folder(cell_name)
        if cell_path is None:
            self.logger.warning(f"No test record for the {cell_name} in our network drive. Please check if the cell name is correct.")
            return None, None, None, None
        # Filepaths for cycle metrics, cell data, cell data vdf and rpt
        filepath_ccm = os.path.join(cell_path, 'CCM.pkl.gz') 
        csv_string = self.load_csv(filepath_ccm)
        return csv_string

    def get_calibration_parameters(self):
        """
        Get the calibration parameters

        Returns
        -------
        dict
            The dictionary of calibration parameters (device name to X1, X2, C)
        """
        self.logger.info(f"Loading calibration parameters from {SANITY_CHECK_CSV_PATH}")
        calibration_csv = self.read_sanity_check_csv()
        header = next(calibration_csv)
        # Get the index of the columns
        project_index, cell_name_index = header.index('Project'), header.index('Cell Name')
        x1_index, x2_index, c_index = header.index('X1'), header.index('X2'), header.index('C')
        start_date_index, removal_date_index = header.index('Start Date (Aging)'), header.index('Removal Date')
        # Set the default values for X1, X2, C
        default_X1, default_X2, default_C = X1, X2, C
        calibration_parameters = {}
        for row in calibration_csv:
            project, cell_number, start_date, removal_date = row[project_index], row[cell_name_index], row[start_date_index], row[removal_date_index]
            x1 = float(row[x1_index]) if row[x1_index] != '' else default_X1
            x2 = float(row[x2_index]) if row[x2_index] != '' else default_X2
            c = float(row[c_index]) if row[c_index] != '' else default_C
            cell_name = project + "_CELL" + cell_number.zfill(3)
            if cell_name not in calibration_parameters:
                calibration_parameters[cell_name] = [] # initialize key with cell name
            calibration_parameters[cell_name].append((start_date, removal_date, x1, x2, c))
        return calibration_parameters

    def save_figs(self, figs, cell_name, time_name, keep_open=False):
        """
        Save the figures to the processed folder

        Parameters
        ----------
        figs: list of Figure objects
            The list of figures to be saved
        cell_name: str
            The name of the cell
        time_name: str
            The name of the time slot
        
        Returns
        -------
        None
        """
        cell_path = self.dirStructure.load_processed_dev_folder(cell_name)
        if cell_path is None:
            self.logger.warning(f"No test record for the {cell_name} in our network drive. Please check if the cell name is correct.")
            return None
        # Filepaths for figures
        filepath_figs = os.path.join(cell_path, 'figs')
        # Save figures
        self._create_directory(filepath_figs)
        for i, fig in enumerate(figs):
            self.logger.info(f"Saving figure {i} to {filepath_figs}")
            fig.savefig(os.path.join(filepath_figs, f'{cell_name}_{time_name}_fig{i}.png'))
            if not keep_open:
                plt.close(fig)

    def _load_pickles(self, file_paths):
        return [self._load_pickle(file_path) for file_path in file_paths]

    def _load_pickle(self, file_path):
        if self.redisClient is not None:
            # Use the SHA256 hash of the file path as the key for Redis
            redis_key = hashlib.sha256(file_path.encode()).hexdigest()
            # Try to load from Redis cache first
            data_from_cache = self.redisClient.get_pickle(redis_key)
            if data_from_cache is not None:
                self.logger.info(f"Loaded data from Redis cache for {file_path}")
                return data_from_cache
        # If not found in Redis cache, load from local disk
        try:
            with gzip.open(file_path, "rb") as f:
                record = pickle.load(f)
            self.logger.info(f"Loaded pickle file from {file_path} successfully")
            # Save to Redis cache
            if self.redisClient is not None:
                self.logger.debug(f"Saving data to Redis cache for {file_path}")
                self.redisClient.set_pickle(redis_key, record)
            return record
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            return None
            
    def _check_folders(self):
        # Use path depth to decide which folders to consider.
        # For example, 'voltaiq_data/GMJuly2022/GMJuly2022_CELL102/' has a depth of 3.
        min_depth = len(self.rootPath.rstrip(os.sep).split(os.sep)) + 2
        empty_folders = []
        valid_folders = []

        for root, _, files in os.walk(self.rootPath):
            # Ignore the root directory itself
            if root == self.rootPath:
                continue
            # # Ignore the Processed data folder
            # TODO: Do not hardcode the folder name
            if 'voltaiq_data/PROCESSED' in root:
                continue
            if 'voltaiq_data/CCM' in root:
                continue
            # If this folder is not deep enough, we skip it
            depth = len(root.rstrip(os.sep).split(os.sep))
            if depth <= min_depth:
                continue
            # Check for the required files
            file_set = set(files)
            if "directory_structure.json" in file_set:
                continue
            elif "tr.pkl.gz" in file_set and "df.pkl.gz" in file_set:
                valid_folders.append(root)
            else:
                self.logger.warning(f"Folder {root} is not complete. It contains the following files: {files}")
                empty_folders.append(root)
        return empty_folders, valid_folders

    def merge_folders(self, src, dest):
        """
        Merge the source folder into the destination folder

        Parameters
        ----------
        src: str
            The path of the source folder
        dest: str
            The path of the destination folder
        
        Returns
        -------
        None
        """
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dest, item)
            # If item is a folder, recursively merge it
            if os.path.isdir(s):
                if not os.path.exists(d):
                    os.makedirs(d)
                self.merge_folders(s, d)
            else:
                if not os.path.exists(d):
                    shutil.copy2(s, d)
        # Remove the source folder
        shutil.rmtree(src)

    def move_tr(self, old_path, new_path):
        """
        Move the test record from the old path to the new path

        Parameters
        ----------
        old_path: str
            The path of the old test record
        new_path: str
            The path of the new test record
        
        Returns
        -------
        None
        """
        self.logger.info(f"Moving test record from {old_path} to {new_path}")
        try:
            shutil.move(old_path, new_path)
        except Exception as e:
            self.logger.error(f"Error moving test record from {old_path} to {new_path}: {e}")

    def copy_file(self, src, dest):
        """
        Copy the file from the source path to the destination path

        Parameters
        ----------
        src: str
            The path of the source file
        dest: str
            The path of the destination file
        
        Returns
        -------
        None
        """
        self.logger.info(f"Copying file from {src} to {dest}")
        try:
            shutil.copy2(src, dest)
        except Exception as e:
            self.logger.error(f"Error copying file from {src} to {dest}: {e}")
