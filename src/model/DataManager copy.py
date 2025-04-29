from src.model.DirStructure import DirStructure
from src.model.DataFetcher import DataFetcher
from src.model.DataIO import DataIO
from src.model.DataDeleter import DataDeleter
from src.model.DataFilter import DataFilter
from src.model.DataProcessor import DataProcessor
from src.utils.Logger import setup_logger
from src.utils.SinglentonMeta import SingletonMeta
from src.utils.DateConverter import DateConverter
from src.presenter.Presenter import Presenter
import os
import gc
import re


class DataManager(metaclass=SingletonMeta):
    """
    The class to manage all the local data

    Attributes
    ----------
    dataIO: DataIO object
        The object to save and load data
    dataFetcher: DataFetcher object
        The object to fetch data from Voltaiq Studio
    dataDeleter: DataDeleter object
        The object to delete data
    dataFilter: DataFilter object
        The object to filter data from the local disk
    dataProcessor: DataProcessor object
        The object to process data
    dirStructure: DirStructure object
        The object to manage the directory structure for the local data
    logger: logger object
        The object to log information
    
    Methods
    -------
    _createdb()
        Create the local database with all the test records and devices
    _updatedb(device_id=None, project_name=None, start_before=None, start_after=None)
        Update the local database with all the test records
    update_test_data(trs=None, devs=None, batch_size=60)
        Update the test data and directory structure with the specified test records and devices
    filter_trs(device_id=None, tr_name_substring=None, start_time=None, tags=None)
        Filter the test records locally with the specified device id or name or start time or tags
    filter_dfs(device_id=None, tr_name_substring=None, start_time=None, tags=None)
        Filter the dataframes locally with the specified device id or name or start time or tags
    filter_trs_and_dfs(device_id=None, tr_name_substring=None, start_time=None, tags=None)
        Filter the test records and dataframes locally with the specified device id or name or start time or tags
    check_and_repair_consistency()
        Check the consistency between the directory structure and local database, and repair the inconsistency
    process_tr(tr_name)
        Process the single test record
    process_cell(cell_name, numFiles = 1000, update_local_db=False, reset=False)
        Process the data for a cell and save the processed cell cycle metrics, cell data and cell data vdf to local disk
    process_project(project_name, numFiles = 1000)
        Process all the cells in a project and save the processed cell cycle metrics, cell data and cell data vdf to local disk
    save_figs(figs, cell_name, time_name)
        Save the figures to local disk, used by callback function
    load_processed_data(cell_name)
        Get the processed data for a cell
    load_ccm_csv(cell_name)
        Get the cycle metrics csv for a cell
    """
    _is_initialized = False
    def __init__(self, presenter: Presenter, use_redis=False):
        if DataManager._is_initialized:
            return
        self.dirStructure = DirStructure()
        self.dataFetcher = DataFetcher()
        self.dataDeleter = DataDeleter()
        self.dataIO = DataIO(self.dirStructure, self.dataDeleter, use_redis)
        self.dataFilter = DataFilter(self.dataIO, self.dirStructure)
        self.dateConverter = DateConverter()
        self.dataProcessor = DataProcessor(self.dataFilter, self.dirStructure, self.dateConverter)
        self.logger = setup_logger()
        self.presenter = presenter
        DataManager._is_initialized = True

    def _createdb(self):
        """
        Create the local database with all the test records and devices

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Fetch test records and devices
        trs = self.dataFetcher.fetch_trs()
        devs = self.dataFetcher.fetch_devs()
        
        if trs is None or devs is None:
            self.logger.error('Failed to fetch data')
            return
        
        # Create device folder dictionary
        devices_id, devices_name, projects_name = self.dataIO.create_dev_dic(devs)
        # Save test data and update directory structure
        self._update_batch_data(trs, devices_id, devices_name, projects_name)

    def _updatedb(self, device_id=None, project_name=None, start_before=None, start_after=None):
        """
        Update the local database with all the test records

        Parameters
        ----------
        device_id: int, optional
            The device id to be updated
        project_name: str, optional
            The project name to be updated
        start_before: str, optional
            The start time of test records to be updated should be earlier than this time, in the format of 'YYYY-MM-DD_HH-MM-SS'
        start_after: str, optional
            The start time of test records to be updated should be later than this time, in the format of 'YYYY-MM-DD_HH-MM-SS'        

        Returns
        -------
        None
        """
        # Fetch test records and devices
        trs = self.dataFetcher.fetch_trs()
        if device_id:
            trs = [tr for tr in trs if tr.device_id == device_id]
        if project_name:
            project_devices_id = self.dirStructure.project_to_devices_id(project_name)
            trs = [tr for tr in trs if tr.device_id in project_devices_id]
        if start_before:
            start_before = self.dateConverter._str_to_timestamp(start_before)
            trs = [tr for tr in trs if self.dateConverter._datetime_to_timestamp(tr.start_time) < start_before]
        if start_after:
            start_after = self.dateConverter._str_to_timestamp(start_after)
            trs = [tr for tr in trs if self.dateConverter._datetime_to_timestamp(tr.start_time) > start_after]
        self.logger.info(f'Find {len(trs)} test records meeting the criteria')
        devs = self.dataFetcher.fetch_devs()
        self.update_test_data(trs, devs, len(trs))
    
    def update_test_data(self, trs=None, devs=None, num_new_trs=60):
        """
        Update the test data and directory structure with the specified test records and devices

        Parameters
        ----------
        trs: list of TestRecord objects (optional)
            The list of test records to be saved
        devs: list of Device objects (optional)
            The list of devices to be saved
        num_new_trs: int
            The number of test records to be updated
        
        Returns
        -------
        None
        """
        existing_uuids = self.dirStructure.load_uuid()
        uuid_to_last_dp_timestamp = self.dirStructure.load_uuid_to_last_dp_timestamp()
        uuid_to_tr_df_cs_path = self.dirStructure.load_uuid_to_tr_df_cs_path()

        # Filter out existing test records
        new_trs = []

        # Process all the trs and devs if they are not provided
        if not trs:
            trs = self.dataFetcher.fetch_trs()
        if not devs:
            devs = self.dataFetcher.fetch_devs()

        for tr in trs:
            if tr.uuid not in existing_uuids:
                self.logger.info(f'New test record found: {tr.uuid}')
                new_trs.append(tr)
            else:
                last_dp_timestamp = uuid_to_last_dp_timestamp[tr.uuid]
                if last_dp_timestamp >= tr.last_dp_timestamp:   
                    self.logger.info(f'No new data found for test record {tr.uuid}') 
                    continue
                # Delete the old test data and update the directory structure
                old_tr_file, old_df_file, old_cs_file = uuid_to_tr_df_cs_path[tr.uuid]
                self.logger.info(f'Deleting old test data: {old_tr_file}, {old_df_file}')
                self.dataDeleter.delete_file(old_tr_file)
                self.dataDeleter.delete_file(old_df_file)
                self.dataDeleter.delete_file(old_cs_file)
                self.dirStructure.delete_record(tr.uuid)
                new_trs.append(tr)
                if len(new_trs) >= num_new_trs:
                    break

        if not new_trs:
            self.logger.info('No new test data found after filtering out existing records')
            return

        devices_id, devices_name, projects_name  = self.dataIO.create_dev_dic(devs)
        # Check the directory structure and update it
        self.dirStructure.check_project_name(devices_id, projects_name, devices_name)
        self.dirStructure.update_project_devices(devices_id, devices_name, projects_name)
        self._update_batch_data(new_trs, devices_id, devices_name, projects_name) 
    
    def _update_batch_data(self, new_trs, devices_id, devices_name, projects_name, batch_size=5):
        for i in range(0, len(new_trs), batch_size):
            new_trs_batch = new_trs[i:i+batch_size]
            # Get dataframes 
            dfs_batch = self.dataFetcher.get_dfs_from_trs(new_trs_batch)
            cycle_stats_batch = self.dataFetcher.get_cycle_stats_from_trs(new_trs_batch)
            # Save new test data and update directory structure
            self.dataIO.save_test_data_update_dict(new_trs_batch, dfs_batch, cycle_stats_batch, devices_id, devices_name, projects_name)
            gc.collect()

    def update_cycle_stats(self):
        """
        Update the cycle status for local recorded test records

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.logger.info('Updating cycle status...')
        uuid_to_tr_df_cs_path = self.dirStructure.load_uuid_to_tr_df_cs_path()
        for uuid, (tr_path, _, cycle_stats_path) in uuid_to_tr_df_cs_path.items():
            if not os.path.exists(cycle_stats_path):
                self.logger.info(f'Updating cycle status for test record {uuid}')
                tr = self.dataIO.load_tr(tr_path)
                cycle_stats = self.dataFetcher.get_cycle_stats(tr)
                self.dataIO._save_to_pickle(cycle_stats, cycle_stats_path)


    def filter_trs(self, device_id=None, tr_name_substring=None, start_time=None, tags=None):
        """
        Filter the test records locally with the specified device id or name, and start time

        Parameters
        ----------
        device_id: str, optional
            The device id of the test record to be found
        tr_name_substring: str, optional
            The substring of the device name of the test record to be found
        start_time: str, optional
            The start time of the test record to be found, in the format of 'YYYY-MM-DD_HH-MM-SS'
        tags: list of str, optional
            The list of tags of the test record to be found
        
        Returns
        -------
        list of test records
            The list of test records that match the specified device id or name, and start time and tags
        
        """
        return self.dataFilter.filter_trs(device_id, tr_name_substring, start_time, tags)
    
    def filter_dfs(self, device_id=None, tr_name_substring=None, start_time=None, tags=None):
        """
        Filter the dataframes locally with the specified device id or name, and start time

        Parameters
        ----------
        device_id: str, optional
            The device id of the dataframe to be found
        tr_name_substring: str, optional
            The substring of the device name of the dataframe to be found
        start_time: str, optional
            The start time of the dataframe to be found, in the format of 'YYYY-MM-DD_HH-MM-SS'
        
        Returns
        -------
        list of dataframes
            The list of dataframes that match the specified device id or name, and start time and tags
        """
        return self.dataFilter.filter_dfs(device_id, tr_name_substring, start_time, tags)
    

    def filter_trs_and_dfs(self, device_id=None, tr_name_substring=None, start_time=None, tags=None):
        """
        Filter the test records and dataframes locally with the specified device id or name, and start time

        Parameters
        ----------
        device_id: str, optional
            The device id of the dataframe to be found
        tr_name_substring: str, optional
            The substring of the tr name of the dataframe to be found
        start_time: str, optional
            The start time of the dataframe to be found, in the format of 'YYYY-MM-DD_HH-MM-SS'
        
        Returns
        -------
        list of test records
            The list of test records that match the specified device id or name, and start time and tags
        list of dataframes
            The list of dataframes that match the specified device id or name, and start time and tags
        """
        return self.dataFilter.filter_trs_and_dfs(device_id, tr_name_substring, start_time, tags)
    
    def check_and_repair_consistency(self):
        """
        Check the consistency between the directory structure and local database, and repair the inconsistency

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.logger.info('Starting consistency check between directory structure and local database...')

        # Step 1: Check if the device folders are in the corrosponding peoject folders
        self.logger.info('Checking if device folders are in the corrosponding peoject folders...')
        devs = self.dataFetcher.fetch_devs()
        devices_id, devices_name, projects_name = self.dataIO.create_dev_dic(devs)
        for dev in devs:
            if os.path.exists(os.path.join(self.dirStructure.rootPath, dev.name)):
                self.logger.warning(f'Found device folder {dev.name} not in the corrosponding peoject folder')
                src_folder = os.path.join(self.dirStructure.rootPath, dev.name)
                project_name = projects_name[devices_id.index(dev.id)]
                if project_name is None:
                    self.logger.error(f'No project name found for device {dev.name}')
                    project_name = 'UNKNOWN_PROJECT'
                self.logger.info(f'Moving device folder {dev.name} to project folder {project_name}')
                dst_folder = os.path.join(self.dirStructure.rootPath, project_name, dev.name)
                self.dataIO.merge_folders(src_folder, dst_folder)
        
        # Step 2: Check if the project name be recorded in the directory structure is the same as the project name in the tags
        self.dirStructure.check_project_name(devices_id, projects_name)
            
        # Step 3: Check for empty or incomplete folders and delete them.
        # TODO: The empty folders check is disabled for now
        empty_folders, valid_folders = self.dataIO._check_folders()
        # if empty_folders:
        #     self.logger.info(f'Empty or incomplete folders found: {empty_folders}')
        #     self.dataDeleter.delete_folders(empty_folders)

        # Convert to sets for easier operations
        valid_folders_set = set(valid_folders)
        recorded_folders_set = set(self.dirStructure.load_test_folders())

        # Step 4: Check for folders present on disk but not recorded in the directory structure.
        unrecorded_folders = valid_folders_set - recorded_folders_set
        if unrecorded_folders:
            self.logger.info(f'{len(unrecorded_folders)} folders not recorded in directory structure')
            trs = self.dataIO.load_trs(list(unrecorded_folders))
            for tr, test_folder in zip(trs, unrecorded_folders):
                if tr is None:
                    self.logger.info(f'No test record found for folder {test_folder}')
                    continue
                try:
                    dev_name = devices_name[devices_id.index(tr.device_id)]
                    project_name = projects_name[devices_id.index(tr.device_id)]
                except Exception as e:
                    self.logger.error(f'Error {e} while getting device name or project name for test record {tr.name} by device id {tr.device_id}')
                    continue
                if dev_name:
                    self.dirStructure.append_record(tr, dev_name, project_name)
                    self.logger.info(f'Appended record for folder {test_folder}')
            # self.dirStructure.save_dir_structure()

        # Step 5: Check for records in the directory structure that don't have corresponding folders on disk.
        # TODO: The orphaned records check is disabled for now
        # orphaned_records = recorded_folders_set - valid_folders_set
        # if orphaned_records:
        #     self.logger.info(f'Orphaned records found without corresponding folders on disk: {orphaned_records}')
        #     for orphaned_record in orphaned_records:
        #         # Delete the orphaned record from the directory structure based on its test_folder.
        #         self.dirStructure.delete_record(test_folder=orphaned_record)
        #         self.logger.info(f'Deleted {len(orphaned_records)} orphaned records from directory structure.')

        # Step 6: Check for local test records that are not consistent with the test records in Voltaiq Studio
        self.logger.info('Checking for local test records that are not consistent with the test records in Voltaiq Studio...')
        trs = self.dataFetcher.fetch_trs()
        tr_uuid_to_tr = {tr.uuid: tr for tr in trs}
        expired_folders = []
        for record in self.dirStructure.load_records():
            if record['uuid'] not in tr_uuid_to_tr:
                self.logger.error(f'Local test record {record["uuid"]} not found in Voltaiq Studio')
                # Delete the test record from the directory structure based on its uuid.
                expired_folders.append(self.dirStructure.get_test_folder(record))
                self.dirStructure.delete_record(record['uuid'])
            elif record['device_id'] != tr_uuid_to_tr[record['uuid']].device_id:
                self.logger.error(f'Local test record {record["uuid"]} has wrong device id')
                # Move the test record to the correct device folder and update the directory structure
                dev_name = devices_name[devices_id.index(tr_uuid_to_tr[record['uuid']].device_id)]
                project_name = projects_name[devices_id.index(tr_uuid_to_tr[record['uuid']].device_id)]
                if dev_name is None or project_name is None:
                    self.logger.error(f'No device name or project name found for test record {record["uuid"]}')
                    continue
                old_path = self.dirStructure.get_test_folder(record)
                new_path = os.path.join(self.dirStructure.rootPath, project_name, dev_name, record['start_time'])
                self.dataIO.move_tr(old_path, new_path)
                self.dirStructure.delete_record(record['uuid'])
                self.dirStructure.append_record(tr_uuid_to_tr[record['uuid']], dev_name, project_name)  
        if expired_folders:
            self.logger.info(f'Expired folders found: {expired_folders}')
            self.dataDeleter.delete_folders(expired_folders)
        self.logger.info('Consistency check completed.')

    def process_tr(self, tr_name):
        """
        Process the data for a test record and save the processed data to local disk

        Parameters
        ----------
        tr_name: str
            The name of the test record to be processed

        Returns
        -------
        None
        """
        cell_cycle_metrics, cell_data, cell_data_vdf = None, None, None
    
        records_neware = self.dataFilter.filter_records(tr_name_substring=tr_name, tags=['neware_xls_4000'])
        records_arbin = self.dataFilter.filter_records(tr_name_substring=tr_name, tags=['arbin'])
        records_biologic = self.dataFilter.filter_records(tr_name_substring=tr_name, tags=['biologic'])
        records_vdf = self.dataFilter.filter_records(tr_name_substring=tr_name, tags=['vdf'])     
        # Sort trs
        records_neware = self.dataProcessor.sort_records(records_neware)
        records_arbin = self.dataProcessor.sort_records(records_arbin)
        records_biologic = self.dataProcessor.sort_records(records_biologic)
        records_cycler = self.dataProcessor.sort_records(records_neware + records_arbin + records_biologic)
        records_vdf = self.dataProcessor.sort_records(records_vdf)

        # Get parameters for calibration
        calibration_parameters = None
        try:
            calibration_parameters = self.dataIO.get_calibration_parameters()
        except Exception as e:
            self.logger.error(f'Error {e} while getting calibration parameters')

        # Process data
        cell_cycle_metrics, cell_data, cell_data_vdf, project_name = self.dataProcessor.process_cell(records_cycler, records_vdf, cell_cycle_metrics, cell_data, cell_data_vdf, calibration_parameters)
        #Save new data to pickle if there was new data
        self.presenter.update(tr_name, cell_cycle_metrics, cell_data, cell_data_vdf, None, None, None)
        return cell_cycle_metrics, cell_data, cell_data_vdf


    def process_cell(self, cell_name, start_time=None, end_time=None, numFiles = 1000, reset = False):
        """
        Process the data for a cell and save the processed data to local disk

        Parameters
        ----------
        cell_name: str
            The name of the cell to be processed
        numFiles: int
            The number of files to be processed
        start_time: str, optional
            The start time of the test records to be processed, in the format of 'YYYY-MM-DD_HH-MM-SS'
        end_time: str, optional
            The end time of the test records to be processed, in the format of 'YYYY-MM-DD_HH-MM-SS'
        reset: bool
            Whether to reset the processed data
        
        Returns
        -------
        cell_cycle_metrics: dataframe
            The dataframe of cycle metrics for the cell
        cell_data: dataframe
            The dataframe of cell data for the cell
        cell_data_vdf: dataframe
            The dataframe of cell data vdf for the cell
        cell_data_rpt: dataframe    
            The dataframe of cell data rpt for the cell
        project_name: str
            Name of project that the cell belongs to
        """
        cell_name = cell_name.upper()
        cell_cycle_metrics, cell_data, cell_data_vdf = None, None, None
        if not reset:
            cell_cycle_metrics, cell_data, cell_data_vdf, _ = self.load_processed_data(cell_name)
    
        records_neware = self.dataFilter.filter_records(tr_name_substring=cell_name, tags=['neware_xls_4000'])
        records_arbin = self.dataFilter.filter_records(tr_name_substring=cell_name, tags=['arbin'])
        records_biologic = self.dataFilter.filter_records(tr_name_substring=cell_name, tags=['biologic'])
        records_vdf = self.dataFilter.filter_records(tr_name_substring=cell_name, tags=['vdf'])     
        # Sort trs
        records_neware = self.dataProcessor.sort_records(records_neware)
        records_arbin = self.dataProcessor.sort_records(records_arbin)
        records_biologic = self.dataProcessor.sort_records(records_biologic)
        records_cycler = self.dataProcessor.sort_records(records_neware + records_arbin + records_biologic)
        records_vdf = self.dataProcessor.sort_records(records_vdf)

        # Get parameters for calibration
        calibration_parameters = None
        try:
            calibration_parameters = self.dataIO.get_calibration_parameters()
        except Exception as e:
            self.logger.error(f'Error {e} while getting calibration parameters')

        # Process data
        project_name = self.dirStructure.cell_to_project(cell_name)
        cell_cycle_metrics, cell_data, cell_data_vdf, update = self.dataProcessor.process_cell(records_cycler, records_vdf, project_name, cell_cycle_metrics, cell_data, cell_data_vdf, calibration_parameters, numFiles)
        #Save new data to pickle if there was new data
        cell_data_rpt = None
        if update:
            self.logger.info(f'Updating processed data for cell {cell_name}...')
            # project_name = self.dirStructure.cell_to_project(cell_name)
            cell_data_rpt,cell_cycle_metrics = self.dataProcessor.summarize_rpt_data_2(cell_data, cell_data_vdf, cell_cycle_metrics, project_name)
            self.dataIO.save_processed_data(cell_name, cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt)

        # Present the data
        fig_1, fig_2, fig_3 = self.presenter.update(cell_name, cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt, start_time, end_time)
        time_name = f"{start_time}To{end_time}" if start_time and end_time else ""

        self.save_figs([fig_1, fig_2, fig_3], cell_name, time_name, keep_open=True)

        return cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt, project_name
    
    def process_project(self, project_name, numFiles = 1000):
        """
        Process the data for a project and save the processed data to local disk

        Parameters
        ----------
        project_name: str
            The name of the project to be processed
        numFiles: int
            The number of files to be processed for each cell

        Returns
        -------
        None
        """
        cells_name = self.dirStructure.project_to_devices_name(project_name)
        for cell_name in cells_name:
            _, _, _, _ = self.process_cell(cell_name, numFiles)


    def save_figs(self, figs, cell_name, time_name, keep_open=False):
        self.dataIO.save_figs(figs, cell_name, time_name, keep_open)

   
    def load_processed_data(self, cell_name):
        """
        Get the processed data for a cell

        Parameters
        ----------
        cell_name: str
            The name of the cell to be processed

        Returns
        -------
        cell_cycle_metrics: dataframe
            The dataframe of cycle metrics for the cell
        cell_data: dataframe
            The dataframe of cell data for the cell
        cell_data_vdf: dataframe
            The dataframe of cell data vdf for the cell
        cell_data_rpt: dataframe
            The dataframe of cell data rpt for the cell
        """
        return self.dataIO.load_processed_data(cell_name)
    
    def load_ccm_csv(self, cell_name):
        """
        Get the cycle metrics csv for a cell

        Parameters
        ----------
        cell_name: str
            The name of the cell to be processed

        Returns
        -------
        ccm_csv: str
            The csv string of the cycle metrics for the cell
        """
        return self.dataIO.load_ccm_csv(cell_name)

    def sanity_check(self):
        self.logger.info('Starting sanity check...')
        trs = self.dataFetcher.fetch_trs()
        # Read sanity check csv line by line
        sanity_csv = self.dataIO.read_sanity_check_csv()
        header = next(sanity_csv)
        # Get the index of the columns
        project_index, cell_name_index, channel_index = header.index('Project'), header.index('Cell Name'), header.index('Channel')
        start_date_index, removal_date_index = header.index('Start Date (Aging)'), header.index('Removal Date')
        
        wrong_trs = {}

        for row in sanity_csv:
            try:
                # Get the test records for the cell
                project, cell_number, correct_channel = row[project_index], row[cell_name_index], row[channel_index]
                start_date, removal_date = row[start_date_index], row[removal_date_index]
                cell_name = project + "_CELL" + cell_number.zfill(3)
                cell_trs = [tr for tr in trs if tr.name.startswith(cell_name)]
                # Check if there is overlapping between the time range of the test records and the time range in the sanity check csv
                if start_date:
                    start_date = self.dateConverter._str_to_timestamp(self.dateConverter._format_date_str(start_date))
                    cell_trs = [tr for tr in cell_trs if tr.last_dp_timestamp >= start_date]
                if removal_date:
                    removal_date = self.dateConverter._str_to_timestamp(self.dateConverter._format_date_str(removal_date))
                    cell_trs = [tr for tr in cell_trs if self.dateConverter._datetime_to_timestamp(tr.start_time) <= removal_date]
                neware_trs = [tr for tr in cell_trs if 'neware_xls_4000' in tr.tags]
                arbin_trs = [tr for tr in cell_trs if 'arbin' in tr.tags]
            except Exception as e:
                self.logger.error(f'Error {e} while processing row {row}')
                continue
            
            # Check the neware trs
            for tr in neware_trs:
                try:
                    channel = tr.name.split("_")[-4] + tr.name.split("_")[-3] + '-' +tr.name.split("_")[-2]
                    if channel != correct_channel:
                        self.logger.warning(f'Neware tr: {tr.name} has wrong neware rack or channel')
                        wrong_trs[tr.name] = [correct_channel]
                except Exception as e:
                    self.logger.error(f"Error while checking neware tr: {tr.name}, {e}")
                    continue

            # Check the arbin trs
            for tr in arbin_trs:
                try:
                    comments = tr.comments
                    comments_str = ''.join([str(comment) for comment in comments])
                    match = re.search(r'Channel Index: #(\d+)', comments_str)
                    channel_idx = int(match.group(1)) if match else None
                    if channel_idx != correct_channel:
                        self.logger.warning(f'Arbin tr: {tr.name} has wrong channel index')
                        wrong_trs[tr.name] = [correct_channel]
                except Exception as e:
                    self.logger.error(f"Error while checking arbin tr: {tr.name}, {e}")
                    continue

        self.logger.info(f'{len(wrong_trs)} wrong trs found')  
        # Save the wrong trs to json
        self.dataIO.save_wrong_trs_name(wrong_trs)

    def duplicate_ccm(self):
        """
        Duplicate the ccm csv and pkl.gz files into the CCM folder in Processed for all cells
        """
        self.logger.info('Starting duplicating ccm csv and pkl.gz files...')
        processed_folder = self.dirStructure.load_processed_folder()
        ccm_folder = self.dirStructure.load_ccm_folder()
        pkl_folder = os.path.join(ccm_folder, 'pkl')
        csv_folder = os.path.join(ccm_folder, 'csv')
        # Create the CCM folder if it does not exist
        os.makedirs(pkl_folder, exist_ok=True)
        os.makedirs(csv_folder, exist_ok=True)

        # Walk through the Processed folder
        for subdir, dirs, files in os.walk(processed_folder):

            # Skip the ccm folder
            if os.path.commonpath([subdir, ccm_folder]) == ccm_folder:
                continue

            for filename in files:
                # Check if the file is a ccm csv or pkl.gz file
                if 'CCM' not in filename:
                    continue

                # Get the file extension
                file_extension = os.path.splitext(filename)[1].lower()

                # Check if the file extension is '.gz' or '.csv'
                if file_extension in ['.gz', '.csv']:
                    # Get the parent directory name for pkl.gz files
                    if file_extension == '.gz' and filename.endswith('.pkl.gz'):
                        parent_dir_name = os.path.basename(subdir)
                        target_filename = f"{parent_dir_name}_{filename}"
                        target_subfolder = pkl_folder
                    elif file_extension == '.csv':
                        target_subfolder = csv_folder
                        target_filename = filename
                    else:
                        continue

                    # Get the target file path
                    source_file = os.path.join(subdir, filename)
                    target_file = os.path.join(target_subfolder, target_filename)

                    # Copy the file to the target folder
                    self.dataIO.copy_file(source_file, target_file)
        self.logger.info('Duplicating ccm csv and pkl.gz files completed.')

    def clean_unknown_project(self):
        """
        Clean the unknown project folder, and move the device folders to the correct project folder.
        Also update the directory structure.
        """
        self.logger.info('Starting cleaning unknown project folder...')
        proj_to_dev_id_name = self.dirStructure.load_project_devices()
        print(proj_to_dev_id_name.keys())
        dev_to_id_proj = {}
        # Wrong id to correct id and project
        wrong_id_to_id_proj = {}
        for proj, devs_id_name in proj_to_dev_id_name.items():
            if proj == 'UNKNOWN_PROJECT':
                continue
            for dev_id, dev_name in devs_id_name:
                dev_to_id_proj[dev_name] = (dev_id, proj)
        print(dev_to_id_proj['GMJULY2022_CELL093'])
        # Check the unknown project
        for dev_id_name in proj_to_dev_id_name['UNKNOWN_PROJECT']:
            wrong_id, dev_name = dev_id_name[0], dev_id_name[1]
            possible_proj = dev_name.split('_')[0].upper()
            if dev_name == 'GMJULY2022_CELL093':
                print(f"!!!GMJULY2022_CELL093: {dev_id}, {proj}")
            if dev_name in dev_to_id_proj or possible_proj in proj_to_dev_id_name:
                if dev_name in dev_to_id_proj:
                    dev_id, proj = dev_to_id_proj[dev_name]
                else:
                    dev_id, proj = wrong_id, possible_proj
                if dev_name == 'GMJULY2022_CELL093':
                    print(f"!GMJULY2022_CELL093: {dev_id}, {proj}")
                wrong_id_to_id_proj[wrong_id] = (dev_id, proj)
                self.logger.info(f'Moving device folder {dev_name} to project folder {proj}')
                src_folder = os.path.join(self.dirStructure.rootPath, 'UNKNOWN_PROJECT', dev_name)
                if not os.path.exists(src_folder):
                    # self.logger.warning(f'Device folder {dev_name} not found in UNKNOWN_PROJECT folder')
                    continue
                dst_folder = os.path.join(self.dirStructure.rootPath, proj, dev_name)
                self.dataIO.merge_folders(src_folder, dst_folder)
                # Delete the record in the project_devices.json

        # Check the directory structure and update it
        self.dirStructure.fix_unknown_project(wrong_id_to_id_proj)

    # Below are the methods for testing
    def test_createdb(self):
        """
        Create the local database for testing

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Fetch test records and devices
        trs = self.dataFetcher.fetch_trs()
        devs = self.dataFetcher.fetch_devs()

        test_trs = trs[:50]
        
        if trs is None or devs is None:
            self.logger.error('Failed to fetch data')
            return
        
        # Create device folder dictionary
        devices_id, devices_name, projects_name = self.dataIO.create_dev_dic(devs)
        # Save test data and update directory structure
        self._update_batch_data(test_trs, devices_id, devices_name, projects_name)
