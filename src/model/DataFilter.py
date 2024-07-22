from datetime import datetime

from src.utils.Logger import setup_logger
from src.model.DataIO import DataIO
from src.model.DirStructure import DirStructure
from src.config.time_config import DATE_FORMAT, TIME_TOLERANCE

#TODO: We don't need this class anymore, the functions can separate into DataIO and DirStructure
class DataFilter:
    """
    The class to filter data from the local disk

    Attributes
    ----------
    dataIO: DataIO object
        The object to save and load test data
    dirStructure: DirStructure object
        The object to manage the directory structure for the local data
    logger: logger object
        The object to log information
    
    Methods
    -------
    filter_records(device_id=None, tr_name_substring=None, start_time=None, tags=None)
        Filter the records with the specified device id or name or start time or tags
    filter_trs(device_id=None, tr_name_substring=None, start_time=None, tags=None)
        Filter the test records with the specified device id or name or start time or tags
    filter_dfs(device_id=None, tr_name_substring=None, start_time=None, tags=None)
        Filter the dataframes with the specified device id or name or start time or tags
    filter_trs_and_dfs(device_id=None, tr_name_substring=None, start_time=None, tags=None)
        Filter the test records and dataframes with the specified device id or name or start time or tags
    filter_df_by_tr(tr, trace_keys=None)
        Filter the dataframe with the specified test record
    """
    def __init__(self, dataIO: DataIO, dirStructure: DirStructure):
        self.dataIO = dataIO
        self.dirStructure = dirStructure
        self.logger = setup_logger()
    
    def filter_records(self, device_id=None, tr_name_substring=None, start_time=None, tags=None):
        """
        Fileter the records with the specified device id or name, and start time

        Parameters
        ----------
        device_id: str, optional
            The device id of the record to be found
        tr_name_substring: str, optional
            The substring of the device name of the record to be found
        start_time: str, optional
            The start time of the record to be found, in the format of 'YYYY-MM-DD_HH-MM-SS'
        tags: list of str, optional
            The list of tags of the record to be found
        
        Returns
        -------
        list of dict
            The list of records that match the specified device id or name, and start time
        """
        self.logger.info(f"Finding records with device_id={device_id}, tr_name_substring={tr_name_substring}, start_time={start_time}, tags={tags}") 
        if start_time is not None:
            start_time = datetime.strptime(start_time, DATE_FORMAT)

        dir_structure = self.dirStructure.load_records()

        matching_records = [
            record
            for record in dir_structure
            if (device_id is None or record['device_id'] == device_id) and
            (tr_name_substring is None or tr_name_substring.upper() in record['tr_name'].upper()) and
            (start_time is None or abs(datetime.strptime(record['start_time'], DATE_FORMAT) - start_time) <= TIME_TOLERANCE) and
            (tags is None or all(tag in record['tags'] for tag in tags)) and 
            (record['project_name'] != 'BAD TESTS')
        ]
        self.logger.info(f"Found {len(matching_records)} matching records")
        return matching_records

    def filter_trs(self, device_id=None, tr_name_substring=None, start_time=None, tags=None):
        """
        Filter the test records with the specified device id or name, and start time

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
        list of TestRecord object
            The list of test records that match the specified device id or name, and start time
        """
        #TODO: Change the pipeline
        matching_records = self.filter_records(device_id=device_id, tr_name_substring=tr_name_substring, start_time=start_time, tags=tags)
        matching_test_folders = [self.dirStructure.get_test_folder(record) for record in matching_records]
        return self.dataIO.load_trs(matching_test_folders)

    def filter_dfs(self, device_id=None, tr_name_substring=None, start_time=None, tags=None):
        """
        Filter the dataframes with the specified device id or name, and start time

        Parameters
        ----------
        device_id: str, optional
            The device id of the dataframe to be found
        tr_name_substring: str, optional
            The substring of the device name of the dataframe to be found
        start_time: str, optional
            The start time of the dataframe to be found, in the format of 'YYYY-MM-DD_HH-MM-SS'
        tags: list of str, optional
            The list of tags of the dataframe to be found
        
        Returns
        -------
        list of Dataframe
            The list of dataframes that match the specified device id or name, and start time
        """
        matching_records = self.filter_records(device_id=device_id, tr_name_substring=tr_name_substring, start_time=start_time, tags=tags)
        matching_test_folders = [self.dirStructure.get_test_folder(record) for record in matching_records]
        return self.dataIO.load_dfs(matching_test_folders)

    def filter_trs_and_dfs(self, device_id=None, tr_name_substring=None, start_time=None, tags=None):
        """
        Filter the test records and dataframes with the specified device id or name, start time and tags

        Parameters
        ----------
        device_id: str, optional
            The device id of the dataframe to be found
        tr_name_substring: str, optional
            The substring of the device name of the dataframe to be found
        start_time: str, optional
            The start time of the dataframe to be found, in the format of 'YYYY-MM-DD_HH-MM-SS'
        tags: list of str, optional
            The list of tags of the dataframe to be found
        
        Returns
        -------
        list of TestRecord object
            The list of test records that match the specified device id or name, start time and tags
        list of Dataframe
            The list of dataframes that match the specified device id or name, start time and tags
        """
        matching_records = self.filter_records(device_id=device_id, tr_name_substring=tr_name_substring, start_time=start_time, tags=tags)
        matching_test_folders = [self.dirStructure.get_test_folder(record) for record in matching_records]
        return self.dataIO.load_trs(matching_test_folders), self.dataIO.load_dfs(matching_test_folders)

    def filter_cycle_stats(self, device_id=None, tr_name_substring=None, start_time=None, tags=None):
        """
        Filter the cycle stats with the specified device id or name, and start time

        Parameters
        ----------
        device_id: str, optional
            The device id of the cycle stats to be found
        tr_name_substring: str, optional
            The substring of the device name of the cycle stats to be found
        start_time: str, optional
            The start time of the cycle stats to be found, in the format of 'YYYY-MM-DD_HH-MM-SS'
        tags: list of str, optional
            The list of tags of the cycle stats to be found
        
        Returns
        -------
        list of DataFrame
            The list of cycle stats that match the specified device id or name, and start time
        """
        matching_records = self.filter_records(device_id=device_id, tr_name_substring=tr_name_substring, start_time=start_time, tags=tags)
        matching_test_folders = [self.dirStructure.get_test_folder(record) for record in matching_records]
        return self.dataIO.load_cycle_stats(matching_test_folders[0])

    def filter_df_by_record(self, record, trace_keys=None):
        """
        Filter the dataframe with the specified test record

        Parameters
        ----------
        record: dict
            The test record to be found
        trace_keys: list of str, optional
            The list of trace keys to be found
        
        Returns
        -------
        DataFrame
            The dataframe that matches the specified test record
        """
        test_folder = self.dirStructure.get_test_folder(record)
        if test_folder == "":
            self.logger.warning(f"No dataframe found that matches test record {record['tr_name']}, need to update the local data")
            return None
        self.logger.info(f"Found dataframe that matches test record {record['tr_name']}")
        return self.dataIO.load_df(test_folder, trace_keys=trace_keys)
    
    def filter_cycle_end_times(self, record):
        """
        Filter the cycle end times with the specified test record

        Parameters
        ----------
        record: dict
            The test record to be found

        Returns
        -------
        list of str
            The list of cycle end times that matches the specified test record
        """
        test_folder = self.dirStructure.get_test_folder(record)
        cycle_stats = self.dataIO.load_cycle_stats(test_folder)
        return cycle_stats.cyc_end_datapoint_time 