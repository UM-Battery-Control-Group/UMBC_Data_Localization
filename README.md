# UMBC_Data_Localization

This project seamlessly fetches data from Voltaiq and stores it locally, offering an efficient solution for managing and processing test records and device data. Additionally, it features functionality to visualize the processed results.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setting up a Virtual Environment](#setting-up-a-virtual-environment)
3. [Installing Dependencies](#installing-dependencies)
4. [Voltaiq Env](#voltaiq-env)
5. [Usage](#usage)
6. [License](#license)
7. [Questions](#questions)

## Prerequisites

- Python 3.7 or higher
- pip (comes with Python)
- Redis
- Voltaiq Studio token

## Setting up a Virtual Environment

Virtual environments allow you to manage project-specific dependencies, which can prevent conflicts between versions.

1. **Install `virtualenv`** (If not installed)

    ```bash
    pip install virtualenv
    ```

2. **Navigate to your project directory**:

    ```bash
    cd /path/to/your/project
    ```

3. **Create a virtual environment**:

    ```bash
    virtualenv venv
    ```

4. **Activate the virtual environment**:

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

    - On Windows:

        ```bash
        .\venv\Scripts\activate
        ```

    After activation, your command prompt should show the name of the virtual environment (`venv` in this case).

## Installing Dependencies

Once the virtual environment is activated, you can install the project's dependencies.

```bash
pip install -r requirements.txt
```

Then install Redis (If you're using Windows or prefer not to use this feature, please skip this section.):

**On macOS:**

```bash
brew install redis
```

**On Linux:**

```bash
sudo apt update
sudo apt install redis-server
```

**On Windows:**
    Check this web: https://redis.io/docs/getting-started/installation/install-redis-on-windows/#:~:text=Redis%20is%20not%20officially%20supported,Linux%20binaries%20natively%20on%20Windows.
        

## Voltaiq Env

Add your voltaiq studio token in the first line of the .env file

## Usage

1. Connect our network drive to your PC.  
2. Navigate to 'src/config/path_config.py'.  
3. Update the 'ROOT_PATH' value to reflect the path of the newly connected network drive.  


### Sample Useage:
```python
    dataManager = DataManager(use_redis=True)
    def save_figs(figs, cell_name, time_name):
        dataManager.save_figs(figs, cell_name, time_name)
    presenter = Presenter()
    viewer = Viewer(call_back=save_figs)
    cell_name = "UMBL2022FEB_CELL152051"
    dataManager.attach(presenter)
    presenter.attach(viewer)
    cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt = dataManager.process_cell(cell_name)
```

### Update local database
**Jason will handle updating our local database daily, so we won't need to do it ourselves.**  

Use the file src/updatedb.py, in this line, you can specify the device id and start time of the test record you want to update. Or you can leave the parameter empty, it will update all the database.
```
dataManager._updatedb(device_id= 1778, start_before='2023-06-22_23-59-59',start_after='2023-06-22_00-00-00')
```
Or we can update by project:
```
dataManager._updatedb(project_name='GMJuly2022')
```
### Directory Structure
#### Folder Structure
The folder structure of voltaiq data looks like this, the tr file is the metadata, and df file is the real data:
```
voltaiq_data/
|-- directory_structure.json
|-- project_devices.json
|-- project_1
|   |-- cell_1/
|       |-- test_start_time_1/
|           |-- tr.pkl.gz
|           |-- df.pkl.gz
|           |-- cycle_stats.pkl.gz                                     
|       |-- test_start_time_2/
|           |-- tr.pkl.gz
|           |-- df.pkl.gz
|           |-- cycle_stats.pkl.gz   
|       |   |-- ...
|   |-- cell_2/
|       |-- test_start_time_1/
|           |-- tr.pkl.gz
|           |-- df.pkl.gz
|           |-- cycle_stats.pkl.gz   
|       |-- test_start_time_2/
|           |-- tr.pkl.gz
|           |-- df.pkl.gz
|           |-- cycle_stats.pkl.gz   
|   |   |-- ...
|   |-- ...
|-- project_N
|   |-- cell_N/
|       |-- test_start_time_1/
|       |   |-- tr.pkl.gz
|       |   |-- df.pkl.gz
|           |-- cycle_stats.pkl.gz   
|       |-- ...
```
#### directory_structure.json 
This file contains the useful metadata for us to locate the real data, the structure of this file looks like this: 
```
{
        "uuid": "f91ca7b0-dde6-4743-b68a-c6cbd22984ec",
        "device_id": 17154,
        "tr_name": "GMFEB23S_CELL009_RPT_6_P25C_15P0PSI_20230815_R0-01-008",
        "dev_name": "GMFEB23S_CELL009",
        "start_time": "2023-08-15_08-57-21",
        "last_dp_timestamp": 1692369188000,
        "tags": [
            "Test Type: Reference Performance Test",
            "Procedure Version: 6",
            "arbin",
            "Temperature: 25C",
            "Run Number: R0-01-008",
            "Test Date: 20230815",
            "Pressure: 15.0 PSI"
        ]
    }, ...
```


#### project_devices.json 
This file maps project names to their corresponding device names and IDs.
```
{
    "GMJuly2022": [
            [
                3110,
                "GMJuly2022_CELL087"
            ],
            [
                3594,
                "GMJuly2022_CELL034"
            ],
            ...]
    "UMBL2022FEB": [
        [
            3255,
            "UMBL2022FEB_CELL152057"
        ],
        [
            3257,
            "UMBL2022FEB_CELL152088"
        ],
        ...]
    ...
    }, ...
```

### DataManager Usage
DataManager is a robust utility class designed to manage local data, ensuring seamless interaction with the Voltaiq Studio. It encompasses functions to fetch, delete, update, filter, and process data pertaining to test records and devices. 

##### Initialization
If have trouble with using Redis:
```python
manager = DataManager()
```
If you want to use Redis as local cache:
```python
manager = DataManager(use_redis=True)
```

##### Check consistency
 Check the consistency between the directory structure and local database, and repair the inconsistency
```python
manager.check_and_repair_consistency()
```
##### Filtering

Filter test records based on certain parameters:

```python
filtered_test_records = manager.filter_trs(device_id="your_device_id")
```

Similarly, for filtering dataframes:

```python
filtered_dataframes = manager.filter_dfs(tags="your_tags")
```

Or, to filter both:

```python
test_records, dataframes = manager.filter_trs_and_dfs(tr_name_substring="your_tr_name_substring")
```

The processed data and generated image can also be find in the Processed folder in our network drive.  
Please make sure process the cell first, then try to get the processed data. 

```python
cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt = load_processed_data(cell_name)
```

Get the csv file of the cell cycle metrics:

```python
cell_name = "your_cell_name"
ccm_csv = dataManager.load_ccm_csv(cell_name)
```  

You can find the generated image at:   
".../voltaiq_data/Processed/*project name*/*cell name*/figs/"

##### Data Processing

To process data for a specific cell:

```python
cell_name = "your_cell_name"
cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt = manager.process_cell(cell_name)
```
You can also process the cell data based on project, it will return nothing but the processed data will be saved to our network drive:

```python
project_name = "your_project_name"
manager.process_cell(project_name)
```

If you want to process the single test record:
```python
tr_name = "your_tr_name"
cell_cycle_metrics, cell_data, cell_data_vdf = manager.process_tr(tr_name)
plt.show()
```

The data and figures generated by this process will be saved at:   
".../voltaiq_data/Processed/*project name*/*cell name*/"


### Presenter

The Presenter class is designed to manage the presentation of data to a frontend. It works in tandem with a data manager to handle data processing and querying. 

#### Initialization

Create an instance of the Presenter, then attach it to the dataManager:

```python
presenter = Presenter()
dataManager.attach(presenter)
```

### Viewer

The Viewer class is a utility designed to visualize data from a local disk pertaining to cells, particularly their timeseries data, expansions, cycle metrics, and index metrics. This class is a comprehensive tool for researchers, engineers, and anyone interested in analyzing and visualizing cell data.

#### Example useage


Add the call back function when creating the Viewer, the figures will be saved by datamanager:
```python
def save_figs(figs, cell_name, time_name):
    dataManager.save_figs(figs, cell_name, time_name)
viewer = Viewer(call_back=save_figs)
presenter.attach(viewer)
```
#### Recommendations
While the default downsample value is set to 100, users can adjust this parameter for finer or coarser visualizations as required.

## License
This project is licensed under the MIT License.

## Questions

If you have any questions or encounter any bugs, please raise an issue in our repository or send an email to ziyiliu@umich.edu. We greatly appreciate your feedback.