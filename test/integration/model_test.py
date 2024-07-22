import sys,os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath("__file__")))
if os.name=="nt":
    sys.path.append(os.path.dirname(os.path.abspath("__file__"))+"\\src")
else:
    sys.path.append(os.path.dirname(os.path.abspath("__file__"))+"/src")


#os.environ['DISPLAY'] = "localhost:10.0" 

from src.model.DataManager import DataManager
from src.presenter.Presenter import Presenter 
from src.viewer.Viewer import Viewer
from matplotlib import pyplot as plt
import matplotlib   
#matplotlib.use('QtAgg')
matplotlib.use('TkAgg')
#matplotlib.use('GTK3Agg') 
# Importing gc module
import gc

# Integration test
def createdb_test():
    dataManager = DataManager()
    # test create db
    dataManager.test_createdb()#

def filter_test():
    dataManager = DataManager()
    # test filter
    trs, dfs = dataManager.filter_trs_and_dfs(tr_name_substring='GMJuly2022_CELL018')
    for tr in trs:
        print(tr)
        print(dfs)
    # dfs = dataManager.filter_dfs(tr_name_substring='GMJuly2022_CELL018')
    # for df in dfs:
    #     print(df.columns)

def consistency_test():
    dataManager = DataManager()
    # test consistency
    dataManager.check_and_repair_consistency()

def process_cell_test():
    # test process_cell
    umbl_cell_nums_neware = [152065,152041,152046,152057,152078,152033,152088,152036,152084,152085,152047,152072,152048, 152097, 152091,152051,152087,152042,152094,152045,152054,152039,]
    umbl_cell_nums_arbin = [152066, 152060, 152069,152061, 152093,152076,152052,152079,152089,152063,152068,152075,152055,152050,152067,152077,152059,152056,152058,152053,152082] #152070, 152062, 152086
    umbl_cell_nums = [152048] #umbl_cell_nums_arbin + umbl_cell_nums_neware 
    # done = 152070, 152062, 152086
    # LLI/LAM error cells
    # umbl_cell_nums = [152056, 152059, 152077, 152052, 152055, 152068, 152075, 152079, 152082, 152050, 152062, 152067, 152039]
    umbl_cells = ['UMBL2022FEB' +'_CELL' + f'{cell:03d}' for cell in umbl_cell_nums]
    # cell_names = ["GMFEB23S_CELL003"]
    # cell_names=["GMJuly2022_CELL"+f'{cell_num:03d}' for cell_num in range(21,50) ]
    # cell_names=["GMFEB23S_CELL"+f'{cell_num:03d}' for cell_num in range(5,77) ]
    cell_names = umbl_cells

    cell_errors = [] # init list of dict with cell name and error message
    for cell_name in cell_names:
        try:
            viewer = Viewer()
            presenter = Presenter(viewer=viewer)
            dataManager = DataManager(presenter=presenter ,use_redis=False)

            # test process_cell

            cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt, junk = dataManager.process_cell(cell_name, reset=True);#, reset=True)#, start_time='2023-07-01_00-00-00', end_time='2023-07-28_23-59-59')
        #    cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt = dataManager.process_cell(cell_name, reset=True, start_time='2022-09-13_10-00-00', end_time='2022-09-24_10-00-00');#, reset=True)#)
        except Exception as error:
            print("An exception occurred:", error) 
            cell_error = {'name': cell_name, 'error': error}
            cell_errors.append(cell_error) # make a list of unprocessed cells and errors

        # Display cells that couldn't be processed and their errors
        df_errors = pd.DataFrame.from_dict(cell_errors)
        print(df_errors)

        # Returns the number of objects it has collected and deallocated
        collected = gc.collect()
        
        # Prints Garbage collector 
        # as 0 object
        print("Garbage collector: collected",
                "%d objects." % collected)


def process_tr_test():
    tr_name = 'GMJuly2022_CELL025_Test7B_1_P0C_5P0PSI_20230412_R0_CH055'
    viewer = Viewer()
    presenter = Presenter(viewer=viewer)
    dataManager = DataManager(presenter=presenter ,use_redis=False)
    cell_cycle_metrics, cell_data, cell_data_vdf = dataManager.process_tr(tr_name)
    print(cell_data)
    print(cell_data_vdf)
    print(cell_cycle_metrics)
    plt.show()

def read_csv_test():
    dataManager = DataManager()
    # test read_csv
    ccm_csv = dataManager.load_ccm_csv("GMJuly2022_CELL050")
    print(ccm_csv)
    with open("V:\\voltaiq_data\\Processed\\GMJuly2022\\output.csv", "w", encoding="utf-8") as f:
        f.write(ccm_csv)

def update_cycle_stats_test():
    dataManager = DataManager()
    # test update_cs
    dataManager.update_cycle_stats()

def sanity_check_test():
    dataManager = DataManager()
    # test sanity_check
    dataManager.sanity_check()

def duplicate_ccm_test():
    dataManager = DataManager()
    # test duplicate_ccm
    dataManager.duplicate_ccm()


def clean_unknown_project():
    dataManager = DataManager()
    dataManager.clean_unknown_project()

if __name__ == '__main__':
    process_cell_test()

