import sys
sys.path.insert(0, '/Users/yiliu/Documents/GitHub/UMBC_Data_Localization/UMBC_Data_Localization/src')


from src.model.DataManager import DataManager
from src.presenter.Presenter import Presenter 
from src.viewer.Viewer import Viewer

# Integration test
def present_cell():
    dataManager = DataManager(use_redis=True)
    def save_figs(figs, cell_name):
        dataManager.save_figs(figs, cell_name)
    presenter = Presenter()
    viewer = Viewer(call_back=save_figs)
    cell_name = "UMBL2022FEB_CELL152051"
    dataManager.attach(presenter)
    presenter.attach(viewer)

    # test process_cell
    cell_cycle_metrics, cell_data, cell_data_vdf, cell_data_rpt = dataManager.process_cell(cell_name, start_time='2023-07-01_00-00-00', end_time='2023-07-28_23-59-59')
if __name__ == '__main__':
    present_cell()