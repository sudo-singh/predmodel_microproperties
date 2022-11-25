from extractor import RawDataCreation
from fit_to_constants import FitToConstants
from gradiennt_boosting import GradientBoosting

recreate_data = input('Do you want to re-create all the data from scratch?')
if recreate_data == 'y':
    print('Gathering data from the simulation results in one excel sheet..........')
    rdc = RawDataCreation()
    rdc.create_master_excel()
    print('Simulation data has been updated in raw_data.xlsx..........')
    print("Gathering Buckingham constants..........")
    ftd = FitToConstants()
    ftd.set_r_squared_data()
    ftd.export_master_frame_to_sheet()

gb = GradientBoosting()
gb.set_working_frame()
gb.feature_label_splitter()
gb.test_train_split()
gb.standard_normalize()
gb.rfr()
# gb.boost()
gb.set_unseen_frame()
gb.normalize_unseen()
gb.predict()
gb.export_results()
