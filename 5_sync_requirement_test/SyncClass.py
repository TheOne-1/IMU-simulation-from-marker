from const import RESULT_PATH, ALL_SUB_FILE, PROCESSED_DATA_PATH, SEGMENT_NAMES, SPEEDS
from SubjectData import SubjectData
from XYGenerator import XYGeneratorUni
from OffsetClass import MultiAxisOffset
import pickle
import pandas as pd
import numpy as np


def get_result(input_names, output_names, result_column, i_sub, model_name, my_delays):
    delays = my_delays.get_delays()
    delay_ms = my_delays.get_delay_ms()
    file_path = RESULT_PATH + 'evaluator\\' + model_name + '\\' + model_name + '_subject_' + str(i_sub) + '.pkl'
    with open(file_path, 'rb') as fp:
        my_evaluator = pickle.load(fp)

    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    total_sub_df = pd.DataFrame()
    test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub]
    del all_sub_data
    for speed in SPEEDS:
        test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
        x_test = test_speed_data[input_names]
        y_test = test_speed_data[output_names].as_matrix()
        x_test, y_test = delay_data(x_test, y_test, delays)
        y_pred = my_evaluator.evaluate_sklearn(x_test, y_test)
        R2, RMSEs, NRMSEs = np.zeros([1, len(output_names)]), np.zeros([1, len(output_names)]), np.zeros([1, len(output_names)])
        R2[:], RMSEs[:], NRMSEs[:] = my_evaluator.get_all_scores(y_test, y_pred)
        evaluation_result = np.column_stack([R2, RMSEs, NRMSEs])
        scores_df = pd.DataFrame(evaluation_result, columns=result_column)
        scores_df = scores_df.reset_index(drop=True)
        scores_df.insert(loc=0, column='speed', value=str(speed))
        scores_df.insert(loc=0, column='subject_id', value=i_sub)
        scores_df.insert(loc=0, column='delay_ms', value=delay_ms)
        total_sub_df = pd.concat([total_sub_df, scores_df], axis=0)

    return total_sub_df

def delay_data(x_test, y_test, delays):
    max_delay = np.max(np.abs(delays)) + 1
    x_test = x_test.as_matrix()
    # cut the data
    x_test_new = x_test[max_delay:-max_delay, :].copy()
    y_test_new = y_test[max_delay:-max_delay, :].copy()
    for i_sensor in range(len(SEGMENT_NAMES)):
        delay = delays[i_sensor]
        x_test_new[:, i_sensor*3:(i_sensor+1)*3] = x_test[delay+max_delay:delay-max_delay, i_sensor*3:(i_sensor+1)*3]

    for i_sensor in range(len(SEGMENT_NAMES)):
        delay = delays[i_sensor]
        x_test_new[:, (8+i_sensor)*3:(9+i_sensor)*3] = x_test[delay+max_delay:delay-max_delay, (8+i_sensor)*3:(9+i_sensor)*3]

    return x_test_new, y_test_new
