import pandas as pd
import multiprocessing
from EvaluationClass import Evaluation
from const import SUB_NUM, SEGMENT_NAMES, MOCAP_SAMPLE_RATE
from DelayClass import RandomDelayClass
from SyncClass import get_result

if __name__ == '__main__':
    output_names = [
        'FP1.ForX',
        'FP2.ForX',
        'FP1.ForY',
        'FP2.ForY',
        'FP1.ForZ',
        'FP2.ForZ'
    ]

    R2_column = [output + '_R2' for output in output_names]
    RMSE_column = [output + '_RMSE' for output in output_names]
    NRMSE_column = [output + '_NRMSE' for output in output_names]
    result_column = R2_column + RMSE_column + NRMSE_column

    input_names = [
        'trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z',
        'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z',
        'l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z',
        'r_thigh_acc_x', 'r_thigh_acc_y', 'r_thigh_acc_z',
        'l_shank_acc_x', 'l_shank_acc_y', 'l_shank_acc_z',
        'r_shank_acc_x', 'r_shank_acc_y', 'r_shank_acc_z',
        'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z',
        'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z',
        'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z',
        'pelvis_gyr_x', 'pelvis_gyr_y', 'pelvis_gyr_z',
        'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
        'r_thigh_gyr_x', 'r_thigh_gyr_y', 'r_thigh_gyr_z',
        'l_shank_gyr_x', 'l_shank_gyr_y', 'l_shank_gyr_z',
        'r_shank_gyr_x', 'r_shank_gyr_y', 'r_shank_gyr_z',
        'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z',
        'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z',
    ]

    model_name = 'MLPRegressor'

    thread_number = multiprocessing.cpu_count() - 2  # allowed thread number
    pool = multiprocessing.Pool(processes=thread_number)

    sub_df_list = []
    for delay_ms in range(0, 101, 10):
        my_random_delay = RandomDelayClass(len(SEGMENT_NAMES), MOCAP_SAMPLE_RATE, delay_ms)
        for i_sub in range(1):
            pool.apply_async(get_result,
                             args=(input_names, output_names, result_column, i_sub, model_name, my_random_delay),
                             callback=sub_df_list.append)
    pool.close()
    pool.join()

    total_result_df = pd.DataFrame()
    for sub_df in sub_df_list:
        total_result_df = pd.concat([total_result_df, sub_df], axis=0)
    Evaluation.save_result(total_result_df, 'sync_test', 'GradientBoostingRegressor')
