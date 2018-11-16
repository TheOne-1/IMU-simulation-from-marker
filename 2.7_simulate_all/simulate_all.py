# divide model training and IMU simulation to save computation time
import multiprocessing
from datetime import datetime
from sklearn import ensemble
from SubThreadModels import *

if __name__ == '__main__':
    start_time = datetime.now()
    output_names = ['FP1.ForX', 'FP2.ForX', 'FP1.ForY', 'FP2.ForY', 'FP1.ForZ', 'FP2.ForZ']

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

    model_names = ['SVR', 'RandomForestRegressor', 'GradientBoostingRegressor', 'MLPRegressor']

    R2_column = [output + '_' + model + '_R2' for model in model_names for output in output_names]
    RMSE_column = [output + '_' + model + '_RMSE' for model in model_names for output in output_names]
    NRMSE_column = [output + '_' + model + '_NRMSE' for model in model_names for output in output_names]
    result_column = R2_column + RMSE_column + NRMSE_column

    thread_number = multiprocessing.cpu_count() - 2  # allowed thread number
    pool = multiprocessing.Pool(processes=6)

    df_list = []
    sub_num = 12

    for i_sub in range(sub_num):
        pool.apply_async(get_segment_translation_result, args=(input_names, output_names, result_column, i_sub,
                                                               model_names), callback=df_list.append)
    for i_sub in range(sub_num):
        pool.apply_async(get_segment_rotation_result, args=(input_names, output_names, result_column, i_sub,
                                                            model_names), callback=df_list.append)
    for i_sub in range(sub_num):
        pool.apply_async(get_segment_trans_rota, args=(input_names, output_names, result_column, i_sub,
                                                       model_names), callback=df_list.append)
    for i_sub in range(sub_num):
        pool.apply_async(get_all_translation_result, args=(input_names, output_names, result_column, i_sub,
                                                           model_names), callback=df_list.append)
    for i_sub in range(sub_num):
        pool.apply_async(get_all_rotation_result, args=(input_names, output_names, result_column, i_sub,
                                                        model_names), callback=df_list.append)
    for i_sub in range(sub_num):
        pool.apply_async(get_all_trans_rota, args=(input_names, output_names, result_column, i_sub,
                                                   model_names), callback=df_list.append)
    pool.close()
    pool.join()

    total_result_df = pd.DataFrame()
    for df in df_list:
        total_result_df = pd.concat([total_result_df, df], axis=0)
    Evaluation.save_result_models(total_result_df, 'result_all', model_names)
    end_time = datetime.now()
    print('Duration: ' + str(end_time - start_time))
