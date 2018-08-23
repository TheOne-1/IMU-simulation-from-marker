# divide model training and IMU simulation to save computation time
from sklearn.svm import SVR
from sklearn import ensemble
from SubThreadSaveModel import *
import multiprocessing
from datetime import datetime

if __name__ == '__main__':
    start_time = datetime.now()
    output_names = [
        'FP1.ForX',
        'FP2.ForX',
        'FP1.ForY',
        'FP2.ForY',
        'FP1.ForZ',
        'FP2.ForZ',
        # 'FP1.CopX', 'FP1.CopY',
        # 'FP2.CopX', 'FP2.CopY'
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

    # evaluators = ensemble.RandomForestRegressor()
    model = ensemble.RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=7)
    # model = SVR(C=200, epsilon=0.02, gamma=0.1, max_iter=4)
    # model = ensemble.GradientBoostingRegressor(
    #     learning_rate=0.1, min_impurity_decrease=0.001, min_samples_split=6, n_estimators=500)

    thread_number = multiprocessing.cpu_count() - 2  # allowed thread number
    pool = multiprocessing.Pool(processes=thread_number)

    stage = 1       # 0 for model training, 1 for placement error simulation

    if stage == 0:
        # model training
        print('model training started')
        write_specification_file(model, input_names, output_names, 'evaluator')
        for i_sub in range(SUB_NUM):
            pool.apply_async(train_models, args=(model, input_names, output_names, i_sub, 'evaluator'))
        pool.close()
        pool.join()

        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))

    elif stage == 1:
        print('placement error simulation started')
        sub_df_list = []
        for i_sub in range(1):
            pool.apply_async(get_segment_rotation_result, args=(input_names, output_names, result_column, i_sub, 'evaluator', '20180820'),
                             callback=sub_df_list.append)
        pool.close()
        pool.join()

        total_result_df = pd.DataFrame()
        for sub_df in sub_df_list:
            total_result_df = pd.concat([total_result_df, sub_df], axis=0)
        Evaluation.save_result(total_result_df, model, input_names, output_names, 'result_segment_rotation')
        end_time = datetime.now()
        print('Duration: ' + str(end_time - start_time))








