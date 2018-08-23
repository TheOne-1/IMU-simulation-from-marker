# this file is used to evaluate all the thigh marker position and find the best location

from sklearn import ensemble
from sklearn import preprocessing

from OffsetClass import *
from SubjectDataUni import SubjectData
from XYGeneratorUni import XYGeneratorUni
from abandoned.EvaluationUniClass import EvaluationUni

output_names = [
    'FP1.ForX',
    # 'FP2.ForX',
    'FP1.ForY',
    # 'FP2.ForY',
    'FP1.ForZ',
    # 'FP2.ForZ',
    # 'FP1.CopX', 'FP1.CopY',
    # 'FP2.CopX', 'FP2.CopY'
]

input_names = [
    'trunk_acc_x', 'trunk_acc_y', 'trunk_acc_z',
    'pelvis_acc_x', 'pelvis_acc_y', 'pelvis_acc_z',
    'l_thigh_acc_x', 'l_thigh_acc_y', 'l_thigh_acc_z',
    'r_thigh_acc_x', 'r_thigh_acc_y', 'r_thigh_acc_z',
    'l_shank_acc_x', 'l_shank_acc_y', 'l_shank_acc_z',
    'r_shank_acc_x', 'r_shank_acc_y', 'r_shank_acc_z',
    'l_foot_acc_x', 'l_foot_acc_y', 'l_foot_acc_z',
    'r_foot_acc_x', 'r_foot_acc_y', 'r_foot_acc_z',
    # 'trunk_gyr_x', 'trunk_gyr_y', 'trunk_gyr_z',
    # 'pelvis_gyr_x', 'pelvis_gyr_y', 'pelvis_gyr_z',
    # 'l_thigh_gyr_x', 'l_thigh_gyr_y', 'l_thigh_gyr_z',
    # 'r_thigh_gyr_x', 'r_thigh_gyr_y', 'r_thigh_gyr_z',
    # 'l_shank_gyr_x', 'l_shank_gyr_y', 'l_shank_gyr_z',
    # 'r_shank_gyr_x', 'r_shank_gyr_y', 'r_shank_gyr_z',
    # 'l_foot_gyr_x', 'l_foot_gyr_y', 'l_foot_gyr_z',
    # 'r_foot_gyr_x', 'r_foot_gyr_y', 'r_foot_gyr_z',
]

R2_column = [output + '_R2' for output in output_names]
RMSE_column = [output + '_RMSE' for output in output_names]
NRMSE_column = [output + '_NRMSE' for output in output_names]
result_column = R2_column + RMSE_column + NRMSE_column
offset_value_list = [50, 50, 20, 20, 5, 20, 5, 20, 5, 20, 5, 20, 20, 20, 20, 20]

model = ensemble.RandomForestRegressor(n_jobs=6)
# evaluators = ensemble.RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=6)
# evaluators = SVR(C=200, epsilon=0.02, gamma=0.1, max_iter=400)
# evaluators = ensemble.GradientBoostingRegressor(
#     learning_rate=0.1, min_impurity_decrease=0.001, min_samples_split=6, n_estimators=500)
x_scalar, y_scalar = preprocessing.MinMaxScaler(), preprocessing.MinMaxScaler()

all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)

total_result_df = pd.DataFrame()

for i_sub_test in range(SUB_NUM):
    print('subject: ' + str(i_sub_test))
    other_sub_data = all_sub_data[all_sub_data['subject_id'] != i_sub_test]
    x_train = other_sub_data[input_names]
    y_train = other_sub_data[output_names]
    my_evaluator = EvaluationUni(output_names, model)
    my_evaluator.set_train(x_train, y_train)

    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub_test)

    # the range have to be defined after subject data to get the diameter

    shank_diameter = subject_data.get_cylinder_diameter('l_shank')
    thigh_diameter = subject_data.get_cylinder_diameter('l_thigh')

    segment_combo_class = TranslationCombos(offset_value_list, thigh_diameter, shank_diameter)
    combos_list = segment_combo_class.get_segment_combos()
    combo_len = len(combos_list)
    all_offsets_df = segment_combo_class.get_offset_df()

    my_evaluator.train_sklearn()        # very time consuming

    for speed in SPEEDS:
        test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub_test]
        test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
        height = test_speed_data['height'].as_matrix()[0]
        x_test = test_speed_data[input_names]
        y_test = test_speed_data[output_names]
        my_evaluator.set_x_test(x_test)
        my_evaluator.set_y_test(y_test)
        my_evaluator.evaluate_sklearn()
        scores = np.zeros([combo_len, len(output_names)])
        RMSEs = np.zeros([combo_len, len(output_names)])
        NRMSEs = np.zeros([combo_len, len(output_names)])
        my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
        for i_combo in range(len(combos_list)):
            x_test_modified = my_xy_generator.modify_x_all_combined(x_test, combos_list[i_combo], height)
            my_evaluator.set_x_test(x_test_modified)
            my_evaluator.evaluate_sklearn()
            scores[i_combo, :] = my_evaluator.get_scores()
            RMSEs[i_combo, :] = my_evaluator.get_RMSE()
            NRMSEs[i_combo, :] = my_evaluator.get_NRMSE()
        evaluation_result = np.column_stack([scores, RMSEs, NRMSEs])
        scores_df = pd.DataFrame(evaluation_result, columns=result_column)
        scores_df = scores_df.reset_index(drop=True)
        speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
        speed_df.insert(loc=0, column='speed', value=str(speed))
        speed_df.insert(loc=0, column='subject_id', value=i_sub_test)
        total_result_df = pd.concat([total_result_df, speed_df], axis=0)

    EvaluationUni.save_result(total_result_df, model, input_names, output_names, 'result_all_combined')








