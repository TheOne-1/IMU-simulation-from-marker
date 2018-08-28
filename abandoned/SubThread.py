import pickle
import random
from datetime import datetime

from EvaluationClass import Evaluation
from OffsetClass import *
from SubjectData import SubjectData
from XYGenerator import XYGeneratorUni


def get_sub_result_df(base_model, input_names, output_names, result_column, i_sub):
    total_sub_df = pd.DataFrame()
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))
    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    other_sub_data = all_sub_data[all_sub_data['subject_id'] != i_sub]
    x_train = other_sub_data[input_names]
    y_train = other_sub_data[output_names]
    my_evaluator = Evaluation(output_names, base_model)

    my_evaluator.train_sklearn(x_train, y_train)  # very time consuming
    test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub]
    del all_sub_data
    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    # the range have to be defined after subject data to get the diameter
    shank_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_shank')
    thigh_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_thigh')
    for segment_moved in SEGMENT_NAMES:
        if i_sub == 0:      # for test
            print('subject 0, ' + segment_moved)
        multi_offset = MultiAxisOffset.get_segment_multi_translation(segment_moved, shank_diameter, thigh_diameter)
        offset_combo_list = multi_offset.get_combos()
        offset_combo_len = len(offset_combo_list)
        all_offsets_df = multi_offset.get_offset_df()
        for speed in SPEEDS:
            test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
            height = test_speed_data['height'].as_matrix()[0]
            x_test = test_speed_data[input_names]
            y_test = test_speed_data[output_names].as_matrix()
            R2 = np.zeros([offset_combo_len, len(output_names)])
            RMSEs = np.zeros([offset_combo_len, len(output_names)])
            NRMSEs = np.zeros([offset_combo_len, len(output_names)])
            my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
            for i_combo in range(len(offset_combo_list)):
                x_test_modified = my_xy_generator.modify_x_segment(x_test, offset_combo_list[i_combo], height)
                y_pred = my_evaluator.evaluate_sklearn(x_test_modified, y_test)
                R2[i_combo, :], RMSEs[i_combo, :], NRMSEs[i_combo, :] = my_evaluator.get_all_scores(y_test, y_pred)
            evaluation_result = np.column_stack([R2, RMSEs, NRMSEs])
            scores_df = pd.DataFrame(evaluation_result, columns=result_column)
            scores_df = scores_df.reset_index(drop=True)
            speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
            speed_df.insert(loc=0, column='segment', value=segment_moved)
            speed_df.insert(loc=0, column='speed', value=str(speed))
            speed_df.insert(loc=0, column='subject_id', value=i_sub)
            total_sub_df = pd.concat([total_sub_df, speed_df], axis=0)

    end_time = datetime.now()
    print('Subject ' + str(i_sub) + ', finished at ' + end_time.strftime('%H:%M:%S'))
    return total_sub_df


def get_combined_result_df(input_names, output_names, result_column, base_model, i_sub):
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))

    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    other_sub_data = all_sub_data[all_sub_data['subject_id'] != i_sub]
    x_train = other_sub_data[input_names]
    y_train = other_sub_data[output_names]
    my_evaluator = Evaluation(output_names, base_model)
    my_evaluator.train_sklearn(x_train, y_train)  # very time consuming

    total_sub_df = pd.DataFrame()
    test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub]
    del all_sub_data
    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    # the range have to be defined after subject data to get the diameter
    thigh_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_thigh')
    shank_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_shank')
    combo_value = np.array([[100, 100], [100, 100], [25, 100], [25, 100], [25, 100], [25, 100], [50, 50], [50, 50]])
    segment_combo_class = TranslationCombos(combo_value, thigh_diameter, shank_diameter)
    combos_list = segment_combo_class.get_segment_combos()
    combo_len = len(combos_list)
    all_offsets_df = segment_combo_class.get_offset_df()
    for speed in SPEEDS:
        test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
        x_test = test_speed_data[input_names]
        y_test = test_speed_data[output_names].as_matrix()
        height = test_speed_data['height'].as_matrix()[0]
        R2 = np.zeros([combo_len+1, len(output_names)])
        RMSEs = np.zeros([combo_len+1, len(output_names)])
        NRMSEs = np.zeros([combo_len+1, len(output_names)])
        y_pred = my_evaluator.evaluate_sklearn(x_test, y_test)
        R2[0, :], RMSEs[0, :], NRMSEs[0, :] = my_evaluator.get_all_scores(y_test, y_pred)
        my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
        for i_combo in range(len(combos_list)):
            x_test_modified = my_xy_generator.modify_x_all_combined(x_test, combos_list[i_combo], height)
            y_pred = my_evaluator.evaluate_sklearn(x_test_modified, y_test)
            R2[i_combo+1, :], RMSEs[i_combo+1, :], NRMSEs[i_combo+1, :] = my_evaluator.get_all_scores(y_test, y_pred)
        evaluation_result = np.column_stack([R2, RMSEs, NRMSEs])
        scores_df = pd.DataFrame(evaluation_result, columns=result_column)
        scores_df = scores_df.reset_index(drop=True)
        speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
        speed_df.insert(loc=0, column='speed', value=str(speed))
        speed_df.insert(loc=0, column='subject_id', value=i_sub)
        total_sub_df = pd.concat([total_sub_df, speed_df], axis=0)

    end_time = datetime.now()
    print('Subject ' + str(i_sub) + ', finished at ' + end_time.strftime('%H:%M:%S'))
    return total_sub_df




def get_error_combos(date, combo_num=100):
    with open(RESULT_PATH + 'result_segment\\' + date + '\\segment_area_all.txt', 'rb') as fp:
        area_all = pickle.load(fp)
    index_all = []
    for i_combo in range(combo_num):
        index = np.zeros([len(SEGMENT_NAMES), 2], dtype=int)
        for i_segment in range(len(SEGMENT_NAMES)):
            index[i_segment, :] = select_point_randomly(area_all[i_segment])
        index_all.append(index)
    return index_all


def select_point_randomly(segment_acceptable_area):
    point_found_flag = False
    axis_0_range = segment_acceptable_area.shape[1]
    axis_1_range = segment_acceptable_area.shape[0]
    x_index, y_index = 0, 0
    while not point_found_flag:
        x_index = random.randint(0, axis_0_range - 1)
        y_index = random.randint(0, axis_1_range - 1)
        if segment_acceptable_area[y_index, x_index] == 1:
            point_found_flag = True
    return [x_index, y_index]





