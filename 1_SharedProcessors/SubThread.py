from const import *
from SubjectDataUni import SubjectData
from EvaluationThreadClass import EvaluationThread
from OffsetClass import *
from XYGeneratorUni import XYGeneratorUni
from datetime import datetime


def get_sub_result_df(base_model, input_names, output_names, result_column, i_sub):
    total_sub_df = pd.DataFrame()
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))
    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    other_sub_data = all_sub_data[all_sub_data['subject_id'] != i_sub]
    x_train = other_sub_data[input_names]
    y_train = other_sub_data[output_names]
    my_evaluator = EvaluationThread(output_names, base_model)

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
        multi_offset = MultiAxisOffset.get_segment_multi_axis_offset(segment_moved, shank_diameter, thigh_diameter)
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

