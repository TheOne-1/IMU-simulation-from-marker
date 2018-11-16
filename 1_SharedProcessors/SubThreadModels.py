# multiple machine models are processed together to save computing time

import json
import os
import pickle
import time
from datetime import datetime
from EvaluationClass import Evaluation
from OffsetClass import *
from SubjectData import SubjectData
from XYGenerator import XYGeneratorUni


def train_models(base_model, input_names, output_names, i_sub):
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))
    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    other_sub_data = all_sub_data[all_sub_data['subject_id'] != i_sub]
    x_train = other_sub_data[input_names]
    y_train = other_sub_data[output_names]
    my_evaluator = Evaluation(output_names, base_model)
    my_evaluator.train_sklearn(x_train, y_train)  # very time consuming

    root_dir = os.path.abspath(os.path.dirname(os.getcwd()))
    os.path.join(root_dir, 'resource\models')
    model_name = base_model.__class__.__name__
    # save results
    file_path = RESULT_PATH + 'evaluator\\' + model_name + '\\' + model_name + '_subject_' + str(i_sub) + '.pkl'
    with open(file_path, 'wb') as fp:
        pickle.dump(my_evaluator, fp, protocol=4)

    end_time = datetime.now()
    print('Subject ' + str(i_sub) + ', finished at ' + end_time.strftime('%H:%M:%S'))


def write_specification_file(model, input_names, output_names):
    date = time.strftime('%Y%m%d')
    model_name = model.__class__.__name__
    specification_txt_file = RESULT_PATH + 'evaluator\\' + model_name + '\\' + model_name + '_' + date + '_specification.txt'

    # write a specification file about details
    input_str = ', '.join(input_names)
    output_str = ', '.join(output_names)

    if DO_SCALING:
        scaling_str = 'Scaling: MinMaxScaler'
    else:
        scaling_str = 'Scaling: None'
    if DO_PCA:
        pca_str = 'Feature selection: PCA, n component = ' + str(N_COMPONENT)
    else:
        pca_str = 'Feature selection: None'

    content = 'Machine learning evaluators: ' + model.__class__.__name__ + '\n' + \
              'Model parameters: ' + json.dumps(model.get_params()) + '\n' + \
              'Input: ' + input_str + '\n' + \
              'Output: ' + output_str + '\n' + scaling_str + '\n' + pca_str + '\n'

    with open(specification_txt_file, 'w') as file:
        file.write(content)


def read_evaluator(date, folder_name):
    file_path = RESULT_PATH + folder_name + '\\model_' + date + '.txt'
    with open(file_path, 'rb') as fp:
        evaluators = pickle.load(fp)
    return evaluators


def get_segment_translation_result(input_names, output_names, result_column, i_sub, model_names):
    print('one trans')
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))

    my_evaluators = []
    for model_name in model_names:
        file_path = RESULT_PATH + 'evaluator\\' + model_name + '\\' + model_name + '_subject_' + str(i_sub) + '.pkl'
        with open(file_path, 'rb') as fp:
            my_evaluators.append(pickle.load(fp))

    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    total_sub_df = pd.DataFrame()
    test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub]
    del all_sub_data

    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    # the range have to be defined after subject data to get the diameter
    shank_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_shank')
    thigh_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_thigh')
    for segment_moved in SEGMENT_NAMES:
        # if i_sub == 0:  # for test
            # print('subject 0, ' + segment_moved)
        multi_offset = MultiAxisOffset.get_segment_multi_translation(segment_moved, shank_diameter, thigh_diameter)
        offset_combo_list = multi_offset.get_combos()
        offset_combo_len = len(offset_combo_list)
        all_offsets_df = multi_offset.get_offset_df()
        for speed in SPEEDS:
            test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
            height = test_speed_data['height'].as_matrix()[0]
            x_test = test_speed_data[input_names]
            y_test = test_speed_data[output_names].as_matrix()
            R2 = np.zeros([offset_combo_len, len(output_names)*len(my_evaluators)])
            RMSEs = np.zeros([offset_combo_len, len(output_names)*len(my_evaluators)])
            NRMSEs = np.zeros([offset_combo_len, len(output_names)*len(my_evaluators)])
            my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
            for i_combo in range(len(offset_combo_list)):
                x_test_modified = my_xy_generator.modify_x_segment(x_test, offset_combo_list[i_combo], height)
                for i_eval in range(len(my_evaluators)):
                    y_pred = my_evaluators[i_eval].evaluate_sklearn(x_test_modified, y_test)
                    R2[i_combo, i_eval*len(output_names):(i_eval+1)*len(output_names)], \
                        RMSEs[i_combo, i_eval*len(output_names):(i_eval+1)*len(output_names)],\
                        NRMSEs[i_combo, i_eval*len(output_names):(i_eval+1)*len(output_names)] =\
                        my_evaluators[i_eval].get_all_scores(y_test, y_pred)
            evaluation_result = np.column_stack([R2, RMSEs, NRMSEs])
            scores_df = pd.DataFrame(evaluation_result, columns=result_column)
            scores_df = scores_df.reset_index(drop=True)
            speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
            speed_df.insert(loc=0, column='segment', value=segment_moved)
            speed_df.insert(loc=0, column='speed', value=str(speed))
            speed_df.insert(loc=0, column='subject_id', value=i_sub)
            speed_df.insert(loc=0, column='experiment', value='one_trans')
            total_sub_df = pd.concat([total_sub_df, speed_df], axis=0)

    end_time = datetime.now()
    print('Subject ' + str(i_sub) + ', finished at ' + end_time.strftime('%H:%M:%S'))
    return total_sub_df


def get_all_translation_result(input_names, output_names, result_column, i_sub, model_names):
    print('all trans')
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))

    my_evaluators = []
    for model_name in model_names:
        file_path = RESULT_PATH + 'evaluator\\' + model_name + '\\' + model_name + '_subject_' + str(i_sub) + '.pkl'
        with open(file_path, 'rb') as fp:
            my_evaluators.append(pickle.load(fp))

    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
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
        R2 = np.zeros([combo_len + 1, len(output_names)*len(my_evaluators)])
        RMSEs = np.zeros([combo_len + 1, len(output_names)*len(my_evaluators)])
        NRMSEs = np.zeros([combo_len + 1, len(output_names)*len(my_evaluators)])
        # get the first data (no placement error)
        for i_eval in range(len(my_evaluators)):
            y_pred = my_evaluators[i_eval].evaluate_sklearn(x_test, y_test)
            R2[0, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                RMSEs[0, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                NRMSEs[0, i_eval * len(output_names):(i_eval + 1) * len(output_names)] = \
                my_evaluators[i_eval].get_all_scores(y_test, y_pred)
        my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
        for i_combo in range(len(combos_list)):
            x_test_modified = my_xy_generator.modify_x_all_combined(x_test, combos_list[i_combo], height)
            # get all other data
            for i_eval in range(len(my_evaluators)):
                y_pred = my_evaluators[i_eval].evaluate_sklearn(x_test_modified, y_test)
                R2[i_combo+1, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                    RMSEs[i_combo + 1, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                    NRMSEs[i_combo + 1, i_eval * len(output_names):(i_eval + 1) * len(output_names)] = \
                    my_evaluators[i_eval].get_all_scores(y_test, y_pred)

        evaluation_result = np.column_stack([R2, RMSEs, NRMSEs])
        scores_df = pd.DataFrame(evaluation_result, columns=result_column)
        scores_df = scores_df.reset_index(drop=True)
        speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
        speed_df.insert(loc=0, column='speed', value=str(speed))
        speed_df.insert(loc=0, column='subject_id', value=i_sub)
        speed_df.insert(loc=0, column='experiment', value='all_trans')
        total_sub_df = pd.concat([total_sub_df, speed_df], axis=0)

    end_time = datetime.now()
    print('Subject ' + str(i_sub) + ', finished at ' + end_time.strftime('%H:%M:%S'))
    return total_sub_df


def get_segment_rotation_result(input_names, output_names, result_column, i_sub, model_names):
    print('one rota')
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))

    my_evaluators = []
    for model_name in model_names:
        file_path = RESULT_PATH + 'evaluator\\' + model_name + '\\' + model_name + '_subject_' + str(i_sub) + '.pkl'
        with open(file_path, 'rb') as fp:
            my_evaluators.append(pickle.load(fp))

    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    total_sub_df = pd.DataFrame()
    test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub]
    del all_sub_data

    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    for segment_moved in SEGMENT_NAMES:
        if i_sub == 0:  # for test
            print('subject 0, ' + segment_moved)
        rotation_offsets = OneAxisRotation.get_one_axis_rotation(segment_moved)
        all_offsets_df = rotation_offsets.get_offset_df()
        rotation_len = len(rotation_offsets)

        for speed in SPEEDS:
            test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
            height = test_speed_data['height'].as_matrix()[0]
            x_test = test_speed_data[input_names]
            y_test = test_speed_data[output_names].as_matrix()
            R2 = np.zeros([rotation_len, len(output_names)*len(my_evaluators)])
            RMSEs = np.zeros([rotation_len, len(output_names)*len(my_evaluators)])
            NRMSEs = np.zeros([rotation_len, len(output_names)*len(my_evaluators)])
            my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
            for i_rotation in range(rotation_len):
                x_test_modified = my_xy_generator.modify_x_segment(x_test, rotation_offsets[i_rotation], height)
                # get all other data
                for i_eval in range(len(my_evaluators)):
                    y_pred = my_evaluators[i_eval].evaluate_sklearn(x_test_modified, y_test)
                    R2[i_rotation, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                        RMSEs[i_rotation, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                        NRMSEs[i_rotation, i_eval * len(output_names):(i_eval + 1) * len(output_names)] = \
                        my_evaluators[i_eval].get_all_scores(y_test, y_pred)
            evaluation_result = np.column_stack([R2, RMSEs, NRMSEs])
            scores_df = pd.DataFrame(evaluation_result, columns=result_column)
            scores_df = scores_df.reset_index(drop=True)
            speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
            speed_df.insert(loc=0, column='segment', value=segment_moved)
            speed_df.insert(loc=0, column='speed', value=str(speed))
            speed_df.insert(loc=0, column='subject_id', value=i_sub)
            speed_df.insert(loc=0, column='experiment', value='one_rota')
            total_sub_df = pd.concat([total_sub_df, speed_df], axis=0)

    end_time = datetime.now()
    print('Subject ' + str(i_sub) + ', finished at ' + end_time.strftime('%H:%M:%S'))
    return total_sub_df


def get_all_rotation_result(input_names, output_names, result_column, i_sub, model_names, error_value=25):
    print('all rota')
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))

    my_evaluators = []
    for model_name in model_names:
        file_path = RESULT_PATH + 'evaluator\\' + model_name + '\\' + model_name + '_subject_' + str(i_sub) + '.pkl'
        with open(file_path, 'rb') as fp:
            my_evaluators.append(pickle.load(fp))

    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    total_sub_df = pd.DataFrame()
    test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub]
    del all_sub_data
    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    segment_combo_class = RotationCombos(error_value=error_value)
    combos_list = segment_combo_class.get_segment_combos()
    combo_len = len(combos_list)
    all_offsets_df = segment_combo_class.get_offset_df()
    for speed in SPEEDS:
        test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
        x_test = test_speed_data[input_names]
        y_test = test_speed_data[output_names].as_matrix()
        height = test_speed_data['height'].as_matrix()[0]
        R2 = np.zeros([combo_len + 1, len(output_names)*len(my_evaluators)])
        RMSEs = np.zeros([combo_len + 1, len(output_names)*len(my_evaluators)])
        NRMSEs = np.zeros([combo_len + 1, len(output_names)*len(my_evaluators)])
        # get the first data (no placement error)
        for i_eval in range(len(my_evaluators)):
            y_pred = my_evaluators[i_eval].evaluate_sklearn(x_test, y_test)
            R2[0, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                RMSEs[0, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                NRMSEs[0, i_eval * len(output_names):(i_eval + 1) * len(output_names)] = \
                my_evaluators[i_eval].get_all_scores(y_test, y_pred)
        my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
        for i_combo in range(len(combos_list)):
            x_test_modified = my_xy_generator.modify_x_all_combined(x_test, combos_list[i_combo], height)
            # get all other data
            for i_eval in range(len(my_evaluators)):
                y_pred = my_evaluators[i_eval].evaluate_sklearn(x_test_modified, y_test)
                R2[i_combo + 1, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                    RMSEs[i_combo + 1, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                    NRMSEs[i_combo + 1, i_eval * len(output_names):(i_eval + 1) * len(output_names)] = \
                    my_evaluators[i_eval].get_all_scores(y_test, y_pred)
        evaluation_result = np.column_stack([R2, RMSEs, NRMSEs])
        scores_df = pd.DataFrame(evaluation_result, columns=result_column)
        scores_df = scores_df.reset_index(drop=True)
        speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
        speed_df.insert(loc=0, column='speed', value=str(speed))
        speed_df.insert(loc=0, column='subject_id', value=i_sub)
        speed_df.insert(loc=0, column='experiment', value='all_rota')
        total_sub_df = pd.concat([total_sub_df, speed_df], axis=0)

    end_time = datetime.now()
    print('Subject ' + str(i_sub) + ', finished at ' + end_time.strftime('%H:%M:%S'))
    return total_sub_df


def get_segment_trans_rota(input_names, output_names, result_column, i_sub, model_names):
    print('one trans rota')
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))

    my_evaluators = []
    for model_name in model_names:
        file_path = RESULT_PATH + 'evaluator\\' + model_name + '\\' + model_name + '_subject_' + str(i_sub) + '.pkl'
        with open(file_path, 'rb') as fp:
            my_evaluators.append(pickle.load(fp))

    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    total_sub_df = pd.DataFrame()
    test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub]
    del all_sub_data

    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    # the range have to be defined after subject data to get the diameter
    shank_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_shank')
    thigh_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_thigh')
    for segment_moved in SEGMENT_NAMES:
        if i_sub == 0:  # for test
            print('subject 0, ' + segment_moved)
        multi_offset = MultiAxisOffset.get_segment_multi_trans_rota(segment_moved, shank_diameter, thigh_diameter)
        offset_combo_list = multi_offset.get_combos()
        offset_combo_len = len(offset_combo_list)
        all_offsets_df = multi_offset.get_offset_df()
        for speed in SPEEDS:
            test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
            height = test_speed_data['height'].as_matrix()[0]
            x_test = test_speed_data[input_names]
            y_test = test_speed_data[output_names].as_matrix()
            R2 = np.zeros([offset_combo_len, len(output_names)*len(my_evaluators)])
            RMSEs = np.zeros([offset_combo_len, len(output_names)*len(my_evaluators)])
            NRMSEs = np.zeros([offset_combo_len, len(output_names)*len(my_evaluators)])
            my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
            for i_combo in range(len(offset_combo_list)):
                x_test_modified = my_xy_generator.modify_x_segment(x_test, offset_combo_list[i_combo], height)
                # get all other data
                for i_eval in range(len(my_evaluators)):
                    y_pred = my_evaluators[i_eval].evaluate_sklearn(x_test_modified, y_test)
                    R2[i_combo, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                        RMSEs[i_combo, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                        NRMSEs[i_combo, i_eval * len(output_names):(i_eval + 1) * len(output_names)] = \
                        my_evaluators[i_eval].get_all_scores(y_test, y_pred)
            evaluation_result = np.column_stack([R2, RMSEs, NRMSEs])
            scores_df = pd.DataFrame(evaluation_result, columns=result_column)
            scores_df = scores_df.reset_index(drop=True)
            speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
            speed_df.insert(loc=0, column='segment', value=segment_moved)
            speed_df.insert(loc=0, column='speed', value=str(speed))
            speed_df.insert(loc=0, column='subject_id', value=i_sub)
            speed_df.insert(loc=0, column='experiment', value='one_trans_rota')
            total_sub_df = pd.concat([total_sub_df, speed_df], axis=0)

    end_time = datetime.now()
    print('Subject ' + str(i_sub) + ', finished at ' + end_time.strftime('%H:%M:%S'))
    return total_sub_df


def get_all_trans_rota(input_names, output_names, result_column, i_sub, model_names, rotation_value=25):
    print('all trans rota')
    start_time = datetime.now()
    print('Subject ' + str(i_sub) + ', start at ' + start_time.strftime('%H:%M:%S'))

    my_evaluators = []
    for model_name in model_names:
        file_path = RESULT_PATH + 'evaluator\\' + model_name + '\\' + model_name + '_subject_' + str(i_sub) + '.pkl'
        with open(file_path, 'rb') as fp:
            my_evaluators.append(pickle.load(fp))

    all_sub_data = pd.read_csv(ALL_SUB_FILE, index_col=False)
    total_sub_df = pd.DataFrame()
    test_sub_data = all_sub_data[all_sub_data['subject_id'] == i_sub]
    del all_sub_data
    subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
    thigh_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_thigh')
    shank_diameter = SubjectData.get_cylinder_diameter(test_sub_data, 'l_shank')
    combo_value = np.array([[100, 100], [100, 100], [25, 100], [25, 100], [25, 100], [25, 100], [50, 50], [50, 50]])
    segment_combo_class = TransRotaCombos(combo_value, thigh_diameter, shank_diameter, rotation_value=rotation_value)
    combos_list = segment_combo_class.get_segment_combos()
    combo_len = len(combos_list)
    all_offsets_df = segment_combo_class.get_offset_df()
    for speed in SPEEDS:
        test_speed_data = test_sub_data[test_sub_data['speed'] == float(speed)]
        x_test = test_speed_data[input_names]
        y_test = test_speed_data[output_names].as_matrix()
        height = test_speed_data['height'].as_matrix()[0]
        R2 = np.zeros([combo_len + 1, len(output_names)*len(my_evaluators)])
        RMSEs = np.zeros([combo_len + 1, len(output_names)*len(my_evaluators)])
        NRMSEs = np.zeros([combo_len + 1, len(output_names)*len(my_evaluators)])
        # get the first data (no placement error)
        for i_eval in range(len(my_evaluators)):
            y_pred = my_evaluators[i_eval].evaluate_sklearn(x_test, y_test)
            R2[0, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                RMSEs[0, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                NRMSEs[0, i_eval * len(output_names):(i_eval + 1) * len(output_names)] = \
                my_evaluators[i_eval].get_all_scores(y_test, y_pred)
        my_xy_generator = XYGeneratorUni(subject_data, speed, output_names, input_names)
        for i_combo in range(len(combos_list)):
            x_test_modified = my_xy_generator.modify_x_all_combined(x_test, combos_list[i_combo], height)
            # get all other data
            for i_eval in range(len(my_evaluators)):
                y_pred = my_evaluators[i_eval].evaluate_sklearn(x_test_modified, y_test)
                R2[i_combo + 1, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                    RMSEs[i_combo + 1, i_eval * len(output_names):(i_eval + 1) * len(output_names)], \
                    NRMSEs[i_combo + 1, i_eval * len(output_names):(i_eval + 1) * len(output_names)] = \
                    my_evaluators[i_eval].get_all_scores(y_test, y_pred)
        evaluation_result = np.column_stack([R2, RMSEs, NRMSEs])
        scores_df = pd.DataFrame(evaluation_result, columns=result_column)
        scores_df = scores_df.reset_index(drop=True)
        speed_df = pd.concat([all_offsets_df, scores_df], axis=1)
        speed_df.insert(loc=0, column='speed', value=str(speed))
        speed_df.insert(loc=0, column='subject_id', value=i_sub)
        speed_df.insert(loc=0, column='experiment', value='all_trans_rota')
        total_sub_df = pd.concat([total_sub_df, speed_df], axis=0)

    end_time = datetime.now()
    print('Subject ' + str(i_sub) + ', finished at ' + end_time.strftime('%H:%M:%S'))
    return total_sub_df

