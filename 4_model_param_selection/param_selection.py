# this file is used to evaluate all the thigh marker position and find the best location
# 用processor之前一定要set_segment

from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

from DatabaseInfo import DatabaseInfo
from EvaluationClass import Evaluation
from SubjectData import SubjectData
from XYGenerator import XYGenerator
from const import *

output_names = [
    'FP1.ForX',
    # 'FP2.ForX',
    # 'FP1.ForY', 'FP2.ForY',
    # 'FP1.ForZ', 'FP2.ForZ',
    # 'FP1.CopX', 'FP1.CopY',
    # 'FP2.CopX',
    # 'FP2.CopY'
]
total_result_columns = ['segment', 'subject_id', 'speed', 'x_offset', 'y_offset', 'z_offset', 'theta_offset']
total_result_columns.extend(output_names)  # change TOTAL_RESULT_COLUMNS

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
my_database_info = DatabaseInfo()
total_score_df = Evaluation.initialize_result_df(total_result_columns)

# model = neighbors.KNeighborsRegressor()
# model = ensemble.AdaBoostRegressor()
# model = tree.DecisionTreeRegressor()
# model = SVR(C=200, epsilon=0.02, gamma=0.1, max_iter=4000000)
# model = ensemble.RandomForestRegressor(random_state=100)
# model = GridSearchCV(ensemble.RandomForestRegressor(random_state=0, n_jobs=4), param_grid={'n_estimators': [40, 200, 1000], 'min_impurity_decrease': [0, 1e-3]})
# model = GridSearchCV(ensemble.AdaBoostRegressor())
model = GridSearchCV(ensemble.GradientBoostingRegressor(), param_grid={'n_estimators': [20, 100, 500],
                                                                       'max_depth': [2, 6, 18],
                                                                       'learning_rate': [0.01, 0.1],
                                                                       'min_samples_split': [2, 6, 18],
                                                                       'min_impurity_decrease': [1e-3, 1e-2]})

# model = GridSearchCV(SVR(verbose=2, max_iter=4000000), param_grid={'C': [100, 1000, 10000], 'gamma': [0.001, 0.01, 0.1], 'epsilon': [0.001, 0.01, 0.1]})
# model = GridSearchCV(tree.DecisionTreeRegressor(), param_grid={'max_depth': [10, 100, 1000], 'min_samples_leaf': [2, 20, 200], 'max_features': [5, 10, 24]})

# model = GridSearchCV(SVR(tol=0.01), param_grid={'kernel': ['rbf', 'sigmoid', 'poly']})

segment_moved = SEGMENT_NAMES[0]
i_sub = 0
speed = SPEEDS[0]

subject_data = SubjectData(PROCESSED_DATA_PATH, i_sub)
my_xy_generator = XYGenerator(subject_data, segment_moved, speed, output_names, input_names)
x_raw, y_raw = my_xy_generator.get_xy()
x, y = x_raw[input_names], y_raw[output_names]

x_scaler = preprocessing.StandardScaler()
x_scaler.fit(x)
x = x_scaler.transform(x)
y_scaler = preprocessing.StandardScaler()
y_scaler.fit(y)
y = y_scaler.transform(y)
# shuffle the data
x, y = shuffle(x, y)

model.fit(x, y[:, 0])

# print scores of each estiamtor
means = model.cv_results_['mean_test_score']
stds = model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, model.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print('\n')
print(model.best_estimator_)

# print feature importances
print('\n')
# feature_importance = model.best_estimator_.feature_importances_
# print(feature_importance)
# plt.bar(input_names, feature_importance)
# plt.show()
