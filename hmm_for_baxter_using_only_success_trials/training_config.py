import os
import util

from model_config_store import model_store

score_metric_options = [
    '_score_metric_last_time_stdmeanratio_',
    '_score_metric_worst_stdmeanratio_in_10_slice_',
    '_score_metric_sum_stdmeanratio_using_fast_log_cal_',
    '_score_metric_mean_of_std_using_fast_log_cal_',
    '_score_metric_hamming_distance_using_fast_log_cal_',
    '_score_metric_std_of_std_using_fast_log_cal_',
    '_score_metric_mean_of_std_divied_by_final_log_mean_',
    '_score_metric_mean_of_std_of_gradient_divied_by_final_log_mean_',
    '_score_metric_minus_diff_btw_1st_2ed_emissionprob_',
    '_score_metric_minus_diff_btw_1st_2ed(>=0)_divide_maxeprob_emissionprob_',
    '_score_metric_minus_diff_btw_1st_2ed(delete<0)_divide_maxeprob_emissionprob_',
    '_score_metric_mean_of_(std_of_(max_emissionprob_of_trials))_',
    '_score_metric_duration_of_(diff_btw_1st_2ed_emissionprob_<_10)_',
]

anomaly_detection_metric_options = [
    'loglik<threshold=(mean-c*std)',
    'gradient<threshold=(min-range/2)',
    'deri_of_diff',
]

base_path_options = [
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170711',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_20170724_6states_vision (delete a bad training data)',
    '/home/sklaw/Desktop/ex/birl/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_20170724_6states_vision (delete a bad training data)',
    '/home/sklaw/Desktop/ex/birl/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170711',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170908',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170909(wrench data synced)',
    '/home/sklaw/Desktop/ex/birl/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170909_calibrated_wrench_data',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170909_calibrated_wrench_data',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170914',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170914_state_transition_wait_2s',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170914_use_old_tagpuber',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_Shuangqi/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170918_old_puber',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_HONGMINWU/iiwa_varying_pose_record/train',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_HONGMINWU/samePoint_3targetPoints/test',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_HONGMINWU/move_to_random_pose/',
    '/home/vmrguser/Files_from_Shuangqi_to_Workstation/birl/data_for_or_from_HMM/ML_DATA_HONGMINWU/anomaly_identification/',
	'/home/birl_wu/HMM/kitting_experiment_data/'
]

# hardcoded constants.
data_type_options = [
    'endpoint_pose',
    'wrench',
    'endpoint_pose_and_wrench'
]

model_type_options = [
    'hmmlearn\'s HMM',
    'hmmlearn\'s GMMHMM',
    'BNPY\'s HMM',
    'PYHSMM\'s HMM',
]

import robot_introspection_pkg.multi_modal_config as mmc

# config provided by the user
config_by_user = {
    # config for types
    'data_type_chosen' :  mmc.modality_chosen,
    'model_type_chosen':  'BNPY\'s HMM', #'BNPY\'s HMM','hmmlearn\'s HMM'
    'score_metric': '_score_metric_last_time_stdmeanratio_',
    'anomaly_detection_metric': anomaly_detection_metric_options[1],

    # config for dataset folder
    'base_path': base_path_options[-1],

    # config for preprocessing
    'preprocessing_scaling': False,   # scaled data has zero mean and unit variance
    'preprocessing_normalize': False, # normalize the individual samples to have unit norm "l1" or 'l2'
    'norm_style': 'l2',
    'pca_components': 0, # cancel the pca processing
    # threshold of derivative used in hmm online anomaly detection
    'deri_threshold': 200,

    # threshold training c value in threshold=mean-c*std
    'threshold_c_value': 5
}

interested_data_fields = mmc.interested_data_fields

model_config_set_name = model_store[config_by_user['model_type_chosen']]['use']
model_config = model_store[config_by_user['model_type_chosen']]['config_set'][model_config_set_name]

model_id     = util.get_model_config_id(model_config)
model_id     = config_by_user['score_metric']+model_id
norm_style   = config_by_user['norm_style']


success_path = os.path.join(config_by_user['base_path'], "success")
test_success_data_path = os.path.join(config_by_user['base_path'], "success_for_test")
model_save_path = os.path.join(config_by_user['base_path'], "model", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)
figure_save_path = os.path.join(config_by_user['base_path'], "figure", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)

# for anomaly analysis
anomaly_data_path = os.path.join(config_by_user['base_path'], 'anomalies')
anomaly_data_path_for_testing = os.path.join(config_by_user['base_path'], 'anomalies_for_testing')
anomaly_model_save_path = os.path.join(config_by_user['base_path'], "anomaly_models")
anomaly_identification_figure_path = os.path.join(config_by_user['base_path'], "figure", config_by_user['data_type_chosen'], config_by_user['model_type_chosen'], model_id)

exec '\n'.join("%s=%r"%i for i in config_by_user.items())
