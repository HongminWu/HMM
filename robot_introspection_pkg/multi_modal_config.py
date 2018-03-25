modalities_store = {
    "endpoint_state_pose": [
        '.endpoint_state.pose.position.x',
        '.endpoint_state.pose.position.y',
        '.endpoint_state.pose.position.z',
        '.endpoint_state.pose.orientation.x',
        '.endpoint_state.pose.orientation.y',
        '.endpoint_state.pose.orientation.z',
        '.endpoint_state.pose.orientation.w',
    ],

    'endpoint_state_twist':[
        '.endpoint_state.twist.linear.x',
        '.endpoint_state.twist.linear.y',
        '.endpoint_state.twist.linear.z',
        '.endpoint_state.twist.angular.x',
        '.endpoint_state.twist.angular.y',
        '.endpoint_state.twist.angular.z',
    ],

    'wrench': [
         '.wrench_stamped.wrench.force.x',
         '.wrench_stamped.wrench.force.y',
         '.wrench_stamped.wrench.force.z',
         '.wrench_stamped.wrench.torque.x',
         '.wrench_stamped.wrench.torque.y',
         '.wrench_stamped.wrench.torque.z',
    ],

    'delta_wrench': [
         '.delta_wrench.force.x',
         '.delta_wrench.force.y',
         '.delta_wrench.force.z',
         '.delta_wrench.torque.x',
         '.delta_wrench.torque.y',
         '.delta_wrench.torque.z',
    ],
    
    'joint_position': [
    '.joint_state.position.right_s0',
    '.joint_state.position.right_s1',
    '.joint_state.position.right_e0',
    '.joint_state.position.right_e1',
    '.joint_state.position.right_w0',
    '.joint_state.position.right_w1',
    '.joint_state.position.right_w2',
    ],

    'joint_velocity': [
    '.joint_state.velocity.right_s0',
    '.joint_state.velocity.right_s1',
    '.joint_state.velocity.right_e0',
    '.joint_state.velocity.right_e1',
    '.joint_state.velocity.right_w0',
    '.joint_state.velocity.right_w1',
    '.joint_state.velocity.right_w2',
    ],

    'joint_effort': [
    '.joint_state.effort.right_s0',
    '.joint_state.effort.right_s1',
    '.joint_state.effort.right_e0',
    '.joint_state.effort.right_e1',
    '.joint_state.effort.right_w0',
    '.joint_state.effort.right_w1',
    '.joint_state.effort.right_w2',
    ],

#the following items for IIWA-ROBOT
    'CartesianWrench':[
    '.CartesianWrench.wrench.force.x',
    '.CartesianWrench.wrench.force.y',
    '.CartesianWrench.wrench.force.z',
    '.CartesianWrench.wrench.torque.x',
    '.CartesianWrench.wrench.torque.y',
    '.CartesianWrench.wrench.torque.z',
    ],

    'JointPosition':[
    '.JointPosition.position.a1',
    '.JointPosition.position.a2',
    '.JointPosition.position.a3',
    '.JointPosition.position.a4',
    '.JointPosition.position.a5',
    '.JointPosition.position.a6',
    '.JointPosition.position.a7',
    ],

	'tactile_sensor_data':[
		'.tactile_values.tactile_values_0',
		'.tactile_values.tactile_values_1',
		'.tactile_values.tactile_values_2',
		'.tactile_values.tactile_values_3',
		'.tactile_values.tactile_values_4',
		'.tactile_values.tactile_values_5',
		'.tactile_values.tactile_values_6',
		'.tactile_values.tactile_values_7',
		'.tactile_values.tactile_values_8',
		'.tactile_values.tactile_values_9',
		],
}

#for iiwa_robo
#modality_chosen = 'CartesianWrench+JointPosition'

#for baxter_robot
modality_chosen = 'endpoint_state_twist+wrench+delta_wrench+tactile_sensor_data'

modality_split = modality_chosen.split("+")
interested_data_fields = []
for modality in modality_split:
    interested_data_fields += modalities_store[modality]
interested_data_fields.append('.tag')
