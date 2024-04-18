
#!/usr/bin/env python
from numpy import genfromtxt
import rospy
import csv
import numpy as np
import dvrk
import os.path
import os
import errno
import copy
from utils import load_data
from optimization import FourierTraj
import matplotlib.pyplot as plt

robot_model_name = 'psm'
robot_name = 'PSM'

# Transformation matrices for motor to dvrk mapping
motor_to_dvrk_psm = np.array([[1.0186, 0, 0], [-.8306, .6089, .6089], [0, -1.2177, 1.2177]])

# Define scales for different joints
scales = np.array([1, 1, 1, 1, 1, 1, 1])

# Define time parameters
stable_time = 5
sampling_time = 30
sampling_rate = 200
speed_scale = 1.0

# Define trajectory parameters
trajectory_name = 'fourier_identification'

# Define folder paths
model_folder = 'models/'
trajectory_folder = 'optimization/'

# Load robot model and trajectory data
robot_model_data = load_data(model_folder, robot_model_name)
trajectory_data = load_data(trajectory_folder, trajectory_name)

# Extract necessary information from loaded data
num_degrees_of_freedom, fourier_order, base_frequency, trajectory_result, reg_norm_matrix = trajectory_data

# Check for consistency in scales and degrees of freedom
if scales.shape[0] != num_degrees_of_freedom:
    raise Exception("Inconsistent scales and degrees of freedom")

# Scale the trajectory results
param_num_for_one_joint = 1 + 2 * fourier_order
for i in range(num_degrees_of_freedom):
    trajectory_result[1 + i * param_num_for_one_joint: 7 + i * param_num_for_one_joint] *= scales[i]

# Generate trajectory data with ramp-up
trajectory_generator = FourierTraj(num_degrees_of_freedom, fourier_order, base_frequency,
                                   frequency=sampling_rate, stable_time=stable_time, final_time=sampling_time)
joint_angles, _, _ = trajectory_generator.fourier_base_x2q(trajectory_result)

# Initialize variables for joint states
measured_states = copy.deepcopy(joint_angles)

# Initialize dVRK robot object based on robot name
is_psm = robot_name.startswith('PSM')
robot_arm = dvrk.psm(robot_name) if is_psm else dvrk.mtm(robot_name)

# Home the robot
robot_arm.home()
rospy.sleep(3)

# Move to the start position of the trajectory
if is_psm and num_degrees_of_freedom == 7:
    selected_joints = np.array([0, 1, 2, 3, 4, 5])
    robot_arm.move_joint_some(measured_states[0, 0:num_degrees_of_freedom - 1], selected_joints)
    robot_arm.move_jaw(measured_states[0, -1])
else:
    selected_joints = np.array([d for d in range(num_degrees_of_freedom)])
    robot_arm.move_joint_some(measured_states[0, :], selected_joints)

# Initialize variables for storing robot states
start_count = int(sampling_rate * stable_time)
max_state_number = len(joint_angles) - start_count
states = np.zeros((max_state_number, 3 * num_degrees_of_freedom))

rospy.sleep(3)

rate = rospy.Rate(sampling_rate * speed_scale)
state_count = 0

# Record robot states during trajectory execution
for i in range(len(joint_angles)):
    if is_psm:
        robot_arm.move_joint_some(joint_angles[i, 0:num_degrees_of_freedom - 1], selected_joints, interpolate=False, blocking=False)
        robot_arm.move_jaw(joint_angles[i, -1], interpolate=False, blocking=False)

        if i >= start_count:
            state_count = i - start_count
            states[state_count][0:num_degrees_of_freedom - 1] = robot_arm.get_current_joint_position()[0:num_degrees_of_freedom - 1]
            states[state_count][num_degrees_of_freedom - 1] = robot_arm.get_current_jaw_position()
            states[state_count][num_degrees_of_freedom:2 * num_degrees_of_freedom - 1] = robot_arm.get_current_joint_velocity()[0:num_degrees_of_freedom - 1]
            states[state_count][2 * num_degrees_of_freedom - 1] = robot_arm.get_current_jaw_velocity()
            states[state_count][2 * num_degrees_of_freedom:3 * num_degrees_of_freedom - 1] = robot_arm.get_current_joint_effort()[0:num_degrees_of_freedom - 1]
            states[state_count][3 * num_degrees_of_freedom - 1] = robot_arm.get_current_jaw_effort()

    else:
        robot_arm.move_joint_some(joint_angles[i, :], selected_joints, interpolate=False, blocking=False)

        if i >= start_count:
            state_count = i - start_count
            states[state_count][0:num_degrees_of_freedom] = robot_arm.get_current_joint_position()[0:num_degrees_of_freedom]
            states[state_count][num_degrees_of_freedom:2 * num_degrees_of_freedom] = robot_arm.get_current_joint_velocity()[0:num_degrees_of_freedom]
            states[state_count][2 * num_degrees_of_freedom:3 * num_degrees_of_freedom] = robot_arm.get_current_joint_effort()[0:num_degrees_of_freedom]

    rate.sleep()

# Reverse motor coupling
motor_states = copy.deepcopy(states)
if is_psm:
    for i in range(states.shape[0]):
        motor_states[i, 4:7] = np.matmul(np.linalg.inv(motor_to_dvrk_psm), states[i, 4:7])
        motor_states[i, 11:14] = np.matmul(np.linalg.inv(motor_to_dvrk_psm), states[i, 11:14])
        motor_states[i, 18:22] = np.matmul(motor_to_dvrk_psm.transpose(), states[i, 18:22])
else:
    for i in range(states.shape[0]):
        motor_states[i, 1:4] = np.matmul(np.linalg.inv(motor_to_dvrk_mtm), states[i, 1:4])
        motor_states[i, 8:11] = np.matmul(np.linalg.inv(motor_to_dvrk_mtm), states[i, 8:11])
        motor_states[i, 15:18] = np.matmul(motor_to_dvrk_mtm.transpose(), states[i, 15:18])

states = motor_states

# Save measured trajectory data to a CSV file
data_file_directory = './experiments' + trajectory_name + '_feedback.csv'
if not os.path.exists(os.path.dirname(data_file_directory)):
    try:
        os.makedirs(os.path.dirname(data_file_directory))
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

with open(data_file_directory, 'w+') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE)
    for i in range(np.size(states, 0) - 1):
        writer.writerow(states[i])
