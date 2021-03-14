from utils import *
from visualization import *

if __name__ == '__main__':

    # only choose ONE of the following data
	    
	# data 1. this data has features, use this if you plan to skip the extra credit feature detection and tracking part 
	filename = "./data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)
	step=100
	features=features[:,::step,:]
	# data 2. this data does NOT have features, you need to do feature detection and tracking but will receive extra credit
	#filename = "./data/03.npz"
	#t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

    # (a) IMU Localization via EKF Prediction

	imu_T_world, world_T_imu = motionModelPrediction(t, linear_velocity, angular_velocity, w_scale=10e-7)

	# (b) Feature detection and matching

	# (c) Landmark Mapping via EKF Update
	landmarks = landmarkMapping(features, imu_T_world, K, b, imu_T_cam)
	# (d) Visual-Inertial SLAM
	slam_iTw, slam_wTi, slam_landmarks = visual_slam(t, linear_velocity, angular_velocity, features, K, b, imu_T_cam)
	compareResult(world_T_imu, slam_wTi, landmarks, slam_landmarks)
	# You can use the function below to visualize the robot pose over time
	visualize_trajectory_2d(world_T_imu, show_ori = True)







