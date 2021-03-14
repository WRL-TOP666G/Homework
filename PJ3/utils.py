import numpy as np
from numpy.linalg import norm, inv

#Loading Data
def load_data(file_name, load_features = False):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
        load_features: a boolean variable to indicate whether to load features
    Output:
        t: time stamp
            with shape 1*t
        features: visual feature point coordinates in stereo images,
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:

        t = data["time_stamps"] # time_stamps
        features = None

        # only load features for 03.npz
        # 10.npz already contains feature tracks
        if load_features:
            features = data["features"] # 4 x num_features : pixel coordinates of features
        
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame

    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam

#Math Tool
def HatMap(vec):
    '''
        Transform vec in R^3 to a skew-symmetric vec_hat in R^3x3
        
        Input:
            vec - a 3x1 vector
        Output:
            vec_hat - a 3x3 matrix
    '''
    vec_hat = np.zeros((3,3))
    vec_hat[2,1] = vec[0]
    vec_hat[1,2] = -vec[0]
    vec_hat[2,0] = -vec[1]
    vec_hat[0,2] = vec[1]
    vec_hat[0,1] = -vec[2]
    vec_hat[1,0] = vec[2]
    return vec_hat

def adjoint(p, theta):
    '''
        Map u=[p,theta]^T in R^6 to se(3) in 4x4
        
        Input:
            p     - a 3x1 vector
            theta - a 3x1 vector
        Output:
            u_adjoint - a 6x6 matrix in SE(3)
    ''' 
    p_hat     = HatMap(p)
    theta_hat = HatMap(theta)
    u_adjoint = np.zeros((6,6))
    u_adjoint[:3,:3] = theta_hat
    u_adjoint[3:,3:] = theta_hat
    u_adjoint[:3,3:] = p_hat
    return u_adjoint

def twist(p, theta):
    '''
        Map u=[p,theta]^T in R^6 to the adjoint of SE(3) in 6x6
        
        Input:
            p     - a 3x1 vector
            theta - a 3x1 vector
        Output:
            twist - a 4x4 matrix in se(3)
    '''
    twist        = np.zeros((4,4))
    theta_hat    = HatMap(theta)
    twist[:3,:3] = theta_hat
    twist[:3,3]  = p
    return twist

def Rodrigues_3(p, theta):
    '''
        u = [p,theta]^T
        Rodrigues formula for u where u is 6x1
    '''
    u  = twist(p,theta) # get u in SE(3)
    u2 = np.dot(u, u)   # u^2
    u3 = np.dot(u, u2)  # u^3
    u2_coeff = (1-np.cos(norm(theta)))/(norm(theta)**2)
    u3_coeff = (norm(theta)-np.sin(norm(theta)))/(np.power(norm(theta),3))
    
    T = np.eye(4) + u + u2_coeff*u2 + u3_coeff*u3
    return T

def approxRodrigues_3(p, theta):
    '''
        u = [v,w]^T
        Approximate Rodrigues formula for u where u is 6x1 to avoid nan
    '''
    u  = twist(p,theta) # get u in SE(3)
    T = np.eye(4) + u
    return T

def Rodrigues_6(p, theta):
    '''
        Rodrigues formula for u where u is 6x6
    '''
    u = adjoint(p, theta) # get u in adjoint of SE(3)
    u2 = np.dot(u, u)     # u^2
    u3 = np.dot(u, u2)    # u^3
    u4 = np.dot(u, u3)    # u^4
    u_coeff  = (3*np.sin(norm(theta)) - norm(theta)*np.cos(norm(theta))) / (2 * norm(theta))
    u2_coeff = (4 - norm(theta)*np.sin(norm(theta)) - 4*np.cos(norm(theta))) / (2 * norm(theta)**2)
    u3_coeff = (np.sin(norm(theta)) - norm(theta)*np.cos(norm(theta))) / (2 * np.power(norm(theta),3))
    u4_coeff = (2 - norm(theta)*np.sin(norm(theta)) - 2*np.cos(norm(theta))) / (2 * np.power(norm(theta),4))
    
    T = np.eye(6) + u_coeff*u + u2_coeff*u2 + u3_coeff*u3 + u4_coeff*u4
    return T
    
def meanPredict(mu, v, w, dt):
    '''
        EKF mean prediction
        
        Input:
            mu - current mean
            v  - current linear velocity
            w  - current rotational velocity
            dt - time interval
        Outputs:
            mu_pred - predicted mean
    '''
    p       = -dt * v
    theta   = -dt * w
    mu_pred = np.dot(Rodrigues_3(p, theta), mu)
    return mu_pred

def covPredict(cov, v, w, dt, noise):
    '''
        EKF covariance prediction
        
        Input:
            cov   - current covariance
            v     - current linear velocity
            w     - current rotational velocity
            dt    - time interval
            niose - motion noise covariance
        Outputs:
            cov_pred - predicted covariance
    '''
    p        = -dt * v
    theta    = -dt * w
    cov_pred = np.dot(Rodrigues_6(p, theta), cov)
    cov_pred = np.dot(cov_pred, Rodrigues_6(p, theta).T)
    cov_pred = cov_pred + noise
    return cov_pred

def getCalibration(K, b):
    '''
        Get calibration matrix M from K and b
        
        Input:
            K - camera calibration matrix
            b - stereo baseline
        Output:
            M - stereo camera calibration matrix
    '''
    M   = np.vstack([K[:2], K[:2]])
    arr = np.array([0, 0, -K[0,0]*float(b), 0]).reshape((4,1))
    M   = np.hstack([M, arr])
    return M

def projection(q):
    '''
        Get the projection of a vector in R^4
        
        Input:
            q  - a vector in R^4
        Output:
            pi - corresponding projection
    '''
    pi = q / q[2]
    return pi

def d_projection(q):
    '''
        Take a R^4 vector and return the derivative of its projection function
        
        Input:
            q  - a vector in R^4
        Output:
            dq - corresponding derivative of the projection function, size 4x4
    '''
    dq = np.zeros((4,4))
    dq[0,0] = 1
    dq[1,1] = 1
    dq[0,2] = -q[0]/q[2]
    dq[1,2] = -q[1]/q[2]
    dq[3,2] = -q[3]/q[2]
    dq[3,3] = 1
    dq = dq / q[2]
    return dq

def pixel_to_world(p, i_T_w, o_T_i, K, b):
    '''
        Get homogeneous coordinates xyz in world frame from left right pixels
        
        Input:
            p     - left right pixels, size 1x4
            i_T_w - current pose of IMU, size 4x4
            o_T_i - imu to optical frame rotation, size 3x3
            K     - camera calibration matrix
            b     - stereo baseline
        Output:
            m_w   - homogeneous coordinates
    '''
    uL, vL, uR, vR = p
    fsu = K[0,0]
    fsv = K[1,1]
    cu  = K[0,2]
    cv  = K[1,2]
    z   = (fsu*b) / (uL-uR)
    x   = z * (uL-cu) / fsu
    y   = z * (vL-cv) / fsv
    m_o = np.array([x,y,z,1]).reshape([4,1])
    m_i = np.dot(inv(o_T_i), m_o)
    m_w = np.dot(inv(i_T_w), m_i)
    return m_w

def get_Ht(H_list, num_feature, isSLAM=False):
    '''
        Get the model Jacobian
    '''
    if isSLAM:
        Nt = len(H_list)
        Ht = np.zeros([4*Nt, 3*num_feature+6])
        for i in range(Nt):
            j = H_list[i][0]      # landmark index
            H_obs  = H_list[i][1]
            H_pose = H_list[i][2]
            Ht[i*4:(i+1)*4, 3*j:3*(j+1)] = H_obs
            Ht[i*4:(i+1)*4, -6:] = H_pose
    else:
        Nt = len(H_list)
        Ht = np.zeros([4*Nt, 3*num_feature])
        for i in range(Nt):
            j = H_list[i][0]      # landmark index
            H = H_list[i][1]      # current Hij
            Ht[i*4:(i+1)*4,3*(j):3*(j+1)] = H
    return Ht

def get_Kt(cov, Ht, v):
    '''
        Get the Kalman gain
    '''
    V_noise  = np.eye(Ht.shape[0]) * v
    inv_term = np.dot(Ht, np.dot(cov, Ht.T)) + V_noise
    Kt       = np.dot(np.dot(cov, Ht.T), inv(inv_term))
    return Kt

def circle(m):
    '''
        circle operator
        
        Input:
            m - a vector in R^4, [x,y,z,1]
        Output:
            result - a matrix of size 4x6
    '''
    s      = m[:3]
    s_hat  = HatMap(s)
    result = np.hstack((np.eye(3), -s_hat))
    result = np.vstack((result, np.zeros((1,6))))
    return result

#Model
def motionModelPrediction(t, v, w, w_scale):
    '''
        Get IMU pose using EKF prediction

        Input:
            t - time stamps
            v - linear velocity
            w - angular velocity
            w_scale - scale of the motion noise
        Outputs:
            pose     - world to IMU frame T over time, size 4x4xN
            inv_pose - IMU to world frame T over time, size 4x4xN
    '''
    # get time discretization
    tau = t[:, 1:] - t[:, :-1]
    n = tau.shape[1]

    # initialize mu, covariance, and noise
    mu = np.eye(4)
    cov = np.eye(6)
    W_noise = np.eye(6) * w_scale

    # poses
    imuPose = np.zeros((4, 4, n + 1))  # w_T_i
    invImuPose = np.zeros((4, 4, n + 1))  # i_T_w
    imuPose[:, :, 0] = mu
    invImuPose[:, :, 0] = inv(mu)

    for i in range(n):
        dt = tau[:, i]
        linear_noise = np.random.randn(3) * w_scale
        angular_noise = np.random.randn(3) * w_scale
        v_curr = v[:, i] + linear_noise
        w_curr = w[:, i] + angular_noise
        mu = meanPredict(mu, v_curr, w_curr, dt)
        cov = covPredict(cov, v_curr, w_curr, dt, W_noise)
        invImuPose[:, :, i + 1] = mu
        imuPose[:, :, i + 1] = inv(mu)

    return invImuPose, imuPose

# Land
def landmarkMapping(features, i_T_w, K, b, imu_T_cam, v_scale=100):
    from time import sleep

    '''
        Get landmarks position using EKF update

        Input:
            features  - landmarks
            i_T_w     - inverse imu pose
            K         - camera calibration matrix
            b         - stereo baseline
            imu_T_cam - camera to imu transformation
            v_scale   - scale of the observation noise
        Outputs:
            landmarks - landmarks position in the world frame
    '''
    # print(type(features))
    num_feature = features.shape[1]
    mean_hasinit = np.zeros(num_feature)
    mean = np.zeros((4 * num_feature, 1))
    cov = np.eye(3 * num_feature)
    M = getCalibration(K, b)
    P = np.vstack([np.eye(3), np.zeros([1, 3])])  # projection matrix
    P_block = np.tile(P, [num_feature, num_feature])

    #######
    total = features.shape[2]
    print(total)
    ######
    for i in range(features.shape[2]):
        print('\r' + '[Progress]:[%s%s]%.2f%%;' % (
        '|' * int(i * 20 / total), ' ' * (20 - int(i * 20 / total)), float(i / total * 100)), end='')
        # if(i%100==0):
        # print(i)
        Ut = i_T_w[:, :, i]  # current inverse IMU pose
        feature = features[:, :, i]  # current landmarks
        zt = np.array([])  # to store zt
        zt_hat = np.array([])  # to store zt_hat
        H_list = []  # to store Hij
        observation_noise = np.random.randn() * np.sqrt(v_scale)
        for j in range(feature.shape[1]):
            # if is a valid feature
            if (feature[:, j] != np.array([-1, -1, -1, -1])).all():
                # check if has seen before
                # initialize if not seen before
                # update otherwise
                if (mean_hasinit[j] == 0):
                    m = pixel_to_world(feature[:, j], Ut, imu_T_cam, K, b)
                    mean[4 * j:4 * (j + 1)] = m
                    cov[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] = np.eye(3) * 1e-3
                    mean_hasinit[j] = 1  # mark as seen
                else:
                    mean_curr = mean[4 * j:4 * (j + 1)]
                    q = np.dot(imu_T_cam, np.dot(Ut, mean_curr))
                    zt = np.concatenate((zt, feature[:, j] + observation_noise), axis=None)
                    zt_hat = np.concatenate((zt_hat, np.dot(M, projection(q))), axis=None)
                    # compute H_ij
                    H = ((M.dot(d_projection(q))).dot(imu_T_cam).dot(Ut)).dot(P)
                    H_list.append((j, H))

        Nt = len(H_list)
        zt = zt.reshape([4 * Nt, 1])
        zt_hat = zt_hat.reshape([4 * Nt, 1])
        Ht = get_Ht(H_list, num_feature)
        Kt = get_Kt(cov, Ht, v_scale)

        # update mu and cov
        mean = mean + P_block.dot(Kt.dot(zt - zt_hat))
        cov = np.dot((np.eye(3 * num_feature) - np.dot(Kt, Ht)), cov)

    landmarks = mean.reshape([num_feature, 4])
    return landmarks

#Visual
def visual_slam(t, v, w, features, K, b, imu_T_cam, v_scale=100, w_scale=10e-5):
    # get time discretization
    tau = t[:, 1:] - t[:, :-1]
    n = tau.shape[1]

    numFeature = features.shape[1]

    # initialize mean and covariance
    mean_imu = np.eye(4)
    mean_obs = np.zeros((4 * numFeature, 1))
    cov = np.eye(3 * numFeature + 6)
    W_noise = np.eye(6) * w_scale

    # poses
    imu_pose = np.zeros((4, 4, n + 1))  # w_T_i
    inv_imu_pose = np.zeros((4, 4, n + 1))  # i_T_w
    imu_pose[:, :, 0] = mean_imu
    inv_imu_pose[:, :, 0] = inv(mean_imu)

    mean_hasinit = np.zeros(numFeature)
    M = getCalibration(K, b)
    P = np.vstack([np.eye(3), np.zeros([1, 3])])  # projection matrix
    P_block = np.tile(P, [numFeature, numFeature])

    for i in range(n):
        if (i % 100 == 0):
            print(i)
        dt = tau[:, i]  # time interval
        Ut = mean_imu  # current inverse IMU pose
        feature = features[:, :, i]  # current landmarks
        zt = np.array([])  # to store zt
        ztHat = np.array([])  # to store zt_hat
        H_list = []  # to store Hij
        observation_noise = np.random.randn() * np.sqrt(v_scale)
        for j in range(feature.shape[1]):
            # if is a valid feature
            if (feature[:, j] != np.array([-1, -1, -1, -1])).all():
                # check if has seen before
                # initialize if not seen before
                # update otherwise
                if (mean_hasinit[j] == 0):
                    m = pixel_to_world(feature[:, j], Ut, imu_T_cam, K, b)
                    mean_obs[4 * j:4 * (j + 1)] = m
                    cov[3 * j:3 * (j + 1), 3 * j:3 * (j + 1)] = np.eye(3) * 1e-4
                    mean_hasinit[j] = 1  # mark as seen
                else:
                    mean_curr = mean_obs[4 * j:4 * (j + 1)]
                    q1 = np.dot(imu_T_cam, np.dot(Ut, mean_curr))
                    q2 = circle(np.dot(Ut, mean_curr))
                    zt = np.concatenate((zt, feature[:, j] + observation_noise), axis=None)
                    ztHat = np.concatenate((ztHat, np.dot(M, projection(q1))), axis=None)
                    # compute H
                    H_obs = ((M.dot(d_projection(q1))).dot(imu_T_cam).dot(Ut)).dot(P)
                    H_pose = (M.dot(d_projection(q1))).dot(imu_T_cam).dot(q2)
                    H_list.append((j, H_obs, H_pose))

        Nt = len(H_list)
        zt = zt.reshape([4 * Nt, 1])
        ztHat = ztHat.reshape([4 * Nt, 1])
        Ht = get_Ht(H_list, numFeature, isSLAM=True)
        Kt = get_Kt(cov, Ht, v_scale)

        # update mean and cov
        Kt_obs = Kt[:-6, :]  # 3M x 4Nt
        Kt_pose = Kt[-6:, :]  # 6 x 4Nt

        p = np.dot(Kt_pose, zt - ztHat)[:3].flatten()
        theta = np.dot(Kt_pose, zt - ztHat)[-3:].flatten()

        mean_obs = mean_obs + P_block.dot(Kt_obs.dot(zt - ztHat))

        mean_imu = np.dot(approxRodrigues_3(p, theta), mean_imu)
        cov    = np.dot((np.eye(3*numFeature+6) - np.dot(Kt,Ht)),cov)

        # store imu pose
        inv_imu_pose[:, :, i + 1] = mean_imu
        imu_pose[:, :, i + 1] = inv(mean_imu)

        # predict mean and cov
        noise = np.random.randn() * w_scale
        v_curr = v[:, i] + noise
        w_curr = w[:, i] + noise
        mean_imu = meanPredict(mean_imu, v_curr, w_curr, dt)
        cov[-6:, -6:] = covPredict(cov[-6:, -6:], v_curr, w_curr, dt, W_noise)

    mean_obs = mean_obs.reshape((numFeature, 4))
    return inv_imu_pose, imu_pose, mean_obs
