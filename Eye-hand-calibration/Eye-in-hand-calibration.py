import cv2
import numpy as np
import scipy.linalg
import os

def read_robot_poses(file_path):
    """读取机械臂末端位姿文件"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    matrices = []
    matrix = []
    for line in lines:
        line = line.strip()
        if not line:
            if matrix:
                matrices.append(np.array(matrix, dtype=np.float32))
                matrix = []
        else:
            values = list(map(float, line.split()))
            matrix.append(values)
    if matrix:
        matrices.append(np.array(matrix, dtype=np.float32))
    return matrices

def camera_calibration(image_folder, board_size=(11, 8)):
    """相机标定函数"""
    # 准备棋盘格角点世界坐标（单位：mm）
    objp = np.zeros((board_size[0]*board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * 15.0  # 棋盘格尺寸为15mm

    objpoints = []  # 3D点
    imgpoints = []  # 2D点

    # 获取所有图像路径并排序
    images = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]
    images.sort()

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            # 绘制角点并保存
            cv2.drawChessboardCorners(img, board_size, corners, ret)
            cv2.imwrite('corners_' + os.path.basename(fname), img)
        else:
            print(f"无法找到角点：{fname}")

    # 执行相机标定
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    print("\n相机内参矩阵 (K):")
    print(K)
    print("\n畸变系数 (dist):")
    print(dist)

    # 计算每个图像的外参矩阵（相机到标定板的变换）
    extrinsics = []
    for i in range(len(rvecs)):
        R, _ = cv2.Rodrigues(rvecs[i])
        t = tvecs[i].reshape(3, 1)
        # 构建4x4齐次变换矩阵
        A = np.hstack((R, t))
        A = np.vstack((A, [0, 0, 0, 1]))
        extrinsics.append(A)
    
    return K, dist, extrinsics

def hand_eye_calibration(extrinsics, robot_poses):
    """手眼标定函数"""
    # 计算相邻位姿的相对变换
    C_list = []
    D_list = []
    for i in range(1, len(extrinsics)):
        # 相机坐标系变换 C = A_current * A_prev^{-1}
        A_prev = extrinsics[i-1]
        A_current = extrinsics[i]
        C = A_current @ np.linalg.inv(A_prev)
        C_list.append(C)
        
        # 机械臂末端变换 D = B_current * B_prev^{-1}
        B_prev = robot_poses[i-1]
        B_current = robot_poses[i]
        D = B_current @ np.linalg.inv(B_prev)
        D_list.append(D)

    # 构建旋转部分的方程矩阵
    kron_blocks = []
    for i in range(len(C_list)):
        R_c = C_list[i][:3, :3]
        R_d = D_list[i][:3, :3]
        # 方程：R_c R_x = R_x R_d → kron(R_c, I) - kron(I, R_d^T)
        kron = np.kron(R_c, np.eye(3)) - np.kron(np.eye(3), R_d.T)
        kron_blocks.append(kron)
    
    C_total = np.vstack(kron_blocks)

    # SVD求解旋转矩阵
    U, S, Vh = np.linalg.svd(C_total)
    vec_Rx = Vh[-1, :]  # 最小奇异值对应的右奇异向量
    Rx = vec_Rx.reshape(3, 3)
    
    # 正交化处理
    Rx = scipy.linalg.orth(Rx)
    if np.linalg.det(Rx) < 0:
        Rx[:, -1] *= -1  # 确保行列式为1

    # 构建平移部分的方程
    A_blocks = []
    b_blocks = []
    for i in range(len(C_list)):
        C = C_list[i]
        D = D_list[i]
        R_c = C[:3, :3]
        t_c = C[:3, 3].reshape(3, 1)
        R_d = D[:3, :3]
        t_d = D[:3, 3].reshape(3, 1)
        
        # 方程：R_c t_x + t_c = Rx R_d t_x + Rx t_d
        A_block = R_c - Rx @ R_d
        b_block = Rx @ t_d - t_c
        A_blocks.append(A_block)
        b_blocks.append(b_block)
    
    A = np.vstack(A_blocks)
    b = np.vstack(b_blocks)

    # 最小二乘求解平移向量
    t_x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    t_x = t_x.reshape(3, 1)

    # 组合成最终变换矩阵
    X = np.hstack((Rx, t_x))
    X = np.vstack((X, [0, 0, 0, 1]))
    return X
    
if __name__ == '__main__':
    IMAGE_FOLDER = '.\\calib_images_4'  # 包含30张标定图片的文件夹
    ROBOT_POSES_FILE = 'robot_end.txt'  # 机械臂位姿文件
    
    # 读取机械臂位姿
    robot_poses = read_robot_poses(ROBOT_POSES_FILE)
    print(f"成功读取 {len(robot_poses)} 个机械臂位姿")

    # 执行相机标定
    K, dist, extrinsics = camera_calibration(IMAGE_FOLDER)
    
    # 执行手眼标定
    X = hand_eye_calibration(extrinsics, robot_poses)
    
    print("\n最终手眼标定结果 (机械臂末端 → 相机坐标系):")
    print(X)