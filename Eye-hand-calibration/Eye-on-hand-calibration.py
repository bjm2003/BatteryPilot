import cv2
import numpy as np
import glob
import re

# 配置参数
CHECKERBOARD = (11, 8)           # 棋盘格内角点数量
SQUARE_SIZE = 0.015            # 棋盘格方块实际尺寸（单位：米）
CAM_CALIB_IMAGES = ".\\calib_images\\*.jpg"  # 标定图像路径
POSE_FILE = ".\\poses.txt"             # 机械臂位姿文件

def load_robot_poses(filename):
    """解析特殊格式的机械臂位姿文件"""
    poses = []
    with open(filename, 'r') as f:
        content = f.read()
    
    # 使用正则表达式匹配所有矩阵
    matrix_pattern = r"\[\[\s*([^]]+?)\s*\]\]"
    matrices = re.findall(matrix_pattern, content, re.DOTALL)
    
    for mat_str in matrices:
        # 清理分号和逗号，转换为numpy数组
        cleaned = re.sub(r"[;\t]", " ", mat_str)
        cleaned = re.sub(r",", " ", cleaned)
        data = np.fromstring(cleaned, sep=' ').reshape(4, 4)
        poses.append(data.astype(np.float32))
    
    return poses

def camera_calibration():
    """完整的相机标定流程"""
    # 准备3D对象点
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    objpoints = []  # 3D点
    imgpoints = []  # 2D点

    images = glob.glob(CAM_CALIB_IMAGES)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret:
            objpoints.append(objp)
            
            # 亚像素角点检测
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners_refined)

    # 执行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    return mtx, dist

def compute_handeye(robot_poses, mtx, dist):
    """执行手眼标定"""
    cam_to_base_transforms = []
    image_files = sorted(glob.glob(CAM_CALIB_IMAGES))
    
    # 准备3D对象点（实际尺寸）
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * SQUARE_SIZE

    for img_path, X_base_end in zip(image_files, robot_poses):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if not ret:
            continue
            
        # 亚像素细化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        
        # 求解PnP
        ret, rvec, tvec = cv2.solvePnP(objp, corners_refined, mtx, dist)
        
        # 构建相机到标定板的变换矩阵
        R_cam_marker, _ = cv2.Rodrigues(rvec)
        X_cam_marker = np.eye(4)
        X_cam_marker[:3, :3] = R_cam_marker
        X_cam_marker[:3, 3] = tvec.flatten()
        
        # 计算机器人基座到相机的变换
        X_base_end_inv = np.linalg.inv(X_base_end)
        X_base_cam = X_cam_marker @ X_base_end_inv
        
        cam_to_base_transforms.append(X_base_cam)
    
    # 平均变换矩阵
    def average_transforms(transforms):
        avg_trans = np.mean([t[:3, 3] for t in transforms], axis=0)
        
        # 四元数平均法
        quats = []
        for t in transforms:
            R = t[:3, :3]
            # 转换为四元数
            q = np.array([
                0.5 * np.sqrt(1 + R[0,0] + R[1,1] + R[2,2]),
                0.5 * np.sign(R[2,1] - R[1,2]) * np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]),
                0.5 * np.sign(R[0,2] - R[2,0]) * np.sqrt(1 - R[0,0] + R[1,1] - R[2,2]),
                0.5 * np.sign(R[1,0] - R[0,1]) * np.sqrt(1 - R[0,0] - R[1,1] + R[2,2])
            ])
            quats.append(q / np.linalg.norm(q))
        
        avg_quat = np.mean(quats, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)
        
        # 四元数转旋转矩阵
        w, x, y, z = avg_quat
        R_avg = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])
        
        avg_transform = np.eye(4)
        avg_transform[:3, :3] = R_avg
        avg_transform[:3, 3] = avg_trans
        return avg_transform
    
    return average_transforms(cam_to_base_transforms)

if __name__ == "__main__":
    # 1. 加载机械臂位姿
    robot_poses = load_robot_poses(POSE_FILE)
    print(f"成功加载 {len(robot_poses)} 个机械臂位姿")
    
    # 2. 相机标定
    mtx, dist = camera_calibration()
    print("\n相机内参矩阵:")
    print(mtx)
    
    # 3. 执行手眼标定
    if len(robot_poses) != len(glob.glob(CAM_CALIB_IMAGES)):
        print("警告：位姿数量与图像数量不一致")
    
    X_base_cam = compute_handeye(robot_poses, mtx, dist)
    
    print("\n最终相机到基座的变换矩阵:")
    print(X_base_cam)