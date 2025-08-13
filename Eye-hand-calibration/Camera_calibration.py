import cv2
import numpy as np
import glob
import os
import json

# 设置棋盘格尺寸（内部角点数，例如：9x6表示每行9个角点，每列6个角点）
CHECKERBOARD = (11, 8)  # 请根据实际标定板修改

# 终止亚像素细化条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 创建输出目录
os.makedirs('output_calib_images_2', exist_ok=True)
os.makedirs('calibration_results_2', exist_ok=True)

# 准备对象点（3D点）
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 存储对象点和图像点的数组
objpoints = []  # 真实世界3D点
imgpoints = []  # 图像中的2D点

# 读取所有标定图像
images = glob.glob('C:\\Users\\15594\\OneDrive\\Desktop\\Eye-hand-calibration\\calib_images_3\\*.jpg')

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"无法读取图像文件: {fname}，请检查文件路径和文件格式。")
        continue
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

    if ret:
        objpoints.append(objp)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_refined)
        
        # 绘制并保存带角点的图像
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
        output_path = os.path.join('output_calib_images_2', os.path.basename(fname))
        cv2.imwrite(output_path, img)
        
        cv2.imshow('Corners', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # 构建保存数据结构
    calibration_data = {
        "intrinsic_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist(),
        "extrinsic_parameters": []
    }
    
    # 转换外参数据
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        R, _ = cv2.Rodrigues(rvec)
        calibration_data["extrinsic_parameters"].append({
            "image_index": i+1,
            "rotation_matrix": R.tolist(),
            "translation_vector": tvec.squeeze().tolist()
        })
    
    # 保存到JSON文件
    with open('calibration_results_2/calibration_data.json', 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    # 控制台输出
    print("标定结果已保存到 calibration_data.json")
    print("\n相机内参矩阵:")
    print(mtx)
    
else:
    print("错误：未检测到任何棋盘格角点，请检查：")
    print("1. CHECKERBOARD尺寸设置是否正确")
    print("2. 图像路径是否正确")
    print("3. 标定板是否清晰可见且角度合适")