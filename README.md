# BatteryPilot Project Description

## Project Purpose
BatteryPilot is a project integrating camera calibration, hand-eye calibration, and battery detection functionalities. It is designed to enable the collaboration between robotic arms and vision systems, specifically for automated battery inspection and processing scenarios.

## File Structure
```
BatteryPilot/
├── Eye-hand-calibration/          # Hand-eye calibration related code
│   ├── Eye-in-hand-calibration.py # Implementation of Eye-in-hand calibration
│   ├── Eye-on-hand-calibration.py # Implementation of Eye-on-hand calibration
│   ├── Eye-Hand_with_UI_loss.py   # Hand-eye calibration with UI (including error calculation)
│   ├── Eye_in_hand_with_UI.py     # Eye-in-hand calibration with UI
│   ├── Camera_calibration.py      # Camera calibration implementation
│   └── Cam_cal_with_UI.py         # Camera calibration with UI
├── calib_images                   # Referable calibration board image
├── data_for_PCL                   # A test file in .ply format
└── DetectBattery.cpp              # Battery detection (based on point cloud processing)
```

## Usage Instructions

### 1. Camera Calibration
- Basic calibration: Run `Camera_calibration.py`, ensuring chessboard images are prepared in advance
- Calibration with UI: Run `Cam_cal_with_UI.py`, set parameters and select image directory through the interface

### 2. Hand-Eye Calibration
- Select the corresponding script based on the installation method of the robotic arm and camera:
  - Eye-in-hand configuration: `Eye-in-hand-calibration.py` or the UI version `Eye_in_hand_with_UI.py`
  - Other installation methods: `Eye-on-hand-calibration.py`
- Requirements: Camera calibration images, robotic arm pose files

### 3. Battery Detection
- Compile and run `DetectBattery.cpp` (requires PCL library)
- Can read PLY format point cloud files to realize battery recognition and parameter extraction (center coordinates, axis direction, radius, etc.)

## Notes
- Ensure chessboard images are clear and captured from multiple angles before calibration
- Robotic arm pose files must correspond one-to-one with calibration images
- The point cloud processing part requires a configured PCL library environment

##
## BatteryPilot 项目说明

## 项目用途
BatteryPilot 是一个包含相机标定、手眼标定以及电池检测功能的项目，主要用于实现机械臂与视觉系统的协同工作，可应用于电池的自动化检测与处理场景。

## 文件结构
```
BatteryPilot/
├── Eye-hand-calibration/          # 手眼标定相关代码
│   ├── Eye-in-hand-calibration.py # 眼在手上（Eye-in-hand）标定实现
│   ├── Eye-on-hand-calibration.py # 眼在手上（Eye-on-hand）标定实现
│   ├── Eye-Hand_with_UI_loss.py   # 带UI的手眼标定（含误差计算）
│   ├── Eye_in_hand_with_UI.py     # 带UI的眼在手上标定
│   ├── Camera_calibration.py      # 相机标定实现
│   └── Cam_cal_with_UI.py         # 带UI的相机标定
├── calib_images                   # 可参考的标定板图片
├── data_for_PCL                   # 一个.ply格式的测试文件
└── DetectBattery.cpp              # 电池检测（基于点云处理）
```

## 用法说明

### 1. 相机标定
- 基础标定：运行 `Camera_calibration.py`，需提前准备棋盘格图像
- 带UI的标定：运行 `Cam_cal_with_UI.py`，通过界面设置参数并选择图像目录

### 2. 手眼标定
- 根据机械臂与相机安装方式选择对应的脚本：
  - 眼在手上：`Eye-in-hand-calibration.py` 或带UI的 `Eye_in_hand_with_UI.py`
  - 其他安装方式：`Eye-on-hand-calibration.py`
- 需准备：相机标定图像、机械臂位姿文件

### 3. 电池检测
- 编译并运行 `DetectBattery.cpp`（依赖PCL库）
- 可读取PLY格式点云文件，实现电池的识别与参数提取（中心坐标、轴线方向、半径等）

## 注意事项
- 标定前需确保棋盘格图像清晰，多角度拍摄
- 机械臂位姿文件需与标定图像一一对应
- 点云处理部分需配置PCL库环境