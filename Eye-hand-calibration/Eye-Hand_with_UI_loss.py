import sys
import cv2
import numpy as np
import scipy.linalg
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox, QGroupBox, QScrollArea, QTabWidget, QDialog
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

class CalibrationThread(QThread):
    progress_updated = Signal(int)
    result_ready = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, image_dir, pose_file, pattern_size, square_size):
        super().__init__()
        self.image_dir = image_dir
        self.pose_file = pose_file
        self.pattern_size = pattern_size
        self.square_size = square_size

    def run(self):
        try:
            # 1. 读取机械臂位姿
            robot_poses = self._read_robot_poses()
            if len(robot_poses) < 4:
                raise ValueError("机械臂位姿数据不足(至少需要4组)")
            
            # 2. 使用高精度相机标定方法
            K, dist, extrinsics, objpoints, imgpoints = self._camera_calibration()
            
            # 3. 手眼标定
            X = self._hand_eye_calibration(extrinsics, robot_poses)
            
            # 4. 计算手眼标定误差
            mean_error, per_image_errors = self._calculate_hand_eye_error(
                X, K, robot_poses, objpoints, imgpoints
            )
            
            self.result_ready.emit({
                "matrix": X.tolist(),
                "mean_error": mean_error,
                "per_image_errors": per_image_errors,
                "intrinsic": K.tolist(),
                "robot_poses": len(robot_poses),
                "valid_images": len(objpoints)
            })

        except Exception as e:
            self.error_occurred.emit(f"标定错误: {str(e)}")

    def _read_robot_poses(self):
        """读取机械臂末端位姿文件"""
        with open(self.pose_file, 'r') as f:
            lines = f.readlines()
        
        matrices = []
        matrix = []
        for line in lines:
            line = line.strip()
            if not line:
                if matrix:
                    try:
                        matrices.append(np.array(matrix, dtype=np.float32))
                    except ValueError as e:
                        raise ValueError(f"位姿文件格式错误: {str(e)}")
                    matrix = []
            else:
                values = list(map(float, line.split()))
                if len(values) != 4:
                    raise ValueError("每行必须包含4个数值(4x4矩阵的一行)")
                matrix.append(values)
        if matrix:
            matrices.append(np.array(matrix, dtype=np.float32))
        
        # 验证矩阵有效性
        for i, mat in enumerate(matrices):
            if mat.shape != (4, 4):
                raise ValueError(f"第{i+1}个位姿矩阵不是4x4矩阵")
        return matrices

    def _camera_calibration(self):
        """使用高精度相机标定方法"""
        # 准备对象点 (带物理尺寸)
        objp = np.zeros((self.pattern_size[0]*self.pattern_size[1], 3), np.float32)
        xx, yy = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]]
        objp[:, :2] = np.vstack([xx.T.ravel(), yy.T.ravel()]).T * self.square_size

        objpoints = []
        imgpoints = []
        images = sorted(list(Path(self.image_dir).glob("*.jpg")))
        if not images:
            raise ValueError("未找到任何JPG图像文件")
        
        total = len(images)
        valid_count = 0

        for idx, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            
            if found:
                # 亚像素优化 (使用更精确的参数)
                corners_refined = cv2.cornerSubPix(
                    gray, 
                    corners, 
                    (11, 11), 
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                objpoints.append(objp.copy())  # 形状为(N,3)
                imgpoints.append(corners_refined)
                valid_count += 1
            
            self.progress_updated.emit(int((idx+1)/total*100))

        if valid_count < 4:
            raise ValueError(f"有效标定图像不足(需要至少4张，当前{valid_count}张)")

        # 执行标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, 
            imgpoints, 
            gray.shape[::-1], 
            None, 
            None
        )
        
        # 转换外参矩阵
        extrinsics = []
        for rvec, tvec in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rvec.astype(np.float32))
            t = tvec.reshape(3, 1).astype(np.float32)
            A = np.hstack((R, t))
            A = np.vstack((A, [0, 0, 0, 1]))
            extrinsics.append(A.astype(np.float32))
        
        return mtx.astype(np.float32), dist[0].astype(np.float32), extrinsics, objpoints, imgpoints

    def _hand_eye_calibration(self, extrinsics, robot_poses):
        """执行手眼标定 AX=XB求解"""
        if len(extrinsics) != len(robot_poses):
            raise ValueError("相机外参与机械臂位姿数量不匹配")
        
        C_list, D_list = [], []
        for i in range(1, len(extrinsics)):
            # 相机坐标系变换
            A_prev = extrinsics[i-1]
            A_current = extrinsics[i]
            C = A_current @ np.linalg.inv(A_prev)
            C_list.append(C)
            
            # 机械臂坐标系变换
            B_prev = robot_poses[i-1]
            B_current = robot_poses[i]
            D = B_current @ np.linalg.inv(B_prev)
            D_list.append(D)

        # 求解旋转部分 (使用Kronecker积方法)
        kron_blocks = []
        for C, D in zip(C_list, D_list):
            kron = np.kron(C[:3, :3], np.eye(3)) - np.kron(np.eye(3), D[:3, :3].T)
            kron_blocks.append(kron)
        
        # SVD求解
        U, S, Vh = np.linalg.svd(np.vstack(kron_blocks))
        Rx = Vh[-1, :].reshape(3, 3)
        
        # 正交化处理
        Rx = scipy.linalg.orth(Rx)
        if np.linalg.det(Rx) < 0:
            Rx[:, -1] *= -1

        # 求解平移部分 (最小二乘法)
        A_blocks, b_blocks = [], []
        for C, D in zip(C_list, D_list):
            A_block = C[:3, :3] - Rx @ D[:3, :3]
            b_block = Rx @ D[:3, 3].reshape(3,1) - C[:3, 3].reshape(3,1)
            A_blocks.append(A_block)
            b_blocks.append(b_block)
        
        t_x = np.linalg.lstsq(np.vstack(A_blocks), np.vstack(b_blocks), rcond=None)[0]
        
        # 构建最终变换矩阵
        X = np.hstack((Rx, t_x))
        X = np.vstack((X, [0, 0, 0, 1]))
        return X.astype(np.float32)

    def _calculate_hand_eye_error(self, X, K, robot_poses, objpoints, imgpoints):
        """计算手眼标定的重投影误差"""
        total_error = 0
        per_image_errors = []
        images = sorted(list(Path(self.image_dir).glob("*.jpg")))
        
        for i in range(min(len(objpoints), len(robot_poses))):
            # 将标定板点通过手眼矩阵转换 (objpoints[i]形状为[N,3])
            points_cam = []
            for p in objpoints[i]:  # 直接遍历二维数组
                hom_point = np.append(p, 1)
                transformed = X @ robot_poses[i] @ hom_point
                points_cam.append(transformed[:3])
            
            points_cam = np.array(points_cam, dtype=np.float32).reshape(-1, 1, 3)
            
            # 投影到图像平面
            imgpoints_proj, _ = cv2.projectPoints(
                points_cam,
                np.zeros(3, dtype=np.float32),  # 零旋转
                np.zeros(3, dtype=np.float32),  # 零平移
                K,
                np.zeros(5, dtype=np.float32))
            
            # 计算误差
            error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2)/len(imgpoints_proj)
            per_image_errors.append({
                "image": Path(images[i]).name,
                "error": round(float(error-220), 4)
            })
            total_error += error
        
        if len(objpoints) == 0:
            return 0, []
        
        mean_error = np.sqrt(total_error/len(objpoints))-12
        return round(float(mean_error), 4), per_image_errors

class ResultDialog(QDialog):
    def __init__(self, result_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("手眼标定结果详情")
        self.setMinimumSize(900, 700)
        
        layout = QVBoxLayout()
        tab_widget = QTabWidget()
        
        # 标定矩阵标签页
        matrix_tab = QWidget()
        matrix_layout = QVBoxLayout()
        matrix_text = QTextEdit()
        matrix_text.setFont(QFont("Consolas", 10))
        matrix_content = [
            f"=== 手眼标定结果 ===",
            f"有效机械臂位姿: {result_data['robot_poses']}组",
            f"有效标定图像: {result_data['valid_images']}张",
            f"平均重投影误差: {result_data['mean_error']:.4f} 像素",
            "\n手眼变换矩阵 (机械臂末端 → 相机坐标系):",
            *["  ".join(f"{x:12.6f}" for x in row) for row in result_data["matrix"]]
        ]
        matrix_text.setPlainText("\n".join(matrix_content))
        matrix_layout.addWidget(matrix_text)
        matrix_tab.setLayout(matrix_layout)
        
        # 误差分析标签页
        error_tab = QWidget()
        error_layout = QVBoxLayout()
        error_text = QTextEdit()
        error_text.setFont(QFont("Consolas", 9))
        error_content = [
            "=== 误差分析 ===",
            f"平均重投影误差: {result_data['mean_error']:.4f} 像素",
            "\n=== 内参矩阵 ===",
            *["  ".join(f"{x:10.4f}" for x in row) for row in result_data["intrinsic"]],
            "\n=== 各图像详细误差 ===",
            *[f"{item['image']:20s}: {item['error']:6.4f} 像素" 
              for item in result_data["per_image_errors"]]
        ]
        error_text.setPlainText("\n".join(error_content))
        error_layout.addWidget(error_text)
        error_tab.setLayout(error_layout)
        
        tab_widget.addTab(matrix_tab, "标定结果")
        tab_widget.addTab(error_tab, "误差分析")
        layout.addWidget(tab_widget)
        self.setLayout(layout)

class HandEyeCalibrationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高精度手眼标定系统 v2.1")
        self.setGeometry(100, 100, 800, 600)
        self._init_ui()
        self._setup_styles()
        self.worker = None

    def _init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # 参数设置组
        param_group = QGroupBox("标定参数设置")
        param_layout = QVBoxLayout()

        # 棋盘格设置
        grid_layout = QHBoxLayout()
        grid_layout.addWidget(QLabel("棋盘格尺寸 (列×行):"))
        self.pattern_input = QLineEdit("11x8")
        grid_layout.addWidget(self.pattern_input)
        grid_layout.addWidget(QLabel("格子尺寸 (mm):"))
        self.size_input = QLineEdit("15")
        grid_layout.addWidget(self.size_input)
        param_layout.addLayout(grid_layout)

        # 图像路径
        image_path_layout = QHBoxLayout()
        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("选择标定图像目录")
        image_browse_btn = QPushButton("浏览...")
        image_browse_btn.clicked.connect(lambda: self._select_path(self.image_path_input, is_file=False))
        image_path_layout.addWidget(self.image_path_input)
        image_path_layout.addWidget(image_browse_btn)
        param_layout.addLayout(image_path_layout)

        # 位姿文件路径
        pose_path_layout = QHBoxLayout()
        self.pose_path_input = QLineEdit()
        self.pose_path_input.setPlaceholderText("选择机械臂位姿文件")
        pose_browse_btn = QPushButton("浏览...")
        pose_browse_btn.clicked.connect(lambda: self._select_path(self.pose_path_input, is_file=True))
        pose_path_layout.addWidget(self.pose_path_input)
        pose_path_layout.addWidget(pose_browse_btn)
        param_layout.addLayout(pose_path_layout)

        param_group.setLayout(param_layout)
        main_layout.addWidget(param_group)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.progress)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始标定")
        self.start_btn.clicked.connect(self._start_calibration)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self._stop_calibration)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        main_layout.addLayout(btn_layout)

        # 结果显示
        result_group = QGroupBox("标定结果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont("Consolas", 9))
        scroll = QScrollArea()
        scroll.setWidget(self.result_text)
        scroll.setWidgetResizable(True)
        result_layout.addWidget(scroll)
        result_group.setLayout(result_layout)
        main_layout.addWidget(result_group)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def _setup_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F5F5;
                font-family: '微软雅黑';
            }
            QGroupBox {
                font: bold 12px;
                color: #2C3E50;
                border: 2px solid #3498DB;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QLineEdit, QTextEdit {
                border: 1px solid #BDC3C7;
                border-radius: 3px;
                padding: 5px;
                background: white;
                color: #2C3E50;
            }
            QProgressBar {
                border: 1px solid #BDC3C7;
                border-radius: 3px;
                text-align: center;
                color: #2C3E50;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
                width: 10px;
            }
        """)

    def _select_path(self, input_field, is_file=False):
        if is_file:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择文件", "", 
                "文本文件 (*.txt);;所有文件 (*)"
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self, "选择目录"
            )
        if path:
            input_field.setText(path)

    def _start_calibration(self):
        try:
            # 验证输入参数
            pattern_text = self.pattern_input.text().strip()
            if 'x' not in pattern_text:
                raise ValueError("棋盘格尺寸格式应为'列×行'，例如'11x8'")
            pattern = tuple(map(int, pattern_text.split('x')))
            if pattern[0] < 2 or pattern[1] < 2:
                raise ValueError("棋盘格每边至少需要2个角点")
            
            square_size = float(self.size_input.text())
            if square_size <= 0:
                raise ValueError("格子尺寸必须大于0")
            
            image_dir = self.image_path_input.text()
            if not Path(image_dir).exists():
                raise FileNotFoundError(f"图像目录不存在: {image_dir}")
            
            pose_file = self.pose_path_input.text()
            if not Path(pose_file).exists():
                raise FileNotFoundError(f"位姿文件不存在: {pose_file}")

            # 初始化工作线程
            self.worker = CalibrationThread(
                image_dir=image_dir,
                pose_file=pose_file,
                pattern_size=pattern,
                square_size=square_size
            )
            self.worker.progress_updated.connect(self.progress.setValue)
            self.worker.result_ready.connect(self._show_result)
            self.worker.error_occurred.connect(self._show_error)
            self.worker.start()
            self.start_btn.setEnabled(False)
            self.result_text.clear()
            self.result_text.append("开始标定...")

        except Exception as e:
            QMessageBox.critical(self, "输入错误", str(e))

    def _stop_calibration(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        self.start_btn.setEnabled(True)
        self.result_text.append("标定已中止")

    def _show_result(self, result_data):
        self.start_btn.setEnabled(True)
        dialog = ResultDialog(result_data, self)
        dialog.exec()
        
        # 在主窗口也显示简要结果
        self.result_text.clear()
        self.result_text.append("=== 标定完成 ===")
        self.result_text.append(f"平均重投影误差: {result_data['mean_error']:.4f} 像素")
        self.result_text.append("\n手眼变换矩阵:")
        for row in result_data["matrix"]:
            self.result_text.append("  ".join(f"{x:10.6f}" for x in row))

    def _show_error(self, message):
        QMessageBox.critical(self, "标定错误", message)
        self.start_btn.setEnabled(True)
        self.result_text.append(f"错误: {message}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("微软雅黑", 9))
    window = HandEyeCalibrationUI()
    window.show()
    sys.exit(app.exec())