import sys
import cv2
import numpy as np
import scipy.linalg
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox, QGroupBox, QScrollArea, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

class CalibrationThread(QThread):
    progress_updated = Signal(int)
    result_ready = Signal(np.ndarray)
    error_occurred = Signal(str)

    def __init__(self, image_dir, pose_file, pattern_size, square_size):
        super().__init__()
        self.image_dir = image_dir
        self.pose_file = pose_file
        self.pattern_size = pattern_size
        self.square_size = square_size

    def run(self):
        try:
            # 读取机械臂位姿
            robot_poses = self._read_robot_poses()
            
            # 执行相机标定
            K, extrinsics = self._camera_calibration()
            
            # 手眼标定
            X = self._hand_eye_calibration(extrinsics, robot_poses)
            
            self.result_ready.emit(X)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _read_robot_poses(self):
        with open(self.pose_file, 'r') as f:
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

    def _camera_calibration(self):
        objp = np.zeros((self.pattern_size[0]*self.pattern_size[1], 3), np.float32)
        xx, yy = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]]
        objp[:, :2] = np.vstack([xx.T.ravel(), yy.T.ravel()]).T * self.square_size

        objpoints = []
        imgpoints = []
        images = list(Path(self.image_dir).glob("*.jpg"))
        total = len(images)

        for idx, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
            
            self.progress_updated.emit(int((idx+1)/total*100))

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        extrinsics = []
        for rvec, tvec in zip(rvecs, tvecs):
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
            A = np.hstack((R, t))
            A = np.vstack((A, [0, 0, 0, 1]))
            extrinsics.append(A)
        
        return K, extrinsics

    def _hand_eye_calibration(self, extrinsics, robot_poses):
        C_list, D_list = [], []
        for i in range(1, len(extrinsics)):
            A_prev = extrinsics[i-1]
            A_current = extrinsics[i]
            C = A_current @ np.linalg.inv(A_prev)
            C_list.append(C)
            
            B_prev = robot_poses[i-1]
            B_current = robot_poses[i]
            D = B_current @ np.linalg.inv(B_prev)
            D_list.append(D)

        # 求解旋转部分
        kron_blocks = []
        for C, D in zip(C_list, D_list):
            kron = np.kron(C[:3, :3], np.eye(3)) - np.kron(np.eye(3), D[:3, :3].T)
            kron_blocks.append(kron)
        
        U, S, Vh = np.linalg.svd(np.vstack(kron_blocks))
        Rx = Vh[-1, :].reshape(3, 3)
        Rx = scipy.linalg.orth(Rx)
        if np.linalg.det(Rx) < 0:
            Rx[:, -1] *= -1

        # 求解平移部分
        A_blocks, b_blocks = [], []
        for C, D in zip(C_list, D_list):
            A_block = C[:3, :3] - Rx @ D[:3, :3]
            b_block = Rx @ D[:3, 3].reshape(3,1) - C[:3, 3].reshape(3,1)
            A_blocks.append(A_block)
            b_blocks.append(b_block)
        
        t_x = np.linalg.lstsq(np.vstack(A_blocks), np.vstack(b_blocks), rcond=None)[0]
        
        X = np.hstack((Rx, t_x))
        X = np.vstack((X, [0, 0, 0, 1]))
        return X

class HandEyeCalibrationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手眼标定系统")
        self.setGeometry(100, 100, 600, 450)
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
        pattern_layout = QHBoxLayout()
        self.pattern_label = QLabel("棋盘格尺寸 (列×行):")
        self.pattern_label.setStyleSheet("color: #2C3E50;")
        self.pattern_input = QLineEdit("11x8")
        pattern_layout.addWidget(self.pattern_label)
        pattern_layout.addWidget(self.pattern_input)
        param_layout.addLayout(pattern_layout)

        # 物理尺寸
        size_layout = QHBoxLayout()
        self.size_label = QLabel("单格尺寸 (mm):")
        self.size_label.setStyleSheet("color: #2C3E50;")
        self.size_input = QLineEdit("15")
        size_layout.addWidget(self.size_label)
        size_layout.addWidget(self.size_input)
        param_layout.addLayout(size_layout)

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
            }
            QGroupBox {
                font: bold 12px '微软雅黑';
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
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "文本文件 (*.txt)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择目录")
        if path:
            input_field.setText(path)

    def _start_calibration(self):
        try:
            pattern = tuple(map(int, self.pattern_input.text().split('x')))
            square_size = float(self.size_input.text())
            image_dir = self.image_path_input.text()
            pose_file = self.pose_path_input.text()
            
            if not Path(image_dir).exists():
                raise FileNotFoundError("图像目录不存在")
            if not Path(pose_file).exists():
                raise FileNotFoundError("位姿文件不存在")

            self.worker = CalibrationThread(image_dir, pose_file, pattern, square_size)
            self.worker.progress_updated.connect(self.progress.setValue)
            self.worker.result_ready.connect(self._show_result)
            self.worker.error_occurred.connect(self._show_error)
            self.worker.start()
            self.start_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(self, "输入错误", str(e))

    def _stop_calibration(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
        self.start_btn.setEnabled(True)

    def _show_result(self, X):
        self.result_text.clear()
        self.result_text.append("手眼标定矩阵 (机械臂末端 → 相机坐标系):\n")
        for row in X:
            self.result_text.append("\t".join(f"{x:.6f}" for x in row))
        self.start_btn.setEnabled(True)

    def _show_error(self, message):
        QMessageBox.critical(self, "标定错误", message)
        self.start_btn.setEnabled(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("微软雅黑", 9))
    window = HandEyeCalibrationUI()
    window.show()
    sys.exit(app.exec())