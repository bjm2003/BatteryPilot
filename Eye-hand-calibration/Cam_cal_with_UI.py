import sys
import cv2
import numpy as np
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTextEdit, QProgressBar,
    QMessageBox, QScrollArea, QGroupBox, QDialog, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont

class CalibrationWorker(QThread):
    progress_updated = Signal(int)
    image_ready = Signal(str, np.ndarray)
    calibration_done = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, image_dir, pattern_size, square_size):
        super().__init__()
        self.image_dir = image_dir
        self.pattern_size = pattern_size
        self.square_size = square_size
        self._is_running = True

    def run(self):
        try:
            # 准备对象点 (带物理尺寸)
            objp = np.zeros((self.pattern_size[0]*self.pattern_size[1], 3), np.float32)
            xx, yy = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]]
            objp[:, :2] = np.vstack([xx.T.ravel(), yy.T.ravel()]).T * self.square_size

            # 收集标定数据
            objpoints = []
            imgpoints = []
            images = list(Path(self.image_dir).glob("*.jpg"))
            total = len(images)

            for idx, img_path in enumerate(images):
                if not self._is_running:
                    return

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                # 查找棋盘格角点
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
                
                if found:
                    # 亚像素优化
                    corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                    objpoints.append(objp)
                    imgpoints.append(corners_refined)

                    # 绘制结果并发送
                    cv2.drawChessboardCorners(img, self.pattern_size, corners_refined, found)
                    self.image_ready.emit(img_path.name, img)

                self.progress_updated.emit(int((idx+1)/total*100))

            # 执行相机标定
            if len(objpoints) > 3:
                # 移除CALIB_USE_INTRINSIC_GUESS标志
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, gray.shape[::-1], None, None
                )
                
                # 计算重投影误差
                total_error = 0
                per_image_errors = []
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(
                        objpoints[i], rvecs[i], tvecs[i], mtx, dist
                    )
                    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                    per_image_errors.append({
                        "image": Path(images[i]).name,
                        "error": round(float(error), 4)
                    })
                    total_error += error**2

                mean_error = np.sqrt(total_error/len(objpoints))

                # 准备结果数据
                result = {
                    "intrinsic": mtx.tolist(),
                    "distortion": dist.tolist(),
                    "extrinsics": [
                        {
                            "image": Path(images[i]).name,
                            "rotation": rvecs[i].tolist(),
                            "translation": tvecs[i].squeeze().tolist()
                        } for i in range(len(rvecs))
                    ],
                    "mean_error": round(float(mean_error), 4),
                    "per_image_errors": per_image_errors
                }
                self.calibration_done.emit(result)
            else:
                self.error_occurred.emit("有效标定图像不足(至少需要4张)")

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        self._is_running = False

class ResultDialog(QDialog):
    def __init__(self, result_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标定结果详情")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout()
        tab_widget = QTabWidget()
        
        # 内参标签页
        intrinsic_tab = QWidget()
        intrinsic_layout = QVBoxLayout()
        intrinsic_text = QTextEdit()
        intrinsic_text.setPlainText(
            "内参矩阵:\n" + 
            "\n".join(["\t".join(f"{x:.4f}" for x in row) for row in result_data["intrinsic"]]) + 
            "\n\n畸变系数:\n" + 
            "\t".join(f"{x:.6f}" for x in result_data["distortion"][0])
        )
        intrinsic_layout.addWidget(intrinsic_text)
        intrinsic_tab.setLayout(intrinsic_layout)
        
        # 外参标签页
        extrinsic_tab = QWidget()
        extrinsic_layout = QVBoxLayout()
        extrinsic_text = QTextEdit()
        content = []
        for ext in result_data["extrinsics"]:
            content.append(f"图像: {ext['image']}")
            content.append("旋转向量:")
            content.append("\n".join(["\t".join(f"{x:.6f}" for x in row) for row in ext["rotation"]]))
            content.append("平移向量(mm):")
            content.append("\t".join(f"{x:.2f}" for x in ext["translation"]))
            content.append("-"*50)
        extrinsic_text.setPlainText("\n".join(content))
        extrinsic_layout.addWidget(extrinsic_text)
        extrinsic_tab.setLayout(extrinsic_layout)
        
        # 误差分析标签页
        error_tab = QWidget()
        error_layout = QVBoxLayout()
        error_text = QTextEdit()
        error_content = [
            f"平均重投影误差: {result_data['mean_error']} 像素",
            "\n各图像详细误差:"
        ]
        for item in result_data["per_image_errors"]:
            error_content.append(f"{item['image']}: {item['error']} 像素")
        error_text.setPlainText("\n".join(error_content))
        error_layout.addWidget(error_text)
        error_tab.setLayout(error_layout)
        
        tab_widget.addTab(intrinsic_tab, "内参信息")
        tab_widget.addTab(extrinsic_tab, "外参信息")
        tab_widget.addTab(error_tab, "误差分析")
        
        layout.addWidget(tab_widget)
        self.setLayout(layout)

class CalibrationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("相机标定系统")
        self.setGeometry(100, 100, 600, 400)
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
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("选择标定图像目录")
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._select_directory)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(browse_btn)
        param_layout.addLayout(path_layout)

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

    def _select_directory(self):
        path = QFileDialog.getExistingDirectory(self, "选择标定图像目录")
        if path:
            self.path_input.setText(path)

    def _start_calibration(self):
        try:
            pattern = tuple(map(int, self.pattern_input.text().split('x')))
            square_size = float(self.size_input.text())
            image_dir = self.path_input.text()
            
            if not Path(image_dir).exists():
                raise FileNotFoundError

            self.worker = CalibrationWorker(image_dir, pattern, square_size)
            self.worker.image_ready.connect(self._show_preview)
            self.worker.progress_updated.connect(self.progress.setValue)
            self.worker.calibration_done.connect(self._show_results)
            self.worker.error_occurred.connect(self._show_error)
            self.worker.start()
            self.start_btn.setEnabled(False)

        except ValueError:
            QMessageBox.warning(self, "输入错误", "请检查参数格式\n正确示例：\n棋盘格：11x8\n尺寸：15")
        except FileNotFoundError:
            QMessageBox.warning(self, "路径错误", "指定的目录不存在")

    def _stop_calibration(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self.start_btn.setEnabled(True)

    def _show_preview(self, filename, img):
        preview = QDialog(self)
        preview.setWindowTitle(f"处理结果: {filename}")
        preview.setWindowModality(Qt.NonModal)
        
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)
        
        label = QLabel()
        label.setPixmap(pixmap.scaledToWidth(600))
        layout = QVBoxLayout()
        layout.addWidget(label)
        preview.setLayout(layout)
        
        preview.show()
        QTimer.singleShot(500, preview.close)

    def _show_results(self, result):
        self.start_btn.setEnabled(True)
        dialog = ResultDialog(result, self)
        dialog.exec()

    def _show_error(self, message):
        self.start_btn.setEnabled(True)
        QMessageBox.critical(self, "标定错误", message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("微软雅黑", 9))
    window = CalibrationApp()
    window.show()
    sys.exit(app.exec())