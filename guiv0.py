import sys
import os
import cv2
import csv
import time
import threading
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QTextEdit,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from ultralytics import YOLO


class VehicleEntryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Entry Detection")
        self.setGeometry(100, 100, 400, 400)

        self.video_path = ""
        self.video_paths = []
        self.log_folder = ""
        self.entry_line = []
        self.line_defined = False
        self.stop_flag = False

        self.model = YOLO("yolov8n.pt")

        # UI elements
        self.upload_btn = QPushButton("Upload Video")
        self.folder_btn = QPushButton("Select Log Folder")
        self.start_btn = QPushButton("Start Detection")
        self.stop_btn = QPushButton("Stop Detection")
        self.log_screen = QTextEdit()
        self.log_screen.setReadOnly(True)
        self.preview_label = QLabel()

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.upload_btn)
        layout.addWidget(self.folder_btn)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(QLabel("Logs:"))
        layout.addWidget(self.log_screen)
        # Remove live preview from main window
        self.setLayout(layout)

        # Signals
        self.upload_btn.clicked.connect(self.upload_video)
        self.folder_btn.clicked.connect(self.select_folder)
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.stop_btn.setEnabled(False)

    def upload_video(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video(s)", "", "Videos (*.mp4 *.avi)"
        )
        if paths:
            self.video_paths = paths
            self.log_screen.append(f"[INFO] {len(paths)} video(s) selected:")
            for i, path in enumerate(paths):
                self.log_screen.append(f"   {i+1}. {path}")

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Log Folder")
        if folder:
            self.log_folder = folder
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
            self.log_screen.append(f"[INFO] Log folder selected : {folder}")

    def start_detection(self):
        if not self.video_paths or not self.log_folder:
            QMessageBox.warning(
                self, "Missing Input", "Please select both a video and a log folder."
            )
            return

        cap = cv2.VideoCapture(self.video_paths[0])
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.log_screen.append("[ERROR] Failed to read frame for line drawing.")
            return

        self.log_screen.append("[INFO] Click two points to draw entry line...")
        p1, p2 = self.draw_line(frame)
        if not p1 or not p2:
            self.log_screen.append("[INFO] Entry line drawing cancelled.")
            return

        self.entry_line = [p1, p2]
        self.log_screen.append(f"[INFO] Entry line defined: {p1} to {p2}")

        self.stop_flag = False
        self.stop_btn.setEnabled(True)
        self.preview_window = QLabel()
        self.preview_window.setWindowTitle("Live Preview")
        self.preview_window.resize(800, 600)
        self.preview_window.show()
        threading.Thread(target=self.run_detection, daemon=True).start()

    def stop_detection(self):
        self.stop_flag = True
        self.log_screen.append("[INFO] Stopping detection...")

    def draw_line(self, frame):
        clone = frame.copy()
        points = []

        def draw(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) == 2:
                    cv2.line(clone, points[0], points[1], (0, 255, 0), 2)
                    cv2.imshow("Draw Entry Line", clone)

        cv2.namedWindow("Draw Entry Line")
        cv2.setMouseCallback("Draw Entry Line", draw)

        while True:
            cv2.imshow("Draw Entry Line", clone)
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and len(points) == 2:
                cv2.destroyWindow("Draw Entry Line")
                return points[0], points[1]
            elif key == 27:
                cv2.destroyWindow("Draw Entry Line")
                return None, None

    def run_detection(self):
        for video_index, video_path in enumerate(self.video_paths):
            self.log_screen.append(
                f"[INFO] Starting video {video_index + 1}: {os.path.basename(video_path)}"
            )
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.log_screen.append(f"[ERROR] Cannot open {video_path}")
                continue

            ret, frame = cap.read()
            if not ret:
                self.log_screen.append(f"[ERROR] Cannot read frame from {video_path}")
                cap.release()
                continue

            p1, p2 = self.entry_line

            video_name = os.path.splitext(os.path.basename(video_path))[0]
            image_dir = os.path.join(self.log_folder, f"{video_name}_folder")
            os.makedirs(image_dir, exist_ok=True)
            log_path = os.path.join(self.log_folder, f"{video_name}_log.csv")
            log_file = open(log_path, "w", newline="")
            csv_writer = csv.writer(log_file)
            csv_writer.writerow(["TrackID", "Timestamp", "ImagePath"])

            def crossed(prev, now, A, B):
                def ccw(X, Y, Z):
                    return (Z[1] - X[1]) * (Y[0] - X[0]) > (Y[1] - X[1]) * (Z[0] - X[0])

                return ccw(prev, A, B) != ccw(now, A, B) and ccw(prev, now, A) != ccw(
                    prev, now, B
                )

            track_memory = {}
            frame_count = 0
            entry_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if self.stop_flag:
                    self.log_screen.append(" ‼️ Detection stopped by user.")
                    break
                if not ret:
                    break

                frame_count += 1
                if frame_count % 5 != 0:
                    continue

                results = self.model.track(frame, persist=True, classes=[2, 3, 5, 7])
                if results[0].boxes.id is None:
                    continue
                else:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    ids = results[0].boxes.id.cpu().numpy().astype(int)
                    for box, obj_id in zip(boxes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"ID : {obj_id}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                        )

                for box, obj_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    prev = track_memory.get(obj_id, [(cx, cy), False])[0]
                    has_crossed = track_memory.get(obj_id, [(cx, cy), False])[1]

                    if (
                        not has_crossed
                        and crossed(prev, (cx, cy), p1, p2)
                        and cx < prev[0]
                    ):
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        image_path = os.path.join(
                            image_dir, f"{obj_id}_{timestamp}.jpg"
                        )
                        cv2.imwrite(image_path, frame[y1:y2, x1:x2])
                        csv_writer.writerow([obj_id, timestamp, image_path])
                        entry_count += 1
                        self.log_screen.append(
                            f" 📍 Vehicle {obj_id} entered at {timestamp}"
                        )

                    track_memory[obj_id] = [
                        (cx, cy),
                        has_crossed
                        or (cx < prev[0] and crossed(prev, (cx, cy), p1, p2)),
                    ]

                cv2.putText(
                    frame,
                    f"Total cars: {entry_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    800, 600, Qt.KeepAspectRatio
                )
                self.preview_window.setPixmap(pixmap)
                # self.log_screen.append(f"[INFO] Processed frame {frame_count}")

            cap.release()
        log_file.close()
        self.stop_btn.setEnabled(False)
        self.log_screen.append(
            f"[DONE] Processing complete. Total entries: {entry_count}"
        )
        self.preview_window.clear()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VehicleEntryApp()
    window.show()
    sys.exit(app.exec_())
