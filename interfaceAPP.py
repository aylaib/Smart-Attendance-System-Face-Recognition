import sys
import os
import cv2
import json
from pathlib import Path
import numpy as np
from datetime import date, datetime
from PyQt5.QtWidgets import QInputDialog, QDialog
from PyQt5.QtGui import QPalette, QColor

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTextEdit, QLineEdit, QMessageBox, QStackedWidget
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer

# Import your existing facial recognition functions
from faciale import (
    easy_face_reco, load_known_faces, add_new_person,load_pin_database,generate_unique_pin,
    manual_attendance_registration, display_attendance, save_pin_database,
    update_main_performance_evaluation, load_attendance
)
class CaptureWindow(QDialog):  # Changed from QWidget to QDialog
    def __init__(self, person_name, person_dir):
        super().__init__()
        self.person_name = person_name
        self.person_dir = person_dir
        self.image_count = 0
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Capture Photos")
        self.setModal(True)  # Make window modal
        layout = QVBoxLayout()
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        layout.addWidget(self.camera_label)
        
        # Capture button
        self.capture_btn = QPushButton("Capture Photo (or press 'S')")
        self.capture_btn.clicked.connect(self.capture_photo)
        layout.addWidget(self.capture_btn)
        
        # Status label
        self.status_label = QLabel(f"Images captured: {self.image_count}/5")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Setup camera
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
    def keyPressEvent(self, event):
        if event.text().lower() == 's':
            self.capture_photo()
            
    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Add counter to frame
            cv2.putText(frame, f"Images captured: {self.image_count}/5", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert frame for display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
            self.camera_label.setPixmap(scaled_pixmap)
            
    def capture_photo(self):
        ret, frame = self.capture.read()
        if ret and self.image_count < 5:
            # Save image
            image_path = os.path.join(self.person_dir, f"{self.person_name}_{self.image_count}.jpg")
            cv2.imwrite(image_path, frame)
            self.image_count += 1
            self.status_label.setText(f"Images captured: {self.image_count}/5")
            
            if self.image_count >= 5:
                self.timer.stop()
                self.capture.release()
                self.accept()  # Close dialog with accept status
                
    def closeEvent(self, event):
        self.timer.stop()
        self.capture.release()
        event.accept()

class FacialRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Attendance System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QLabel {
                color: white;
                font-size: 14px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 10px;
                padding: 10px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
            }
        """)
        
        # Central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left side layout with gradient background
        left_side = QWidget()
        left_side.setObjectName("leftPanel")
        left_side.setStyleSheet("""
            QWidget#leftPanel {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2c3e50, stop:1 #3498db);
                border-radius: 15px;
                padding: 20px;
            }
        """)
        
        left_layout = QVBoxLayout(left_side)
        left_layout.setSpacing(15)
        
        # Camera display area with border
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setStyleSheet("""
            QLabel {
                background-color: #1a1a1a;
                border: 2px solid #3498db;
                border-radius: 10px;
            }
        """)
        left_layout.addWidget(self.camera_label)
        
        # Button container
        button_container = QWidget()
        button_layout = QVBoxLayout(button_container)
        button_layout.setSpacing(10)
        
        # Define buttons with their properties
        buttons_data = [
            ("Add Person", "#2ecc71", self.show_add_person),
            ("View Report", "#e74c3c", self.show_attendance_report),
            ("Register Attendance", "#f1c40f", self.show_attendance_registration),
            ("Performance", "#9b59b6", self.show_performance)
        ]
        
        for text, color, callback in buttons_data:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                    color: white;
                    border: none;
                    padding: 15px;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 8px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: {color}dd;
                }}
                QPushButton:pressed {{
                    background-color: {color}aa;
                }}
            """)
            btn.clicked.connect(callback)  # Connect the button to its callback
            button_layout.addWidget(btn)
        
        left_layout.addWidget(button_container)
        
        # Right side - display area
        right_side = QWidget()
        right_side.setObjectName("rightPanel")
        right_side.setStyleSheet("""
            QWidget#rightPanel {
                background-color: #2d2d2d;
                border-radius: 15px;
                padding: 20px;
            }
        """)
        
        right_layout = QVBoxLayout(right_side)
        
        # Add title for display area
        display_title = QLabel("Activity Log")
        display_title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
            }
        """)
        right_layout.addWidget(display_title)
        
        # Enhanced text display
        self.display_area = QTextEdit()
        self.display_area.setReadOnly(True)
        self.display_area.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #ecf0f1;
                border: 1px solid #3498db;
                border-radius: 10px;
                padding: 15px;
                font-family: 'Consolas', monospace;
                font-size: 13px;
                line-height: 1.6;
            }
            QScrollBar:vertical {
                border: none;
                background: #2d2d2d;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #3498db;
                border-radius: 5px;
            }
        """)
        right_layout.addWidget(self.display_area)
        
        # Add layouts to main layout
        main_layout.addWidget(left_side, stretch=60)
        main_layout.addWidget(right_side, stretch=40)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Initialize camera
        self.capture = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        
        # Load known faces
        self.dataset_path = Path("Dataset")
        self.known_face_encodings, self.known_face_names = load_known_faces(self.dataset_path)
        
        # Attendance tracking
        self.attendance_path = "attendance.json"
        self.attendance_records = load_attendance(self.attendance_path)
    
    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Perform face recognition
            easy_face_reco(
                frame, 
                self.known_face_encodings, 
                self.known_face_names, 
                self.attendance_records, 
                self.attendance_path
            )
            
            # Convert frame to QImage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Scale pixmap to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.camera_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            QApplication.processEvents()  # Keep UI responsive during updates
            self.camera_label.setPixmap(scaled_pixmap)

    def add_new_person(self):
        name, ok = QInputDialog.getText(self, "Add New Person", "Enter name:")
        if ok and name:
            try:
                # Temporarily release camera
                self.timer.stop()
                self.capture.release()
                
                # Add person (uses existing function from faciale.py)
                if add_new_person(self.dataset_path):
                    # Reload known faces
                    self.known_face_encodings, self.known_face_names = load_known_faces(self.dataset_path)
                    
                    # Display success message
                    self.display_area.setText(f"Person '{name}' added successfully!")
                    
                    # Refresh attendance records
                    self.attendance_records = load_attendance(self.attendance_path)
                    
                    QMessageBox.information(self, "Success", f"Person {name} added successfully!")
                else:
                    QMessageBox.critical(self, "Error", "Failed to add person. Please try again.")
            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
            finally:
                # Restart camera
                self.capture = cv2.VideoCapture(0)
                self.timer.start(30)

    def show_add_person(self):
        name, ok = QInputDialog.getText(self, "Add New Person", "Enter name:")
        if ok and name:
            try:
                # Update display area with status
                self.display_area.clear()
                self.display_area.append(f"Adding new person: {name}")
                self.display_area.append("Please wait...")
                
                # Create directory for the new person
                person_dir = os.path.join(self.dataset_path, name)
                os.makedirs(person_dir, exist_ok=True)
                
                # Generate PIN
                pin_database = load_pin_database("pin_database.json")
                new_pin = generate_unique_pin(list(pin_database.keys()))
                
                # Update PIN database
                pin_database[name] = {
                    "pin": new_pin,
                    "attendance": {}
                }
                save_pin_database(pin_database, "pin_database.json")
                
                # Update display with PIN
                self.display_area.append(f"\nGenerated PIN for {name}: {new_pin}")
                
                # Temporarily stop the main camera feed
                self.timer.stop()
                self.capture.release()
                
                # Create and show capture window
                capture_window = CaptureWindow(name, person_dir)
                if capture_window.exec_() == QDialog.Accepted:  # Use exec_() for modal dialog
                    # Reload known faces only if photos were captured successfully
                    self.known_face_encodings, self.known_face_names = load_known_faces(self.dataset_path)
                    self.display_area.append(f"\nPerson '{name}' added successfully!")
                    QMessageBox.information(self, "Success", f"Person {name} added successfully!")
                else:
                    self.display_area.append("\nCapture cancelled or incomplete")
                
                # Restart main camera feed
                self.capture = cv2.VideoCapture(0)
                self.timer.start(30)
                
            except Exception as e:
                self.display_area.append(f"\nError: {str(e)}")
                QMessageBox.critical(self, "Error", str(e))
                
                # Ensure camera is restarted even if there's an error
                self.capture = cv2.VideoCapture(0)
                self.timer.start(30)

        
    def show_attendance_report(self):
        report = ""
        for date_key, entries in sorted(self.attendance_records.items()):
            report += f"\nDate: {date_key}\n"
            for name, time in entries.items():
                report += f"{name}: {time}\n"
        
        self.display_area.setText(report)
    
    def show_attendance_registration(self):
        try:
            # Get list of currently recognized faces from the video feed
            current_faces = list(set([name for name in self.known_face_names if name != "inconnu"]))
            
            # Show recognized faces in display area
            self.display_area.clear()
            self.display_area.append("Recognized faces in frame:")
            self.display_area.append(", ".join(current_faces))
            self.display_area.append("\n")
            
            # Get name input
            name, ok = QInputDialog.getText(self, "Register Attendance", "Enter name:")
            if not ok or not name:
                return
                
            if name not in current_faces:
                self.display_area.append(f"[ERROR] {name} not in recognized faces")
                return
                
            # Get PIN input
            pin, ok = QInputDialog.getText(self, "PIN Required", "Enter PIN:", QLineEdit.Password)
            if not ok or not pin:
                return
                
            # Load PIN database
            pin_database_path = "pin_database.json"
            with open(pin_database_path, 'r') as f:
                pin_database = json.load(f)
                
            # Verify PIN
            if name not in pin_database:
                self.display_area.append(f"[ERROR] {name} not found in database")
                return
                
            correct_pin = pin_database[name]['pin']
            if pin != correct_pin:
                self.display_area.append("[ERROR] Incorrect PIN")
                return
                
            # Record attendance
            today = date.today().isoformat()
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Check if already recorded today
            if today in self.attendance_records and name in self.attendance_records[today]:
                self.display_area.append(f"[INFO] Attendance already recorded for {name} today")
                return
                
            # Record new attendance
            if today not in self.attendance_records:
                self.attendance_records[today] = {}
            self.attendance_records[today][name] = current_time
            
            # Save to file
            with open(self.attendance_path, 'w') as f:
                json.dump(self.attendance_records, f, indent=4)
                
            self.display_area.append(f"[SUCCESS] Attendance recorded for {name}")
            
            # Show updated attendance report
            self.display_area.append("\n--- ATTENDANCE REPORT ---")
            for date_key, entries in sorted(self.attendance_records.items()):
                self.display_area.append(f"\nDate: {date_key}")
                for person, time in entries.items():
                    self.display_area.append(f"{person}: {time}")
            self.display_area.append("------------------------")
            
        except Exception as e:
            self.display_area.append(f"[ERROR] An error occurred: {str(e)}")
    
    def show_performance(self):
        try:
            # Temporarily capture stdout
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            # Run performance evaluation
            update_main_performance_evaluation(self.dataset_path)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Get captured output
            performance_text = captured_output.getvalue()
            
            # Display in text area
            self.display_area.setText(performance_text)
            
            # Show performance image if generated
            if os.path.exists('performance_comprehensive_report.png'):
                QMessageBox.information(
                    self, 
                    "Performance Report", 
                    "Performance report image has been saved as 'performance_comprehensive_report.png'"
                )
        except Exception as e:
            QMessageBox.critical(self, "Performance Error", str(e))
    
    def closeEvent(self, event):
        # Release camera and close application
        self.timer.stop()
        self.capture.release()
        cv2.destroyAllWindows()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern, flat style
    
    # Dark fusion style
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(dark_palette)
    
    # Custom font
    app.setFont(QFont('Segoe UI', 10))
    
    window = FacialRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()