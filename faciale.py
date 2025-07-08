import cv2
import dlib
import PIL.Image
import numpy as np
from imutils import face_utils
import argparse
from pathlib import Path
import os
import ntpath
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from datetime import datetime, date
import json
import random
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Easy Facial Recognition App with Attendance')
parser.add_argument('-i', '--input', type=str, required=True, help='directory of input known faces')
parser.add_argument('-a', '--attendance', type=str, default='attendance.json', help='path to attendance file')

print('[INFO] Starting System...')
print('[INFO] Importing pretrained model..')
pose_predictor_68_point = dlib.shape_predictor("pretrained_model/shape_predictor_68_face_landmarks.dat")
pose_predictor_5_point = dlib.shape_predictor("pretrained_model/shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("pretrained_model/dlib_face_recognition_resnet_model_v1.dat")
face_detector = dlib.get_frontal_face_detector()
print('[INFO] Importing pretrained model done')

def transform(image, face_locations):
    coord_faces = []
    for face in face_locations:
        rect = face.top(), face.right(), face.bottom(), face.left()
        coord_face = max(rect[0], 0), min(rect[1], image.shape[1]), min(rect[2], image.shape[0]), max(rect[3], 0)
        coord_faces.append(coord_face)
    return coord_faces

def encode_face(image):
    # Convert to RGB if image is in BGR
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image

    face_locations = face_detector(image_rgb, 1)
    face_encodings_list = []
    landmarks_list = []
    
    for face_location in face_locations:
        # DETECT FACES
        shape = pose_predictor_68_point(image_rgb, face_location)
        # Compute face descriptor with padding
        face_encodings_list.append(np.array(face_encoder.compute_face_descriptor(image_rgb, shape, num_jitters=1, padding=0.25)))
        # GET LANDMARKS
        shape = face_utils.shape_to_np(shape)
        landmarks_list.append(shape)
    
    face_locations = transform(image_rgb, face_locations)
    return face_encodings_list, face_locations, landmarks_list


def easy_face_reco(frame, known_face_encodings, known_face_names, attendance_records, attendance_path):
    # Convert to RGB
    rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ENCODING FACE
    try:
        face_encodings_list, face_locations_list, landmarks_list = encode_face(rgb_small_frame)
    except Exception as e:
        print(f"[ERROR] Face encoding failed: {e}")
        return
    
    face_names = []
    for face_encoding in face_encodings_list:
        if len(face_encoding) == 0:
            continue
        
        # CHECK DISTANCE BETWEEN KNOWN FACES AND FACES DETECTED
        vectors = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
        tolerance = 0.6
        result = vectors <= tolerance
        
        if result.any():
            first_match_index = result.argmax()
            name = known_face_names[first_match_index]
            
            # Record attendance
            if record_attendance(name, attendance_records):
                save_attendance(attendance_records, attendance_path)
        else:
            name = "inconnu"
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations_list, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), (255, 0, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 2, bottom - 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    for shape in landmarks_list:
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


def process_image(file_):
    try:
        image = PIL.Image.open(file_)
        image = np.array(image)
        face_encodings_list, _, _ = encode_face(image)
        
        if face_encodings_list:  # Check if any faces were detected
            return face_encodings_list[0], file_.parent.name
        return None, None
    except Exception as e:
        print(f"[WARNING] Could not process {file_}: {e}")
        return None, None

def load_known_faces(face_to_encode_path):
    print('[INFO] Importing faces...')
    files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        files.extend(list(face_to_encode_path.rglob(ext)))

    if not files:
        raise ValueError(f'No faces detected in the directory: {face_to_encode_path}')

    # Use multiprocessing to speed up encoding
    known_face_encodings = []
    known_face_names = []

    # Limit the number of workers to prevent overwhelming the system
    max_workers = min(multiprocessing.cpu_count(), 8)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_image, file_): file_ for file_ in files}
        
        # Process results as they complete
        for future in as_completed(future_to_file):
            encoding, name = future.result()
            if encoding is not None:
                known_face_encodings.append(encoding)
                known_face_names.append(name)

    print(f'[INFO] Imported {len(known_face_encodings)} faces')
    print('[INFO] Faces well imported')
    
    return known_face_encodings, known_face_names

def add_new_person(dataset_path, pin_database_path='pin_database.json'):
    """
    Add a new person to the dataset and PIN database.
    
    Args:
        dataset_path (Path): Path to the dataset directory.
        pin_database_path (str): Path to the PIN database file.
    
    Returns:
        bool: True if the person was added successfully, False otherwise.
    """
    try:
        # Load existing PIN database
        pin_database = load_pin_database(pin_database_path)
        
        # Get new person's name
        person_name = input("Enter the name of the new person: ").strip()
        
        # Generate unique PIN
        new_pin = generate_unique_pin(list(pin_database.keys()))
        
        # Add PIN to database with full structure
        pin_database[person_name] = {
            "pin": new_pin,
            "attendance": {}
        }
        save_pin_database(pin_database, pin_database_path)
        
        # Create directory
        new_person_dir = os.path.join(dataset_path, person_name)
        os.makedirs(new_person_dir, exist_ok=True)
        
        # Display PIN to user
        print(f"Generated PIN for {person_name}: {new_pin}")
        
        # Capture and save images
        video_capture = cv2.VideoCapture(0)
        image_count = 0
        
        print(f"Taking photos for {person_name}. Press 's' to capture, 'q' to finish.")
        
        while image_count < 5:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame")
                video_capture.release()
                cv2.destroyAllWindows()
                return False
            
            cv2.putText(frame, f"Images captured: {image_count}/5", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Capture Images', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Save image
                image_path = os.path.join(new_person_dir, f"{person_name}_{image_count}.jpg")
                cv2.imwrite(image_path, frame)
                image_count += 1
                print(f"Captured image {image_count}")
            
            elif key == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()
        
        return image_count > 0
    
    except Exception as e:
        print(f"[ERROR] Failed to add person: {e}")
        return False
    
def load_attendance(attendance_path):
    """Load attendance records from JSON file."""
    if os.path.exists(attendance_path):
        with open(attendance_path, 'r') as f:
            return json.load(f)
    return {}

def save_attendance(attendance, attendance_path):
    """Save attendance records to JSON file."""
    with open(attendance_path, 'w') as f:
        json.dump(attendance, f, indent=4)

def record_attendance(name, attendance_records):
    """Record attendance for a recognized person."""
    today = date.today().isoformat()
    current_time = datetime.now().strftime("%H:%M:%S")
    
    if today not in attendance_records:
        attendance_records[today] = {}
    
    # Only record first attendance of the day
    if name not in attendance_records[today]:
        attendance_records[today][name] = current_time
        return True
    return False

def display_attendance(attendance_records):
    """Display attendance records in a formatted way."""
    print("\n--- ATTENDANCE REPORT ---")
    for date, entries in sorted(attendance_records.items()):
        print(f"\nDate: {date}")
        for name, time in entries.items():
            print(f"{name}: {time}")
    print("------------------------")

def generate_unique_pin(existing_pins):
    """Generate a unique 4-digit PIN."""
    while True:
        pin = str(random.randint(1000, 9999))
        if pin not in existing_pins:
            return pin

def load_pin_database(pin_database_path):
    try:
        with open(pin_database_path, 'r') as f:
            # Read the entire file content and parse it as a single JSON object
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        # If parsing fails, try to manually merge the JSON objects
        pin_database = {}
        with open(pin_database_path, 'r') as f:
            for line in f:
                try:
                    line_data = json.loads(line)
                    pin_database.update(line_data)
                except json.JSONDecodeError:
                    print(f"[WARNING] Could not parse line: {line}")
        return pin_database

def save_pin_database(pin_database, pin_database_path):
    with open(pin_database_path, 'w') as f:
        json.dump(pin_database, f, indent=4)

def manual_attendance_registration(name, pin_database_path='pin_database.json', attendance_path='attendance.json'):
    # Load PIN database
    try:
        with open(pin_database_path, 'r') as f:
            # Read the entire file as a single JSON object
            pin_database = json.load(f)
        
        # Check if name exists in database
        if name not in pin_database:
            print(f"[ERROR] {name} not found in database")
            return False
        
        # Ask for PIN verification
        correct_pin = pin_database[name]['pin']
        entered_pin = input(f"Enter PIN for {name}: ")
        
        # Verify PIN
        if entered_pin != correct_pin:
            print("[ERROR] Incorrect PIN")
            return False
        
        # Record attendance
        today = date.today().isoformat()
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Load existing attendance
        try:
            with open(attendance_path, 'r') as f:
                attendance_records = json.load(f)
        except FileNotFoundError:
            attendance_records = {}

        # Add or update attendance
        if today not in attendance_records:
            attendance_records[today] = {}
        
        # Only record if not already recorded today
        if name not in attendance_records[today]:
            attendance_records[today][name] = current_time
            
            # Save updated attendance
            with open(attendance_path, 'w') as f:
                json.dump(attendance_records, f, indent=4)
            
            print(f"[SUCCESS] Attendance recorded for {name}")
            return True
        else:
            print(f"[INFO] Attendance already recorded for {name} today")
            return False
    
    except json.JSONDecodeError:
        # Fallback method to parse line by line if loading fails
        pin_database = {}
        with open(pin_database_path, 'r') as f:
            for line in f:
                try:
                    line_data = json.loads(line)
                    pin_database.update(line_data)
                except json.JSONDecodeError:
                    print(f"[WARNING] Could not parse line: {line}")
        
        # Retry the entire process with parsed database
        if name in pin_database:
            correct_pin = pin_database[name]['pin']
            # Rest of the function remains the same...
    
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        return False

def prepare_face_data(dataset_path):
    """
    Prepare face encodings and labels from dataset
    
    Args:
        dataset_path (Path): Path to dataset
    
    Returns:
        tuple: Face encodings and corresponding labels
    """
    face_data = []
    labels = []
    
    for person_dir in dataset_path.iterdir():
        if not person_dir.is_dir():
            continue
        
        images = list(person_dir.glob('*.jpg')) + \
                 list(person_dir.glob('*.png')) + \
                 list(person_dir.glob('*.jpeg'))
        
        for image_path in images:
            image = cv2.imread(str(image_path))
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_detector(image_rgb, 1)
                
                for face_location in face_locations:
                    shape = pose_predictor_68_point(image_rgb, face_location)
                    encoding = np.array(face_encoder.compute_face_descriptor(
                        image_rgb, shape, num_jitters=1, padding=0.25
                    ))
                    face_data.append(encoding)
                    labels.append(person_dir.name)
    
    return face_data, labels

def advanced_performance_evaluation(dataset_path, tolerances=[0.5, 0.6, 0.7], num_jitters_list=[1, 2, 3]):
    """
    Advanced performance evaluation with comprehensive metrics
    """
    import warnings
    from sklearn.exceptions import UndefinedMetricWarning
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

    # Prepare face data
    face_data, labels = prepare_face_data(dataset_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        face_data, labels, test_size=0.2, random_state=42
    )
    
    performance_results = {
        'global_summary': [],
        'class_metrics': {},
        'configurations': []
    }
    
    # Test different configurations
    for tolerance in tolerances:
        for num_jitters in num_jitters_list:
            # Encode training faces
            known_face_encodings = [encoding for encoding in X_train]
            known_face_names = [name for name in y_train]
            
            # Predict test faces
            predicted_names = []
            for test_encoding in X_test:
                vectors = np.linalg.norm(known_face_encodings - test_encoding, axis=1)
                result = vectors <= tolerance
                
                if result.any():
                    first_match_index = result.argmax()
                    predicted_names.append(known_face_names[first_match_index])
                else:
                    predicted_names.append("Unknown")
            
            # Calculate comprehensive metrics
            report = classification_report(y_test, predicted_names, output_dict=True)
            
            # Store configuration results
            config_result = {
                'tolerance': tolerance,
                'num_jitters': num_jitters,
                'accuracy': report['accuracy'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['f1-score']
            }
            performance_results['configurations'].append(config_result)
            
            # Collect global summary
            performance_results['global_summary'].append({
                'Tolerance': tolerance,
                'Num Jitters': num_jitters,
                'Accuracy': f"{report['accuracy']:.2%}",
                'Macro Precision': f"{report['macro avg']['precision']:.2%}",
                'Macro Recall': f"{report['macro avg']['recall']:.2%}"
            })
            
            # Collect class-level metrics
            performance_results['class_metrics'][f'T{tolerance}_J{num_jitters}'] = {
                name: {
                    'Precision': f"{report[name]['precision']:.2%}",
                    'Recall': f"{report[name]['recall']:.2%}",
                    'F1-Score': f"{report[name]['f1-score']:.2%}",
                    'Support': report[name]['support']
                } for name in report.keys() if name not in ['accuracy', 'macro avg', 'weighted avg']
            }
    
    return performance_results

def visualize_performance(performance_metrics, X_test, y_test):
    plt.figure(figsize=(20, 15))
    
    # Global Performance Summary
    plt.subplot(2, 2, 1)
    summary_df = pd.DataFrame(performance_metrics['global_summary'])
    plt.title('Performance Summary')
    plt.axis('off')
    plt.table(cellText=summary_df.values, 
              colLabels=summary_df.columns, 
              loc='center', 
              cellLoc='center')
    
    # Accuracy Comparison
    plt.subplot(2, 2, 2)
    accuracies = [config['accuracy'] for config in performance_metrics['configurations']]
    configs = [f"T{config['tolerance']}/J{config['num_jitters']}" for config in performance_metrics['configurations']]
    plt.bar(configs, accuracies)
    plt.title('Accuracy by Configuration')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Confusion Matrix 
    plt.subplot(2, 2, 3)
    # Use the first configuration for demonstration
    tolerance = performance_metrics['configurations'][0]['tolerance']
    num_jitters = performance_metrics['configurations'][0]['num_jitters']
    
    # Recreate prediction for this specific configuration
    known_face_encodings = [encoding for encoding in X_test]
    known_face_names = [name for name in y_test]
    
    predicted_names = []
    for test_encoding in X_test:
        vectors = np.linalg.norm(known_face_encodings - test_encoding, axis=1)
        result = vectors <= tolerance
        
        if result.any():
            first_match_index = result.argmax()
            predicted_names.append(known_face_names[first_match_index])
        else:
            predicted_names.append("Unknown")
    
    # Create confusion matrix
    conf_matrix = confusion_matrix(y_test, predicted_names)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (T:{tolerance}, J:{num_jitters})')
    
    # Performance Recommendations
    plt.subplot(2, 2, 4)
    plt.title('Performance Recommendations')
    plt.axis('off')
    recommendations = [
        "Increase training data if accuracy < 80%",
        "Adjust face matching tolerance",
        "Consider more face augmentation techniques",
        "Validate face recognition in varied lighting"
    ]
    plt.text(0.1, 0.5, '\n'.join(recommendations), fontsize=10, 
             verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('performance_comprehensive_report.png')
    plt.close()

def update_main_performance_evaluation(face_to_encode_path):
    print("[INFO] Performing Advanced Performance Evaluation...")
    
    # Prepare face data
    face_data, labels = prepare_face_data(face_to_encode_path)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        face_data, labels, test_size=0.2, random_state=42
    )
    
    performance_metrics = advanced_performance_evaluation(face_to_encode_path)
    
    # Print simplified performance summary
    print("\n--- PERFORMANCE SUMMARY ---")
    
    # Get the top-performing configuration
    top_config = performance_metrics['configurations'][0]
    
    print(f"Tolerance: {top_config['tolerance']}, Jitters: {top_config['num_jitters']}")
    print(f"Accuracy: {top_config['accuracy']:.2%}")
    
    print("\nPerformance Recommendations:")
    print("Increase training data if accuracy < 80%")
    
    # Visualize performance with test data
    visualize_performance(performance_metrics, X_test, y_test)
    print("Performance report saved as 'performance_comprehensive_report.png'")


def main(face_to_encode_path, attendance_path):
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces(face_to_encode_path)
    
    # Load existing attendance records
    attendance_records = load_attendance(attendance_path)
    
    # Use webcam instead of video file
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("[ERROR] Could not open webcam")
        exit()

    print('[INFO] Detecting...')
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break
        
        easy_face_reco(frame, known_face_encodings, known_face_names, attendance_records, attendance_path)
        
        # Add text instructions on the frame
        cv2.putText(frame, "Press 'q' to quit, 'a' to add new person, 'r' to view report, 'e' to register attendance, 'p' to evaluate performance", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Easy Facial Recognition App', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Add new person logic
            video_capture.release()
            if add_new_person(face_to_encode_path):
                known_face_encodings, known_face_names = load_known_faces(face_to_encode_path)
                cv2.destroyAllWindows()
                video_capture = cv2.VideoCapture(0)
        elif key == ord('r'):
            # Display attendance report
            display_attendance(attendance_records)
        
        elif key == ord('e'):
            # Manually register attendance for recognized faces
            current_faces = list(set([name for name in known_face_names if name != "inconnu"]))
            
            if not current_faces:
                print("[INFO] No recognized faces in the frame")
                continue
            
            print("Recognized faces:", current_faces)
            
            # Prompt user to select a face for attendance
            name = input("Enter name to register attendance: ").strip()
            
            if name in current_faces:
                # Use the pin_database_path here
                pin_database_path = os.path.join(os.path.dirname(__file__), 'pin_database.json')
                manual_attendance_registration(name, pin_database_path=pin_database_path)
            else:
                print("[ERROR] Selected name not in recognized faces")

        elif key == ord('p'):
            update_main_performance_evaluation(face_to_encode_path)


    print('[INFO] Stopping System')
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = parser.parse_args()
    face_to_encode_path = Path(args.input)
    attendance_path = args.attendance
    main(face_to_encode_path, attendance_path)

