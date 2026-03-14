"""
Smart CCTV Surveillance System with Women Safety Features
Hackathon Prototype - Violence & Harassment Detection System
"""

import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import urllib.request
from datetime import datetime
import threading
import queue

# Audio detection imports (optional if not available)
try:
    import speech_recognition as sr
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠ SpeechRecognition not available. Audio detection disabled.")

class SmartCCTVSystem:
    def __init__(self, model_path='yolov8n.pt', video_source=0, 
                 camera_name="Camera_02", location="MG Road"):
        """
        Initialize Smart CCTV Surveillance System
        """
        print("=" * 60)
        print("🔴 INITIALIZING SMART CCTV SURVEILLANCE SYSTEM")
        print("=" * 60)
        
        # Camera metadata
        self.camera_name = camera_name
        self.location = location
        
        # Download Pose Model if needed
        self.pose_model_path = 'pose_landmarker_lite.task'
        if not os.path.exists(self.pose_model_path):
            print(f"📥 Downloading Pose Model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            try:
                urllib.request.urlretrieve(url, self.pose_model_path)
                print("✅ Download complete.")
            except Exception as e:
                print(f"❌ Failed to download pose model: {e}")
                raise e

        # Initialize YOLOv8
        try:
            self.model = YOLO(model_path)
            print(f"✅ YOLOv8 Model loaded.")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
            raise e

        # Initialize MediaPipe Pose Landmarker
        print("🔄 Initializing MediaPipe Pose Landmarker...")
        base_options = python.BaseOptions(model_asset_path=self.pose_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            num_poses=1
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        print("✅ Pose detection ready.")

        # Video Capture
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            print(f"❌ Error: Could not open video source {video_source}")
            raise ValueError("Could not open video source")

        # State Variables
        self.previous_frame_landmarks = {}
        self.fight_detected = False
        self.fight_counter = 0
        self.normal_counter = 0
        
        # Women Safety States
        self.female_detected = False
        self.distress_gesture_detected = False
        self.harassment_detected = False
        self.audio_distress_detected = False
        
        # Detection Parameters
        self.FIGHT_DISTANCE_THRESHOLD = 150
        self.MOVEMENT_SPEED_THRESHOLD = 30
        self.CONFIDENCE_THRESHOLD = 7
        self.DISTRESS_GESTURE_THRESHOLD = 5
        self.distress_gesture_counter = 0
        
        # Incident Logging
        self.incident_log_file = "incident_log.txt"
        self.last_incident_time = 0
        self.incident_cooldown = 10  # seconds between same incident logs
        
        # Audio Detection Setup
        self.audio_queue = queue.Queue()
        self.audio_thread = None
        if AUDIO_AVAILABLE:
            self.setup_audio_detection()
        
        print("=" * 60)
        print("🟢 SYSTEM READY - Monitoring Started")
        print(f"📍 Location: {self.location}")
        print(f"📹 Camera: {self.camera_name}")
        print("=" * 60)

    def setup_audio_detection(self):
        """Initialize background audio detection thread"""
        def audio_listener():
            recognizer = sr.Recognizer()
            mic = sr.Microphone()
            
            # Distress keywords
            distress_keywords = ["help", "bachao", "save me", "stop", "mujhe bachao"]
            
            print("🎤 Audio monitoring started...")
            
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                
                while True:
                    try:
                        audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
                        text = recognizer.recognize_google(audio).lower()
                        
                        # Check for distress keywords
                        for keyword in distress_keywords:
                            if keyword in text:
                                self.audio_queue.put(("distress", text))
                                break
                                
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except Exception as e:
                        continue
        
        self.audio_thread = threading.Thread(target=audio_listener, daemon=True)
        self.audio_thread.start()

    def process_live_feed(self):
        """Main processing loop"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📺 Video Resolution: {width}x{height}")
        print("⌨️  Press 'ESC' to exit.")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Process audio queue
            self.check_audio_alerts()

            # Data container
            current_frame_data = {
                'people_count': 0,
                'people_boxes': [],
                'people_landmarks': {},
                'female_boxes': []
            }

            # Detect People using YOLOv8
            results = self.model.track(frame, persist=True, classes=0, verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                current_frame_data['people_count'] = len(track_ids)
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    current_frame_data['people_boxes'].append((x1, y1, x2, y2, track_id))
                    
                    # Simulate female detection (for demo: every 3rd person)
                    is_female = (track_id % 3 == 0)
                    if is_female:
                        current_frame_data['female_boxes'].append((x1, y1, x2, y2, track_id))
                    
                    # Draw bounding box
                    color = (255, 0, 255) if is_female else (0, 255, 0)
                    if self.fight_detected or self.harassment_detected:
                        color = (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} {'[F]' if is_female else ''}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update female detection state
            self.female_detected = len(current_frame_data['female_boxes']) > 0

            # Pose Detection (if 2+ people or female detected)
            if current_frame_data['people_count'] >= 2 or self.female_detected:
                self.process_poses(frame, current_frame_data)

            # Violence Detection
            self.detect_fight(current_frame_data)
            
            # Women Safety Detection
            self.detect_harassment(current_frame_data)
            self.detect_distress_gesture(current_frame_data)

            # Display CCTV UI
            self.display_cctv_ui(frame, current_frame_data)

            # Show Frame
            cv2.imshow("Smart CCTV Surveillance System", frame)
            
            # Update state
            self.previous_frame_landmarks = current_frame_data['people_landmarks']

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("🔴 SYSTEM SHUTDOWN")
        print("=" * 60)

    def process_poses(self, frame, frame_data):
        """Process pose detection for all people"""
        for (x1, y1, x2, y2, track_id) in frame_data['people_boxes']:
            h, w, _ = frame.shape
            padding = 20
            roi_x1 = max(0, x1 - padding)
            roi_y1 = max(0, y1 - padding)
            roi_x2 = min(w, x2 + padding)
            roi_y2 = min(h, y2 + padding)
            
            person_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            
            if person_roi.size != 0:
                roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
                detection_result = self.detector.detect(mp_image)
                
                if detection_result.pose_landmarks:
                    pose_landmarks = detection_result.pose_landmarks[0]
                    global_landmarks = []
                    roi_w = roi_x2 - roi_x1
                    roi_h = roi_y2 - roi_y1
                    
                    for lm in pose_landmarks:
                        global_x = int(lm.x * roi_w) + roi_x1
                        global_y = int(lm.y * roi_h) + roi_y1
                        global_landmarks.append((global_x, global_y))
                    
                    frame_data['people_landmarks'][track_id] = global_landmarks
                    self.draw_pose_skeleton(frame, global_landmarks)

    def draw_pose_skeleton(self, frame, landmarks):
        """Draw pose skeleton"""
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
        ]
        
        for x, y in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                cv2.line(frame, landmarks[start_idx], landmarks[end_idx], (0, 255, 0), 1)

    def detect_fight(self, frame_data):
        """Detect violence/fighting"""
        people_landmarks = frame_data['people_landmarks']
        track_ids = list(people_landmarks.keys())
        fight_score = 0
        
        for i in range(len(track_ids)):
            for j in range(i + 1, len(track_ids)):
                id1 = track_ids[i]
                id2 = track_ids[j]
                
                lm1 = people_landmarks[id1]
                lm2 = people_landmarks[id2]
                
                if len(lm1) < 17 or len(lm2) < 17:
                    continue

                # Proximity check
                p1_center = np.mean([lm1[11], lm1[12]], axis=0)
                p2_center = np.mean([lm2[11], lm2[12]], axis=0)
                dist = np.linalg.norm(p1_center - p2_center)
                
                if dist > self.FIGHT_DISTANCE_THRESHOLD:
                    continue

                # Behavior analysis
                speed1 = self.calculate_movement_speed(id1, lm1)
                speed2 = self.calculate_movement_speed(id2, lm2)
                combined_speed = speed1 + speed2

                # Hands-up check
                p1_hands_up = (lm1[15][1] < lm1[13][1]) or (lm1[16][1] < lm1[14][1])
                p2_hands_up = (lm2[15][1] < lm2[13][1]) or (lm2[16][1] < lm2[14][1])
                any_hands_up = p1_hands_up or p2_hands_up

                # Critical area check
                p2_neck = np.mean([lm2[11], lm2[12]], axis=0)
                p1_wrists = [np.array(lm1[15]), np.array(lm1[16])]
                hitting_neck = any(np.linalg.norm(w - p2_neck) < 60 for w in p1_wrists)
                
                p1_neck = np.mean([lm1[11], lm1[12]], axis=0)
                p2_wrists = [np.array(lm2[15]), np.array(lm2[16])]
                hitting_neck = hitting_neck or any(np.linalg.norm(w - p1_neck) < 60 for w in p2_wrists)

                # Decision logic
                if combined_speed > self.MOVEMENT_SPEED_THRESHOLD and any_hands_up:
                    fight_score += 2
                elif hitting_neck:
                    fight_score += 3
                elif combined_speed > (self.MOVEMENT_SPEED_THRESHOLD * 1.5):
                    fight_score += 1

        # Smooth detection
        if fight_score > 0:
            self.fight_counter += 1
            self.normal_counter = 0
        else:
            self.fight_counter = max(0, self.fight_counter - 1)
            self.normal_counter += 1
        
        prev_fight_state = self.fight_detected
        
        if self.fight_counter >= self.CONFIDENCE_THRESHOLD:
            self.fight_detected = True
        elif self.normal_counter >= 10:
            self.fight_detected = False
        
        # Log incident if newly detected
        if self.fight_detected and not prev_fight_state:
            self.log_incident("Violence Detected", frame_data['people_count'])

    def detect_distress_gesture(self, frame_data):
        """Detect distress gesture (hands raised above head)"""
        if not self.female_detected:
            return
        
        people_landmarks = frame_data['people_landmarks']
        distress_detected = False
        
        for track_id, landmarks in people_landmarks.items():
            # Check if this person is female (simplified check)
            is_female = any(track_id == fb[4] for fb in frame_data['female_boxes'])
            if not is_female or len(landmarks) < 17:
                continue
            
            # Check if both wrists are above the head/nose
            left_wrist_y = landmarks[15][1]
            right_wrist_y = landmarks[16][1]
            nose_y = landmarks[0][1] if len(landmarks) > 0 else landmarks[11][1]
            
            # Both hands raised above head
            if left_wrist_y < nose_y and right_wrist_y < nose_y:
                distress_detected = True
                break
        
        # Apply confidence threshold
        if distress_detected:
            self.distress_gesture_counter += 1
        else:
            self.distress_gesture_counter = max(0, self.distress_gesture_counter - 1)
        
        prev_distress_state = self.distress_gesture_detected
        
        if self.distress_gesture_counter >= self.DISTRESS_GESTURE_THRESHOLD:
            self.distress_gesture_detected = True
        elif self.distress_gesture_counter == 0:
            self.distress_gesture_detected = False
        
        # Log incident
        if self.distress_gesture_detected and not prev_distress_state:
            self.log_incident("Women Distress Gesture", frame_data['people_count'])

    def detect_harassment(self, frame_data):
        """Detect potential harassment (female surrounded by multiple people)"""
        if not self.female_detected or frame_data['people_count'] < 2:
            self.harassment_detected = False
            return
        
        female_boxes = frame_data['female_boxes']
        all_boxes = frame_data['people_boxes']
        
        harassment_score = 0
        
        for fx1, fy1, fx2, fy2, f_id in female_boxes:
            f_center = np.array([(fx1 + fx2) / 2, (fy1 + fy2) / 2])
            nearby_count = 0
            
            for bx1, by1, bx2, by2, b_id in all_boxes:
                if b_id == f_id:
                    continue
                
                b_center = np.array([(bx1 + bx2) / 2, (by1 + by2) / 2])
                dist = np.linalg.norm(f_center - b_center)
                
                if dist < 200:  # Within close proximity
                    nearby_count += 1
            
            # If 2+ people near female
            if nearby_count >= 2:
                harassment_score += 1
        
        prev_harassment_state = self.harassment_detected
        self.harassment_detected = harassment_score > 0
        
        # Log incident
        if self.harassment_detected and not prev_harassment_state:
            self.log_incident("Possible Harassment", frame_data['people_count'])

    def check_audio_alerts(self):
        """Check audio queue for distress signals"""
        try:
            while not self.audio_queue.empty():
                alert_type, text = self.audio_queue.get_nowait()
                if alert_type == "distress":
                    prev_audio_state = self.audio_distress_detected
                    self.audio_distress_detected = True
                    
                    if not prev_audio_state:
                        self.log_incident(f"Distress Audio: '{text}'", 0)
                    
                    # Reset after 5 seconds
                    threading.Timer(5.0, lambda: setattr(self, 'audio_distress_detected', False)).start()
        except queue.Empty:
            pass

    def calculate_movement_speed(self, track_id, current_landmarks):
        """Calculate movement speed"""
        if track_id not in self.previous_frame_landmarks:
            return 0
        
        prev_landmarks = self.previous_frame_landmarks[track_id]
        keypoints = [13, 14, 15, 16]
        max_speed = 0
        
        for kp in keypoints:
            if kp < len(current_landmarks) and kp < len(prev_landmarks):
                p1 = np.array(current_landmarks[kp])
                p2 = np.array(prev_landmarks[kp])
                dist = np.linalg.norm(p1 - p2)
                if dist > max_speed:
                    max_speed = dist
        
        return max_speed

    def log_incident(self, event_type, person_count):
        """Log incident to console and file"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_incident_time < self.incident_cooldown:
            return
        
        self.last_incident_time = current_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_short = datetime.now().strftime("%H:%M")
        
        # Console Report
        print("\n" + "=" * 60)
        print("🚨 INCIDENT DETECTED")
        print("=" * 60)
        print(f"Event:           {event_type}")
        print(f"Location:        {self.location}")
        print(f"Camera:          {self.camera_name}")
        print(f"Time:            {timestamp}")
        print(f"Persons Involved: {person_count}")
        print(f"Female Detected:  {'YES' if self.female_detected else 'NO'}")
        print("=" * 60 + "\n")
        
        # File Logging
        log_entry = f"Event: {event_type} | Location: {self.location} | Camera: {self.camera_name} | Time: {timestamp} | Persons: {person_count}\n"
        
        try:
            with open(self.incident_log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"⚠ Failed to write to log file: {e}")

    def display_cctv_ui(self, frame, frame_data):
        """Display CCTV-style UI"""
        height, width, _ = frame.shape
        
        # Top bar (CCTV Info)
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        
        # Camera name
        cv2.putText(frame, f"📹 {self.camera_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Location
        cv2.putText(frame, f"📍 {self.location}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Date and Time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (width - 320, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Bottom status bar
        cv2.rectangle(frame, (0, height-50), (width, height), (0, 0, 0), -1)
        
        status_text = "🟢 Normal Monitoring"
        color = (0, 255, 0)
        
        # Check for critical alerts
        if self.fight_detected:
            status_text = "🚨 VIOLENCE DETECTED"
            color = (0, 0, 255)
            cv2.rectangle(frame, (5, 5), (width-5, height-5), (0, 0, 255), 8)
            
            # Alert overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (width//2 - 250, 100), (width//2 + 250, 180), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "⚠ VIOLENCE ALERT ⚠", (width//2 - 200, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        elif self.distress_gesture_detected:
            status_text = "🚨 WOMEN DISTRESS SIGNAL"
            color = (0, 0, 255)
            cv2.rectangle(frame, (5, 5), (width-5, height-5), (255, 0, 255), 8)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (width//2 - 300, 100), (width//2 + 300, 180), (255, 0, 255), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, "🚨 WOMEN DISTRESS SIGNAL", (width//2 - 280, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        
        elif self.harassment_detected:
            status_text = "⚠ POSSIBLE HARASSMENT"
            color = (0, 165, 255)
            cv2.rectangle(frame, (5, 5), (width-5, height-5), (0, 165, 255), 8)
        
        elif self.audio_distress_detected:
            status_text = "🚨 DISTRESS AUDIO DETECTED"
            color = (0, 0, 255)
            cv2.putText(frame, "🔊 DISTRESS CALL", (width//2 - 150, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        elif frame_data['people_count'] >= 2:
            status_text = "🟡 Analyzing Behavior..."
            color = (0, 215, 255)
        
        # Status text
        text = f"People: {frame_data['people_count']} | {status_text}"
        cv2.putText(frame, text, (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


if __name__ == "__main__":
    print("\n")
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║  SMART CCTV SURVEILLANCE SYSTEM - HACKATHON PROTOTYPE    ║")
    print("║  Violence Detection + Women Safety Monitoring            ║")
    print("╚═══════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Initialize system
    app = SmartCCTVSystem(
        video_source=0,
        camera_name="Camera_02",
        location="MG Road"
    )
    
    # Start monitoring
    app.process_live_feed()
