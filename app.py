from flask import Flask, request, jsonify, render_template, send_from_directory, Response, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import openai
import os
import time
import threading
import uuid
import json
from collections import deque
import sys
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("dribbling_analysis.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='.', template_folder='.')
app.secret_key = 'football_analysis_session_key'  # For session management


# Store analysis results
analysis_storage = {}

# Replace with your API key securely
openai.api_key = os.environ.get("OPENAI_API_KEY", "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")

# MediaPipe setup with improved configuration
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Improved drawing specs for better visualization
pose_connection_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
pose_landmark_drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=2)

# Global variables for video streaming
output_frame = None
lock = threading.Lock()
analysis_results = {"dribbles": 0, "Feinting": 0, "warnings": "", "tips": "", "time": 0, "bad_dribbles": 0}
is_analyzing = False

# Define a class for movement tracking with Kalman filtering for smoother predictions
class MovementTracker:
    def __init__(self, smooth_factor=0.8):
        self.prev_positions = []
        self.smooth_factor = smooth_factor
        self.history = deque(maxlen=30)  # Keep last 30 positions for trajectory analysis
        self.velocity = [0, 0]
        self.last_update_time = None
    
    def update(self, position):
        if position is None:
            return None
        
        try:
            current_time = time.time()
            
            # Add to history
            self.history.append((position, current_time))
            
            # Calculate velocity if possible
            if self.last_update_time is not None and len(self.prev_positions) > 0:
                dt = current_time - self.last_update_time
                if dt > 0:
                    dx = position[0] - self.prev_positions[-1][0]
                    dy = position[1] - self.prev_positions[-1][1]
                    self.velocity = [dx/dt, dy/dt]
            
            # Apply smoothing
            smoothed_position = position
            if self.prev_positions:
                smoothed_position = (
                    int(self.smooth_factor * position[0] + (1 - self.smooth_factor) * self.prev_positions[-1][0]),
                    int(self.smooth_factor * position[1] + (1 - self.smooth_factor) * self.prev_positions[-1][1])
                )
            
            self.prev_positions.append(smoothed_position)
            if len(self.prev_positions) > 10:
                self.prev_positions.pop(0)
                
            self.last_update_time = current_time
            return smoothed_position
        except Exception as e:
            logger.error(f"Error in movement tracker update: {e}")
            return position  # Return original position if smoothing fails
    
    def get_speed(self):
        try:
            return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        except Exception as e:
            logger.error(f"Error calculating speed: {e}")
            return 0
    
    def get_acceleration(self, window=5):
        """Calculate acceleration based on velocity changes"""
        try:
            if len(self.history) < window + 1:
                return 0
            
            recent = list(self.history)[-window:]
            if len(recent) < 2:
                return 0
            
            # Get first and last velocity measurement
            p1, t1 = recent[0]
            p2, t2 = recent[-1]
            
            if t2 == t1:
                return 0
                
            v1 = np.array([0, 0])  # Initial velocity
            v2 = np.array([(p2[0] - p1[0])/(t2 - t1), (p2[1] - p1[1])/(t2 - t1)])  # Final velocity
            
            # Acceleration = change in velocity / change in time
            acceleration = np.linalg.norm(v2 - v1) / (t2 - t1)
            return acceleration
        except Exception as e:
            logger.error(f"Error calculating acceleration: {e}")
            return 0
    
    def get_movement_direction(self):
        """Get the current movement direction as an angle in degrees"""
        try:
            if abs(self.velocity[0]) < 0.1 and abs(self.velocity[1]) < 0.1:
                return None  # No significant movement
            
            angle_rad = np.arctan2(self.velocity[1], self.velocity[0])
            angle_deg = np.degrees(angle_rad)
            return angle_deg
        except Exception as e:
            logger.error(f"Error calculating movement direction: {e}")
            return None
    
    def get_trajectory(self):
        """Return a list of position points for drawing trajectory"""
        try:
            return [pos for pos, _ in self.history]
        except Exception as e:
            logger.error(f"Error getting trajectory: {e}")
            return []

# Create trackers
ball_tracker = MovementTracker(smooth_factor=0.7)  # Less smoothing for the ball
left_foot_tracker = MovementTracker(smooth_factor=0.85)
right_foot_tracker = MovementTracker(smooth_factor=0.85)

@app.route('/')
def index():
    try:
        return render_template('home_dribbling.html')
    except Exception as e:
        logger.error(f"Error rendering home template: {e}")
        return "Error loading home page. Check if home_dribbling.html exists."

@app.route('/<path:path>')
def serve_html(path):
    try:
        if path.endswith('.html'):
            return render_template(path)
        return send_from_directory('.', path)
    except Exception as e:
        logger.error(f"Error serving {path}: {e}")
        return f"Error loading {path}. File may not exist."

# Route for serving static files like audio files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def generate_frames():
    global output_frame, lock, is_analyzing
    
    while True:
        try:
            with lock:
                if output_frame is None or not is_analyzing:
                    # If no frame is being processed, send a placeholder
                    if not is_analyzing:
                        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 255
                        cv2.putText(placeholder, "Waiting for dribbling video analysis to start...", 
                                   (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        _, buffer = cv2.imencode('.jpg', placeholder)
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                        time.sleep(0.5)
                        continue
                    continue
                
                # Encode the frame to JPEG
                _, buffer = cv2.imencode('.jpg', output_frame)
                frame = buffer.tobytes()
            
            # Yield the frame in the response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
        except Exception as e:
            logger.error(f"Error in generate_frames: {e}")
            # Create error frame
            error_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(error_frame, "Error processing video frame", 
                       (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', error_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1.0)  # Longer delay on error

@app.route('/video_feed')
def video_feed():
    """Route for streaming the analyzed video."""
    try:
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video_feed: {e}")
        return "Video feed error", 500

@app.route('/analysis_status')
def analysis_status():
    """Route to get the current analysis results."""
    global analysis_results, is_analyzing
    try:
        has_new_warning = False
        if analysis_results.get("warnings", "") and not hasattr(analysis_status, "last_warning"):
            analysis_status.last_warning = ""
            
        if analysis_results.get("warnings", "") != getattr(analysis_status, "last_warning", ""):
            has_new_warning = True
            analysis_status.last_warning = analysis_results.get("warnings", "")
        
        return jsonify({
            "is_analyzing": is_analyzing,
            "results": analysis_results,
            "new_warning": has_new_warning
        })
    except Exception as e:
        logger.error(f"Error in analysis_status: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_analysis_thumbnail', methods=['POST'])
def save_analysis_thumbnail():
    """Save a thumbnail from the analyzed video for coach view"""
    global output_frame, lock
    
    try:
        analysis_id = request.form.get('analysis_id')
        if not analysis_id or analysis_id not in analysis_storage:
            return jsonify({"error": "Invalid analysis ID"}), 400
        
        with lock:
            if output_frame is not None:
                # Save thumbnail
                thumbnail_path = f"analysis_{analysis_id}_thumbnail.jpg"
                cv2.imwrite(thumbnail_path, output_frame)
                analysis_storage[analysis_id]["thumbnail"] = thumbnail_path
                
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error saving thumbnail: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_analysis/<analysis_id>')
def get_analysis(analysis_id):
    """Get specific analysis results"""
    try:
        if analysis_id not in analysis_storage:
            return jsonify({"error": "Analysis not found"}), 404
            
        return jsonify(analysis_storage[analysis_id])
    except Exception as e:
        logger.error(f"Error getting analysis {analysis_id}: {e}")
        return jsonify({"error": str(e)}), 500

def detect_ball(frame, min_radius=10, max_radius=30):
    """Improved ball detection with multi-technique approach"""
    try:
        # Safety check for frame
        if frame is None or frame.size == 0:
            return None, 0, frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Try to detect circular objects (balls) using Hough Circle Transform
        try:
            circles = cv2.HoughCircles(
                blurred, 
                cv2.HOUGH_GRADIENT, 
                dp=1.2,            # Resolution ratio
                minDist=50,        # Min distance between circles
                param1=100,        # Higher threshold for Canny edge detection
                param2=30,         # Threshold for circle detection
                minRadius=min_radius,
                maxRadius=max_radius
            )
        except cv2.error as cv_err:
            logger.warning(f"OpenCV error in HoughCircles: {cv_err}")
            return None, 0, frame
        
        ball_position = None
        confidence = 0
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # Find the most likely ball (strongest circle)
            best_circle = None
            best_score = 0
            
            for circle in circles[0, :]:
                # Extract the region of the potential ball
                x, y, r = circle
                
                # Safety check for boundaries
                if x - r < 0 or y - r < 0 or x + r >= frame.shape[1] or y + r >= frame.shape[0]:
                    continue
                    
                # Calculate confidence score based on circularity and color
                mask = np.zeros_like(gray)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # Check for typical ball colors (white/black areas)
                ball_region = cv2.bitwise_and(frame, frame, mask=mask)
                
                # Calculate average color and standard deviation
                if np.sum(mask) > 0:  # Avoid division by zero
                    mean_color = cv2.mean(ball_region, mask=mask)
                    
                    # For a typical ball, we expect high variance in colors (black and white patterns)
                    # And generally brighter than surroundings
                    brightness = (mean_color[0] + mean_color[1] + mean_color[2])/3
                    
                    # Simple scoring system
                    score = brightness * r  # Favor larger, brighter circles
                    
                    if score > best_score:
                        best_score = score
                        best_circle = circle
                        confidence = min(1.0, score / 5000)  # Normalize confidence
            
            if best_circle is not None:
                x, y, r = best_circle
                ball_position = (x, y)
                
                # Draw the ball with confidence indicator
                color = (0, int(255 * confidence), 0)  # More green = higher confidence
                cv2.circle(frame, ball_position, r, color, 2)
                cv2.circle(frame, ball_position, 2, (0, 0, 255), 3)
                
                # Show confidence
                cv2.putText(frame, f"Ball {confidence:.2f}", (x - r, y - r - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return ball_position, confidence, frame
    except Exception as e:
        logger.error(f"Error in ball detection: {e}")
        return None, 0, frame

def calculate_angle(a, b, c):
    """Calculate angle between three points (used for joint angles)"""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Handle potential errors with invalid coordinates
        if np.array_equal(a, b) or np.array_equal(b, c):
            return 0
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle
    except Exception as e:
        logger.error(f"Error calculating angle: {e}")
        return 0

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    try:
        return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    except Exception as e:
        logger.error(f"Error calculating distance: {e}")
        return 0

def analyze_video_thread(video_path, analysis_id):
    global output_frame, lock, analysis_results, is_analyzing, analysis_storage
    global ball_tracker, left_foot_tracker, right_foot_tracker
    
    # Reset trackers for this new analysis
    ball_tracker = MovementTracker(smooth_factor=0.7)
    left_foot_tracker = MovementTracker(smooth_factor=0.85)
    right_foot_tracker = MovementTracker(smooth_factor=0.85)
    
    try:
        logger.info(f"Starting dribbling analysis of video: {video_path}")
        is_analyzing = True
        
        # Check if file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file does not exist: {video_path}")
            with lock:
                analysis_results = {"error": "Video file not found."}
                if analysis_id in analysis_storage:
                    analysis_storage[analysis_id]["error"] = "Video file not found."
            is_analyzing = False
            return
        
        # Try to open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {video_path}")
            with lock:
                analysis_results = {"error": "Could not open the video file."}
                if analysis_id in analysis_storage:
                    analysis_storage[analysis_id]["error"] = "Could not open the video file."
            is_analyzing = False
            return
        
        # Check video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
        
        if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
            logger.error(f"Invalid video properties: fps={fps}, frames={frame_count}, size={width}x{height}")
            with lock:
                analysis_results = {"error": "Invalid video format."}
                if analysis_id in analysis_storage:
                    analysis_storage[analysis_id]["error"] = "Invalid video format."
            is_analyzing = False
            cap.release()
            return
        
        # Football metrics - FOCUS ONLY ON DRIBBLING
        dribbles_counter = 0
        bad_dribbles_counter = 0  # New counter for bad dribbles
        ball_control_score = 100
        dribble_quality_sum = 0  # For tracking average quality
        
        # Bad dribble reasons tracking
        bad_dribble_reasons = {}  # To track reasons for bad dribbles
        
        # Skill-specific metrics for Feinting (rapid dribbling)
        Feinting_counter = 0
        Feinting_quality = 0  # Initialize the missing variable
        last_Feinting_time = 0
        Feinting_sequence = []
        
        # Advanced metrics for dribbling only
        movements_data = {
            "dribbles": [],
            "Feinting": []
        }
        
        # Game state tracking
        start_time = time.time()
        gpt_tips_for_overlay = ""
        warnings_list = []
        performance_score = 100
        
        # Event cooldowns to prevent duplicate detections
        event_cooldowns = {
            "dribble": 0,
            "Feinting": 0,
            "warning": 0  # Add cooldown for warnings
        }
        
        # Use a less complex model for better performance and stability
        model_complexity = 1  # Medium complexity (0=light, 1=medium, 2=heavy)
        
        # Set up the improved pose detector with balanced confidence
        with mp_pose.Pose(
            min_detection_confidence=0.6,  # Lower threshold to avoid missing poses
            min_tracking_confidence=0.6,
            model_complexity=model_complexity
        ) as pose:
            frame_count = 0
            skip_frames = 2  # Process every nth frame for performance
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video reached")
                    break
                
                frame_count += 1
                
                # Process only every nth frame for efficiency
                if frame_count % skip_frames != 0:
                    continue
                
                # Initialize warning variable
                warning = ""
                
                try:
                    # Resize for faster processing and consistent display
                    frame = cv2.resize(frame, (640, 480))
                    
                    # Process frame for pose detection
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Process with timeout protection
                    start_process = time.time()
                    results = pose.process(image)
                    process_time = time.time() - start_process
                    
                    # If processing takes too long, it might be stuck
                    if process_time > 1.0:
                        logger.warning(f"Pose detection took {process_time:.2f}s - may be resource intensive")
                    
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
                    # Update cooldowns
                    current_time = time.time()
                    for event_type in event_cooldowns:
                        if event_cooldowns[event_type] > 0:
                            event_cooldowns[event_type] = max(0, event_cooldowns[event_type] - 1)
                    
                    # Try to detect the ball with improved confidence
                    ball_position, ball_confidence, image = detect_ball(image)
                    
                    # Update ball tracker with the detected position
                    ball_detected = False
                    if ball_position and ball_confidence > 0.4:  # Minimum confidence threshold
                        smoothed_ball_pos = ball_tracker.update(ball_position)
                        ball_detected = True
                        
                        # Draw "BALL DETECTED" indicator with confidence
                        conf_color = (0, int(255 * ball_confidence), 0)
                        cv2.putText(image, f"Ball {ball_confidence:.2f}", 
                                    (ball_position[0] - 40, ball_position[1] - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 2)
                        
                        # Draw ball trajectory
                        trajectory = ball_tracker.get_trajectory()
                        if len(trajectory) > 2:
                            for i in range(1, len(trajectory)):
                                # Fade color based on how old the point is
                                alpha = (i / len(trajectory))
                                color = (0, int(255 * alpha), int(255 * (1-alpha)))
                                cv2.line(image, trajectory[i-1], trajectory[i], color, 2)
                    
                    # Process pose landmarks if detected
                    if not results.pose_landmarks:
                        cv2.putText(image, "No player detected", (20, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        # Draw pose landmarks
                        try:
                            mp_drawing.draw_landmarks(
                                image, 
                                results.pose_landmarks, 
                                mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=pose_landmark_drawing_spec,
                                connection_drawing_spec=pose_connection_drawing_spec
                            )
                        except Exception as draw_error:
                            logger.error(f"Error drawing landmarks: {draw_error}")
                        
                        try:
                            # Extract landmarks for football analysis
                            landmarks = results.pose_landmarks.landmark
                            
                            # Check if we have all the landmarks we need
                            required_landmarks = [
                                mp_pose.PoseLandmark.LEFT_ANKLE, 
                                mp_pose.PoseLandmark.RIGHT_ANKLE,
                                mp_pose.PoseLandmark.LEFT_KNEE,
                                mp_pose.PoseLandmark.RIGHT_KNEE,
                                mp_pose.PoseLandmark.LEFT_HIP,
                                mp_pose.PoseLandmark.RIGHT_HIP,
                                mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
                                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
                            ]
                            
                            missing_landmarks = False
                            for landmark in required_landmarks:
                                if landmark.value >= len(landmarks) or landmarks[landmark.value].visibility < 0.5:
                                    missing_landmarks = True
                                    break
                            
                            if not missing_landmarks:
                                # Get key points for football movements
                                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                                
                                # Get additional points for detailed analysis
                                left_foot = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                           landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                                right_foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
                                
                                # Calculate angles for leg movements with error handling
                                left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
                                right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)
                                
                                # Calculate angle between feet - useful for Feinting detection
                                feet_distance = calculate_distance(left_foot, right_foot)
                                
                                # Convert normalized coordinates to pixel coordinates
                                h, w, _ = image.shape
                                left_ankle_px = (int(left_ankle[0] * w), int(left_ankle[1] * h))
                                right_ankle_px = (int(right_ankle[0] * w), int(right_ankle[1] * h))
                                left_foot_px = (int(left_foot[0] * w), int(left_foot[1] * h))
                                right_foot_px = (int(right_foot[0] * w), int(right_foot[1] * h))
                                
                                # Update foot trackers
                                smoothed_left_foot = left_foot_tracker.update(left_foot_px)
                                smoothed_right_foot = right_foot_tracker.update(right_foot_px)
                                
                                # Show angles for debugging
                                cv2.putText(image, f"L Leg: {left_leg_angle:.1f}", 
                                           (left_ankle_px[0] - 30, left_ankle_px[1] - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                cv2.putText(image, f"R Leg: {right_leg_angle:.1f}", 
                                           (right_ankle_px[0] + 10, right_ankle_px[1] - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                
                                # Draw foot speed indicators
                                left_speed = left_foot_tracker.get_speed()
                                right_speed = right_foot_tracker.get_speed()
                                
                                # Visualize foot movement speed
                                left_speed_color = (0, min(255, int(left_speed/5)), 0)
                                right_speed_color = (0, min(255, int(right_speed/5)), 0)
                                
                                cv2.circle(image, left_foot_px, int(5 + min(10, left_speed/50)), left_speed_color, -1)
                                cv2.circle(image, right_foot_px, int(5 + min(10, right_speed/50)), right_speed_color, -1)
                                
                                # ------------------- DRIBBLING SKILL DETECTION -------------------
                                
                                try:
                                    # ----- تعديل خوارزمية اكتشاف المراوغة -----
                                    # 1. Dribble detection - FOCUS ON THIS - تحسين معايير الاكتشاف
                                    if ball_detected and event_cooldowns["dribble"] == 0:
                                        # حساب المسافة بين الكرة وأقرب قدم
                                        left_foot_ball_dist = calculate_distance(left_foot_px, ball_position)
                                        right_foot_ball_dist = calculate_distance(right_foot_px, ball_position)
                                        min_foot_dist = min(left_foot_ball_dist, right_foot_ball_dist)
                                        
                                        # حساب سرعة القدمين
                                        foot_speeds = [left_speed, right_speed]
                                        max_foot_speed = max(foot_speeds)
                                        
                                        # ===== معايير اكتشاف المراوغة الأساسية - أكثر حساسية =====
                                        # المراوغة الأساسية: الكرة قريبة من القدمين + حركة القدمين
                                        basic_dribble = min_foot_dist < 80 and max_foot_speed > 60
                                        
                                        # فحص إضافي لتحديد المراوغات الخاطئة
                                        is_bad_dribble = False
                                        bad_dribble_reason = ""
                                        
                                        # المراوغة تعتبر خاطئة إذا:
                                        if min_foot_dist > 80 and min_foot_dist < 120 and max_foot_speed > 40:
                                            # الكرة بعيدة عن القدمين لكن هناك محاولة للمراوغة
                                            is_bad_dribble = True
                                            bad_dribble_reason = "The ball is too far from feet"
                                        elif min_foot_dist < 80 and max_foot_speed < 40:
                                            # الكرة قريبة لكن حركة القدمين بطيئة جداً
                                            is_bad_dribble = True
                                            bad_dribble_reason = "Foot movement is too slow"
                                        elif left_leg_angle > 160 and right_leg_angle > 160 and min_foot_dist < 100:
                                            # الساقين مستقيمتين أثناء المراوغة (غير مرن)
                                            is_bad_dribble = True
                                            bad_dribble_reason = "Knees are too straight"
                                        
                                        # اكتشاف المراوغة بغض النظر عن انثناء الركبتين
                                        if basic_dribble:
                                            # إذا كان المعيار الأساسي محققاً، نعتبرها مراوغة صحيحة
                                            dribbles_counter += 1
                                            
                                            # حساب مؤشر جودة المراوغة (لكن لا نستخدمه كمعيار للاكتشاف)
                                            quality_score = (1 - (min_foot_dist / 150)) * 0.6  # 60% للمسافة
                                            quality_score += (min(max_foot_speed, 400) / 400) * 0.4  # 40% للسرعة
                                            
                                            # إضافة لقاعدة البيانات
                                            dribble_quality_sum += quality_score
                                            event_cooldowns["dribble"] = 8  # تقليل كولداون لاكتشاف المزيد من المراوغات
                                            
                                            # إضافة بيانات المراوغة
                                            movements_data["dribbles"].append({
                                                "time": current_time - start_time,
                                                "foot_dist": min_foot_dist,
                                                "foot_speed": max_foot_speed,
                                                "quality": round(quality_score * 100)
                                            })
                                            
                                            # عرض الاكتشاف والجودة
                                            quality_percent = int(quality_score * 100)
                                            cv2.putText(image, f"DRIBBLE! {quality_percent}%", (150, 60), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                        elif is_bad_dribble:
                                            # تسجيل المراوغة الخاطئة
                                            bad_dribbles_counter += 1
                                            event_cooldowns["dribble"] = 8
                                            
                                            # تتبع أسباب المراوغة الخاطئة
                                            if bad_dribble_reason in bad_dribble_reasons:
                                                bad_dribble_reasons[bad_dribble_reason] += 1
                                            else:
                                                bad_dribble_reasons[bad_dribble_reason] = 1
                                                
                                            # عرض تحذير المراوغة الخاطئة
                                            cv2.putText(image, f"BAD DRIBBLE: {bad_dribble_reason}", (120, 60), 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                                                       
                                            # تعيين التحذير لتشغيل الصوت
                                            warning = bad_dribble_reason
                                            event_cooldowns["warning"] = 5  # حتى لا يتكرر التحذير كثيراً
                                    
                                    # 2. Feinting detection - تحسين اكتشاف الخوزامية - معايير أكثر حساسية
                                    if ball_detected and event_cooldowns["Feinting"] == 0:
                                        # التحقق من حركة القدمين
                                        left_accel = left_foot_tracker.get_acceleration()
                                        right_accel = right_foot_tracker.get_acceleration()
                                        
                                        # الحصول على مواضع القدمين
                                        if len(left_foot_tracker.history) > 3 and len(right_foot_tracker.history) > 3:
                                            # معايير الخوزامية الأساسية: الكرة قريبة + حركة القدمين السريعة
                                            min_foot_dist = min(
                                                calculate_distance(left_foot_px, ball_position),
                                                calculate_distance(right_foot_px, ball_position)
                                            )
                                            
                                            # معايير أكثر حساسية لاكتشاف الخوزامية
                                            basic_Feinting = (
                                                min_foot_dist < 80 and  # الكرة قريبة من القدمين
                                                (left_speed > 80 or right_speed > 80) and  # حركة سريعة للقدمين
                                                (left_accel > 300 or right_accel > 300)  # بعض التسارع للقدمين
                                            )
                                            
                                            if basic_Feinting and current_time - last_Feinting_time > 1.0:
                                                # اكتشفنا خوزامية
                                                Feinting_counter += 1
                                                Feinting_quality_score = 0.7  # جودة افتراضية عالية
                                                Feinting_quality += Feinting_quality_score
                                                last_Feinting_time = current_time
                                                event_cooldowns["Feinting"] = 15
                                                
                                                # تسجيل البيانات
                                                Feinting_sequence.append({
                                                    "time": current_time - start_time,
                                                    "quality": Feinting_quality_score,
                                                    "foot_dist": min_foot_dist,
                                                    "ball_control": 1 - (min_foot_dist / 100)
                                                })
                                                
                                                # إضافة للحركات
                                                movements_data["Feinting"].append({
                                                    "time": current_time - start_time,
                                                    "quality": Feinting_quality_score,
                                                    "ball_control": 1 - (min_foot_dist / 100)
                                                })
                                                
                                                # عرض اكتشاف الخوزامية مع الجودة
                                                quality_percent = int(Feinting_quality_score * 100)
                                                cv2.putText(image, f"Feinting! {quality_percent}%", (220, 80), 
                                                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                                    
                                    # تحسين التحذيرات لتشغيل التنبيهات الصوتية
                                    if warning == "" and ball_detected and event_cooldowns["warning"] == 0:
                                        # حساب المسافة بين الكرة وأقرب قدم
                                        min_foot_dist = min(
                                            calculate_distance(left_foot_px, ball_position),
                                            calculate_distance(right_foot_px, ball_position)
                                        )
                                        
                                        # Warning conditions
                                        if min_foot_dist > 120:
                                            warning = "Try to keep the ball closer to your feet for better control."
                                            ball_control_score -= 0.1
                                            event_cooldowns["warning"] = 10
                                        elif min_foot_dist > 100 and (left_speed > 50 or right_speed > 50):
                                            warning = "The ball is getting too far from your feet while moving."
                                            ball_control_score -= 0.05
                                            event_cooldowns["warning"] = 10
                                        elif left_leg_angle > 160 and right_leg_angle > 160 and min_foot_dist < 100:
                                            warning = "Keep your knees slightly bent for better balance and control."
                                            performance_score -= 0.1
                                            event_cooldowns["warning"] = 10
                                        elif max(left_speed, right_speed) < 30 and min_foot_dist < 80:
                                            warning = "Try to increase your foot speed during dribbling."
                                            performance_score -= 0.1
                                            event_cooldowns["warning"] = 10
                                    
                                    if warning:
                                        warnings_list.append(warning)
                                    
                                except Exception as event_error:
                                    logger.error(f"Error in event detection: {event_error}")
                                
                                # Keep performance score in valid range
                                performance_score = max(0, min(100, performance_score))
                                ball_control_score = max(0, min(100, ball_control_score))
                            else:
                                # Some landmarks are missing
                                cv2.putText(image, "Some body landmarks not detected", (20, 80), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        except Exception as landmark_error:
                            logger.error(f"Error processing landmarks: {landmark_error}")
                            cv2.putText(image, "Error in landmark processing", (20, 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display information on frame
                    elapsed_time = current_time - start_time
                    
                    # Create transparent overlay for stats
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0), (640, 150), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
                    
                    # Top row stats with improved styling - FOCUS ON DRIBBLING ONLY
                    cv2.putText(image, f"Dribbles: {dribbles_counter}", (30, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Add bad dribbles counter
                    cv2.putText(image, f"Bad: {bad_dribbles_counter}", (180, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                    
                    # Add Feinting counter
                    Feinting_color = (0, 255, 255) if Feinting_counter > 0 else (200, 200, 200)
                    cv2.putText(image, f"Feinting: {Feinting_counter}", (280, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, Feinting_color, 2)
                    
                    # Time display
                    cv2.putText(image, f"Time: {elapsed_time:.1f}s", (460, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Warning message if any
                    if warning:
                        cv2.putText(image, f"Tip: {warning}", (30, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Performance scores
                    cv2.putText(image, f"Overall: {int(performance_score)}%", (30, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(image, f"Ball Control: {int(ball_control_score)}%", (230, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Mode indicator
                    cv2.putText(image, "Mode: Dribbling/Feinting Analysis", (30, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show tips from GPT - call GPT less frequently to avoid API errors
                    if (dribbles_counter > 0 or Feinting_counter > 0) and (dribbles_counter + Feinting_counter) % 3 == 0 and not gpt_tips_for_overlay:
                        # Try to get tips from GPT
                        try:
                            # Request tips only if we've detected some activity
                            user_msg = (
                                f"Football/Soccer dribbling technique analysis. Details:\n"
                                f"- Dribbles detected: {dribbles_counter}\n"
                                f"- Feinting moves: {Feinting_counter}\n"
                                f"- Current knee bend: Left leg {left_leg_angle:.1f}°, Right leg {right_leg_angle:.1f}°\n"
                                f"- Foot speed: Left {left_speed:.1f}, Right {right_speed:.1f}\n"
                                f"Please provide ONE specific positive tip to enhance the correct dribbling and Feinting technique, "
                                f"focusing only on how to improve the good technique the player already shows. "
                                f"Do not mention weaknesses or mistakes."
                            )
                            
                            # Try to get tips from GPT, with error handling
                            try:
                                response = openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                        {"role": "user", "content": user_msg}
                                    ],
                                    max_tokens=100  # Limit response size
                                )
                                gpt_tips_for_overlay = response.choices[0].message.content.strip()
                                logger.info(f"Got GPT tip: {gpt_tips_for_overlay[:30]}...")
                            except Exception as gpt_error:
                                logger.error(f"Error calling GPT: {gpt_error}")
                                # نصائح افتراضية أفضل تركز على الإيجابيات
                                dribbling_tips = [
                                    "Continue to keep the ball close to your feet while dribbling to increase control.",
                                    "To improve your dribbling, try increasing the speed at which you switch feet while maintaining the same control.",
                                    "Try changing direction suddenly while dribbling to improve the effectiveness of your dribbling.",
                                    "Maintain a low center of gravity while dribbling for better balance and increased speed.",
                                    "Focus on using the inside of your foot for precise touches while dribbling."
                                ]
                                import random
                                gpt_tips_for_overlay = random.choice(dribbling_tips)
                        except Exception as tip_error:
                            logger.error(f"Error generating tips: {tip_error}")
                    
                    # Show tips on overlay if available
                    if gpt_tips_for_overlay:
                        y0, dy = 420, 20
                        # Add a "Coach Tips" header with background
                        tip_overlay = image.copy()
                        cv2.rectangle(tip_overlay, (10, 380), (630, 460), (0, 0, 0), -1)
                        cv2.addWeighted(tip_overlay, 0.7, image, 0.3, 0, image)
                        
                        cv2.putText(image, "Coach Tips:", (30, 400), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                        
                        # Display tips
                        for i, line in enumerate(gpt_tips_for_overlay.split('\n')[:2]):
                            cv2.putText(image, line[:60], (30, y0 + i * dy), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Update the output frame with lock
                    with lock:
                        output_frame = image.copy()
                        
                        # Calculate averages
                        avg_dribble_quality = 0
                        if dribbles_counter > 0:
                            avg_dribble_quality = dribble_quality_sum / dribbles_counter
                        
                        # Calculate Feinting quality if applicable
                        Feinting_avg_quality = 0
                        if Feinting_counter > 0:
                            Feinting_avg_quality = Feinting_quality / Feinting_counter
                        
                        # Update analysis results - DRIBBLING FOCUSED with quality metrics
                        current_results = {
                            "dribbles": dribbles_counter,
                            "dribble_quality": round(avg_dribble_quality * 100, 1),
                            "Feinting": Feinting_counter,
                            "Feinting_quality": round(Feinting_avg_quality * 100, 1),
                            "warnings": warning if warning else "",  # Add latest warning for sound alert
                            "tips": gpt_tips_for_overlay,
                            "time": round(elapsed_time, 2),
                            "performance": int(performance_score),
                            "ball_control": int(ball_control_score),
                            "bad_dribbles": bad_dribbles_counter,
                            "bad_dribble_reasons": bad_dribble_reasons
                        }
                        analysis_results = current_results
                        
                        # Also update the stored analysis
                        if analysis_id in analysis_storage:
                            analysis_storage[analysis_id].update(current_results)
                    
                except Exception as frame_error:
                    logger.error(f"Error processing frame {frame_count}: {frame_error}")
                    traceback.print_exc()
                    
                    # Create error frame
                    error_frame = frame.copy() if frame is not None else np.ones((480, 640, 3), dtype=np.uint8) * 255
                    cv2.putText(error_frame, f"Frame processing error: {str(frame_error)[:50]}", 
                               (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    with lock:
                        output_frame = error_frame
  
                # Slow down processing slightly for smooth display on slower systems
                time.sleep(0.01)
        
        # Release resources
        cap.release()
        logger.info(f"Video analysis completed: {dribbles_counter} dribbles, {Feinting_counter} Feinting moves, {bad_dribbles_counter} bad dribbles")
        
        # Final update with analysis complete
        # Calculate averages
        avg_dribble_quality = 0
        if dribbles_counter > 0:
            avg_dribble_quality = dribble_quality_sum / dribbles_counter
            
        # Calculate Feinting quality
        Feinting_avg_quality = 0
        if Feinting_counter > 0:
           Feinting_avg_quality = Feinting_quality / Feinting_counter
        
        # Create final summary
        with lock:
            analysis_results["analysis_complete"] = True
            analysis_results["bad_dribbles"] = bad_dribbles_counter
            
            # Transform bad_dribble_reasons to a list of tuples for easier handling in frontend
            bad_dribble_reasons_list = []
            for reason, count in bad_dribble_reasons.items():
                bad_dribble_reasons_list.append((reason, count))
            analysis_results["bad_dribble_reasons"] = sorted(bad_dribble_reasons_list, key=lambda x: x[1], reverse=True)
            
            if analysis_id in analysis_storage:
                analysis_storage[analysis_id]["completed"] = True
                analysis_storage[analysis_id]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                analysis_storage[analysis_id]["bad_dribbles"] = bad_dribbles_counter
                analysis_storage[analysis_id]["bad_dribble_reasons"] = sorted(bad_dribble_reasons_list, key=lambda x: x[1], reverse=True)
                
                # Calculate dribbling-specific skill levels
                skill_ratings = {
                    "dribbling": min(10, max(1, int(dribbles_counter * 0.8))),  # Scale more favorably
                    "ball_control": min(10, max(1, round(ball_control_score/10))),
                    "Feinting_technique": min(10, max(1, round(Feinting_counter * 0.7))),  # Scale more favorably
                    "agility": min(10, max(1, round((performance_score / 10))))
                }
                
                analysis_storage[analysis_id]["skills"] = skill_ratings
                analysis_storage[analysis_id]["movement_data"] = movements_data
                
                # Identify top errors with counts but only for major issues
                if warnings_list:
                    warnings_count = {}
                    for warning in warnings_list:
                        if warning in warnings_count:
                            warnings_count[warning] += 1
                        else:
                            warnings_count[warning] = 1
                    
                    # Get top 2 most common errors
                    top_errors = sorted(warnings_count.items(), key=lambda x: x[1], reverse=True)[:2]
                    analysis_storage[analysis_id]["top_errors"] = [{"error": e, "count": c} for e, c in top_errors]
                else:
                    analysis_storage[analysis_id]["top_errors"] = []
                    
                # Add exercise type
                analysis_storage[analysis_id]["exercise_type"] = "dribbling"
                
                # Generate final summary
                try:
                    # Try to get a final summary from GPT
                    if openai.api_key:
                        summary_msg = (
                            f"Please provide a concise positive summary for a football player who completed a dribbling training session, focusing on Feinting technique.\n\n"
                            f"Session statistics:\n"
                            f"- Duration: {round(elapsed_time, 1)} seconds\n"
                            f"- Dribbles: {dribbles_counter}\n"
                            f"- Dribble quality score: {round(avg_dribble_quality * 100)}%\n"
                            f"- Bad dribbles: {bad_dribbles_counter}\n"
                            f"- Feinting moves: {Feinting_counter}\n"
                            f"- Feinting quality: {round(Feinting_avg_quality * 100)}%\n"
                            f"- Ball control: {int(ball_control_score)}%\n\n"
                            f"Create a positive and encouraging summary focusing on strengths and correct techniques. "
                            f"Then provide 2-3 specific tips for further enhancing these already good techniques. "
                            f"Your response should be upbeat and focus only on what the player did well, assuming they have good dribbling skills."
                        )
                        
                        try:
                            response = openai.ChatCompletion.create(
                                model="gpt-4",
                                messages=[
                                    {"role": "user", "content": summary_msg}
                                ],
                                max_tokens=200
                            )
                            final_summary = response.choices[0].message.content.strip()
                            analysis_storage[analysis_id]["summary"] = final_summary
                        except Exception as gpt_error:
                            logger.error(f"Error generating GPT summary: {gpt_error}")
                            # Create a positive fallback summary for dribbling
                            analysis_storage[analysis_id]["summary"] = (
                                f"You showed outstanding performance with {dribbles_counter} Strong dribble and {Feinting_counter} Good technique Feinting movement "
                                f"Your ball control was {int(ball_control_score)}%. "
                                f"\n\nTo develop, focus on: "
                                f"\n1. Keeping your knees slightly bent for more flexibility and control while dribbling."
                                f"\n2. Accelerating your footwork during the dribbling while maintaining the same level of accuracy."
                                f"\n3. Developing your ability to change direction suddenly to increase dribbling effectiveness."
                            )
                    else:
                        # No API key available
                        analysis_storage[analysis_id]["summary"] = (
                            f"Dribbling analysis complete. {dribbles_counter} showed good dribbling and "
                            f"{Feinting_counter} Feinting movement over {round(elapsed_time)} seconds with a control level of {int(ball_control_score)}%."
                            f" Continue to keep the ball close to your feet and develop speed in changing feet in Feinting!"
                        )
                except Exception as summary_error:
                    logger.error(f"Error creating summary: {summary_error}")
                    analysis_storage[analysis_id]["summary"] = "Dribbling analysis completed successfully. You showed excellent ball control skills!"
        
    except Exception as e:
        logger.error(f"Error in video analysis: {e}")
        traceback.print_exc()
        with lock:
            analysis_results = {"error": str(e)}
            if analysis_id in analysis_storage:
                analysis_storage[analysis_id]["error"] = str(e)
    finally:
        is_analyzing = False
        logger.info("Analysis thread completed")

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Route to handle video upload and start analysis."""
    global is_analyzing, analysis_storage
    
    try:
        # Check if already analyzing
        if is_analyzing:
            return jsonify({"error": "Already analyzing a video. Please wait."}), 400
        
        video_file = request.files.get('video')
        if not video_file:
            return jsonify({"error": "No video file provided."}), 400
        
        # Validate video file
        if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.wmv', '.mkv')):
            return jsonify({"error": "Invalid video format. Please upload MP4, AVI, MOV, WMV, or MKV files."}), 400
        
        # Create a unique ID for this analysis
        analysis_id = str(uuid.uuid4())
        
        # Save the uploaded video
        video_path = f'uploaded_video_{analysis_id}.mp4'
        video_file.save(video_path)
        
        # Verify the video can be opened
        try:
            test_cap = cv2.VideoCapture(video_path)
            if not test_cap.isOpened():
                os.remove(video_path)
                return jsonify({"error": "The uploaded file is not a valid video."}), 400
            
            # Get basic video info
            fps = test_cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
                test_cap.release()
                os.remove(video_path)
                return jsonify({"error": "Invalid video format or corrupted file."}), 400
                
            # Check video is not too long
            duration = frame_count / fps if fps > 0 else 0
            if duration > 300:  # 5 minutes max
                test_cap.release()
                os.remove(video_path)
                return jsonify({"error": "Video too long. Please limit to 5 minutes maximum."}), 400
                
            test_cap.release()
        except Exception as e:
            logger.error(f"Error validating video: {e}")
            if os.path.exists(video_path):
                os.remove(video_path)
            return jsonify({"error": f"Could not process video: {str(e)}"}), 400
        
        # Initialize analysis storage
        analysis_storage[analysis_id] = {
            "id": analysis_id,
            "video_path": video_path,
            "exercise_type": "dribbling",  # Always dribbling
            "completed": False
        }
        
        # Start analysis in a separate thread
        thread = threading.Thread(target=analyze_video_thread, args=(video_path, analysis_id))
        thread.daemon = True
        thread.start()
        
        # Store the analysis ID in session
        session['current_analysis'] = analysis_id
        
        return jsonify({
            "message": "Dribbling video analysis started", 
            "stream_url": "/video_feed", 
            "analysis_id": analysis_id
        })
    except Exception as e:
        logger.error(f"Error in analyze_video route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/finish_analysis')
def finish_analysis():
    """Redirect to coach view with current analysis"""
    try:
        analysis_id = session.get('current_analysis')
        if not analysis_id:
            return redirect('/coach_dribbling.html')
        
        return redirect(url_for('serve_html', path=f'coach_dribbling.html?analysis_id={analysis_id}'))
    except Exception as e:
        logger.error(f"Error in finish_analysis: {e}")
        return redirect(url_for('serve_html', path='coach_dribbling.html'))

@app.route('/clear_old_analysis', methods=['POST'])
def clear_old_analysis():
    """Clean up old analysis data to free up memory"""
    try:
        # Keep only the 10 most recent analyses
        if len(analysis_storage) > 10:
            # Sort by timestamp (or creation time if no timestamp)
            sorted_analyses = sorted(
                analysis_storage.items(),
                key=lambda x: x[1].get('timestamp', '1970-01-01')
            )
            
            # Remove oldest analyses and their video files
            for analysis_id, analysis in sorted_analyses[:-10]:
                if 'video_path' in analysis and os.path.exists(analysis['video_path']):
                    try:
                        os.remove(analysis['video_path'])
                    except:
                        pass
                        
                if 'thumbnail' in analysis and os.path.exists(analysis['thumbnail']):
                    try:
                        os.remove(analysis['thumbnail'])
                    except:
                        pass
                        
                del analysis_storage[analysis_id]
        
        return jsonify({"success": True, "remaining": len(analysis_storage)})
    except Exception as e:
        logger.error(f"Error clearing old analyses: {e}")
        return jsonify({"error": str(e)}), 500

# Make sure we have a static directory for audio files
if not os.path.exists('static'):
    os.makedirs('static')

# Create a default alert sound if one doesn't exist
if not os.path.exists('static/alert.mp3'):
    try:
        # Try to download a default alert sound
        import urllib.request
        urllib.request.urlretrieve(
            "https://assets.mixkit.co/active_storage/sfx/2869/2869-preview.mp3",
            "static/alert.mp3"
        )
        logger.info("Downloaded default alert sound")
    except Exception as e:
        logger.error(f"Could not download alert sound: {e}")

if __name__ == '__main__':
    # Make sure the necessary files exist
    for required_file in ['home_dribbling.html', 'coach_dribbling.html']:
        if not os.path.exists(required_file):
            logger.error(f"Required file {required_file} not found")
            print(f"ERROR: Required file {required_file} not found in {os.getcwd()}")
            print("Make sure these HTML files are in the same directory as app.py")
    
    logger.info("Starting football dribbling analysis server")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)