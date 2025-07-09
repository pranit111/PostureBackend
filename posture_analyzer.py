import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
import math


class PostureAnalyzer:
    """Main class for analyzing posture from image frames"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
    def analyze_frame(self, image: np.ndarray) -> dict:
        """
        Analyze a single frame for posture
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with posture analysis results
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.pose.process(rgb_image)
            
            if results.pose_landmarks:
                # Extract landmarks
                landmarks = self._extract_landmarks(results.pose_landmarks)
                
                # Detect view orientation
                view_type = self._detect_view_orientation(landmarks)
                
                # Calculate back angle
                back_angle = self._calculate_back_angle(landmarks)
                
                # Determine posture status and bad_posture flag
                posture_status, bad_posture = self._determine_posture_status(back_angle)
                
                # Generate reason
                reason = self._generate_reason(back_angle, bad_posture, view_type, posture_status)
                
                return {
                    "bad_posture": bad_posture,
                    "reason": reason,
                    "back_angle": back_angle,  # Frontend expects this field name
                    "angle": back_angle,       # Keep for compatibility
                    "view_type": view_type,
                    "analysis_method": "side_view" if view_type == "side" else "front_view",
                    "posture_status": posture_status
                }
            else:
                return {
                    "bad_posture": False,
                    "reason": "No pose detected - please position yourself in front of the camera",
                    "back_angle": None,
                    "angle": None,
                    "view_type": "unknown",
                    "analysis_method": "none",
                    "posture_status": "no_detection"
                }
                
        except Exception as e:
            return {
                "bad_posture": False,
                "reason": f"Analysis error: {str(e)}",
                "back_angle": None,
                "angle": None,
                "view_type": "error",
                "analysis_method": "error",
                "posture_status": "error"
            }
    
    def _extract_landmarks(self, pose_landmarks) -> dict:
        """Extract relevant landmarks from MediaPipe results"""
        landmarks = {}
        
        # Get specific landmarks we need
        landmark_indices = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'left_knee': self.mp_pose.PoseLandmark.LEFT_KNEE,
            'right_knee': self.mp_pose.PoseLandmark.RIGHT_KNEE,
            'nose': self.mp_pose.PoseLandmark.NOSE,
            'left_ear': self.mp_pose.PoseLandmark.LEFT_EAR,
            'right_ear': self.mp_pose.PoseLandmark.RIGHT_EAR
        }
        
        for name, idx in landmark_indices.items():
            landmark = pose_landmarks.landmark[idx]
            landmarks[name] = {
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            }
        
        return landmarks
    
    def _calculate_back_angle(self, landmarks: dict) -> Optional[float]:
        """
        Calculate back angle using shoulder, hip, and knee landmarks
        Works for both side view and front view
        
        Args:
            landmarks: Dictionary of pose landmarks
            
        Returns:
            Back angle in degrees or None if calculation fails
        """
        try:
            # Detect view orientation first
            view_type = self._detect_view_orientation(landmarks)
            
            if "side" in view_type:
                return self._calculate_side_view_angle(landmarks)
            elif view_type == "front":
                return self._calculate_front_view_angle(landmarks)
            else:
                # Default to side view calculation
                return self._calculate_side_view_angle(landmarks)
            
        except Exception as e:
            print(f"Error calculating back angle: {e}")
            return None
    
    def _detect_view_orientation(self, landmarks: dict) -> str:
        """Detect if person is facing front or side"""
        try:
            # Check nose position relative to ears
            nose_x = landmarks['nose']['x']
            left_ear_x = landmarks['left_ear']['x']
            right_ear_x = landmarks['right_ear']['x']
            left_ear_vis = landmarks['left_ear']['visibility']
            right_ear_vis = landmarks['right_ear']['visibility']
            
            # If both ears are visible and nose is between them, likely front view
            if left_ear_vis > 0.5 and right_ear_vis > 0.5:
                ear_distance = abs(left_ear_x - right_ear_x)
                if ear_distance > 0.08 and min(left_ear_x, right_ear_x) < nose_x < max(left_ear_x, right_ear_x):
                    return "front"
            
            # If only one ear is clearly visible, it's side view
            if left_ear_vis > 0.7 and right_ear_vis < 0.3:
                return "side_left"
            elif right_ear_vis > 0.7 and left_ear_vis < 0.3:
                return "side_right"
            
            return "side"  # Default to side view
        except:
            return "side"  # Default to side view
    
    def _calculate_side_view_angle(self, landmarks: dict) -> Optional[float]:
        """Calculate angle for side view - better for detecting forward lean"""
        try:
            # Get key points for side view analysis
            nose_x = landmarks['nose']['x']
            nose_y = landmarks['nose']['y']
            
            shoulder_x = (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2
            shoulder_y = (landmarks['left_shoulder']['y'] + landmarks['right_shoulder']['y']) / 2
            
            hip_x = (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2
            hip_y = (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2
            
            # Calculate spine angle (hip to shoulder to head)
            # Vector from hip to shoulder
            hip_shoulder_vector = np.array([shoulder_x - hip_x, shoulder_y - hip_y])
            # Vector from shoulder to head
            shoulder_head_vector = np.array([nose_x - shoulder_x, nose_y - shoulder_y])
            
            # Calculate angle between vectors
            dot_product = np.dot(hip_shoulder_vector, shoulder_head_vector)
            magnitude_hip_shoulder = np.linalg.norm(hip_shoulder_vector)
            magnitude_shoulder_head = np.linalg.norm(shoulder_head_vector)
            
            if magnitude_hip_shoulder == 0 or magnitude_shoulder_head == 0:
                return None
            
            cos_angle = dot_product / (magnitude_hip_shoulder * magnitude_shoulder_head)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            
            angle_rad = np.arccos(cos_angle)
            angle_deg = np.degrees(angle_rad)
            
            # Convert to back angle (straight alignment should be close to 180°)
            back_angle = angle_deg
            
            # Normalize to expected range (160-180 for good posture)
            if back_angle < 90:
                back_angle = 180 - back_angle
            
            return round(back_angle, 1)
            
        except Exception as e:
            print(f"Error calculating side view angle: {e}")
            return None
    
    def _calculate_front_view_angle(self, landmarks: dict) -> Optional[float]:
        """Calculate posture angle for front view - better for alignment"""
        try:
            # For front view, we focus on head-shoulder alignment and shoulder level
            nose_x = landmarks['nose']['x']
            nose_y = landmarks['nose']['y']
            
            left_shoulder_x = landmarks['left_shoulder']['x']
            left_shoulder_y = landmarks['left_shoulder']['y']
            right_shoulder_x = landmarks['right_shoulder']['x']
            right_shoulder_y = landmarks['right_shoulder']['y']
            
            shoulder_center_x = (left_shoulder_x + right_shoulder_x) / 2
            shoulder_center_y = (left_shoulder_y + right_shoulder_y) / 2
            
            hip_x = (landmarks['left_hip']['x'] + landmarks['right_hip']['x']) / 2
            hip_y = (landmarks['left_hip']['y'] + landmarks['right_hip']['y']) / 2
            
            # Calculate head position relative to shoulders
            head_deviation = abs(nose_x - shoulder_center_x)
            
            # Calculate shoulder alignment (should be level)
            shoulder_tilt = abs(left_shoulder_y - right_shoulder_y)
            
            # Calculate spine alignment (shoulder to hip should be vertical)
            spine_lean = abs(shoulder_center_x - hip_x)
            
            # Combine metrics to get overall posture score
            # Lower values = better posture
            total_deviation = head_deviation + shoulder_tilt + spine_lean
            
            # Convert to angle scale (good posture = higher angle)
            max_deviation = 0.15  # Maximum acceptable combined deviation
            posture_score = max(0, 1 - (total_deviation / max_deviation))
            
            # Map to angle range (165-180 for good posture in front view)
            angle = 165 + (posture_score * 15)
            
            return round(angle, 1)
            
        except Exception as e:
            print(f"Error calculating front view angle: {e}")
            return None
    
    def _determine_posture_status(self, angle: Optional[float]) -> tuple[str, bool]:
        """Determine posture status and bad_posture flag based on angle"""
        if angle is None:
            return "no_detection", False
        
        # Enhanced posture classification
        if angle >= 175:
            return "excellent", False
        elif angle >= 165:
            return "good", False
        elif angle >= 155:
            return "fair", True
        elif angle >= 145:
            return "poor", True
        elif angle >= 135:
            return "bad", True
        else:
            return "very_bad", True
    
    def _generate_reason(self, angle: Optional[float], bad_posture: bool, view_type: str = "side", posture_status: str = "unknown") -> Optional[str]:
        """Generate reason for posture assessment with updated thresholds"""
        if angle is None:
            return "Unable to detect pose landmarks"
        
        # Don't show reason for good posture
        if posture_status in ["excellent", "good"]:
            return None
        
        view_prefix = f"[{view_type.upper()} VIEW] "
        
        if posture_status == "fair":
            return f"{view_prefix}Slightly poor posture - minor adjustments needed (angle: {angle}°)"
        elif posture_status == "poor":
            return f"{view_prefix}Moderate slouching detected - straighten your back (angle: {angle}°)"
        elif posture_status == "bad":
            return f"{view_prefix}Significant slouching - major posture correction needed (angle: {angle}°)"
        elif posture_status == "very_bad":
            return f"{view_prefix}Severe slouching - urgent posture correction required (angle: {angle}°)"
        else:
            return f"{view_prefix}Posture analysis complete (angle: {angle}°)"
    
    def get_additional_metrics(self, landmarks: dict) -> dict:
        """
        Calculate additional posture metrics for future use
        
        Args:
            landmarks: Dictionary of pose landmarks
            
        Returns:
            Dictionary with additional metrics
        """
        try:
            # Head forward posture
            head_forward = self._calculate_head_forward_posture(landmarks)
            
            # Shoulder alignment
            shoulder_alignment = self._calculate_shoulder_alignment(landmarks)
            
            # Overall confidence
            confidence = self._calculate_confidence(landmarks)
            
            return {
                "head_forward": head_forward,
                "shoulder_alignment": shoulder_alignment,
                "confidence": confidence
            }
        except Exception as e:
            return {
                "head_forward": 0.0,
                "shoulder_alignment": 0.0,
                "confidence": 0.0
            }
    
    def _calculate_head_forward_posture(self, landmarks: dict) -> float:
        """Calculate head forward posture metric"""
        try:
            # Get ear and shoulder positions
            ear_x = (landmarks['left_ear']['x'] + landmarks['right_ear']['x']) / 2
            shoulder_x = (landmarks['left_shoulder']['x'] + landmarks['right_shoulder']['x']) / 2
            
            # Calculate forward lean
            forward_lean = abs(ear_x - shoulder_x)
            return round(forward_lean * 100, 1)  # Convert to percentage
        except:
            return 0.0
    
    def _calculate_shoulder_alignment(self, landmarks: dict) -> float:
        """Calculate shoulder alignment metric"""
        try:
            left_shoulder_y = landmarks['left_shoulder']['y']
            right_shoulder_y = landmarks['right_shoulder']['y']
            
            # Calculate height difference
            height_diff = abs(left_shoulder_y - right_shoulder_y)
            return round(height_diff * 100, 1)  # Convert to percentage
        except:
            return 0.0
    
    def _calculate_confidence(self, landmarks: dict) -> float:
        """Calculate overall confidence of the pose detection"""
        try:
            key_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            visibilities = [landmarks[lm]['visibility'] for lm in key_landmarks]
            average_visibility = sum(visibilities) / len(visibilities)
            return round(average_visibility * 100, 1)
        except:
            return 0.0


# Utility functions
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better pose detection
    
    Args:
        image: Input image
        
    Returns:
        Preprocessed image
    """
    # Resize if too large
    height, width = image.shape[:2]
    if width > 1280 or height > 720:
        aspect_ratio = width / height
        if aspect_ratio > 1:
            new_width = 1280
            new_height = int(1280 / aspect_ratio)
        else:
            new_height = 720
            new_width = int(720 * aspect_ratio)
        image = cv2.resize(image, (new_width, new_height))
    
    # Enhance contrast
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    enhanced = cv2.merge([l, a, b])
    image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    return image


def validate_image(image: np.ndarray) -> bool:
    """
    Validate if image is suitable for pose detection
    
    Args:
        image: Input image
        
    Returns:
        True if image is valid, False otherwise
    """
    if image is None:
        return False
    
    height, width = image.shape[:2]
    if height < 100 or width < 100:
        return False
    
    # Check if image is not too dark
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 20:
        return False
    
    return True
