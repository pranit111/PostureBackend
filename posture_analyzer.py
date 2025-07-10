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


class VideoPostureAnalyzer:
    """Video posture analyzer for detecting activities and providing activity-specific feedback"""
    
    def __init__(self):
        self.frame_analyzer = PostureAnalyzer()
        self.activity_detectors = {
            'squat': self._detect_squat_activity,
            'sitting': self._detect_sitting_activity,
            'standing': self._detect_standing_activity,
            'walking': self._detect_walking_activity
        }
    
    def analyze_video(self, video_path: str) -> dict:
        """
        Analyze entire video for posture and activity
        
        Args:
            video_path: Path to video file
            
        Returns:
            Complete video analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Sample frames (analyze every 10th frame for performance)
            frame_analyses = []
            frame_count = 0
            analyzed_frames = 0
            activity_votes = {}
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for performance (analyze every 10th frame)
                if frame_count % 10 != 0:
                    continue
                
                analyzed_frames += 1
                timestamp = frame_count / fps
                
                # Analyze this frame
                frame_result = self.frame_analyzer.analyze_frame(frame)
                
                # Detect activity in this frame
                detected_activity = self._detect_activity_in_frame(frame_result)
                
                # Vote for activity
                if detected_activity in activity_votes:
                    activity_votes[detected_activity] += 1
                else:
                    activity_votes[detected_activity] = 1
                
                # Get activity-specific analysis
                activity_issues, suggestions = self._get_activity_specific_feedback(
                    detected_activity, frame_result, frame
                )
                
                # Store frame analysis
                frame_analysis = {
                    "frame_number": frame_count,
                    "timestamp": timestamp,
                    "bad_posture": frame_result.get("bad_posture", True),
                    "back_angle": frame_result.get("back_angle"),
                    "posture_status": frame_result.get("posture_status", "unknown"),
                    "activity_specific_issues": activity_issues,
                    "suggestions": suggestions
                }
                
                frame_analyses.append(frame_analysis)
            
            cap.release()
            
            # Determine dominant activity
            dominant_activity = max(activity_votes, key=activity_votes.get) if activity_votes else "unknown"
            
            # Calculate overall posture score
            overall_score = self._calculate_overall_score(frame_analyses)
            
            # Generate activity-specific feedback
            activity_feedback = self._generate_activity_feedback(dominant_activity, frame_analyses)
            
            # Generate summary
            summary = self._generate_summary(frame_analyses, dominant_activity, overall_score)
            
            return {
                "activity_detected": dominant_activity,
                "overall_posture_score": overall_score,
                "frame_analyses": frame_analyses,
                "activity_specific_feedback": activity_feedback,
                "summary": summary,
                "total_frames": total_frames,
                "analyzed_frames": analyzed_frames
            }
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return {
                "activity_detected": "error",
                "overall_posture_score": 0,
                "frame_analyses": [],
                "activity_specific_feedback": {},
                "summary": {"error": str(e)},
                "total_frames": 0,
                "analyzed_frames": 0
            }
    
    def _detect_activity_in_frame(self, frame_result: dict) -> str:
        """Detect activity type based on frame analysis"""
        view_type = frame_result.get("view_type", "unknown")
        back_angle = frame_result.get("back_angle", 180)
        
        # Simple activity detection based on posture characteristics
        if back_angle and back_angle < 120:
            return "squat"
        elif view_type == "front" and back_angle and 140 < back_angle < 170:
            return "sitting"
        elif back_angle and back_angle > 170:
            return "standing"
        else:
            return "general"
    
    def _get_activity_specific_feedback(self, activity: str, frame_result: dict, frame: np.ndarray) -> tuple:
        """Get activity-specific issues and suggestions"""
        issues = []
        suggestions = []
        
        back_angle = frame_result.get("back_angle", 180)
        posture_status = frame_result.get("posture_status", "unknown")
        
        if activity == "squat":
            # Squat-specific checks
            if back_angle and back_angle < 150:
                issues.append("Back angle too forward during squat")
                suggestions.append("Keep your chest up and back straight")
            
            # Additional squat checks could include knee-toe alignment
            if posture_status in ["poor", "bad", "very_bad"]:
                issues.append("Poor squat form detected")
                suggestions.append("Ensure knees don't go beyond toes")
        
        elif activity == "sitting":
            # Sitting-specific checks
            if back_angle and back_angle < 160:
                issues.append("Slouching while sitting")
                suggestions.append("Sit up straight with back against chair")
            
            if frame_result.get("view_type") == "side":
                issues.append("Head forward posture detected")
                suggestions.append("Keep monitor at eye level to avoid neck strain")
        
        elif activity == "standing":
            # Standing-specific checks
            if back_angle and back_angle < 170:
                issues.append("Not standing straight")
                suggestions.append("Stand tall with shoulders back")
        
        return issues, suggestions
    
    def _calculate_overall_score(self, frame_analyses: list) -> float:
        """Calculate overall posture score for the video"""
        if not frame_analyses:
            return 0.0
        
        good_frames = sum(1 for frame in frame_analyses if not frame["bad_posture"])
        total_frames = len(frame_analyses)
        
        return round((good_frames / total_frames) * 100, 1)
    
    def _generate_activity_feedback(self, activity: str, frame_analyses: list) -> dict:
        """Generate activity-specific feedback summary"""
        total_frames = len(frame_analyses)
        poor_posture_frames = sum(1 for frame in frame_analyses if frame["bad_posture"])
        
        # Collect common issues
        all_issues = []
        for frame in frame_analyses:
            all_issues.extend(frame["activity_specific_issues"])
        
        # Count issue frequency
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Get most common issues
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        common_issues = [issue for issue, count in common_issues[:5]]
        
        # Activity-specific suggestions
        suggestions = self._get_activity_suggestions(activity)
        
        return {
            "activity": activity,
            "total_frames": total_frames,
            "poor_posture_frames": poor_posture_frames,
            "common_issues": common_issues,
            "improvement_suggestions": suggestions,
            "specific_metrics": {
                "posture_consistency": round((1 - poor_posture_frames / total_frames) * 100, 1) if total_frames > 0 else 0
            }
        }
    
    def _get_activity_suggestions(self, activity: str) -> list:
        """Get general suggestions for specific activities"""
        suggestions_map = {
            "squat": [
                "Keep your chest up and shoulders back",
                "Ensure knees track over toes, not inward",
                "Maintain neutral spine throughout the movement",
                "Go down until thighs are parallel to floor",
                "Push through heels when standing up"
            ],
            "sitting": [
                "Sit with back straight against chair back",
                "Keep feet flat on floor or footrest",
                "Position monitor at eye level",
                "Take breaks every 30 minutes to stand and stretch",
                "Keep shoulders relaxed and down"
            ],
            "standing": [
                "Stand tall with crown of head reaching toward ceiling",
                "Keep shoulders back and down",
                "Engage core muscles lightly",
                "Distribute weight evenly on both feet",
                "Avoid locking knees"
            ],
            "general": [
                "Maintain awareness of your posture throughout activities",
                "Strengthen core muscles to support good posture",
                "Stretch regularly to maintain flexibility",
                "Consider ergonomic adjustments to your workspace"
            ]
        }
        
        return suggestions_map.get(activity, suggestions_map["general"])
    
    def _generate_summary(self, frame_analyses: list, activity: str, overall_score: float) -> dict:
        """Generate summary of video analysis"""
        total_frames = len(frame_analyses)
        if total_frames == 0:
            return {"message": "No frames analyzed"}
        
        poor_frames = sum(1 for frame in frame_analyses if frame["bad_posture"])
        good_frames = total_frames - poor_frames
        
        # Performance rating
        if overall_score >= 80:
            rating = "Excellent"
        elif overall_score >= 60:
            rating = "Good"
        elif overall_score >= 40:
            rating = "Fair"
        else:
            rating = "Needs Improvement"
        
        return {
            "overall_rating": rating,
            "posture_score": overall_score,
            "activity_detected": activity,
            "frames_analyzed": total_frames,
            "good_posture_frames": good_frames,
            "poor_posture_frames": poor_frames,
            "recommendation": self._get_overall_recommendation(overall_score, activity)
        }
    
    def _get_overall_recommendation(self, score: float, activity: str) -> str:
        """Get overall recommendation based on score and activity"""
        if score >= 80:
            return f"Great job! Your {activity} posture is excellent. Keep it up!"
        elif score >= 60:
            return f"Good {activity} posture overall. Focus on consistency and minor improvements."
        elif score >= 40:
            return f"Your {activity} posture needs some work. Practice the suggested improvements."
        else:
            return f"Significant improvement needed in your {activity} posture. Consider professional guidance."

    def _detect_squat_activity(self, landmarks) -> bool:
        """Detect if current pose indicates squatting activity"""
        if not landmarks:
            return False
        
        try:
            # Get relevant landmarks for squat detection
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            
            # Calculate knee angle (simple approximation)
            hip_knee_dist = ((left_hip.x - left_knee.x) ** 2 + (left_hip.y - left_knee.y) ** 2) ** 0.5
            knee_ankle_dist = ((left_knee.x - left_ankle.x) ** 2 + (left_knee.y - left_ankle.y) ** 2) ** 0.5
            
            # Squat typically has bent knees (knee below hip level)
            return left_knee.y > left_hip.y and hip_knee_dist > 0.1
            
        except Exception:
            return False

    def _detect_sitting_activity(self, landmarks) -> bool:
        """Detect if current pose indicates sitting activity"""
        if not landmarks:
            return False
        
        try:
            # Get relevant landmarks
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            
            # Sitting typically has hips and knees at similar height
            hip_knee_height_diff = abs(left_hip.y - left_knee.y)
            
            # Also check if torso is relatively upright
            shoulder_hip_dist = ((left_shoulder.x - left_hip.x) ** 2 + (left_shoulder.y - left_hip.y) ** 2) ** 0.5
            
            return hip_knee_height_diff < 0.15 and shoulder_hip_dist < 0.3
            
        except Exception:
            return False

    def _detect_standing_activity(self, landmarks) -> bool:
        """Detect if current pose indicates standing activity"""
        if not landmarks:
            return False
        
        try:
            # Get relevant landmarks
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP]
            left_knee = landmarks[mp.solutions.pose.PoseLandmark.LEFT_KNEE]
            left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            
            # Standing typically has legs relatively straight
            hip_ankle_dist = abs(left_hip.y - left_ankle.y)
            knee_position = (left_hip.y + left_ankle.y) / 2
            
            # Check if knee is roughly between hip and ankle (straight leg)
            return abs(left_knee.y - knee_position) < 0.1 and hip_ankle_dist > 0.3
            
        except Exception:
            return False

    def _detect_walking_activity(self, landmarks) -> bool:
        """Detect if current pose indicates walking activity"""
        if not landmarks:
            return False
        
        try:
            # Get foot landmarks
            left_ankle = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE]
            
            # Walking typically has one foot ahead of the other
            foot_separation = abs(left_ankle.x - right_ankle.x)
            
            # Also check if one foot is lifted (different y positions)
            foot_height_diff = abs(left_ankle.y - right_ankle.y)
            
            return foot_separation > 0.1 or foot_height_diff > 0.05
            
        except Exception:
            return False
