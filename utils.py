import io
import os
import tempfile
import uuid
import time
import cv2
import numpy as np
from PIL import Image
from typing import Optional


def convert_upload_to_image(file_content: bytes) -> Optional[np.ndarray]:
    """
    Convert uploaded file content to OpenCV image
    
    Args:
        file_content: Raw file content as bytes
        
    Returns:
        OpenCV image as numpy array or None if conversion fails
    """
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    
    except Exception as e:
        print(f"Error converting upload to image: {e}")
        return None


def validate_file_type(filename: str) -> bool:
    """
    Validate if uploaded file is a supported image type
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        True if file type is supported, False otherwise
    """
    if not filename:
        return False
    
    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    file_extension = filename.lower().split('.')[-1]
    
    return f'.{file_extension}' in supported_extensions


def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """
    Validate file size
    
    Args:
        file_size: Size of the file in bytes
        max_size_mb: Maximum allowed size in MB
        
    Returns:
        True if file size is acceptable, False otherwise
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


def create_error_response(message: str, error_code: str = "PROCESSING_ERROR") -> dict:
    """
    Create standardized error response
    
    Args:
        message: Error message
        error_code: Error code for categorization
        
    Returns:
        Error response dictionary
    """
    return {
        "bad_posture": True,
        "reason": message,
        "angle": None,
        "error_code": error_code
    }


def create_success_response(bad_posture: bool, reason: Optional[str], angle: Optional[float]) -> dict:
    """
    Create standardized success response
    
    Args:
        bad_posture: Whether bad posture was detected
        reason: Reason for bad posture (if any)
        angle: Calculated back angle
        
    Returns:
        Success response dictionary
    """
    response = {
        "bad_posture": bad_posture,
        "angle": angle
    }
    
    if reason:
        response["reason"] = reason
    
    return response


def log_analysis_result(result: dict, processing_time: float = 0.0):
    """
    Log analysis result for debugging/monitoring
    
    Args:
        result: Analysis result dictionary
        processing_time: Time taken to process the request
    """
    print(f"Analysis completed in {processing_time:.2f}s")
    print(f"Bad posture: {result.get('bad_posture', 'Unknown')}")
    if result.get('angle'):
        print(f"Back angle: {result.get('angle')}°")
    if result.get('reason'):
        print(f"Reason: {result.get('reason')}")
    print("---")


def calculate_processing_stats(start_time: float, end_time: float) -> dict:
    """
    Calculate processing statistics
    
    Args:
        start_time: Start time of processing
        end_time: End time of processing
        
    Returns:
        Statistics dictionary
    """
    processing_time = end_time - start_time
    
    return {
        "processing_time": round(processing_time, 3),
        "fps_potential": round(1 / processing_time, 2) if processing_time > 0 else 0
    }


def validate_video_file_type(filename: Optional[str]) -> bool:
    """
    Validate if the uploaded file is an allowed video type
    
    Args:
        filename: Name of the uploaded file
        
    Returns:
        True if file type is allowed, False otherwise
    """
    if not filename:
        return False
    
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    file_extension = filename.lower().split('.')[-1]
    
    return f'.{file_extension}' in allowed_extensions


def save_temp_video(file_content: bytes, filename: str) -> Optional[str]:
    """
    Save uploaded video content to a temporary file
    
    Args:
        file_content: Raw video file bytes
        filename: Original filename
        
    Returns:
        Path to temporary file or None if failed
    """
    try:
        # Create temporary directory if it doesn't exist
        temp_dir = tempfile.gettempdir()
        
        # Generate unique filename
        file_extension = filename.lower().split('.')[-1] if '.' in filename else 'mp4'
        temp_filename = f"video_{uuid.uuid4().hex}.{file_extension}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Write file content
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(file_content)
        
        return temp_path
        
    except Exception as e:
        print(f"Error saving temporary video: {e}")
        return None


def cleanup_temp_file(file_path: str) -> None:
    """
    Clean up temporary file
    
    Args:
        file_path: Path to the temporary file to delete
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error cleaning up temporary file {file_path}: {e}")


def log_video_analysis_result(result: dict, processing_time: float) -> None:
    """
    Log the video analysis result for monitoring and debugging
    
    Args:
        result: Video analysis result dictionary
        processing_time: Time taken to process the video
    """
    try:
        activity = result.get("activity_detected", "unknown")
        score = result.get("overall_posture_score", 0)
        total_frames = result.get("total_frames", 0)
        analyzed_frames = result.get("analyzed_frames", 0)
        
        print(f"[VIDEO ANALYSIS] Activity: {activity} | "
              f"Score: {score:.1f}% | Frames: {analyzed_frames}/{total_frames} | "
              f"Time: {processing_time:.3f}s")
              
    except Exception as e:
        print(f"Error logging video analysis result: {e}")
