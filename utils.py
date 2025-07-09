import io
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
