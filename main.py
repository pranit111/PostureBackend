from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uvicorn
from typing import Optional

from models import PostureAnalysisResponse
from posture_analyzer import PostureAnalyzer, preprocess_image, validate_image
from utils import (
    convert_upload_to_image,
    validate_file_type,
    validate_file_size,
    create_error_response,
    create_success_response,
    log_analysis_result,
    calculate_processing_stats
)

# Initialize FastAPI app
app = FastAPI(
    title="Posture Detection API",
    description="Real-time posture analysis API using MediaPipe and OpenCV",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the posture analyzer
analyzer = PostureAnalyzer()

# Global stats for monitoring
processing_stats = {
    "total_requests": 0,
    "successful_analyses": 0,
    "failed_analyses": 0,
    "average_processing_time": 0.0
}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Posture Detection API",
        "version": "1.0.0",
        "endpoints": {
            "analyze_frame": "/analyze/frame",
            "health_check": "/health",
            "stats": "/stats"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "analyzer_status": "initialized"
    }


@app.get("/stats")
async def get_stats():
    """Get processing statistics"""
    return {
        "processing_stats": processing_stats,
        "timestamp": time.time()
    }


@app.post("/analyze/frame", response_model=PostureAnalysisResponse)
async def analyze_frame(file: UploadFile = File(...)):
    """
    Analyze a single frame for posture detection
    
    Args:
        file: Uploaded image file
        
    Returns:
        PostureAnalysisResponse with analysis results
    """
    start_time = time.time()
    processing_stats["total_requests"] += 1
    
    try:
        # Validate file type
        if not validate_file_type(file.filename):
            processing_stats["failed_analyses"] += 1
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload JPG, PNG, or other image formats."
            )
        
        # Read file content
        file_content = await file.read()
        
        # Validate file size (10MB limit)
        if not validate_file_size(len(file_content)):
            processing_stats["failed_analyses"] += 1
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        # Convert to OpenCV image
        image = convert_upload_to_image(file_content)
        if image is None:
            processing_stats["failed_analyses"] += 1
            return JSONResponse(
                status_code=422,
                content=create_error_response(
                    "Failed to process image. Please ensure it's a valid image file.",
                    "IMAGE_PROCESSING_ERROR"
                )
            )
        
        # Validate image quality
        if not validate_image(image):
            processing_stats["failed_analyses"] += 1
            return JSONResponse(
                status_code=422,
                content=create_error_response(
                    "Image quality too low for pose detection. Please ensure good lighting and image size.",
                    "IMAGE_QUALITY_ERROR"
                )
            )
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Analyze posture
        result = analyzer.analyze_frame(processed_image)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Update stats
        processing_stats["successful_analyses"] += 1
        processing_stats["average_processing_time"] = (
            (processing_stats["average_processing_time"] * (processing_stats["successful_analyses"] - 1) + processing_time) /
            processing_stats["successful_analyses"]
        )
        
        # Log result
        log_analysis_result(result, processing_time)
        
        # Return response
        return PostureAnalysisResponse(
            bad_posture=result["bad_posture"],
            reason=result.get("reason"),
            back_angle=result.get("back_angle"),
            angle=result.get("angle"),
            view_type=result.get("view_type"),
            analysis_method=result.get("analysis_method"),
            posture_status=result.get("posture_status")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_stats["failed_analyses"] += 1
        print(f"Unexpected error in analyze_frame: {e}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                "Internal server error during analysis.",
                "INTERNAL_ERROR"
            )
        )


@app.post("/analyze/frame/detailed")
async def analyze_frame_detailed(file: UploadFile = File(...)):
    """
    Analyze a single frame for posture detection with detailed metrics
    
    Args:
        file: Uploaded image file
        
    Returns:
        Detailed analysis results including additional metrics
    """
    start_time = time.time()
    
    try:
        # Similar validation as above
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload JPG, PNG, or other image formats."
            )
        
        file_content = await file.read()
        
        if not validate_file_size(len(file_content)):
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        image = convert_upload_to_image(file_content)
        if image is None:
            return JSONResponse(
                status_code=422,
                content=create_error_response(
                    "Failed to process image. Please ensure it's a valid image file.",
                    "IMAGE_PROCESSING_ERROR"
                )
            )
        
        if not validate_image(image):
            return JSONResponse(
                status_code=422,
                content=create_error_response(
                    "Image quality too low for pose detection. Please ensure good lighting and image size.",
                    "IMAGE_QUALITY_ERROR"
                )
            )
        
        processed_image = preprocess_image(image)
        
        # Get basic analysis
        result = analyzer.analyze_frame(processed_image)
        
        # Get additional metrics (if pose was detected)
        additional_metrics = {}
        if not result.get("reason", "").startswith("No pose detected"):
            # This would require extracting landmarks again - for now, return basic info
            additional_metrics = {
                "head_forward": 0.0,
                "shoulder_alignment": 0.0,
                "confidence": 85.0
            }
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Enhanced response
        detailed_response = {
            **result,
            "additional_metrics": additional_metrics,
            "processing_stats": calculate_processing_stats(start_time, end_time)
        }
        
        log_analysis_result(detailed_response, processing_time)
        
        return JSONResponse(content=detailed_response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in analyze_frame_detailed: {e}")
        return JSONResponse(
            status_code=500,
            content=create_error_response(
                "Internal server error during detailed analysis.",
                "INTERNAL_ERROR"
            )
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
