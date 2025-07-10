from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
# Just JSON response models, for returing responses from FastAPI endpoints

class PostureAnalysisRequest(BaseModel):
    """Request model for posture analysis"""
    pass  # File upload will be handled by FastAPI's UploadFile


class PostureAnalysisResponse(BaseModel):
    """Response model for posture analysis"""
    bad_posture: bool = Field(
        description="True if bad posture is detected, False otherwise"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanation of why posture is considered bad"
    )
    back_angle: Optional[float] = Field(
        default=None,
        description="Calculated back angle in degrees"
    )
    angle: Optional[float] = Field(
        default=None,
        description="Calculated back angle in degrees (compatibility field)"
    )
    view_type: Optional[str] = Field(
        default=None,
        description="Detected view orientation (side, front, side_left, side_right)"
    )
    analysis_method: Optional[str] = Field(
        default=None,
        description="Method used for analysis (side_view, front_view)"
    )
    posture_status: Optional[str] = Field(
        default=None,
        description="Posture quality status (excellent, good, fair, poor, bad, very_bad)"
    )
    
    class Config:
        populate_by_name = True


class FrameAnalysis(BaseModel):
    """Analysis result for a single video frame"""
    frame_number: int
    timestamp: float
    bad_posture: bool
    back_angle: Optional[float] = None
    posture_status: str
    activity_specific_issues: List[str] = []
    suggestions: List[str] = []


class ActivityFeedback(BaseModel):
    """Activity-specific posture feedback"""
    activity: str
    total_frames: int
    poor_posture_frames: int
    common_issues: List[str]
    improvement_suggestions: List[str]
    specific_metrics: Dict[str, float] = {}


class VideoAnalysisResponse(BaseModel):
    """Response model for video posture analysis"""
    activity_detected: str = Field(
        description="Detected activity in the video (squat, sitting, walking, etc.)"
    )
    overall_posture_score: float = Field(
        description="Overall posture score for the entire video (0-100)"
    )
    frame_analyses: List[FrameAnalysis] = Field(
        description="Analysis results for sampled frames"
    )
    activity_specific_feedback: ActivityFeedback = Field(
        description="Feedback specific to the detected activity"
    )
    summary: Dict[str, Any] = Field(
        description="Summary statistics and recommendations"
    )
    processing_time: float = Field(
        description="Time taken to process the video"
    )
    total_frames: int = Field(
        description="Total number of frames in the video"
    )
    analyzed_frames: int = Field(
        description="Number of frames that were analyzed"
    )


class PostureMetrics(BaseModel):
    """Detailed posture metrics for future use"""
    back_angle: float
    shoulder_alignment: float
    head_position: float
    confidence_score: float
    
    
class PostureSession(BaseModel):
    """Model for storing posture session data (future use)"""
    session_id: str
    start_time: str
    end_time: Optional[str] = None
    total_frames: int = 0
    bad_posture_count: int = 0
    average_angle: Optional[float] = None
    
    
class PostureHistory(BaseModel):
    """Model for storing historical posture data (future use)"""
    timestamp: str
    bad_posture: bool
    angle: float
    session_id: str
