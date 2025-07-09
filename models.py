from pydantic import BaseModel, Field
from typing import Optional
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
