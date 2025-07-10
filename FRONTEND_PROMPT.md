## 🎯 Frontend Integration Prompt for Video Posture Analysis API

**CONTEXT**: Integrate a video posture analysis API endpoint into the frontend that analyzes uploaded videos for posture detection and provides activity-specific feedback.

### 🔗 API DETAILS
- **Endpoint**: `POST http://localhost:8000/analyze/video`
- **Content-Type**: `multipart/form-data`
- **File field name**: `file`
- **Supported formats**: MP4, AVI, MOV (max 50MB)

### 📥 RESPONSE STRUCTURE
```typescript
interface VideoAnalysisResponse {
  activity_detected: "squat" | "sitting" | "standing" | "walking" | "general";
  overall_posture_score: number; // 0-100
  frame_analyses: Array<{
    frame_number: number;
    timestamp: number;
    bad_posture: boolean;
    back_angle?: number;
    posture_status: "excellent" | "good" | "fair" | "poor" | "bad" | "very_bad";
    activity_specific_issues: string[];
    suggestions: string[];
  }>;
  activity_specific_feedback: {
    activity: string;
    total_frames: number;
    poor_posture_frames: number;
    common_issues: string[];
    improvement_suggestions: string[];
    specific_metrics: Record<string, number>;
  };
  summary: {
    total_duration: number;
    good_posture_percentage: number;
    main_activity: string;
    key_recommendations: string[];
  };
  processing_time: number;
  total_frames: number;
  analyzed_frames: number;
}
```

### 🎯 ACTIVITY-SPECIFIC FEATURES TO HIGHLIGHT
1. **Squat Analysis**: Flags back angle <150°, knee tracking, form issues
2. **Sitting Analysis**: Detects slouching, forward head posture, spinal alignment
3. **Standing Analysis**: Monitors posture alignment, head position, shoulder alignment

### 💻 BASIC FETCH IMPLEMENTATION
```javascript
const analyzeVideo = async (videoFile) => {
  const formData = new FormData();
  formData.append('file', videoFile);
  
  const response = await fetch('http://localhost:8000/analyze/video', {
    method: 'POST',
    body: formData
  });
  
  if (!response.ok) {
    throw new Error(`Analysis failed: ${response.status}`);
  }
  
  return await response.json();
};
```

### 🖥️ UI COMPONENTS NEEDED
1. **File Upload Component**: Video file input with validation
2. **Loading State**: Progress indicator during analysis
3. **Results Display**: 
   - Overall posture score (0-100 with color coding)
   - Detected activity with icon
   - Activity-specific issues and suggestions
   - Frame-by-frame timeline visualization
   - Summary statistics
4. **Error Handling**: User-friendly error messages

### 🎨 KEY UI ELEMENTS
- **Score Circle**: Visual representation of overall_posture_score
- **Activity Badge**: Show activity_detected with appropriate icon
- **Issues List**: Display activity_specific_feedback.common_issues
- **Suggestions List**: Show improvement_suggestions
- **Timeline**: Visualize frame_analyses as a timeline with good/bad indicators
- **Stats Cards**: Show summary statistics

### 🚫 ERROR HANDLING
- **400**: File type/size validation errors
- **422**: Video processing errors
- **500**: Server errors
- Validate file type and size before upload
- Show appropriate user messages for each error type

### 📱 REQUIREMENTS
- File input with video/* accept
- Progress indicator during upload/analysis
- Responsive design for mobile/desktop
- Error boundaries for failed requests
- Loading states with informative messages

### 🎯 USER FLOW
1. User selects video file
2. Validate file (type/size)
3. Show upload progress
4. Display analysis results with:
   - Overall score visualization
   - Activity detection
   - Specific feedback and suggestions
   - Timeline of posture quality
5. Handle errors gracefully

**GOAL**: Create an intuitive interface that makes video posture analysis accessible and actionable for users, with clear feedback and improvement suggestions based on detected activities.
