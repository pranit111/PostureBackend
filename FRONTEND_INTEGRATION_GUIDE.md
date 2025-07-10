# Frontend Integration Guide: Video Posture Analysis API

## 📋 Context and Overview

This guide provides all the necessary information to integrate the **Video Posture Analysis API** into your frontend application. The API analyzes video files for posture detection, activity recognition, and provides activity-specific feedback.

## 🔗 API Endpoint Information

### Base URL
```
http://localhost:8000
```

### Video Analysis Endpoint
```
POST /analyze/video
```

### Content Type
```
multipart/form-data
```

## 📤 Request Format

### File Upload Requirements
- **Supported formats**: MP4, AVI, MOV
- **Maximum file size**: 50MB
- **Field name**: `file`
- **Request type**: `multipart/form-data`

### Example JavaScript/Fetch Request
```javascript
const analyzeVideo = async (videoFile) => {
  const formData = new FormData();
  formData.append('file', videoFile);

  try {
    const response = await fetch('http://localhost:8000/analyze/video', {
      method: 'POST',
      body: formData,
      headers: {
        // Don't set Content-Type header - let browser set it for multipart/form-data
      }
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error analyzing video:', error);
    throw error;
  }
};
```

### Example React Hook
```javascript
import { useState } from 'react';

const useVideoAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [error, setError] = useState(null);

  const analyzeVideo = async (videoFile) => {
    setIsAnalyzing(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', videoFile);

      const response = await fetch('http://localhost:8000/analyze/video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.reason || 'Analysis failed');
      }

      const result = await response.json();
      setAnalysisResult(result);
      return result;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setIsAnalyzing(false);
    }
  };

  return { analyzeVideo, isAnalyzing, analysisResult, error };
};
```

## 📥 Response Format

### Complete Response Structure
```typescript
interface VideoAnalysisResponse {
  activity_detected: string;           // "squat" | "sitting" | "standing" | "walking" | "general"
  overall_posture_score: number;       // 0-100 score
  frame_analyses: FrameAnalysis[];     // Array of frame-by-frame results
  activity_specific_feedback: ActivityFeedback;
  summary: Summary;
  processing_time: number;             // Seconds
  total_frames: number;               // Total frames in video
  analyzed_frames: number;            // Frames actually analyzed
}

interface FrameAnalysis {
  frame_number: number;
  timestamp: number;                  // Seconds
  bad_posture: boolean;
  back_angle?: number;               // Degrees
  posture_status: string;            // "excellent" | "good" | "fair" | "poor" | "bad" | "very_bad"
  activity_specific_issues: string[];
  suggestions: string[];
}

interface ActivityFeedback {
  activity: string;
  total_frames: number;
  poor_posture_frames: number;
  common_issues: string[];
  improvement_suggestions: string[];
  specific_metrics: Record<string, number>;
}

interface Summary {
  total_duration: number;
  good_posture_percentage: number;
  main_activity: string;
  key_recommendations: string[];
}
```

## 🎯 Activity-Specific Analysis Rules

### 🏋️ Squat Analysis
- **Detection criteria**: Back angle < 120°
- **Issues flagged**:
  - Back angle < 150° (poor form)
  - Knee tracking beyond toes
  - Insufficient depth
- **Suggestions**:
  - "Keep your chest up and back straight"
  - "Ensure knees don't go beyond toes"
  - "Focus on controlled movement"

### 💺 Sitting Analysis
- **Detection criteria**: Front view with back angle 140-170°
- **Issues flagged**:
  - Back angle < 160° (slouching)
  - Forward head posture
  - Poor spinal alignment
- **Suggestions**:
  - "Sit up straight with back against chair"
  - "Keep monitor at eye level to avoid neck strain"
  - "Take breaks to stand and stretch"

### 🚶 Standing Analysis
- **Detection criteria**: Back angle > 170°
- **Issues flagged**:
  - Back angle < 170° (not standing straight)
  - Forward head posture
  - Shoulder misalignment
- **Suggestions**:
  - "Stand tall with shoulders back"
  - "Distribute weight evenly on both feet"
  - "Keep head aligned with spine"

## 🖥️ Frontend Implementation Examples

### File Upload Component (React)
```jsx
import React, { useRef, useState } from 'react';

const VideoUploader = ({ onAnalysisComplete }) => {
  const fileInputRef = useRef(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileSelect = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime'];
    if (!validTypes.includes(file.type)) {
      alert('Please select a valid video file (MP4, AVI, MOV)');
      return;
    }

    // Validate file size (50MB)
    if (file.size > 50 * 1024 * 1024) {
      alert('File size must be less than 50MB');
      return;
    }

    setIsUploading(true);
    
    try {
      const result = await analyzeVideo(file);
      onAnalysisComplete(result);
    } catch (error) {
      alert('Analysis failed: ' + error.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="video-uploader">
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/avi,video/quicktime"
        onChange={handleFileSelect}
        disabled={isUploading}
        style={{ display: 'none' }}
      />
      
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={isUploading}
        className="upload-button"
      >
        {isUploading ? 'Analyzing...' : 'Upload Video for Analysis'}
      </button>
      
      {isUploading && (
        <div className="upload-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
          <p>Analyzing your video...</p>
        </div>
      )}
    </div>
  );
};
```

### Results Display Component (React)
```jsx
const VideoAnalysisResults = ({ analysisResult }) => {
  if (!analysisResult) return null;

  const {
    activity_detected,
    overall_posture_score,
    frame_analyses,
    activity_specific_feedback,
    summary
  } = analysisResult;

  const getScoreColor = (score) => {
    if (score >= 80) return '#4CAF50'; // Green
    if (score >= 60) return '#FF9800'; // Orange
    return '#F44336'; // Red
  };

  const getActivityIcon = (activity) => {
    const icons = {
      squat: '🏋️',
      sitting: '💺',
      standing: '🚶',
      walking: '🚶‍♂️',
      general: '🤸'
    };
    return icons[activity] || '📊';
  };

  return (
    <div className="analysis-results">
      {/* Overall Score */}
      <div className="score-section">
        <h2>Overall Posture Score</h2>
        <div 
          className="score-circle"
          style={{ 
            background: `conic-gradient(${getScoreColor(overall_posture_score)} ${overall_posture_score}%, #e0e0e0 0)` 
          }}
        >
          <span className="score-text">{overall_posture_score.toFixed(1)}</span>
        </div>
      </div>

      {/* Activity Detection */}
      <div className="activity-section">
        <h3>
          {getActivityIcon(activity_detected)} 
          Detected Activity: {activity_detected.charAt(0).toUpperCase() + activity_detected.slice(1)}
        </h3>
      </div>

      {/* Activity-Specific Feedback */}
      <div className="feedback-section">
        <h3>Activity-Specific Feedback</h3>
        
        {activity_specific_feedback.common_issues.length > 0 && (
          <div className="issues">
            <h4>⚠️ Issues Detected:</h4>
            <ul>
              {activity_specific_feedback.common_issues.map((issue, index) => (
                <li key={index} className="issue-item">{issue}</li>
              ))}
            </ul>
          </div>
        )}

        {activity_specific_feedback.improvement_suggestions.length > 0 && (
          <div className="suggestions">
            <h4>💡 Improvement Suggestions:</h4>
            <ul>
              {activity_specific_feedback.improvement_suggestions.map((suggestion, index) => (
                <li key={index} className="suggestion-item">{suggestion}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Summary Stats */}
      <div className="summary-section">
        <h3>Summary</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Duration:</span>
            <span className="stat-value">{summary.total_duration.toFixed(1)}s</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Good Posture:</span>
            <span className="stat-value">{summary.good_posture_percentage.toFixed(1)}%</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Frames Analyzed:</span>
            <span className="stat-value">{analysisResult.analyzed_frames}/{analysisResult.total_frames}</span>
          </div>
        </div>
      </div>

      {/* Frame Timeline */}
      <div className="timeline-section">
        <h3>Posture Timeline</h3>
        <div className="frame-timeline">
          {frame_analyses.map((frame, index) => (
            <div
              key={index}
              className={`timeline-point ${frame.bad_posture ? 'bad' : 'good'}`}
              title={`Frame ${frame.frame_number}: ${frame.posture_status} posture`}
              style={{
                left: `${(frame.timestamp / summary.total_duration) * 100}%`
              }}
            />
          ))}
        </div>
      </div>

      {/* Key Recommendations */}
      {summary.key_recommendations && summary.key_recommendations.length > 0 && (
        <div className="recommendations-section">
          <h3>🎯 Key Recommendations</h3>
          <ul>
            {summary.key_recommendations.map((rec, index) => (
              <li key={index} className="recommendation-item">{rec}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
```

## 🎨 CSS Styling Examples

```css
.analysis-results {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.score-section {
  text-align: center;
  margin-bottom: 30px;
}

.score-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 20px auto;
  position: relative;
}

.score-circle::before {
  content: '';
  position: absolute;
  width: 80px;
  height: 80px;
  background: white;
  border-radius: 50%;
  z-index: 1;
}

.score-text {
  font-size: 24px;
  font-weight: bold;
  z-index: 2;
}

.activity-section {
  background: #f5f5f5;
  padding: 15px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.issues {
  background: #ffebee;
  padding: 15px;
  border-left: 4px solid #f44336;
  margin-bottom: 15px;
}

.suggestions {
  background: #e8f5e8;
  padding: 15px;
  border-left: 4px solid #4caf50;
  margin-bottom: 15px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-top: 15px;
}

.stat-item {
  background: white;
  padding: 15px;
  border-radius: 6px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  display: flex;
  justify-content: space-between;
}

.frame-timeline {
  position: relative;
  height: 20px;
  background: #e0e0e0;
  border-radius: 10px;
  margin-top: 10px;
}

.timeline-point {
  position: absolute;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  top: 4px;
  transform: translateX(-50%);
}

.timeline-point.good {
  background: #4caf50;
}

.timeline-point.bad {
  background: #f44336;
}

.upload-button {
  background: #2196f3;
  color: white;
  border: none;
  padding: 12px 24px;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  transition: background 0.3s;
}

.upload-button:hover:not(:disabled) {
  background: #1976d2;
}

.upload-button:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.upload-progress {
  margin-top: 15px;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: #2196f3;
  transition: width 0.3s ease;
}
```

## 🚫 Error Handling

### Common Error Responses
```javascript
// File type validation error (400)
{
  "detail": "Unsupported video file type. Please upload MP4, AVI, or MOV files."
}

// File size validation error (400)
{
  "detail": "Video file too large. Maximum size is 50MB."
}

// Processing error (422)
{
  "bad_posture": true,
  "reason": "Failed to process video file.",
  "error_code": "VIDEO_PROCESSING_ERROR"
}

// Internal server error (500)
{
  "bad_posture": true,
  "reason": "Internal server error during video analysis.",
  "error_code": "INTERNAL_ERROR"
}
```

### Error Handling Implementation
```javascript
const handleAnalysisError = (error, response) => {
  if (response?.status === 400) {
    // File validation error
    return "Please check your file format and size.";
  } else if (response?.status === 422) {
    // Processing error
    return "Unable to process video. Please try a different file.";
  } else if (response?.status === 500) {
    // Server error
    return "Server error. Please try again later.";
  } else {
    // Network or other errors
    return "Connection error. Please check your internet connection.";
  }
};
```

## 🔧 Testing and Validation

### API Health Check
```javascript
const checkAPIHealth = async () => {
  try {
    const response = await fetch('http://localhost:8000/health');
    const health = await response.json();
    return health.status === 'healthy';
  } catch (error) {
    return false;
  }
};
```

### File Validation
```javascript
const validateVideoFile = (file) => {
  const validTypes = ['video/mp4', 'video/avi', 'video/quicktime'];
  const maxSize = 50 * 1024 * 1024; // 50MB

  if (!validTypes.includes(file.type)) {
    throw new Error('Invalid file type. Please upload MP4, AVI, or MOV files.');
  }

  if (file.size > maxSize) {
    throw new Error('File too large. Maximum size is 50MB.');
  }

  return true;
};
```

## 📱 Mobile Considerations

### Responsive Design
```css
@media (max-width: 768px) {
  .analysis-results {
    padding: 15px;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .score-circle {
    width: 100px;
    height: 100px;
  }
  
  .upload-button {
    width: 100%;
    padding: 15px;
  }
}
```

### File Input for Mobile
```jsx
<input
  type="file"
  accept="video/*"
  capture="environment" // For camera recording on mobile
  onChange={handleFileSelect}
/>
```

## 🚀 Performance Optimization

### Loading States
```jsx
const LoadingSpinner = () => (
  <div className="loading-spinner">
    <div className="spinner"></div>
    <p>Analyzing your video...</p>
    <small>This may take a few moments depending on video length</small>
  </div>
);
```

### Progress Tracking
```javascript
// For large files, you might want to implement progress tracking
const uploadWithProgress = (file, onProgress) => {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('file', file);

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable) {
        const percentComplete = (e.loaded / e.total) * 100;
        onProgress(percentComplete);
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status === 200) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        reject(new Error(`Upload failed: ${xhr.status}`));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new Error('Upload failed'));
    });

    xhr.open('POST', 'http://localhost:8000/analyze/video');
    xhr.send(formData);
  });
};
```

## 📋 Integration Checklist

- [ ] Set up file upload component with proper validation
- [ ] Implement error handling for all response codes
- [ ] Add loading states and progress indicators
- [ ] Create results display components
- [ ] Style components for desktop and mobile
- [ ] Test with various video files and sizes
- [ ] Implement API health checks
- [ ] Add user feedback for successful/failed uploads
- [ ] Consider implementing file preview before upload
- [ ] Add analytics/tracking for usage patterns

## 🎯 Ready to Integrate!

This guide provides everything needed to integrate the video posture analysis API into your frontend application. The API is production-ready and provides comprehensive posture analysis with activity-specific feedback.

For questions or issues, check the API documentation at `http://localhost:8000/docs` when the server is running.
