# Posture Detection System

A comprehensive posture analysis system with real-time camera scanning and video analysis capabilities. The system detects activities (squats, sitting, standing, walking) and provides activity-specific posture feedback and improvement suggestions.

##  Live Demo

**Deployed Application:** https://posture-backend-k51h.vercel.app/

## Note that the file analyzer doesn't work as expexted on deployed as it is deployed on a low powered server and it exceeds the ram of the hosted server it works perfectly file on local

**Demo Video:** https://drive.google.com/file/d/1cWiF8n8ltY8yRRbbeVgiG7HNLXn3PzSJ/view?usp=sharing

##  Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **MediaPipe** - Google's framework for pose detection and analysis
- **OpenCV** - Computer vision library for image processing
- **Pydantic** - Data validation using Python type hints
- **Uvicorn** - ASGI server for FastAPI
- **Python 3.8+** - Programming language

### Frontend
- **React 18** - Frontend JavaScript library
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide React** - Beautiful icon library
- **Axios** - HTTP client for API requests
- **React Webcam** - Webcam component for React
- **Vite** - Fast build tool and dev server

### Deployment
- **Backend:** [Add your backend deployment platform]
- **Frontend:** [Add your frontend deployment platform]
## ✨ Features

### Real-Time Analysis
- Live camera feed posture scanning
- Instant feedback and posture status
- Real-time posture score and recommendations
- Multiple view detection (side, front view)

### Video Analysis
- Upload and analyze video files (MP4, AVI, MOV)
- Activity detection (squats, sitting, standing, walking)
- Frame-by-frame posture analysis
- Activity-specific feedback and suggestions
- Overall posture scoring (0-100%)

### Posture Detection
- **Forward head posture** detection
- **Shoulder misalignment** analysis
- **Spine slouching** identification
- **Back angle** calculation and assessment
- **Activity-specific** posture rules:
  - **Squats:** Knee-toe alignment, back angle monitoring
  - **Sitting:** Slouching detection, neck angle analysis
  - **Standing:** Spine alignment, head position
  - **Walking:** Dynamic posture assessment

### User Experience
- Clean, responsive web interface
- Tabbed navigation (Live vs Video analysis)
- Real-time feedback with visual indicators
- Detailed improvement suggestions
- Progress tracking and metrics

## 🏗️ Project Structure

```
posture-detection-system/
├── frontend/                    # React frontend application
│   ├── src/
│   │   ├── App.tsx             # Main application component
│   │   ├── index.tsx           # Application entry point
│   │   └── index.css           # Global styles
│   ├── package.json            # Frontend dependencies
│   └── README.md               # Frontend setup guide
├── backend/                     # FastAPI backend application
│   ├── main.py                 # FastAPI application and endpoints
│   ├── models.py               # Pydantic data models
│   ├── posture_analyzer.py     # Core posture analysis logic
│   ├── utils.py                # Utility functions
│   ├── requirements.txt        # Backend dependencies
│   └── test_video_endpoint.py  # API testing script
└── README.md                   # This file
```

## 🚀 Setup Instructions

### Prerequisites
- **Python 3.8+** installed on your system
- **Node.js 16+** and **npm** for frontend
- **Git** for cloning the repository

### Backend Setup
Github url:https://github.com/pranit111/PostureBackend
1. **Clone the repository:**
```bash
git clone https://github.com/pranit111/PostureBackend
cd posture-detection-system/backend
```

2. **Create virtual environment:**
```bash
python -m venv env

# On Windows:
env\Scripts\activate

# On macOS/Linux:
source env/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the FastAPI server:**
```bash
python main.py
```

5. **Verify backend is running:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health check: http://localhost:8000/health

### Frontend Setup
Github Url:https://github.com/pranit111/PostureFront


1. **Install dependencies:**
```bash
npm install
```

2. **Start development server:**
```bash
npm run dev
```

3. **Access the application:**
   - Frontend: http://localhost:5173
   - The app will automatically connect to the backend API

### Environment Configuration

**Backend Environment Variables:**
```bash
# Optional: Create .env file in backend directory
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE_MB=50
```

**Frontend Environment Variables:**
```bash
# Optional: Create .env file in frontend directory
VITE_API_BASE_URL=http://localhost:8000
```

## 📡 API Endpoints

### Frame Analysis
- **POST /analyze/frame** - Analyze single frame for posture
- **POST /analyze/frame/detailed** - Detailed frame analysis with metrics

### Video Analysis
- **POST /analyze/video** - Upload and analyze video file
  - Supports: MP4, AVI, MOV (max 50MB)
  - Returns: Activity detection, posture score, frame analysis, suggestions

### System Endpoints
- **GET /** - API information and available endpoints
- **GET /health** - Health check and system status
- **GET /stats** - Processing statistics and performance metrics

### Example API Response (Video Analysis)
```json
{
  "activity_detected": "squat",
  "overall_posture_score": 75.5,
  "frame_analyses": [
    {
      "frame_number": 10,
      "timestamp": 0.33,
      "bad_posture": true,
      "back_angle": 145.2,
      "posture_status": "poor",
      "activity_specific_issues": ["Back angle too forward during squat"],
      "suggestions": ["Keep your chest up and back straight"]
    }
  ],
  "activity_specific_feedback": {
    "activity": "squat",
    "total_frames": 150,
    "poor_posture_frames": 45,
    "common_issues": ["Back angle too forward during squat"],
    "improvement_suggestions": ["Keep your chest up and back straight"],
    "specific_metrics": {"posture_consistency": 70.0}
  },
  "summary": {
    "overall_rating": "Good",
    "recommendation": "Good squat posture overall. Focus on consistency."
  }
}
```

## 🔧 How It Works

### 1. Image/Video Processing
- Uploaded media is preprocessed for optimal pose detection
- MediaPipe extracts pose landmarks from frames
- Image quality validation and error handling

### 2. Pose Analysis
- **Back angle calculation** using shoulder, hip, and knee positions
- **View detection** (side view vs front view) for appropriate analysis
- **Posture assessment** with configurable thresholds

### 3. Activity Detection
- **Frame-by-frame analysis** for activity classification
- **Activity-specific rules:**
  - **Squats:** Back angle < 150° flagged, knee-toe alignment
  - **Sitting:** Slouching detection, neck angle > 30° flagged
  - **Standing:** Spine alignment monitoring
  - **Walking:** Dynamic posture assessment

### 4. Feedback Generation
- **Real-time suggestions** based on detected issues
- **Activity-specific recommendations** for improvement
- **Overall scoring** and progress tracking

## ⚙️ Configuration

The posture analysis can be customized by modifying parameters in `posture_analyzer.py`:

```python
# Detection confidence thresholds
POSE_DETECTION_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE = 0.5

# Posture angle thresholds
BAD_POSTURE_ANGLE_THRESHOLD = 150  # degrees
EXCELLENT_POSTURE_THRESHOLD = 170
FAIR_POSTURE_THRESHOLD = 160

# Activity-specific thresholds
SQUAT_BACK_ANGLE_THRESHOLD = 150
SITTING_NECK_ANGLE_THRESHOLD = 30
```

## 🚀 Deployment

### Backend Deployment
The FastAPI backend can be deployed on:
- **Render** (recommended for free tier)
- **Heroku**
- **Railway**
- **AWS/GCP/Azure**

### Frontend Deployment
The React frontend can be deployed on:
- **Vercel** (recommended)
- **Netlify**
- **GitHub Pages**
- **Firebase Hosting**

### Environment Variables for Production
```bash
# Backend
CORS_ORIGINS=["https://your-frontend-domain.com"]
MAX_FILE_SIZE_MB=50

# Frontend
VITE_API_BASE_URL=https://your-backend-api.com
```

## 📊 Performance & Features

### Performance Metrics
- **Frame Analysis:** ~0.1-0.3 seconds per frame
- **Video Processing:** ~2-5 seconds per minute of video
- **Concurrent Requests:** Supports multiple simultaneous analyses
- **Memory Efficient:** Optimized for real-time processing
- **File Size Limits:** 10MB for images, 50MB for videos

### Error Handling
The system includes comprehensive error handling for:
- Invalid file types and formats
- File size limit violations
- Image/video processing errors
- Pose detection failures
- Network connectivity issues
- Malformed requests

### Browser Compatibility
- **Chrome/Edge:** Full support including webcam
- **Firefox:** Full support including webcam
- **Safari:** Full support including webcam
- **Mobile browsers:** Responsive design, limited webcam support

## 🎯 Usage Examples

### Live Camera Analysis
1. Open the application
2. Click "Live Analysis" tab
3. Allow camera permissions
4. Click "Start Scanning"
5. Receive real-time posture feedback

### Video Analysis
1. Click "Video Analysis" tab
2. Upload a video file (MP4, AVI, MOV)
3. Click "Analyze Video"
4. View detailed results:
   - Activity detection
   - Overall posture score
   - Frame-by-frame analysis
   - Improvement suggestions




**Made by PranitBhopi **
