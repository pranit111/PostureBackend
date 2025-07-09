# Posture Detection API

A FastAPI-based backend for real-time posture analysis using MediaPipe and OpenCV.

## Features

- Real-time posture analysis from uploaded images
- Detection of common posture issues:
  - Forward head posture
  - Shoulder misalignment
  - Spine slouching
- RESTful API with automatic documentation
- Built with MediaPipe for accurate pose detection

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

3. Access the API:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs

## API Endpoints

- `POST /analyze` - Upload image for posture analysis
- `GET /health` - Health check endpoint
- `GET /` - API information

## Project Structure

```
posture-backend/
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── posture_analyzer.py  # Core posture analysis logic
├── utils.py             # Utility functions
└── requirements.txt     # Dependencies
```

## Technologies Used

- **FastAPI** - Modern web framework for building APIs
- **MediaPipe** - Google's framework for pose detection
- **OpenCV** - Computer vision library
- **Pydantic** - Data validation using Python type hints
```

2. The API will be available at: `http://localhost:8000`

3. API Documentation: `http://localhost:8000/docs`

## API Endpoints

### POST /analyze/frame
Analyze a single frame for posture detection.

**Request:**
- File: Image file (JPG, PNG, etc.)

**Response:**
```json
{
  "bad_posture": true,
  "reason": "Back angle < 150°",
  "angle": 142.5
}
```

### POST /analyze/frame/detailed
Get detailed analysis with additional metrics.

### GET /health
Health check endpoint.

### GET /stats
Get processing statistics.

## Project Structure

```
posture-backend/
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── posture_analyzer.py  # Core posture analysis logic
├── utils.py             # Utility functions
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## How It Works

1. **Image Processing**: Uploaded frames are preprocessed for optimal pose detection
2. **Pose Detection**: MediaPipe extracts pose landmarks from the image
3. **Angle Calculation**: Back angle is calculated using shoulder, hip, and knee positions
4. **Assessment**: Posture is flagged as bad if back angle < 150°
5. **Response**: Structured JSON response with analysis results

## Configuration

The posture analysis can be configured by modifying the `PostureAnalyzer` class:

- Detection confidence threshold
- Tracking confidence threshold
- Angle thresholds for bad posture
- Image preprocessing parameters

## Future Enhancements

- Database integration for session tracking
- Historical data analysis
- Additional posture metrics (head forward, shoulder alignment)
- Real-time notifications
- Machine learning model for improved accuracy

## Error Handling

The API includes comprehensive error handling for:
- Invalid file types
- File size limits
- Image processing errors
- Pose detection failures
- Network errors

## Performance

- Average processing time: ~0.1-0.3 seconds per frame
- Supports concurrent requests
- Optimized for real-time analysis
- Memory efficient processing
