#!/usr/bin/env python3
"""
Video Posture Analysis API Demo
Demonstrates how to use the video analysis endpoint with detailed examples.
"""

import requests
import json
import time

def demo_video_analysis_endpoint():
    """Demonstrate the video analysis endpoint capabilities"""
    
    print("🎥 VIDEO POSTURE ANALYSIS API DEMO")
    print("=" * 60)
    print()
    
    # API endpoint information
    base_url = "http://localhost:8000"
    video_endpoint = f"{base_url}/analyze/video"
    
    print("📊 ENDPOINT INFORMATION:")
    print(f"• Video Analysis: POST {video_endpoint}")
    print("• Supported formats: MP4, AVI, MOV")
    print("• Maximum file size: 50MB")
    print("• Response format: JSON with detailed analysis")
    print()
    
    print("🔍 ANALYSIS FEATURES:")
    print("• Activity Detection: squat, sitting, standing, walking")
    print("• Frame-by-frame posture analysis")
    print("• Activity-specific feedback and suggestions")
    print("• Overall posture scoring (0-100)")
    print("• Per-frame detailed analysis")
    print()
    
    print("🎯 ACTIVITY-SPECIFIC FEEDBACK:")
    print()
    print("📐 SQUAT ANALYSIS:")
    print("  • Flags if knee goes beyond toe")
    print("  • Checks back angle (<150° flagged as poor form)")
    print("  • Monitors depth and alignment")
    print("  • Provides form improvement suggestions")
    print()
    
    print("💺 DESK SITTING ANALYSIS:")
    print("  • Detects slouching (back angle monitoring)")
    print("  • Flags forward head posture (neck >30°)")
    print("  • Checks if back isn't straight")
    print("  • Suggests ergonomic improvements")
    print()
    
    print("🚶 STANDING/WALKING ANALYSIS:")
    print("  • Monitors posture alignment")
    print("  • Detects forward head posture")
    print("  • Checks shoulder alignment")
    print("  • Provides standing posture tips")
    print()
    
    print("📋 EXAMPLE RESPONSE STRUCTURE:")
    print()
    
    # Example response structure
    example_response = {
        "activity_detected": "squat",
        "overall_posture_score": 75.5,
        "frame_analyses": [
            {
                "frame_number": 10,
                "timestamp": 0.33,
                "bad_posture": True,
                "back_angle": 145.2,
                "posture_status": "poor",
                "activity_specific_issues": [
                    "Back angle too forward during squat"
                ],
                "suggestions": [
                    "Keep your chest up and back straight"
                ]
            },
            {
                "frame_number": 20,
                "timestamp": 0.67,
                "bad_posture": False,
                "back_angle": 165.8,
                "posture_status": "good",
                "activity_specific_issues": [],
                "suggestions": []
            }
        ],
        "activity_specific_feedback": {
            "activity": "squat",
            "total_frames": 30,
            "poor_posture_frames": 12,
            "common_issues": [
                "Back angle too forward during squat",
                "Poor squat form detected"
            ],
            "improvement_suggestions": [
                "Keep your chest up and back straight",
                "Ensure knees don't go beyond toes"
            ],
            "specific_metrics": {
                "average_back_angle": 155.3,
                "posture_consistency": 60.0
            }
        },
        "summary": {
            "total_duration": 1.0,
            "good_posture_percentage": 60.0,
            "main_activity": "squat",
            "key_recommendations": [
                "Focus on maintaining proper back alignment",
                "Practice controlled movements"
            ]
        },
        "processing_time": 2.45,
        "total_frames": 30,
        "analyzed_frames": 3
    }
    
    print(json.dumps(example_response, indent=2))
    print()
    
    print("💡 HOW TO USE:")
    print("1. Prepare a video file (MP4, AVI, or MOV)")
    print("2. Make a POST request to the endpoint")
    print("3. Include the video file in the request body")
    print("4. Receive detailed analysis response")
    print()
    
    print("🐍 PYTHON EXAMPLE:")
    print("""
import requests

# Upload and analyze video
with open('your_video.mp4', 'rb') as video_file:
    files = {'file': video_file}
    response = requests.post(
        'http://localhost:8000/analyze/video',
        files=files
    )
    
    if response.status_code == 200:
        analysis = response.json()
        print(f"Activity: {analysis['activity_detected']}")
        print(f"Score: {analysis['overall_posture_score']}")
        print(f"Issues: {analysis['activity_specific_feedback']['common_issues']}")
    else:
        print(f"Error: {response.status_code}")
""")
    
    print("🌐 JAVASCRIPT/FETCH EXAMPLE:")
    print("""
const formData = new FormData();
formData.append('file', videoFile);

fetch('http://localhost:8000/analyze/video', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Activity:', data.activity_detected);
    console.log('Score:', data.overall_posture_score);
    console.log('Feedback:', data.activity_specific_feedback);
})
.catch(error => console.error('Error:', error));
""")
    
    print("📝 CURL EXAMPLE:")
    print("""
curl -X POST "http://localhost:8000/analyze/video" \\
     -H "accept: application/json" \\
     -H "Content-Type: multipart/form-data" \\
     -F "file=@your_video.mp4"
""")
    
    print("⚡ PERFORMANCE NOTES:")
    print("• Videos are analyzed frame-by-frame (every 10th frame)")
    print("• Processing time depends on video length and resolution")
    print("• Temporary files are automatically cleaned up")
    print("• Real-time feedback for activities and posture")
    print()
    
    print("🔧 TESTING THE API:")
    print("• Check API status: GET http://localhost:8000/health")
    print("• View API docs: http://localhost:8000/docs")
    print("• Get processing stats: GET http://localhost:8000/stats")
    print()
    
    # Test API availability
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is running and healthy!")
            health_data = response.json()
            print(f"   Status: {health_data['status']}")
            print(f"   Analyzer: {health_data['analyzer_status']}")
        else:
            print("❌ API health check failed")
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to API: {e}")
        print("   Make sure the server is running: python -m uvicorn main:app --host 0.0.0.0 --port 8000")
    
    print()
    print("🎯 Ready to analyze your posture videos!")
    print("📁 Place your video files and start making requests!")

if __name__ == "__main__":
    demo_video_analysis_endpoint()
