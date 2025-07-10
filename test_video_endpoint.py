#!/usr/bin/env python3
"""
Test script for the video posture analysis endpoint
"""

import requests
import json

def test_video_analysis_endpoint():
    """Test the video analysis endpoint"""
    
    # API endpoint
    url = "http://localhost:8000/analyze/video"
    
    # You would need to provide an actual video file path
    # video_file_path = "path/to/your/test/video.mp4"
    
    print("🎥 Video Posture Analysis Endpoint")
    print("=" * 50)
    print()
    print("📝 Endpoint: POST /analyze/video")
    print("📎 Accepts: MP4, AVI, MOV video files (up to 50MB)")
    print()
    print("🔍 What it analyzes:")
    print("• Activity detection (squat, sitting, standing, walking)")
    print("• Frame-by-frame posture analysis")
    print("• Activity-specific feedback")
    print("• Overall posture score")
    print()
    print("📊 Response includes:")
    print("• activity_detected: Type of activity in video")
    print("• overall_posture_score: 0-100 score")
    print("• frame_analyses: Array of frame-by-frame results")
    print("• activity_specific_feedback: Tailored suggestions")
    print("• summary: Overall performance and recommendations")
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
            }
        ],
        "activity_specific_feedback": {
            "activity": "squat",
            "total_frames": 120,
            "poor_posture_frames": 30,
            "common_issues": [
                "Back angle too forward during squat",
                "Poor squat form detected"
            ],
            "improvement_suggestions": [
                "Keep your chest up and shoulders back",
                "Ensure knees track over toes, not inward",
                "Maintain neutral spine throughout the movement"
            ],
            "specific_metrics": {
                "posture_consistency": 75.0
            }
        },
        "summary": {
            "overall_rating": "Good",
            "posture_score": 75.5,
            "activity_detected": "squat",
            "frames_analyzed": 120,
            "good_posture_frames": 90,
            "poor_posture_frames": 30,
            "recommendation": "Good squat posture overall. Focus on consistency and minor improvements."
        },
        "processing_time": 12.34,
        "total_frames": 1200,
        "analyzed_frames": 120
    }
    
    print("📋 Example Response Structure:")
    print(json.dumps(example_response, indent=2))
    print()
    
    print("🎯 Activity-Specific Checks:")
    print()
    print("🏋️ SQUAT Analysis:")
    print("• Flags if back angle < 150°")
    print("• Checks for knee-toe alignment")
    print("• Monitors squat depth and form")
    print()
    print("💺 SITTING Analysis:")
    print("• Flags if back angle < 160° (slouching)")
    print("• Detects forward head posture")
    print("• Checks shoulder alignment")
    print()
    print("🧍 STANDING Analysis:")
    print("• Flags if back angle < 170°")
    print("• Checks overall spinal alignment")
    print("• Monitors shoulder position")
    print()
    
    print("💡 Usage Example:")
    print("""
# Python usage:
import requests

with open('squat_video.mp4', 'rb') as video_file:
    files = {'file': video_file}
    response = requests.post('http://localhost:8000/analyze/video', files=files)
    result = response.json()
    
    print(f"Activity: {result['activity_detected']}")
    print(f"Score: {result['overall_posture_score']}%")
    print(f"Rating: {result['summary']['overall_rating']}")
    
# JavaScript/Frontend usage:
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
});
""")

if __name__ == "__main__":
    test_video_analysis_endpoint()
