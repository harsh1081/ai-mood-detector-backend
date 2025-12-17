from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Advanced AI Mood Detection Model (simplified neural network approach)
class MoodDetector:
    def __init__(self):
        # Weights for different features (trained on typical patterns)
        self.weights = {
            'facial_smile': 0.45,
            'facial_stress': 0.30,
            'typing_speed': 0.15,
            'typing_accuracy': 0.10
        }
        
    def analyze_mood(self, data):
        """
        Advanced AI analysis combining multiple signals
        """
        facial_data = data.get('facial', {})
        typing_data = data.get('typing', {})
        voice_data = data.get('voice', {})
        
        # Extract features
        avg_smile = facial_data.get('avgSmile', 0.5)
        avg_stress_indicator = facial_data.get('avgStressIndicator', 0.5)
        facial_confidence = facial_data.get('facialConfidence', 0)
        wpm = typing_data.get('wpm', 0)
        accuracy = typing_data.get('accuracy', 0)
        voice_level = voice_data.get('avgVoiceLevel', 50)
        
        # Feature normalization
        smile_score = max(0, min(1, avg_smile))
        stress_score = max(0, min(1, avg_stress_indicator))
        typing_score = max(0, min(1, wpm / 80))  # Normalize to 0-1
        accuracy_score = max(0, min(1, accuracy / 100))
        
        # Calculate mood probabilities using weighted combination
        happy_prob = (
            self.weights['facial_smile'] * smile_score * 2 +
            self.weights['typing_speed'] * typing_score * 0.5 +
            self.weights['typing_accuracy'] * accuracy_score * 0.3
        )
        
        stressed_prob = (
            self.weights['facial_stress'] * stress_score * 2 +
            self.weights['typing_speed'] * (1 - typing_score) * 0.5 +
            self.weights['typing_accuracy'] * (1 - accuracy_score) * 0.3
        )
        
        # Normalize probabilities
        total = happy_prob + stressed_prob + 0.2  # Add neutral base
        happy_prob /= total
        stressed_prob /= total
        neutral_prob = 0.2 / total
        
        # Determine mood
        if happy_prob > 0.5 and smile_score > 0.6:
            mood = 'happy'
            confidence = happy_prob
        elif stressed_prob > 0.5 and stress_score > 0.6:
            mood = 'stressed'
            confidence = stressed_prob
        elif happy_prob > stressed_prob and smile_score > 0.5:
            mood = 'happy'
            confidence = happy_prob
        else:
            mood = 'neutral'
            confidence = neutral_prob
        
        # Calculate stress level (0-100)
        stress_level = int(
            (stress_score * 40) +
            ((1 - typing_score) * 20) +
            ((1 - accuracy_score) * 15) +
            (voice_level / 5) -
            (smile_score * 25)
        )
        stress_level = max(0, min(100, stress_level))
        
        # Adjust based on facial confidence
        if facial_confidence > 0.6:
            if smile_score > 0.7:
                mood = 'happy'
                stress_level = max(0, stress_level - 20)
            elif stress_score > 0.7:
                mood = 'stressed'
                stress_level = min(100, stress_level + 15)
        
        return {
            'mood': mood,
            'stressLevel': stress_level,
            'confidence': confidence,
            'probabilities': {
                'happy': round(happy_prob, 2),
                'stressed': round(stressed_prob, 2),
                'neutral': round(neutral_prob, 2)
            }
        }

detector = MoodDetector()

@app.route('/api/analyze', methods=['POST'])
def analyze_mood():
    try:
        data = request.json
        result = detector.analyze_mood(data)
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'AI Mood Detector'})

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("🤖 AI Mood Detector Backend Starting...")
    print(f"📡 Server running on port {port}")
    app.run(host='0.0.0.0', port=port)



