from flask import Flask, request, jsonify
import speech_recognition as sr
import librosa
import numpy as np
import soundfile as sf
import io
from pydub import AudioSegment

# 初始化Flask应用
app = Flask(__name__)

# 使用Flask的after_request装饰器手动添加CORS头
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 用于语音识别的识别器实例
r = sr.Recognizer()

def analyze_audio_features(audio_data, sr_librosa):
    # 查找非静音部分
    non_silent_intervals = librosa.effects.split(audio_data, top_db=20)
    
    pauses = []
    # 计算每次停顿的持续时间
    for i in range(len(non_silent_intervals) - 1):
        pause_start = non_silent_intervals[i][1] / sr_librosa
        pause_end = non_silent_intervals[i+1][0] / sr_librosa
        pause_duration = pause_end - pause_start
        if pause_duration > 0.1: # 只考虑超过0.1秒的停顿
            pauses.append(pause_duration)
            
    if not pauses:
        return 0.0

    # 返回平均停顿时长
    return np.mean(pauses)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    try:
        # 将传入的音频转换为WAV格式
        audio_segment = AudioSegment.from_file(audio_file)
        
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)

        # 语音转文字
        with sr.AudioFile(wav_io) as source:
            audio_data_sr = r.record(source)
            transcript = r.recognize_google(audio_data_sr, language='en-US')
            
        word_count = len(transcript.split())

        # 语音特征分析
        wav_io.seek(0)
        audio_data, sample_rate = sf.read(wav_io)
        
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        total_duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        
        duration_minutes = total_duration / 60
        if duration_minutes > 0:
            words_per_minute = round(word_count / duration_minutes)
        else:
            words_per_minute = 0
            
        avg_pause_duration = analyze_audio_features(audio_data, sample_rate)

        unique_words = len(set(transcript.lower().split()))
        lexical_richness = round((unique_words / word_count * 100), 1) if word_count > 0 else 0

        # 风险评估
        risk = 'Low Risk'
        risk_suggestion = 'All indicators are within the normal range. Please continue to maintain a healthy lifestyle.'

        if words_per_minute < 140 and words_per_minute >= 100:
            risk = 'Medium Risk'
            risk_suggestion = 'Some indicators show slight abnormalities. We recommend that you keep monitoring and consider regular check-ups.'
        elif words_per_minute < 100:
            risk = 'High Risk'
            risk_suggestion = 'Significant linguistic abnormalities have been detected. We strongly recommend consulting a doctor for a comprehensive evaluation.'
        
        if avg_pause_duration > 0.9 and risk == 'Low Risk':
            risk = 'Medium Risk'
            risk_suggestion = 'Your speaking rate is normal, but with slightly longer pauses. We recommend that you keep monitoring and consider regular check-ups.'

        return jsonify({
            "speakingRate": f"{words_per_minute} WPM",
            "pauseDuration": f"{avg_pause_duration:.2f} s",
            "lexicalRichness": f"{lexical_richness}%",
            "riskLevel": risk,
            "suggestion": risk_suggestion,
            "transcript": transcript
        })

    except sr.UnknownValueError:
        return jsonify({"error": "Speech recognition could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition service error; {e}"}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

# 注意：用于生产环境的代码不应包含 if __name__ == '__main__': app.run()
