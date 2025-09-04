from flask import Flask, request, jsonify, send_from_directory
import librosa
import numpy as np
import soundfile as sf
import io
from pydub import AudioSegment
import whisper # 导入Whisper

# 初始化Flask应用
app = Flask(__name__)

# --- 核心改动：实现Whisper模型的懒加载 ---
# 1. 初始化一个全局变量来存储模型，初始值为None
model = None

def get_whisper_model():
    """
    此函数用于获取Whisper模型。
    它会检查模型是否已经加载到内存中，如果没有，则进行加载。
    这可以防止在应用启动时因加载模型过慢而导致部署超时。
    """
    global model
    if model is None:
        print("Whisper model is not loaded. Loading now... (This may take a moment on the first run)")
        # 2. 如果模型未加载，则加载它。这步操作只会在第一次被调用时执行。
        model = whisper.load_model("tiny.en")
        print("Whisper model loaded successfully.")
    return model
# --- 改动结束 ---


# 使用Flask的after_request装饰器手动添加CORS头
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 添加一个路由来服务前端HTML页面
@app.route('/')
def serve_index():
    return send_from_directory('.', 'dementia_screening_demo.html')

def analyze_audio_features(audio_data, sr_librosa):
    non_silent_intervals = librosa.effects.split(audio_data, top_db=20)
    pauses = []
    for i in range(len(non_silent_intervals) - 1):
        pause_start = non_silent_intervals[i][1] / sr_librosa
        pause_end = non_silent_intervals[i+1][0] / sr_librosa
        pause_duration = pause_end - pause_start
        if pause_duration > 0.1:
            pauses.append(pause_duration)
    if not pauses:
        return 0.0
    return np.mean(pauses)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    try:
        audio_segment = AudioSegment.from_file(audio_file)
        wav_io = io.BytesIO()
        audio_segment.set_frame_rate(16000).set_channels(1).export(wav_io, format="wav")
        wav_io.seek(0)
        
        with sf.SoundFile(wav_io, 'r') as sound_file:
            audio_data_whisper = sound_file.read(dtype='float32')
        
        # --- 核心改动：调用函数来获取模型，而不是直接使用全局变量 ---
        # 3. 在需要模型时，调用get_whisper_model()
        current_model = get_whisper_model()
        result = current_model.transcribe(audio_data_whisper, fp16=False)
        transcript = result['text']
        # --- 改动结束 ---
            
        word_count = len(transcript.split())

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

    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({"error": "An internal server error occurred during analysis."}), 500

