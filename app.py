from flask import Flask, request, jsonify
# 我们不再需要 flask_cors, 所以可以移除这个导入
# from flask_cors import CORS
import speech_recognition as sr
import librosa
import numpy as np
import soundfile as sf
import io
from pydub import AudioSegment

# 初始化Flask应用
app = Flask(__name__)

# --- 核心改动：使用Flask的after_request装饰器手动添加CORS头 ---
@app.after_request
def after_request(response):
    """
    这个函数会在每个请求处理完毕后、返回给前端前执行。
    我们在这里手动为响应添加CORS相关的头信息。
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
# --- 改动结束 ---


# 用于语音识别的识别器实例
r = sr.Recognizer()

def analyze_audio_features(audio_data, sr_librosa):
    """
    使用Librosa分析音频特征
    :param audio_data: 音频数据 (numpy array)
    :param sr_librosa: 采样率
    :return: 平均停顿时长
    """
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
    # 检查请求中是否包含文件
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    try:
        # --- 核心改动：将传入的音频转换为WAV格式 ---
        # pydub可以自动识别传入的音频格式 (webm, ogg, mp3等)
        audio_segment = AudioSegment.from_file(audio_file)
        
        # 创建一个内存中的WAV文件对象
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0) # 重置指针到文件开头
        # --- 转换完成 ---

        # --- 语音转文字 (现在使用转换后的WAV文件) ---
        with sr.AudioFile(wav_io) as source:
            audio_data_sr = r.record(source)
            # 使用 Whisper API (需要联网) 或本地Whisper模型进行识别
            # 这里使用Google的API作为示例, 您可以换成recognize_whisper
            transcript = r.recognize_google(audio_data_sr, language='en-US')
            
        word_count = len(transcript.split())

        # --- 语音特征分析 (Librosa) ---
        # 为了让Librosa也能处理, 再次重置指针
        wav_io.seek(0)
        audio_data, sample_rate = sf.read(wav_io)
        
        # 如果是多声道, 转为单声道
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        total_duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        
        # 计算语速 (WPM)
        duration_minutes = total_duration / 60
        if duration_minutes > 0:
            words_per_minute = round(word_count / duration_minutes)
        else:
            words_per_minute = 0
            
        # 分析停顿
        avg_pause_duration = analyze_audio_features(audio_data, sample_rate)

        # 词汇丰富度
        unique_words = len(set(transcript.lower().split()))
        lexical_richness = round((unique_words / word_count * 100), 1) if word_count > 0 else 0


        # --- 风险评估 (与前端类似的逻辑) ---
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


        # 将结果打包成JSON格式返回
        return jsonify({
            "speakingRate": f"{words_per_minute} WPM",
            "pauseDuration": f"{avg_pause_duration:.2f} s",
            "lexicalRichness": f"{lexical_richness}%",
            "riskLevel": risk,
            "suggestion": risk_suggestion,
            "transcript": transcript # 附上识别的文本
        })

    except sr.UnknownValueError:
        return jsonify({"error": "Speech recognition could not understand audio"}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition service error; {e}"}), 500
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

# --- 核心改动：移除或注释掉仅供本地开发使用的 app.run() ---
# if __name__ == '__main__':
#    # 启动服务器, 监听在所有网络接口的5000端口
#    app.run(host='0.0.0.0', port=5000, debug=True)
# --- 改动结束 ---


### 第2步：将您的代码上传到 GitHub

1.  **创建GitHub仓库**：在 [GitHub](https://github.com/) 网站上创建一个新的、公开的（Public）代码仓库，例如 `dementia-screening-app`。
2.  **上传您的项目**：将您的 `app.py`, `dementia_screening_demo.html`, 和 `requirements.txt` 这三个文件上传到这个新的仓库中。您可以使用网页上传，也可以使用 Git 命令行工具。

### 第3步：在 Render 平台上进行部署

Render 提供免费的套餐，非常适合托管您这样的小项目。

1.  **注册并登录 Render**：
    访问 [https://render.com/](https://render.com/)，建议直接使用您的 GitHub 账户进行注册和登录。

2.  **创建新的 Web 服务**：
    * 登录后，进入您的 Dashboard（仪表盘）。
    * 点击 **"New +"** 按钮，然后选择 **"Web Service"**。

3.  **连接您的 GitHub 仓库**：
    * 选择 “Build and deploy from a Git repository”。
    * 授权 Render 访问您的 GitHub 账户，然后从列表中选择您刚刚创建的 `dementia-screening-app` 仓库，点击 **"Connect"**。

4.  **配置您的服务**：
    现在您会看到一个配置页面，这是最关键的一步。请按如下方式填写：
    * **Name**：给您的应用起一个独一无二的名字，比如 `dementia-screener`。您的应用最终将通过 `dementia-screener.onrender.com` 这个网址访问。
    * **Region**：选择一个离您近的地区（例如 Singapore）。
    * **Branch**：保持 `main` 或 `master` 不变。
    * **Runtime**：选择 **`Python 3`**。
    * **Build Command**：`pip install -r requirements.txt` (这通常是默认值，不用改)。
    * **Start Command**：填写 `gunicorn app:app`
        * 这里的 `app:app` 分别指：`文件名app.py` : `文件中的Flask实例app`。
    * **Instance Type**：选择 **Free**（免费套餐）。

5.  **解决 `ffmpeg` 依赖（非常重要！）**
    `ffmpeg` 不是一个 Python 库，而是一个系统级的软件。我们需要告诉 Render 在构建环境时安装它。
    * 在配置页面向下滚动，找到 **"Advanced"** 设置并展开它。
    * 点击 **"Add Buildpack"**。
    * 在弹出的输入框中，粘贴以下这个 buildpack 的 URL，然后保存：
        ```
        https://github.com/jonathanong/heroku-buildpack-ffmpeg-latest
        


