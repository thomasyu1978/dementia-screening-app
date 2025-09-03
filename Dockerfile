# 1. 使用官方的Python 3.11基础镜像
FROM python:3.11-slim

# 2. 在容器内部创建一个工作目录
WORKDIR /app

# 3. 更新包管理器并安装所有系统级依赖
#    我们现在一次性安装 ffmpeg, build-essential, 和 portaudio19-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    portaudio19-dev

# 4. 复制依赖文件并安装Python库
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制您项目的所有文件到工作目录
COPY . .

# 6. 暴露Gunicorn将要监听的端口 (Render会自动映射)
EXPOSE 10000

# 7. 容器启动时运行的命令
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
