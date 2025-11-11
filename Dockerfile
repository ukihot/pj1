FROM python:3.11-slim

WORKDIR /app

# OpenCV、ffmpeg、X11に必要な最小限のパッケージ
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY src/ ./src/

# Pythonパスにsrcを追加
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# エントリーポイント
ENTRYPOINT ["python3", "src/hippolytica.py"]
