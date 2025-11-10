# ビルドステージ：依存関係のインストール
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

WORKDIR /build

# ビルドに必要なパッケージ
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 仮想環境を作成してパッケージをインストール
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 実行ステージ：軽量なruntimeイメージ
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 実行に必要な最小限のパッケージのみ
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ビルドステージから仮想環境をコピー
COPY --from=builder /opt/venv /opt/venv

# 仮想環境をアクティブ化
ENV PATH="/opt/venv/bin:$PATH"

# ソースコードをコピー
COPY . .

# エントリーポイント：バッチスクリプト起動
ENTRYPOINT ["python3", "hippolytica.py"]
