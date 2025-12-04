# CUDA 12.1 ベースイメージ（PyTorch 2.0.0 との互換性を考慮）
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=Asia/Tokyo

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    ca-certificates \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Miniconda のインストール
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# conda-forge のみを使用するように設定（Anaconda の利用規約問題を回避）
RUN conda config --remove channels defaults || true && \
    conda config --remove channels https://repo.anaconda.com/pkgs/main || true && \
    conda config --remove channels https://repo.anaconda.com/pkgs/r || true && \
    conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda config --show channels

# Conda 環境の作成
COPY sema_env.yaml /tmp/sema_env.yaml

# PyTorch を CUDA 対応版に変更して環境を作成（--override-channels で conda-forge のみ使用）
RUN conda create -n sema_env python=3.11 --override-channels -c conda-forge -y && \
    /bin/bash -c "source activate sema_env && \
    pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install timm==0.9.8 numpy==1.24.3 scipy tqdm easydict ftfy ipdb"

# 作業ディレクトリの設定
WORKDIR /workspace

# デフォルトで sema_env を有効化
RUN echo "source activate sema_env" >> ~/.bashrc
SHELL ["/bin/bash", "-c"]

# エントリポイント
CMD ["bash", "-c", "source activate sema_env && bash"]
