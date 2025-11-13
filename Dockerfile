# ベースイメージ
FROM ubuntu:24.04

# 作業ディレクトリの設定
WORKDIR /app

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Groonga, PGroonga, PostgreSQLのインストール
RUN apt-get update && apt-get install -y \
    ca-certificates \
    lsb-release \
    wget && \
    wget https://packages.groonga.org/ubuntu/groonga-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt install -y ./groonga-apt-source-latest-$(lsb_release --codename --short).deb && \
    apt-get update && \
    apt-get install -y \
    groonga \
    groonga-tokenizer-mecab \
    postgresql-16-pgroonga && \
    apt-get clean && rm -rf /var/lib/apt/lists/* ./groonga-apt-source-latest-*.deb

# Pythonのインストール
RUN apt-get update && apt-get install -Vy \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python仮想環境を作成し、PATHを通す
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pythonパッケージのインストール
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    datasets \
    huggingface_hub \
    psycopg2-binary \
    mcp \
    pyyaml \
    starlette \
    uvicorn

# その他雑多なツールのインストール
RUN apt-get update && apt-get install -y \
    curl \
    sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
# sudoは必須
# curlは必須ではないがデバッグのため

# 各ファイルのコピー
COPY src /app/src
# 設定ファイル(デフォルト設定)のコピー
COPY config.yaml .

ENTRYPOINT ["src/start.sh"]
# 仮EntryPoint
# ENTRYPOINT ["tail", "-f", "/dev/null"]