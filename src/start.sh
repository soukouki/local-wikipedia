#!/bin/bash

set -e

# DBの初期設定
/app/src/db-init.sh

# Wikipediaデータのダウンロード・保存
/app/src/download.py

# MCPサーバーの起動
exec /app/src/local-wikipedia.py
