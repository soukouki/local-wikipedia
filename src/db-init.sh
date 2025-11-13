#!/bin/bash

# DBの初期設定を行うスクリプト

set -e

DBNAME="finewiki"
DBUSER="dbuser"
DBPASS="dbpass"

# DBの起動
mkdir -p /app/data/pgroonga_test
pg_createcluster 16 finewiki --datadir=/app/data/pgroonga --port=5432 || true
pg_ctlcluster 16 finewiki start
echo "PostgreSQL started."

# ユーザーと新しいデータベースを作成
sudo -u postgres psql postgres <<SQL
CREATE USER "${DBUSER}" WITH PASSWORD '${DBPASS}';
CREATE DATABASE "${DBNAME}" OWNER "${DBUSER}";
GRANT ALL PRIVILEGES ON DATABASE "${DBNAME}" TO "${DBUSER}";
SQL
echo "If not exists, user '${DBUSER}' and database '${DBNAME}' created."

# 作成した'finewiki'データベースに接続して、拡張機能を作成
sudo -u postgres psql "${DBNAME}" <<SQL
CREATE EXTENSION pgroonga;
SQL
echo "If not enabled, extension 'pgroonga' created in database '${DBNAME}'."

