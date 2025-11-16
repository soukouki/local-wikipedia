#!/usr/bin/env python3

import os
import re
import gzip
import requests
import yaml
import csv
import requests
from datasets import load_dataset
from pathlib import Path
from typing import List, Dict, Iterator, Tuple
import sys
import os
import json
import psycopg2
import psycopg2.extras
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import time
import traceback

# ========================================
# パス設定
# ========================================
ROOT_DIR = Path("/app")
CONFIG_FILE = ROOT_DIR / "config.yaml"
DATA_DIR = ROOT_DIR / "data"
METADATA_FILE = DATA_DIR / "languages.json"
DATASETS_DIR = DATA_DIR / "dataset"
CACHE_DIR = DATA_DIR / "cache"

# ========================================
# グローバル設定
# ========================================
NUM_CONSUMERS = os.cpu_count() or 4
DATA_QUEUE = Queue(maxsize=NUM_CONSUMERS * 2)
CHUNK_ITEMS = 2000
MAX_RETRIES = 10
DB_PARAMS = dict(host="localhost", port=5432, dbname="finewiki", user="dbuser", password="dbpass")


# ========================================
# データベース接続ヘルパー
# ========================================

@contextmanager
def db_cursor(autocommit=False):
    """
    データベースカーソルを提供（接続とカーソルを自動管理）
    
    Args:
        autocommit: Trueの場合は自動コミットモード（インデックス作成用）
    """
    with psycopg2.connect(**DB_PARAMS) as conn:
        if autocommit:
            conn.autocommit = True
        with conn.cursor() as cur:
            yield cur
            if not autocommit:
                conn.commit()


# ========================================
# ユーティリティ関数
# ========================================

def setup_directories():
    """必要なディレクトリを作成"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)


def format_bytes(bytes_size: int) -> str:
    """バイト数を人間が読みやすい形式に変換"""
    if bytes_size == 0: return "0 B"
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while bytes_size >= 1024 and i < len(units) - 1:
        bytes_size /= 1024.0
        i += 1
    return f"{bytes_size:.1f} {units[i]}"


def iterate_in_chunks(iterator: Iterator[any], chunk_size: int) -> Iterator[List[any]]:
    """イテレータを指定された件数ずつのチャンク(リスト)に分割する"""
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def clear_queue():
    """
    キューをクリアする
    各処理フェーズの間で呼ばれ、前のフェーズの終了シグナル(None)を削除する
    """
    while not DATA_QUEUE.empty():
        try:
            DATA_QUEUE.get_nowait()
        except:
            break


# ========================================
# 設定とメタデータ管理
# ========================================

def load_config() -> Dict:
    """config.yamlを読み込む"""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict) or 'source' not in config or 'language' not in config['source']:
            print("ERROR: Invalid config.yaml format.", file=sys.stderr)
            sys.exit(1)
        return config
    except FileNotFoundError:
        print(f"ERROR: Config file not found: {CONFIG_FILE}", file=sys.stderr)
        sys.exit(1)


def download_languages() -> Dict[str, Dict]:
    """HuggingFaceから言語メタデータ(ページ数、サイズ等)をダウンロード"""
    url = "https://huggingface.co/datasets/HuggingFaceFW/finewiki/resolve/main/language_subsets.csv"
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cached metadata: {e}. Downloading fresh data.")

    print("Downloading language metadata...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        lines = response.text.strip().split('\n')
        reader = csv.DictReader(lines)
        lang_metadata = {
            row['subset']: {k: (int(v) if v.isdigit() else v) for k, v in row.items()}
            for row in reader
        }
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(lang_metadata, f, ensure_ascii=False, indent=2)
        print("✓ Downloaded and cached metadata.")
        return lang_metadata
    except requests.RequestException as e:
        print(f"ERROR: Failed to download language metadata: {e}", file=sys.stderr)
        sys.exit(1)


# ========================================
# データベース操作
# ========================================

def setup_database_schema():
    """データベーススキーマを初期化"""
    print("Setting up database schema...")
    with db_cursor() as cur:
        # 言語管理テーブル
        cur.execute("""
            CREATE TABLE IF NOT EXISTS languages (
                language_code VARCHAR(20) PRIMARY KEY,
                is_page_complete BOOLEAN DEFAULT FALSE,
                is_document_complete BOOLEAN DEFAULT FALSE,
                is_redirection_complete BOOLEAN DEFAULT FALSE
            );
        """)
        # ページテーブル（Wikimedia dumps由来）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id BIGSERIAL PRIMARY KEY,
                page_id BIGINT NOT NULL,
                page_namespace INTEGER NOT NULL,
                page_title TEXT NOT NULL,
                language_code VARCHAR(20) NOT NULL REFERENCES languages(language_code) ON DELETE CASCADE,
                UNIQUE(page_id, language_code)
            );
        """)
        # ドキュメントテーブル（FineWeb-Edu由来）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                text_body TEXT NOT NULL,
                language_code VARCHAR(20) NOT NULL REFERENCES languages(language_code) ON DELETE CASCADE,
                page_id BIGINT NOT NULL,
                UNIQUE(page_id, language_code)
            );
        """)
        # リダイレクションテーブル（Wikimedia dumps由来）
        cur.execute("""
            CREATE TABLE IF NOT EXISTS redirections (
                id BIGSERIAL PRIMARY KEY,
                from_page_id BIGINT NOT NULL,
                to_title TEXT NOT NULL,
                language_code VARCHAR(20) NOT NULL REFERENCES languages(language_code) ON DELETE CASCADE
            );
        """)
    print("✓ Database schema is ready.")


def check_pages_complete(lang_code: str) -> bool:
    """指定された言語のページデータが完全に存在するか確認"""
    with db_cursor() as cur:
        cur.execute("SELECT is_page_complete FROM languages WHERE language_code = %s;", (lang_code,))
        result = cur.fetchone()
        return result is not None and result[0] is True


def check_documents_complete(lang_code: str) -> bool:
    """指定された言語のドキュメントデータが完全に存在するか確認"""
    with db_cursor() as cur:
        cur.execute("SELECT is_document_complete FROM languages WHERE language_code = %s;", (lang_code,))
        result = cur.fetchone()
        return result is not None and result[0] is True


def check_redirections_complete(lang_code: str) -> bool:
    """指定された言語のリダイレクションデータが完全に存在するか確認"""
    with db_cursor() as cur:
        cur.execute("SELECT is_redirection_complete FROM languages WHERE language_code = %s;", (lang_code,))
        result = cur.fetchone()
        return result is not None and result[0] is True


def check_indexes_exist() -> bool:
    """ドキュメントテーブルの主要インデックスが存在するか確認"""
    with db_cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'documents' AND indexname IN (
                'idx_documents_language_code',
                'idx_documents_title',
                'idx_documents_text_body'
            );
        """)
        result = cur.fetchone()
        return result is not None and result[0] == 3


def check_page_indexes_exist() -> bool:
    """ページテーブルのインデックスが存在するか確認"""
    with db_cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'pages' AND indexname IN (
                'idx_pages_language_code',
                'idx_pages_page_id'
            );
        """)
        result = cur.fetchone()
        return result is not None and result[0] == 2


def check_redirection_indexes_exist() -> bool:
    """リダイレクションテーブルのインデックスが存在するか確認"""
    with db_cursor() as cur:
        cur.execute("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'redirections' AND indexname IN (
                'idx_redirections_language_code',
                'idx_redirections_from_page_id',
                'idx_redirections_to_title'
            );
        """)
        result = cur.fetchone()
        return result is not None and result[0] == 3


def mark_pages_complete(lang_code: str):
    """言語のページ取得完了をマーク"""
    with db_cursor() as cur:
        cur.execute(
            "UPDATE languages SET is_page_complete = TRUE WHERE language_code = %s;",
            (lang_code,)
        )


def mark_documents_complete(lang_code: str):
    """言語のドキュメント取得完了をマーク"""
    with db_cursor() as cur:
        cur.execute(
            "UPDATE languages SET is_document_complete = TRUE WHERE language_code = %s;",
            (lang_code,)
        )


def mark_redirections_complete(lang_code: str):
    """言語のリダイレクション取得完了をマーク"""
    with db_cursor() as cur:
        cur.execute(
            "UPDATE languages SET is_redirection_complete = TRUE WHERE language_code = %s;",
            (lang_code,)
        )


def ensure_language_exists(lang_code: str):
    """言語レコードが存在することを保証"""
    with db_cursor() as cur:
        cur.execute("INSERT INTO languages(language_code) VALUES (%s) ON CONFLICT DO NOTHING;", (lang_code,))


def drop_indexes():
    """ドキュメントテーブルのインデックスを削除（大量挿入のパフォーマンス向上のため）"""
    print("Dropping existing document indexes (if any)...")
    with db_cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_documents_language_code;")
        cur.execute("DROP INDEX IF EXISTS idx_documents_title;")
        cur.execute("DROP INDEX IF EXISTS idx_documents_text_body;")
    print("✓ Existing document indexes dropped.")


def create_indexes():
    """ドキュメントテーブルのインデックスを作成"""
    print("\n--- Creating document indexes... ---", flush=True)
    with db_cursor(autocommit=True) as cur:
        print("Creating index on 'language_code'...", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_language_code ON documents(language_code);")

        print("Creating PGroonga index on 'title' (this may take a while)...", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_title ON documents USING pgroonga (title);")

        print("Creating PGroonga index on 'text_body' (this may take several hours)...", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_text_body ON documents USING pgroonga (text_body);")
    print("✓ All document indexes are created.", flush=True)


def create_page_indexes():
    """ページテーブルのインデックスを作成"""
    print("\n--- Creating page indexes... ---", flush=True)
    with db_cursor(autocommit=True) as cur:
        print("Creating index on pages 'language_code'...", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_language_code ON pages(language_code);")

        print("Creating index on pages 'page_id'...", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_page_id ON pages(page_id, language_code);")
    print("✓ All page indexes are created.", flush=True)


def create_redirection_indexes():
    """リダイレクションテーブルのインデックスを作成"""
    print("\n--- Creating redirection indexes... ---", flush=True)
    with db_cursor(autocommit=True) as cur:
        print("Creating index on redirections 'language_code'...", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_redirections_language_code ON redirections(language_code);")

        print("Creating index on redirections 'from_page_id'...", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_redirections_from_page_id ON redirections(from_page_id, language_code);")

        print("Creating index on redirections 'to_title'...", flush=True)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_redirections_to_title ON redirections(to_title);")
    print("✓ All redirection indexes are created.", flush=True)


# ========================================
# ページ取得処理
# ========================================

def plan_page_downloads(languages: List[str], lang_metadata: Dict) -> Dict:
    """
    ページのダウンロード計画を作成
    Returns: 計画を含む辞書 {'targets': [(lang_code, metadata), ...]}
    """
    print("\n--- Planning Page Downloads ---")
    
    targets = []
    
    for lang_code in languages:
        if lang_code not in lang_metadata:
            print(f"✗ {lang_code:6s} | Not available for pages")
            continue
        
        meta = lang_metadata[lang_code]
        
        if check_pages_complete(lang_code):
            print(f"Skipping [{lang_code}]: Pages already exist and seem complete.")
        else:
            print(f"Planning page download for [{lang_code}]")
            targets.append((lang_code, meta))
            ensure_language_exists(lang_code)
    
    return {'targets': targets}


def page_producer(lang_code: str, meta: Dict) -> Tuple[str, bool, int]:
    """
    ページデータをWikimedia dumpsからダウンロード・解析し、チャンクをキューに入れる
    namespace 0（記事ページ）のみを抽出する
    Returns: (lang_code, is_complete, processed_items)
    """
    print(f"  Starting page download for [{lang_code}]...")
    
    base_url = f"https://dumps.wikimedia.org/{lang_code}wiki/latest/"
    file_name = f"{lang_code}wiki-latest-page.sql.gz"
    download_url = base_url + file_name
    local_gz_file = CACHE_DIR / file_name
    extracted_sql_file = CACHE_DIR / f"{lang_code}wiki-latest-page.sql"
    
    retries = 0
    processed_items = 0
    
    while retries <= MAX_RETRIES:
        try:
            # ダウンロード
            if not os.path.exists(local_gz_file):
                print(f"  Downloading {download_url}...")
                response = requests.get(download_url, stream=True, timeout=60)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(local_gz_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0 and downloaded_size % (1024 * 1024 * 10) == 0:
                                progress_pct = (downloaded_size / total_size) * 100
                                print(f"    Download progress: {format_bytes(downloaded_size)} / {format_bytes(total_size)} ({progress_pct:.1f}%)", flush=True)
                
                print(f"  ✓ Downloaded {file_name} ({format_bytes(downloaded_size)})")
            else:
                print(f"  File already exists: {local_gz_file}")
            
            # 解凍
            if not os.path.exists(extracted_sql_file):
                print(f"  Extracting {local_gz_file}...")
                with gzip.open(local_gz_file, "rb") as gz_file:
                    with open(extracted_sql_file, "wb") as sql_file:
                        while True:
                            chunk = gz_file.read(1024 * 1024)
                            if not chunk:
                                break
                            sql_file.write(chunk)
                print(f"  ✓ Extracted to {extracted_sql_file}")
            else:
                print(f"  Extracted file already exists: {extracted_sql_file}")
            
            # パースと挿入（namespace 0のみ）
            print(f"  Parsing {extracted_sql_file} (namespace 0 only)...")
            chunk = []
            
            with open(extracted_sql_file, "r", encoding="utf-8", errors='ignore') as f:
                in_page_section = False
                
                for line in f:
                    line = line.strip()
                    
                    if "ALTER TABLE `page` DISABLE KEYS" in line:
                        in_page_section = True
                        continue
                    
                    if "ALTER TABLE `page` ENABLE KEYS" in line:
                        in_page_section = False
                        break
                    
                    if in_page_section and line.startswith("INSERT INTO"):
                        values_start = line.find("VALUES")
                        if values_start == -1:
                            continue
                        
                        values_part = line[values_start + 6:].strip()
                        if values_part.endswith(';'):
                            values_part = values_part[:-1]
                        
                        try:
                            # パターン: (page_id, page_namespace, 'page_title', ...)
                            pattern = r"\((\d+),(\d+),'([^']*(?:''[^']*)*)'"
                            
                            for match in re.finditer(pattern, values_part):
                                page_id = int(match.group(1))
                                page_namespace = int(match.group(2))
                                page_title = match.group(3).replace("''", "'")
                                
                                # namespace 0（記事）のみを処理
                                if page_namespace == 0:
                                    chunk.append((page_id, page_namespace, page_title, lang_code))
                                    processed_items += 1
                                    
                                    if len(chunk) >= CHUNK_ITEMS:
                                        DATA_QUEUE.put(('page', lang_code, chunk.copy()), timeout=10)
                                        chunk = []
                                    
                        except Exception as parse_error:
                            print(f"⚠ Warning: Failed to parse line, skipping: {parse_error}", file=sys.stderr)
                            print(f"  Type: {type(parse_error).__name__})", file=sys.stderr)
                            print(f"  Message: {str(parse_error)}", file=sys.stderr)
                            print(f"  Stacktrace:", file=sys.stderr)
                            traceback.print_exc(file=sys.stderr)
                            continue

            
            # 残りのチャンクを追加
            if chunk:
                DATA_QUEUE.put(('page', lang_code, chunk), timeout=10)
            
            print(f"  ✓ Finished parsing for [{lang_code}], {processed_items:,} pages queued.", flush=True)
            
            return (lang_code, True, processed_items)
        
        except requests.RequestException as e:
            retries += 1
            print(f"\n  ✗ ERROR downloading for [{lang_code}]: {e}", file=sys.stderr, flush=True)
            
            if retries > MAX_RETRIES:
                print(f"  ✗ CRITICAL: Page producer for [{lang_code}] failed after {MAX_RETRIES} retries.", file=sys.stderr)
                return (lang_code, False, processed_items)
            
            wait_time = 15 * retries
            print(f"  ... Retrying ({retries}/{MAX_RETRIES}) in {wait_time} seconds.", file=sys.stderr, flush=True)
            time.sleep(wait_time)
            
        except Exception as e:
            retries += 1
            print(f"\n  ✗ ERROR in page producer for [{lang_code}]: {e}", file=sys.stderr, flush=True)
            
            if retries > MAX_RETRIES:
                print(f"  ✗ CRITICAL: Page producer for [{lang_code}] failed after {MAX_RETRIES} retries.", file=sys.stderr)
                return (lang_code, False, processed_items)
            
            wait_time = 15 * retries
            print(f"  ... Retrying ({retries}/{MAX_RETRIES}) in {wait_time} seconds.", file=sys.stderr, flush=True)
            time.sleep(wait_time)
    
    return (lang_code, False, processed_items)


def page_consumer(db_params: Dict, progress: Dict):
    """
    キューからページデータを取り出し、DBに一括挿入する
    各スレッドが独立したDB接続を持つ
    """
    tid = threading.get_ident()
    print(f"  [Page Consumer {tid}] Starting...", flush=True)
    
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                while True:
                    item = DATA_QUEUE.get(timeout=300)
                    if item is None:
                        DATA_QUEUE.put(None, timeout=10)
                        break
                    
                    data_type, lang_code, chunk = item
                    
                    if data_type == 'page':
                        psycopg2.extras.execute_values(
                            cur,
                            "INSERT INTO pages(page_id, page_namespace, page_title, language_code) VALUES %s ON CONFLICT DO NOTHING",
                            chunk
                        )
                        conn.commit()
                        
                        with progress['lock']:
                            progress['count'] += len(chunk)
                            print(f"  Processed {progress['count']:,} pages...", flush=True)
    
    except psycopg2.Error as e:
        print(f"\n  ✗ ERROR in page consumer {tid}: {e}", file=sys.stderr, flush=True)
        raise e
    
    print(f"\n  [Page Consumer {tid}] Finished.", flush=True)


def execute_page_downloads(plan: Dict):
    """ページのダウンロード計画を実行"""
    targets = plan['targets']
    if not targets:
        print("\nAll page data already exists.")
        return
    
    print("\n--- Executing Page Downloads ---")
    print(f"Using {NUM_CONSUMERS} parallel DB writers.")
    
    # 進捗管理
    # 総ページはユーザーページなどの不要なnamespaceも含まれるため、ここでは扱わない
    progress = {
        "count": 0,
        "lock": threading.Lock()
    }
    
    producer_results = []
    
    # Producer数を制限（同時ダウンロード数を制御）
    max_concurrent_producers = min(2, len(targets))
    
    with ThreadPoolExecutor(max_workers=NUM_CONSUMERS + max_concurrent_producers) as executor:
        # Consumer起動
        consumer_futures = [
            executor.submit(page_consumer, DB_PARAMS, progress)
            for _ in range(NUM_CONSUMERS)
        ]
        
        # Producer起動（各言語ごと）
        producer_futures = [
            executor.submit(page_producer, lang_code, meta)
            for lang_code, meta in targets
        ]
        
        # 各producerの結果を個別に収集
        for future in producer_futures:
            try:
                result = future.result()
                producer_results.append(result)
            except Exception as e:
                print(f"\n  ✗ Page producer failed with exception: {e}", file=sys.stderr, flush=True)
        
        print("\nAll page downloads finished. Signaling consumers to stop...", flush=True)
        DATA_QUEUE.put(None, timeout=10)
        
        # Consumer終了待ち
        for future in consumer_futures:
            try:
                future.result()
            except Exception as e:
                print(f"\n  ✗ Page consumer failed with exception: {e}", file=sys.stderr, flush=True)
    
    # 完了フラグを立てる（成功した言語のみ）
    for lang_code, is_complete, items in producer_results:
        if is_complete:
            mark_pages_complete(lang_code)
            print(f"  ✓ Marked [{lang_code}] pages as complete ({items:,} items).", flush=True)
        else:
            print(f"  ⚠ Skipping completion flag for [{lang_code}] pages due to incomplete download.", file=sys.stderr, flush=True)


# ========================================
# ドキュメント取得処理
# ========================================

def plan_document_downloads(languages: List[str], lang_metadata: Dict) -> Dict:
    """
    ドキュメントのダウンロード計画を作成
    Returns: 計画を含む辞書 {'targets': [...], 'total_items': int, 'already_processed': int}
    """
    print("\n--- Planning Document Downloads ---")
    
    targets = []
    total_items = 0
    already_processed = 0

    for lang_code in languages:
        if lang_code not in lang_metadata:
            print(f"✗ {lang_code:6s} | Not available")
            continue
        
        meta = lang_metadata[lang_code]
        pages = meta.get('pages', 0)
        size = meta.get('size_bytes', 0)
        
        if check_documents_complete(lang_code):
            print(f"Skipping [{lang_code}]: Documents already exist and seem complete.")
            total_items += pages
            already_processed += pages
        else:
            print(f"Planning download for [{lang_code}]: {pages:,} pages, {format_bytes(size)} total size.")
            targets.append((lang_code, meta))
            total_items += pages
            ensure_language_exists(lang_code)

    return {
        'targets': targets,
        'total_items': total_items,
        'already_processed': already_processed
    }


def document_producer(lang_code: str, meta: Dict) -> Tuple[str, bool, int]:
    """
    HuggingFaceからドキュメントデータをダウンロードし、チャンクをキューに入れる
    エラー発生時にはリトライし、処理済みの箇所から再開を試みる
    Returns: (lang_code, is_complete, processed_items)
    """
    print(f"  Starting download for [{lang_code}]...")
    os.environ['HF_HOME'] = str(CACHE_DIR)
    os.environ['HF_DATASETS_CACHE'] = str(CACHE_DIR / 'datasets')

    retries = 0
    processed_items = 0

    while retries <= MAX_RETRIES:
        try:
            dataset_stream = load_dataset(
                "HuggingFaceFW/finewiki", name=lang_code,
                cache_dir=str(CACHE_DIR / 'datasets'), streaming=True
            )['train']

            if processed_items > 0:
                print(f"\n  ... Resuming for [{lang_code}] after {processed_items:,} items.", file=sys.stderr, flush=True)
                dataset_stream = dataset_stream.skip(processed_items)

            chunk_iterator = iterate_in_chunks(dataset_stream, CHUNK_ITEMS)

            for chunk in chunk_iterator:
                DATA_QUEUE.put(('document', lang_code, chunk), timeout=10)
                processed_items += len(chunk)
                retries = 0

            print(f"  ✓ Finished download for [{lang_code}], {processed_items:,} items queued.", flush=True)
            return (lang_code, True, processed_items)

        except Exception as e:
            retries += 1
            print(f"\n  ✗ ERROR in producer for [{lang_code}]: {e}", file=sys.stderr, flush=True)

            if retries > MAX_RETRIES:
                print(f"  ✗ CRITICAL: Producer for [{lang_code}] failed after {MAX_RETRIES} retries.", file=sys.stderr)
                print(f"  ... Language [{lang_code}] will be automatically retried on the next run.", file=sys.stderr, flush=True)
                return (lang_code, False, processed_items)

            wait_time = 15 * retries
            print(f"  ... Retrying ({retries}/{MAX_RETRIES}) in {wait_time} seconds.", file=sys.stderr, flush=True)
            time.sleep(wait_time)

    return (lang_code, False, processed_items)


def document_consumer(db_params: Dict, progress: Dict):
    """
    キューからドキュメントデータを取り出し、DBに一括挿入する
    各スレッドが独立したDB接続を持つ
    """
    tid = threading.get_ident()
    print(f"  [Document Consumer {tid}] Starting...", flush=True)

    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                while True:
                    item = DATA_QUEUE.get(timeout=300)
                    if item is None:
                        DATA_QUEUE.put(None, timeout=10)
                        break

                    data_type, lang_code, chunk = item
                    
                    if data_type == 'document':
                        psycopg2.extras.execute_values(
                            cur,
                            "INSERT INTO documents(title, text_body, language_code, page_id) VALUES %s ON CONFLICT DO NOTHING",
                            [(d['title'], d['text'], lang_code, d['page_id']) for d in chunk],
                        )
                        conn.commit()

                        with progress['lock']:
                            progress['count'] += len(chunk)
                            if progress['total'] > 0:
                                pct = (progress['count'] / progress['total']) * 100
                                print(f"  Processed {progress['count']:,} / {progress['total']:,} items ({pct:.2f}%)", flush=True)
                            else:
                                print(f"  Processed {progress['count']:,} items...", flush=True)

    except psycopg2.Error as e:
        print(f"\n  ✗ ERROR in consumer {tid}: {e}", file=sys.stderr, flush=True)
        raise e

    print(f"\n  [Document Consumer {tid}] Finished.", flush=True)


def execute_document_downloads(plan: Dict):
    """ドキュメントのダウンロード計画を実行"""
    targets = plan['targets']
    
    if not targets:
        print("\nAll document data already exists.")
        return
    
    print("\n--- Executing Document Downloads ---")
    print(f"Total items to process: {plan['total_items']:,}")
    print(f"Using {NUM_CONSUMERS} parallel DB writers.")
    
    # インデックスを削除してパフォーマンス向上
    drop_indexes()

    progress = {
        "count": plan['already_processed'],
        "total": plan['total_items'],
        "lock": threading.Lock()
    }

    producer_results = []
    
    with ThreadPoolExecutor(max_workers=NUM_CONSUMERS + 1) as executor:
        consumer_futures = [executor.submit(document_consumer, DB_PARAMS, progress) for _ in range(NUM_CONSUMERS)]
        producer_futures = [executor.submit(document_producer, lang_code, meta) for lang_code, meta in targets]

        # 各producerの結果を個別に収集
        for future in producer_futures:
            try:
                result = future.result()
                producer_results.append(result)
            except Exception as e:
                print(f"\n  ✗ Producer failed with exception: {e}", file=sys.stderr, flush=True)

        print("\nAll downloads finished. Signaling consumers to stop...", flush=True)
        DATA_QUEUE.put(None, timeout=10)

        for future in consumer_futures:
            try:
                future.result()
            except Exception as e:
                print(f"\n  ✗ Consumer failed with exception: {e}", file=sys.stderr, flush=True)

    # 完了フラグを立てる（成功した言語のみ）
    for lang_code, is_complete, items in producer_results:
        if is_complete:
            mark_documents_complete(lang_code)
            print(f"  ✓ Marked [{lang_code}] as complete ({items:,} items).", flush=True)
        else:
            print(f"  ⚠ Skipping completion flag for [{lang_code}] due to incomplete download ({items:,} items processed).", file=sys.stderr, flush=True)


# ========================================
# リダイレクト取得処理
# ========================================

def plan_redirection_downloads(languages: List[str], lang_metadata: Dict) -> Dict:
    """
    リダイレクションのダウンロード計画を作成
    Returns: 計画を含む辞書 {'targets': [(lang_code, metadata), ...]}
    """
    print("\n--- Planning Redirection Downloads ---")
    
    targets = []
    
    for lang_code in languages:
        if lang_code not in lang_metadata:
            print(f"✗ {lang_code:6s} | Not available for redirections")
            continue
        
        meta = lang_metadata[lang_code]
        
        if check_redirections_complete(lang_code):
            print(f"Skipping [{lang_code}]: Redirections already exist and seem complete.")
        else:
            redirects = meta.get('redirections', 0)
            print(f"Planning redirection download for [{lang_code}]: {redirects:,} redirections.")
            targets.append((lang_code, meta))
            ensure_language_exists(lang_code)
    
    return {'targets': targets}


def redirection_producer(lang_code: str, meta: Dict) -> Tuple[str, bool, int]:
    """
    リダイレクションデータをWikimedia dumpsからダウンロード・解析し、チャンクをキューに入れる
    rd_from（リダイレクト元のpage_id）とrd_title（リダイレクト先のタイトル）を抽出する
    Returns: (lang_code, is_complete, processed_items)
    """
    print(f"  Starting redirection download for [{lang_code}]...")
    
    base_url = f"https://dumps.wikimedia.org/{lang_code}wiki/latest/"
    file_name = f"{lang_code}wiki-latest-redirect.sql.gz"
    download_url = base_url + file_name
    local_gz_file = CACHE_DIR / file_name
    extracted_sql_file = CACHE_DIR / f"{lang_code}wiki-latest-redirect.sql"
    
    retries = 0
    processed_items = 0
    
    while retries <= MAX_RETRIES:
        try:
            # ダウンロード
            if not os.path.exists(local_gz_file):
                print(f"  Downloading {download_url}...")
                response = requests.get(download_url, stream=True, timeout=60)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(local_gz_file, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            if total_size > 0 and downloaded_size % (1024 * 1024 * 10) == 0:
                                progress_pct = (downloaded_size / total_size) * 100
                                print(f"    Download progress: {format_bytes(downloaded_size)} / {format_bytes(total_size)} ({progress_pct:.1f}%)", flush=True)
                
                print(f"  ✓ Downloaded {file_name} ({format_bytes(downloaded_size)})")
            else:
                print(f"  File already exists: {local_gz_file}")
            
            # 解凍
            if not os.path.exists(extracted_sql_file):
                print(f"  Extracting {local_gz_file}...")
                with gzip.open(local_gz_file, "rb") as gz_file:
                    with open(extracted_sql_file, "wb") as sql_file:
                        while True:
                            chunk = gz_file.read(1024 * 1024)
                            if not chunk:
                                break
                            sql_file.write(chunk)
                print(f"  ✓ Extracted to {extracted_sql_file}")
            else:
                print(f"  Extracted file already exists: {extracted_sql_file}")
            
            # パースと挿入
            print(f"  Parsing {extracted_sql_file}...")
            chunk = []
            
            with open(extracted_sql_file, "r", encoding="utf-8", errors='ignore') as f:
                in_redirect_section = False
                
                for line in f:
                    line = line.strip()
                    
                    # リダイレクトセクションの開始
                    if "ALTER TABLE `redirect` DISABLE KEYS" in line:
                        in_redirect_section = True
                        continue
                    
                    # リダイレクトセクションの終了
                    if "ALTER TABLE `redirect` ENABLE KEYS" in line:
                        in_redirect_section = False
                        break
                    
                    # INSERT文のパース
                    if in_redirect_section and line.startswith("INSERT INTO"):
                        values_start = line.find("VALUES")
                        if values_start == -1:
                            continue
                        
                        values_part = line[values_start + 6:].strip()
                        if values_part.endswith(';'):
                            values_part = values_part[:-1]
                        
                        # 各レコードをパース
                        # フォーマット: (rd_from, rd_namespace, 'rd_title', ...)
                        # rd_from: リダイレクト元のページID
                        # rd_title: リダイレクト先のタイトル
                        try:
                            pattern = r"\((\d+),\d+,'([^']*(?:''[^']*)*)'"
                            
                            for match in re.finditer(pattern, values_part):
                                from_page_id = int(match.group(1))
                                to_title = match.group(2).replace("''", "'")
                                
                                chunk.append((from_page_id, to_title, lang_code))
                                processed_items += 1
                                
                                # チャンクサイズに達したらキューに追加
                                if len(chunk) >= CHUNK_ITEMS:
                                    DATA_QUEUE.put(('redirection', lang_code, chunk.copy()), timeout=10)
                                    chunk = []
                                    
                        except Exception as parse_error:
                            print(f"⚠ Warning: Failed to parse line, skipping: {parse_error}", file=sys.stderr)
                            print(f"  Type: {type(parse_error).__name__})", file=sys.stderr)
                            print(f"  Message: {str(parse_error)}", file=sys.stderr)
                            print(f"  Stacktrace:", file=sys.stderr)
                            traceback.print_exc(file=sys.stderr)
                            continue
            
            # 残りのチャンクを追加
            if chunk:
                DATA_QUEUE.put(('redirection', lang_code, chunk), timeout=10)
            
            print(f"  ✓ Finished parsing for [{lang_code}], {processed_items:,} redirections queued.", flush=True)
            
            return (lang_code, True, processed_items)
        
        except requests.RequestException as e:
            retries += 1
            print(f"\n  ✗ ERROR downloading for [{lang_code}]: {e}", file=sys.stderr, flush=True)
            
            if retries > MAX_RETRIES:
                print(f"  ✗ CRITICAL: Redirection producer for [{lang_code}] failed after {MAX_RETRIES} retries.", file=sys.stderr)
                return (lang_code, False, processed_items)
            
            wait_time = 15 * retries
            print(f"  ... Retrying ({retries}/{MAX_RETRIES}) in {wait_time} seconds.", file=sys.stderr, flush=True)
            time.sleep(wait_time)
            
        except Exception as e:
            retries += 1
            print(f"\n  ✗ ERROR in redirection producer for [{lang_code}]: {e}", file=sys.stderr, flush=True)
            
            if retries > MAX_RETRIES:
                print(f"  ✗ CRITICAL: Redirection producer for [{lang_code}] failed after {MAX_RETRIES} retries.", file=sys.stderr)
                return (lang_code, False, processed_items)
            
            wait_time = 15 * retries
            print(f"  ... Retrying ({retries}/{MAX_RETRIES}) in {wait_time} seconds.", file=sys.stderr, flush=True)
            time.sleep(wait_time)
    
    return (lang_code, False, processed_items)


def redirection_consumer(db_params: Dict, progress: Dict):
    """
    キューからリダイレクションデータを取り出し、DBに一括挿入する
    各スレッドが独立したDB接続を持つ
    """
    tid = threading.get_ident()
    print(f"  [Redirection Consumer {tid}] Starting...", flush=True)
    
    try:
        with psycopg2.connect(**db_params) as conn:
            with conn.cursor() as cur:
                while True:
                    item = DATA_QUEUE.get(timeout=300)
                    if item is None:
                        DATA_QUEUE.put(None, timeout=10)
                        break
                    
                    data_type, lang_code, chunk = item
                    
                    if data_type == 'redirection':
                        psycopg2.extras.execute_values(
                            cur,
                            "INSERT INTO redirections(from_page_id, to_title, language_code) VALUES %s",
                            chunk
                        )
                        conn.commit()
                        
                        with progress['lock']:
                            progress['count'] += len(chunk)
                            if progress['total'] > 0:
                                print(f"  Processed {progress['count']:,} / {progress['total']:,} redirections...", flush=True)
                            else:
                                print(f"  Processed {progress['count']:,} redirections...", flush=True)
    
    except psycopg2.Error as e:
        print(f"\n  ✗ ERROR in redirection consumer {tid}: {e}", file=sys.stderr, flush=True)
        raise e
    
    print(f"\n  [Redirection Consumer {tid}] Finished.", flush=True)


def execute_redirection_downloads(plan: Dict):
    """リダイレクションのダウンロード計画を実行"""
    targets = plan['targets']
    if not targets:
        print("\nAll redirection data already exists.")
        return
    
    print("\n--- Executing Redirection Downloads ---")
    print(f"Using {NUM_CONSUMERS} parallel DB writers.")
    
    # 進捗管理
    total_redirections = sum(meta.get('redirections', 0) for _, meta in targets)
    progress = {
        "count": 0,
        "total": total_redirections,
        "lock": threading.Lock()
    }
    
    producer_results = []
    
    # Producer数を制限（同時ダウンロード数を制御）
    max_concurrent_producers = min(2, len(targets))
    
    with ThreadPoolExecutor(max_workers=NUM_CONSUMERS + max_concurrent_producers) as executor:
        # Consumer起動
        consumer_futures = [
            executor.submit(redirection_consumer, DB_PARAMS, progress) 
            for _ in range(NUM_CONSUMERS)
        ]
        
        # Producer起動（各言語ごと）
        producer_futures = [
            executor.submit(redirection_producer, lang_code, meta) 
            for lang_code, meta in targets
        ]
        
        # 各producerの結果を個別に収集
        for future in producer_futures:
            try:
                result = future.result()
                producer_results.append(result)
            except Exception as e:
                print(f"\n  ✗ Redirection producer failed with exception: {e}", file=sys.stderr, flush=True)
        
        print("\nAll redirection downloads finished. Signaling consumers to stop...", flush=True)
        DATA_QUEUE.put(None, timeout=10)
        
        # Consumer終了待ち
        for future in consumer_futures:
            try:
                future.result()
            except Exception as e:
                print(f"\n  ✗ Redirection consumer failed with exception: {e}", file=sys.stderr, flush=True)
    
    # 完了フラグを立てる（成功した言語のみ）
    for lang_code, is_complete, items in producer_results:
        if is_complete:
            mark_redirections_complete(lang_code)
            print(f"  ✓ Marked [{lang_code}] redirections as complete ({items:,} items).", flush=True)
        else:
            print(f"  ⚠ Skipping completion flag for [{lang_code}] redirections due to incomplete download.", file=sys.stderr, flush=True)


# ========================================
# メイン処理
# ========================================

def main():
    # 初期化フェーズ
    setup_directories()
    config = load_config()
    languages = config['source']['language']
    print(f"Target languages: {', '.join(languages)}")
    
    lang_metadata = download_languages()
    
    try:
        setup_database_schema()
    except psycopg2.Error as e:
        print(f"ERROR: Database setup failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # ページ処理フェーズ
    page_plan = plan_page_downloads(languages, lang_metadata)
    try:
        execute_page_downloads(page_plan)
    except Exception as e:
        print(f"ERROR: Page processing failed: {e}", file=sys.stderr)
        sys.exit(1)
    clear_queue()  # フェーズ間でキューをクリア
    
    # ドキュメント処理フェーズ
    doc_plan = plan_document_downloads(languages, lang_metadata)
    try:
        execute_document_downloads(doc_plan)
    except Exception as e:
        print(f"ERROR: Document processing failed: {e}", file=sys.stderr)
        sys.exit(1)
    clear_queue()  # フェーズ間でキューをクリア
    
    # リダイレクション処理フェーズ
    redir_plan = plan_redirection_downloads(languages, lang_metadata)
    try:
        execute_redirection_downloads(redir_plan)
    except Exception as e:
        print(f"ERROR: Redirection processing failed: {e}", file=sys.stderr)
        sys.exit(1)
    clear_queue()  # フェーズ間でキューをクリア
    
    # インデックス作成フェーズ（ページ）
    try:
        if not check_page_indexes_exist():
            create_page_indexes()
        else:
            print("\n--- Page indexes already exist. Skipping creation. ---", flush=True)
    except psycopg2.Error as e:
        print(f"ERROR: Page index creation failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # インデックス作成フェーズ（ドキュメント）
    try:
        if not check_indexes_exist():
            create_indexes()
        else:
            print("\n--- Document indexes already exist. Skipping creation. ---", flush=True)
    except psycopg2.Error as e:
        print(f"ERROR: Index creation failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # インデックス作成フェーズ（リダイレクション）
    try:
        if not check_redirection_indexes_exist():
            create_redirection_indexes()
        else:
            print("\n--- Redirection indexes already exist. Skipping creation. ---", flush=True)
    except psycopg2.Error as e:
        print(f"ERROR: Redirection index creation failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("\n--- All Processing Complete ---", flush=True)


if __name__ == "__main__":
    main()
