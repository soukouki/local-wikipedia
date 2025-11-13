#!/usr/bin/env python3

import re
import yaml
import psycopg2
import logging
from contextlib import contextmanager
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount
import uvicorn
from typing import Optional, Tuple, List
from collections import OrderedDict

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s' # 時刻はどうせDockerログに入るので不要
)
logger = logging.getLogger(__name__)

DB_PARAMS = dict(host="localhost", port=5432, dbname="finewiki", user="dbuser", password="dbpass")

# config.yaml読み込み
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        LANGUAGES = config["source"]["language"]
        server = config.get("server", {})
        logger.info(f"Available languages from config: {LANGUAGES}")
        PORT = server.get("port", 29423)
        logger.info(f"Server port from config: {PORT}")
        MAX_SEARCH_RESULTS = server.get("max_search_results", 20)
        logger.info(f"Max search results from config: {MAX_SEARCH_RESULTS}")
    logger.info(f"Loaded languages from config: {LANGUAGES}")
except Exception as e:
    logger.error(f"Failed to load config.yaml: {e}")
    raise

# 利用可能な言語リストを文字列化
AVAILABLE_LANGUAGES_STR = "[" + ", ".join(LANGUAGES) + "]"

mcp = FastMCP("wikipedia-mcp")


# ========================================
# データベース接続ヘルパー
# ========================================

@contextmanager
def db_cursor(autocommit: bool = False):
    """
    データベースカーソルを提供（接続とカーソルを自動管理）
    
    Args:
        autocommit: 自動コミットするかどうか（デフォルト: False）
    
    Yields:
        psycopg2.cursor: データベースカーソル
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        logger.debug("Database connection established")
        with conn.cursor() as cur:
            yield cur
            if not autocommit:
                conn.commit()
    except Exception as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed")


# ========================================
# データアクセス層
# ========================================

def get_document_by_title(cur, title: str, lang: str) -> Optional[Tuple[str, str]]:
    """
    タイトルから記事を取得
    
    Args:
        cur: データベースカーソル
        title: 記事タイトル
        lang: 言語コード
    
    Returns:
        (title, text_body) または None
    """
    logger.debug(f"DB Query: get_document_by_title with title='{title}', lang='{lang}'")
    # Wikipedia内部でのタイトルは"_"で区切られているが、finewikiのdocumentsテーブルではスペースで保存されているため、変換する
    title = title.replace("_", " ")
    cur.execute(
        "SELECT title, text_body FROM documents WHERE title = %s AND language_code = %s LIMIT 1",
        (title, lang)
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else None


def get_page_id_by_title(cur, title: str, lang: str) -> Optional[int]:
    """
    タイトルから page_id を取得
    
    Args:
        cur: データベースカーソル
        title: 記事タイトル
        lang: 言語コード
    
    Returns:
        page_id または None
    """
    normalized_title = normalize_title_for_page(title)
    logger.debug(f"DB Query: get_page_id_by_title with title='{normalized_title}', lang='{lang}'")
    cur.execute(
        "SELECT page_id FROM pages WHERE page_title = %s AND language_code = %s LIMIT 1",
        (normalized_title, lang)
    )
    row = cur.fetchone()
    return row[0] if row else None


def get_redirect_target(cur, page_id: int, lang: str) -> Optional[str]:
    """
    page_id からリダイレクト先タイトルを取得
    
    Args:
        cur: データベースカーソル
        page_id: ページID
        lang: 言語コード
    
    Returns:
        リダイレクト先タイトル または None
    """
    logger.debug(f"DB Query: get_redirect_target with page_id='{page_id}', lang='{lang}'")
    cur.execute(
        "SELECT to_title FROM redirections WHERE from_page_id = %s AND language_code = %s LIMIT 1",
        (page_id, lang)
    )
    row = cur.fetchone()
    return row[0] if row else None


def search_exact_match(cur, query: str, lang: str) -> Optional[Tuple[str, str]]:
    """
    完全一致検索
    
    Args:
        cur: データベースカーソル
        query: 検索クエリ
        lang: 言語コード
    
    Returns:
        (title, text_body) または None
    """
    logger.debug(f"DB Query: search_exact_match with query='{query}', lang='{lang}'")
    cur.execute(
        "SELECT title, text_body FROM documents WHERE title = %s AND language_code = %s LIMIT 1",
        (query, lang)
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else None


def search_title_match(cur, query: str, lang: str, limit: int) -> List[Tuple[str, str]]:
    """
    タイトル部分一致検索
    
    Args:
        cur: データベースカーソル
        query: 検索クエリ
        lang: 言語コード
        limit: 最大取得件数
    
    Returns:
        [(title, snippet), ...] のリスト
    """
    if not query: return [] # 空クエリによるエラーを防止
    logger.debug(f"DB Query: search_title_match with query='{query}', lang='{lang}', limit={limit}")
    cur.execute(
        "SELECT title, pgroonga_snippet_html(title, pgroonga_query_extract_keywords(%s)) "
        "FROM documents WHERE title &@~ %s AND language_code = %s LIMIT %s",
        (query, query, lang, limit)
    )
    return [(row[0], row[1][0]) for row in cur.fetchall()]


def search_redirect_match(cur, query: str, lang: str, limit: int) -> List[Tuple[str, str]]:
    """
    リダイレクト一致検索
    
    Args:
        cur: データベースカーソル
        query: 検索クエリ
        lang: 言語コード
        limit: 最大取得件数
    
    Returns:
        [(from_title, to_title), ...] のリスト
    """
    normalized_query = normalize_title_for_page(query)
    logger.debug(f"DB Query: search_redirect_match with query='{normalized_query}', lang='{lang}', limit={limit}")
    cur.execute(
        """
        SELECT p.page_title, r.to_title
        FROM pages p
        JOIN redirections r ON p.page_id = r.from_page_id AND p.language_code = r.language_code
        WHERE (p.page_title ILIKE %s OR p.page_title = %s) AND p.language_code = %s
        LIMIT %s
        """,
        (f"%{normalized_query}%", normalized_query, lang, limit)
    )
    return [(row[0].replace("_", " "), row[1]) for row in cur.fetchall()]


def search_body_match(cur, query: str, lang: str, exclude_title: str, limit: int) -> List[Tuple[str, str]]:
    """
    本文一致検索
    
    Args:
        cur: データベースカーソル
        query: 検索クエリ
        lang: 言語コード
        exclude_title: 除外するタイトル（完全一致を避けるため）
        limit: 最大取得件数
    
    Returns:
        [(title, snippet), ...] のリスト
    """
    if not query: return [] # 空クエリによるエラーを防止
    logger.debug(f"DB Query: search_body_match with query='{query}', lang='{lang}', exclude_title='{exclude_title}', limit={limit}")
    cur.execute(
        "SELECT title, pgroonga_snippet_html(text_body, pgroonga_query_extract_keywords(%s)) "
        "FROM documents WHERE text_body &@~ %s AND language_code = %s AND title != %s LIMIT %s",
        (query, query, lang, exclude_title, limit)
    )
    return [(row[0], row[1][0]) for row in cur.fetchall()]


def get_random_article(cur, langs: List[str]) -> Optional[Tuple[str, str]]:
    """
    ランダム記事を取得
    
    Args:
        cur: データベースカーソル
        langs: 言語コード
    
    Returns:
        (title, text_body) または None
    """
    logger.debug(f"DB Query: get_random_article with langs='{langs}'")
    # documentsテーブルからその言語の中でランダムに1件取得
    cur.execute(
        "SELECT title, text_body FROM documents "
        "WHERE language_code = ANY(%s) "
        "ORDER BY RANDOM() LIMIT 1",
        (langs,)
    )
    row = cur.fetchone()
    return (row[0], row[1]) if row else None


# ========================================
# ヒューリスティック検索ロジック
# ========================================
def generate_heuristic_queries(query: str, search_languages: List[str]) -> List[str]:
    """
    元のクエリから、ヒューリスティックに基づいて複数の検索候補を生成する。
    優先度順にソートされたクエリのリストを返す。

    Args:
        query: 元の検索クエリ
        search_languages: 検索対象言語リスト

    Returns:
        検索クエリ候補のリスト
    """
    if not query:
        return []

    # OrderedDictを使い、順序を維持しつつ重複を排除
    queries = OrderedDict()
    queries[query.strip()] = None  # 元のクエリを最優先

    # 括弧内の文字列を後で追加するために保持（優先度を下げる）
    bracket_contents = []
    
    # CJKが含まれるかチェック
    contains_cjk = any(lang in ['ja', 'zh', 'ko'] for lang in search_languages)
    meaningful_length = 3 if contains_cjk else 6

    # 1. 言語コード `(ja)` などを除去
    lang_code_pattern = re.compile(r'(.+?)\s+\(([a-z]{2,3})\)$', re.IGNORECASE)
    match = lang_code_pattern.match(query)
    if match:
        stripped_query = match.group(1).strip()
        queries[stripped_query] = None

    # 2. 括弧の処理
    bracket_pairs = [('「', '」'), ('『', '』'), ('(', ')'), ('[', ']'), ('【', '】')]
    
    # 現在の候補リストをコピーしてイテレート
    current_queries = list(queries.keys())
    for q in current_queries:
        # 2a. まず括弧内の内容を抽出して保持（後で追加するため）
        for start, end in bracket_pairs:
            escaped_start = re.escape(start)
            escaped_end = re.escape(end)
            inner_matches = re.findall(f'{escaped_start}(.+?){escaped_end}', q)
            for inner in inner_matches:
                inner_stripped = inner.strip()
                # 抽出した文字列が短すぎる場合は無視
                if inner_stripped and len(inner_stripped) >= 2:
                    bracket_contents.append(inner_stripped)
        
        # 2b. 括弧を単純に除去したバージョン（こちらを先に追加して優先度を上げる）
        stripped_q = q
        for start, end in bracket_pairs:
            stripped_q = stripped_q.replace(start, "").replace(end, "")
        
        stripped_q = stripped_q.strip()
        if stripped_q and stripped_q != q:
            queries[stripped_q] = None
    
    # 3. その他LLMの誤りやすいパターン（冗長な表現）を除去
    current_queries = list(queries.keys())
    for q in current_queries:
        modified_q = q
        
        # 多言語対応の接頭辞・接尾辞パターン
        prefix_patterns = [
            r"^(?:tell me about|explain|what is|what's|describe)\s+", # en
            r"^(?:was ist|erkläre mir|erzähl mir von|beschreibe)\s+", # de
            r"^(?:qu'est-ce que|qu'est-ce qu'un|qu'est-ce qu'une|parlez-moi de|expliquez-moi|décris-moi)\s+", # fr
            r"^(?:что такое|расскажи(?:те)? о|опиши(?:те)?)\s+", # ru
            r"^(?:qué es|explícame|háblame de|descríbeme)\s+", # es
            r"^(?:che cos'è|cos'è|spiegami|parlami di|descrivimi)\s+", # it
        ]
        suffix_patterns = [
            r"(?:について(?:教えて|おしえて)?|とは|を(?:教えて|おしえて|調べて|しらべて))$", # ja
            r"\s+(?:about|on the topic of|regarding)$", # en
            r"\s+(?:über|bezüglich|hinsichtlich)$", # de
            r"\s+(?:sur|à propos de|concernant)$", # fr
            r"\s+(?:о|об|про)$", # ru (限定的)
            r"\s+(?:sobre|acerca de)$", # es
            r"\s+(?:su|riguardo a)$", # it
        ]
        split_patterns = [
            r"\s+of\s+", # en
            r"[のでにをは]", # ja
            r"\s+von\s+", # de
            r"\s+de\s+", # fr, es
            r"\s+di\s+", # it
            r"\s+о\s+", # ru
            r"\s*[_\-<>|:;/\\]\s*", # 記号類
        ]
        
        # 接頭辞の除去
        for pattern in prefix_patterns:
            modified_q = re.sub(pattern, '', modified_q, flags=re.IGNORECASE)
        # 接尾辞の除去
        for pattern in suffix_patterns:
            modified_q = re.sub(pattern, '', modified_q, flags=re.IGNORECASE)

        # "of" や "の" で分割して主要部分を抽出
        for pattern in split_patterns:
            parts = re.split(pattern, modified_q, flags=re.IGNORECASE)
            # 長さが指定文字以上の部分について抽出
            valid_parts = []
            for part in parts:
                part = part.strip()
                # 分割後のパーツからも括弧を除去
                for start, end in bracket_pairs:
                    part = part.replace(start, "").replace(end, "")
                part = part.strip()
                if len(part) >= meaningful_length:
                    valid_parts.append(part)
            
            # CJKの場合は前方優先（出現順）、それ以外は長い順
            if contains_cjk:
                # 出現順で追加（前方に出現した文字列を優先）
                for part in valid_parts:
                    if part and part != q:
                        queries[part] = None
            else:
                # 長い順にソート
                valid_parts.sort(key=len, reverse=True)
                for part in valid_parts:
                    if part and part != q:
                        queries[part] = None

        # もし検索言語が日本語・中国語・韓国語なら、スペースでの分割も試みる
        if contains_cjk:
            space_parts = modified_q.split()
            # 前方から順に追加（出現順を優先）
            for part in space_parts:
                part = part.strip()
                # スペース分割後のパーツからも括弧を除去
                for start, end in bracket_pairs:
                    part = part.replace(start, "").replace(end, "")
                part = part.strip()
                if len(part) >= meaningful_length and part not in queries:
                    queries[part] = None
        
        # 処理後に残った両端の括弧や引用符を除去
        bracket_chars = "「」『』\"'()[]【】"
        modified_q = modified_q.strip(bracket_chars)

        modified_q = modified_q.strip()
        if modified_q and modified_q != q:
            queries[modified_q] = None

    # 4. 最後に括弧内の文字列を追加（優先度を最も低くする）
    for content in bracket_contents:
        if content not in queries:
            queries[content] = None

    final_queries = list(queries.keys())
    logger.info(f"Generated heuristic queries for '{query}': {final_queries}")
    return final_queries


# ========================================
# ビジネスロジック層
# ========================================

def normalize_title_for_page(title: str) -> str:
    """
    Wikipediaのpage_title形式に正規化（スペース→アンダースコア、先頭大文字化）
    
    Args:
        title: 元のタイトル
    
    Returns:
        正規化されたタイトル
    """
    normalized = title.replace(" ", "_")
    if normalized:
        normalized = normalized[0].upper() + normalized[1:]
    return normalized


def resolve_redirect(cur, title: str, lang: str) -> Optional[Tuple[str, str]]:
    """
    リダイレクト先のタイトルと元タイトルを取得
    
    Args:
        cur: データベースカーソル
        title: 元のタイトル
        lang: 言語コード
    
    Returns:
        (redirect_target_title, original_title) または None
    """
    page_id = get_page_id_by_title(cur, title, lang)
    if not page_id:
        return None
    
    redirect_target = get_redirect_target(cur, page_id, lang)
    if redirect_target:
        logger.info(f"Redirect found: {title} -> {redirect_target} (lang: {lang})")
        return (redirect_target, title)
    
    return None


def validate_languages(languages: Optional[List[str]]) -> Tuple[bool, List[str], str]:
    """
    言語リストを検証し、検索対象言語を決定
    
    Args:
        languages: 言語コードリスト（オプション）
    
    Returns:
        (is_valid, search_languages, error_message)
    """
    if languages is None or languages == []:
        return (True, LANGUAGES, "")
    
    validated_languages = [lang.lower() for lang in languages]
    for lang in validated_languages:
        if lang not in LANGUAGES:
            error_msg = f"Error: Language '{lang}' is not available. Available languages: {AVAILABLE_LANGUAGES_STR}"
            return (False, [], error_msg)
    return (True, validated_languages, "")


def normalize_languages_input(languages: Optional[list[str] | str]) -> Optional[list[str]]:
    """
    languages引数を正規化してlist[str]に変換する。
    
    Args:
        languages: 言語指定（list, str, またはNone）
    
    Returns:
        正規化されたlist[str]、またはNone
    """
    if languages is None:
        return None
    
    # すでにリストの場合
    if isinstance(languages, list):
        return languages
    
    # 文字列の場合
    if isinstance(languages, str):
        # 空白を除去してから処理
        languages = languages.strip()
        
        if not languages:
            return None
        
        # カンマ区切り（スペース有無両対応: "en,ja" or "en, ja, de"）
        if ',' in languages:
            return [lang.strip() for lang in languages.split(',') if lang.strip()]
        
        # スペース区切り（言語コードには空白が含まれないため安全に分割可能）
        if ' ' in languages:
            return [lang.strip() for lang in languages.split() if lang.strip()]
        
        # 単一の言語コード
        return [languages]
    
    # その他の型の場合はログに記録してNoneを返す
    logger.warning(f"Unexpected type for languages: {type(languages)}, value: {languages}")
    return None


# ========================================
# プレゼンテーション層
# ========================================

def extract_summary(text_body: str) -> str:
    """
    記事本文から概要を抽出（最初の200文字まで）
    
    Args:
        text_body: 記事本文
    
    Returns:
        抽出された概要テキスト
    """
    # 概要は##までのテキストとし、200文字までに制限
    heading_pattern = re.compile(r'^(#{2,6})\s+(.*)$', re.MULTILINE)
    match = heading_pattern.search(text_body)
    snippet_end = match.start() if match else len(text_body)
    snippet = text_body[:snippet_end][:200]
    return snippet

def extract_snippet(text_body: str, is_full_text: bool = False) -> str:
    """
    記事本文からスニペットを抽出
    
    Args:
        text_body: 記事本文
        is_full_text: True の場合は全文、False の場合は概要のみ
    
    Returns:
        抽出されたテキスト
    """
    if is_full_text:
        return text_body
    # Markdown記法から概要を抽出し、もし短い(200文字以内)なら2番めの見出し(ただし1000文字以降はカット)まで含める
    # どちらにせよ省略した見出しがある場合は見出し一覧を箇条書き形式で追加(章・節・項の区別をするためにレベルに応じたインデントを付与)
    # finewikiのテキストは以下のような構造になっている
    # ---
    # # タイトル
    # 本文（概要）
    # ## 見出し1
    # 本文（見出し1の内容）
    # ...
    heading_pattern = re.compile(r'^(#{2,6})\s+(.*)$', re.MULTILINE)
    headings = list(heading_pattern.finditer(text_body))

    if not headings:
        snippet = text_body[:200]
        if len(snippet) < len(text_body):
            snippet += "..."
        return snippet

    first_heading = headings[0]
    snippet_end = first_heading.start()
    if len(text_body[:snippet_end]) <= 200 and len(headings) > 1:
        second_heading = headings[1]
        snippet_end = second_heading.start()
        if snippet_end > 1000:
            snippet_end = first_heading.start() + 1000
    snippet = text_body[:snippet_end]
    if len(snippet) < len(text_body):
        snippet += "..."

    # 省略された見出しを箇条書きで追加
    if len(headings) > 1:
        snippet += "\n## Omitted Headings\n"
        for heading in headings[1:]:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            indent = "  " * (level - 2) # Open WebUIでは行頭の空白が表示されないのでバグに見えるが、きちんとインデントされている
            snippet += f"{indent}- {title}\n"
        snippet += "\nIf you want to read more, please use the `search_article` tool with `is_full_text=True`.\n"
    return snippet


def format_html_snippet(html_snippet: str, max_length: int = 200) -> str:
    """
    HTMLスニペットを整形（タグ除去、強調変換、長さ制限）
    
    Args:
        html_snippet: HTMLスニペット
        max_length: 最大文字数
    
    Returns:
        整形されたテキスト
    """
    formatted = html_snippet.replace('<span class="keyword">', '**').replace('</span>', '**')
    if len(formatted) > max_length:
        formatted = formatted[:max_length] + "..."
    return formatted


def format_article_with_redirect_notice(text_body: str, from_title: str, to_title: str, is_full_text: bool) -> str:
    """
    リダイレクト通知付きで記事を整形
    
    Args:
        text_body: 記事本文
        from_title: リダイレクト元タイトル
        to_title: リダイレクト先タイトル
        is_full_text: 全文表示かどうか
    
    Returns:
        整形された記事テキスト
    """
    redirect_notice = f"(Redirected from '{from_title}' to '{to_title}')\n\n"
    snippet = extract_snippet(text_body, is_full_text)
    return redirect_notice + snippet


# ========================================
# MCPツール
# ========================================

@mcp.tool()
def search_article(
    title: str,
    is_full_text: bool = False,
    languages: Optional[list[str] | str] = None,
) -> str:
    f"""
    Search and read a Wikipedia article by title. The search process includes exact title match, redirect resolution, partial title match, and full-text search.
    
    Args:
        title: Article title to read. such as "Wikipedia"
        is_full_text: Set to false to output only summaries. Defaults to false.
        languages: Specific language code list (optional). Available languages: {AVAILABLE_LANGUAGES_STR}. If specified, **it must always be in list format**.
    
    If user want to search in detail, **please set `is_full_text=True`** to read the full text of the article.
    
    **Be careful when setting arguments when using the tool**.
    """
    logger.info(f"search_article called with title: {title}, languages: {languages}, is_full_text: {is_full_text}")

    # 言語パラメータを正規化
    normalized_languages = normalize_languages_input(languages)
    logger.info(f"Normalized languages: {normalized_languages}")

    # 言語パラメータを検証
    is_valid, search_languages, error_msg = validate_languages(normalized_languages)
    if not is_valid:
        return error_msg
    
    # ヒューリスティックに基づくクエリバリエーション生成
    queries_to_try = generate_heuristic_queries(title, search_languages)
    if not queries_to_try:
        return f"Article not found: Invalid title '{title}'"
        
    try:
        with db_cursor() as cur:

            # 素直にクエリバリエーションと各言語で順次検索
            for query_variant in queries_to_try:
                for lang in search_languages:

                    # 1. タイトルで完全一致検索
                    logger.info(f"Trying exact match for '{query_variant}' in {lang}")
                    result = get_document_by_title(cur, query_variant, lang)
                    if result:
                        found_title, text_body = result
                        snippet = extract_snippet(text_body, is_full_text)
                        notice = ""
                        if query_variant != title:
                            notice = f"(Found article '{found_title}' based on your query '{title}')\n\n"
                        logger.info(f"Article found: {found_title} in {lang}")
                        return notice + snippet
                    
                    # 2. リダイレクトで完全一致検索
                    logger.info(f"Checking redirect for '{query_variant}' in {lang}")
                    redirect_info = resolve_redirect(cur, query_variant, lang)
                    if redirect_info:
                        redirect_title, original_title_from_redirect = redirect_info
                        logger.info(f"Following redirect: {query_variant} -> {redirect_title} in {lang}")
                        
                        article_result = get_document_by_title(cur, redirect_title, lang)
                        if article_result:
                            _, text_body = article_result
                            logger.info(f"Article found via redirect: {query_variant} -> {redirect_title} in {lang}")
                            
                            from_display = f"'{original_title_from_redirect}'"
                            if query_variant != title:
                                from_display = f"'{query_variant}' (from query '{title}')"

                            return format_article_with_redirect_notice(text_body, from_display, redirect_title, is_full_text)

            # それでも見つからなければ、部分一致で補足検索
            results = []
            for query_variant in queries_to_try:
                for lang in search_languages:

                    # 3. タイトル部分一致検索
                    logger.info(f"Trying title match for '{query_variant}' in {lang}")
                    title_matches = search_title_match(cur, query_variant, lang, MAX_SEARCH_RESULTS - len(results))
                    for match_title, snippet in title_matches:
                        results.append(f"## [Title Match] {match_title} ({lang})\n{format_html_snippet(snippet)}\n")
                        if len(results) >= MAX_SEARCH_RESULTS:
                            break

                    # 4. リダイレクト部分一致検索
                    logger.info(f"Trying redirect match for '{query_variant}' in {lang}")
                    redirect_matches = search_redirect_match(cur, query_variant, lang, MAX_SEARCH_RESULTS - len(results))
                    for from_t, to_t in redirect_matches:
                        doc = get_document_by_title(cur, to_t, lang)
                        # ここではsnippetよりも短いsummaryを表示
                        summary = extract_summary(doc[1]) if doc else "(Article not found)"
                        results.append(f"## [Redirect Match] {from_t} -> {to_t} ({lang})\n{summary}\n")
                        if len(results) >= MAX_SEARCH_RESULTS:
                            break
            
            # それでも20件に満たなければ、本文部分一致検索
            for query_variant in queries_to_try:
                for lang in search_languages:
                    logger.info(f"Trying body match for '{query_variant}' in {lang}")
                    body_matches = search_body_match(cur, query_variant, lang, query_variant, MAX_SEARCH_RESULTS - len(results))
                    for match_title, snippet in body_matches:
                        results.append(f"## [Body Match] {match_title} ({lang})\n{format_html_snippet(snippet)}\n")
                        if len(results) >= MAX_SEARCH_RESULTS:
                            break
            
            if results:
                logger.info(f"Partial matches found for title: {title}")
                return "The following articles were found in your search:\n\n" + "\n---\n".join(results)
            
            logger.warning(f"Article not found for any variation of: {title}")
            return f"Article not found: {title}\nPlease try different keywords."
    except Exception as e:
        logger.error(f"Error in search_article: {e}", exc_info=True)
        return f"Error reading article: {str(e)}"


@mcp.tool()
def read_random_article(
    languages: Optional[list[str] | str] = None,
) -> str:
    f"""
    Read a random Wikipedia article. Automatically excludes redirect pages to ensure actual content is returned.
    
    Args:
        languages: Specific language code list (optional). Available languages: {AVAILABLE_LANGUAGES_STR}. If specified, **it must always be in list format**.

    Since this tool returns a random article each time it's called, if you want to view the same article again in detail, **NEVER USE THIS TOOL** and **ALWAYS USE THE `search_article` FUNCTION** instead**.
    """
    logger.info(f"read_random_article called with languages: {languages}")

    # 言語パラメータを正規化
    normalized_languages = normalize_languages_input(languages)
    logger.info(f"Normalized languages: {normalized_languages}")
    
    # 言語パラメータを検証
    is_valid, search_languages, error_msg = validate_languages(normalized_languages)
    if not is_valid:
        return error_msg
    
    try:
        with db_cursor() as cur:
            logger.info(f"Fetching random article from languages: {search_languages}")
            result = get_random_article(cur, search_languages)

            if result:
                title, text_body = result
                summary = extract_snippet(text_body)
                logger.info(f"Random article found: {title}")
                return f"Random Article: '{title}'\n\n{summary}"
            
            logger.warning("No articles found in any language")
            return "No articles found"
    except Exception as e:
        logger.error(f"Error in read_random_article: {e}", exc_info=True)
        return f"Error reading random article: {str(e)}"


# ========================================
# アプリケーション起動
# ========================================

app = Starlette(
    routes=[
        Mount("/", app=mcp.sse_app()),
    ]
)

if __name__ == "__main__":
    logger.info("Starting MCP Wikipedia server...")
    print("Starting MCP Wikipedia server...")
    uvicorn.run("local-wikipedia:app", host="0.0.0.0", port=PORT, log_level="info")
