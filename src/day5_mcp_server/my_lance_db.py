# -*- coding: utf-8 -*-
"""
day5_mcp_server - LanceDB Access (로컬 벡터 DB 접근 계층)

[이 모듈의 역할]
my_repositories.py 가 사용할 LanceDB 저수준 접근 계층입니다.
PostgreSQL 과 달리 별도 서버 없이 로컬 디렉터리(D:/lance_db)에 데이터를 저장합니다.

[mcp_server03 db_access.py 와의 대응 관계]
  mcp_server03 db_access  →  이 모듈
  get_connection()         →  get_db()
  fetch_all()              →  search_table() / scan_table()
  DBUnavailableError       →  LanceDBUnavailableError
  is_postgres_available()  →  is_lancedb_available()

[저장 위치]
  LANCE_DB_PATH = "D:/lance_db"
  테이블별 서브디렉터리가 자동 생성됩니다.

[보안 원칙]
- raw SQL / 임의 쿼리를 외부 Tool argument 로 받지 않는다.
  my_repositories 의 고정 함수만 이 모듈을 호출한다.
- 검색어는 pandas 필터링으로만 처리한다(문자열 결합 → SQL 이 없으므로 인젝션 위험 없음).
- DB 경로(LANCE_DB_PATH)는 환경변수로 override 가능하지만 출력에 노출하지 않는다.

[예외 정책]
- LanceDBUnavailableError: 경로 없음·import 실패. 호출부가 빈 list 로 fallback.
- ValueError: 테이블 없음·형식 오류. 숨기지 않고 전달.
"""
from __future__ import annotations

import os
from pathlib import Path

# LanceDB 선택 의존성 — 미설치 시 LanceDBUnavailableError 로 안전 처리
try:
    import lancedb
    _HAS_LANCEDB = True
except ImportError:
    lancedb = None
    _HAS_LANCEDB = False

LANCE_DB_PATH = os.getenv("LANCE_DB_PATH", "D:/lance_db")

# 가용성 캐시 (DB 경로 존재 여부는 런타임 중 잘 바뀌지 않으므로 1회 확인 후 재사용)
_availability_cache: dict = {"checked": False, "value": False}


class LanceDBUnavailableError(Exception):
    """LanceDB 접근 불가(경로 없음 / 패키지 미설치). fallback 대상 신호."""


def get_db():
    """LanceDB 연결을 생성한다.

    [반환] lancedb.LanceDBConnection 객체.
    [예외] 패키지 미설치 또는 경로 오류 시 LanceDBUnavailableError.
    """
    if not _HAS_LANCEDB:
        raise LanceDBUnavailableError("lancedb 패키지가 설치되지 않았습니다(pip install lancedb).")
    try:
        return lancedb.connect(LANCE_DB_PATH)
    except Exception as error:
        raise LanceDBUnavailableError(f"LanceDB 연결 실패: {type(error).__name__}") from None


def open_table(table_name: str):
    """테이블을 열어 반환한다. 없으면 ValueError.

    [보안] table_name 은 my_repositories 내부 상수만 전달한다(외부 인자 금지).
    """
    db = get_db()
    existing = db.table_names()
    if table_name not in existing:
        raise ValueError(f"테이블 '{table_name}' 이 존재하지 않습니다. 먼저 데이터를 적재하세요.")
    return db.open_table(table_name)


def search_table(table_name: str, query: str, filter_col: str = None,
                 filter_val: str = None, limit: int = 10) -> list[dict]:
    """테이블에서 키워드 검색(pandas ILIKE 방식)을 수행한다.

    [동작]
        1) table_name 의 LanceDB 테이블을 pandas DataFrame 으로 로드한다.
        2) 모든 문자열 컬럼에서 query 를 대소문자 무시 포함 검색한다.
        3) filter_col / filter_val 로 추가 필터링한다(카테고리·소스 등).
        4) 최대 limit 건 반환한다.

    [입력]
        table_name : 테이블 이름(my_repositories 상수만 전달).
        query      : 검색 키워드(사용자 입력 — 문자열 비교만 하므로 인젝션 위험 없음).
        filter_col : 추가 필터 컬럼 이름(없으면 None).
        filter_val : 추가 필터 값(없으면 None).
        limit      : 반환 최대 건수(안전 상한 100 적용).
    [반환] list[dict]. 결과 없으면 빈 list.
    [예외]
        LanceDBUnavailableError : DB 접근 불가(호출부가 빈 list fallback).
        ValueError              : 테이블 없음 등(숨기지 않음).
    """
    safe_limit = max(1, min(int(limit) if str(limit).isdigit() else 10, 100))
    table = open_table(table_name)
    try:
        df = table.to_pandas()
    except Exception as error:
        raise ValueError(f"테이블 로드 실패: {type(error).__name__}") from None

    # query 키워드 포함 검색 (모든 object 컬럼 대상, 대소문자 무시)
    if query and not df.empty:
        q_lower = query.lower()
        str_cols = df.select_dtypes(include="object").columns
        mask = df[str_cols].apply(
            lambda col: col.str.lower().str.contains(q_lower, na=False)
        ).any(axis=1)
        df = df[mask]

    # 추가 필터 (카테고리 / 소스 등)
    if filter_col and filter_val and filter_col in df.columns:
        df = df[df[filter_col].str.lower() == filter_val.lower()]

    return df.head(safe_limit).to_dict(orient="records")


def scan_table(table_name: str, filter_col: str = None,
               filter_val: str = None, limit: int = 20) -> list[dict]:
    """테이블 전체를 스캔한다(키워드 없는 목록 조회용)."""
    safe_limit = max(1, min(int(limit) if str(limit).isdigit() else 20, 100))
    table = open_table(table_name)
    try:
        df = table.to_pandas()
    except Exception as error:
        raise ValueError(f"테이블 로드 실패: {type(error).__name__}") from None

    if filter_col and filter_val and filter_col in df.columns:
        df = df[df[filter_col].str.lower() == filter_val.lower()]

    return df.head(safe_limit).to_dict(orient="records")


def is_lancedb_available(use_cache: bool = True) -> bool:
    """LanceDB 접근 가능 여부를 반환한다(예외 없이 True/False).

    패키지 설치 여부 + D:/lance_db 경로 존재 여부를 함께 확인한다.
    """
    if use_cache and _availability_cache["checked"]:
        return _availability_cache["value"]
    available = False
    try:
        if _HAS_LANCEDB and Path(LANCE_DB_PATH).exists():
            get_db()
            available = True
    except Exception:
        available = False
    _availability_cache["checked"] = True
    _availability_cache["value"] = available
    return available


def reset_availability_cache() -> None:
    """가용성 캐시를 초기화한다(테스트/재확인용)."""
    _availability_cache["checked"] = False
    _availability_cache["value"] = False


if __name__ == "__main__":
    print("[my_lance_db] LanceDB 가용성 확인")
    print(f"  lancedb 설치: {_HAS_LANCEDB}")
    print(f"  DB 경로: {LANCE_DB_PATH}  존재: {Path(LANCE_DB_PATH).exists()}")
    print(f"  is_lancedb_available={is_lancedb_available(use_cache=False)}")
    if is_lancedb_available(use_cache=False):
        db = get_db()
        print(f"  테이블 목록: {db.list_tables()}")
