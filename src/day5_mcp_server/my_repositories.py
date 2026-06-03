# -*- coding: utf-8 -*-
"""
day5_mcp_server - Repositories (로봇 기술 LanceDB 고정 조회 함수)

[조회 대상 테이블: robot_multi_source_matrix (D:/lance_db/)]
  컬럼: vector, title, content, source_type, url, origin, publish_date

[검색 방식]
  - pandas 키워드 필터링(title/content ILIKE) — SQL 인젝션 위험 없음.
  - source_type 으로 소스 필터링(선택).

[보안 원칙]
  - 테이블 이름은 이 모듈 내부 상수로 고정(외부 인자 금지).
  - vector 컬럼은 반환하지 않는다(불필요한 고차원 데이터 노출 방지).
  - my_lance_db 를 통해서만 DB 에 접근한다.
"""
from my_lance_db import LanceDBUnavailableError, open_table  # noqa: F401

_TABLE = "robot_multi_source_matrix"

# 반환 허용 컬럼 (vector 제외)
_RETURN_COLS = ["title", "content", "source_type", "url", "origin", "publish_date"]


def _to_safe_records(df, limit: int) -> list[dict]:
    """vector 컬럼 제외 후 limit 건 dict 목록으로 반환한다."""
    cols = [c for c in _RETURN_COLS if c in df.columns]
    return df[cols].head(limit).to_dict(orient="records")


def search_robot_cache(query: str, source_type: str = None, limit: int = 10) -> list:
    """로봇 기술 캐시(robot_multi_source_matrix)에서 키워드로 검색한다.

    [검색] title / content 에 query 포함(대소문자 무시).
    [조건] source_type('irobotnews'/'arxiv'/'reddit' 등)으로 추가 필터 가능.
    [반환] list[dict] — vector 컬럼 제외, limit 건.
    """
    safe_limit = max(1, min(int(limit) if str(limit).isdigit() else 10, 100))
    table = open_table(_TABLE)
    df = table.to_pandas()

    if query:
        q = query.lower()
        mask = (
            df["title"].str.lower().str.contains(q, na=False) |
            df["content"].str.lower().str.contains(q, na=False)
        )
        df = df[mask]

    if source_type:
        df = df[df["source_type"].str.lower() == source_type.lower()]

    return _to_safe_records(df, safe_limit)


def list_robot_cache(source_type: str = None, limit: int = 20) -> list:
    """캐시 전체 목록을 반환한다(키워드 없는 조회용).

    [조건] source_type 으로 필터 가능.
    """
    safe_limit = max(1, min(int(limit) if str(limit).isdigit() else 20, 100))
    table = open_table(_TABLE)
    df = table.to_pandas()

    if source_type:
        df = df[df["source_type"].str.lower() == source_type.lower()]

    return _to_safe_records(df, safe_limit)


if __name__ == "__main__":
    from my_lance_db import is_lancedb_available
    print("[my_repositories] LanceDB 캐시 조회 테스트")
    if is_lancedb_available(use_cache=False):
        rows = search_robot_cache("로봇", limit=3)
        print(f"  search_robot_cache('로봇', limit=3) → {len(rows)}건")
        for r in rows:
            print(f"    [{r.get('source_type')}] {r.get('title','')[:50]}")
    else:
        print("  DB 미가동")
