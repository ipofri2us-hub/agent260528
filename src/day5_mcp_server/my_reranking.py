# -*- coding: utf-8 -*-
"""
day5_mcp_server - 운영형 재정렬(reranking) 모듈

[출처/유래]
이 파일은 Day5 mcp_server02/rag_search/reranking.py 의 설계를 참고해,
크롤러(RSS/arXiv) 결과에 맞게 재작성한 운영형 reranker 입니다.

[mcp_server02 reranking 과의 차이]
- mcp_server02: Chroma 벡터 검색 chunk 를 alarm_code/equipment_type/metadata 로 재정렬.
- 이 모듈: RSS/arXiv 크롤러 결과(title·preview·summary)를 query 관련성과
  소스 선호도로 재정렬. 정답 라벨(expected_*) 없이 query 만 사용한다.

[운영형 정렬이 쓰는 신호 — 정답 라벨이 아님]
  1) query 키워드가 title 에 포함되면 가점(+3)
  2) query 키워드가 preview/summary 에 포함되면 가점(+2)
  3) query 의도(학술·커뮤니티·한국·업계)에 맞는 소스면 가점(+2)
  4) 동점은 원래 크롤러 반환 순서를 유지(안정 정렬)

import 계층 규칙:
  my_reranking.py 는 표준 라이브러리만 import 합니다(다른 my_* 모듈을 import하지 않음).
  → 단방향 구조라 순환 import 가 생기지 않습니다.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# 소스 routing 키워드 (정답 라벨이 아닌 query 의도 추론용)
# ---------------------------------------------------------------------------
_ACADEMIC_KEYWORDS = [
    "논문", "연구", "알고리즘", "학술", "리뷰", "실험",
    "paper", "research", "algorithm", "study", "survey", "experiment",
]
_COMMUNITY_KEYWORDS = [
    "토론", "커뮤니티", "의견", "추천", "질문", "포럼",
    "reddit", "forum", "discussion", "community", "opinion", "recommend",
]
_KOREAN_KEYWORDS = [
    "한국", "국내", "한국어", "국산", "로봇신문",
    "korean", "korea", "domestic",
]
_INDUSTRY_KEYWORDS = [
    "뉴스", "출시", "제품", "기업", "시장", "산업", "동향",
    "news", "launch", "product", "company", "market", "industry", "trend",
]


def infer_preferred_sources_for_query(query: str) -> list[str]:
    """사용자 query 만으로 우선 검색 소스(tool_name)를 추론한다(운영형 소스 routing).

    [반환]
        우선 tool_name 리스트(중복 제거, 등장 순서 유지). 매칭 없으면 빈 리스트.
        현재 지식베이스(4개 소스):
          - search_arxiv       : 학술 논문 / 연구 / 알고리즘
          - search_reddit      : 커뮤니티 토론 / 의견 / 추천
          - search_irobotnews  : 한국어 뉴스 / 국내 동향
          - search_robot_news  : 영문 업계 뉴스 / 제품·시장
    """
    text = str(query or "").lower()
    preferred: list[str] = []

    if any(kw in text for kw in _ACADEMIC_KEYWORDS):
        preferred.append("search_arxiv")
    if any(kw in text for kw in _COMMUNITY_KEYWORDS):
        preferred.append("search_reddit")
    if any(kw in text for kw in _KOREAN_KEYWORDS):
        preferred.append("search_irobotnews")
    if any(kw in text for kw in _INDUSTRY_KEYWORDS):
        preferred.append("search_robot_news")

    result: list[str] = []
    for item in preferred:
        if item not in result:
            result.append(item)
    return result


def _query_tokens(query: str) -> list[str]:
    """query 를 소문자 토큰으로 분리한다(길이 2 이상만, 중복 제거).

    너무 짧은 토큰(조사/한 글자)은 잡음이 많아 제외한다.
    title/preview 텍스트에 query 핵심어가 있는지 보는 가벼운 보조 신호다.
    """
    raw = str(query or "").lower().split()
    tokens: list[str] = []
    for token in raw:
        token = token.strip()
        if len(token) >= 2 and token not in tokens:
            tokens.append(token)
    return tokens


def rerank_crawler_results(query: str, items: list, tool_name: str = None) -> list:
    """크롤러 결과를 운영형 신호로 재정렬한다(정답 라벨 미사용).

    [입력]
        query    : 사용자 질의문.
        items    : 크롤러 반환 list[dict].
                   각 항목: title / preview(또는 summary) / source 등.
        tool_name: 실행 tool 이름(선택). 소스 선호도 가점 판단에 사용한다.
    [반환]
        재정렬된 list[dict]. 각 항목에 진단 필드를 덧붙인다:
          original_rank  : 크롤러 원래 순서(0-based)
          rerank_score   : 점수 합계
          rerank_reasons : 가점 이유 목록

    [정렬 규칙]
        1) 크롤러 반환 순서를 기본으로 유지(안정 정렬).
        2) title 에 query 키워드가 있으면 우선(+3).
        3) preview/summary 에 query 키워드가 있으면 우선(+2).
        4) 질문 의도에 맞는 소스면 우선(+2).
        5) 동일 점수면 원래 크롤러 순서를 유지(안정 정렬).

    [안정성]
        items 가 비어 있거나 형식이 어긋나면 원본을 그대로(또는 빈 리스트) 돌려준다.
    """
    if not isinstance(items, list) or not items:
        return items if isinstance(items, list) else []

    preferred_sources = infer_preferred_sources_for_query(query)
    tokens = _query_tokens(query)

    reranked: list[dict] = []
    for original_rank, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        title   = str(item.get("title",   "") or "").lower()
        preview = str(item.get("preview", item.get("summary", "")) or "").lower()

        score = 0
        reasons: list[str] = []

        # (2) title 키워드 매칭 가점
        if tokens and any(token in title for token in tokens):
            score += 3
            reasons.append("title_keyword_match")

        # (3) preview/summary 키워드 매칭 가점
        if tokens and any(token in preview for token in tokens):
            score += 2
            reasons.append("preview_keyword_match")

        # (4) 소스 선호도 가점
        if tool_name and preferred_sources and tool_name in preferred_sources:
            score += 2
            reasons.append(f"preferred_source:{tool_name}")

        new_item = dict(item)
        new_item["original_rank"]  = original_rank
        new_item["rerank_score"]   = score
        new_item["rerank_reasons"] = reasons
        reranked.append(new_item)

    # 점수 내림차순 정렬. Python sort 는 안정 정렬이라 동점은 original_rank 순서가 유지된다.
    reranked.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return reranked
