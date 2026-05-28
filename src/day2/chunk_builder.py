# -*- coding: utf-8 -*-
"""
2일차 RAG 실습 - Markdown 문서 Chunk 생성기 (검색 가능한 지식 단위 설계)

[2일차 RAG Agent v1 흐름에서의 역할]
chunk 생성은 문서를 길이로 자르는 작업이 아니라, RAG 검색이 답변 근거를
"문단 단위로 찾고 추적"할 수 있게 만드는 검색 가능한 지식 단위 설계 단계입니다.

[입력 / 산출물]
- 입력: docs 폴더의 교육용 제조 기술 문서(rag_document_loader.py가 검증한 문서)
- 산출물1: outputs/day2/chunk_preview.json
  → 이후 chroma_index_builder / rag_search 가 그대로 재사용하는 구조화 chunk 데이터
- 산출물2: outputs/day2/chunk_build_result.md
  → chunk 생성 결과를 사람이 검토하기 위한 요약 자료

[핵심 설계 의도]
- chunk_id, doc_name 은 검색 결과와 Trace에서 "이 답변이 어느 문서 어느 조각에
  근거했는지"를 추적하기 위한 기준값입니다.
- metadata(alarm_code, equipment_id 등)는 검색·필터링·3일차 Tool 입력값으로
  재사용되는 구조화 정보입니다. 즉 여기서의 metadata 설계가 곧
  3일차 search_manual Tool의 입력/필터 설계와 직접 연결됩니다.

이 파일은 Chroma Vector DB 저장이나 Ollama embedding은 만들지 않습니다(다음 단계 책임).

설치 명령어:
pip install pystache

실행 명령어:
python src/day2/chunk_builder_20260524_020457.py
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pystache


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# 너무 짧은 문단은 검색 근거로 쓰기 어려워 제외합니다.
MIN_PARAGRAPH_LENGTH = 15

# 교육용 가상 제조 문서에서 metadata를 추출하기 위한 키워드입니다.
EQUIPMENT_TYPE_KEYWORDS = [
    "증착 설비",
    "검사 설비",
    "세정 설비",
    "이송 설비",
    "패널 제조 설비",
    "설비",
]

PROCESS_NAME_KEYWORDS = [
    "증착 공정",
    "검사 공정",
    "세정 공정",
    "이송 공정",
    "공정 상태",
    "제조 라인",
    "디스플레이 패널 제조 라인",
]

SYMPTOM_KEYWORDS = [
    "온도 상승",
    "반복 알람",
    "압력 불안정",
    "진공 상태",
    "냉각 상태",
    "공정 부하",
    "센서 값 변동",
    "검사 결과 변화",
]

QUALITY_METRIC_KEYWORDS = [
    "품질 지표",
    "불량률",
    "수율",
    "검사 결과",
    "품질 영향",
    "품질 이상",
    "검사 공정",
]

ACTION_KEYWORDS = [
    "확인 필요",
    "점검 필요",
    "추가 검토",
    "조치 방향",
    "원인 후보",
    "정비 이력",
    "설정 변경",
    "근거 확인",
    "품질 영향 여부 확인",
]


def get_project_root() -> Path:
    """
    현재 파일 위치를 기준으로 프로젝트 루트 폴더를 찾습니다.

    docs(입력)와 outputs/day2(산출물) 경로를 루트 기준으로 일관되게 잡기 위한 기준점입니다.
    상위 폴더로 올라가며 src 폴더와 docs 또는 outputs 폴더가 함께 있는 위치를 루트로 봅니다.
    """
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent, *current_file.parents]:
        has_src = (parent / "src").exists()
        has_docs_or_outputs = (parent / "docs").exists() or (parent / "outputs").exists()
        if has_src and has_docs_or_outputs:
            return parent

    return current_file.parents[2]


def extract_metadata(text: str) -> Dict[str, str]:
    """
    chunk 본문에서 알람 코드, 설비 ID, 증상, 조치 방향 같은 metadata를 추출합니다.

    여기서 뽑는 metadata는 단순 태그가 아니라, 이후 검색 필터링과
    3일차 search_manual Tool의 입력값으로 재사용되는 구조화 정보입니다.
    alarm_code/equipment_id는 정규식으로, 나머지는 교육용 키워드 사전 매칭으로 추출합니다.
    """

    def unique_join(items: List[str]) -> str:
        """중복 값을 제거하고 쉼표로 연결합니다."""
        unique_items: List[str] = []
        for item in items:
            if item and item not in unique_items:
                unique_items.append(item)
        return ", ".join(unique_items)

    alarm_code = unique_join(re.findall(r"\bALM-[A-Z]+-[0-9]+\b", text))
    equipment_id = unique_join(re.findall(r"\bEQP-[A-Z]+-[0-9]+\b", text))
    equipment_type = unique_join([keyword for keyword in EQUIPMENT_TYPE_KEYWORDS if keyword in text])
    process_name = unique_join([keyword for keyword in PROCESS_NAME_KEYWORDS if keyword in text])
    symptom = unique_join([keyword for keyword in SYMPTOM_KEYWORDS if keyword in text])
    quality_metric = unique_join([keyword for keyword in QUALITY_METRIC_KEYWORDS if keyword in text])
    action = unique_join([keyword for keyword in ACTION_KEYWORDS if keyword in text])

    keyword_parts: List[str] = []
    for value in [alarm_code, equipment_id, equipment_type, process_name, symptom, quality_metric, action]:
        for part in [item.strip() for item in value.split(",")]:
            if part and part not in keyword_parts:
                keyword_parts.append(part)

    return {
        "alarm_code": alarm_code,
        "equipment_id": equipment_id,
        "equipment_type": equipment_type,
        "process_name": process_name,
        "symptom": symptom,
        "quality_metric": quality_metric,
        "action": action,
        "keywords": ", ".join(keyword_parts),
    }


def build_chunk_records(docs_dir: Path) -> List[Dict[str, str]]:
    """
    docs 폴더의 교육용 제조 문서를 읽어 검색 가능한 chunk record 목록을 생성합니다.

    각 record는 chunk_id(근거 위치 추적용 고유 ID), doc_name, section_title,
    metadata, text 로 구성됩니다. 이 구조가 곧 검색 결과 1건의 모양이 되고,
    Trace에서 "어느 문서 어느 조각을 근거로 썼는지"를 되짚는 기준이 됩니다.
    문단은 빈 줄 기준으로 나누며, 흐름을 따라가기 쉽게 단순 for 반복으로 작성합니다.
    """
    records: List[Dict[str, str]] = []
    chunk_number = 1

    if not docs_dir.exists():
        logger.warning("docs 폴더를 찾지 못했습니다: %s", docs_dir)
        return records

    markdown_files = sorted(docs_dir.glob("*.md"))
    if not markdown_files:
        logger.warning("docs 폴더에 Markdown(.md) 파일이 없습니다: %s", docs_dir)
        return records

    for md_file in markdown_files:
        logger.info("Markdown 문서 읽기: %s", md_file.name)

        # Windows에서 한글 Markdown 파일은 UTF-8-SIG로 저장되는 경우가 많습니다.
        text = md_file.read_text(encoding="utf-8-sig")
        section_title = ""

        for block in text.split("\n\n"):
            paragraph = block.strip()
            if not paragraph:
                continue

            # Markdown 제목(#, ##, ###)을 만나면 현재 section_title만 갱신합니다.
            if re.match(r"^#{1,3}\s+", paragraph):
                section_title = re.sub(r"^#{1,3}\s+", "", paragraph).strip()
                continue

            compact_text = paragraph.replace(" ", "").replace("\n", "")
            if len(compact_text) < MIN_PARAGRAPH_LENGTH:
                continue

            # chunk record 1건 = 검색 결과 1건의 원형.
            # chunk_id/doc_name/section_title 은 근거 위치 추적의 기준이고,
            # metadata는 검색 필터·Tool 입력값, text는 답변 grounding의 실제 근거 본문이 됩니다.
            metadata = extract_metadata(paragraph)
            record = {
                "chunk_id": f"CHUNK-{chunk_number:04d}",
                "doc_name": md_file.name,
                "section_title": section_title,
                **metadata,
                "text": paragraph,
            }

            records.append(record)
            chunk_number += 1

    return records


def save_results(
    records: List[Dict[str, str]],
    output_json_path: Path,
    output_md_path: Path,
    template_path: Path,
    next_file_name: str,
) -> None:
    """
    chunk_preview.json을 저장하고, Mustache 템플릿으로 chunk_build_result.md를 생성합니다.

    - chunk_preview.json: 이후 검색/인덱싱 단계가 그대로 다시 읽어 쓰는 구조화 chunk 데이터
    - chunk_build_result.md: 문서별 chunk 개수와 샘플을 담아, chunk 생성 품질과
      Trace 추적 가능성(근거 위치를 되짚을 수 있는지)을 사람이 검토하는 요약 자료
    """

    def preview_text(text: str, max_length: int = 120) -> str:
        one_line_text = " ".join(text.split())
        if len(one_line_text) <= max_length:
            return one_line_text
        return one_line_text[:max_length] + "..."

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8-sig",
    )

    doc_counter = Counter(record["doc_name"] for record in records)
    template_data = {
        "doc_count": len(doc_counter),
        "chunk_count": len(records),
        "chunk_preview_path": str(output_json_path),
        "next_file_name": next_file_name,
        "doc_counts": [
            {"doc_name": doc_name, "chunk_count": count}
            for doc_name, count in sorted(doc_counter.items())
        ],
        "sample_chunks": [
            {
                "chunk_id": record["chunk_id"],
                "doc_name": record["doc_name"],
                "section_title": record["section_title"],
                "alarm_code": record["alarm_code"],
                "equipment_id": record["equipment_id"],
                "keywords": record["keywords"],
                "text_preview": preview_text(record["text"]),
            }
            for record in records[:5]
        ],
    }

    if not template_path.exists():
        logger.warning("Mustache 템플릿 파일을 찾지 못했습니다: %s", template_path)
        logger.warning("templates/day2/chunk_build_result.mustache 파일을 먼저 생성해 주세요.")
        return

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    template_text = template_path.read_text(encoding="utf-8-sig")
    rendered_text = pystache.render(template_text, template_data)
    output_md_path.write_text(rendered_text, encoding="utf-8-sig")


def main() -> None:
    """전체 실행 흐름을 담당합니다."""
    project_root = get_project_root()
    docs_dir = project_root / "docs"
    output_dir = project_root / "outputs" / "day2"
    output_json_path = output_dir / "chunk_preview.json"
    output_md_path = output_dir / "chunk_build_result.md"
    template_path = project_root / "templates" / "day2" / "chunk_build_result.mustache"

    current_stem = Path(__file__).stem
    parts = current_stem.split("_")
    if len(parts) >= 4 and parts[-2].isdigit() and parts[-1].isdigit():
        timestamp = f"{parts[-2]}_{parts[-1]}"
        next_file_name = f"chroma_index_builder_{timestamp}.py"
    else:
        next_file_name = "chroma_index_builder_날짜_시간.py"

    logger.info("Day2 chunk 생성을 시작합니다.")
    logger.info("프로젝트 루트: %s", project_root)
    logger.info("Mustache 템플릿 경로: %s", template_path)
    logger.info("다음 단계 파일: %s", next_file_name)

    records = build_chunk_records(docs_dir)
    save_results(
        records=records,
        output_json_path=output_json_path,
        output_md_path=output_md_path,
        template_path=template_path,
        next_file_name=next_file_name,
    )

    logger.info("chunk preview JSON을 저장했습니다: %s", output_json_path)
    logger.info("chunk 생성 요약 파일을 저장했습니다: %s", output_md_path)
    logger.info("생성된 chunk 수: %s", len(records))

    if not records:
        logger.warning("생성된 chunk가 없습니다. docs 폴더와 Markdown 파일을 확인해 주세요.")


if __name__ == "__main__":
    main()
