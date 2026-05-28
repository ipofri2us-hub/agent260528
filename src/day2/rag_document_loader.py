# -*- coding: utf-8 -*-
"""
2일차 RAG 실습 - Markdown 문서 로더 (RAG 대상 문서 입력 구조 검증)

[2일차 RAG Agent v1 흐름에서의 역할]
이 파일은 RAG 파이프라인의 첫 단계로, 단순한 Markdown 읽기가 아니라
"RAG 검색 대상 문서가 정상적으로 준비되었는지"를 점검하는
RAG 대상 문서 입력 구조 검증 단계입니다.

[입력 - 교육용 제조 기술 문서]
docs/alarm_manual.md, docs/troubleshooting_guide.md, docs/quality_standard.md
세 문서는 모두 가상의 DisplayEdu Fab 시나리오에 기반한 교육용 제조 기술 문서이며,
이후 RAG Agent가 답변 근거로 참조할 지식 원본입니다.

[산출물]
outputs/day2/document_load_result.md
검색 대상 문서가 제대로 로드되었는지 확인하는 입력 검증 자료입니다.

[다음 단계 연결]
여기서 정리된 문단은 다음 단계 chunk_builder.py에서
검색 가능한 지식 단위(chunk)로 분리됩니다.

실행 명령어:
python src/day2/rag_document_loader_YYYYMMDD_HHMMSS.py
"""

import logging
from pathlib import Path


# 실행 단계를 로그로 남겨, 어떤 문서가 몇 개 로드되었는지 콘솔에서 바로 확인하도록 합니다.
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def get_project_root():
    """
    현재 파일 위치를 기준으로 프로젝트 루트 경로를 찾습니다.

    docs/outputs 같은 입력·산출물 경로를 루트 기준으로 일관되게 잡기 위한 기준점입니다.
    프로젝트 폴더가 다른 드라이브로 이동해도 동작하도록 특정 경로를 하드코딩하지 않습니다.
    """
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent] + list(current_file.parents):
        src_dir = parent / "src"
        docs_dir = parent / "docs"
        outputs_dir = parent / "outputs"

        if src_dir.exists() and (docs_dir.exists() or outputs_dir.exists()):
            return parent

    # 프로젝트 루트를 찾지 못하면 현재 파일 기준 두 단계 위를 기본값으로 사용합니다.
    return current_file.parents[2]


def load_documents(project_root):
    """
    docs 폴더의 교육용 제조 기술 문서를 읽어 RAG 검색 대상 입력 구조로 정리합니다.

    파일명·글자 수·문단 수를 함께 모아두는 이유는, 검색 대상 문서가
    비어 있거나 누락되지 않았는지(=입력 구조 검증) 다음 단계 전에 확인하기 위함입니다.
    문단은 빈 줄 기준으로 나누며, 여기서는 chunk 분리나 metadata 추출은 하지 않습니다.
    그 작업은 다음 단계 chunk_builder.py의 책임입니다.
    """
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        logger.warning("docs 폴더를 찾을 수 없습니다.")
        logger.warning("프로젝트 루트 아래에 docs 폴더를 만들고 Markdown 문서를 넣어 주세요.")
        return []

    markdown_files = sorted(docs_dir.glob("*.md"))

    if not markdown_files:
        logger.warning("docs 폴더에 Markdown(.md) 문서가 없습니다.")
        logger.warning("docs 폴더에 실습용 Markdown 문서를 추가한 뒤 다시 실행해 주세요.")
        return []

    results = []

    for file_path in markdown_files:
        logger.info("Markdown 문서를 읽는 중입니다: %s", file_path.name)

        text = file_path.read_text(encoding="utf-8-sig")

        paragraphs = []
        for block in text.split("\n\n"):
            paragraph = block.strip()
            if paragraph:
                paragraphs.append(paragraph)

        results.append(
            {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "char_count": len(text),
                "paragraph_count": len(paragraphs),
                "paragraphs": paragraphs,
            }
        )

    return results


def save_result_markdown(results, output_path):
    """
    문서 로드 결과를 outputs/day2/document_load_result.md 파일로 저장합니다.

    이 산출물은 최종 답변이 아니라, RAG 검색 대상 문서가 정상적으로 준비되었는지
    강사·수강생이 눈으로 확인하는 입력 검증 자료입니다.
    문서 수가 0이거나 문단 수가 비정상이면 이후 chunk·검색 품질이 모두 흔들리므로
    이 단계에서 먼저 점검합니다.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_paragraphs = sum(item["paragraph_count"] for item in results)
    total_chars = sum(item["char_count"] for item in results)

    lines = []
    lines.append("# Day2 Markdown 문서 로드 결과")
    lines.append("")
    lines.append("## 1. 로드 요약")
    lines.append("")
    lines.append(f"- 읽은 문서 수: {len(results)}")
    lines.append(f"- 전체 문단 수: {total_paragraphs}")
    lines.append(f"- 전체 글자 수: {total_chars}")
    lines.append("")
    lines.append("## 2. 문서별 요약")
    lines.append("")

    if not results:
        lines.append("- 분석할 Markdown 문서가 없습니다.")
        lines.append("- docs 폴더에 Markdown 문서를 추가한 뒤 다시 실행해 주세요.")
        lines.append("")
    else:
        for item in results:
            lines.append(f"### {item['file_name']}")
            lines.append("")
            lines.append(f"- 파일명: {item['file_name']}")
            lines.append(f"- 글자 수: {item['char_count']}")
            lines.append(f"- 문단 수: {item['paragraph_count']}")
            lines.append("")

    lines.append("## 3. 다음 단계 안내")
    lines.append("")
    lines.append("다음 단계에서는 `chunk_builder.py`를 실행해 문단을 chunk로 나누고 metadata를 생성합니다.")
    lines.append("")
    lines.append("문서 로더가 Markdown 파일을 읽어 문단 단위로 정리하면, chunk_builder.py는 이 문단을 검색하기 좋은 작은 단위로 나눕니다.")
    lines.append("")
    lines.append("이후 RAG 검색 단계에서는 사용자의 질문과 관련이 높은 chunk를 찾아 AI Agent 답변의 근거로 사용합니다.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    """
    전체 실행 흐름을 담당합니다.

    1. 프로젝트 루트를 찾습니다.
    2. docs 폴더의 Markdown 문서를 읽습니다.
    3. 결과를 outputs/day2/document_load_result.md 파일로 저장합니다.
    4. 다음 단계인 chunk_builder.py 실행을 안내합니다.
    """
    project_root = get_project_root()
    output_path = project_root / "outputs" / "day2" / "document_load_result.md"

    logger.info("Day2 RAG Markdown 문서 로드를 시작합니다.")
    logger.info("프로젝트 루트: %s", project_root)

    results = load_documents(project_root)
    save_result_markdown(results, output_path)

    logger.info("문서 로드 결과를 저장했습니다: %s", output_path)
    logger.info("다음 단계에서는 chunk_builder.py를 실행해 문단을 chunk로 나누고 metadata를 생성합니다.")


if __name__ == "__main__":
    main()
