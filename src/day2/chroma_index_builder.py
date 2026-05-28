# -*- coding: utf-8 -*-
"""
2일차 RAG 실습 - Chroma Vector DB 저장기 (선택/확장 실습)

[위치와 목적 - 먼저 읽어둘 것]
이 파일은 선택/확장 실습입니다. 2일차의 기본 목표는 Vector DB 구축 자체가 아니라,
"검색 결과가 어떻게 State(retrieved_docs)와 Trace로 연결되는지"를 이해하는 것입니다.
Vector DB는 그 검색을 실제로 가능하게 해 주는 인프라일 뿐입니다.

[하는 일]
outputs/day2/chunk_preview.json(앞 단계 chunk 산출물)을 읽어,
Ollama nomic-embed-text 모델로 각 chunk를 embedding(숫자 벡터)으로 바꾸고
Chroma Vector DB(vector_db/chroma_db)에 저장합니다. Chroma는 이 벡터로
의미가 가까운 문서를 찾아주는 Vector DB 역할을 합니다.

[Saved Result Review - 운영 주석]
Chroma/Ollama 환경 오류로 인덱싱이 안 되면, 강사 데모 또는 미리 만들어 둔
outputs/day2의 검색 결과(rag_test_results.md 등)로 진행하는 Saved Result Review로 대체합니다.
이 단계에서 오래 막히지 않는 것이 수업 운영상 중요합니다.

결과 Markdown 문서는 Mustache 템플릿으로 생성합니다.
템플릿 위치:
templates/day2/chroma_build_result.mustache

필요 패키지:
pip install chromadb
pip install pystache
pip install tqdm

실행 명령어:
python src/day2/chroma_index_builder_20260524_031526.py

Ollama 사전 준비 명령어:
ollama list
ollama pull nomic-embed-text
ollama serve
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pystache
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


COLLECTION_NAME = "manufacturing_rag_docs"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"
OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embeddings"


def get_project_root() -> Path:
    """
    현재 파일 위치를 기준으로 프로젝트 루트 폴더를 찾습니다.

    src 폴더가 있고, docs 또는 outputs 폴더가 있는 위치를 프로젝트 루트로 봅니다.
    프로젝트 폴더 위치가 바뀌어도 동작하도록 특정 경로를 하드코딩하지 않습니다.
    """
    current_file = Path(__file__).resolve()

    for parent in [current_file.parent, *current_file.parents]:
        has_src = (parent / "src").exists()
        has_docs_or_outputs = (parent / "docs").exists() or (parent / "outputs").exists()
        if has_src and has_docs_or_outputs:
            return parent

    return current_file.parents[2]


def load_chunk_preview(input_json_path: Path) -> List[Dict[str, str]]:
    """chunk_builder가 생성한 chunk_preview.json 파일을 읽습니다."""
    if not input_json_path.exists():
        logger.error("chunk_preview.json 파일을 찾지 못했습니다: %s", input_json_path)
        logger.error("먼저 chunk_builder_날짜_시간.py를 실행해 outputs/day2/chunk_preview.json 파일을 생성해 주세요.")
        raise FileNotFoundError(input_json_path)

    return json.loads(input_json_path.read_text(encoding="utf-8-sig"))


class SimpleOllamaEmbeddingFunction:
    """
    Chroma가 chunk text를 숫자 벡터로 바꿀 때 사용하는 도구입니다.

    이 클래스는 Ollama embedding API를 호출합니다.
    text embedding은 문장을 숫자 리스트로 바꾸는 과정입니다.
    Chroma는 이 숫자 리스트를 사용해 의미가 가까운 문서를 찾습니다.
    """

    def __init__(self, model_name: str, url: str, timeout_seconds: int = 60) -> None:
        self.model_name = model_name
        self.url = url
        self.timeout_seconds = timeout_seconds

    def name(self) -> str:
        """Chroma가 embedding function을 구분할 때 사용하는 이름입니다."""
        return "simple_ollama_nomic_embed_text"

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Chroma가 documents를 embedding으로 바꿀 때 호출합니다."""
        return self.embed_documents(input)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 chunk text를 embedding 리스트로 변환합니다."""
        embeddings: List[List[float]] = []

        for text in tqdm(texts, desc="Ollama embedding 생성", unit="chunk"):
            embeddings.append(self._request_embedding(str(text or "")))

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """검색 질문을 embedding으로 변환할 때 재사용할 수 있는 메서드입니다."""
        return self._request_embedding(str(text or ""))

    def _request_embedding(self, text: str) -> List[float]:
        """Ollama embedding API를 호출해 embedding 숫자 리스트를 받습니다."""
        payload = {
            "model": self.model_name,
            "prompt": text,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
        except urllib.error.URLError as error:
            logger.error("Ollama 서버에 연결하지 못했습니다.")
            logger.error("확인 명령어: ollama serve")
            logger.error("모델 설치 명령어: ollama pull nomic-embed-text")
            raise RuntimeError(f"Ollama embedding API 연결 오류: {error}") from error

        response_json = json.loads(response_text)
        embedding = response_json.get("embedding")

        if not isinstance(embedding, list) or not embedding:
            raise RuntimeError(
                "Ollama 응답에서 embedding 필드를 찾지 못했습니다. "
                "모델명과 Ollama API 상태를 확인해 주세요."
            )

        return [float(value) for value in embedding]


def save_chunks_to_chroma(records: List[Dict[str, str]], chroma_dir: Path, collection_name: str) -> int:
    """chunk records를 Chroma collection에 저장합니다.

    text는 검색 본문(document), chunk_id는 ID, 나머지는 metadata로 저장합니다.
    여기서 함께 저장하는 metadata(alarm_code/equipment_id 등)가 rag_search 결과에
    그대로 실려 나와 State의 retrieved_docs가 되고, 근거 위치 추적의 기준이 됩니다.
    재실행 시 같은 collection을 지우고 새로 만들어, 항상 최신 chunk 기준으로 인덱싱합니다.
    """

    def sanitize_metadata_value(value: Any) -> str | int | float | bool:
        """Chroma metadata에 안정적으로 들어갈 수 있는 값으로 변환합니다."""
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if item is not None)
        return str(value)

    try:
        import chromadb
    except ImportError:
        logger.error("chromadb 패키지가 설치되어 있지 않습니다.")
        logger.error("설치 명령: pip install chromadb")
        raise

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        client.delete_collection(name=collection_name)
        logger.info("기존 Chroma collection을 삭제했습니다: %s", collection_name)
    except Exception:
        logger.debug("기존 Chroma collection이 없어서 새로 생성합니다: %s", collection_name)

    embedding_function = SimpleOllamaEmbeddingFunction(
        model_name=OLLAMA_EMBEDDING_MODEL,
        url=OLLAMA_EMBEDDING_URL,
    )

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
    )

    if not records:
        return collection.count()

    documents: List[str] = []
    ids: List[str] = []
    metadatas: List[Dict[str, str | int | float | bool]] = []

    for record in tqdm(records, desc="Chroma 저장 데이터 준비", unit="chunk"):
        documents.append(record["text"])
        ids.append(record["chunk_id"])
        metadatas.append(
            {
                "chunk_id": sanitize_metadata_value(record.get("chunk_id", "")),
                "doc_name": sanitize_metadata_value(record.get("doc_name", "")),
                "section_title": sanitize_metadata_value(record.get("section_title", "")),
                "alarm_code": sanitize_metadata_value(record.get("alarm_code", "")),
                "equipment_id": sanitize_metadata_value(record.get("equipment_id", "")),
                "equipment_type": sanitize_metadata_value(record.get("equipment_type", "")),
                "process_name": sanitize_metadata_value(record.get("process_name", "")),
                "symptom": sanitize_metadata_value(record.get("symptom", "")),
                "quality_metric": sanitize_metadata_value(record.get("quality_metric", "")),
                "action": sanitize_metadata_value(record.get("action", "")),
                "keywords": sanitize_metadata_value(record.get("keywords", "")),
            }
        )

    logger.info("Chroma 저장을 시작합니다. chunk 수: %s", len(records))
    collection.add(documents=documents, metadatas=metadatas, ids=ids)

    return collection.count()


def save_results(
    records: List[Dict[str, str]],
    chroma_dir: Path,
    collection_name: str,
    chroma_count: int,
    output_md_path: Path,
    template_path: Path,
) -> None:
    """Mustache 템플릿을 사용해 chroma_build_result.md 파일을 저장합니다."""
    if not template_path.exists():
        logger.warning("Mustache 템플릿 파일을 찾지 못했습니다: %s", template_path)
        logger.warning("templates/day2/chroma_build_result.mustache 파일을 먼저 생성해 주세요.")
        return

    doc_counter = Counter(record["doc_name"] for record in records)

    template_data = {
        "chunk_count": len(records),
        "chroma_dir": str(chroma_dir),
        "collection_name": collection_name,
        "chroma_count": chroma_count,
        "embedding_provider": "Ollama",
        "embedding_model": OLLAMA_EMBEDDING_MODEL,
        "embedding_api_url": OLLAMA_EMBEDDING_URL,
        "doc_counts": [
            {"doc_name": doc_name, "chunk_count": count}
            for doc_name, count in sorted(doc_counter.items())
        ],
    }

    template_text = template_path.read_text(encoding="utf-8-sig")
    rendered_text = pystache.render(template_text, template_data)

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text(rendered_text, encoding="utf-8-sig")


def main() -> None:
    """전체 실행 흐름을 담당합니다."""
    project_root = get_project_root()

    input_json_path = project_root / "outputs" / "day2" / "chunk_preview.json"
    output_md_path = project_root / "outputs" / "day2" / "chroma_build_result.md"
    chroma_dir = project_root / "vector_db" / "chroma_db"
    template_path = project_root / "templates" / "day2" / "chroma_build_result.mustache"

    logger.info("Day2 Chroma Vector DB 생성을 시작합니다.")
    logger.info("프로젝트 루트: %s", project_root)
    logger.info("입력 JSON 경로: %s", input_json_path)
    logger.debug("Mustache 템플릿 경로: %s", template_path)

    records = load_chunk_preview(input_json_path)
    chroma_count = save_chunks_to_chroma(records, chroma_dir, COLLECTION_NAME)
    save_results(
        records=records,
        chroma_dir=chroma_dir,
        collection_name=COLLECTION_NAME,
        chroma_count=chroma_count,
        output_md_path=output_md_path,
        template_path=template_path,
    )

    logger.info("Chroma Vector DB 저장을 완료했습니다.")
    logger.info("Chroma DB 저장 경로: %s", chroma_dir)
    logger.info("Chroma collection 이름: %s", COLLECTION_NAME)
    logger.info("Chroma collection 저장 개수: %s", chroma_count)
    logger.info("요약 파일 경로: %s", output_md_path)

    if not records:
        logger.warning("저장할 chunk가 없습니다. chunk_preview.json 내용을 확인해 주세요.")


if __name__ == "__main__":
    main()
