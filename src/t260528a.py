# -*- coding: utf-8 -*-
"""
공정 5~8단계: 외부 Config 기반 동적 Reranking 및 팩트 기반 RAG 질의응답 솔버 엔진
"""

import json
import logging
import urllib.request
from pathlib import Path
from typing import List, Dict, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 설정 파일 로드
CONFIG_PATH = Path(r"C:\work\agent260528\config.yaml")
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {CONFIG_PATH}")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    import yaml
    CONFIG = yaml.safe_load(f)

class SimpleOllamaClient:
    def __init__(self, embedding_url: str, generate_url: str):
        self.embedding_url = embedding_url
        self.generate_url = generate_url

    def get_embedding(self, text: str) -> List[float]:
        body = json.dumps({"model": CONFIG["vector_db"]["embedding_model"], "prompt": text}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.embedding_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return [float(v) for v in data.get("embedding", [])]

    def generate_answer(self, prompt: str) -> str:
        body = json.dumps({
            "model": CONFIG["llm"]["model_name"],
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": CONFIG["llm"]["temperature"]}
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.generate_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()


def run_rag_pipeline(query_text: str):
    import chromadb
    
    chroma_dir = Path(CONFIG["vector_db"]["chroma_path"])
    if not chroma_dir.exists():
        logger.error("Chroma DB 폴더가 없습니다. 먼저 robot_news_rag_builder.py를 실행하세요.")
        return

    client = chromadb.PersistentClient(path=str(chroma_dir))
    ollama = SimpleOllamaClient(
        embedding_url=CONFIG["vector_db"]["embedding_url"],
        generate_url=CONFIG["llm"]["api_url"]
    )
    collection = client.get_collection(name=CONFIG["vector_db"]["collection_name"])

    print(f"\n🙋 사용자 질문: {query_text}")
    print("🔍 [STAGE 1] 1차 벡터 주소 검색 실시 (후보군 10건 선별)...")
    
    query_vector = ollama.get_embedding(query_text)
    results = collection.query(query_embeddings=[query_vector], n_results=10)
    raw_docs = results["documents"][0]
    raw_metas = results["metadatas"][0]

    print("🔄 [STAGE 2] 동적 Config 기반 2차 재정렬(Reranking) 공정 가동...")
    
    # 💥 [자유도 극대화 파트] 하드코딩 완전 제거
    target_keywords = CONFIG["rerank"]["target_keywords"]
    weight_score = CONFIG["rerank"]["weight_score"]
    top_k = CONFIG["rerank"]["top_k"]

    reranked_list = []
    for doc, meta in zip(raw_docs, raw_metas):
        score = 0
        # Config에서 읽어온 복수 키워드 매칭 검증
        if any(kw in doc or kw in meta.get("section_title", "") for kw in target_keywords):
            score += weight_score
            
        reranked_list.append((score, doc, meta))
    
    # 가중치 점수 기반 내림차순 정렬 후 최종 Top K 상위 추출
    reranked_list.sort(key=lambda x: x[0], reverse=True)
    final_targets = reranked_list[:top_k]

    print(f"📰 [STAGE 3] 최적의 정밀 지식 파편 {len(final_targets)}건 프롬프트 바인딩 완료.")
    
    context_str = ""
    sources = []
    for i, (score, doc, meta) in enumerate(final_targets, 1):
        context_str += f"[참조 지식 파편 {i} (가중치 점수: {score}): {meta.get('section_title')}]\n{doc}\n\n"
        sources.append(f"- {meta.get('section_title')} (출처 링크: {meta.get('action')})")

    system_prompt = (
        "당신은 로봇 산업 분석가입니다. 아래 제공된 [로봇신문 뉴스 기사 문맥]을 철저히 바탕으로 답변해 주세요.\n"
        "기사에 없는 가짜 정보(Hallucination)는 절대로 지어내지 말고 오직 사실만을 기술해야 합니다.\n"
        "마치 보고서 형식처럼 마크다운 문법을 사용하여 깔끔하게 정리해 대답하세요.\n\n"
        f"[로봇신문 뉴스 기사 문맥]\n{context_str}\n"
        f"[사용자 질문]\n{query_text}\n\n"
        "답변:"
    )

    print(f"🤖 [STAGE 4] 로컬 LLM ({CONFIG['llm']['model_name']}) 지식 컨텍스트 연산 및 답변 생성 중...")
    answer = ollama.generate_answer(system_prompt)
    
    print("\n" + "="*60)
    print("✨ Config-driven Advanced RAG 시스템 최종 시연 결과:")
    print("="*60)
    print(answer)
    print("\n📌 답변의 타당성 및 영구 추적 조사를 위한 근거 소스(Trace):")
    for source in sources: 
        print(source)
    print("="*60 + "\n")

if __name__ == "__main__":
    test_query = "삼성 관련 로봇 최신 뉴스 동향을 요약해 주고 어떤 기술 스택을 쓰는지 알려줘"
    run_rag_pipeline(test_query)