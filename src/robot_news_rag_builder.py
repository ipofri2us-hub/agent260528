# -*- coding: utf-8 -*-
"""
공정 1~4단계: 다중 매체(로봇신문/전자신문 등) 동적 루프 수집 및 통합 Chroma DB 적재 엔진
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Any

import pystache
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

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

# 메타데이터 추출용 키워드 사전
ROBOT_TYPE_KEYWORDS = ["자율주행", "협동로봇", "물류로봇", "휴머노이드", "서비스로봇", "산업용로봇", "드론"]
TECH_KEYWORDS = ["AI", "인공지능", "비전", "센서", "SLAM", "그리퍼", "액추에이터", "5G"]
COMPANY_KEYWORDS = ["삼성", "LG", "현대", "두산", "레인보우", "보스턴", "테슬라"]


def extract_news_metadata(text: str) -> Dict[str, str]:
    """본문 문맥에서 키워드를 필터링하여 구조화 메타데이터를 뽑아냅니다."""
    def unique_join(items: List[str]) -> str:
        return ", ".join(sorted(list(set([i for i in items if i]))))

    robot_type = unique_join([k for k in ROBOT_TYPE_KEYWORDS if k in text])
    tech_stack = unique_join([k for k in TECH_KEYWORDS if k in text])
    companies = unique_join([k for k in COMPANY_KEYWORDS if k in text])
    
    keyword_parts = []
    for val in [robot_type, tech_stack, companies]:
        for part in [item.strip() for item in val.split(",")]:
            if part and part not in keyword_parts:
                keyword_parts.append(part)

    return {
        "robot_type": robot_type,
        "tech_stack": tech_stack,
        "companies": companies,
        "keywords": ", ".join(keyword_parts),
    }


def parse_article_content(article_url: str, selectors: List[str]) -> str:
    """매체별 본문 셀렉터 풀을 활용하여 유연하게 본문 텍스트 추출"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(article_url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return ""
        
        # 종합지(전자신문 등)의 한글 깨짐 방지 인코딩 보정
        resp.encoding = resp.apparent_encoding if resp.encoding == 'ISO-8859-1' else 'utf-8'
        soup = BeautifulSoup(resp.text, "html.parser")
        
        # Config에서 넘어온 셀렉터들을 순차 대입 (하나라도 걸리면 매칭)
        content_div = None
        for selector in selectors:
            content_div = soup.select_one(selector)
            if content_div: break
            
        if content_div:
            for s in content_div(["script", "style", "iframe", "figure", "figcaption"]):
                s.decompose()
            return content_div.get_text(strip=True)
    except Exception as e:
        logger.debug(f"본문 파싱 실패 ({article_url}): {e}")
    return ""


def crawl_all_sources_to_chunks() -> List[Dict[str, str]]:
    """Config에 정의된 다중 매체 리스트를 순회하며 지식을 통합 수집 및 분할합니다."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    all_records = []
    chunk_number = 1
    seen_links = set()
    
    chunk_size = CONFIG["chunk"]["size"]
    chunk_overlap = CONFIG["chunk"]["overlap"]
    sources_list = CONFIG["sources"]

    # 💥 [엔지니어링 고도화: 멀티 타겟 동적 루프 구동]
    for src in sources_list:
        media = src["media_name"]
        logger.info(f"▶️ [매체 가동] {media} 수집 파이프라인 시작 (목표: {src['max_pages']} 페이지)")
        
        for page in range(1, src["max_pages"] + 1):
            try:
                resp = requests.get(src["target_url"], headers=headers, params={"page": page}, timeout=10)
                if resp.status_code != 200: continue
                
                resp.encoding = 'utf-8'
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # 매체별 정규식 주소 패턴 로드 및 매칭
                all_links = soup.find_all("a", href=re.compile(src["url_pattern"]))
                
                for a_tag in all_links:
                    title = a_tag.get_text(strip=True)
                    if len(title) < 6 or title.startswith("사진") or "기자" in title: 
                        continue
                        
                    raw_href = a_tag["href"]
                    link = src["prefix_url"] + raw_href if raw_href.startswith("/") else raw_href
                    
                    if link in seen_links: continue
                    seen_links.add(link)
                    
                    # 유니크 ID 생성
                    article_id = link.split("idxno=")[-1].split("/")[-1].split("&")[0].split("?")[0]
                    
                    # 매체 전용 셀렉터 세트를 전달하여 본문 파싱
                    content = parse_article_content(link, src["content_selectors"])
                    if len(content) < 40: continue
                    
                    # 슬라이딩 윈도우 분할
                    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size - chunk_overlap)]
                    
                    for sub_idx, chunk_text in enumerate(chunks):
                        meta = extract_news_metadata(chunk_text)
                        
                        record = {
                            "chunk_id": f"INTEG-NEWS-{chunk_number:04d}",
                            "doc_name": f"{media}_{article_id}_{sub_idx}.md", # 파일명에 매체명 이식
                            "section_title": f"[{media}] {title} (파트 {sub_idx+1})", # 제목에 매체 꼬리표 명시
                            "robot_type": meta["robot_type"],
                            "tech_stack": meta["tech_stack"],
                            "companies": meta["companies"],
                            "keywords": meta["keywords"],
                            "alarm_code": "", "equipment_id": "", "equipment_type": "", "process_name": "", "symptom": "", "quality_metric": "",
                            "action": link,
                            "text": chunk_text
                        }
                        all_records.append(record)
                        chunk_number += 1
                    
                    time.sleep(0.15)
            except Exception as e:
                logger.error(f"{media} {page}페이지 구동 중 오류: {e}")
                
    return all_records


class SimpleOllamaEmbeddingFunction:
    def __init__(self, model_name: str, url: str) -> None:
        self.model_name = model_name
        self.url = url

    def name(self) -> str:
        return "simple_ollama_nomic_embed_text"

    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = []
        for text in tqdm(input, desc="통합 오프라인 임베딩 연산 중", unit="chunk"):
            body = json.dumps({"model": self.model_name, "prompt": str(text or "")}, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=30) as response:
                response_json = json.loads(response.read().decode("utf-8"))
            embeddings.append([float(v) for v in response_json.get("embedding", [])])
        return embeddings


def save_chunks_to_chroma(records: List[Dict[str, str]]) -> int:
    import chromadb
    
    chroma_dir = Path(CONFIG["vector_db"]["chroma_path"])
    collection_name = CONFIG["vector_db"]["collection_name"]
    
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    embedding_function = SimpleOllamaEmbeddingFunction(
        model_name=CONFIG["vector_db"]["embedding_model"], 
        url=CONFIG["vector_db"]["embedding_url"]
    )
    collection = client.create_collection(name=collection_name, embedding_function=embedding_function)

    if not records: return 0

    documents, ids, metadatas = [], [], []
    meta_keys = ["chunk_id", "doc_name", "section_title", "robot_type", "tech_stack", "companies", "keywords", "action"]
    
    for r in records:
        documents.append(r["text"])
        ids.append(r["chunk_id"])
        metadatas.append({k: str(r.get(k, "")) for k in meta_keys})

    logger.info(f"통합 가상 창고 적재 가동... (총 {len(records)} 개 멀티 매체 청크)")
    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection.count()


def main():
    logger.info("=== [STAGE 1] 다중 매체 동적 루프 수집 및 지식 파편화 ===")
    records = crawl_all_sources_to_chunks()
    
    if not records:
        logger.error("수집된 통합 뉴스가 존재하지 않습니다.")
        return

    logger.info("=== [STAGE 2] 통합 Vector 지식창고 인덱싱 적재 ===")
    chroma_count = save_chunks_to_chroma(records)
    logger.info(f"=== [완공] 총 {chroma_count}개 청크 연동 다중 매체 RAG 데이터베이스 구축 완료 ===")


if __name__ == "__main__":
    main()