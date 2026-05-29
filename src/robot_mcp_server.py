# C:\work\agent260528\src\robot_mcp_server.py
from __future__ import annotations
import sys
import os
import yaml
import requests
from bs4 import BeautifulSoup
import urllib.parse
from pathlib import Path
from fastmcp import FastMCP

SERVER_NAME = "enterprise_robotics_pure_mcp_server"
mcp = FastMCP(SERVER_NAME)

def load_registry_config() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                return cfg.get("search_registry", {})
        except Exception:
            pass
    return {}

def _execute_real_arxiv_api(search_query: str, limit: int) -> list:
    encoded_query = urllib.parse.quote(search_query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&max_results={limit}&sortBy=submittedDate&sortOrder=descending"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code != 200: return []
        soup = BeautifulSoup(response.content, "xml")
        entries = soup.find_all("entry")
        
        results = []
        for entry in entries:
            raw_link = entry.id.text.strip()
            if not raw_link.startswith("http"):
                raw_link = f"https://arxiv.org/abs/{raw_link}"
            results.append({
                "type": "Academic Paper", 
                "source": "ArXiv Repository",
                "title": entry.title.text.strip().replace("\n", " "),
                "link": raw_link, 
                "date": entry.published.text.strip()[:10]
            })
        return results
    except Exception:
        return []

def _execute_real_google_news_api(search_query: str, limit: int) -> list:
    encoded_query = urllib.parse.quote(search_query.encode('utf-8'))
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code != 200: return []
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
        
        results = []
        for item in items:
            title_text = item.title.text.strip().replace("\n", " ")
            source_name = "구글 뉴스망"
            if " - " in title_text:
                parts = title_text.split(" - ")
                title_text = parts[0]
                source_name = parts[-1]

            results.append({
                "type": "로봇전문지 (Press)", 
                "source": source_name,
                "title": title_text, 
                "link": item.link.text.strip(),
                "date": "최신 시황"
            })
            if len(results) >= limit: break
        return results
    except Exception:
        return []

# =====================================================================
# [순수 라이브 API 라우터] 💥 어떠한 가짜 데이터셋도 섞지 않는 순수 순정 로직
# =====================================================================
@mcp.tool(name="search_humanoid_ai_model")
def search_humanoid_ai_model(keyword: str = "") -> dict:
    reg = load_registry_config().get("robot", {})
    total_needed = int(reg.get("fetch_limit", 10))
    half = total_needed // 2
    
    arxiv = _execute_real_arxiv_api(reg.get("arxiv_keywords", "VLA robot foundation"), half)
    news = _execute_real_google_news_api(reg.get("news_keywords", "로봇 인공지능"), half)
    return {"domain": reg.get("domain_name"), "data": arxiv + news}

@mcp.tool(name="search_amr_logistics_tech")
def search_amr_logistics_tech(keyword: str = "") -> dict:
    reg = load_registry_config().get("amr", {})
    total_needed = int(reg.get("fetch_limit", 10))
    half = total_needed // 2
    
    arxiv = _execute_real_arxiv_api(reg.get("arxiv_keywords", "AMR SLAM robot"), half)
    news = _execute_real_google_news_api(reg.get("news_keywords", "물류 자율주행 로봇"), half)
    return {"domain": reg.get("domain_name"), "data": arxiv + news}

@mcp.tool(name="search_robot_hardware_tech")
def search_robot_hardware_tech(keyword: str = "") -> dict:
    reg = load_registry_config().get("hw", {})
    total_needed = int(reg.get("fetch_limit", 10))
    half = total_needed // 2
    
    arxiv = _execute_real_arxiv_api(reg.get("actuator motor", "robotics actuator motor"), half)
    news = _execute_real_google_news_api(reg.get("news_keywords", "로봇 모터 액추에이터"), half)
    return {"domain": reg.get("domain_name"), "data": arxiv + news}

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8765, path="/mcp")