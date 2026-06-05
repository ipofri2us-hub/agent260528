# 트러블슈팅 로그

---

## 2026-06-05 — MCP 서버 크롤러 0건 반환

### 증상
- `search_rfm_news`, `search_vla_news`, `search_imitation_learning_news` 실행 시 ArXiv 논문 0건
- `search_reddit_robotics` 실행 시 Reddit 게시물 0건

### 원인

#### 1. lxml 미설치 (Reddit, Google News 0건)
`my_mcp_server.py`의 `_reddit_robotics_feed`, `_google_news_search` 함수는 `BeautifulSoup(content, "xml")` 파서를 사용한다.  
이 파서는 내부적으로 `lxml`을 요구하는데, venv가 uv Python(`cpython-3.11.15`)으로 생성되어 있어 시스템에 lxml이 있어도 venv에는 포함되지 않는다.  
결과적으로 `FeatureNotFound` 예외가 발생하고, 각 함수의 `except Exception: return []` 블록에서 조용히 빈 리스트를 반환했다.

> 주의: `pip show lxml`이 설치된 것처럼 보이는 건 시스템 패키지(`/usr/lib/python3/dist-packages`) 때문이며, venv `sys.path`에는 포함되지 않는다.

#### 2. ArXiv HTTP 429 (ArXiv 0건)
`_arxiv_search` 함수가 `requests.get` 호출 시 헤더를 전달하지 않았다.  
ArXiv `export.arxiv.org` 엔드포인트는 헤더 없는 요청에 대해 레이트 리밋(429)을 반환하고, 코드는 `status_code != 200`이면 즉시 `[]`를 반환했다.

### 조치

#### lxml 설치
```bash
source /home/kjs0503/src/agent260528/.venv/bin/activate
uv pip install lxml
```
`pip install lxml`은 PEP 668로 차단됨 — 반드시 `uv pip install` 사용.

#### ArXiv 재시도 로직 추가 (`my_mcp_server.py`)
- `_HEADERS` User-Agent 헤더를 ArXiv 요청에도 전달
- 429 응답 시 `Retry-After` 헤더 값만큼 대기 후 최대 3회 재시도

```python
import time

arxiv_headers = {**_HEADERS, "Accept": "application/atom+xml"}
for attempt in range(3):
    resp = requests.get(url, params=params, headers=arxiv_headers, timeout=15)
    if resp.status_code == 429:
        wait = int(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
        time.sleep(wait)
        continue
    break
```

### 확인
수정 후 세 소스 모두 정상 반환:
- Reddit: 5건
- Google News: 3건
- ArXiv: 3건

### 재발 방지
venv를 재생성하거나 환경을 옮길 때마다 lxml 재설치 필요:
```bash
uv pip install lxml
```
