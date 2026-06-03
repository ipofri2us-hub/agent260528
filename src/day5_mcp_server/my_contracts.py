# -*- coding: utf-8 -*-
"""
my_mcp_server - Contracts (허용 Tool 목록 및 Tool Contract 기준표)

[이 모듈의 역할]
my_mcp_server 파이프라인(schemas → selector → executor → server)이
공통으로 참조하는 '기준표(Tool Contract)' 모듈입니다.

[크롤링 대상]
  1. Reddit r/robotics  : Google News RSS 경유 (봇 차단 우회)
  2. arXiv cs.RO        : arXiv Atom API
  3. 영문 로봇 뉴스     : Robot Report / Robohub RSS
  4. 로봇신문           : irobotnews.com (Google News RSS 한국어)

[설계 원칙]
- 이 파일은 다른 my_* 모듈을 import하지 않는 leaf 모듈입니다.
- Tool 이름·argument·기본값의 single source of truth 입니다.
  새 크롤러 tool을 추가할 때 이 파일과 my_crawlers.py 두 곳만 수정하면 됩니다.
- my_tool_contracts.json의 내용을 Python 상수로 정의합니다
  (런타임 JSON 파일 의존성 제거 → import 오류 방지).
"""

# ---------------------------------------------------------------------------
# Allowed Tools (허용 크롤러 tool 목록)
# ---------------------------------------------------------------------------
ALLOWED_TOOLS = [
    "search_reddit",           # r/robotics 커뮤니티 게시물 (Google News RSS)
    "search_arxiv",            # arXiv cs.RO 학술 논문
    "search_robot_news",       # 영문 로봇 뉴스 RSS (Robot Report / Robohub)
    "search_irobotnews",       # 로봇신문 한국어 뉴스 (Google News RSS)
    "search_robot_db",         # 로봇 기술 논문/뉴스 캐시 DB 조회 (LanceDB D:/lance_db)
]

# ---------------------------------------------------------------------------
# Tool Contract (각 tool의 목적·인수·기본값 기준표)
# ---------------------------------------------------------------------------
TOOL_CONTRACTS = {
    "search_reddit": {
        "purpose": "r/robotics 커뮤니티 최신 토론·게시물을 수집한다 (Google News RSS 경유).",
        "when_to_use": "로봇 커뮤니티 동향, Reddit 토론, r/robotics 게시물 검색 요청.",
        "when_not_to_use": "학술 논문이나 공식 뉴스만 필요한 경우.",
        "required_arguments": ["query"],
        "any_of_required_arguments": [],
        "optional_arguments": ["limit", "subreddit"],
        "default_values": {"limit": 5, "subreddit": "robotics"},
    },
    "search_arxiv": {
        "purpose": "arXiv cs.RO 카테고리 최신 로봇공학 논문을 검색한다.",
        "when_to_use": "학술 논문, 연구 트렌드, 알고리즘·기술 깊이 있는 내용 요청.",
        "when_not_to_use": "커뮤니티 동향이나 뉴스만 필요한 경우.",
        "required_arguments": ["query"],
        "any_of_required_arguments": [],
        "optional_arguments": ["max_results", "category"],
        "default_values": {"max_results": 5, "category": "cs.RO"},
    },
    "search_robot_news": {
        "purpose": "영문 로봇 전문 뉴스 RSS(Robot Report / Robohub)에서 기사를 수집한다.",
        "when_to_use": "영문 업계 뉴스, 제품 출시, 회사·시장 동향 요청.",
        "when_not_to_use": "한국어 뉴스나 학술 자료만 필요한 경우.",
        "required_arguments": ["query"],
        "any_of_required_arguments": [],
        "optional_arguments": ["limit", "source"],
        "default_values": {"limit": 5, "source": "all"},
    },
    "search_irobotnews": {
        "purpose": "로봇신문(irobotnews.com) 한국어 뉴스를 수집한다 (Google News RSS 경유).",
        "when_to_use": "한국어 로봇 뉴스, 국내 로봇 산업 동향 요청.",
        "when_not_to_use": "영문 자료나 학술 논문만 필요한 경우.",
        "required_arguments": ["query"],
        "any_of_required_arguments": [],
        "optional_arguments": ["limit"],
        "default_values": {"limit": 5},
    },
    "search_robot_db": {
        "purpose": "D:/lance_db 에 캐싱된 로봇 기술 논문·뉴스·토픽을 로컬 LanceDB 에서 조회한다.",
        "when_to_use": "크롤러가 이미 수집·저장한 데이터를 빠르게 재조회할 때. "
                       "DB 미준비 시 빈 결과를 반환하고 크롤러 Tool 로 보완한다.",
        "when_not_to_use": "실시간 최신 뉴스·논문이 필요한 경우(크롤러 Tool 을 사용).",
        "required_arguments": ["query"],
        "any_of_required_arguments": [],
        "optional_arguments": ["data_type", "category", "source"],
        "default_values": {"data_type": "papers"},
    },
}

# ---------------------------------------------------------------------------
# Plan item 허용 필드 / reason 최소 길이
# ---------------------------------------------------------------------------
ALLOWED_PLAN_ITEM_FIELDS = {"step", "tool_name", "arguments", "condition", "reason"}
MIN_REASON_LENGTH = 10

# ---------------------------------------------------------------------------
# RSS 소스 (search_robot_news 가 참조하는 영문 뉴스 피드)
# ---------------------------------------------------------------------------
RSS_SOURCES = {
    "robot_report": "https://www.therobotreport.com/feed/",
    "robohub":      "https://robohub.org/feed/",
}

# ---------------------------------------------------------------------------
# 주제 키워드 / 검색어 확장
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS = {
    "HW":  ["hardware", "hw", "센서", "액추에이터", "그리퍼", "서보", "모터",
            "actuator", "sensor", "gripper", "servo", "motor", "pcb", "embedded"],
    "ROS": ["ros", "ros2", "rclpy", "colcon", "gazebo", "rviz",
            "navigation", "node", "topic", "service", "action", "launch"],
    "AMR": ["amr", "autonomous mobile", "자율주행", "slam", "lidar",
            "path planning", "obstacle", "fleet", "warehouse", "agv"],
    "RFM": ["rfm", "framework", "middleware", "dds", "protocol", "communication"],
}

TOPIC_ENRICHMENT = {
    "HW":  "robot hardware actuator sensor gripper servo motor pcb embedded",
    "ROS": "ROS2 robot operating system navigation2 rclpy gazebo colcon launch",
    "AMR": "autonomous mobile robot SLAM lidar path planning obstacle fleet warehouse",
    "RFM": "robot framework middleware DDS communication protocol ROS bridge",
}

# Reddit Google News 결과가 0건일 때 사용하는 대체 토픽
REDDIT_FALLBACK_TOPICS = [
    ["Is ROS2 Humble stable enough for production logistics AMRs?",
     "https://www.reddit.com/r/robotics/search/?q=ROS2+AMR+production"],
    ["Unitree G1 vs Figure 01: actuator torque density comparison",
     "https://www.reddit.com/r/robotics/search/?q=humanoid+actuator"],
    ["Best SLAM libraries for ROS2 in 2025: Cartographer vs RTAB-Map vs LIO-SAM",
     "https://www.reddit.com/r/robotics/search/?q=SLAM+ROS2"],
    ["How to choose between servo, BLDC, and stepper for a 6-DOF arm HW",
     "https://www.reddit.com/r/robotics/search/?q=servo+motor+arm"],
    ["Open-source AMR fleet management: ROS2 Nav2 vs custom middleware",
     "https://www.reddit.com/r/robotics/search/?q=AMR+fleet+ROS2"],
]

# ---------------------------------------------------------------------------
# Severity 순서
# ---------------------------------------------------------------------------
SEVERITY_ORDER = {
    "PASS": 0,
    "WARNING": 1,
    "FAIL": 2,
    "NEEDS_REVIEW": 3,
}


if __name__ == "__main__":
    print(f"[my_contracts] ALLOWED_TOOLS: {ALLOWED_TOOLS}")
    for name, c in TOOL_CONTRACTS.items():
        print(f"  {name}: required={c['required_arguments']}, "
              f"optional={c['optional_arguments']}")
