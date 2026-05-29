# C:\work\agent260528\src\robot_mcp_client.py
from __future__ import annotations
import os
import yaml
import json
import requests
from pathlib import Path

SUPPORTED_TOOLS = ["search_robot_hardware_tech", "search_humanoid_ai_model", "search_amr_logistics_tech"]

def load_config_values() -> tuple[str, str]:
    mode = "fastmcp"
    url = "http://127.0.0.1:8765"
    try:
        project_root = Path(__file__).resolve().parents[1]
        config_path = project_root / "config.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
                mcp_cfg = cfg.get("mcp", {})
                mode = mcp_cfg.get("mode", mode)
                url = mcp_cfg.get("fastmcp_url", url)
    except Exception:
        pass
    if url.endswith("/mcp"): 
        url = url[:-4]
    return mode, url

def load_tool_caller(mode: str = None) -> tuple:
    return call_tool_via_fastmcp, "fastmcp"

def call_tool_via_fastmcp(tool_name: str, tool_input: dict) -> dict:
    if tool_name not in SUPPORTED_TOOLS: 
        raise ValueError(f"보안 제한 도구: {tool_name}")
    _, server_url = load_config_values()
    url = f"{server_url}/tools/{tool_name}"
    try:
        response = requests.post(url, json=tool_input, timeout=10)
        if response.status_code != 200: 
            return {"status": "error", "data": []}
        resp_json = response.json()
        if isinstance(resp_json, dict) and "content" in resp_json:
            contents = resp_json["content"]
            if contents and isinstance(contents, list) and len(contents) > 0:
                text_payload = contents[0].get("text", "")
                if text_payload: 
                    return json.loads(text_payload)
        return resp_json if (isinstance(resp_json, dict) and "data" in resp_json) else {"status": "error", "data": []}
    except Exception:
        return {"status": "error", "data": []}