# -*- coding: utf-8 -*-
"""
my_mcp_server - MCP Schemas (contracts → MCP Tool schema 변환)

[이 모듈의 역할]
my_contracts.py의 Tool 정의(ALLOWED_TOOLS, TOOL_CONTRACTS)를 읽어,
MCP Tool의 input schema(JSON Schema 형태 dict)로 '변환만' 합니다.

[가장 중요한 원칙 — schema 중복 정의 금지]
Tool 이름이나 argument 목록을 새로 정의하지 않습니다.
모든 기준은 my_contracts.py 한 곳(single source of truth)에서만 옵니다.

[전체 흐름에서의 위치]
    my_contracts.py (기준)
        ↓ 변환
    [my_mcp_schemas]  → list_mcp_tools()가 노출할 Tool schema 생성
        ↓
    my_mcp_server.list_mcp_tools()
"""
from my_contracts import ALLOWED_TOOLS, TOOL_CONTRACTS

# ---------------------------------------------------------------------------
# argument 이름 → JSON Schema type 매핑
# ---------------------------------------------------------------------------
_ARG_TYPE_HINTS = {
    "limit":       "integer",
    "max_results": "integer",
}


def _arg_type(arg_name):
    """argument 이름으로 JSON Schema type 문자열을 돌려준다(기본 string)."""
    return _ARG_TYPE_HINTS.get(arg_name, "string")


def get_exposed_mcp_tool_names():
    """MCP로 노출할 Tool 이름 목록을 돌려준다.

    [반환] list[str] — ALLOWED_TOOLS와 동일(크롤러 서버는 금지 tool이 없음).
    """
    return list(ALLOWED_TOOLS)


def build_single_tool_schema(tool_name):
    """tool_name 하나에 대한 MCP Tool schema dict를 contracts 기준으로 만든다.

    [반환]
        {
          "name": <tool_name>,
          "description": <purpose>,
          "inputSchema": {
            "type": "object",
            "properties": { <arg>: {"type":..., "description":...}, ... },
            "required": [ <required_arguments> ],
            "additionalProperties": False
          }
        }
    """
    if tool_name not in TOOL_CONTRACTS:
        raise ValueError(f"노출할 수 없는 Tool입니다: {tool_name}")

    contract = TOOL_CONTRACTS[tool_name]
    required = list(contract.get("required_arguments", []) or [])
    any_of   = list(contract.get("any_of_required_arguments", []) or [])
    optional = list(contract.get("optional_arguments", []) or [])

    # properties: required + any_of + optional 모든 argument 포함(중복 방지)
    properties = {}
    for arg in required + any_of + optional:
        properties[arg] = {
            "type":        _arg_type(arg),
            "description": _arg_description(tool_name, arg, required, any_of),
        }

    # default_values를 JSON Schema default로 추가
    defaults = contract.get("default_values", {}) or {}
    for arg, default in defaults.items():
        if arg in properties:
            properties[arg]["default"] = default

    description = contract.get("purpose", "")
    if any_of:
        description += f" (다음 argument 중 하나 이상 필요: {', '.join(any_of)})"

    return {
        "name":        tool_name,
        "description": description,
        "inputSchema": {
            "type":                 "object",
            "properties":           properties,
            "required":             required,
            "additionalProperties": False,
        },
    }


def _arg_description(tool_name, arg_name, required, any_of):
    """argument 한 개의 description 문자열을 만든다(required/any_of 여부 표시)."""
    if arg_name in required:
        role = "필수"
    elif arg_name in any_of:
        role = "다음 중 하나 이상"
    else:
        role = "선택"
    when = TOOL_CONTRACTS.get(tool_name, {}).get("when_to_use", "")
    defaults = TOOL_CONTRACTS.get(tool_name, {}).get("default_values", {}) or {}
    default_str = f" (기본값: {defaults[arg_name]})" if arg_name in defaults else ""
    return f"{arg_name} ({role}){default_str}. 사용 맥락: {when}"


def build_mcp_tool_schemas():
    """노출 대상 전체 Tool의 MCP schema 리스트를 만든다.

    [반환] list[dict] — 각 Tool에 대한 build_single_tool_schema() 결과 목록.
    """
    return [build_single_tool_schema(name) for name in get_exposed_mcp_tool_names()]


if __name__ == "__main__":
    import json
    schemas = build_mcp_tool_schemas()
    print(f"[my_mcp_schemas] 노출 Tool 수: {len(schemas)}")
    for s in schemas:
        props = list(s["inputSchema"]["properties"].keys())
        print(f"  {s['name']}: properties={props}, required={s['inputSchema']['required']}")
