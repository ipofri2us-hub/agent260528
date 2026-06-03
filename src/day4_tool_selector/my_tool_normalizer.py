# -*- coding: utf-8 -*-
"""
로봇 정보 검색 Tool Plan 정규화 모듈

my_tool_selection.select_tools()가 만든 raw plan을 검증 전에 보정합니다.

정규화 순서:
  1. 허용되지 않은 tool 제거 (ALLOWED_TOOLS 기준)
  2. Tool Contract 기본값으로 누락 argument 채우기
  3. step 번호 1부터 재정렬

핵심 원칙:
  - expected 정답(없음)을 참조하지 않습니다.
  - Tool Contract와 쿼리 텍스트만을 근거로 보정합니다.
  - 정규화는 Contract 완화가 아닌 구조 보정입니다.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from my_tool_selection import TOOL_CONTRACTS, ALLOWED_TOOLS


# ───────────────────────────────────────────────────────────
# 내부 헬퍼
# ───────────────────────────────────────────────────────────
def _fill_defaults(item: dict) -> tuple[dict, list]:
    """
    Tool Contract 기본값으로 누락된 optional argument를 채웁니다.

    반환: (보정된 item dict, 채워진 argument 이름 목록)
    """
    tool_name = item.get("tool_name", "")
    contract  = TOOL_CONTRACTS.get(tool_name, {})
    defaults  = contract.get("default_values", {})
    args      = dict(item.get("arguments") or {})
    filled    = []

    for key, default_val in defaults.items():
        if args.get(key) in (None, ""):
            args[key] = default_val
            filled.append(key)

    return {**item, "arguments": args}, filled


def _renumber(items: list) -> list:
    """plan item의 step 번호를 1부터 순서대로 재정렬합니다."""
    for idx, item in enumerate(items, 1):
        item["step"] = idx
    return items


# ───────────────────────────────────────────────────────────
# 공개 인터페이스
# ───────────────────────────────────────────────────────────
def normalize_plan(plan_result: dict) -> dict:
    """
    plan_result["plan"]을 정규화하고 보정된 plan_result를 반환합니다.

    매개변수:
      plan_result : select_tools()가 반환한 dict.
                    "plan" 키에 list[dict]가 있어야 합니다.

    반환:
      보정된 plan_result (원본 dict의 shallow copy + "plan" 키 교체).
      "normalization_log" 키에 보정 이력이 추가됩니다.

    보정 단계:
      1. 허용되지 않은 tool 제거
         → ALLOWED_TOOLS에 없는 tool_name을 가진 item을 plan에서 제거합니다.
      2. 기본값 채우기
         → Tool Contract의 default_values에서 누락된 argument를 채웁니다.
      3. step 번호 재정렬
         → 제거된 항목이 있으면 step이 불연속이 되므로 1부터 재번호합니다.
    """
    raw_items = plan_result.get("plan", [])
    log       = []
    new_items = []

    for item in raw_items:
        if not isinstance(item, dict):
            log.append({"action": "skip", "reason": "item이 dict가 아님"})
            continue

        tool_name = item.get("tool_name", "")

        # 단계 1: 허용되지 않은 tool 제거
        if tool_name not in ALLOWED_TOOLS:
            log.append({
                "action":  "removed",
                "tool":    tool_name,
                "reason":  f"'{tool_name}'은 ALLOWED_TOOLS에 없음",
            })
            continue

        # 단계 2: 기본값 채우기
        item, filled = _fill_defaults(item)
        if filled:
            log.append({
                "action":  "default_filled",
                "tool":    tool_name,
                "filled":  filled,
            })

        new_items.append(item)

    # 단계 3: step 번호 재정렬
    new_items = _renumber(new_items)

    # 로그 출력 (정보용)
    for entry in log:
        action = entry["action"]
        if action == "removed":
            print(f"  [정규화] ❌ 제거 — {entry['tool']}: {entry['reason']}")
        elif action == "default_filled":
            print(f"  [정규화] ✅ 기본값 채움 — {entry['tool']}: {entry['filled']}")

    return {
        **plan_result,
        "plan":              new_items,
        "normalization_log": log,
    }


# ───────────────────────────────────────────────────────────
# 단독 실행용 self-test
# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from my_tool_selection import select_tools

    test_query = "ROS2 navigation AMR"
    raw = select_tools(test_query)
    print(f"[raw plan]  항목 수: {len(raw['plan'])}")

    # 의도적으로 unknown tool 삽입
    raw["plan"].append({
        "step": 99,
        "tool_name": "bad_tool",
        "arguments": {},
        "condition": "always",
        "reason": "테스트용 잘못된 tool",
    })

    normalized = normalize_plan(raw)
    print(f"[정규화 후] 항목 수: {len(normalized['plan'])}")
    for item in normalized["plan"]:
        print(f"  step={item['step']} | {item['tool_name']} | args={item['arguments']}")
