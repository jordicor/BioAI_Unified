#!/usr/bin/env python3
"""
Quick test for Gemini JSON Schema + Thinking compatibility.

Tests the exact scenario described in revisa-esto.txt:
- gemini-3-pro-preview with "reasoning" capability
- json_schema provided
- NO thinking_budget_tokens sent (will auto-apply 32768)

Usage:
    python dev_tests/test_gemini_json_thinking_quick.py
"""

import asyncio
import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json_utils as json
from ai_service import AIService
from datetime import datetime

MODEL_ID = "gemini-3-pro-preview"

PROMPT = "Generate 3 video titles about cooking pasta. Output JSON only."

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"}
                },
                "required": ["number", "title"]
            }
        }
    },
    "required": ["titles"]
}


async def test_scenario(ai_service: AIService, name: str, **kwargs):
    """Run a single test scenario."""
    print(f"\n--- {name} ---")
    print(f"    json_output: {kwargs.get('json_output', False)}")
    print(f"    json_schema: {'YES' if kwargs.get('json_schema') else 'NO'}")
    print(f"    thinking_budget_tokens: {kwargs.get('thinking_budget_tokens', 'NOT SENT')}")

    start = datetime.now()
    try:
        content = await ai_service.generate_content(
            prompt=PROMPT,
            model=MODEL_ID,
            temperature=0.7,
            max_tokens=2000,
            system_prompt="Generate YouTube titles.",
            **kwargs
        )
        elapsed = (datetime.now() - start).total_seconds()
        word_count = len(content.split()) if content else 0

        print(f"    Result: {word_count} words in {elapsed:.1f}s")
        if content:
            print(f"    Preview: {content[:150]}...")

        return {"success": word_count > 0, "word_count": word_count, "elapsed": elapsed}

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"    ERROR: {e}")
        return {"success": False, "error": str(e), "elapsed": elapsed}


async def main():
    print("=" * 60)
    print("  QUICK GEMINI JSON SCHEMA + THINKING TEST")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Started: {datetime.now().isoformat()}")

    ai_service = AIService()

    if not ai_service.google_new_client:
        print("\n[ERROR] Google Gemini client not initialized")
        return

    results = {}

    # Test 1: JSON Schema + NO thinking_budget_tokens (auto-applies)
    results["auto_thinking"] = await test_scenario(
        ai_service,
        "Test 1: JSON Schema + AUTO thinking (NO thinking_budget_tokens sent)",
        json_output=True,
        json_schema=JSON_SCHEMA,
        # NOT sending thinking_budget_tokens
    )

    # Test 2: JSON Schema + explicit thinking_budget_tokens
    results["explicit_thinking"] = await test_scenario(
        ai_service,
        "Test 2: JSON Schema + EXPLICIT thinking (8192)",
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget_tokens=8192,
    )

    # Test 3: JSON mode (no schema) + NO thinking_budget_tokens
    results["flexible_auto"] = await test_scenario(
        ai_service,
        "Test 3: JSON Mode (no schema) + AUTO thinking",
        json_output=True,
        # No json_schema - flexible mode
        # Not sending thinking_budget_tokens
    )

    # Test 4: No JSON mode at all + AUTO thinking
    results["no_json"] = await test_scenario(
        ai_service,
        "Test 4: Plain text (no JSON) + AUTO thinking",
        json_output=False,
        # Not sending thinking_budget_tokens
    )

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for name, r in results.items():
        status = "[OK]" if r.get("success") else "[FAIL]"
        error = r.get("error", "")
        words = r.get("word_count", 0)
        print(f"  {name}: {status} - {words} words {error}")

    # Analysis
    print("\n" + "-" * 50)
    if results["auto_thinking"].get("success"):
        print("[OK] JSON Schema + auto-thinking works correctly")
        print("    The issue described in revisa-esto.txt is NOT reproducible")
    else:
        print("[PROBLEM] JSON Schema + auto-thinking FAILS")
        if results["explicit_thinking"].get("success"):
            print("    But explicit thinking works - issue may be with auto-apply logic")
        if results["flexible_auto"].get("success"):
            print("    JSON flexible mode works - issue is specific to response_schema + thinking_config")

    # Save results
    output = {
        "test_date": datetime.now().isoformat(),
        "model": MODEL_ID,
        "results": results
    }
    output_file = "dev_tests/gemini_quick_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
