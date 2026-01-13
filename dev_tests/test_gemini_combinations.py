#!/usr/bin/env python3
"""
Test different combinations of json_schema and thinking_budget_tokens
using the EXACT prompt from failing session 38408446-2822-43b8-82ec-c5cb6db9559d.

Usage:
    python dev_tests/test_gemini_combinations.py
"""

import sqlite3
import json
import requests
import time
import sys
import io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE_URL = "http://localhost:8000"

def get_original_request():
    """Get the exact request from the failing session."""
    conn = sqlite3.connect('S:/01.Coding/GranSabio_LLM/debugger_history.db')
    cursor = conn.execute(
        'SELECT request_json FROM sessions WHERE session_id = ?',
        ('38408446-2822-43b8-82ec-c5cb6db9559d',)
    )
    payload = json.loads(cursor.fetchone()[0])
    conn.close()
    return payload


def run_test(name: str, payload: dict) -> dict:
    """Run a single test and return result."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    print(f"  json_schema: {'YES' if payload.get('json_schema') else 'NO'}")
    print(f"  thinking_budget_tokens: {payload.get('thinking_budget_tokens', 'NOT SET (auto)')}")

    start = time.time()

    try:
        resp = requests.post(f"{BASE_URL}/generate", json=payload, timeout=15)
        if resp.status_code != 200:
            print(f"  HTTP Error: {resp.status_code}")
            return {"success": False, "error": f"HTTP {resp.status_code}"}

        session_id = resp.json().get("session_id")
        print(f"  Session: {session_id}")

        # Poll for result
        for i in range(90):
            time.sleep(1)
            status = requests.get(f"{BASE_URL}/status/{session_id}").json()

            if status["status"] in ["completed", "failed"]:
                result = requests.get(f"{BASE_URL}/result/{session_id}").json()
                content = result.get("content", "")
                if isinstance(content, dict):
                    content = json.dumps(content)

                elapsed = time.time() - start
                final_iter = result.get("final_iteration")
                generator_empty = final_iter == "Gran Sabio"

                print(f"  Elapsed: {elapsed:.1f}s")
                print(f"  Final iteration: {final_iter}")
                print(f"  Content length: {len(content)} chars")

                if generator_empty:
                    print(f"  >>> GENERATOR RETURNED EMPTY - Gran Sabio saved it <<<")
                elif len(content) > 0:
                    print(f"  [OK] Generator produced content")
                else:
                    print(f"  [FAIL] No content at all")

                return {
                    "success": len(content) > 0 and not generator_empty,
                    "generator_empty": generator_empty,
                    "content_length": len(content),
                    "elapsed": elapsed,
                    "final_iteration": final_iter,
                    "session_id": session_id
                }

            if i % 20 == 0 and i > 0:
                print(f"  [{i}s] waiting...")

        return {"success": False, "error": "Timeout"}

    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    print("=" * 70)
    print("  GEMINI COMBINATIONS TEST")
    print("  Using exact prompt from failing session")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    # Get original request
    original = get_original_request()
    original_schema = original.get("json_schema")

    print(f"\nOriginal prompt length: {len(original['prompt'])} chars")
    print(f"Original model: {original['generator_model']}")

    # Base payload (copy from original)
    def make_payload(**overrides):
        p = {
            "prompt": original["prompt"],
            "generator_model": original["generator_model"],
            "temperature": original.get("temperature", 0.7),
            "max_tokens": original.get("max_tokens", 8000),
            "json_output": True,
            "qa_layers": [],
            "max_iterations": 1,
            "gran_sabio_fallback": True,  # Keep enabled to see if generator fails
            "gran_sabio_model": original.get("gran_sabio_model", "claude-opus-4-5-20251101"),
            "verbose": True,
        }
        p.update(overrides)
        return p

    results = {}

    # Test 1: Original failing case (schema + auto thinking)
    results["1_schema_auto_thinking"] = run_test(
        "1. JSON Schema + AUTO thinking (original failing case)",
        make_payload(json_schema=original_schema)
        # No thinking_budget_tokens = auto-applies 32768
    )

    # Test 2: Schema + MINIMAL thinking
    results["2_schema_min_thinking"] = run_test(
        "2. JSON Schema + MINIMAL thinking (thinking_budget_tokens=1024)",
        make_payload(json_schema=original_schema, thinking_budget_tokens=1024)
    )

    # Test 3: Schema + LOW thinking
    results["3_schema_low_thinking"] = run_test(
        "3. JSON Schema + LOW thinking (thinking_budget_tokens=4096)",
        make_payload(json_schema=original_schema, thinking_budget_tokens=4096)
    )

    # Test 4: Schema + MEDIUM thinking
    results["4_schema_medium_thinking"] = run_test(
        "4. JSON Schema + MEDIUM thinking (thinking_budget_tokens=16384)",
        make_payload(json_schema=original_schema, thinking_budget_tokens=16384)
    )

    # Test 5: Schema + HIGH thinking (same as auto default)
    results["5_schema_high_thinking"] = run_test(
        "5. JSON Schema + HIGH thinking (thinking_budget_tokens=32768)",
        make_payload(json_schema=original_schema, thinking_budget_tokens=32768)
    )

    # Test 6: NO schema + AUTO thinking
    results["6_no_schema_auto_thinking"] = run_test(
        "6. NO JSON Schema + AUTO thinking",
        make_payload(json_schema=None)
        # No thinking_budget_tokens = auto-applies 32768
    )

    # Test 7: NO schema + HIGH thinking (explicit)
    results["7_no_schema_high_thinking"] = run_test(
        "7. NO JSON Schema + HIGH thinking (32768)",
        make_payload(json_schema=None, thinking_budget_tokens=32768)
    )

    # Test 8: NO schema + MINIMAL thinking
    results["8_no_schema_min_thinking"] = run_test(
        "8. NO JSON Schema + MINIMAL thinking (1024)",
        make_payload(json_schema=None, thinking_budget_tokens=1024)
    )

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n| # | Test                              | Schema | Thinking | Result      |")
    print("|---|-----------------------------------|--------|----------|-------------|")

    for key, r in results.items():
        num = key.split("_")[0]
        name = key[2:].replace("_", " ")[:30]
        schema = "YES" if "schema" in key and "no_schema" not in key else "NO"

        if "auto" in key:
            thinking = "AUTO(32k)"
        elif "min_thinking" in key:
            thinking = "1024"
        elif "low" in key:
            thinking = "4096"
        elif "medium" in key:
            thinking = "16384"
        elif "high" in key:
            thinking = "32768"
        else:
            thinking = "?"

        if r.get("generator_empty"):
            result = "GEN EMPTY"
        elif r.get("success"):
            result = "OK"
        else:
            result = f"FAIL: {r.get('error', '?')[:15]}"

        print(f"| {num} | {name:<33} | {schema:<6} | {thinking:<8} | {result:<11} |")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print("-" * 70)

    schema_thinking_fails = [k for k, v in results.items()
                            if "schema" in k and "no_schema" not in k
                            and v.get("generator_empty")]

    no_schema_thinking_works = [k for k, v in results.items()
                                if "no_schema" in k and v.get("success")]

    if schema_thinking_fails:
        print(f"\nFailed with JSON Schema + thinking: {schema_thinking_fails}")
    if no_schema_thinking_works:
        print(f"\nWorked without JSON Schema: {no_schema_thinking_works}")

    # Save results
    with open("dev_tests/gemini_combinations_results.json", "w") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    print("\nResults saved to: dev_tests/gemini_combinations_results.json")


if __name__ == "__main__":
    main()
