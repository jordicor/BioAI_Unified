#!/usr/bin/env python3
"""
Test to verify if Gemini auto-applies thinking budget when user doesn't send it.

According to revisa-esto.txt, the problem is:
1. gemini-3-pro-preview has "reasoning" in capabilities
2. It has thinking_budget.default_tokens: 32768 in model_specs.json
3. When user does NOT send thinking_budget_tokens, config.py applies it automatically
4. This creates the combination JSON Schema + thinking_config that may cause empty responses

This test:
1. Verifies config.py behavior directly
2. Tests API without thinking_budget_tokens to see auto-application
3. Compares with explicit thinking_budget_tokens

Usage:
    python dev_tests/test_gemini_auto_thinking_json_schema.py
"""

import asyncio
import requests
import time
import sys
import os
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import json_utils as json
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
MODEL_ID = "gemini-3-pro-preview"

PROMPT = """Generate 3 creative YouTube video titles about cooking pasta.
Respond with valid JSON containing a "titles" array with "number" and "title" fields."""

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


def print_header(title: str):
    """Print a header with separators."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_config_behavior():
    """Test 1: Verify config.py auto-applies thinking budget."""
    print_header("TEST 1: CONFIG.PY BEHAVIOR ANALYSIS")

    cfg = config.config

    # Test 1a: With NO thinking_budget_tokens
    print("\n[1a] Calling validate_token_limits WITHOUT thinking_budget_tokens...")
    result_no_thinking = cfg.validate_token_limits(
        model_name=MODEL_ID,
        max_tokens=8000,
        reasoning_effort=None,
        thinking_budget_tokens=None  # <-- Not sending this
    )

    auto_thinking = result_no_thinking.get("adjusted_thinking_budget_tokens")
    print(f"    Model: {MODEL_ID}")
    print(f"    Input thinking_budget_tokens: None")
    print(f"    Output adjusted_thinking_budget_tokens: {auto_thinking}")

    if auto_thinking and auto_thinking > 0:
        print(f"    [!] CONFIG AUTO-APPLIED thinking_budget: {auto_thinking} tokens")
    else:
        print(f"    [OK] No auto-application of thinking budget")

    # Test 1b: With explicit thinking_budget_tokens = 0
    print("\n[1b] Calling validate_token_limits WITH thinking_budget_tokens=0...")
    result_zero_thinking = cfg.validate_token_limits(
        model_name=MODEL_ID,
        max_tokens=8000,
        reasoning_effort=None,
        thinking_budget_tokens=0  # <-- Explicitly zero
    )

    zero_thinking_result = result_zero_thinking.get("adjusted_thinking_budget_tokens")
    print(f"    Input thinking_budget_tokens: 0")
    print(f"    Output adjusted_thinking_budget_tokens: {zero_thinking_result}")

    # Test 1c: With explicit thinking_budget_tokens
    print("\n[1c] Calling validate_token_limits WITH thinking_budget_tokens=8192...")
    result_explicit = cfg.validate_token_limits(
        model_name=MODEL_ID,
        max_tokens=8000,
        reasoning_effort=None,
        thinking_budget_tokens=8192
    )

    explicit_result = result_explicit.get("adjusted_thinking_budget_tokens")
    print(f"    Input thinking_budget_tokens: 8192")
    print(f"    Output adjusted_thinking_budget_tokens: {explicit_result}")

    # Summary
    print("\n" + "-" * 50)
    print("CONFIG BEHAVIOR SUMMARY:")
    print("-" * 50)
    print(f"  - None  -> {auto_thinking} (auto-applied: {'YES' if auto_thinking else 'NO'})")
    print(f"  - 0     -> {zero_thinking_result} (auto-applied: {'YES' if zero_thinking_result else 'NO'})")
    print(f"  - 8192  -> {explicit_result}")

    return {
        "no_thinking_input": auto_thinking,
        "zero_thinking_input": zero_thinking_result,
        "explicit_thinking_input": explicit_result,
        "auto_applies": auto_thinking is not None and auto_thinking > 0
    }


def test_api_without_thinking(session_timeout: int = 60) -> dict:
    """Test 2: API request WITHOUT thinking_budget_tokens."""
    print_header("TEST 2: API REQUEST WITHOUT thinking_budget_tokens")

    print(f"\n[API] Sending request to {BASE_URL}/generate")
    print(f"      Model: {MODEL_ID}")
    print(f"      json_schema: YES")
    print(f"      thinking_budget_tokens: NOT SENT (will config auto-apply?)")

    payload = {
        "prompt": PROMPT,
        "generator_model": MODEL_ID,
        "temperature": 0.7,
        "max_tokens": 2000,
        "qa_layers": [],
        "max_iterations": 1,
        "gran_sabio_fallback": False,
        "json_output": True,
        "json_schema": JSON_SCHEMA,
        "verbose": True,
        # NOT sending thinking_budget_tokens
    }

    start_time = time.time()
    error = None
    content = ""
    session_id = None

    try:
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        session_id = data.get("session_id")
        print(f"      Session ID: {session_id}")

        # Poll for result
        for i in range(session_timeout):
            time.sleep(1)
            status_resp = requests.get(f"{BASE_URL}/status/{session_id}", timeout=5)
            status_data = status_resp.json()
            status = status_data.get("status")

            if status in ["completed", "failed", "error"]:
                result_resp = requests.get(f"{BASE_URL}/result/{session_id}", timeout=5)
                result_data = result_resp.json()

                raw_content = result_data.get("content", "")
                if isinstance(raw_content, dict):
                    content = json.dumps(raw_content)
                else:
                    content = raw_content or ""

                if status == "failed":
                    error = result_data.get("failure_reason", "Unknown failure")
                elif not content:
                    error = "Empty content returned"
                break

            if i % 10 == 0 and i > 0:
                print(f"      [{i}s] Status: {status}")
        else:
            error = f"Timeout ({session_timeout}s)"

    except Exception as e:
        error = str(e)

    elapsed = time.time() - start_time

    print(f"\n[RESULT]")
    print(f"  Elapsed: {elapsed:.2f}s")

    if error:
        print(f"  ERROR: {error}")
    elif content:
        word_count = len(content.split())
        print(f"  SUCCESS: {word_count} words, {len(content)} chars")
        print(f"  Content preview: {content[:200]}...")
    else:
        print(f"  EMPTY RESPONSE (0 words)")

    return {
        "scenario": "API without thinking_budget_tokens",
        "content": content,
        "elapsed": elapsed,
        "error": error,
        "word_count": len(content.split()) if content else 0,
        "session_id": session_id
    }


def test_api_with_explicit_thinking_zero(session_timeout: int = 60) -> dict:
    """Test 3: API request WITH thinking_budget_tokens=0 to disable auto-apply."""
    print_header("TEST 3: API REQUEST WITH thinking_budget_tokens=0 (DISABLED)")

    print(f"\n[API] Sending request to {BASE_URL}/generate")
    print(f"      Model: {MODEL_ID}")
    print(f"      json_schema: YES")
    print(f"      thinking_budget_tokens: 0 (explicitly disabled)")

    payload = {
        "prompt": PROMPT,
        "generator_model": MODEL_ID,
        "temperature": 0.7,
        "max_tokens": 2000,
        "qa_layers": [],
        "max_iterations": 1,
        "gran_sabio_fallback": False,
        "json_output": True,
        "json_schema": JSON_SCHEMA,
        "verbose": True,
        "thinking_budget_tokens": 0,  # Explicitly disabled
    }

    start_time = time.time()
    error = None
    content = ""
    session_id = None

    try:
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        session_id = data.get("session_id")
        print(f"      Session ID: {session_id}")

        for i in range(session_timeout):
            time.sleep(1)
            status_resp = requests.get(f"{BASE_URL}/status/{session_id}", timeout=5)
            status_data = status_resp.json()
            status = status_data.get("status")

            if status in ["completed", "failed", "error"]:
                result_resp = requests.get(f"{BASE_URL}/result/{session_id}", timeout=5)
                result_data = result_resp.json()

                raw_content = result_data.get("content", "")
                if isinstance(raw_content, dict):
                    content = json.dumps(raw_content)
                else:
                    content = raw_content or ""

                if status == "failed":
                    error = result_data.get("failure_reason", "Unknown failure")
                elif not content:
                    error = "Empty content returned"
                break

            if i % 10 == 0 and i > 0:
                print(f"      [{i}s] Status: {status}")
        else:
            error = f"Timeout ({session_timeout}s)"

    except Exception as e:
        error = str(e)

    elapsed = time.time() - start_time

    print(f"\n[RESULT]")
    print(f"  Elapsed: {elapsed:.2f}s")

    if error:
        print(f"  ERROR: {error}")
    elif content:
        word_count = len(content.split())
        print(f"  SUCCESS: {word_count} words, {len(content)} chars")
        print(f"  Content preview: {content[:200]}...")
    else:
        print(f"  EMPTY RESPONSE (0 words)")

    return {
        "scenario": "API with thinking_budget_tokens=0",
        "content": content,
        "elapsed": elapsed,
        "error": error,
        "word_count": len(content.split()) if content else 0,
        "session_id": session_id
    }


def test_api_flexible_json_without_schema(session_timeout: int = 60) -> dict:
    """Test 4: API request with json_output but NO json_schema (flexible mode)."""
    print_header("TEST 4: API REQUEST WITH JSON MODE (NO SCHEMA) - AUTO THINKING")

    print(f"\n[API] Sending request to {BASE_URL}/generate")
    print(f"      Model: {MODEL_ID}")
    print(f"      json_output: YES")
    print(f"      json_schema: NO (flexible mode)")
    print(f"      thinking_budget_tokens: NOT SENT (will auto-apply)")

    payload = {
        "prompt": PROMPT,
        "generator_model": MODEL_ID,
        "temperature": 0.7,
        "max_tokens": 2000,
        "qa_layers": [],
        "max_iterations": 1,
        "gran_sabio_fallback": False,
        "json_output": True,
        # NO json_schema - flexible mode
        "verbose": True,
        # NOT sending thinking_budget_tokens
    }

    start_time = time.time()
    error = None
    content = ""
    session_id = None

    try:
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        session_id = data.get("session_id")
        print(f"      Session ID: {session_id}")

        for i in range(session_timeout):
            time.sleep(1)
            status_resp = requests.get(f"{BASE_URL}/status/{session_id}", timeout=5)
            status_data = status_resp.json()
            status = status_data.get("status")

            if status in ["completed", "failed", "error"]:
                result_resp = requests.get(f"{BASE_URL}/result/{session_id}", timeout=5)
                result_data = result_resp.json()

                raw_content = result_data.get("content", "")
                if isinstance(raw_content, dict):
                    content = json.dumps(raw_content)
                else:
                    content = raw_content or ""

                if status == "failed":
                    error = result_data.get("failure_reason", "Unknown failure")
                elif not content:
                    error = "Empty content returned"
                break

            if i % 10 == 0 and i > 0:
                print(f"      [{i}s] Status: {status}")
        else:
            error = f"Timeout ({session_timeout}s)"

    except Exception as e:
        error = str(e)

    elapsed = time.time() - start_time

    print(f"\n[RESULT]")
    print(f"  Elapsed: {elapsed:.2f}s")

    if error:
        print(f"  ERROR: {error}")
    elif content:
        word_count = len(content.split())
        print(f"  SUCCESS: {word_count} words, {len(content)} chars")
        print(f"  Content preview: {content[:200]}...")
    else:
        print(f"  EMPTY RESPONSE (0 words)")

    return {
        "scenario": "API with JSON mode (no schema) + auto thinking",
        "content": content,
        "elapsed": elapsed,
        "error": error,
        "word_count": len(content.split()) if content else 0,
        "session_id": session_id
    }


def main():
    print_header("GEMINI AUTO-THINKING + JSON SCHEMA COMPATIBILITY TEST")
    print(f"Model: {MODEL_ID}")
    print(f"Test started: {datetime.now().isoformat()}")

    results = {"test_date": datetime.now().isoformat(), "model": MODEL_ID}

    # Test 1: Config behavior
    config_result = test_config_behavior()
    results["config_analysis"] = config_result

    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"\nServer is running: {health.json()}")
    except:
        print(f"\nServer not running at {BASE_URL}. Skipping API tests.")
        print("\nTo run full tests, start the server and run this script again.")

        # Save partial results
        output_file = "dev_tests/gemini_auto_thinking_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nPartial results saved to: {output_file}")
        return

    api_results = []

    # Test 2: API without thinking_budget_tokens
    result2 = test_api_without_thinking()
    api_results.append(result2)

    # Test 3: API with thinking_budget_tokens=0
    result3 = test_api_with_explicit_thinking_zero()
    api_results.append(result3)

    # Test 4: JSON flexible mode (no schema) with auto thinking
    result4 = test_api_flexible_json_without_schema()
    api_results.append(result4)

    results["api_tests"] = api_results

    # Summary
    print_header("FINAL SUMMARY")

    print("\n[CONFIG ANALYSIS]")
    print(f"  Auto-applies thinking when not specified: {'YES' if config_result['auto_applies'] else 'NO'}")
    if config_result['auto_applies']:
        print(f"  Default thinking budget: {config_result['no_thinking_input']} tokens")

    print("\n[API TEST RESULTS]")
    print("| # | Scenario                                    | Words | Time   | Status |")
    print("|---|---------------------------------------------|-------|--------|--------|")

    for i, r in enumerate(api_results, 1):
        status = "[OK]" if r["word_count"] > 0 and not r["error"] else "[FAIL]"
        name = r["scenario"][:43]
        print(f"| {i} | {name:<43} | {r['word_count']:>5} | {r['elapsed']:>5.1f}s | {status} |")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print("-" * 70)

    test_no_thinking = api_results[0]  # Without thinking_budget_tokens
    test_zero_thinking = api_results[1]  # With thinking_budget_tokens=0
    test_flexible = api_results[2]  # JSON flexible mode

    if config_result['auto_applies']:
        print("\n[!] Config AUTO-APPLIES thinking budget for Gemini reasoning models")
        print(f"    Default value: {config_result['no_thinking_input']} tokens")

        if test_no_thinking["word_count"] == 0:
            print("\n[PROBLEM CONFIRMED] API without thinking_budget_tokens FAILS")
            print("    This confirms the JSON Schema + auto-thinking incompatibility")

            if test_zero_thinking["word_count"] > 0:
                print("\n[WORKAROUND] Setting thinking_budget_tokens=0 explicitly WORKS")

            if test_flexible["word_count"] > 0:
                print("\n[ALTERNATIVE] Using JSON flexible mode (no schema) + thinking WORKS")

            print("\n[RECOMMENDED FIX]")
            print("    Option A: When json_schema is provided for Gemini + thinking,")
            print("              use JSON flexible mode instead and let json_guard validate")
            print("    Option B: Allow users to disable auto-thinking with thinking_budget_tokens=0")
        else:
            print("\n[OK] API without thinking_budget_tokens WORKS")
            print("    The issue may have been fixed or is intermittent")
    else:
        print("\n[OK] Config does NOT auto-apply thinking budget")
        print("    The issue may have been fixed")

    # Save results
    output_file = "dev_tests/gemini_auto_thinking_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
