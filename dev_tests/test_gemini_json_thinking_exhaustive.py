#!/usr/bin/env python3
"""
Exhaustive test for Gemini JSON Schema + Thinking compatibility.

According to revisa-esto.txt, there's an issue where Gemini returns empty responses
when combining JSON Schema (response_schema) with thinking mode (thinking_config).

This test tries multiple scenarios to reproduce the issue:
1. Different prompt sizes (short, medium, long)
2. Different max_tokens values
3. Different schema complexities
4. Repeated runs to catch intermittent issues

Usage:
    python dev_tests/test_gemini_json_thinking_exhaustive.py
"""

import asyncio
import sys
import os
import io
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json_utils as json
from ai_service import AIService

# Configuration
MODEL_ID = "gemini-3-pro-preview"
TEST_RUNS = 3  # Repeat each test to catch intermittent issues

# Test prompts of different sizes
SHORT_PROMPT = "Generate 3 video titles about cooking."

MEDIUM_PROMPT = """Generate 5 creative YouTube video titles about Italian pasta recipes.
The titles should be SEO-optimized, engaging, and include relevant keywords.
Focus on authentic Italian cuisine and home cooking techniques."""

LONG_PROMPT = """Generate 10 creative YouTube video titles about cooking pasta.

## VIDEO CONTENT CONTEXT
The video is about a Spanish YouTuber who reads awkward stories from her followers.
She discusses embarrassing moments like:
- Being caught in embarrassing situations
- Public mishaps and social awkwardness
- Funny stories from social media
- Relatable life experiences

## STYLE REQUIREMENTS
- Titles should be in Spanish
- Between 50-70 characters
- SEO-optimized with relevant keywords
- Engaging but not clickbait
- Include numbers when appropriate

## CHANNEL INFORMATION
Channel: FocusingsVlogs
Content style: Comedy, storytime, relatable content
Target audience: Spanish-speaking teens and young adults

Generate exactly 10 titles now, each with a unique angle on the content."""

# Test schemas of different complexity
SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["titles"]
}

MEDIUM_SCHEMA = {
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

COMPLEX_SCHEMA = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "total_count": {"type": "integer"},
                "generated_at": {"type": "string"}
            },
            "required": ["total_count"]
        },
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "title": {"type": "string"},
                    "seo_score": {"type": "number"},
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["number", "title"]
            }
        }
    },
    "required": ["titles"]
}


class TestResult:
    def __init__(self, scenario: str, run: int):
        self.scenario = scenario
        self.run = run
        self.success = False
        self.content = ""
        self.word_count = 0
        self.elapsed = 0.0
        self.error = None


async def run_test(
    ai_service: AIService,
    scenario: str,
    prompt: str,
    json_schema: dict,
    max_tokens: int,
    run_number: int
) -> TestResult:
    """Run a single test scenario."""
    result = TestResult(scenario, run_number)

    start = datetime.now()
    try:
        content = await ai_service.generate_content(
            prompt=prompt,
            model=MODEL_ID,
            temperature=0.7,
            max_tokens=max_tokens,
            system_prompt="You are a helpful assistant that generates YouTube video titles.",
            json_output=True,
            json_schema=json_schema,
            # NOT passing thinking_budget_tokens - let it auto-apply
        )

        result.content = content or ""
        result.word_count = len(result.content.split()) if result.content else 0
        result.success = result.word_count > 0

    except Exception as e:
        result.error = str(e)

    result.elapsed = (datetime.now() - start).total_seconds()
    return result


async def run_all_tests():
    """Run all test scenarios."""
    print("=" * 70)
    print("  EXHAUSTIVE GEMINI JSON SCHEMA + THINKING TEST")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Runs per scenario: {TEST_RUNS}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    ai_service = AIService()

    if not ai_service.google_new_client:
        print("[ERROR] Google Gemini client not initialized")
        return

    all_results = []

    # Define test matrix
    test_matrix = [
        # (name, prompt, schema, max_tokens)
        ("Short+Simple+2k", SHORT_PROMPT, SIMPLE_SCHEMA, 2000),
        ("Short+Medium+2k", SHORT_PROMPT, MEDIUM_SCHEMA, 2000),
        ("Short+Complex+2k", SHORT_PROMPT, COMPLEX_SCHEMA, 2000),
        ("Medium+Simple+4k", MEDIUM_PROMPT, SIMPLE_SCHEMA, 4000),
        ("Medium+Medium+4k", MEDIUM_PROMPT, MEDIUM_SCHEMA, 4000),
        ("Medium+Complex+4k", MEDIUM_PROMPT, COMPLEX_SCHEMA, 4000),
        ("Long+Simple+8k", LONG_PROMPT, SIMPLE_SCHEMA, 8000),
        ("Long+Medium+8k", LONG_PROMPT, MEDIUM_SCHEMA, 8000),
        ("Long+Complex+8k", LONG_PROMPT, COMPLEX_SCHEMA, 8000),
        # High max_tokens scenarios (closer to original issue)
        ("Short+Medium+40k", SHORT_PROMPT, MEDIUM_SCHEMA, 40000),
        ("Long+Medium+40k", LONG_PROMPT, MEDIUM_SCHEMA, 40000),
    ]

    for name, prompt, schema, max_tokens in test_matrix:
        print(f"\n--- Testing: {name} ---")

        for run in range(1, TEST_RUNS + 1):
            result = await run_test(ai_service, name, prompt, schema, max_tokens, run)
            all_results.append(result)

            status = "[OK]" if result.success else "[FAIL]"
            if result.error:
                print(f"  Run {run}: {status} {result.elapsed:.1f}s - ERROR: {result.error}")
            else:
                print(f"  Run {run}: {status} {result.elapsed:.1f}s - {result.word_count} words")

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    # Group by scenario
    scenarios = {}
    for r in all_results:
        if r.scenario not in scenarios:
            scenarios[r.scenario] = []
        scenarios[r.scenario].append(r)

    print("\n| Scenario             | Run 1 | Run 2 | Run 3 | Pass Rate |")
    print("|----------------------|-------|-------|-------|-----------|")

    total_pass = 0
    total_fail = 0

    for scenario, results in scenarios.items():
        statuses = ["[OK]" if r.success else "[FAIL]" for r in results]
        pass_count = sum(1 for r in results if r.success)
        fail_count = len(results) - pass_count
        total_pass += pass_count
        total_fail += fail_count
        pass_rate = f"{pass_count}/{len(results)}"

        # Pad statuses if needed
        while len(statuses) < 3:
            statuses.append("-")

        print(f"| {scenario:<20} | {statuses[0]:^5} | {statuses[1]:^5} | {statuses[2]:^5} | {pass_rate:^9} |")

    print("\n" + "-" * 50)
    print(f"Total: {total_pass} passed, {total_fail} failed")

    # Identify failures
    failures = [r for r in all_results if not r.success]
    if failures:
        print("\n[!] FAILURES DETECTED:")
        for f in failures:
            print(f"  - {f.scenario} (run {f.run}): {f.error or 'Empty response'}")
    else:
        print("\n[OK] All tests passed - JSON Schema + Thinking combination works correctly")

    # Save results
    output = {
        "test_date": datetime.now().isoformat(),
        "model": MODEL_ID,
        "runs_per_scenario": TEST_RUNS,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "results": [
            {
                "scenario": r.scenario,
                "run": r.run,
                "success": r.success,
                "word_count": r.word_count,
                "elapsed": r.elapsed,
                "error": r.error,
                "content_preview": r.content[:100] if r.content else None
            }
            for r in all_results
        ]
    }

    output_file = "dev_tests/gemini_exhaustive_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())
