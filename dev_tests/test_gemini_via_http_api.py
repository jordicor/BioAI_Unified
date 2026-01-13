#!/usr/bin/env python3
"""
Test via HTTP API to reproduce the exact bug.

Uses the same HTTP endpoint as the external application.

Usage:
    python dev_tests/test_gemini_via_http_api.py
"""

import requests
import time
import sys
import io
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, '.')
import json_utils as json

BASE_URL = "http://localhost:8000"
MODEL_ID = "gemini-3-pro-preview"

LONG_PROMPT = """You are an expert YouTube title creator. Generate 10 unique video titles.

## VIDEO INFORMATION
Original Title: Las situaciones incomodas de mis chipirones ^^ (con @SitaKarstica).mp4

Video Content Summary:
================================================================================

... Bueno, que? Como estais? Yo no voy a mentir, la verdad es que no me encuentro muy bien, por eso voy tan abrigada. Antes de empezar el video, queria disculparme porque no va a poder estar todo lo bien currado que yo queria. Hoy, el dia que estoy grabando, es 7 de enero y manana dia 8 me voy a Madrid unos dias. Por lo tanto, tengo que grabar el video hoy y, como ya os he dicho, pues no voy a poder trabajarmelo tanto, lo voy a tener que hacer mas en plan expres. Asi que nada, ya en otro video os compensare. [musica animada] Hola, chipirones! Soy Focusin y el video de hoy trata sobre situaciones incomodas. Todos hemos pasado por ellas y seguiremos sufriendolas a lo largo de la vida, porque si, amigos, la vida misma esta repleta de situaciones incomodas. Algunas suceden por accidente y otras nos las buscamos nosotros mismos sin ser conscientes de ello. Son momentos en los que deseas que la tierra te trague, pero en lugar de eso permaneces alli, en medio del meollo. Y lo unico que puede pasar es que, aparte de la verguenza y la tension que estas sufriendo, tu cuerpo te la juegue con reacciones como pueden ser enrojecerse, sudar, tener la boca seca, temblar, tartamudear, etcetera, para poder sufrir de una forma un poco mas llevadera estas situaciones. Cuando era nina, lei algo que me sirvio de mucho en los libros de Manolito Gafotas. En uno de ellos, Manolito cuenta que cuando su madre le iba a echar la bronca y darle una colleja, siempre pensaba en esto: lo que me esta sucediendo ahora pasara, y cuando pase un tiempo, parecera que le ha pasado a otra persona. Y oye, funciona! Cuando pasa un tiempo desde que vives aquella situacion embarazosa, la ves como lejana, como ajena a ti. Y es por eso que al cabo de un tiempo te ves capacitado para contarla y para reirte de ella.

## STYLE INSTRUCTIONS
Generate SEO-optimized YouTube video titles.
The titles should be:
- Include relevant keywords naturally
- Front-load important keywords
- Between 50-70 characters to maximize visibility

## OUTPUT REQUIREMENTS
- Generate exactly 10 titles
- All titles must be in Spanish
- Respond with a JSON object containing a "titles" array

Generate the titles now:"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "titles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "Sequential number of the title (1, 2, 3...)"
                    },
                    "title": {
                        "type": "string",
                        "description": "The generated video title text"
                    }
                },
                "required": ["number", "title"],
                "additionalProperties": False
            }
        }
    },
    "required": ["titles"],
    "additionalProperties": False
}


def test_via_api(name: str, payload: dict, timeout: int = 120) -> dict:
    """Test via HTTP API."""
    print(f"\n--- {name} ---")
    print(f"    Model: {payload.get('generator_model')}")
    print(f"    json_schema: {'YES' if payload.get('json_schema') else 'NO'}")
    print(f"    thinking_budget_tokens: {payload.get('thinking_budget_tokens', 'NOT IN PAYLOAD')}")
    print(f"    gran_sabio_fallback: {payload.get('gran_sabio_fallback', True)}")

    start = time.time()
    error = None
    content = ""
    session_id = None
    generator_output_empty = False

    try:
        # Start generation
        resp = requests.post(f"{BASE_URL}/generate", json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        session_id = data.get("session_id")
        print(f"    Session ID: {session_id}")

        # Poll for result
        for i in range(timeout):
            time.sleep(1)
            status_resp = requests.get(f"{BASE_URL}/status/{session_id}", timeout=5)
            status_data = status_resp.json()
            status = status_data.get("status")
            phase = status_data.get("current_phase", "")

            if i % 15 == 0 and i > 0:
                print(f"    [{i}s] {status} - {phase}")

            if status in ["completed", "failed", "error"]:
                result_resp = requests.get(f"{BASE_URL}/result/{session_id}", timeout=5)
                result_data = result_resp.json()

                # Check if Gran Sabio was invoked (indicates generator failed)
                final_iteration = result_data.get("final_iteration")
                if final_iteration == "Gran Sabio":
                    print(f"    [!] Gran Sabio was invoked - generator likely returned empty")
                    generator_output_empty = True

                raw_content = result_data.get("content", "")
                if isinstance(raw_content, dict):
                    content = json.dumps(raw_content)
                else:
                    content = raw_content or ""

                if status == "failed":
                    error = result_data.get("failure_reason", "Unknown")
                elif not content:
                    error = "Empty final content"
                break
        else:
            error = f"Timeout ({timeout}s)"

    except Exception as e:
        error = str(e)

    elapsed = time.time() - start
    word_count = len(content.split()) if content else 0

    print(f"    Elapsed: {elapsed:.1f}s")
    if error:
        print(f"    [FAIL] {error}")
    elif generator_output_empty:
        print(f"    [BUG] Generator returned empty, Gran Sabio saved it ({word_count} words)")
    else:
        print(f"    [OK] {word_count} words")
        print(f"    Preview: {content[:150]}...")

    return {
        "success": word_count > 0 and not generator_output_empty,
        "word_count": word_count,
        "elapsed": elapsed,
        "error": error,
        "generator_empty": generator_output_empty,
        "session_id": session_id
    }


def main():
    print("=" * 70)
    print("  TEST VIA HTTP API - EXACT REPRODUCTION")
    print("=" * 70)
    print(f"URL: {BASE_URL}")
    print(f"Started: {datetime.now().isoformat()}")

    # Check server
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server: {health.json()}")
    except:
        print(f"[ERROR] Server not running at {BASE_URL}")
        return

    results = {}

    # Test 1: EXACT failing params with gran_sabio_fallback DISABLED
    payload1 = {
        "prompt": LONG_PROMPT,
        "generator_model": MODEL_ID,
        "temperature": 0.7,
        "max_tokens": 8000,
        "json_output": True,
        "json_schema": JSON_SCHEMA,
        "qa_layers": [],
        "max_iterations": 1,
        "gran_sabio_fallback": False,  # Disable so we see the real error
        "verbose": True,
        # NO thinking_budget_tokens - will auto-apply
    }
    results["no_fallback"] = test_via_api("Test 1: Without Gran Sabio fallback", payload1)

    # Test 2: Same but WITH gran_sabio_fallback (like the real case)
    payload2 = {
        "prompt": LONG_PROMPT,
        "generator_model": MODEL_ID,
        "temperature": 0.7,
        "max_tokens": 8000,
        "json_output": True,
        "json_schema": JSON_SCHEMA,
        "qa_layers": [],
        "max_iterations": 1,
        "gran_sabio_fallback": True,  # Enabled like real case
        "verbose": True,
        # NO thinking_budget_tokens - will auto-apply
    }
    results["with_fallback"] = test_via_api("Test 2: With Gran Sabio fallback", payload2)

    # Test 3: Without json_schema (flexible mode)
    payload3 = {
        "prompt": LONG_PROMPT,
        "generator_model": MODEL_ID,
        "temperature": 0.7,
        "max_tokens": 8000,
        "json_output": True,
        # NO json_schema
        "qa_layers": [],
        "max_iterations": 1,
        "gran_sabio_fallback": False,
        "verbose": True,
        # NO thinking_budget_tokens - will auto-apply
    }
    results["flexible"] = test_via_api("Test 3: JSON flexible (no schema)", payload3)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    bug_found = False
    for name, r in results.items():
        if r.get("generator_empty"):
            status = "[BUG]"
            bug_found = True
        elif r.get("success"):
            status = "[OK]"
        else:
            status = "[FAIL]"
            bug_found = True
        print(f"  {name}: {status} - {r.get('word_count', 0)} words")

    print("\n" + "-" * 50)
    if bug_found:
        print("[BUG CONFIRMED] Generator returns empty with json_schema + auto-thinking")
        print("\n[PROPOSED FIX]")
        print("  In ai_service.py _generate_gemini_new_sdk():")
        print("  When thinking_config is active, skip response_schema")
        print("  and use only response_mime_type='application/json'")
    else:
        print("[OK] Bug not reproduced in this run")

    # Save
    output = {"test_date": datetime.now().isoformat(), "results": results}
    with open("dev_tests/gemini_http_api_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to: dev_tests/gemini_http_api_results.json")


if __name__ == "__main__":
    main()
