#!/usr/bin/env python3
"""
Test Gemini with parameters closer to the original failing request.

Original parameters from error.txt:
- thinking_budget_tokens: 32768
- max_tokens: 40768
- Long prompt with video transcription

Usage:
    python dev_tests/test_gemini_json_thinking_original_params.py
"""

import requests
import time
import json
import sys
import io
from datetime import datetime

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Configuration - matching original parameters
BASE_URL = "http://192.168.50.209:8000"
MODEL_ID = "gemini-3-pro-preview"
THINKING_BUDGET = 32768  # Original value
MAX_TOKENS = 40768  # Original value

# Long prompt similar to original (with video transcription)
LONG_PROMPT = """You are an expert YouTube title creator. Generate 10 unique video titles.

## VIDEO INFORMATION
Original Title: Las situaciones incomodas de mis chipirones ^^ (con @SitaKarstica).mp4

Video Content Summary:
================================================================================

... Bueno, ¿qué? ¿Cómo estáis? Yo no voy a mentir, la verdad es que no me encuentro muy bien, por eso voy tan abrigada. Antes de empezar el vídeo, quería disculparme porque no va a poder estar todo lo bien currado que yo quería. Hoy, el día que estoy grabando, es 7 de enero y mañana día 8 me voy a Madrid unos días. Por lo tanto, tengo que grabar el vídeo hoy y, como ya os he dicho, pues no voy a poder trabajármelo tanto, lo voy a tener que hacer más en plan exprés. Así que nada, ya en otro vídeo os compensaré. [música animada] ¡Hola, chipirones! Soy Focusin y el vídeo de hoy trata sobre situaciones incómodas. Todos hemos pasado por ellas y seguiremos sufriéndolas a lo largo de la vida, porque sí, amigos, la vida misma está repleta de situaciones incómodas. Algunas suceden por accidente y otras nos las buscamos nosotros mismos sin ser conscientes de ello. Son momentos en los que deseas que la tierra te trague, pero en lugar de eso permaneces allí, en medio del meollo. Y lo único que puede pasar es que, aparte de la vergüenza y la tensión que estás sufriendo, tu cuerpo te la juegue con reacciones como pueden ser enrojecerse, sudar, tener la boca seca, temblar, tartamudear, etcétera, para poder sufrir de una forma un poco más llevadera estas situaciones. Cuando era niña, leí algo que me sirvió de mucho en los libros de Manolito Gafotas. En uno de ellos, Manolito cuenta que cuando su madre le iba a echar la bronca y darle una colleja, siempre pensaba en esto: lo que me está sucediendo ahora pasará, y cuando pase un tiempo, parecerá que le ha pasado a otra persona. Y oye, ¡funciona! Cuando pasa un tiempo desde que vives aquella situación embarazosa, la ves como lejana, como ajena a ti. Y es por eso que al cabo de un tiempo te ves capacitado para contarla y para reírte de ella. Y por eso os voy a leer algunas situaciones incómodas que me habéis pasado tanto por Twitter como por mi página de Facebook. Como ya os he dicho, no voy a tener nada de tiempo para poder representarlas, pero hoy me acompaña mi amiga y compañera de piso, Sita Cártica, y así la comentaremos entre las dos y, bueno, se hará muchísimo más ameno que si solo las comento yo. A ver, vamos a leer primero las situaciones incómodas que nos han pasado por Twitter. Panda Cornea dice: Ir por la calle y cambiar de acera porque viene un tío chungo. [risas] Eso, eso yo lo he hecho alguna vez también, pero con un poquito de disimulo. Es como: «Ay, que no se note, que le tengo miedo, ¿sabes?». ¿A ti te ha pasado alguna vez? Sí, me agacho y salgo corriendo entre los coches y que no me vean. [risas] A ver, Mushroom Man, léelo tú. Ir por la calle equivocándote de dirección, [risas] sacar el móvil para simular algo y cambiar de sentido. Eso sí que lo he hecho yo muchas veces. Vas por la calle y dices: «Uy, mierda, que es por el otro lado». Y ahora, ¿cómo me doy la vuelta para no quedar como el tonto? Y es como: «Ah, sí, bueno...» ¿En qué estáis pensando? Videópatas, que se te escape un pedo en medio de la clase. [risas] [sonido de pedo] Mel... El gato. ¡Ya! The Big Vlogs. No, The no, De. De, no, De Big Vlogs. Me mata. Mi madre me pilló de semidesnudo viendo porno. Fue extremadamente incómodo. ¿Hola? [risas] Mister Prea: Ver una peli con tu familia y que haya una escena subida de tono. Yo siempre miro la cara de mi abuelo. [risas] Sí, pero la verdad es que eso es una situación bastante incómoda. Por eso me vine a vivir con ella, [risas] para no tener que, para no tener que aguantar la mirada de mis padres. Pues, pues sí. [risas] Unbroken. Un, Unbroken. Una de las mayores que he tenido, situaciones incómodas que he tenido, fue que me bajó la regla en clase. ¡Eh! Cuando ya empieza la adolescencia, tienes que ir pensando que te puede pasar, entonces tienes que llevarte cosas, Sabadill, compresa, etcétera. Cuando a mí me vino la regla, fue en casa de mi tía, que era el cumpleaños de mi primo. Me fui al baño, vi que me había venido la regla, llamé a mi prima, le dije: «Ay, que me ha venido la regla, no se lo cuentes a nadie». Y me dijo: «No, no, tranquila, si yo no se lo voy a contar a nadie». Luego se fue al comedor y dijo: «¡A la Mel le ha venido la regla!» Delante de mi primo, de sus amigos, de mis tíos, de... ¡Fiesta, que le ha venido la regla y hay que celebrarlo! Luego, todo el mundo felicitándome: «¡Ay, felicidades!». ¿Y yo qué? ¿Cómo que felicidades? No. Qué cosas más bonitas.

## STYLE INSTRUCTIONS
Generate SEO-optimized YouTube video titles.
The titles should be:
- Include relevant keywords naturally
- Front-load important keywords
- Use searchable phrases people actually search for
- Include numbers when relevant (e.g., "5 Tips", "2024 Guide")
- Between 50-70 characters to maximize visibility
- Avoid clickbait but still be compelling

## ADDITIONAL INSTRUCTIONS FROM USER
añade " - FocusingsVlogs" al final, que uno de los títulos sea "Las situaciones incomodas de mis chipirones ^^ (con @SitaKarstica) - FocusignsVlogs" (tal cual, es que quiero tener ese en la lista).

## OUTPUT REQUIREMENTS
- Generate exactly 10 titles
- All titles must be in Spanish
- Respond with a JSON object containing a "titles" array
- Each item in the array must have "number" (1, 2, 3...) and "title" (the text)

Example format:
{
  "titles": [
    {"number": 1, "title": "Your first title here"},
    {"number": 2, "title": "Your second title here"}
  ]
}

Generate the titles now:"""

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


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_scenario(
    scenario_name: str,
    json_output: bool,
    json_schema: dict = None,
    thinking_budget: int = 0,
    max_tokens: int = 2000
) -> dict:
    """Test a specific scenario via API."""

    print(f"\n🔄 Testing: {scenario_name}")
    print(f"   json_output={json_output}, json_schema={'YES' if json_schema else 'NO'}")
    print(f"   thinking_budget={thinking_budget}, max_tokens={max_tokens}")
    print(f"   Prompt length: {len(LONG_PROMPT)} chars")

    payload = {
        "prompt": LONG_PROMPT,
        "generator_model": MODEL_ID,
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "qa_layers": [],
        "max_iterations": 1,
        "gran_sabio_fallback": False,
        "json_output": json_output,
        "verbose": True,
    }

    if json_schema:
        payload["json_schema"] = json_schema

    if thinking_budget > 0:
        payload["thinking_budget_tokens"] = thinking_budget

    start_time = time.time()
    error = None
    content = ""
    session_id = None

    try:
        response = requests.post(f"{BASE_URL}/generate", json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        session_id = data.get("session_id")
        print(f"   Session ID: {session_id}")

        # Poll for result (max 120 seconds for longer requests)
        for i in range(120):
            time.sleep(1)
            status_resp = requests.get(f"{BASE_URL}/status/{session_id}", timeout=5)
            status_data = status_resp.json()
            status = status_data.get("status")

            if status in ["completed", "failed", "error"]:
                result_resp = requests.get(f"{BASE_URL}/result/{session_id}", timeout=5)
                result_data = result_resp.json()

                raw_content = result_data.get("content", "")
                if isinstance(raw_content, dict):
                    content = json.dumps(raw_content, ensure_ascii=False)
                else:
                    content = raw_content or ""

                if status == "failed":
                    error = result_data.get("failure_reason", "Unknown failure")
                elif not content:
                    error = "Empty content returned"
                break

            if i % 10 == 0:
                current_phase = status_data.get("current_phase", "")
                print(f"   [{i}s] Status: {status} ({current_phase})")

        else:
            error = "Timeout waiting for result (120s)"

    except requests.exceptions.RequestException as e:
        error = f"HTTP error: {e}"
    except Exception as e:
        error = f"Error: {e}"

    elapsed = time.time() - start_time

    print(f"\n📋 Scenario: {scenario_name}")
    print(f"⏱️  Elapsed: {elapsed:.2f}s")

    if error:
        print(f"❌ ERROR: {error}")
    elif content:
        word_count = len(content.split())
        print(f"✅ SUCCESS: {word_count} words, {len(content)} chars")
        print(f"📄 Content preview (first 600 chars):")
        print("-" * 40)
        print(content[:600] + ("..." if len(content) > 600 else ""))
        print("-" * 40)

        try:
            clean = content.strip()
            if clean.startswith("```"):
                lines = clean.split("\n")
                clean = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            parsed = json.loads(clean)
            print(f"✅ Valid JSON with {len(parsed.get('titles', []))} titles")
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parse error: {e}")
    else:
        print(f"⚠️  EMPTY RESPONSE (0 words)")

    return {
        "scenario": scenario_name,
        "content": content,
        "elapsed": elapsed,
        "error": error,
        "word_count": len(content.split()) if content else 0,
        "json_schema": json_schema is not None,
        "thinking_budget": thinking_budget,
        "max_tokens": max_tokens,
    }


def main():
    print_separator("GEMINI TEST WITH ORIGINAL PARAMETERS")
    print(f"Model: {MODEL_ID}")
    print(f"Original Thinking Budget: {THINKING_BUDGET} tokens")
    print(f"Original Max Tokens: {MAX_TOKENS}")
    print(f"API URL: {BASE_URL}")
    print(f"Test started: {datetime.now().isoformat()}")

    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"✅ Server is running: {health.json()}")
    except:
        print(f"❌ Server not running at {BASE_URL}")
        return

    results = []

    # Test 1: Original params (JSON Schema + Thinking + High tokens)
    print_separator("TEST 1: ORIGINAL PARAMS - JSON Schema + Thinking (32768) + Max tokens (40768)")
    result = test_scenario(
        "ORIGINAL: JSON Schema + Thinking 32768 + MaxTokens 40768",
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget=THINKING_BUDGET,
        max_tokens=MAX_TOKENS
    )
    results.append(result)

    # Test 2: Same but without thinking
    print_separator("TEST 2: NO THINKING - JSON Schema + No Thinking + Max tokens (40768)")
    result = test_scenario(
        "NO THINKING: JSON Schema + MaxTokens 40768",
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget=0,
        max_tokens=MAX_TOKENS
    )
    results.append(result)

    # Test 3: Same but without JSON Schema (flexible mode)
    print_separator("TEST 3: NO SCHEMA - JSON mode + Thinking (32768)")
    result = test_scenario(
        "NO SCHEMA: JSON mode + Thinking 32768",
        json_output=True,
        json_schema=None,
        thinking_budget=THINKING_BUDGET,
        max_tokens=MAX_TOKENS
    )
    results.append(result)

    # Summary
    print_separator("TEST SUMMARY")
    print("\n| # | Scenario                                    | Schema | Think  | Words | Time   | Status |")
    print("|---|---------------------------------------------|--------|--------|-------|--------|--------|")

    for i, r in enumerate(results, 1):
        schema_str = "YES" if r["json_schema"] else "NO"
        thinking_str = f"{r['thinking_budget']}" if r["thinking_budget"] > 0 else "NO"
        status = "✅ OK" if r["word_count"] > 0 and not r["error"] else "❌ FAIL"
        name = r["scenario"][:41]
        print(f"| {i} | {name:<41} | {schema_str:^6} | {thinking_str:>6} | {r['word_count']:>5} | {r['elapsed']:>5.1f}s | {status} |")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print("-" * 70)

    original_result = results[0]
    if original_result["word_count"] == 0 or original_result["error"]:
        print("\n🎯 CONFIRMED: The ORIGINAL parameters cause failure!")
        print(f"   Error: {original_result['error']}")

        no_thinking = results[1]
        no_schema = results[2]

        if no_thinking["word_count"] > 0 and not no_thinking["error"]:
            print("\n   ✅ Without thinking: WORKS")
            print("   → Problem is thinking_budget with JSON Schema")
        if no_schema["word_count"] > 0 and not no_schema["error"]:
            print("\n   ✅ Without JSON Schema (flexible mode): WORKS")
            print("   → Problem is specifically response_schema + thinking_config")

        print("\n💡 SOLUTION: When thinking is enabled, use JSON mode (flexible)")
        print("   and let json_guard validate the output.")
    else:
        print("\n✅ Original parameters worked! Issue may be intermittent.")

    # Save results
    output_file = "dev_tests/gemini_original_params_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_date": datetime.now().isoformat(),
            "model": MODEL_ID,
            "thinking_budget": THINKING_BUDGET,
            "max_tokens": MAX_TOKENS,
            "prompt_length": len(LONG_PROMPT),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
