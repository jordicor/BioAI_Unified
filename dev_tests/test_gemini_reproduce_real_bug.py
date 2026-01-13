#!/usr/bin/env python3
"""
Test to reproduce the REAL bug from session 82fac4d0-1a83-4cba-b444-91143f3eb04d.

The exact parameters that cause empty response:
- gemini-3-pro-preview
- json_output: true
- json_schema with additionalProperties: false
- thinking_budget_tokens: null (auto-applies 32768)
- Long prompt (10K+ chars)

Usage:
    python dev_tests/test_gemini_reproduce_real_bug.py
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

# The EXACT long prompt from the failing session
LONG_PROMPT = """You are an expert YouTube title creator. Generate 10 unique video titles.

## VIDEO INFORMATION
Original Title: Las situaciones incomodas de mis chipirones ^^ (con @SitaKarstica).mp4

Video Content Summary:
================================================================================

... Bueno, que? Como estais? Yo no voy a mentir, la verdad es que no me encuentro muy bien, por eso voy tan abrigada. Antes de empezar el video, queria disculparme porque no va a poder estar todo lo bien currado que yo queria. Hoy, el dia que estoy grabando, es 7 de enero y manana dia 8 me voy a Madrid unos dias. Por lo tanto, tengo que grabar el video hoy y, como ya os he dicho, pues no voy a poder trabajarmelo tanto, lo voy a tener que hacer mas en plan expres. Asi que nada, ya en otro video os compensare. [musica animada] Hola, chipirones! Soy Focusin y el video de hoy trata sobre situaciones incomodas. Todos hemos pasado por ellas y seguiremos sufriendolas a lo largo de la vida, porque si, amigos, la vida misma esta repleta de situaciones incomodas. Algunas suceden por accidente y otras nos las buscamos nosotros mismos sin ser conscientes de ello. Son momentos en los que deseas que la tierra te trague, pero en lugar de eso permaneces alli, en medio del meollo. Y lo unico que puede pasar es que, aparte de la verguenza y la tension que estas sufriendo, tu cuerpo te la juegue con reacciones como pueden ser enrojecerse, sudar, tener la boca seca, temblar, tartamudear, etcetera, para poder sufrir de una forma un poco mas llevadera estas situaciones. Cuando era nina, lei algo que me sirvio de mucho en los libros de Manolito Gafotas. En uno de ellos, Manolito cuenta que cuando su madre le iba a echar la bronca y darle una colleja, siempre pensaba en esto: lo que me esta sucediendo ahora pasara, y cuando pase un tiempo, parecera que le ha pasado a otra persona. Y oye, funciona! Cuando pasa un tiempo desde que vives aquella situacion embarazosa, la ves como lejana, como ajena a ti. Y es por eso que al cabo de un tiempo te ves capacitado para contarla y para reirte de ella. Y por eso os voy a leer algunas situaciones incomodas que me habeis pasado tanto por Twitter como por mi pagina de Facebook. Como ya os he dicho, no voy a tener nada de tiempo para poder representarlas, pero hoy me acompana mi amiga y companera de piso, Sita Cartica, y asi la comentaremos entre las dos y, bueno, se hara muchisimo mas ameno que si solo las comento yo. A ver, vamos a leer primero las situaciones incomodas que nos han pasado por Twitter. Panda Cornea dice: Ir por la calle y cambiar de acera porque viene un tio chungo. [risas] Eso, eso yo lo he hecho alguna vez tambien, pero con un poquito de disimulo. Es como: Ay, que no se note, que le tengo miedo, sabes?. A ti te ha pasado alguna vez? Si, me agacho y salgo corriendo entre los coches y que no me vean. [risas] A ver, Mushroom Man, leelo tu. Ir por la calle equivocandote de direccion, [risas] sacar el movil para simular algo y cambiar de sentido. Eso si que lo he hecho yo muchas veces. Vas por la calle y dices: Uy, mierda, que es por el otro lado. Y ahora, como me doy la vuelta para no quedar como el tonto? Y es como: Ah, si, bueno... En que estais pensando? Videopatas, que se te escape un pedo en medio de la clase. [risas] [sonido de pedo] Mel... El gato. Ya! The Big Vlogs. No, The no, De. De, no, De Big Vlogs. Me mata. Mi madre me pillo de semidesnudo viendo porno. Fue extremadamente incomodo. Hola? [risas] Mister Prea: Ver una peli con tu familia y que haya una escena subida de tono. Yo siempre miro la cara de mi abuelo. [risas] Si, pero la verdad es que eso es una situacion bastante incomoda. Por eso me vine a vivir con ella, [risas] para no tener que, para no tener que aguantar la mirada de mis padres. Pues, pues si. [risas] Unbroken. Un, Unbroken. Una de las mayores que he tenido, situaciones incomodas que he tenido, fue que me bajo la regla en clase. Eh! Cuando ya empieza la adolescencia, tienes que ir pensando que te puede pasar, entonces tienes que llevarte cosas, Sabadill, compresa, etcetera. Cuando a mi me vino la regla, fue en casa de mi tia, que era el cumpleanos de mi primo. Me fui al bano, vi que me habia venido la regla, llame a mi prima, le dije: Ay, que me ha venido la regla, no se lo cuentes a nadie. Y me dijo: No, no, tranquila, si yo no se lo voy a contar a nadie. Luego se fue al comedor y dijo: A la Mel le ha venido la regla! Delante de mi primo, de sus amigos, de mis tios, de... Fiesta, que le ha venido la regla y hay que celebrarlo! Luego, todo el mundo felicitandome: Ay, felicidades!. Y yo que? Como que felicidades? No. Que cosas mas bonitas.

## STYLE INSTRUCTIONS
Generate SEO-optimized YouTube video titles.
The titles should be:
- Include relevant keywords naturally
- Front-load important keywords
- Use searchable phrases people actually search for
- Include numbers when relevant (e.g., "5 Tips", "2024 Guide")
- Between 50-70 characters to maximize visibility
- Avoid clickbait but still be compelling

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

# The EXACT json_schema from the failing session
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
                "additionalProperties": False  # <-- KEY: This is in the real schema
            }
        }
    },
    "required": ["titles"],
    "additionalProperties": False  # <-- KEY: This is in the real schema
}


async def run_test(name: str, ai_service: AIService, **kwargs):
    """Run a single test."""
    print(f"\n--- {name} ---")
    print(f"    Prompt length: {len(LONG_PROMPT)} chars")
    for k, v in kwargs.items():
        if k == 'json_schema':
            print(f"    {k}: {'YES with additionalProperties:false' if v else 'NO'}")
        elif k == 'thinking_budget_tokens':
            print(f"    {k}: {v if v is not None else 'NOT SENT (auto-applies 32768)'}")
        else:
            print(f"    {k}: {v}")

    start = datetime.now()
    try:
        content = await ai_service.generate_content(
            prompt=LONG_PROMPT,
            model=MODEL_ID,
            temperature=0.7,
            max_tokens=8000,
            **kwargs
        )
        elapsed = (datetime.now() - start).total_seconds()
        word_count = len(content.split()) if content else 0

        if word_count == 0:
            print(f"    [FAIL] EMPTY RESPONSE in {elapsed:.1f}s")
            print(f"    Content repr: {repr(content)}")
        else:
            print(f"    [OK] {word_count} words in {elapsed:.1f}s")
            print(f"    Preview: {content[:200]}...")

        return {"success": word_count > 0, "word_count": word_count, "content": content, "elapsed": elapsed}

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        print(f"    [ERROR] {e}")
        return {"success": False, "error": str(e), "elapsed": elapsed}


async def main():
    print("=" * 70)
    print("  REPRODUCE REAL BUG FROM SESSION 82fac4d0-1a83-4cba-b444-91143f3eb04d")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Started: {datetime.now().isoformat()}")

    ai_service = AIService()

    if not ai_service.google_new_client:
        print("\n[ERROR] Google Gemini client not initialized")
        return

    results = {}

    # Test 1: EXACT params from failing session (should fail)
    results["exact_params"] = await run_test(
        "Test 1: EXACT failing params (json_schema + NO thinking_budget_tokens)",
        ai_service,
        json_output=True,
        json_schema=JSON_SCHEMA,
        # NOT sending thinking_budget_tokens - will auto-apply 32768
    )

    # Test 2: Same but with explicit thinking disabled
    results["no_thinking"] = await run_test(
        "Test 2: Same but WITHOUT thinking (thinking_budget_tokens=1)",
        ai_service,
        json_output=True,
        json_schema=JSON_SCHEMA,
        thinking_budget_tokens=1,  # Minimal thinking
    )

    # Test 3: Same but without json_schema (flexible mode)
    results["no_schema"] = await run_test(
        "Test 3: JSON flexible mode (no schema) + auto thinking",
        ai_service,
        json_output=True,
        # No json_schema
        # No thinking_budget_tokens - will auto-apply
    )

    # Test 4: Schema without additionalProperties:false
    simple_schema = {
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
                    # NO additionalProperties
                }
            }
        },
        "required": ["titles"]
        # NO additionalProperties
    }
    results["simple_schema"] = await run_test(
        "Test 4: Schema WITHOUT additionalProperties:false + auto thinking",
        ai_service,
        json_output=True,
        json_schema=simple_schema,
        # No thinking_budget_tokens - will auto-apply
    )

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_passed = True
    for name, r in results.items():
        status = "[OK]" if r.get("success") else "[FAIL]"
        if not r.get("success"):
            all_passed = False
        words = r.get("word_count", 0)
        error = r.get("error", "")
        print(f"  {name}: {status} - {words} words {error}")

    print("\n" + "-" * 50)
    if not all_passed:
        print("[BUG REPRODUCED] Some tests failed - the issue exists")
        if not results["exact_params"].get("success") and results["no_schema"].get("success"):
            print("    -> JSON Schema + thinking causes empty response")
            print("    -> JSON flexible mode + thinking WORKS")
            print("\n[RECOMMENDED FIX]")
            print("    When thinking is enabled for Gemini, use JSON flexible mode")
            print("    instead of response_schema, and let json_guard validate.")
    else:
        print("[OK] All tests passed - bug not reproduced in this run")

    # Save results
    output = {
        "test_date": datetime.now().isoformat(),
        "model": MODEL_ID,
        "prompt_length": len(LONG_PROMPT),
        "results": {k: {**v, "content": v.get("content", "")[:500]} for k, v in results.items()}
    }
    output_file = "dev_tests/gemini_real_bug_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
