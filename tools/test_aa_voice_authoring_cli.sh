#!/bin/bash
# test_aa_voice_authoring_cli.sh — smoke test for aa-voice-authoring-cli.
#
# (a) --selftest-pcm16 exact-value check: catches a float->PCM16 conversion
#     regression (e.g. *32768/lrintf instead of *32767/truncating cast)
#     before any Metal inference runs.
# (b) 1-family batch run against the real authoring models: writes a
#     throwaway batch plan JSON from one PENDING row of the production
#     manifest (read-only — this script never writes back to the manifest,
#     the manifest write-back transaction is Task 3's job, not this CLI's),
#     runs the CLI, and asserts every expected artifact exists plus that
#     results.csv's header equals RESULT_COLUMNS parsed live from
#     generate_voice_variants.gd (never a hardcoded column list/count) with
#     reload_status=ok.
#
# Byte-identity between this CLI's output and the Godot generator's output
# at the same seed is Task 4's job (the A/B pilot), not this script's — this
# script only proves the CLI runs end to end and emits the right shape.
#
# All (b) output lands under research/generated/ab-pilot/smoke, which is
# gitignored (research/generated/ is a blanket-ignored scratch tree).

set -euo pipefail

REPO_ROOT="/Users/roberthyatt/Code/roleplaying-agents"
QWENTTS_DIR="$REPO_ROOT/godot/artificial-adventures/addons/ai_inference_extension/thirdparty/qwentts.cpp"
CLI="$QWENTTS_DIR/build/aa-voice-authoring-cli"
GD_GENERATOR="$REPO_ROOT/godot/artificial-adventures/tools/voice_catalog/generate_voice_variants.gd"
CLI_SRC="$QWENTTS_DIR/tools/aa_voice_authoring_cli.cpp"
MANIFEST="$REPO_ROOT/research/notes/2026-07-25-voice-generation-manifest.csv"
MODEL_ROOT="$REPO_ROOT/godot/artificial-adventures/models/qwen3-tts-gguf"
SMOKE_DIR="$REPO_ROOT/research/generated/ab-pilot/smoke"

if [ ! -x "$CLI" ]; then
    echo "FAIL: CLI binary not found or not executable: $CLI"
    echo "      (run 'make build-ai-inference' from godot/artificial-adventures first)"
    exit 1
fi

echo "=== (a) selftest-pcm16 ==="
EXPECTED_PCM16="-32767 -32767 -16383 0 10922 16383 32767 32767"
ACTUAL_PCM16="$("$CLI" --selftest-pcm16)"
if [ "$ACTUAL_PCM16" != "$EXPECTED_PCM16" ]; then
    echo "FAIL: selftest-pcm16 mismatch"
    echo "  expected: $EXPECTED_PCM16"
    echo "  actual:   $ACTUAL_PCM16"
    exit 1
fi
echo "PASS: selftest-pcm16 == $ACTUAL_PCM16"

echo
echo "=== (a2) LISTENING_SEED + LISTENING_LINES parity (CLI source vs .gd) ==="
# The CLI hardcodes LISTENING_SEED/LISTENING_LINES (it cannot link the Godot-typed
# generator). Assert those hardcoded values still match generate_voice_variants.gd
# so an edit to the .gd listening lines cannot silently break byte-identity.
set +e
python3 -u - "$GD_GENERATOR" "$CLI_SRC" <<'PYEOF'
import re
import sys

gd_path, cpp_path = sys.argv[1:3]
gd = open(gd_path).read()
cpp = open(cpp_path).read()

def seed(text):
    m = re.search(r"LISTENING_SEED\s*(?::=|=)\s*(\d+)", text)
    return m.group(1) if m else None

def lines_text(text, open_delim, close_delim):
    # Concatenate every double-quoted string inside the LISTENING_LINES block.
    # Element boundaries are irrelevant: GDScript stores each line as one string
    # while the C++ vector splits long lines across adjacent literals that the
    # compiler concatenates, so the joined text is the byte-identical comparand.
    start = text.index(open_delim) + len(open_delim)
    end = text.index(close_delim, start)
    return "".join(re.findall(r'"([^"]*)"', text[start:end]))

gd_seed, cpp_seed = seed(gd), seed(cpp)
gd_lines = lines_text(gd, "LISTENING_LINES := [", "]")
cpp_lines = lines_text(cpp, "LISTENING_LINES = {", "}")

fail = False
if gd_seed is None or gd_seed != cpp_seed:
    print("FAIL: LISTENING_SEED drift: gd=%r cli=%r" % (gd_seed, cpp_seed))
    fail = True
else:
    print("OK: LISTENING_SEED == %s" % gd_seed)
if not gd_lines or gd_lines != cpp_lines:
    print("FAIL: LISTENING_LINES text drift between .gd and CLI source")
    print("  gd  (%d chars): %r" % (len(gd_lines), gd_lines[:120]))
    print("  cli (%d chars): %r" % (len(cpp_lines), cpp_lines[:120]))
    fail = True
else:
    print("OK: LISTENING_LINES text matches (%d chars)" % len(gd_lines))
sys.exit(1 if fail else 0)
PYEOF
PARITY_RC=$?
set -e
if [ "$PARITY_RC" -ne 0 ]; then
    echo "FAIL: LISTENING parity mismatch — CLI hardcoded values diverged from generate_voice_variants.gd"
    exit 1
fi

echo
echo "=== (b) 1-family batch smoke ==="
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"
PLAN_JSON="$SMOKE_DIR/plan.json"
OUT_DIR="$SMOKE_DIR/out"
mkdir -p "$OUT_DIR"

python3 -u - "$MANIFEST" "$MODEL_ROOT" "$PLAN_JSON" "$OUT_DIR" <<'PYEOF'
import csv
import json
import sys

manifest_path, model_root, plan_path, out_dir = sys.argv[1:5]

row = None
with open(manifest_path, newline="") as f:
    for candidate in csv.DictReader(f):
        if candidate["generation_status"] == "pending":
            row = candidate
            break
if row is None:
    print("no pending manifest row found in " + manifest_path, file=sys.stderr)
    sys.exit(1)

plan = {
    "q4_base": model_root + "/qwen-talker-1.7b-base-Q4_K_M.gguf",
    "q8_voicedesign": model_root + "/qwen-talker-1.7b-voicedesign-Q8_0.gguf",
    "q4_tok": model_root + "/qwen-tokenizer-12hz-Q4_K_M.gguf",
    "q8_tok": model_root + "/qwen-tokenizer-12hz-Q8_0.gguf",
    "out_dir": out_dir,
    "families": [
        {
            "family_id": row["family_id"],
            "variants": [
                {
                    "variant_index": int(row["variant_index"]),
                    "seed": int(row["seed"]),
                    "maturity_archetype": row["maturity_archetype"],
                    "vocal_presentation_archetype": row["vocal_presentation_archetype"],
                    "register_archetype": row["register_archetype"],
                    "vocal_mechanism_archetype": row["vocal_mechanism_archetype"],
                    "temperament_voice_archetype": row["temperament_voice_archetype"],
                    "accent_archetype": row["accent_archetype"],
                    "voice_design_prompt": row["voice_design_prompt"],
                    "reference_text": row["reference_text"],
                    "reference_language": row["reference_language"],
                    "voice_design_model": row["voice_design_model"],
                    "voice_design_model_revision": row["voice_design_model_revision"],
                    "base_model": row["base_model"],
                    "base_model_revision": row["base_model_revision"],
                }
            ],
        }
    ],
}

with open(plan_path, "w") as f:
    json.dump(plan, f, indent=2)

print("plan written: family=%r variant=%s seed=%s" % (
    row["family_id"], row["variant_index"], row["seed"]))
PYEOF

echo "Running CLI (loads all 4 GGUF models -- may take a few minutes)..."
set +e
"$CLI" --plan "$PLAN_JSON"
CLI_RC=$?
set -e
if [ "$CLI_RC" -ne 0 ]; then
    echo "FAIL: CLI exited $CLI_RC"
    exit 1
fi

set +e
python3 -u - "$GD_GENERATOR" "$OUT_DIR" "$PLAN_JSON" <<'PYEOF'
import csv
import json
import os
import re
import sys

gd_path, out_dir, plan_path = sys.argv[1:4]

with open(gd_path) as f:
    gd_text = f.read()
match = re.search(r"RESULT_COLUMNS\s*:=\s*\[(.*?)\]", gd_text, re.DOTALL)
if not match:
    print("FAIL: could not find RESULT_COLUMNS in " + gd_path, file=sys.stderr)
    sys.exit(1)
expected_columns = re.findall(r'"([^"]+)"', match.group(1))
if not expected_columns:
    print("FAIL: RESULT_COLUMNS parsed empty", file=sys.stderr)
    sys.exit(1)
print("parsed RESULT_COLUMNS (%d): %s" % (len(expected_columns), expected_columns))

with open(plan_path) as f:
    plan = json.load(f)
family = plan["families"][0]
family_id = family["family_id"]
variant_index = family["variants"][0]["variant_index"]

fail = False

variant_dir = os.path.join(out_dir, family_id, str(variant_index))
required_files = [
    "reference.wav", "reference.spk", "reference.rvq", "reference.voice",
    "bundle.json", "line_1.wav", "line_2.wav", "line_3.wav", "line_4.wav",
]
for name in required_files:
    path = os.path.join(variant_dir, name)
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        print("FAIL: missing or empty artifact: " + path)
        fail = True
    else:
        print("OK: %s (%d bytes)" % (path, os.path.getsize(path)))

results_path = os.path.join(out_dir, family_id, "results.csv")
if not os.path.isfile(results_path):
    print("FAIL: missing " + results_path)
    sys.exit(1)
with open(results_path, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

if header != expected_columns:
    print("FAIL: results.csv header != parsed RESULT_COLUMNS")
    print("  header:   %s" % header)
    print("  expected: %s" % expected_columns)
    fail = True
else:
    print("OK: results.csv header == parsed RESULT_COLUMNS")

if len(rows) != 1:
    print("FAIL: expected 1 results row, got %d" % len(rows))
    fail = True
else:
    row = dict(zip(header, rows[0]))
    if row.get("reload_status") != "ok":
        print("FAIL: reload_status != ok (got %r)" % row.get("reload_status"))
        fail = True
    else:
        print("OK: reload_status == ok")
    if row.get("family_id") != family_id or int(row.get("variant_index", -1)) != variant_index:
        print("FAIL: results row family_id/variant_index mismatch")
        fail = True

sys.exit(1 if fail else 0)
PYEOF
PY_RC=$?
set -e

if [ "$PY_RC" -ne 0 ]; then
    echo "=== SMOKE TEST FAILED ==="
    exit 1
fi

echo
echo "=== SMOKE TEST PASSED ==="
