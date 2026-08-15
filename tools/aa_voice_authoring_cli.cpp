// aa_voice_authoring_cli.cpp: Godot-free batch voice-authoring binary.
//
// Loads the qwentts.cpp authoring models ONCE per process and generates
// `.voice` containers + listening artifacts for a batch of voice families,
// reproducing exactly what tools/voice_catalog/generate_voice_variants.gd
// (the Godot generator this CLI replaces) does per family/variant, so a
// later A/B pilot can prove byte-identical output at the same seed.
//
// Byte-identity-by-construction: this binary links the SAME two Godot-free
// units the shipped extension/generator use underneath, unmodified, straight
// out of addons/ai_inference_extension/src/:
//   - qwen3_tts_qwentts_backend.{h,cpp}  (the Backend adapter over qwentts.cpp;
//     already documented as "self-contained; no dependency on Godot")
//   - voice_container_core.{h,cpp}       (the Task-1 std `.voice` pack/inspect/
//     decode core the Godot adapter also calls through)
// No logic from either file is re-derived or copied here — only called.
// The ONE byte-sensitive piece this file DOES re-derive (because it lives
// inside Godot-typed code we cannot link, godot/.../src/qwen3_tts.cpp) is the
// float -> PCM16 conversion + 44-byte WAV header, replicated EXACTLY from:
//   - qwen3_tts.cpp:1044-1049 (float -> PCM16 LE: clamp +-1.0f, *32767.0f,
//     truncating static_cast<int16_t>, LE byte pack)
//   - qwen3_tts.cpp:1867-1892 (s_pcm_to_wav: 44-byte RIFF/WAVE/fmt/data header)
// See selftest_pcm16() below for a standalone check of the first piece.
//
// ============================================================================
// Batch plan-JSON input contract (what a driver — e.g. Task 3's Python
// wiring — writes and this CLI reads via --plan <path>). Field names are
// grounded on generate_voice_variants.gd's PLAN_COLUMNS (also mirrored as
// run_isolated_godot.py's AUTHORING_PLAN_COLUMNS, tools/voice_catalog/
// run_isolated_godot.py:93-100), so a plan built from a manifest row-slice
// needs no field renaming:
//
// {
//   "q4_base":        "<abs path to qwen-talker-1.7b-base-Q4_K_M.gguf>",
//   "q8_voicedesign":  "<abs path to qwen-talker-1.7b-voicedesign-Q8_0.gguf>",
//   "q4_tok":          "<abs path to qwen-tokenizer-12hz-Q4_K_M.gguf>",
//   "q8_tok":          "<abs path to qwen-tokenizer-12hz-Q8_0.gguf>",
//   "out_dir":         "<abs dir; artifacts land at <out_dir>/<family_id>/<variant_index>/...>",
//   "families": [
//     {
//       "family_id": "<string>",
//       "variants": [
//         {
//           "variant_index":                 <int, 0-9>,
//           "seed":                          <int64>,
//           "maturity_archetype":             "<string>",
//           "vocal_presentation_archetype":   "<string>",
//           "register_archetype":             "<string>",
//           "vocal_mechanism_archetype":      "<string>",
//           "temperament_voice_archetype":    "<string>",
//           "accent_archetype":               "<string>",
//           "voice_design_prompt":            "<string>",
//           "reference_text":                 "<string>",
//           "reference_language":             "<string>",
//           "voice_design_model":             "<string>",
//           "voice_design_model_revision":    "<string>",
//           "base_model":                     "<string>",
//           "base_model_revision":            "<string>"
//         }
//       ]
//     }
//   ]
// }
//
// All fields required, string-typed unless noted; "seed" and
// "variant_index" must be bare JSON integers (not strings) so yyjson parses
// them exactly (no double round-trip). The 4 model path basenames are
// validated against the exact filenames qwen3_tts.cpp's exact_filename()
// requires (TASK3_Q4_BASE_IDENTITY etc., qwen3_tts.cpp:286-293) before any
// model load is attempted.
//
// Output per variant, under <out_dir>/<family_id>/<variant_index>/ (same
// layout as generate_voice_variants.gd's bundle_paths(), which nests
// directly under the (already family-scoped) --out argument the Godot
// script receives — this CLI adds the <family_id> path segment itself
// since one batch spans multiple families):
//   reference.wav, reference.spk, reference.rvq, reference.voice,
//   bundle.json, line_1.wav, line_2.wav, line_3.wav, line_4.wav
//
// One results.csv is written PER FAMILY, at <out_dir>/<family_id>/results.csv
// (not a single batch-root file) — grounded on run_isolated_godot.py's
// _publish_authoring_scratch (:1763-1827), which reads `scratch_root /
// "results.csv"` where scratch_root is a SINGLE family's own directory (its
// immediate subdirectories must be exactly that family's variant-index
// dirs, :1788-1793) and Task 3's own plan text ("spawns the CLI ONCE with a
// plan JSON covering N families writing <scratch>/<family>/<variant>/... +
// per-family results.csv, then loops _publish_authoring_scratch per family
// with scratch_root=<scratch>/<family>"). Writing it per family (rather
// than once at the end of the whole batch) also means a crash partway
// through a multi-family batch leaves every already-finished family in a
// state _publish_authoring_scratch can consume immediately, instead of
// losing every family's results to an unwritten final CSV.
// Header/column order pinned to generate_voice_variants.gd's RESULT_COLUMNS
// (:221-229) verbatim (18 columns, family_id..reload_verify_ms).
//
// Modes:
//   --plan <path>       batch mode (above contract)
//   --selftest-pcm16     converts a fixed float vector through the exact
//                        qwen3_tts.cpp:1044-1049 conversion and prints the
//                        resulting int16 values, space-separated, then exits
//                        0. No model paths needed. Catches a *32768/lrintf
//                        conversion regression before any Metal inference runs.
//
// Hard-fail policy (matches the rest of this codebase and the shipped
// runtime's CrashWrapper mandate, translated to a Godot-free CLI: no
// exceptions, no fallback data): the first family/variant failure prints an
// error to stderr and the process exits nonzero. Never substitute
// placeholder data for a failed step. stdout is fflush'd after every family
// so a crash mid-batch leaves prior families' artifacts intact. Each family's
// results.csv is written under <out_dir>/<family_id>/ only after ALL that
// family's variants succeed (a mid-family failure exits before that CSV is
// written), so no partial or fabricated-hash rows are ever emitted.

#include "qwen3_tts_qwentts_backend.h"
#include "voice_container_core.h"

#include "yyjson.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ============================================================================
// Model identity constants. Mirror qwen3_tts.cpp:286-293 (TASK3_Q4_BASE_IDENTITY
// etc.) exactly — these are the literal basenames exact_filename() there checks,
// and (for the Q4 pair) the literal strings voice_container_core.cpp's pack()/
// inspect() hardcode as the only accepted q4_base_identity/q4_tokenizer_identity
// values (voice_container_core.cpp:24-25). Duplicated here (not #include'd,
// since qwen3_tts.cpp is Godot-typed code this Godot-free binary cannot link)
// so a batch plan pointing at the wrong GGUF fails closed before any load.
constexpr const char *Q4_BASE_IDENTITY = "qwen-talker-1.7b-base-Q4_K_M.gguf";
constexpr const char *Q4_TOKENIZER_IDENTITY = "qwen-tokenizer-12hz-Q4_K_M.gguf";
constexpr const char *Q8_VOICE_DESIGN_IDENTITY = "qwen-talker-1.7b-voicedesign-Q8_0.gguf";
constexpr const char *Q8_TOKENIZER_IDENTITY = "qwen-tokenizer-12hz-Q8_0.gguf";

// LISTENING_SEED + LISTENING_LINES mirror generate_voice_variants.gd:23,27-32
// verbatim (ASCII only, same rationale: unverified tokenizer handling for
// non-ASCII punctuation would misattribute an artifact to the voice).
constexpr int64_t LISTENING_SEED = 20260731;
const std::vector<std::string> LISTENING_LINES = {
    "The birch canoe slid on the smooth planks.",
    "The corridor narrows as you press forward, and the torchlight begins to "
    "falter against damp stone that has not felt open air in a very long "
    "time. Somewhere ahead, water is moving, steady and patient, and the "
    "sound of it carries further than it should.",
    "Get back! The floor is giving way beneath you!",
    "Keep your voice down. Whatever is in that room has been listening to us "
    "since we came through the gate.",
};

// RESULT_COLUMNS mirrors generate_voice_variants.gd:221-229 verbatim, in
// order (18 columns, trailing reload_verify_ms from the Task-1 Phase-B
// timer addition). This is the single source the results.csv header AND
// row-writer below both draw from.
const std::vector<std::string> RESULT_COLUMNS = {
    "family_id", "variant_index", "seed_used", "model_load_ms",
    "design_inference_ms", "clone_extraction_ms", "reference_wav_sha256",
    "spk_sha256", "rvq_sha256", "voice_sha256", "bundle_sha256",
    "reload_status", "listening_seed",
    "line_1_wav_sha256", "line_2_wav_sha256",
    "line_3_wav_sha256", "line_4_wav_sha256",
    "reload_verify_ms",
};

void fatal(const std::string &message) {
    std::fprintf(stderr, "[aa-voice-authoring-cli] FATAL: %s\n", message.c_str());
    std::fflush(stderr);
}

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool exact_filename(const std::string &path, const char *expected) {
    // Mirrors qwen3_tts.cpp:295-297 (exact_filename): reject a model path
    // whose basename doesn't match the expected GGUF filename, closed by
    // default rather than trusting the plan JSON's declared identity.
    return !path.empty() && fs::path(path).filename().string() == expected;
}

// ============================================================================
// float -> PCM16 LE conversion. EXACT replica of qwen3_tts.cpp:1040-1051 (the
// lambda inside Qwen3TTS::synthesize's qwentts branch): clamp to +-1.0f,
// multiply by 32767.0f, truncating static_cast<int16_t>, little-endian pack.
// This is the byte-authoritative WAV-sample path for line_N.wav; do not
// "improve" it (e.g. to *32768.0f + lrintf rounding) — that changes bytes.
// ============================================================================
void append_pcm16_from_float(std::vector<uint8_t> &pcm16, const float *chunk, int n) {
    size_t base = pcm16.size();
    pcm16.resize(base + static_cast<size_t>(n) * 2);
    for (int i = 0; i < n; ++i) {
        float s = chunk[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t v = static_cast<int16_t>(s * 32767.0f);
        pcm16[base + static_cast<size_t>(i) * 2] = static_cast<uint8_t>(v & 0xFF);
        pcm16[base + static_cast<size_t>(i) * 2 + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    }
}

// EXACT replica of qwen3_tts.cpp:1867-1892 (s_pcm_to_wav): wraps PCM16 LE
// bytes in a 44-byte RIFF/WAVE/fmt /data header, 24kHz mono S16.
std::vector<uint8_t> pcm16_to_wav(const std::vector<uint8_t> &pcm, uint32_t sample_rate) {
    const uint32_t pcm_size = static_cast<uint32_t>(pcm.size());
    const uint32_t wav_size = 36 + pcm_size;
    std::vector<uint8_t> out(44 + static_cast<size_t>(pcm_size));
    uint8_t *p = out.data();
    auto put_le16 = [](uint8_t *dst, uint16_t v) {
        dst[0] = v & 0xFF;
        dst[1] = (v >> 8) & 0xFF;
    };
    auto put_le32 = [](uint8_t *dst, uint32_t v) {
        dst[0] = v & 0xFF;
        dst[1] = (v >> 8) & 0xFF;
        dst[2] = (v >> 16) & 0xFF;
        dst[3] = (v >> 24) & 0xFF;
    };
    std::memcpy(p + 0, "RIFF", 4);
    put_le32(p + 4, wav_size);
    std::memcpy(p + 8, "WAVE", 4);
    std::memcpy(p + 12, "fmt ", 4);
    put_le32(p + 16, 16);
    put_le16(p + 20, 1);
    put_le16(p + 22, 1);
    put_le32(p + 24, sample_rate);
    put_le32(p + 28, sample_rate * 2);
    put_le16(p + 32, 2);
    put_le16(p + 34, 16);
    std::memcpy(p + 36, "data", 4);
    put_le32(p + 40, pcm_size);
    if (pcm_size > 0) {
        std::memcpy(p + 44, pcm.data(), pcm_size);
    }
    return out;
}

bool write_bytes(const std::string &path, const std::vector<uint8_t> &bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    if (!bytes.empty()) {
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    return static_cast<bool>(f);
}

bool write_text(const std::string &path, const std::string &text) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(text.data(), static_cast<std::streamsize>(text.size()));
    return static_cast<bool>(f);
}

// ============================================================================
// .spk / .rvq file I/O. Layout documented in qwen3_tts_qwentts_backend.h:39-46
// and implemented (write-side) as private statics inside
// qwen3_tts_qwentts_backend.cpp (write_spk_file/write_rvq_file) that this
// binary calls indirectly via Backend::extract_voice_reference(). The
// read-side helpers below exist only because pack() needs the raw vectors
// back out, and those statics aren't exported — same documented layout,
// duplicated by necessity, not by choice:
//   .spk: [int32_le dim][dim * float32_le spk_emb]
//   .rvq: [int32_le num_codebooks][int32_le ref_T][num_codebooks*ref_T * int32_le codes]
// ============================================================================
bool read_spk_file(const std::string &path, std::vector<float> &emb) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t dim = 0;
    f.read(reinterpret_cast<char *>(&dim), 4);
    if (!f || dim <= 0 || dim > 65536) return false;
    emb.resize(static_cast<size_t>(dim));
    f.read(reinterpret_cast<char *>(emb.data()), static_cast<std::streamsize>(dim) * 4);
    return static_cast<bool>(f);
}

bool read_rvq_file(const std::string &path, std::vector<int32_t> &codes, int64_t &codebooks, int64_t &tokens) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t k = 0, t = 0;
    f.read(reinterpret_cast<char *>(&k), 4);
    f.read(reinterpret_cast<char *>(&t), 4);
    if (!f || k <= 0 || t <= 0 || k > 64 || t > 16384) return false;
    codebooks = k;
    tokens = t;
    codes.resize(static_cast<size_t>(k) * static_cast<size_t>(t));
    f.read(reinterpret_cast<char *>(codes.data()), static_cast<std::streamsize>(codes.size()) * 4);
    return static_cast<bool>(f);
}

// ============================================================================
// selftest: --selftest-pcm16
// ============================================================================
int run_selftest_pcm16() {
    const std::vector<float> fixed = {
        -1.5f, -1.0f, -0.5f, 0.0f, 0.33333334f, 0.5f, 1.0f, 1.5f,
    };
    std::vector<uint8_t> pcm16;
    append_pcm16_from_float(pcm16, fixed.data(), static_cast<int>(fixed.size()));
    std::string line;
    for (size_t i = 0; i < fixed.size(); ++i) {
        const uint8_t lo = pcm16[i * 2];
        const uint8_t hi = pcm16[i * 2 + 1];
        const int16_t v = static_cast<int16_t>(static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8));
        if (i > 0) line += ' ';
        line += std::to_string(v);
    }
    std::printf("%s\n", line.c_str());
    std::fflush(stdout);
    return 0;
}

// ============================================================================
// Batch plan JSON parsing (yyjson; vendored at vendor/yyjson, already linked
// by this project's tts-server target — no new dependency).
// ============================================================================
struct VariantPlan {
    int64_t variant_index = 0;
    int64_t seed = 0;
    std::string maturity_archetype;
    std::string vocal_presentation_archetype;
    std::string register_archetype;
    std::string vocal_mechanism_archetype;
    std::string temperament_voice_archetype;
    std::string accent_archetype;
    std::string voice_design_prompt;
    std::string reference_text;
    std::string reference_language;
    std::string voice_design_model;
    std::string voice_design_model_revision;
    std::string base_model;
    std::string base_model_revision;
};

struct FamilyPlan {
    std::string family_id;
    std::vector<VariantPlan> variants;
};

struct BatchPlan {
    std::string q4_base;
    std::string q8_voicedesign;
    std::string q4_tok;
    std::string q8_tok;
    std::string out_dir;
    std::vector<FamilyPlan> families;
};

bool json_require_str(yyjson_val *obj, const char *key, std::string &out, std::string &error) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v || !yyjson_is_str(v)) {
        error = std::string("missing or non-string field: ") + key;
        return false;
    }
    out = yyjson_get_str(v);
    return true;
}

bool json_require_int(yyjson_val *obj, const char *key, int64_t &out, std::string &error) {
    yyjson_val *v = yyjson_obj_get(obj, key);
    if (!v || !yyjson_is_int(v)) {
        error = std::string("missing or non-integer field: ") + key;
        return false;
    }
    out = yyjson_get_sint(v);
    return true;
}

bool parse_variant(yyjson_val *v, VariantPlan &out, std::string &error) {
    if (!yyjson_is_obj(v)) {
        error = "variant entry is not an object";
        return false;
    }
    return json_require_int(v, "variant_index", out.variant_index, error) &&
           json_require_int(v, "seed", out.seed, error) &&
           json_require_str(v, "maturity_archetype", out.maturity_archetype, error) &&
           json_require_str(v, "vocal_presentation_archetype", out.vocal_presentation_archetype, error) &&
           json_require_str(v, "register_archetype", out.register_archetype, error) &&
           json_require_str(v, "vocal_mechanism_archetype", out.vocal_mechanism_archetype, error) &&
           json_require_str(v, "temperament_voice_archetype", out.temperament_voice_archetype, error) &&
           json_require_str(v, "accent_archetype", out.accent_archetype, error) &&
           json_require_str(v, "voice_design_prompt", out.voice_design_prompt, error) &&
           json_require_str(v, "reference_text", out.reference_text, error) &&
           json_require_str(v, "reference_language", out.reference_language, error) &&
           json_require_str(v, "voice_design_model", out.voice_design_model, error) &&
           json_require_str(v, "voice_design_model_revision", out.voice_design_model_revision, error) &&
           json_require_str(v, "base_model", out.base_model, error) &&
           json_require_str(v, "base_model_revision", out.base_model_revision, error);
}

bool parse_family(yyjson_val *v, FamilyPlan &out, std::string &error) {
    if (!yyjson_is_obj(v)) {
        error = "family entry is not an object";
        return false;
    }
    if (!json_require_str(v, "family_id", out.family_id, error)) return false;
    // Fail-closed on any family_id that is not a bare catalog name. It becomes a
    // filesystem path segment (fs::path(out_dir) / family_id), so a value like
    // "../../x" or "/etc/x" would escape out_dir. Catalog family_ids are
    // [A-Za-z0-9_-] only; reject empty or any other character (blocks '/', '\',
    // '.', and therefore '..').
    if (out.family_id.empty() ||
        out.family_id.find_first_not_of(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") !=
            std::string::npos) {
        error = "family_id is not a bare catalog name (allowed [A-Za-z0-9_-]): '" +
                out.family_id + "'";
        return false;
    }
    yyjson_val *variants = yyjson_obj_get(v, "variants");
    if (!variants || !yyjson_is_arr(variants) || yyjson_arr_size(variants) == 0) {
        error = "family '" + out.family_id + "' has no variants array";
        return false;
    }
    size_t idx, max;
    yyjson_val *val;
    yyjson_arr_foreach(variants, idx, max, val) {
        VariantPlan vp;
        if (!parse_variant(val, vp, error)) return false;
        out.variants.push_back(vp);
    }
    return true;
}

bool parse_plan(const std::string &path, BatchPlan &out, std::string &error) {
    yyjson_read_err read_err;
    yyjson_doc *doc = yyjson_read_file(path.c_str(), 0, nullptr, &read_err);
    if (!doc) {
        error = std::string("cannot parse plan JSON '") + path + "': " + read_err.msg;
        return false;
    }
    yyjson_val *root = yyjson_doc_get_root(doc);
    bool ok = root && yyjson_is_obj(root) &&
              json_require_str(root, "q4_base", out.q4_base, error) &&
              json_require_str(root, "q8_voicedesign", out.q8_voicedesign, error) &&
              json_require_str(root, "q4_tok", out.q4_tok, error) &&
              json_require_str(root, "q8_tok", out.q8_tok, error) &&
              json_require_str(root, "out_dir", out.out_dir, error);
    if (ok) {
        yyjson_val *families = yyjson_obj_get(root, "families");
        if (!families || !yyjson_is_arr(families) || yyjson_arr_size(families) == 0) {
            error = "plan has no families array";
            ok = false;
        } else {
            size_t idx, max;
            yyjson_val *val;
            yyjson_arr_foreach(families, idx, max, val) {
                FamilyPlan fp;
                if (!parse_family(val, fp, error)) {
                    ok = false;
                    break;
                }
                out.families.push_back(fp);
            }
        }
    }
    yyjson_doc_free(doc);
    return ok;
}

// ============================================================================
// bundle.json writer. Field set + order mirrors generate_voice_variants.gd's
// _bundle_metadata() (:545-570) exactly; exact byte formatting (Godot's
// JSON.stringify vs yyjson's pretty writer) is NOT required to match — Task
// 4's A/B pilot only hashes .voice/.spk/.rvq/reference.wav/line_N.wav, not
// bundle.json.
// ============================================================================
bool write_bundle_json(const std::string &path, const FamilyPlan &family, const VariantPlan &variant,
                        const voice_container_core::PackMetadata &meta) {
    yyjson_mut_doc *doc = yyjson_mut_doc_new(nullptr);
    yyjson_mut_val *obj = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, obj);

    yyjson_mut_obj_add_sint(doc, obj, "schema_version", 1);
    yyjson_mut_obj_add_str(doc, obj, "family_id", family.family_id.c_str());
    yyjson_mut_obj_add_sint(doc, obj, "variant_index", variant.variant_index);
    yyjson_mut_obj_add_str(doc, obj, "maturity_archetype", variant.maturity_archetype.c_str());
    yyjson_mut_obj_add_str(doc, obj, "vocal_presentation_archetype", variant.vocal_presentation_archetype.c_str());
    yyjson_mut_obj_add_str(doc, obj, "register_archetype", variant.register_archetype.c_str());
    yyjson_mut_obj_add_str(doc, obj, "vocal_mechanism_archetype", variant.vocal_mechanism_archetype.c_str());
    yyjson_mut_obj_add_str(doc, obj, "temperament_voice_archetype", variant.temperament_voice_archetype.c_str());
    yyjson_mut_obj_add_str(doc, obj, "accent_archetype", variant.accent_archetype.c_str());
    yyjson_mut_obj_add_str(doc, obj, "voice_design_prompt", variant.voice_design_prompt.c_str());
    yyjson_mut_obj_add_str(doc, obj, "reference_text", variant.reference_text.c_str());
    yyjson_mut_obj_add_str(doc, obj, "reference_language", variant.reference_language.c_str());
    yyjson_mut_obj_add_sint(doc, obj, "seed_used", variant.seed);
    yyjson_mut_obj_add_str(doc, obj, "voice_design_model", variant.voice_design_model.c_str());
    yyjson_mut_obj_add_str(doc, obj, "voice_design_model_revision", variant.voice_design_model_revision.c_str());
    yyjson_mut_obj_add_str(doc, obj, "base_model", variant.base_model.c_str());
    yyjson_mut_obj_add_str(doc, obj, "base_model_revision", variant.base_model_revision.c_str());
    yyjson_mut_obj_add_str(doc, obj, "q4_base_identity", meta.q4_base_identity.c_str());
    yyjson_mut_obj_add_str(doc, obj, "q4_base_sha256", meta.q4_base_sha256.c_str());
    yyjson_mut_obj_add_str(doc, obj, "q4_tokenizer_identity", meta.q4_tokenizer_identity.c_str());
    yyjson_mut_obj_add_str(doc, obj, "q4_tokenizer_sha256", meta.q4_tokenizer_sha256.c_str());
    yyjson_mut_obj_add_sint(doc, obj, "rvq_codebooks", meta.rvq_codebooks);
    yyjson_mut_obj_add_sint(doc, obj, "rvq_tokens", meta.rvq_tokens);

    yyjson_write_err write_err;
    bool ok = yyjson_mut_write_file(path.c_str(), doc, YYJSON_WRITE_PRETTY_TWO_SPACES, nullptr, &write_err);
    yyjson_mut_doc_free(doc);
    return ok;
}

// ============================================================================
// Result row (one per variant) feeding results.csv, column set == RESULT_COLUMNS.
// ============================================================================
struct ResultRow {
    std::string family_id;
    int64_t variant_index = 0;
    int64_t seed_used = 0;
    int64_t model_load_ms = 0;
    int64_t design_inference_ms = 0;
    int64_t clone_extraction_ms = 0;
    std::string reference_wav_sha256;
    std::string spk_sha256;
    std::string rvq_sha256;
    std::string voice_sha256;
    std::string bundle_sha256;
    std::string reload_status;
    int64_t listening_seed = 0;
    std::string line_1_wav_sha256;
    std::string line_2_wav_sha256;
    std::string line_3_wav_sha256;
    std::string line_4_wav_sha256;
    int64_t reload_verify_ms = 0;
};

std::string result_field(const ResultRow &r, const std::string &column) {
    if (column == "family_id") return r.family_id;
    if (column == "variant_index") return std::to_string(r.variant_index);
    if (column == "seed_used") return std::to_string(r.seed_used);
    if (column == "model_load_ms") return std::to_string(r.model_load_ms);
    if (column == "design_inference_ms") return std::to_string(r.design_inference_ms);
    if (column == "clone_extraction_ms") return std::to_string(r.clone_extraction_ms);
    if (column == "reference_wav_sha256") return r.reference_wav_sha256;
    if (column == "spk_sha256") return r.spk_sha256;
    if (column == "rvq_sha256") return r.rvq_sha256;
    if (column == "voice_sha256") return r.voice_sha256;
    if (column == "bundle_sha256") return r.bundle_sha256;
    if (column == "reload_status") return r.reload_status;
    if (column == "listening_seed") return std::to_string(r.listening_seed);
    if (column == "line_1_wav_sha256") return r.line_1_wav_sha256;
    if (column == "line_2_wav_sha256") return r.line_2_wav_sha256;
    if (column == "line_3_wav_sha256") return r.line_3_wav_sha256;
    if (column == "line_4_wav_sha256") return r.line_4_wav_sha256;
    if (column == "reload_verify_ms") return std::to_string(r.reload_verify_ms);
    return "";
}

// Naive comma-join, no quoting: mirrors generate_voice_variants.gd's own
// _write_results_csv (:591-602), which likewise never quotes — every
// RESULT_COLUMNS value here is a hash hex string, an integer, "ok", or a
// family_id (no commas ever appear in any of those).
bool write_results_csv(const std::string &path, const std::vector<ResultRow> &results) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    for (size_t i = 0; i < RESULT_COLUMNS.size(); ++i) {
        if (i > 0) f << ',';
        f << RESULT_COLUMNS[i];
    }
    f << '\n';
    for (const auto &row : results) {
        for (size_t i = 0; i < RESULT_COLUMNS.size(); ++i) {
            if (i > 0) f << ',';
            f << result_field(row, RESULT_COLUMNS[i]);
        }
        f << '\n';
    }
    return static_cast<bool>(f);
}

}  // namespace

// ============================================================================
// Batch mode.
// ============================================================================
int run_batch(const std::string &plan_path) {
    BatchPlan plan;
    std::string error;
    if (!parse_plan(plan_path, plan, error)) {
        fatal(error);
        return 1;
    }
    if (!exact_filename(plan.q4_base, Q4_BASE_IDENTITY) ||
        !exact_filename(plan.q8_voicedesign, Q8_VOICE_DESIGN_IDENTITY) ||
        !exact_filename(plan.q4_tok, Q4_TOKENIZER_IDENTITY) ||
        !exact_filename(plan.q8_tok, Q8_TOKENIZER_IDENTITY)) {
        fatal("plan model paths do not match the required exact GGUF filenames "
              "(qwen-talker-1.7b-base-Q4_K_M.gguf / qwen-talker-1.7b-voicedesign-Q8_0.gguf / "
              "qwen-tokenizer-12hz-Q4_K_M.gguf / qwen-tokenizer-12hz-Q8_0.gguf)");
        return 1;
    }

    std::error_code ec;
    fs::create_directories(plan.out_dir, ec);
    if (ec) {
        fatal("cannot create out_dir: " + plan.out_dir);
        return 1;
    }

    // Load once per process: an authoring backend (base + voice-design
    // talkers, Phase A) and a runtime backend (base talker only, mirroring
    // the shipped Q4-only game runtime, Phase B). Hoisted out of the
    // per-family/per-variant loop below — this is the whole point of this
    // binary versus the Godot generator it replaces, which re-initializes a
    // fresh runtime PER VARIANT for Phase B (generate_voice_variants.gd:397-405).
    // Task 4 explicitly A/B-validates that this container-swap-on-one-runtime
    // reuse matches the Godot fresh-runtime-per-variant behavior.
    const int64_t load_start = now_ms();
    qwen3_tts_qwentts_backend::Backend authoring_backend;
    if (authoring_backend.initialize_authoring(plan.q4_base, plan.q8_voicedesign, plan.q4_tok, plan.q8_tok) != 0) {
        fatal("initialize_authoring failed");
        return 1;
    }
    if (!authoring_backend.can_design_voice()) {
        fatal("authoring backend did not expose VoiceDesign");
        return 1;
    }
    qwen3_tts_qwentts_backend::Backend runtime_backend;
    if (runtime_backend.initialize_runtime(plan.q4_base, plan.q4_tok) != 0) {
        fatal("initialize_runtime failed");
        return 1;
    }
    const int64_t model_load_ms = now_ms() - load_start;

    // Provenance: sha256 of the Q4 base talker + Q4 tokenizer, computed ONCE
    // for the whole batch (matches qwen3_tts.cpp:322-323/374-375 — provenance
    // is a property of which model files were loaded, not of any one variant).
    std::string q4_base_sha, q4_tok_sha;
    if (!voice_container_core::sha256_file(plan.q4_base, q4_base_sha) ||
        !voice_container_core::sha256_file(plan.q4_tok, q4_tok_sha)) {
        fatal("cannot sha256 the Q4 base talker or Q4 tokenizer");
        return 1;
    }

    size_t total_variants = 0;

    for (const FamilyPlan &family : plan.families) {
        const fs::path family_dir = fs::path(plan.out_dir) / family.family_id;
        std::vector<ResultRow> family_results;

        for (const VariantPlan &variant : family.variants) {
            const fs::path variant_dir = family_dir / std::to_string(variant.variant_index);
            fs::create_directories(variant_dir, ec);
            if (ec) {
                fatal("cannot create variant dir: " + variant_dir.string());
                return 1;
            }
            const std::string reference_wav = (variant_dir / "reference.wav").string();
            const std::string spk_path = (variant_dir / "reference.spk").string();
            const std::string rvq_path = (variant_dir / "reference.rvq").string();
            const std::string voice_path = (variant_dir / "reference.voice").string();
            const std::string bundle_path = (variant_dir / "bundle.json").string();

            ResultRow row;
            row.family_id = family.family_id;
            row.variant_index = variant.variant_index;
            row.seed_used = variant.seed;
            row.model_load_ms = model_load_ms;

            // ---- Phase A: design + extract + pack -------------------------
            const int64_t design_start = now_ms();
            std::vector<uint8_t> wav = authoring_backend.design_voice(
                variant.voice_design_prompt, variant.reference_text, variant.seed);
            row.design_inference_ms = now_ms() - design_start;
            if (wav.size() <= 44) {
                fatal("family '" + family.family_id + "' variant " + std::to_string(variant.variant_index) +
                      " produced no audio samples");
                return 1;
            }
            if (!write_bytes(reference_wav, wav)) {
                fatal("cannot write " + reference_wav);
                return 1;
            }

            const int64_t extract_start = now_ms();
            const int extract_rc = authoring_backend.extract_voice_reference(reference_wav, spk_path, rvq_path);
            row.clone_extraction_ms = now_ms() - extract_start;
            if (extract_rc != 0) {
                fatal("family '" + family.family_id + "' variant " + std::to_string(variant.variant_index) +
                      " extraction failed rc=" + std::to_string(extract_rc));
                return 1;
            }

            std::vector<float> spk;
            if (!read_spk_file(spk_path, spk)) {
                fatal("variant produced an unreadable SPK: " + spk_path);
                return 1;
            }
            std::vector<int32_t> rvq_codes;
            int64_t rvq_codebooks = 0, rvq_tokens = 0;
            if (!read_rvq_file(rvq_path, rvq_codes, rvq_codebooks, rvq_tokens)) {
                fatal("variant produced an unreadable RVQ: " + rvq_path);
                return 1;
            }

            voice_container_core::PackMetadata meta;
            meta.schema_version = 1;
            meta.family_id = family.family_id;
            meta.variant_index = variant.variant_index;
            meta.reference_transcript = variant.reference_text;
            meta.q4_base_identity = Q4_BASE_IDENTITY;
            meta.q4_base_sha256 = q4_base_sha;
            meta.q4_tokenizer_identity = Q4_TOKENIZER_IDENTITY;
            meta.q4_tokenizer_sha256 = q4_tok_sha;
            meta.rvq_codebooks = rvq_codebooks;
            meta.rvq_tokens = rvq_tokens;

            std::vector<uint8_t> packed = voice_container_core::pack(meta, spk, rvq_codes);
            if (packed.empty()) {
                fatal("family '" + family.family_id + "' variant " + std::to_string(variant.variant_index) +
                      " container packing failed (validation rejected the metadata/payload)");
                return 1;
            }
            if (!write_bytes(voice_path, packed)) {
                fatal("cannot write " + voice_path);
                return 1;
            }
            if (!write_bundle_json(bundle_path, family, variant, meta)) {
                fatal("cannot write " + bundle_path);
                return 1;
            }

            if (!voice_container_core::sha256_file(reference_wav, row.reference_wav_sha256) ||
                !voice_container_core::sha256_file(spk_path, row.spk_sha256) ||
                !voice_container_core::sha256_file(rvq_path, row.rvq_sha256) ||
                !voice_container_core::sha256_file(voice_path, row.voice_sha256) ||
                !voice_container_core::sha256_file(bundle_path, row.bundle_sha256)) {
                fatal("cannot sha256 a Phase-A artifact for variant " + std::to_string(variant.variant_index));
                return 1;
            }

            // ---- Phase B: reload on the shipped Q4 runtime + listening ----
            const int64_t reload_start = now_ms();
            std::ifstream voice_in(voice_path, std::ios::binary | std::ios::ate);
            if (!voice_in) {
                fatal("cannot reopen " + voice_path);
                return 1;
            }
            const std::streamoff voice_size = voice_in.tellg();
            if (voice_size <= 0) {
                fatal("empty container: " + voice_path);
                return 1;
            }
            voice_in.seekg(0, std::ios::beg);
            std::vector<uint8_t> container(static_cast<size_t>(voice_size));
            voice_in.read(reinterpret_cast<char *>(container.data()), voice_size);
            if (!voice_in) {
                fatal("short read on " + voice_path);
                return 1;
            }

            voice_container_core::ExpectedIdentity expected;
            expected.valid = true;
            expected.family_id = family.family_id;
            expected.variant_index = variant.variant_index;
            expected.q4_base_identity = Q4_BASE_IDENTITY;
            expected.q4_base_sha256 = q4_base_sha;
            expected.q4_tokenizer_identity = Q4_TOKENIZER_IDENTITY;
            expected.q4_tokenizer_sha256 = q4_tok_sha;
            voice_container_core::InspectResult decoded = voice_container_core::decode(container, expected);
            if (!decoded.ok) {
                fatal("family '" + family.family_id + "' variant " + std::to_string(variant.variant_index) +
                      " container rejected by the shipped Q4 runtime: " + decoded.error);
                return 1;
            }

            const int set_rc = runtime_backend.set_reference(
                decoded.speaker_embedding, decoded.rvq_codes,
                static_cast<int>(decoded.rvq_codebooks), static_cast<int>(decoded.rvq_tokens),
                decoded.reference_transcript);
            if (set_rc != 0) {
                fatal("family '" + family.family_id + "' variant " + std::to_string(variant.variant_index) +
                      " set_reference failed rc=" + std::to_string(set_rc));
                return 1;
            }
            row.reload_status = "ok";
            row.listening_seed = LISTENING_SEED;

            for (size_t line_index = 0; line_index < LISTENING_LINES.size(); ++line_index) {
                std::vector<uint8_t> pcm16;
                const int gen_rc = runtime_backend.generate_streaming(
                    LISTENING_LINES[line_index],
                    [&pcm16](const float *chunk, int n) -> bool {
                        append_pcm16_from_float(pcm16, chunk, n);
                        return true;
                    },
                    LISTENING_SEED);
                if (gen_rc != 0) {
                    fatal("family '" + family.family_id + "' variant " + std::to_string(variant.variant_index) +
                          " line " + std::to_string(line_index + 1) +
                          " generate_streaming failed rc=" + std::to_string(gen_rc));
                    return 1;
                }
                std::vector<uint8_t> clip = pcm16_to_wav(pcm16, 24000);
                if (clip.size() <= 44) {
                    fatal("family '" + family.family_id + "' variant " + std::to_string(variant.variant_index) +
                          " line " + std::to_string(line_index + 1) + " produced no audio");
                    return 1;
                }
                const std::string clip_path =
                    (variant_dir / ("line_" + std::to_string(line_index + 1) + ".wav")).string();
                if (!write_bytes(clip_path, clip)) {
                    fatal("cannot write " + clip_path);
                    return 1;
                }
                std::string clip_sha;
                if (!voice_container_core::sha256_file(clip_path, clip_sha)) {
                    fatal("cannot sha256 " + clip_path);
                    return 1;
                }
                switch (line_index) {
                    case 0: row.line_1_wav_sha256 = clip_sha; break;
                    case 1: row.line_2_wav_sha256 = clip_sha; break;
                    case 2: row.line_3_wav_sha256 = clip_sha; break;
                    case 3: row.line_4_wav_sha256 = clip_sha; break;
                    default: break;
                }
            }
            row.reload_verify_ms = now_ms() - reload_start;

            family_results.push_back(row);
            std::printf("VARIANT_OK family=%s index=%lld design_ms=%lld extract_ms=%lld reload_ms=%lld\n",
                         family.family_id.c_str(),
                         static_cast<long long>(variant.variant_index),
                         static_cast<long long>(row.design_inference_ms),
                         static_cast<long long>(row.clone_extraction_ms),
                         static_cast<long long>(row.reload_verify_ms));
        }

        // Per-family results.csv, written the moment this family's variants
        // finish (see the doc comment above run_batch's declaration for why
        // this is per-family rather than batch-root).
        const std::string family_results_path = (family_dir / "results.csv").string();
        if (!write_results_csv(family_results_path, family_results)) {
            fatal("cannot write " + family_results_path);
            return 1;
        }
        total_variants += family_results.size();
        std::printf("FAMILY_OK family=%s variants=%zu results=%s\n",
                     family.family_id.c_str(), family_results.size(), family_results_path.c_str());
        std::fflush(stdout);
    }

    std::printf("BATCH_OK families=%zu variants=%zu model_load_ms=%lld out=%s\n",
                plan.families.size(), total_variants,
                static_cast<long long>(model_load_ms), plan.out_dir.c_str());
    std::fflush(stdout);
    return 0;
}

int main(int argc, char **argv) {
    std::string plan_path;
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--plan") == 0 && i + 1 < argc) {
            plan_path = argv[++i];
        } else if (std::strcmp(argv[i], "--selftest-pcm16") == 0) {
            selftest = true;
        } else {
            fatal(std::string("unknown or incomplete argument: ") + argv[i]);
            return 1;
        }
    }
    if (selftest) {
        return run_selftest_pcm16();
    }
    if (plan_path.empty()) {
        fatal("usage: aa-voice-authoring-cli --plan <batch-plan.json> | --selftest-pcm16");
        return 1;
    }
    return run_batch(plan_path);
}
