"""
Phase 1 自動術語庫（per-manga 名稱記憶）。

從每頁的 (修正後原文, 譯文) 抽取「人/事/地/物的專有名詞」與其譯名，
第一次出現就記住、之後沿用以維持一致性。建議制：寫進 glossary，由現有
extract_relevant_terms 注入 system prompt（不強制取代）。

設計為 fail-soft：任何錯誤都不應中斷翻譯。
"""
import os
import json

# 只抽「人事地物」的專有名詞；嚴格排除一般詞彙/狀聲詞
_SYSTEM = (
    "You extract PROPER NOUNS ONLY from manga text and pair each with the translation "
    "actually used in the provided target text.\n"
    "IN SCOPE (only these): \n"
    "  - person: character names, titles, nicknames\n"
    "  - place: locations, place names\n"
    "  - organization: groups, teams, events, factions with a proper name\n"
    "  - thing: specifically NAMED items, mecha, techniques, products, coined terms\n"
    "STRICTLY EXCLUDE: common nouns, verbs, adjectives, pronouns, onomatopoeia/SFX, "
    "and any generic dialogue words. When unsure, leave it out (be conservative).\n"
    "For each proper noun that actually appears, return the EXACT source substring "
    "(preserve original spelling/characters verbatim) and its translation exactly as used "
    "in the target text.\n"
    "STRICT 1-to-1: treat every distinct source spelling as its own separate entry. NEVER "
    "merge, normalize, or skip a name just because it looks similar to another (e.g. 'luka' "
    "and 'luca' are TWO separate entries even if both translate to the same target).\n"
    "If there are none, return an empty list."
)

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "proper_nouns",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "names": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source": {"type": "string"},
                            "translation": {"type": "string"},
                            "type": {"type": "string",
                                     "enum": ["person", "place", "organization", "thing"]},
                        },
                        "required": ["source", "translation", "type"],
                    },
                }
            },
            "required": ["names"],
        },
    },
}


async def extract_names(client, model, pairs):
    """pairs: list[(source_text, translation_text)] -> list[(source, translation, type)].
    一個純文字 mini 呼叫，無圖、很便宜。失敗回空 list（不中斷翻譯）。"""
    pairs = [(s, t) for s, t in pairs if s and t]
    if not pairs:
        return []
    blocks = []
    for i, (s, t) in enumerate(pairs):
        blocks.append(f"[{i}] SOURCE: {s}\n[{i}] TARGET: {t}")
    user = "Extract proper nouns from these source/target pairs:\n\n" + "\n\n".join(blocks)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _SYSTEM},
                      {"role": "user", "content": user}],
            max_completion_tokens=2000,
            response_format=_SCHEMA,
        )
        if not resp.choices:
            return []
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return []
    out = []
    for n in data.get("names", []):
        src = (n.get("source") or "").strip()
        tgt = (n.get("translation") or "").strip()
        typ = (n.get("type") or "").strip()
        if src and tgt:
            out.append((src, tgt, typ))
    return out


def load_pairs(path):
    """讀取既有 per-manga glossary(mit 格式) -> dict{src: 'tgt #comment'}。失敗回 {}。"""
    entries = {}
    if not path or not os.path.exists(path):
        return entries
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                body = line.split("#", 1)[0].split("//", 1)[0].strip()
                parts = body.split("\t", 1)
                if len(parts) < 2:
                    parts = body.split(None, 1)
                if len(parts) >= 2:
                    entries[parts[0].strip()] = parts[1].strip()
    except Exception:
        pass
    return entries


def append_entries(path, entries):
    """append 新名詞到 per-manga glossary(mit 格式)。entries: list[(src,tgt,type)]。"""
    if not path or not entries:
        return
    try:
        new_file = not os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="\n") as f:
            if new_file:
                f.write("# Auto proper-noun glossary (manga-translator Phase 1). "
                        "First occurrence wins; edit freely.\n")
            for src, tgt, typ in entries:
                tag = f"\t#{typ}" if typ else ""
                f.write(f"{src}\t{tgt}{tag}\n")
    except Exception:
        pass
