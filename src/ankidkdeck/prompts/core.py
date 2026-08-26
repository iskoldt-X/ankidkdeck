"""The byte-frozen prompt cores. THIS TEXT IS THE PRODUCT.

The 22,734 shipped cells per language were produced by PROMPT V4. Changing a
word makes the new cells stylistically foreign to the old ones, so the core is
frozen by BYTE, not by intent, and every enrichment is APPENDED after it. That
is what makes `LEAN` a pure prefix of `RICH`: the rollback from a rich prompt is
"assemble fewer blocks", never "maintain a second text".

The two functions here are the only place the core text exists. s42_translate's
`definition_prompt` / `expression_prompt` delegate to them, so the wire, the
bill, doctor, the cache key and the ledger row all read one string.

One edit is already inside the frozen text (2026-08-26, patch plan 1.7): the
OUTPUT FORMAT line used to interpolate the object count, which made the
"constant" system prompt a different string for every batch size and so made
one-explicit-cache-per-language impossible. 30 measured payloads produced 7
distinct sha256 values, one per value of n, differing by one or two characters.
The count is stated in the user message and enforced by the schema's
minItems == maxItems, which is where it was always actually enforced.

The SIZE of that edit, measured: German went from 5,124 characters to 5,134,
i.e. +10 characters / 0.2%. The "+50 characters, 1.0%" figure that the old
comment and the probe artifact both carried was never measured -- and it was
the number consumption rule 6's size band had been justified with.

The Danish in the structure example is DDO editorial style written by hand for
this repository; no DDO source text is stored here (v3_final_plan.md section 2).
"""


def definition_core(lang: str) -> str:
    """PROMPT V4, the frozen definition core, for one target language.

    English is the recorded special case: it has no CRITICAL RULE block, so the
    English core is 4,985 characters against 5,134 (German) and 5,160
    (Chinese, Spanish). Measured token counts: 1,135 for zh/de/es and 1,092 for
    en, both comfortably over the 1,024 explicit-cache floor.
    """
    critical_rule = ""
    if lang.lower() != "english":
        critical_rule = f"""
    **CRITICAL RULE: Both "lemma" and "gloss" MUST be in {lang}. Do NOT use English in your output, unless there is absolutely no way to express a concept in {lang}.**
    """

    system_prompt_base = f"""
    You are a senior lexicographer creating a new Danish-{lang} dictionary specifically for **language learners**. Your translations must be clear, practical, and intuitive.

    ### CORE TASK
    For each Danish definition provided, you must generate a corresponding `lemma` and `gloss` in **{lang}**.

    ### RULES FOR "lemma" (The {lang} headword):
    1.  **Be a real word/phrase:** It must be a concise, common {lang} word or a widely used, natural-sounding fixed phrase.
    2.  **Be practical, not overly technical:** AVOID purely grammatical descriptions. A conceptual hint is better than a niche technical term. If a grammatical term is widely understood by learners in the context of {lang} (e.g., a {lang} term for "noun suffix" when dealing with a suffix that forms nouns), it can be acceptable if truly the best and most concise fit.
    3.  **Balance specificity, naturalness, and informativeness. Focus on DEFINITION'S CORE SEMANTICS:**
        a.  **Capture Key Nuances:** The `lemma` should accurately reflect the specific and defining characteristics of THAT particular Danish definition.
        b.  **Prioritize Natural & Common Usage:** Choose {lang} words/phrases that are natural and commonly used by native speakers.
        c.  **Handling Multiple Meanings of Same Headword:** If multiple Danish definitions of the *same Danish headword* naturally map to the *same core {lang} lemma* (because it genuinely covers those nuances), you MAY reuse that lemma. Rely on distinct `glosses` for precise differentiation.
        d.  **When to Be More Specific (Informativeness):** However, if a Danish definition highlights a *distinct, crucial aspect* (e.g., a specific skill, a combined action, or a specialized context) that a very general {lang} lemma would obscure, prefer a *slightly more specific (yet still natural and common)* {lang} lemma that better conveys this key information. For example, if a definition describes "kicking a ball with light force AND high control", a lemma reflecting both aspects is better than one reflecting only "light kicking" if a natural {lang} term for the combined concept exists.
        e.  **Avoid Over-Simplification & Literal Traps:** Do NOT over-simplify the lemma to the point of losing essential semantic information from the Danish definition. Also, be cautious of being overly influenced by the literal translation of the *Danish headword itself* if the *definition* points to a more nuanced or specific meaning in {lang}. The definition is paramount.
        f.   **Clarity for Learners:** Ultimately, the lemma (in conjunction with the gloss) should provide maximum clarity for a language learner.
    4.  **Conciseness preferred:** Aim for brevity. Single words or short, common phrases in {lang} are ideal, but not at the cost of Rule 3's requirements.

    ### RULES FOR "gloss" (The explanatory translation):
    1.  **Be a clear, complete thought:** It must be a grammatically correct and natural-sounding sentence or a very clear, self-contained explanatory phrase in {lang}. Full sentences are preferred, but clarity and conciseness for the learner are paramount.
    2.  **Be explanatory, focused, AND faithful:** It should clearly explain the meaning and usage context relevant to THAT specific definition, being faithful to the scope and nuances of the Danish original. AVOID over-explaining with information not directly implied by the Danish definition, but ensure all key semantic components of the Danish definition are represented in the {lang} gloss. Focus on direct translation and essential clarification in {lang}.
    3.  **Match formality:** The tone of the gloss in {lang} should generally match the formality of the Danish definition.

    ### OUTPUT FORMAT
    Return pure JSON that **exactly matches the response schema**. The `definitions` array MUST have exactly as many objects as the user message states.
    """

    unified_example = f"""
    ### EXAMPLE OF STRUCTURE (target language is {lang}):
    Input Headword: "spille" (example headword, not necessarily from your data)
    Input Definitions: {{
        "0": "udføre musik på et instrument",
        "1": "deltage i et spil for fornøjelses skyld",
        "2": "i fodbold, aflevere bolden til en medspiller med præcis kontrol og ofte for at skabe en scoringsmulighed"
    }}

    Expected JSON Output Structure:
    {{
    "headword":"spille",
    "definitions":[
    {{
        "lemma":"[A {lang} lemma for 'to play music']",
        "gloss":"[A {lang} gloss explaining playing music on an instrument]."
    }},
    {{
        "lemma":"[A {lang} lemma for 'to play a game']",
        "gloss":"[A {lang} gloss explaining participating in a game for fun]."
    }},
    {{
        "lemma":"[A {lang} lemma for 'to pass with control (football)', informative and natural phrase]",
        "gloss":"[A {lang} gloss explaining the football-specific action of passing with control to create an opportunity, ensuring all key elements are covered]."
    }}
    ]}}
    """
    return f"{critical_rule}\n{system_prompt_base.strip()}\n\n{unified_example.strip()}"


def expression_core(lang: str) -> str:
    """06's generator prompt, frozen. The Russian clause exists because of a
    real contamination incident; do not remove it.

    `n_items` is gone from the last line and the correction instruction is gone
    from the signature, for the same reason as in the definition core: the
    system prompt is the cached prefix, so nothing per-batch may live in it.
    """
    system_prompt_base = f"""
You are a senior lexicographer translating Danish fixed expressions and idioms into {lang} for a dictionary aimed at **language learners**.
### CORE TASK
For each Danish expression in the input, provide a `lemma` and a `gloss` in {lang}.
### CONTEXT PROVIDED
- `expr`: The Danish expression to translate.
- `hint`: An optional Danish definition to clarify the expression's meaning. Use this hint to find the best translation, but do not translate the hint itself.
### RULES FOR "lemma":
- It must be the most natural and common equivalent idiom or phrase in {lang}.
- If a direct idiom exists, prefer it. If not, use a concise, descriptive phrase.
### RULES FOR "gloss":
- It must be a full, explanatory sentence in {lang} that clarifies the meaning and usage of the Danish expression for a learner.
### OUTPUT FORMAT
Return pure JSON matching the schema. The `fixed_expressions` array MUST have exactly as many objects as the user message states.
"""
    critical_rules = ""
    if lang.lower() != "english":
        critical_rules = f"""
**CRITICAL RULES: These are non-negotiable.**
1.  The primary language for both "lemma" and "gloss" MUST be **{lang}**.
2.  You must avoid all other languages. **DO NOT USE RUSSIAN under any circumstances.**
3.  **AS A LAST RESORT, ONLY IF** a concept is truly untranslatable into {lang}, you are permitted to use a concise English word or phrase in the `gloss`. This should be extremely rare. Never use English in the `lemma`.
"""
    return f"{critical_rules.strip()}\n\n{system_prompt_base.strip()}"
