"""Prompt construction: the byte-frozen cores plus per-language enrichment.

Import this, never the pieces: `s42_translate._SYSTEM_PROMPTS` is the one place
the mapping from request kind to system prompt is written down, and it points
here.
"""

from .builder import (DEFAULT_PROMPT_ID, LEAN, PROMPT_IDS, RAMP_STAGE_NAMES,
                      RAMP_STAGES, RICH, activate, active_prompt_id,
                      active_ramp_stage, build_definition_prompt,
                      build_expression_prompt, build_review_prompt,
                      effective_prompt_id, effective_prompt_ids,
                      estimate_tokens, pack_identity, pack_sha256, pack_version,
                      ramp_stage_blocks, reset,
                      size_report, variant_for)
from .packs import allowed_sets, available

__all__ = [
    "DEFAULT_PROMPT_ID", "LEAN", "PROMPT_IDS", "RAMP_STAGES",
    "RAMP_STAGE_NAMES", "RICH", "activate", "active_prompt_id",
    "active_ramp_stage", "allowed_sets", "available",
    "build_definition_prompt", "build_expression_prompt",
    "build_review_prompt", "effective_prompt_id", "effective_prompt_ids",
    "estimate_tokens", "pack_identity", "pack_sha256",
    "pack_version", "ramp_stage_blocks", "reset",
    "size_report", "variant_for",
]
