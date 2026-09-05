"""Configuration: CLI args + optional ankidkdeck.toml + defaults. No source-code
constants to edit -- that was the v1/v2 workflow this package retires."""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .util import FatalError

LANGS_DEFAULT = ["Chinese", "English", "German", "Spanish"]

UA_DEFAULT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 "
    "(ankidkdeck; +https://github.com/iskoldt-X/ankidkdeck)"
)

# The three transports. They share the request-construction layer, the
# checkpoint/reconciliation layer and the gates; only the wire differs, and the
# choice is a HUMAN one -- nothing downgrades a batch wave to standard by
# itself. "flex" is standard plus serviceTier=flex (preview, best effort).
MODES = ("standard", "batch", "flex")

# The literal the API wants. It is required, not optional: leaving it unset
# means MEDIUM, which was measured at mean 578.7 thought tokens per request and
# is billed at output rates. thinking_budget=0 is NOT a substitute -- it is
# accepted and then clamped (65 thought tokens on a 7-token prompt).
THINKING_LEVELS = ("LOW", "MEDIUM", "HIGH")

# Per-model allow-list. A model belongs here only when BOTH are true: its
# constants have been measured on this project's keys, and its rate card has
# been read off the pricing page on a recorded date. Anything else must be
# refused rather than quoted, because every number downstream -- the output
# fit, the thinking level's behaviour, the cache floor, the price -- is a
# property of the model that produced it.
#
# gemini-2.0-flash was the default until 2026-08-26. It is deliberately NOT on
# this list: none of the v3 probe work was done on it, and it was still the
# effective model on the run host (whose ankidkdeck.toml had no override), so
# --confirm-spend would have paid 2.0-flash prices for 2.0-flash quality and
# welded "gemini:gemini-2.0-flash@..." into every cell's provenance.
VERIFIED_MODELS = {
    "gemini-3.7-flash": {
        "constants_measured_at": "2026-08-26",   # work/probes/stats.json
        "rate_card_read_at": "2026-08-13",
        "context_in": 1048576,
        "context_out": 65536,
    },
}


@dataclass
class Config:
    work_dir: Path = Path("work")
    # Read-only path to the recovered 2025 production workspace (translations,
    # download_map, HTML corpus). Required only by the migrate stage.
    legacy_workspace: Path | None = None
    langs: list[str] = field(default_factory=lambda: list(LANGS_DEFAULT))
    ua: str = UA_DEFAULT
    sleep_min: float = 2.0
    sleep_max: float = 4.0
    pilot_size: int = 300
    # Fixed year keeps rebuilds reproducible; never datetime.now() (the one
    # environment dependency that broke v2.1 byte-parity).
    copyright_year: int = 2026
    # Pin of the 2025 wordlist artifact; changing the wordlist re-ranks the
    # whole deck and must be an explicit decision.
    wordlist_file: Path | None = None
    wordlist_sha256: str | None = None
    gemini_model: str = "gemini-3.7-flash"
    # Expressions, the homograph ranking and the POS table are the three calls
    # whose output is short and whose failure mode is contamination rather than
    # truncation, so they may be pointed at a different (usually cheaper) model
    # without touching the 22,734 definition cells' house style. None = use
    # gemini_model.
    gemini_model_expressions: str | None = None
    # The release label stamped into card_keys.json rows. NOT __version__: those
    # rows are immutable once frozen, so a dev pre-release ("3.0.0a0") would
    # brand every v3.0 family forever in the file whose diff is the release
    # artifact.
    registry_version: str = "3.0"
    # Where the .apkg lands.
    dist_dir: Path = Path("dist")

    # ---------------- the LLM call, one field per decision ----------------
    # None of these existed before 2026-08-26, and _apply() FatalErrors on an
    # unknown TOML key -- so until the fields exist the run host cannot be
    # configured at all, which is why they are all here in one change even
    # though the transports that read some of them land later.
    #
    # Transport. A human picks it (--mode or the TOML); nothing switches by
    # itself, because an automatic fallback from batch to standard doubles the
    # rate silently.
    mode: str = "standard"
    # thinkingLevel, sent as this literal on EVERY request. The program is
    # PINNED to LOW (decision record: gemini-docs-verification.md), and the
    # pin is arithmetic, not taste -- see thinking_level_override_ack.
    thinking_level: str = "LOW"
    # "I know the output-cap math ignores thinking."
    #
    # maxOutputTokens is one budget that thoughts and candidates SHARE, and the
    # cap this pipeline derives is ceil(a*n + b) * 1.5 with NO thinking term.
    # That is exact at LOW (0 derived thinking across 38 observations, including
    # on a 4.5k prompt) and simply WRONG anywhere else: at MEDIUM the measured
    # p95 is 1,042 thought tokens, which is most of an n=20 definition batch's
    # entire 1,115-token cap. Both MAX_TOKENS finishes in the whole probe set
    # came from MEDIUM.
    #
    # So a non-LOW level cannot spend on "it was measured" alone -- a measured
    # thinking cost that the cap formula does not consume is a number nobody is
    # using. Spending above LOW takes BOTH the measurement AND this
    # acknowledgement, set by hand, by someone who has read the sentence above.
    thinking_level_override_ack: bool = False
    # None = derive the cap per request from the measured output fit (the
    # probe's a*n + b, times a 1.5 safety factor, floored at the lowest cap
    # actually tested). An int pins one cap for every request instead, which is
    # only useful when investigating a truncation.
    max_output_tokens: int | None = None
    # The floor for the derived cap. 1024 is the smallest cap that was measured
    # end to end (n=8 produced 250-307 output tokens under it); below that is
    # untested ground, and the cap does not cost anything -- billing is on
    # tokens actually produced, and thinking is 0 at LOW.
    max_output_floor: int = 1024
    # The cap for request kinds whose output was NEVER MEASURED: expressions,
    # the POS table, the homograph ranking. The measured fit (a*n + b) comes
    # entirely from definition requests, and its headroom at a full batch is
    # only 1.42x (n=20: 783 observed against a 1,115 cap) -- so extending it to
    # a kind whose glosses are full sentences would be a guess wearing a
    # measurement's clothes. A flat generous cap costs nothing (billing is on
    # tokens produced) while a wrong derived cap costs a truncated paid call.
    # Lower it only after someone measures those kinds.
    max_output_unmeasured: int = 4096
    # The prompt pack version, stamped into provenance and into the bill. It is
    # part of the identity of a translated cell: the measured thinking constant
    # is only valid for the prompt it was measured on, so a change here must
    # invalidate it rather than be silently carried over.
    prompt_id: str = "v4-frozen"
    # Explicit context caching. Free tier can never do it
    # (TotalCachedContentStorageTokensPerModelFreeTier limit=0) and there is no
    # implicit fallback inside batch, so this is the only discount path.
    cache_enabled: bool = False
    # ttl = cache_ttl_factor * the estimated wall clock of the wave. Batch
    # resolves the cache at EXECUTION time, not at submit time, so the TTL has
    # to cover the whole drain window, not just the submit.
    cache_ttl_factor: float = 1.5
    # A cache is bound to the key/project that created it: another key
    # referencing it gets 403 PERMISSION_DENIED. So a cached run pins ONE key
    # and the rotating pool is off.
    cache_key_index: int = 0
    # Only "flex" is meaningful, only on the standard surface, and mode=flex
    # implies it. A batch row must never carry serviceTier.
    service_tier: str | None = None
    # The self-hosted spend gate. Google's project-level cap does not stop an
    # already-submitted batch wave, so the ceiling has to be enforced here.
    spend_cap_usd: float = 10.0
    spend_cap_period: str = "month"
    # Clean retranslation. Off by default: it archives every existing row for
    # the language and pays for all of them again.
    retranslate_all: bool = False
    retranslate_reason: str = "clean_redo"
    # The probe artifact the measured constants are read from. None = the
    # default location under the workspace. Nothing measured is hard-coded in
    # source: a missing file has to refuse the spend, not fall back to a
    # plausible number.
    probe_stats_file: Path | None = None

    # ---------------- throttling (was five source constants) ----------------
    def_request_interval: float = 2.1
    expr_request_interval: float = 5.0
    pos_request_interval: float = 1.1
    rank_request_interval: float = 1.6
    max_per_api_key: int = 5
    # The real quotas of the project in use, read off AI Studio by a human and
    # stamped with the date they were read. None means "nobody has looked".
    # Measured on the free tier: 20 requests/model/day, and a 503 consumes one.
    # UNMEASURED on paid Tier 1: the probes ran at ~4.6 RPM and never saw a
    # per-minute 429, which is not the same as knowing the limit.
    rpm_limit: float | None = None
    rpd_limit: int | None = None
    rate_limits_measured_at: str | None = None
    # Escape hatch for a model that is not on the verified list. It exists so
    # that trying a new model is possible; it prints as a warning everywhere
    # and no spend gate treats the run as priced.
    allow_unverified_model: bool = False

    @property
    def expressions_model(self) -> str:
        return self.gemini_model_expressions or self.gemini_model

    @property
    def raw_dir(self) -> Path:
        return self.work_dir / "raw"

    @property
    def json_dir(self) -> Path:
        return self.work_dir / "json"

    @property
    def audio_dir(self) -> Path:
        return self.work_dir / "audio"

    @property
    def report_dir(self) -> Path:
        return self.work_dir / "reports"

    @property
    def registry_local(self) -> Path:
        return self.work_dir / "registry"

    @property
    def review_dir(self) -> Path:
        """Artifacts a human is required to read before a release (rejected
        classifications, refused merges, the registry diff)."""
        return self.work_dir / "review"

    @property
    def probe_stats_path(self) -> Path:
        """Where the measured LLM constants live."""
        return (Path(self.probe_stats_file) if self.probe_stats_file
                else self.work_dir / "probes" / "stats.json")

    @property
    def effective_service_tier(self) -> str | None:
        """mode=flex IS serviceTier=flex; batch rows never carry the field."""
        if self.mode == "flex":
            return "flex"
        if self.mode == "batch":
            return None
        return self.service_tier

    def model_is_verified(self, model: str | None = None) -> bool:
        return (model or self.gemini_model) in VERIFIED_MODELS

    def validate(self, spending: bool = False, stats: dict | None = None) -> None:
        """Refuse a configuration that cannot be priced or reproduced.

        Called from load_config(), and again by the stages that spend money --
        a Config built directly in a test or a script has never been through
        load_config().

        `spending=True` adds the checks that only apply when calls are about to
        be placed, and they need the measured constants: this module must not
        import the probe reader (the stage that owns it imports this one), so
        the CALLER passes `stats` in. Everything checked here is checked on the
        dry path too, except what genuinely cannot be: a bill may quote any
        configuration, but a paid run may not.
        """
        if self.mode not in MODES:
            raise FatalError("mode = %r is not one of %s"
                             % (self.mode, ", ".join(MODES)))
        if self.thinking_level not in THINKING_LEVELS:
            raise FatalError(
                "thinking_level = %r is not one of %s. It is sent as this exact "
                "literal; leaving it unset on the API side means MEDIUM, which "
                "was measured at mean 578.7 thought tokens per request."
                % (self.thinking_level, ", ".join(THINKING_LEVELS)))
        if self.mode == "batch" and self.service_tier:
            raise FatalError(
                "service_tier = %r with mode = batch: a batch JSONL row must "
                "not carry serviceTier at all." % (self.service_tier,))
        if self.service_tier not in (None, "flex"):
            raise FatalError("service_tier = %r: only \"flex\" (or unset) is "
                             "supported." % (self.service_tier,))
        if self.spend_cap_usd is not None and self.spend_cap_usd <= 0:
            raise FatalError("spend_cap_usd = %r must be positive; it is the "
                             "ceiling the local billing gate enforces."
                             % (self.spend_cap_usd,))
        if self.max_output_tokens is not None and self.max_output_tokens < 1:
            raise FatalError("max_output_tokens = %r is not a cap"
                             % (self.max_output_tokens,))
        for name, model in (("gemini_model", self.gemini_model),
                            ("gemini_model_expressions",
                             self.gemini_model_expressions)):
            if model and model not in VERIFIED_MODELS \
                    and not self.allow_unverified_model:
                raise FatalError(
                    "%s = %r is not on the verified model list (%s). Every "
                    "measured constant and every price downstream is a "
                    "property of the model that produced it, so an unpriced "
                    "model is refused rather than quoted. Set "
                    "allow_unverified_model = true to experiment -- no spend "
                    "gate will treat that run as priced."
                    % (name, model, ", ".join(sorted(VERIFIED_MODELS))))
        if spending:
            self.validate_spend(stats or {})

    def validate_spend(self, stats: dict) -> None:
        """The checks that only a run about to place calls has to pass.

        THINKING LEVEL. The program is pinned to LOW (decision record:
        gemini-docs-verification.md), and the pin is arithmetic. maxOutputTokens
        is ONE budget that thoughts and candidates SHARE, and the cap this
        pipeline derives is ceil(a*n + b) * 1.5 with NO thinking term: exact at
        LOW (0 derived thinking across 38 observations, including on a 4.5k
        prompt), wrong anywhere else. At MEDIUM the measured p95 is 1,042 thought
        tokens against an n=20 definition batch's whole 1,115-token cap, and both
        MAX_TOKENS finishes in the probe set came from MEDIUM.

        So "it was measured" is NOT sufficient: a measured thinking cost that the
        cap formula does not consume is a number nobody is using. Spending above
        LOW takes BOTH

          (a) a measured THINKING_PER_REQUEST for that exact level on disk, and
          (b) thinking_level_override_ack = true, set by hand.

        Either one missing is a refusal.
        """
        if self.thinking_level == "LOW":
            return
        key = "THINKING_PER_REQUEST_%s" % self.thinking_level
        measured = (stats.get("thinking") or {}).get(key)
        why = ("maxOutputTokens is ONE budget shared by thoughts and "
               "candidates, and the cap this pipeline derives "
               "(ceil(a*n + b) * 1.5) has NO thinking term -- it is only exact "
               "at LOW, where the measured thinking is 0. At MEDIUM the "
               "measured p95 is 1,042 thought tokens against an n=20 batch's "
               "entire 1,115-token cap, and both MAX_TOKENS finishes in the "
               "probe set came from MEDIUM. thinkingLevel is pinned to LOW for "
               "this program (decision record: gemini-docs-verification.md).")
        if not measured:
            raise FatalError(
                "thinking_level = %r cannot place paid calls: there is no "
                "measured thinking cost for it on disk (thinking.%s). %s "
                "Measure the level, or set thinking_level = \"LOW\"."
                % (self.thinking_level, key, why))
        if not self.thinking_level_override_ack:
            raise FatalError(
                "thinking_level = %r cannot place paid calls: it IS measured "
                "(thinking.%s = %s) but a measured thinking cost the cap "
                "formula never reads is a number nobody is using. %s To spend "
                "above LOW anyway, set thinking_level_override_ack = true -- it "
                "means \"I know the output-cap math ignores thinking\"."
                % (self.thinking_level, key, measured, why))
        print("  NOTE: thinking_level = %s with "
              "thinking_level_override_ack = true (measured %s). The derived "
              "output cap still has NO thinking term; the only guard left is "
              "_generate raising the budget once on a MAX_TOKENS finish."
              % (self.thinking_level, measured))


def settable_fields() -> list[str]:
    """The names a TOML file may set: the dataclass FIELDS, never every
    attribute `hasattr` answers True for.

    `expressions_model`, `raw_dir`, `json_dir`, `audio_dir`, `report_dir`,
    `registry_local` and `review_dir` are read-only properties, and hasattr() is
    True for all of them -- so `json_dir = "..."` in a config file raised a bare
    `AttributeError: property 'json_dir' has no setter` out of main(), which only
    catches FatalError. `expressions_model` is the likely typo (the settable
    field is gemini_model_expressions).
    """
    return [f.name for f in dataclasses.fields(Config)]


def _apply(cfg: Config, data: dict, source: str) -> None:
    allowed = settable_fields()
    for k, v in data.items():
        if k in allowed:
            setattr(cfg, k, v)
            continue
        hint = ""
        if isinstance(getattr(type(cfg), k, None), property):
            hint = (" -- that is a derived, read-only property, not a setting"
                    + (" (did you mean gemini_model_expressions?)"
                       if k == "expressions_model" else ""))
        raise FatalError(
            "unknown setting %r in %s%s. Accepted keys: %s"
            % (k, source, hint, ", ".join(allowed)))


def load_config(path: Path | None = None, **overrides) -> Config:
    cfg = Config()
    toml_path = path or Path("ankidkdeck.toml")
    if toml_path.exists():
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        _apply(cfg, data, str(toml_path))
    _apply(cfg, {k: v for k, v in overrides.items() if v is not None},
           "the command line")
    cfg.work_dir = Path(cfg.work_dir)
    cfg.dist_dir = Path(cfg.dist_dir)
    if cfg.legacy_workspace:
        cfg.legacy_workspace = Path(cfg.legacy_workspace)
    if cfg.wordlist_file:
        cfg.wordlist_file = Path(cfg.wordlist_file)
    if cfg.probe_stats_file:
        cfg.probe_stats_file = Path(cfg.probe_stats_file)
    cfg.validate()
    return cfg
