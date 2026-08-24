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
    gemini_model: str = "gemini-2.0-flash"
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
    return cfg
