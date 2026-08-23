"""Configuration: CLI args + optional ankidkdeck.toml + defaults. No source-code
constants to edit -- that was the v1/v2 workflow this package retires."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

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


def load_config(path: Path | None = None, **overrides) -> Config:
    cfg = Config()
    toml_path = path or Path("ankidkdeck.toml")
    if toml_path.exists():
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    for k, v in overrides.items():
        if v is not None and hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.work_dir = Path(cfg.work_dir)
    if cfg.legacy_workspace:
        cfg.legacy_workspace = Path(cfg.legacy_workspace)
    if cfg.wordlist_file:
        cfg.wordlist_file = Path(cfg.wordlist_file)
    return cfg
