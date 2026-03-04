from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog
import yaml

_DEFAULT_YAML = Path(__file__).resolve().parent.parent.parent / "default.yaml"


# ---------------------------------------------------------------------------
# Nested config sections
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WarehouseConfig:
    width: int = 120
    height: int = 80
    stations: int = 20


@dataclass(slots=True)
class OrdersConfig:
    count: int = 600
    burst: bool = False


@dataclass(slots=True)
class PathsConfig:
    ffmpeg: str = "ffmpeg"
    output_dir: str = "outputs"


@dataclass(slots=True)
class VisualizationConfig:
    enabled: bool = False
    fps: int = 20
    steps_per_frame: int = 25


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"
    file: str = "sim.log"


@dataclass(slots=True)
class BaselineConfig:
    seed: int = 44
    robots: int = 20
    width: int = 120
    height: int = 80
    stations: int = 20
    orders_count: int = 600
    orders_burst: bool = True
    strategy_window: int = 50


@dataclass(slots=True)
class OptVarBounds:
    lower: int = 0
    upper: int = 1


@dataclass(slots=True)
class OptObjective:
    name: str = ""
    metric_key: str = ""
    direction: str = "minimize"


@dataclass(slots=True)
class OptimizationConfig:
    algorithm: str = "nsga2"
    population_size: int = 20
    generations: int = 30
    eval_ticks: int = 3000
    n_workers: int = 0
    strategy_window: int = 50
    fixed_area: int = 60000
    output_dir: str = "outputs/moop"
    save_pareto: bool = True
    variables: dict[str, OptVarBounds] = field(default_factory=dict)
    objectives: list[OptObjective] = field(default_factory=list)

    @property
    def var_names(self) -> list[str]:
        return list(self.variables.keys())

    @property
    def n_var(self) -> int:
        return len(self.variables)

    @property
    def n_obj(self) -> int:
        return len(self.objectives)

    @property
    def xl(self) -> list[float]:
        return [float(v.lower) for v in self.variables.values()]

    @property
    def xu(self) -> list[float]:
        return [float(v.upper) for v in self.variables.values()]


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SimConfig:
    scenario: str = "seed42"
    seed: int = 42
    ticks: int = 10000
    robots: int = 20
    warehouse: WarehouseConfig = field(default_factory=WarehouseConfig)
    orders: OrdersConfig = field(default_factory=OrdersConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # -- helpers -------------------------------------------------------------

    def scenario_path(self, filename: str) -> Path:
        """Resolve ``outputs/<scenario>/<filename>``."""
        return Path(self.paths.output_dir) / self.scenario / filename

    def ensure_scenario_dir(self) -> None:
        """Create the scenario output directory if it doesn't exist."""
        (Path(self.paths.output_dir) / self.scenario).mkdir(parents=True, exist_ok=True)

    # -- factory -------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> SimConfig:
        """Load config from *path* (falls back to bundled ``default.yaml``)."""
        yaml_path = Path(path) if path else _DEFAULT_YAML
        raw: dict[str, Any] = {}
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        return _build_config(raw)


def _build_config(raw: dict[str, Any]) -> SimConfig:
    """Merge a raw YAML dict into a ``SimConfig`` with defaults."""

    def _pick(section: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
        return {k: section[k] for k in keys if k in section}

    # Parse optimization section
    opt_raw = raw.get("optimization", {})
    opt_vars: dict[str, OptVarBounds] = {}
    for name, bounds in opt_raw.get("variables", {}).items():
        opt_vars[name] = OptVarBounds(lower=bounds["lower"], upper=bounds["upper"])
    opt_objs: list[OptObjective] = []
    for obj in opt_raw.get("objectives", []):
        opt_objs.append(OptObjective(
            name=obj["name"],
            metric_key=obj["metric_key"],
            direction=obj["direction"],
        ))
    opt_config = OptimizationConfig(
        **_pick(opt_raw, ("algorithm", "population_size", "generations", "eval_ticks",
                          "n_workers", "strategy_window", "fixed_area", "output_dir",
                          "save_pareto")),
        variables=opt_vars,
        objectives=opt_objs,
    )

    baseline_config = BaselineConfig(
        **_pick(raw.get("baseline", {}),
                ("seed", "robots", "width", "height", "stations",
                 "orders_count", "orders_burst", "strategy_window")),
    )

    return SimConfig(
        **_pick(raw, ("scenario", "seed", "ticks", "robots")),
        warehouse=WarehouseConfig(**_pick(raw.get("warehouse", {}), ("width", "height", "stations"))),
        orders=OrdersConfig(**_pick(raw.get("orders", {}), ("count", "burst"))),
        paths=PathsConfig(**_pick(raw.get("paths", {}), ("ffmpeg", "output_dir"))),
        visualization=VisualizationConfig(**_pick(raw.get("visualization", {}), ("enabled", "fps", "steps_per_frame"))),
        logging=LoggingConfig(**_pick(raw.get("logging", {}), ("level", "file"))),
        baseline=baseline_config,
        optimization=opt_config,
    )


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(cfg: LoggingConfig | None = None) -> None:
    """Configure *structlog* to write JSON lines to the log file."""
    cfg = cfg or LoggingConfig()

    file_handler = logging.FileHandler(cfg.file, mode="a", encoding="utf-8")
    file_handler.setLevel(getattr(logging, cfg.level, logging.INFO))

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, cfg.level, logging.INFO),
        handlers=[file_handler],
        force=True,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
