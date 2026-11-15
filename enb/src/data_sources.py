"""Frozen aggregate input interface for the public reproduction package."""
from pathlib import Path
from functools import lru_cache
import yaml
ROOT = Path(__file__).resolve().parents[1]
P1 = "P1-small-gas-midband"
P2 = "P2-medium-electric"
P3 = "P3-large-mixed-low"
PROTOTYPES = [P1,P2,P3]
@lru_cache(maxsize=1)
def load_config():
    return yaml.safe_load((ROOT/"config/accepted_assumptions.yaml").read_text(encoding="utf8"))
