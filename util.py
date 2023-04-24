import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def _tmp() -> Path:
    return Path(os.environ['DNNROOF_TMPDIR'])
