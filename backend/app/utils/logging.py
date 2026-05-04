import logging
import sys

from app.config import get_settings


def setup_logging():
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    # Suppress noisy loggers
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("h5py").setLevel(logging.WARNING)
