import logging
from pathlib import Path

from etl.config import DEFAULT_LOG_DIR

logging_level = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
}

def get_logger(log_name='log', log_path='log.log', mode='w', level='INFO'):
    """Configure and return the ETL logger."""
    logger = logging.getLogger(log_name)

    # Accept either a relative file name or an absolute/Path object
    log_path = Path(log_path)
    if not log_path.is_absolute():
        log_path = DEFAULT_LOG_DIR / log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)

    if logger.handlers:
        return logger

    logger.setLevel(logging_level.get(level, logging.INFO))
    # log_dir = os.path.join(os.path.dirname(__file__), "logging")
    # os.makedirs(log_dir, exist_ok=True)
    # log_path = os.path.join(log_dir, "etl.log")

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
