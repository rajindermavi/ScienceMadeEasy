import logging
import config

def get_logger(log_name='log',log_path='log.log', mode = 'w'):
    """Configure and return the ETL logger."""
    logger = logging.getLogger(log_name)
    log_path = config.DEFAULT_LOG_DIR / log_path
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
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