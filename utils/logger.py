import logging
import sys
import io

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers on re-imports
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # Force UTF-8 console output
        ch = logging.StreamHandler(sys.stdout)
        
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger