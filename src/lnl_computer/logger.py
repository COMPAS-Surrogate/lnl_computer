import logging

import colorlog
import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(name: str):
    logger = colorlog.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = TqdmLoggingHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(name)s | %(asctime)s | %(levelname)s | %(message)s",
            datefmt="%d/%m/%y %H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "SUCCESS:": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    logger.addHandler(handler)
    return logger


logger = setup_logger("lnl_computer")
