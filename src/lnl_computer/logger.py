import logging
import sys

import tqdm


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = TqdmLoggingHandler()
formatter = logging.Formatter(
    "%(asctime)s| LNL-COMPUTER [%(levelname)s]: %(message)s",
    datefmt="%d-%m-%y %H:%M:%S",
)
handler.setFormatter(formatter)


# stream_handler = logging.StreamHandler(sys.stdout)

# stream_handler.setFormatter(formatter)
# logger.addHandler(stream_handler)
logger.addHandler(handler)
