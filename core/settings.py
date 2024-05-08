import logging
import os

# configuration setting for logging
LOGLEVEL = os.environ.get('LOGLEVEL', 'info').upper()
logger = logging.getLogger("EV ASSISTANT")
logger.setLevel(LOGLEVEL)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:   %(asctime)s - %(name)s -  %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
