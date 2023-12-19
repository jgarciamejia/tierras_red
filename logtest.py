#!/usr/bin/env python

import logging
print (logging.__version__)

#logging.basicConfig(filename='example.log', level=logging.DEBUG)
#logging.debug('This message should go to the log file')
#logging.info('So should this')
#logging.warning('And this, too')
#logging.error('And non-ASCII stuff, too, like Øresund and Malmö')

def setup_logger(log_file, log_level=logging.INFO):
    """Set up a logger with a FileHandler and a StreamHandler.
    Args:
        log_file (str): The name of the log file.
        log_level (int, optional): The log level. Defaults to logging.INFO.
    Returns:
        logger: The logger object.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Create a FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create a StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # Add the FileHandler and StreamHandler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logger('test.txt')

logging.info('hi')
logging.info('does this work?')
