"""
A helper script to create logging functionality.
For more logger attributes visit:
https://docs.python.org/3/library/logging.html#logrecord-attributes
"""

# imports
import logging


def setup_logger(self, log_path: str) -> logging.Logger:
    """
    A helper method to set up logging.

    Parameters
    =--------=
    log_path: a python string object
        The path of the file handler for the logger

    Returns
    =-----=
    logger: a python logger object
        The final logger that has been setup up
    """
    try:
        # getting the log path
        log_path = log_path

        # adding logger to the script
        logger = logging.getLogger(__name__)
        print(f'--> {logger}')
        # setting the log level to info
        logger.setLevel(logging.DEBUG)
        # setting up file handler
        file_handler = logging.FileHandler(log_path)

        # setting up formatter
        formatter = logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(funcName)s] [%(levelname)s] " +
                "--> [%(message)s]")

        # setting up file handler and formatter
        file_handler.setFormatter(formatter)
        # adding file handler
        logger.addHandler(file_handler)
        print(f'logger {logger} created at path: {log_path}')
    except Exception as e:
        logger.error(e, exec_info=True)
        print(e)
    finally:
        # return the logger object
        return logger
