import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_file_path: str = 'run.log'):
    if not hasattr(setup_logging, "intialized"):
        # Set up the root logger to handle logs from third-party libraries
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # General file handler for all logs
        general_f_handler = RotatingFileHandler(log_file_path, maxBytes=1000000, backupCount=5)
        general_f_handler.setLevel(logging.INFO)
        general_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')
        general_f_handler.setFormatter(general_format)
        root_logger.addHandler(general_f_handler)

        # Project-specific logger with detailed configuration
        project_logger = logging.getLogger('minipack')
        project_logger.setLevel(logging.INFO)

        # Console handler for immediate output of project logs
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        project_logger.addHandler(c_handler)

        # Set the flag to prevent reinitialization of the logger
        setup_logging.intialized = True
    else:
        project_logger = logging.getLogger('minipack')
    return project_logger