"""
Logging Utility
---------------
Reusable logging setup for the entire project.

Usage:
    from src.utils.logger import setup_logger
    
    logger = setup_logger('my_module', 'outputs/logs/my_log.log')
    logger.info("This is an info message")
    logger.warning("This is a warning")
"""

import logging
import os
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO, console_output=True):
    """
    Setup and return a configured logger
    
    Args:
        name (str): Name of the logger (usually module name)
        log_file (str): Path to log file. If None, only console output
        level (int): Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output (bool): Whether to show logs in terminal
        
    Returns:
        logging.Logger: Configured logger instance
        
    Example:
        logger = setup_logger('training', 'outputs/logs/training.log')
        logger.info("Training started")
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
    '%(asctime)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
  )
    
    # File handler (save to file)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Console handler (show in terminal)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name):
    """
    Get an existing logger by name
    Useful when want to use the same logger in multiple functions
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)


def create_timestamped_log(name, log_dir='outputs/logs', level=logging.INFO):
    """
    Create a logger with timestamp in filename
    Useful for multiple runs/experiments
    
    Args:
        name (str): Logger name
        log_dir (str): Directory for log files
        level (int): Logging level
        
    Returns:
        logging.Logger: Configured logger with timestamped file
        
    Example:
        logger = create_timestamped_log('training')
        # Creates: outputs/logs/training_2024-11-22_15-30-45.log
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
    
    return setup_logger(name, log_file, level)


# Example usage and testing
if __name__ == "__main__":
    """
    Test the logger functionality
    Run this file directly to see examples
    """
    
    print("Testing logger...\n")
    
    print("\n" + "="*60 + "\n")
    
    # Logger with timestamp
    logger2 = create_timestamped_log('test_timestamp')
    logger2.info("This log has timestamp in filename")
    
    print("\n" + "="*60 + "\n")
    
    # Different log levels
    logger3 = setup_logger('test_levels', 'outputs/logs/levels.log', level=logging.DEBUG)
    logger3.debug("Debug message - very detailed")
    logger3.info("Info message - normal operation")
    logger3.warning("Warning - potential issue")
    logger3.error("Error - something went wrong")
    logger3.critical("Critical - fatal error!")
    
    print("\nLogging test complete! Check outputs/logs/ folder")