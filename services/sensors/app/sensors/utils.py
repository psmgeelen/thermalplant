import logging
import logging.handlers
import queue
import threading
import sys
import time
from typing import Optional

# Global queue for all loggers
log_queue = queue.Queue(-1)  # No limit on queue size
queue_handler = None
queue_listener = None


def setup_queue_logging(level=logging.INFO, console=True, file=None) -> None:
    """
    Set up queue-based logging system for thread-safe logging.
    
    Args:
        level: Logging level (default: INFO)
        console: Whether to output logs to console (default: True)
        file: Optional file path to write logs to
    """
    global queue_handler, queue_listener
    
    # Create handlers for the queue listener
    handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # File handler
    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(levelname)s - %(asctime)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Create the queue handler that will be used by all loggers
    queue_handler = logging.handlers.QueueHandler(log_queue)
    
    # Configure the root logger to use the queue handler
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the queue handler to the root logger
    root_logger.addHandler(queue_handler)
    
    # Start the queue listener in a separate thread
    queue_listener = logging.handlers.QueueListener(
        log_queue, *handlers, respect_handler_level=True
    )
    queue_listener.start()
    
    # Log a message to indicate setup is complete
    logging.info("Queue-based logging system initialized")


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger configured to use the queue handler.
    
    Args:
        name: Name for the logger
        level: Optional specific level for this logger
        
    Returns:
        Logger instance configured to use queue logging
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    
    # Make sure we have queue logging set up
    if queue_handler is None:
        # Auto-setup with defaults if not already configured
        setup_queue_logging()
    
    return logger


def shutdown_logging() -> None:
    """
    Properly shut down the logging system, ensuring all logs are processed.
    Should be called before application exit.
    """
    global queue_listener
    
    if queue_listener:
        # Process any remaining logs
        time.sleep(0.1)  # Short delay to allow final logs to be added to queue
        
        # Stop the listener thread
        queue_listener.stop()
        queue_listener = None
        
        logging.info("Queue-based logging system shut down")
