import logging
from megengine.distributed.group import get_rank
from megengine.distributed import is_distributed

logger_initialized = {}

def get_logger(name, log_file=None, log_level=logging.INFO):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    
    if name in logger_initialized:
        return logger

    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name): # child
            return logger

    # fix stream twice bug
    # while logger.handlers:
    #     logger.handlers.pop()

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if is_distributed():
        rank = get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    return logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. 
    By default a StreamHandler will be added. 
    If `log_file` is specified, a FileHandler will also be added. 
    The name of the root logger is the top-level package name, e.g., "edit".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    root_name = __name__.split('.')[0]  # edit.utils.logger
    if is_distributed():
        rank = get_rank()
        root_name = "rank" + str(rank) + "_" + root_name
    logger = get_logger(root_name, log_file, log_level)
    return logger