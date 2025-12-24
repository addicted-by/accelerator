import functools
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

from accelerator.utilities.distributed_state import distributed_state

if os.getenv("ACCELERATOR_DEBUG", None):
    DEFAULT_LOG_LEVEL = logging.DEBUG
    _IS_DEBUG_LEVEL = True
else:
    DEFAULT_LOG_LEVEL = logging.INFO
    _IS_DEBUG_LEVEL = False

DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_FRAMEWORK_NAME = "__main__"
DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MB
DEFAULT_BACKUP_COUNT = 5

COLORS = {
    "DEBUG": "\033[94m",
    "INFO": "\033[92m",
    "WARNING": "\033[93m",
    "ERROR": "\033[91m",
    "CRITICAL": "\033[91;1m",
    "RESET": "\033[0m",
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to levelname in console output"""

    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            colored_levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
            record.levelname = colored_levelname
        return super().format(record)


class Logger:
    """Singleton Logger class for the framework"""

    _instance = None
    _initialized = False
    _loggers: dict[str, logging.Logger] = {}
    _file_handler = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_file: Union[str, Path] = None):
        # Initialize only once
        if Logger._initialized:
            if log_file is not None:
                self._update_log_file(log_file)
            return

        self.log_file = log_file

        # if is_package_installed('hydra'):
        #     import hydra
        #     from hydra.core.hydra_config import HydraConfig

        #     if HydraConfig.initialized():
        #         self._hydra_initialized = True
        #         self.root_logger = logging.getLogger(f'{DEFAULT_FRAMEWORK_NAME}')
        #         self.root_logger.info("Using Hydra configured logger")
        #     else:
        #         self._configure_default_logger()

        # else:
        self._configure_default_logger()

        if self.log_file:
            self.root_logger.info(f"Testing log file writing to {self.log_file}")

        Logger._initialized = True

    def _get_log_level(self) -> int:
        """Get default log level"""
        return DEFAULT_LOG_LEVEL

    def _configure_default_logger(self):
        """Configure default logger with colored console and file handlers"""
        base_logger = logging.getLogger(f"{DEFAULT_FRAMEWORK_NAME}")

        # ---------- NEW: stop bubbling to the real root ----------
        base_logger.propagate = False  # <-- ADD THIS LINE

        self.root_logger = _RankZeroFilter(base_logger)
        self.log_file = Path(self.log_file) if self.log_file else None
        self.log_level = self._get_log_level()
        self.root_logger.setLevel(self.log_level)

        # clear only **our** handlers; leave any pre-existing external ones alone
        for h in list(base_logger.handlers):
            if isinstance(h, (logging.StreamHandler, logging.FileHandler)):
                base_logger.removeHandler(h)

        # ---------- NEW: add console handler only if absent ----------
        if not any(isinstance(h, logging.StreamHandler) for h in base_logger.handlers):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_formatter = ColoredFormatter(DEFAULT_LOG_FORMAT)
            console_handler.setFormatter(console_formatter)
            base_logger.addHandler(console_handler)

        if self.log_file:
            self._add_file_handler(self.log_file)

        self.root_logger.debug("Framework logger initialized with default configuration")

    def _add_file_handler(self, log_file: Union[str, Path]):
        """Add a file handler to the root logger"""
        log_file = Path(log_file)
        try:
            log_dir = log_file.parent
            if log_dir != Path("."):
                log_dir.mkdir(parents=True, exist_ok=True)

            if self._file_handler is not None:
                for handler in self.root_logger.handlers[:]:
                    if isinstance(handler, logging.FileHandler):
                        self.root_logger.removeHandler(handler)
                self._file_handler = None

            self._file_handler = RotatingFileHandler(
                str(log_file),  # Convert to string explicitly
                maxBytes=DEFAULT_MAX_BYTES,
                backupCount=DEFAULT_BACKUP_COUNT,
            )
            self._file_handler.setLevel(self.log_level)
            file_formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
            self._file_handler.setFormatter(file_formatter)
            self.root_logger.addHandler(self._file_handler)

            if not log_file.exists():
                self.root_logger.warning(f"Log file does not exist after writing: {log_file.absolute()}")
        except (PermissionError, FileNotFoundError) as e:
            self.root_logger.warning(f"Could not create log file: {e}. Logging to console only.")

    def _update_log_file(self, log_file: Union[str, Path]):
        """Update the log file configuration for all loggers"""
        log_file = Path(log_file)
        current_log_file = Path(self.log_file) if self.log_file else None

        # print(f"Updating log file from {current_log_file} to {log_file}")

        if current_log_file == log_file and self._file_handler is not None:
            print("Log file path unchanged, skipping update")
            return

        self.log_file = log_file

        self._add_file_handler(log_file)

        for _, logger in self._loggers.items():
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)

            if not logger.propagate:
                logger.addHandler(self._file_handler)

        # self.root_logger.info(f"Updated log file to: {log_file}")

    def get_logger(self, name: str = None) -> logging.Logger:
        """
        Get a logger instance for the specified name.
        If name is None, returns the root framework logger.

        Args:
            name: Optional module/component name to get a specific logger for

        Returns:
            logging.Logger: Configured logger instance
        """
        if name is None:
            logger = self.root_logger

        else:
            if name not in self._loggers:
                logger_name = f"{DEFAULT_FRAMEWORK_NAME}.{name}" if name else f"{DEFAULT_FRAMEWORK_NAME}"
                logger = _RankZeroFilter(logging.getLogger(logger_name))

                self._loggers[name] = logger

            logger = self._loggers[name]

        return logger


def get_logger(name: str = None, log_file: Union[str, Path] = None) -> logging.Logger:
    """
    Get a configured logger for any module in the framework.

    Args:
        name: Optional name for the logger (typically module/component name)
        log_file: Optional log file path (can be set/updated any time)

    Returns:
        logging.Logger: Configured logger instance
    """
    framework_logger = Logger(log_file)
    return framework_logger.get_logger(name)


def set_log_file(log_file: Union[str, Path]) -> None:
    """
    Set or update the log file used by all loggers.
    This can be called at any time to change where logs are written.

    Args:
        log_file: Path to the log file

    Returns:
        None
    """
    log_file = Path(log_file) if log_file else None
    logger_instance = Logger()
    if log_file and logger_instance._initialized:  # and not logger_instance._hydra_initialized:
        logger_instance._update_log_file(log_file)
    root_logger = logger_instance.get_logger()
    root_logger.info(f"Log file set to: {log_file.absolute() if log_file else 'None'}")

    return logger_instance


class _RankZeroFilter:
    """A tiny proxy around a Logger that no-ops on non-zero ranks."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    @property
    def _zr(self):
        return (distributed_state.rank == 0) or _IS_DEBUG_LEVEL

    def __getattr__(self, name):
        attr = getattr(self._logger, name)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            if self._zr:
                return attr(*args, **kwargs)

        return wrapper


def log_function_call(level: str = "DEBUG"):
    """
    Decorator to log function calls, arguments, and return values.

    Args:
        level: Logging level to use
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            level_num = getattr(logging, level.upper(), logging.DEBUG)

            if logger.isEnabledFor(level_num):
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                logger.log(level_num, f"Calling {func.__name__}({signature})")

            try:
                result = func(*args, **kwargs)
                if logger.isEnabledFor(level_num):
                    logger.log(level_num, f"{func.__name__} returned {result!r}")
                return result
            except Exception as e:
                logger.exception(f"Exception in {func.__name__}: {str(e)}")
                raise

        return wrapper

    return decorator
