import sys
from loguru import logger

logger.remove()

logger.add(
    sys.stderr,
    level="DEBUG",  # Set to DEBUG to see everything
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    backtrace=True,  # Show full stack trace in console
    diagnose=True    # Show variable values in console
)