
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import logging
from logging_config import configure_global_logging, RunContext

def test_logging():
    print("--- 1. Setting up Global Logging ---")
    ctx = RunContext()
    ctx.set_step('verification')
    
    # Configure root logger
    logger = configure_global_logging(
        context=ctx,
        level=logging.INFO,
        json_output=False
    )
    
    print("--- 2. Testing Root Logger ---")
    logger.info("This should be GREEN and have [verification] prefix")
    
    print("--- 3. Testing Child Module Logger ---")
    # Simulate a module logger (like in src/embeddings.py)
    child_logger = logging.getLogger('src.embeddings')
    child_logger.info("This should ALSO be GREEN and have prefix")
    child_logger.warning("This should be YELLOW")
    child_logger.error("This should be RED")
    
    print("--- 4. Testing Extra Fields ---")
    child_logger.info("Structured log", extra={'files': 42, 'status': "ok"})

if __name__ == "__main__":
    test_logging()
