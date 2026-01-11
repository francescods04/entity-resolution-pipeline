"""
logging_config.py - Structured Logging System

Professional logging with:
- JSON structured output for parsing
- Run ID for correlation
- Step tracking
- Performance metrics
- Error context

USAGE:
------
from logging_config import get_logger, RunContext

ctx = RunContext()
logger = get_logger('pipeline', ctx)
logger.info("Started processing", files=766, step='ingest')
"""

import json
import logging
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from functools import wraps
import traceback


# =============================================================================
# RUN CONTEXT
# =============================================================================

class RunContext:
    """
    Context object that tracks run-level information.
    
    Provides:
    - run_id: Unique identifier for this execution
    - step: Current pipeline step
    - timing: Start times for duration tracking
    """
    
    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.step = None
        self.step_start = None
        self._step_times: Dict[str, float] = {}
    
    def set_step(self, step: str) -> None:
        """Set current step and record timing."""
        # Record previous step duration
        if self.step and self.step_start:
            duration = time.time() - self.step_start
            self._step_times[self.step] = duration
        
        self.step = step
        self.step_start = time.time()
    
    def get_step_duration(self, step: str) -> Optional[float]:
        """Get duration of a completed step."""
        return self._step_times.get(step)
    
    def get_total_duration(self) -> float:
        """Get total run duration in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_timing_summary(self) -> Dict[str, float]:
        """Get timing for all completed steps."""
        return dict(self._step_times)
    
    def to_dict(self) -> Dict:
        """Convert to dict for logging."""
        return {
            'run_id': self.run_id,
            'step': self.step,
            'elapsed_sec': self.get_total_duration(),
        }


# =============================================================================
# JSON FORMATTER
# =============================================================================

class JsonFormatter(logging.Formatter):
    """
    Formats log records as JSON for structured logging.
    
    Output includes:
    - timestamp
    - level
    - logger name
    - message
    - run context (if available)
    - extra fields
    """
    
    def __init__(self, context: Optional[RunContext] = None):
        super().__init__()
        self.context = context
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add run context
        if self.context:
            log_data['run_id'] = self.context.run_id
            log_data['step'] = self.context.step
        
        # Add extra fields (passed via logger.info("msg", extra={...}))
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno',
                          'module', 'msecs', 'pathname', 'process',
                          'processName', 'relativeCreated', 'stack_info',
                          'thread', 'threadName', 'message', 'exc_info',
                          'exc_text']:
                if not key.startswith('_'):
                    log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = traceback.format_exception(*record.exc_info)
        
        return json.dumps(log_data)


class HumanFormatter(logging.Formatter):
    """
    Human-readable formatter with colors for terminal.
    """
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, context: Optional[RunContext] = None, use_colors: bool = True):
        super().__init__()
        self.context = context
        self.use_colors = use_colors
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime('%H:%M:%S')
        level = record.levelname[:4]
        
        if self.use_colors:
            color = self.COLORS.get(record.levelname, '')
            level = f"{color}{level}{self.RESET}"
        
        # Build prefix
        prefix_parts = [timestamp, level]
        if self.context and self.context.step:
            prefix_parts.append(f"[{self.context.step}]")
        
        prefix = ' | '.join(prefix_parts)
        
        return f"{prefix} | {record.getMessage()}"


# =============================================================================
# LOGGER FACTORY
# =============================================================================

class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that includes context in all log calls.
    
    Allows: logger.info("message", count=10, file="x.xlsx")
    """
    
    def process(self, msg, kwargs):
        # Move extra kwargs into extra dict
        extra = kwargs.get('extra', {})
        
        # Find kwargs that aren't standard logging kwargs
        standard_keys = {'exc_info', 'stack_info', 'stacklevel', 'extra'}
        for key in list(kwargs.keys()):
            if key not in standard_keys:
                extra[key] = kwargs.pop(key)
        
        if extra:
            kwargs['extra'] = extra
        
        return msg, kwargs





def configure_global_logging(
    context: Optional[RunContext] = None,
    level: int = logging.INFO,
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> ContextLogger:
    """
    Configure the ROOT logger so all modules share settings.
    
    Args:
        context: RunContext for correlation
        level: Logging level
        json_output: Use JSON format for console
        log_file: Optional file to write logs to
    
    Returns:
        Root logger (wrapped)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to prevent duplicates
    root_logger.handlers = []
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    if json_output:
        console.setFormatter(JsonFormatter(context))
    else:
        # Pass context so all logs get run_id/step
        console.setFormatter(HumanFormatter(context))
    
    root_logger.addHandler(console)
    
    # File handler
    if log_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(str(file_path))
        file_handler.setFormatter(JsonFormatter(context))
        root_logger.addHandler(file_handler)
    
    # Silence noisy libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return ContextLogger(root_logger, {})



# =============================================================================
# TIMING DECORATOR
# =============================================================================

def timed(logger: logging.Logger = None, level: int = logging.INFO):
    """
    Decorator to time function execution.
    
    Usage:
        @timed(logger)
        def my_function():
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                if logger:
                    logger.log(level, f"{func.__name__} completed", 
                              duration_sec=round(duration, 2))
                return result
            except Exception as e:
                duration = time.time() - start
                if logger:
                    logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
                raise
        return wrapper
    return decorator


# =============================================================================
# PROGRESS TRACKER
# =============================================================================

class ProgressTracker:
    """
    Track progress of long-running operations.
    
    Logs at configurable intervals.
    """
    
    def __init__(
        self,
        total: int,
        logger: logging.Logger,
        desc: str = "Processing",
        log_every: int = 100,
    ):
        self.total = total
        self.logger = logger
        self.desc = desc
        self.log_every = log_every
        self.current = 0
        self.start_time = time.time()
        self.errors = 0
    
    def update(self, n: int = 1, error: bool = False) -> None:
        """Update progress."""
        self.current += n
        if error:
            self.errors += 1
        
        if self.current % self.log_every == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.current) / rate if rate > 0 else 0
            
            self.logger.info(
                f"{self.desc}: {self.current}/{self.total} ({100*self.current/self.total:.1f}%)",
                elapsed_sec=round(elapsed, 1),
                remaining_sec=round(remaining, 1),
                rate=round(rate, 2),
                errors=self.errors,
            )
    
    def finish(self) -> Dict:
        """Finish and return summary."""
        elapsed = time.time() - self.start_time
        return {
            'total': self.total,
            'processed': self.current,
            'errors': self.errors,
            'elapsed_sec': round(elapsed, 1),
            'rate': round(self.current / elapsed, 2) if elapsed > 0 else 0,
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Demo
    ctx = RunContext()
    logger = get_logger('demo', ctx, json_output=False)
    
    ctx.set_step('loading')
    logger.info("Starting pipeline", version="1.0.0")
    
    ctx.set_step('processing')
    logger.info("Processing files", count=766, type="orbis")
    
    ctx.set_step('scoring')
    logger.info("Scoring complete", matches=15000, tier_a=3000)
    
    print("\nTiming summary:", ctx.get_timing_summary())
