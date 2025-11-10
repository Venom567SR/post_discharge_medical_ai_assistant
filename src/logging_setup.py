import sys
from loguru import logger
from src.config import LOG_FILE, LOG_ROTATION, LOG_RETENTION, LOG_LEVEL

def setup_logging():
    """Configure loguru with file rotation and formatting."""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with colored output
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=LOG_LEVEL,
        colorize=True
    )
    
    # Add file handler with rotation
    logger.add(
        LOG_FILE,
        rotation=LOG_ROTATION,
        retention=LOG_RETENTION,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=LOG_LEVEL,
        enqueue=True  # Thread-safe
    )
    
    logger.info("Logging system initialized")
    return logger

def log_agent_action(user_id: str, agent: str, action: str, details: dict = None):
    """Log agent actions with structured data."""
    log_data = {
        "user_id": user_id,
        "agent": agent,
        "action": action,
        "details": details or {}
    }
    logger.info(f"Agent action: {log_data}")

def log_tool_call(user_id: str, tool: str, params: dict, result: str):
    """Log tool invocations."""
    log_data = {
        "user_id": user_id,
        "tool": tool,
        "params": params,
        "result": result[:200] + "..." if len(result) > 200 else result
    }
    logger.info(f"Tool call: {log_data}")

def log_retrieval(user_id: str, query: str, results_count: int, top_scores: list):
    """Log RAG retrieval results."""
    log_data = {
        "user_id": user_id,
        "query": query[:100],
        "results_count": results_count,
        "top_scores": top_scores
    }
    logger.info(f"RAG retrieval: {log_data}")

def log_handoff(user_id: str, from_agent: str, to_agent: str, reason: str):
    """Log agent handoffs."""
    log_data = {
        "user_id": user_id,
        "from": from_agent,
        "to": to_agent,
        "reason": reason
    }
    logger.info(f"Agent handoff: {log_data}")

def log_error(user_id: str, error_type: str, error_msg: str, context: dict = None):
    """Log errors with context."""
    log_data = {
        "user_id": user_id,
        "error_type": error_type,
        "error_msg": error_msg,
        "context": context or {}
    }
    logger.error(f"Error occurred: {log_data}")

def log_llm_call(user_id: str, model: str, prompt_length: int, response_length: int, latency_ms: float):
    """Log LLM API calls."""
    log_data = {
        "user_id": user_id,
        "model": model,
        "prompt_length": prompt_length,
        "response_length": response_length,
        "latency_ms": latency_ms
    }
    logger.info(f"LLM call: {log_data}")

def get_recent_logs(n: int = 100) -> list[str]:
    """Read the last N lines from the log file."""
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return lines[-n:] if n < len(lines) else lines
    except FileNotFoundError:
        return ["Log file not found"]
    except Exception as e:
        return [f"Error reading logs: {str(e)}"]

# Initialize logging on import
setup_logging()
