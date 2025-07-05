from celery import Celery
import os

# Constants
DEFAULT_REDIS_URL = "redis://localhost:6379"

# Get Redis URL from environment
redis_url = os.getenv("REDIS_URL", DEFAULT_REDIS_URL)

# Initialize Celery app
celery = Celery(
    "uni_chatbot_worker",
    broker=redis_url,
    backend=redis_url,
    include=['tasks']
)

# Configure Celery
celery.conf.update(
    broker_url=redis_url,
    result_backend=redis_url,
    broker_connection_retry_on_startup=True,
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    # Task timeout settings - increased for larger chunks
    task_soft_time_limit=900,  # 15 minutes soft limit (increased from 5 min)
    task_time_limit=1800,      # 30 minutes hard limit (increased from 10 min)
    # Result backend settings
    result_expires=3600,       # Results expire after 1 hour
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50
)

@celery.task
def test_task():
    """
    Simple test task to verify worker connectivity.
    
    Returns:
        str: Success message
    """
    return "Worker is running!"
