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
    # Task timeout settings
    task_soft_time_limit=300,  # 5 minutes soft limit
    task_time_limit=600,       # 10 minutes hard limit
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
