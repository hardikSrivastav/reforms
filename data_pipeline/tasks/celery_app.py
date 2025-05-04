"""
Celery application configuration.
"""

import os
from celery import Celery
from kombu import Queue, Exchange

# Load environment variables if needed
from dotenv import load_dotenv
load_dotenv()

# Define Redis URL from environment or use default
redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

# Create Celery instance
app = Celery(
    'data_pipeline',
    broker=redis_url,
    backend=redis_url,
    include=[
        'data_pipeline.tasks.analysis_tasks',
        'data_pipeline.tasks.embedding_tasks'
    ]
)

# Configure Celery
app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Define task queues with priorities
    task_queues=(
        Queue('high', Exchange('high'), routing_key='high', queue_arguments={'x-max-priority': 10}),
        Queue('default', Exchange('default'), routing_key='default', queue_arguments={'x-max-priority': 5}),
        Queue('low', Exchange('low'), routing_key='low', queue_arguments={'x-max-priority': 1}),
    ),
    
    # Default queue and exchange settings
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
    
    # Task result settings
    task_ignore_result=False,
    task_track_started=True,
    result_expires=3600 * 24 * 7,  # 7 days
    
    # Prefetch settings for better resource management
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,

    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_acks_on_failure_or_timeout=False,
    
    # Rate limiting
    task_default_rate_limit='100/m',
)

# Example configuration for task routing based on task name
app.conf.task_routes = {
    'data_pipeline.tasks.analysis_tasks.run_base_analysis': {'queue': 'high'},
    'data_pipeline.tasks.analysis_tasks.run_metric_analysis': {'queue': 'default'},
    'data_pipeline.tasks.analysis_tasks.run_cross_metric_analysis': {'queue': 'low'},
    'data_pipeline.tasks.embedding_tasks.*': {'queue': 'low'},
}


if __name__ == '__main__':
    app.start() 