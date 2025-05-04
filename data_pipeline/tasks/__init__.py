"""
Task queue package for background processing.
This package contains Celery tasks for the data pipeline.
"""

from .celery_app import app as celery_app

__all__ = ['celery_app'] 