"""
DESOM Dashboard and Web Interface module.
"""

from .api import DashboardAPI
from .server import start_dashboard

__all__ = ["DashboardAPI", "start_dashboard"]
