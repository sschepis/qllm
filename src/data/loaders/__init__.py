"""
Data loader implementations for QLLM.

This module provides various data loader implementations that extend the
BaseLoader class to handle loading data from different sources and formats.
"""

from src.data.loaders.wikitext_loader import WikitextLoader
from src.data.loaders.daily_dialog_loader import DailyDialogLoader
#from src.data.loaders.custom_loader import CustomLoader
#from src.data.loaders.dummy_loaders import DummyLoader


# Comment out loaders that don't exist yet
# from src.data.loaders.remote_loader import RemoteLoader

__all__ = [
    'WikitextLoader',
    'DailyDialogLoader',
    'CustomLoader',
    'DummyLoader',
    # 'RemoteLoader'
]