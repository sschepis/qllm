"""
Command Line Interface package for QLLM.

This package provides the CLI functionality for the Quantum Resonance 
Language Model, including both traditional command-line arguments
and a menu-driven interface.
"""

# Import these in a specific order to avoid circular imports
from src.cli.user_interface import TerminalUI
from src.cli.menu_system import MenuSystem, Menu, MenuOption
from src.cli.config_wizard import ConfigWizard

# Import MenuHandler last since it depends on the others
# Note: We use a function to delay import until needed
def get_menu_handler():
    """Get a MenuHandler instance on demand to avoid circular imports."""
    from src.cli.menu_handlers import MenuHandler
    return MenuHandler()

__all__ = [
    'MenuSystem',
    'Menu',
    'MenuOption',
    'ConfigWizard',
    'TerminalUI',
    'get_menu_handler',
]