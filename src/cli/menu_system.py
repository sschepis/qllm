"""
Menu system for the QLLM CLI.

This module provides a menu system for the QLLM CLI, allowing users to navigate
through different options using arrow keys and execute commands.
"""

import os
import sys
from typing import Dict, Any, Optional, List, Callable, Union

from src.cli.user_interface import TerminalUI


class MenuOption:
    """Represents a single menu option."""
    
    def __init__(
        self,
        text: str,
        handler: Optional[Callable] = None,
        should_exit: bool = False,
        is_enabled: Optional[Callable[[], bool]] = None
    ):
        """
        Initialize a menu option.
        
        Args:
            text: Display text for the option
            handler: Function to call when option is selected
            should_exit: Whether selecting this option should exit the menu
            is_enabled: Function to determine if the option is enabled
        """
        self.text = text
        self.handler = handler
        self.should_exit = should_exit
        self.is_enabled = is_enabled or (lambda: True)


class Menu:
    """Represents a menu with multiple options."""
    
    def __init__(
        self,
        title: str,
        parent: Optional['Menu'] = None
    ):
        """
        Initialize a menu.
        
        Args:
            title: Menu title
            parent: Parent menu (for navigation)
        """
        self.title = title
        self.parent = parent
        self.options = []
        self.ui = TerminalUI()
    
    def add_option(self, option: MenuOption) -> None:
        """
        Add an option to the menu.
        
        Args:
            option: Menu option to add
        """
        self.options.append(option)
    
    def display(self) -> None:
        """Display the menu and get user selection using arrow key navigation."""
        while True:
            # Clear screen and display title
            self.ui.clear_screen()
            self.ui.print_banner()
            
            # Get enabled options
            enabled_options = [opt for opt in self.options if opt.is_enabled()]
            option_texts = []
            
            # Prepare option texts with proper coloring for display
            for option in self.options:
                if option.is_enabled():
                    if option.should_exit:
                        if self.parent:
                            # Back option
                            option_texts.append(option.text)
                        else:
                            # Exit option
                            option_texts.append(option.text)
                    else:
                        option_texts.append(option.text)
            
            # Display menu options with arrow key navigation
            selected = self.ui.menu(self.title, option_texts, "Navigate with arrow keys, press Enter to select:")
            
            if selected < 0 or selected >= len(enabled_options):
                continue
            
            # Get the selected option
            selected_option = enabled_options[selected]
            
            if selected_option.should_exit:
                return
            
            if selected_option.handler:
                # Show transition animation
                self.ui.spinner(0.3, f"Loading {selected_option.text}", fps=15)
                
                # Call handler and get result (which may be a submenu)
                result = selected_option.handler()
                
                # If handler returned a menu, display it
                if isinstance(result, Menu):
                    # Set parent menu
                    result.parent = self
                    
                    # Display submenu
                    result.display()
                elif result is not None:
                    # Wait for any key if the handler returned a value
                    self.ui.wait_for_any_key()


class MenuSystem:
    """Menu system for the CLI."""
    
    def __init__(self, main_menu: Menu):
        """
        Initialize the menu system.
        
        Args:
            main_menu: Main menu to display
        """
        self.main_menu = main_menu
        self.ui = TerminalUI()
    
    def run(self) -> None:
        """Run the menu system."""
        try:
            # Display main menu
            self.main_menu.display()
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            self.ui.clear_screen()
            self.ui.print_box(["Exiting QLLM CLI"], "GOODBYE", color="BRIGHT_MAGENTA")
            print()