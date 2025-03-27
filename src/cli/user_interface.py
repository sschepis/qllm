"""
Terminal user interface utilities for the QLLM CLI.

This module provides terminal UI components for the menu-driven CLI,
including colored output, progress bars, arrow key navigation, and other visual elements.
"""

import os
import sys
import time
import shutil
import readchar
import colorama
from typing import List, Dict, Any, Optional, Union, Callable


# Initialize colorama for cross-platform color support
colorama.init()


class TerminalUI:
    """Terminal user interface utilities."""
    
    # ANSI color codes
    COLORS = {
        'BLACK': '\033[30m',
        'RED': '\033[31m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'MAGENTA': '\033[35m',
        'CYAN': '\033[36m',
        'WHITE': '\033[37m',
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'REVERSED': '\033[7m',
        
        # Bright colors
        'BRIGHT_BLACK': '\033[90m',
        'BRIGHT_RED': '\033[91m',
        'BRIGHT_GREEN': '\033[92m',
        'BRIGHT_YELLOW': '\033[93m',
        'BRIGHT_BLUE': '\033[94m',
        'BRIGHT_MAGENTA': '\033[95m',
        'BRIGHT_CYAN': '\033[96m',
        'BRIGHT_WHITE': '\033[97m',
        
        # Background colors
        'BG_BLACK': '\033[40m',
        'BG_RED': '\033[41m',
        'BG_GREEN': '\033[42m',
        'BG_YELLOW': '\033[43m',
        'BG_BLUE': '\033[44m',
        'BG_MAGENTA': '\033[45m',
        'BG_CYAN': '\033[46m',
        'BG_WHITE': '\033[47m',
    }
    
    # Key codes
    KEYS = {
        'UP': readchar.key.UP,
        'DOWN': readchar.key.DOWN,
        'LEFT': readchar.key.LEFT,
        'RIGHT': readchar.key.RIGHT,
        'ENTER': readchar.key.ENTER,
        'BACKSPACE': readchar.key.BACKSPACE,
        'SPACE': ' ',
        'ESC': readchar.key.ESC,
        'TAB': readchar.key.TAB,
        'CTRL_C': readchar.key.CTRL_C
    }
    
    # Box drawing characters
    BOX = {
        'top_left': '╭',
        'top_right': '╮',
        'bottom_left': '╰',
        'bottom_right': '╯',
        'horizontal': '─',
        'vertical': '│',
        'left_t': '├',
        'right_t': '┤',
        'top_t': '┬',
        'bottom_t': '┴',
        'cross': '┼',
    }
    
    def __init__(self):
        """Initialize the terminal UI."""
        self.terminal_width, self.terminal_height = self._get_terminal_size()
    
    def _get_terminal_size(self) -> tuple:
        """Get the terminal size."""
        try:
            columns, lines = shutil.get_terminal_size()
            return columns, lines
        except:
            return 80, 24
    
    def clear_screen(self) -> None:
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_colored(self, text: str, color: str = 'RESET', bold: bool = False, end: str = '\n') -> None:
        """
        Print colored text to the terminal.
        
        Args:
            text: Text to print
            color: Color to use (from self.COLORS)
            bold: Whether to make the text bold
            end: String to print at the end (default: newline)
        """
        if color.upper() not in self.COLORS:
            color = 'RESET'
        
        color_code = self.COLORS[color.upper()]
        bold_code = self.COLORS['BOLD'] if bold else ''
        reset_code = self.COLORS['RESET']
        
        print(f"{color_code}{bold_code}{text}{reset_code}", end=end)
    
    def print_header(self, text: str) -> None:
        """
        Print a section header.
        
        Args:
            text: Header text
        """
        width = min(self.terminal_width, 80)
        
        print()
        self.print_colored("=" * width, 'BRIGHT_CYAN', bold=True)
        self.print_colored(f" {text.upper()} ", 'BRIGHT_CYAN', bold=True)
        self.print_colored("=" * width, 'BRIGHT_CYAN', bold=True)
        print()
    
    def print_section(self, text: str) -> None:
        """
        Print a subsection header.
        
        Args:
            text: Section text
        """
        print()
        self.print_colored(f"▓▒░ {text} ░▒▓", 'BRIGHT_MAGENTA', bold=True)
        print()
    
    def print_info(self, text: str) -> None:
        """
        Print information text.
        
        Args:
            text: Information text
        """
        self.print_colored(f"  {text}", 'BRIGHT_WHITE')
    
    def print_success(self, text: str) -> None:
        """
        Print success text.
        
        Args:
            text: Success text
        """
        self.print_colored(f"✓ {text}", 'BRIGHT_GREEN', bold=True)
    
    def print_warning(self, text: str) -> None:
        """
        Print warning text.
        
        Args:
            text: Warning text
        """
        self.print_colored(f"⚠ {text}", 'BRIGHT_YELLOW', bold=True)
    
    def print_error(self, text: str) -> None:
        """
        Print error text.
        
        Args:
            text: Error text
        """
        self.print_colored(f"✗ {text}", 'BRIGHT_RED', bold=True)
    
    def print_box(self, text: List[str], title: Optional[str] = None, color: str = 'BRIGHT_CYAN') -> None:
        """
        Print text inside a box.
        
        Args:
            text: List of lines to print inside the box
            title: Optional title for the box
            color: Color to use for the box
        """
        # Calculate box dimensions
        max_line_length = max(len(line) for line in text)
        if title and len(title) + 4 > max_line_length:
            max_line_length = len(title) + 4
        
        width = max_line_length + 4
        
        # Print top border with title if provided
        if title:
            title_len = len(title)
            left_padding = (width - title_len - 2) // 2
            right_padding = width - title_len - 2 - left_padding
            
            self.print_colored(
                f"{self.BOX['top_left']}{self.BOX['horizontal'] * left_padding}"
                f" {title} "
                f"{self.BOX['horizontal'] * right_padding}{self.BOX['top_right']}",
                color
            )
        else:
            self.print_colored(
                f"{self.BOX['top_left']}{self.BOX['horizontal'] * (width - 2)}{self.BOX['top_right']}",
                color
            )
        
        # Print content
        for line in text:
            padding = width - len(line) - 4
            self.print_colored(
                f"{self.BOX['vertical']}  {line}{' ' * padding}  {self.BOX['vertical']}",
                color
            )
        
        # Print bottom border
        self.print_colored(
            f"{self.BOX['bottom_left']}{self.BOX['horizontal'] * (width - 2)}{self.BOX['bottom_right']}",
            color
        )
    
    def print_banner(self) -> None:
        """Print a fancy banner for the QLLM CLI."""
        banner = [
            "╭─────╮  ╭────╮ ╭────╮ ╭─╮   ╭─╮",
            "│ ╭──╯  │ ╭╮ │ │ ╭╮ │ │ │   │ │",
            "│ │ ╭╮  │ ╰╯ │ │ ╰╯ │ │ │   │ │",
            "│ │ ││  │ ╭╮ │ │ ╭╮ │ │ │   │ │",
            "│ ╰─╯│  │ │ │ │ │ │ │ │ ╰───╯ │",
            "╰────╯  ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─────╯",
            "",
            "Quantum Resonance Language Model"
        ]
        
        print()
        for line in banner:
            if "Quantum" in line:
                self.print_colored(line, 'BRIGHT_MAGENTA', bold=True)
            else:
                self.print_colored(line, 'BRIGHT_CYAN', bold=True)
        print()
    
    def print_progress_bar(self, progress: float, width: int = 40, prefix: str = '', suffix: str = '') -> None:
        """
        Print a progress bar.
        
        Args:
            progress: Progress value between 0 and 1
            width: Width of the progress bar
            prefix: Text to display before the progress bar
            suffix: Text to display after the progress bar
        """
        progress = max(0, min(1, progress))
        filled_length = int(width * progress)
        bar = '█' * filled_length + '░' * (width - filled_length)
        
        self.print_colored(f'\r{prefix} |{bar}| {int(progress * 100)}% {suffix}', 'BRIGHT_GREEN', end='')
        sys.stdout.flush()
        
        if progress == 1:
            print()
    
    def spinner(self, seconds: float, message: str = 'Processing', fps: int = 10) -> None:
        """
        Display a spinner animation for the specified number of seconds.
        
        Args:
            seconds: How many seconds to display the spinner
            message: Message to display with the spinner
            fps: Frames per second for the spinner
        """
        spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        iterations = int(seconds * fps)
        
        for i in range(iterations):
            char = spinner_chars[i % len(spinner_chars)]
            self.print_colored(f'\r{char} {message}...', 'BRIGHT_CYAN', end='')
            sys.stdout.flush()
            time.sleep(1/fps)
        
        print()
    
    def wait_for_any_key(self, message: str = 'Press any key to continue...') -> str:
        """
        Wait for user to press any key.
        
        Args:
            message: Message to display
            
        Returns:
            The key that was pressed
        """
        self.print_colored(f"\n{message}", 'BRIGHT_WHITE', end='')
        sys.stdout.flush()
        key = readchar.readkey()
        print()
        return key
    
    def prompt(self, prompt_text: str, default: str = '') -> str:
        """
        Prompt for text input with arrow key navigation.
        
        Args:
            prompt_text: Prompt text
            default: Default value
            
        Returns:
            User input
        """
        # Fix for the NoneType error - convert None to empty string
        if default is None:
            default = ''
            
        if default:
            self.print_colored(f"{prompt_text} [{default}]: ", 'BRIGHT_YELLOW', end='')
        else:
            self.print_colored(f"{prompt_text}: ", 'BRIGHT_YELLOW', end='')
        
        # Initialize input buffer
        buffer = list(default)  # This was causing the error with None
        cursor_pos = len(buffer)
        
        # Keep track of current line position
        prompt_length = len(prompt_text) + 4 + len(default)
        current_line_pos = prompt_length
        
        # Display initial buffer
        print(''.join(buffer), end='')
        sys.stdout.flush()
        
        while True:
            key = readchar.readkey()
            
            if key == self.KEYS['ENTER']:
                print()  # Move to next line
                return ''.join(buffer)
            
            elif key == self.KEYS['BACKSPACE']:
                if cursor_pos > 0:
                    buffer.pop(cursor_pos - 1)
                    cursor_pos -= 1
                    
                    # Redraw from cursor position
                    print('\r' + ' ' * current_line_pos, end='\r')
                    if default:
                        self.print_colored(f"{prompt_text} [{default}]: ", 'BRIGHT_YELLOW', end='')
                    else:
                        self.print_colored(f"{prompt_text}: ", 'BRIGHT_YELLOW', end='')
                    print(''.join(buffer), end='')
                    
                    current_line_pos = prompt_length + len(buffer)
            
            elif key == self.KEYS['LEFT']:
                if cursor_pos > 0:
                    cursor_pos -= 1
                    # Move cursor left
                    print('\033[D', end='')
                    sys.stdout.flush()
            
            elif key == self.KEYS['RIGHT']:
                if cursor_pos < len(buffer):
                    cursor_pos += 1
                    # Move cursor right
                    print('\033[C', end='')
                    sys.stdout.flush()
            
            elif key == self.KEYS['CTRL_C']:
                print('^C')
                raise KeyboardInterrupt()
            
            elif len(key) == 1 and ord(key) >= 32:  # Printable character
                buffer.insert(cursor_pos, key)
                cursor_pos += 1
                
                # Redraw from cursor position
                print('\r' + ' ' * current_line_pos, end='\r')
                if default:
                    self.print_colored(f"{prompt_text} [{default}]: ", 'BRIGHT_YELLOW', end='')
                else:
                    self.print_colored(f"{prompt_text}: ", 'BRIGHT_YELLOW', end='')
                print(''.join(buffer), end='')
                
                current_line_pos = prompt_length + len(buffer)
            
            sys.stdout.flush()
    
    def prompt_int(self, prompt_text: str, default: int = 0, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
        """
        Prompt for integer input.
        
        Args:
            prompt_text: Prompt text
            default: Default value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            User input as integer
        """
        while True:
            range_info = ""
            if min_value is not None and max_value is not None:
                range_info = f" ({min_value}-{max_value})"
            elif min_value is not None:
                range_info = f" (min: {min_value})"
            elif max_value is not None:
                range_info = f" (max: {max_value})"
                
            user_input = self.prompt(f"{prompt_text}{range_info}", str(default))
            
            if not user_input:
                value = default
            else:
                try:
                    value = int(user_input)
                except ValueError:
                    self.print_error("Please enter a valid integer.")
                    continue
            
            if min_value is not None and value < min_value:
                self.print_error(f"Value must be at least {min_value}.")
                continue
                
            if max_value is not None and value > max_value:
                self.print_error(f"Value must be at most {max_value}.")
                continue
                
            return value
    
    def prompt_float(self, prompt_text: str, default: float = 0.0, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
        """
        Prompt for float input.
        
        Args:
            prompt_text: Prompt text
            default: Default value
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            User input as float
        """
        while True:
            range_info = ""
            if min_value is not None and max_value is not None:
                range_info = f" ({min_value}-{max_value})"
            elif min_value is not None:
                range_info = f" (min: {min_value})"
            elif max_value is not None:
                range_info = f" (max: {max_value})"
                
            user_input = self.prompt(f"{prompt_text}{range_info}", str(default))
            
            if not user_input:
                value = default
            else:
                try:
                    value = float(user_input)
                except ValueError:
                    self.print_error("Please enter a valid number.")
                    continue
            
            if min_value is not None and value < min_value:
                self.print_error(f"Value must be at least {min_value}.")
                continue
                
            if max_value is not None and value > max_value:
                self.print_error(f"Value must be at most {max_value}.")
                continue
                
            return value
    
    def prompt_bool(self, prompt_text: str, default: bool = False) -> bool:
        """
        Prompt for boolean input with arrow key navigation for Yes/No options.
        
        Args:
            prompt_text: Prompt text
            default: Default value
            
        Returns:
            User input as boolean
        """
        default_str = "Y/n" if default else "y/N"
        options = ["Yes", "No"]
        selected = 0 if default else 1
        
        # Print prompt
        self.print_colored(f"{prompt_text} [{default_str}]: ", 'BRIGHT_YELLOW', end='')
        print()
        
        # Display options with cursor
        while True:
            for i, option in enumerate(options):
                if i == selected:
                    # Selected option
                    self.print_colored(f"  > {option}", 'BRIGHT_CYAN', end='\n')
                else:
                    # Unselected option
                    self.print_colored(f"    {option}", 'BRIGHT_WHITE', end='\n')
            
            # Move cursor back up
            print(f"\033[{len(options)}A", end='')
            
            # Get key press
            key = readchar.readkey()
            
            if key == self.KEYS['ENTER']:
                # Accept selection
                print(f"\033[{len(options)}B")  # Move cursor down
                return selected == 0
            
            elif key == self.KEYS['UP'] and selected > 0:
                selected -= 1
            
            elif key == self.KEYS['DOWN'] and selected < len(options) - 1:
                selected += 1
            
            elif key.lower() == 'y':
                print(f"\033[{len(options)}B")  # Move cursor down
                return True
            
            elif key.lower() == 'n':
                print(f"\033[{len(options)}B")  # Move cursor down
                return False
            
            elif key == self.KEYS['CTRL_C']:
                print(f"\033[{len(options)}B")  # Move cursor down
                print('^C')
                raise KeyboardInterrupt()
    
    def prompt_choice(self, prompt_text: str, choices: List[str], default: int = 0) -> int:
        """
        Prompt for a choice from a list of options with arrow key navigation.
        
        Args:
            prompt_text: Prompt text
            choices: List of choices
            default: Default choice index
            
        Returns:
            Selected choice index
        """
        if not choices:
            return -1
        
        # Set initial selection
        selected = default
        
        # Clear screen and print prompt
        print()
        self.print_colored(prompt_text, 'BRIGHT_YELLOW')
        print()
        
        # Display choices with selection indicator
        while True:
            for i, choice in enumerate(choices):
                prefix = f"  {i+1}. "
                if i == selected:
                    self.print_colored(f"> {prefix}{choice}", 'BRIGHT_CYAN', bold=True)
                else:
                    self.print_colored(f"  {prefix}{choice}", 'BRIGHT_WHITE')
            
            # Move cursor back up to first option
            print(f"\033[{len(choices)}A", end='')
            
            # Get key press
            key = readchar.readkey()
            
            if key == self.KEYS['ENTER']:
                # Accept selection
                print(f"\033[{len(choices)}B")  # Move cursor down
                return selected
            
            elif key == self.KEYS['UP'] and selected > 0:
                selected -= 1
            
            elif key == self.KEYS['DOWN'] and selected < len(choices) - 1:
                selected += 1
            
            elif key.isdigit():
                # Try to select by number
                num = int(key) - 1
                if 0 <= num < len(choices):
                    selected = num
            
            elif key == self.KEYS['CTRL_C']:
                print(f"\033[{len(choices)}B")  # Move cursor down
                print('^C')
                raise KeyboardInterrupt()
    
    def prompt_list(self, prompt_text: str, default: Optional[List[Any]] = None, item_type: str = "str") -> List[Any]:
        """
        Prompt for a list of items.
        
        Args:
            prompt_text: Prompt text
            default: Default list value
            item_type: Type of items ("str", "int", "float")
            
        Returns:
            List of items
        """
        # Handle None default value
        if default is None:
            default = []
            
        default_str = ", ".join(map(str, default))
        
        user_input = self.prompt(f"{prompt_text}", default_str)
        
        if not user_input:
            return default
        
        items = [item.strip() for item in user_input.split(",")]
        
        if item_type == "int":
            try:
                return [int(item) for item in items if item]
            except ValueError:
                self.print_error("Invalid input. Using default.")
                return default
        elif item_type == "float":
            try:
                return [float(item) for item in items if item]
            except ValueError:
                self.print_error("Invalid input. Using default.")
                return default
        else:
            return [item for item in items if item]
    
    def menu(self, title: str, options: List[str], prompt_text: str = "Select an option:") -> int:
        """
        Display a menu with arrow key navigation.
        
        Args:
            title: Menu title
            options: List of menu options
            prompt_text: Prompt text
            
        Returns:
            Selected option index
        """
        if not options:
            return -1
        
        # Print header
        self.print_box([title], color='BRIGHT_MAGENTA')
        print()
        
        # Set initial selection
        selected = 0
        
        # Display the menu once before entering the loop
        menu_height = len(options) + 2  # Add 2 for the prompt and blank line
        
        # First display of prompt and options
        self.print_colored(prompt_text, 'BRIGHT_YELLOW')
        print()
        
        for i, option in enumerate(options):
            if i == selected:
                self.print_colored(f"> {option}", 'BRIGHT_CYAN', bold=True)
            else:
                self.print_colored(f"  {option}", 'BRIGHT_WHITE')
        
        # Main interaction loop
        while True:
            # Get key press
            key = readchar.readkey()
            
            if key == self.KEYS['ENTER']:
                # Accept selection and clean up
                return selected
            
            elif key == self.KEYS['UP'] and selected > 0:
                selected -= 1
            
            elif key == self.KEYS['DOWN'] and selected < len(options) - 1:
                selected += 1
            
            elif key.isdigit():
                # Try to select by number
                num = int(key) - 1
                if 0 <= num < len(options):
                    selected = num
            
            elif key == self.KEYS['CTRL_C']:
                print('^C')
                raise KeyboardInterrupt()
            
            # Clear the menu area and redraw
            # Move to the top of the menu area
            print(f"\033[{menu_height}A", end='')
            
            # Clear each line
            for _ in range(menu_height):
                print(" " * self.terminal_width, end='\r')
                print("\033[1B", end='')  # Move down one line
            
            # Move back to the top
            print(f"\033[{menu_height}A", end='')
            
            # Redraw the menu
            self.print_colored(prompt_text, 'BRIGHT_YELLOW')
            print()
            
            for i, option in enumerate(options):
                if i == selected:
                    self.print_colored(f"> {option}", 'BRIGHT_CYAN', bold=True)
                else:
                    self.print_colored(f"  {option}", 'BRIGHT_WHITE')
    
    def print_values_table(self, data: Dict[str, Any], title: Optional[str] = None) -> None:
        """
        Print a table of key-value pairs.
        
        Args:
            data: Dictionary of data to display
            title: Optional title for the table
        """
        # Find the widest key
        key_width = max(len(key) for key in data.keys())
        
        # Print title if provided
        if title:
            self.print_section(title)
        
        # Print horizontal line
        line_width = key_width + 20
        self.print_colored("┌" + "─" * (key_width + 2) + "┬" + "─" * 17 + "┐", 'BRIGHT_BLUE')
        
        # Print each key-value pair
        for key, value in data.items():
            key_str = f" {key}" + " " * (key_width + 1 - len(key))
            
            # Format value based on type
            if isinstance(value, bool):
                value_str = f" {self.COLORS['BRIGHT_GREEN'] if value else self.COLORS['BRIGHT_RED']}"
                value_str += "✓" if value else "✗"
                value_str += f"{self.COLORS['BRIGHT_BLUE']} "
            elif isinstance(value, (int, float)):
                value_str = f" {self.COLORS['BRIGHT_CYAN']}{value}{self.COLORS['BRIGHT_BLUE']} "
            else:
                value_str = f" {value} "
            
            # Pad value string
            value_str = value_str + " " * (15 - len(str(value)))
            
            self.print_colored(f"│{key_str}│{value_str}│", 'BRIGHT_BLUE')
        
        # Print horizontal line
        self.print_colored("└" + "─" * (key_width + 2) + "┴" + "─" * 17 + "┘", 'BRIGHT_BLUE')
    
    def animate_text(self, text: str, delay: float = 0.03, color: str = 'BRIGHT_CYAN') -> None:
        """
        Print text with a typing animation.
        
        Args:
            text: Text to animate
            delay: Delay between characters in seconds
            color: Color to use for the text
        """
        for char in text:
            self.print_colored(char, color, end='')
            sys.stdout.flush()
            time.sleep(delay)
        print()  # End with newline
    
    def show_loading_screen(self, message: str = "Loading") -> None:
        """
        Show a loading screen.
        
        Args:
            message: Message to display
        """
        self.clear_screen()
        self.print_header(message)
        
        for i in range(3):
            self.print_colored(f"{message}{'.' * (i+1)}", 'BRIGHT_CYAN')
            time.sleep(0.3)
            self.clear_screen()
            self.print_header(message)
    
    def celebrate(self, message: str = "Success!") -> None:
        """
        Show a celebration animation.
        
        Args:
            message: Message to display
        """
        celebration = [
            "      \\o/      ",
            "       |       ",
            "      / \\      "
        ]
        
        colors = ['BRIGHT_RED', 'BRIGHT_YELLOW', 'BRIGHT_GREEN', 'BRIGHT_CYAN', 'BRIGHT_MAGENTA']
        
        for _ in range(3):
            for color in colors:
                self.clear_screen()
                self.print_colored(message, color, bold=True)
                print()
                
                for line in celebration:
                    self.print_colored(line, color, bold=True)
                
                time.sleep(0.1)