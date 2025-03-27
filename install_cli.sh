#!/bin/bash
# Script to install the new menu-driven CLI

echo "Installing new menu-driven CLI for QLLM..."

# Make CLI executable
chmod +x cli.py
echo "Made cli.py executable"

# Create configs directory for storing configurations
mkdir -p configs
echo "Created configs directory for storing configurations"

# Remove old files (no backup since we don't need backward compatibility)
echo "Removing old files..."
rm -f main.py.new
rm -f src/cli/commands.py.new

# Create symlink to make cli.py accessible as 'qllm'
ln -sf $(pwd)/cli.py /usr/local/bin/qllm 2>/dev/null || echo "Could not create symlink. You may need sudo rights or can add the current directory to your PATH."

echo -e "\nInstallation completed!"
echo -e "You can now run the CLI using:\n"
echo -e "  ./cli.py\n"
echo -e "Or if the symlink was created successfully:\n"
echo -e "  qllm\n"