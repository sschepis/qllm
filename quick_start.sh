#!/bin/bash
# Quick start script for the QLLM Menu-Driven CLI
# With fancy colors and visual appeal

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[0;37m'
BOLD='\033[1m'
RESET='\033[0m'

# Function to print colored text
print_color() {
  color=$1
  text=$2
  echo -e "${!color}${text}${RESET}"
}

# Function to print success message
print_success() {
  echo -e "${GREEN}✓${RESET} $1"
}

# Function to print info message
print_info() {
  echo -e "${BLUE}ℹ${RESET} $1"
}

# Function to show a spinner
spin() {
  spinner="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
  message=$1
  while true; do
    for i in $(seq 0 9); do
      echo -ne "\r${CYAN}${spinner:$i:1}${RESET} $message"
      sleep 0.1
    done
  done
}

# Function to show a progress bar
progress_bar() {
  local duration=$1
  local message=$2
  local steps=20
  local sleep_time=$(echo "scale=3; $duration/$steps" | bc)
  
  echo -ne "${message} ["
  for i in $(seq 1 $steps); do
    echo -ne "${MAGENTA}#${RESET}"
    sleep $sleep_time
  done
  echo -e "] ${GREEN}Done!${RESET}"
}

# Clear screen and show banner
clear
echo
echo -e "${CYAN}${BOLD}╭─────╮  ╭────╮ ╭────╮ ╭─╮   ╭─╮${RESET}"
echo -e "${CYAN}${BOLD}│ ╭──╯  │ ╭╮ │ │ ╭╮ │ │ │   │ │${RESET}"
echo -e "${CYAN}${BOLD}│ │ ╭╮  │ ╰╯ │ │ ╰╯ │ │ │   │ │${RESET}"
echo -e "${CYAN}${BOLD}│ │ ││  │ ╭╮ │ │ ╭╮ │ │ │   │ │${RESET}"
echo -e "${CYAN}${BOLD}│ ╰─╯│  │ │ │ │ │ │ │ │ ╰───╯ │${RESET}"
echo -e "${CYAN}${BOLD}╰────╯  ╰─╯ ╰─╯ ╰─╯ ╰─╯ ╰─────╯${RESET}"
echo
echo -e "${MAGENTA}${BOLD}Quantum Resonance Language Model${RESET}"
echo
echo -e "${YELLOW}${BOLD}========================================${RESET}"
echo -e "${YELLOW}${BOLD}           Quick Start Script           ${RESET}"
echo -e "${YELLOW}${BOLD}========================================${RESET}"
echo

# Start spinner in the background
spin "Setting up QLLM environment" &
SPIN_PID=$!

# Make sure we kill the spinner when the script exits
trap "kill -9 $SPIN_PID &>/dev/null" EXIT

# Make CLI executable
chmod +x cli.py
sleep 0.5
kill -9 $SPIN_PID &>/dev/null
wait $SPIN_PID 2>/dev/null
print_success "Made CLI executable"

# Create configs directory if it doesn't exist
spin "Creating directory structure" &
SPIN_PID=$!
mkdir -p configs
sleep 0.5
kill -9 $SPIN_PID &>/dev/null
wait $SPIN_PID 2>/dev/null
print_success "Created configs directory"

# Create runs directory if it doesn't exist
spin "Preparing runs directory" &
SPIN_PID=$!
mkdir -p runs
sleep 0.5
kill -9 $SPIN_PID &>/dev/null
wait $SPIN_PID 2>/dev/null
print_success "Created runs directory"

# Create basic structure for the model if needed
echo -e "\n${BOLD}${BLUE}Checking required directories...${RESET}"
for dir in "src/config" "src/cli" "src/training"; do
  if [ ! -d "$dir" ]; then
    spin "Creating $dir" &
    SPIN_PID=$!
    mkdir -p "$dir"
    sleep 0.5
    kill -9 $SPIN_PID &>/dev/null
    wait $SPIN_PID 2>/dev/null
    print_success "Created $dir"
  else
    print_info "$dir already exists"
  fi
done

# Attempt to create a symlink for convenience
echo -e "\n${BOLD}${BLUE}Setting up command access...${RESET}"
if ln -sf $(pwd)/cli.py /usr/local/bin/qllm 2>/dev/null; then
  print_success "Created symlink: you can now use 'qllm' command from anywhere"
else
  print_info "Could not create symlink (may need admin privileges)"
  print_info "You can still run ./cli.py from this directory"
fi

# Final setup tasks
echo
print_info "Running final configuration..."
progress_bar 2 "Finalizing setup"

# Display completion banner
echo -e "\n${GREEN}${BOLD}✓ Setup complete!${RESET}"
echo -e "${CYAN}╭───────────────────────────────────────────╮${RESET}"
echo -e "${CYAN}│                                           │${RESET}"
echo -e "${CYAN}│  ${YELLOW}To start the QLLM CLI, run:${RESET}                ${CYAN}│${RESET}"
echo -e "${CYAN}│    ${GREEN}./cli.py${RESET}                                ${CYAN}│${RESET}"
echo -e "${CYAN}│                                           │${RESET}"
echo -e "${CYAN}│  ${YELLOW}For more information, see:${RESET}                ${CYAN}│${RESET}"
echo -e "${CYAN}│    ${GREEN}cli_manual.md${RESET}                           ${CYAN}│${RESET}"
echo -e "${CYAN}│                                           │${RESET}"
echo -e "${CYAN}│  ${YELLOW}To test the configuration system:${RESET}         ${CYAN}│${RESET}"
echo -e "${CYAN}│    ${GREEN}python test_config.py${RESET}                    ${CYAN}│${RESET}"
echo -e "${CYAN}│                                           │${RESET}"
echo -e "${CYAN}╰───────────────────────────────────────────╯${RESET}"
echo

# Show a random quantum fact
quantum_facts=(
  "Quantum entanglement allows particles to maintain instantaneous correlations across any distance."
  "Quantum superposition allows particles to exist in multiple states simultaneously."
  "In quantum mechanics, particles can tunnel through barriers that should be impenetrable."
  "Quantum resonance uses harmonic frequencies to enable efficient information transfer."
  "The quantum observer effect suggests that the act of observation changes quantum states."
  "Quantum prime patterns create efficient mathematical representations of complex data."
)
random_index=$((RANDOM % ${#quantum_facts[@]}))
echo -e "${MAGENTA}${BOLD}Did You Know?${RESET}"
echo -e "${CYAN}${quantum_facts[$random_index]}${RESET}"
echo