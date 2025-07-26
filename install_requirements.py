import subprocess
import sys

def install_requirements():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def run_initial_steps():
    # Add your initial manual steps here
    print("Running initial steps...")
    # TODO: Here run initial steps for Ollama.

def main():
    print("Installing required packages...")
    install_requirements()
    
    print("Running initial setup steps...")
    run_initial_steps()
    
    print("Setup complete. Running main script...")
    # Add code to run your main script here

if __name__ == "__main__":
    main()