#!/usr/bin/zsh

# Check if the venv directory exists
if [ ! -d "venv" ]; then
    echo "Virtual environment directory does not exist. Creating it now and configuring settings."
    # Create the virtual environment
    python3 -m venv venv
    echo "Virtual environment created."
    source venv/bin/activate
    echo "Installing requirements now."
    pip install -r requirements.txt    
    echo "Finished you should now be able to run trainplicate with"
    echo "start.sh <dirname> <token> <descritption>"
else
    echo "There is an existing venv. If you want to reinstall remove the venv directory."
fi
