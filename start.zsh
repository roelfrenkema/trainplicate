#!/usr/bin/zsh

source venv/bin/activate

python trainplicate.py $1 $2 $3

deactivate
