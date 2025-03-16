@echo off
echo Creating virtual environment...
python -m venv .venv
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r requirements.txt
pip install -e .
echo Done.
pause