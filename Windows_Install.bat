@echo off

echo WARNING. For this auto installer to work you need to have installed Python 3.10.11 or 3.10.13 and C++ tools 
echo follow this tutorial : https://space.bilibili.com/483532108/

pip install requests
pip install tqdm
git lfs install

git clone https://github.com/cynic1988/IDMVTON_CN

set "url=https://raw.githubusercontent.com/FurkanGozukara/Stable-Diffusion/main/CustomPythonScripts/app_VTON.py"
set "folder=IDM-VTON"
set "filename=app_VTON.py"

rem if not exist "%folder%" mkdir "%folder%"
rem powershell -Command "(New-Object System.Net.WebClient).DownloadFile('%url%', '%folder%\%filename%')"

cd IDM-VTON

py --version >nul 2>&1
if "%ERRORLEVEL%" == "0" (
    echo Python launcher is available. Generating Python 3.10 VENV
    py -3.10 -m venv venv
) else (
    echo Python launcher is not available, generating VENV with default Python. Make sure that it is 3.10
    python -m venv venv
)

call .\venv\Scripts\activate.bat

pip install tqdm

pip install -r requirements.txt

echo installing requirements 

pip install torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade
pip install https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl
pip install xformers==0.0.24
pip install bitsandbytes==0.43.0 --upgrade
pip install gradio

REM Show completion message
echo Virtual environment made and installed properly

REM Pause to keep the command prompt open
pause