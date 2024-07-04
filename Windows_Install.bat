@echo off
chcp 65001

echo WARNING. 安装前需要你已在安装了python3.10.11或3.10.13 和C++生成工具
echo follow this tutorial : https://space.bilibili.com/483532108/

pip install requests -i https://pypi.org/simple
pip install tqdm -i https://pypi.org/simple
git lfs install

set "folder=IDM-VTON"
set "filename=app_VTON.py"
rem set http_proxy=http://127.0.0.1:58309
rem set https_proxy=https://127.0.0.1:58309

cd IDM-VTON

py --version >nul 2>&1
if "%ERRORLEVEL%" == "0" (
    echo 获取python版本，正在创建ven环境
    py -3.10 -m venv venv
) else (
    echo Python版本不正确 ，请安装python3.10
    python -m venv venv
)

call .\venv\Scripts\activate.bat
pip config set global.index-url https://pypi.org/simple/
pip config set install.trusted-host pypi.org
pip config set global.proxy http:127.0.0.1:58309

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