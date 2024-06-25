@echo off
chcp 65001

cd IDM-VTON

call .\venv\Scripts\activate.bat || exit /b
REM SET CUDA_VISIBLE_DEVICES=0  - this is used to set certain CUDA device visible only used
REM set TRANSFORMERS_OFFLINE=1
REM SET CUDA_VISIBLE_DEVICES=1
set PYTHONWARNINGS=ignore
echo 打开Gradio分享，请在启动末尾加上[--share SHARE]参数。
setlocal enabledelayedexpansion
if not exist batconfig.txt (
echo 请根据显存情况进行选择，最快的是FP16和全显存模式 {1+1}
echo "1+1 = ALL, 1+2 = 12 GB , 1+3 = 10 GB, 2+1 = 15 GB, 2+2 = 10.3 GB , 2+3 = 8.2 GB."
echo --------------------------------
echo 设置显卡选项-
echo 1-默认启用显卡全显存模式.
echo 2-启用CPU加载-显存低于6G请启用此项-速度会更慢.
set /p cpuoffload= "请输入显卡模式{1或2}"
echo --------------------------------
echo 请选择模型加载方式-
echo 1-FP16模式-精度更高-对显卡要求也更高.
echo 2-FP8模式-质量稍低的中等显存.
echo 3-FP4模式-质量更低但速度更快.
set /p precision= "请输入你的选择{1至3}"
echo --------------------------------
echo cpuoffload=!cpuoffload! > batconfig.txt
echo precision=!precision! >> batconfig.txt
) else (
	set "cpuoffload="
	set "precision="
	for /f "tokens=1* delims==" %%a in (batconfig.txt) do (
    set /a lineCount+=1
    if !lineCount!==1 (
        if "%%a"=="cpuoffload" (
            set "cpuoffload=%%b"
        )
    ) else if !lineCount!==2 (
        if "%%a"=="precision" (
            set "precision=%%b"
            goto displayValues
        )
    )
	:displayValues
	echo 获取配置文件成功
)
)

set "lowvram="
set "load_mode="

if "!cpuoffload!" == "2" (
    set "lowvram=--lowvram"
)

if "!precision!" == "2" (
    set "load_mode=--load_mode 8bit"
)

if "!precision!" == "3" (
    set "load_mode=--load_mode 4bit"
)
echo 正在启动MojieVTON...
python app_VTON.py !lowvram! !load_mode! 
endlocal
pause
