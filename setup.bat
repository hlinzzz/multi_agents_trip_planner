@echo off
REM 使用 Python 3.12 安装依赖（系统自带 3.13 有依赖冲突）
echo 安装依赖到 E:\python ...
E:\python\python.exe -m pip install -r requirements.txt

echo.
echo 安装完成。
echo.
echo 下一步：复制 .env.example 为 .env 并填入真实凭证。
pause
