@echo off
chcp 65001 >nul
cd /d "%~dp0"

set CURRENT_TIME=%date% %time:~0,8%

echo ==================================================
echo   🚀 开始自动同步 GitHub 代码
echo   🕒 启动时间: %CURRENT_TIME%
echo ==================================================
echo.

echo [1/4] ⬇️ 正在拉取远程最新代码...
:: 使用 --progress 强制 Git 打印下载进度
git pull origin main --progress
echo.

echo [2/4] 📦 正在扫描并添加变更文件...
git add -A
echo --------------------------------
echo   本次检测到以下文件变动：
:: -s 参数表示 short（简短输出），只列出发生变化的文件名单
git status -s
echo --------------------------------
echo.

echo [3/4] 💾 正在打包提交...
git commit -m "Auto Update: %CURRENT_TIME%"
echo.

echo [4/4] ☁️ 正在推送到 GitHub... (受网络影响可能需要几秒到几十秒，请耐心等待)
:: 使用 --progress 强制 Git 打印上传进度
git push origin main --progress
echo.

echo ==================================================
echo   ✅ 全部同步完成！
echo ==================================================
pause
