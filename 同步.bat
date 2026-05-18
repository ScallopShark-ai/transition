@echo off
chcp 65001 >nul
set CURRENT_TIME=%date% %time:~0,8%
echo 拉取最新代码
git pull origin main
echo 添加文件到暂存区
git add .
echo 正在提交代码
git commit -m "%日期+时间%"
echo 推送到远程仓库
git push origin main
pause