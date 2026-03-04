ss -tulpn
# -t：TCP
# -u：UDP
# -l：LISTEN（监听中的端口）
# -p：显示进程
# -n：不反向解析服务名（直接显示端口号）

lsof -i -P -n | grep LISTEN
# COMMAND：进程名
# PID：进程 ID
# USER：所属用户
# NAME：IP:端口
