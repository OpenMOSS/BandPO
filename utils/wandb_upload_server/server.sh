HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
python server.py --host $HOST --port 8000