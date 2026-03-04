#!/bin/bash

IP=$(hostname -I | awk '{print $1}')
RAY_IP=${IP}
RAY_dashboard_port="8265"
RAY_GCS_port="6379"
ray start --head --dashboard-host="${RAY_IP}" --dashboard-port="${RAY_dashboard_port}" --disable-usage-stats
export RAY_ADDRESS=${RAY_IP}:${RAY_GCS_port}
sleep 3s
ray status # 自检