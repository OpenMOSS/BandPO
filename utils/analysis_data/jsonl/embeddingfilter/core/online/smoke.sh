BASE_URL="http://10.176.59.108:8008"
curl -s "${BASE_URL}/health" | jq .

# 深度检查：服务内部会调用一次 smoke_test()
curl -s "${BASE_URL}/health?deep=true" | jq .
# 或者兼容旧用法：
curl -s "${BASE_URL}/smoke" | jq .
