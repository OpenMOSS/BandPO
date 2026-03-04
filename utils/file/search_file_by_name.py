# 快速使用
from file_searcher import AsyncFileSearcher

searcher = AsyncFileSearcher(max_workers=20)
results = searcher.search(
    root_dir="/remote-home1/yli/tmp",
    search_string="offline-run-202509",  # 搜索包含 "wandb" 的文件/文件夹
    search_mode="contains",  # 或 "equals", "starts", "ends", "regex"
    case_sensitive=False
)

for item in results:
    print(f"{item['type']}: {item['path']}")