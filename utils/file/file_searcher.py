import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Optional
from queue import Queue
import threading

class MultiThreadFileSearcher:
    """多线程文件和文件夹搜索器"""
    
    def __init__(self, max_workers: int = 10):
        """
        初始化搜索器
        
        Args:
            max_workers: 最大线程数，默认10
        """
        self.max_workers = max_workers
        self.results = []
        self.results_lock = threading.Lock()
        
    def search_directory(self, 
                        directory: str, 
                        search_string: str, 
                        search_mode: str = 'contains',
                        case_sensitive: bool = False) -> List[Tuple[str, str]]:
        """
        在指定目录下搜索文件和文件夹
        
        Args:
            directory: 要搜索的根目录
            search_string: 要搜索的字符串
            search_mode: 搜索模式 - 'contains'(包含), 'equals'(完全匹配), 'starts'(开头), 'ends'(结尾)
            case_sensitive: 是否区分大小写
            
        Returns:
            包含 (路径, 类型) 元组的列表
        """
        self.results = []
        self.search_string = search_string if case_sensitive else search_string.lower()
        self.case_sensitive = case_sensitive
        self.search_mode = search_mode
        
        # 获取所有子目录
        all_dirs = self._get_all_subdirectories(directory)
        all_dirs.insert(0, directory)  # 添加根目录
        
        # 使用线程池处理每个目录
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for dir_path in all_dirs:
                future = executor.submit(self._search_in_directory, dir_path)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"搜索过程中出错: {e}")
        
        return self.results
    
    def _get_all_subdirectories(self, directory: str) -> List[str]:
        """获取所有子目录（用于分配给不同线程）"""
        subdirs = []
        try:
            for root, dirs, _ in os.walk(directory):
                for d in dirs:
                    subdirs.append(os.path.join(root, d))
        except PermissionError:
            pass
        return subdirs
    
    def _match_name(self, name: str) -> bool:
        """检查名称是否匹配搜索条件"""
        test_name = name if self.case_sensitive else name.lower()
        
        if self.search_mode == 'contains':
            return self.search_string in test_name
        elif self.search_mode == 'equals':
            return self.search_string == test_name
        elif self.search_mode == 'starts':
            return test_name.startswith(self.search_string)
        elif self.search_mode == 'ends':
            return test_name.endswith(self.search_string)
        return False
    
    def _search_in_directory(self, directory: str):
        """在单个目录中搜索（由单个线程执行）"""
        try:
            # 检查当前目录名
            dir_name = os.path.basename(directory)
            if self._match_name(dir_name):
                with self.results_lock:
                    self.results.append((directory, 'directory'))
            
            # 检查目录中的文件
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                
                # 检查是否匹配
                if self._match_name(item):
                    item_type = 'directory' if os.path.isdir(item_path) else 'file'
                    with self.results_lock:
                        self.results.append((item_path, item_type))
                        
        except (PermissionError, OSError) as e:
            # 忽略无权限访问的目录
            pass


class AsyncFileSearcher:
    """使用线程池的异步文件搜索器（更高效的实现）"""
    
    def __init__(self, max_workers: int = 20):
        self.max_workers = max_workers
    
    def search(self, 
               root_dir: str, 
               search_string: str, 
               search_mode: str = 'contains',
               case_sensitive: bool = False,
               include_hidden: bool = False) -> List[dict]:
        """
        高效搜索文件和目录
        
        Args:
            root_dir: 根目录
            search_string: 搜索字符串
            search_mode: 搜索模式
            case_sensitive: 是否区分大小写
            include_hidden: 是否包含隐藏文件（以.开头的文件）
            
        Returns:
            搜索结果列表，每个元素是包含路径、名称、类型等信息的字典
        """
        results = []
        results_lock = threading.Lock()
        search_str = search_string if case_sensitive else search_string.lower()
        
        def check_match(name: str) -> bool:
            """检查名称是否匹配"""
            if not include_hidden and name.startswith('.'):
                return False
                
            test_name = name if case_sensitive else name.lower()
            
            if search_mode == 'contains':
                return search_str in test_name
            elif search_mode == 'equals':
                return search_str == test_name
            elif search_mode == 'starts':
                return test_name.startswith(search_str)
            elif search_mode == 'ends':
                return test_name.endswith(search_str)
            elif search_mode == 'regex':
                import re
                try:
                    return re.search(search_string, name, 
                                    0 if case_sensitive else re.IGNORECASE) is not None
                except:
                    return False
            return False
        
        def process_path(path: str, name: str, is_dir: bool):
            """处理单个路径"""
            if check_match(name):
                with results_lock:
                    results.append({
                        'path': path,
                        'name': name,
                        'type': 'directory' if is_dir else 'file',
                        'size': os.path.getsize(path) if not is_dir else None,
                        'modified': os.path.getmtime(path)
                    })
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for root, dirs, files in os.walk(root_dir):
                # 处理目录
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    futures.append(
                        executor.submit(process_path, dir_path, dir_name, True)
                    )
                
                # 处理文件
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    futures.append(
                        executor.submit(process_path, file_path, file_name, False)
                    )
            
            # 等待所有任务完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    # 忽略单个文件的错误
                    pass
        
        return results


def simple_multithread_search(directory: str, 
                              search_string: str, 
                              max_workers: int = 10,
                              case_sensitive: bool = False) -> List[str]:
    """
    简单的多线程搜索函数
    
    Args:
        directory: 搜索目录
        search_string: 搜索字符串
        max_workers: 最大线程数
        case_sensitive: 是否区分大小写
        
    Returns:
        匹配的文件和文件夹路径列表
    """
    results = []
    lock = threading.Lock()
    search_str = search_string if case_sensitive else search_string.lower()
    
    def search_batch(paths: List[Tuple[str, str, bool]]):
        """处理一批路径"""
        local_results = []
        for path, name, is_dir in paths:
            test_name = name if case_sensitive else name.lower()
            if search_str in test_name or search_str == test_name:
                local_results.append(path)
        
        if local_results:
            with lock:
                results.extend(local_results)
    
    # 收集所有路径
    all_paths = []
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            all_paths.append((os.path.join(root, d), d, True))
        for f in files:
            all_paths.append((os.path.join(root, f), f, False))
    
    # 分批处理
    batch_size = max(1, len(all_paths) // max_workers)
    batches = [all_paths[i:i + batch_size] 
               for i in range(0, len(all_paths), batch_size)]
    
    # 使用线程池
    with ThreadPoolExecutor(max_workers=min(max_workers, len(batches))) as executor:
        futures = [executor.submit(search_batch, batch) for batch in batches]
        for future in as_completed(futures):
            future.result()
    
    return results


# 使用示例
if __name__ == "__main__":
    # 示例1：使用 MultiThreadFileSearcher
    print("=" * 50)
    print("示例1: 使用 MultiThreadFileSearcher")
    print("=" * 50)
    
    searcher = MultiThreadFileSearcher(max_workers=10)
    
    # 搜索包含 "test" 的文件和文件夹
    start_time = time.time()
    results = searcher.search_directory(
        directory=".",  # 当前目录
        search_string="test",
        search_mode="contains",  # contains, equals, starts, ends
        case_sensitive=False
    )
    
    print(f"找到 {len(results)} 个匹配项:")
    for path, item_type in results[:10]:  # 只显示前10个
        print(f"  [{item_type}] {path}")
    
    print(f"搜索耗时: {time.time() - start_time:.2f} 秒\n")
    
    # 示例2：使用 AsyncFileSearcher（更详细的信息）
    print("=" * 50)
    print("示例2: 使用 AsyncFileSearcher")
    print("=" * 50)
    
    async_searcher = AsyncFileSearcher(max_workers=20)
    
    start_time = time.time()
    results = async_searcher.search(
        root_dir=".",
        search_string="py",
        search_mode="ends",  # 搜索以 "py" 结尾的文件
        case_sensitive=False,
        include_hidden=False
    )
    
    print(f"找到 {len(results)} 个匹配项:")
    for item in results[:10]:  # 只显示前10个
        print(f"  [{item['type']}] {item['name']}")
        print(f"    路径: {item['path']}")
        if item['type'] == 'file' and item['size'] is not None:
            print(f"    大小: {item['size']} bytes")
    
    print(f"搜索耗时: {time.time() - start_time:.2f} 秒\n")
    
    # 示例3：使用简单函数
    print("=" * 50)
    print("示例3: 使用简单搜索函数")
    print("=" * 50)
    
    start_time = time.time()
    results = simple_multithread_search(
        directory=".",
        search_string="config",
        max_workers=15,
        case_sensitive=False
    )
    
    print(f"找到 {len(results)} 个匹配项:")
    for path in results[:10]:
        print(f"  {path}")
    
    print(f"搜索耗时: {time.time() - start_time:.2f} 秒")