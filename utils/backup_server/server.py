import os
import shutil
import time
from datetime import datetime

def copy_files(source_path, target_folder):
    """
    核心复制函数：
    1. 判断源路径是否存在
    2. 判断是文件还是文件夹
    3. 执行覆盖复制
    """
    
    # 检查源路径是否存在
    if not os.path.exists(source_path):
        print(f"[错误] 源路径不存在: {source_path}")
        return

    # 确保目标主文件夹存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"[提示] 已创建目标文件夹: {target_folder}")

    # 获取源文件/文件夹的名字 (例如 C:/data/my_file.txt -> my_file.txt)
    base_name = os.path.basename(source_path.rstrip(os.sep))
    # 拼接最终的目标路径
    destination_path = os.path.join(target_folder, base_name)

    try:
        # --- 情况 A: 如果源是一个文件夹 ---
        if os.path.isdir(source_path):
            # 如果目标位置已经存在同名文件夹，先删除它（实现覆盖）
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
            
            # 复制文件夹
            shutil.copytree(source_path, destination_path)
            print(f"[成功] 文件夹已覆盖备份到: {destination_path}")

        # --- 情况 B: 如果源是一个文件 ---
        elif os.path.isfile(source_path):
            # shutil.copy2 会保留文件的元数据（如创建时间、权限）
            # 它会自动覆盖目标位置的同名文件
            shutil.copy2(source_path, destination_path)
            print(f"[成功] 文件已覆盖备份到: {destination_path}")

    except Exception as e:
        print(f"[失败] 复制过程中出错: {e}")

def main():
    # ================= 配置区域 =================
    # 输入源文件或文件夹的路径 (请修改这里)
    # Windows 示例: r"D:\Work\SourceData"
    # Mac/Linux 示例: "/Users/name/Documents/data"
    source_path = r"C:\你的\源文件\地址" 
    
    # 输入目标文件夹地址 (请修改这里)
    target_folder = r"D:\你的\备份\地址"
    
    # 设置时间间隔 (分钟)
    interval_minutes = 10
    # ===========================================

    print(f"--- 自动备份脚本启动 ---")
    print(f"源路径: {source_path}")
    print(f"目标路径: {target_folder}")
    print(f"频率: 每 {interval_minutes} 分钟\n")

    while True:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始执行备份...")
        
        copy_files(source_path, target_folder)
        
        print(f"等待 {interval_minutes} 分钟后进行下一次备份...\n")
        # time.sleep 接收秒数，所以需要乘以 60
        time.sleep(interval_minutes * 60)

if __name__ == "__main__":
    main()