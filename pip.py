import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import os
import time
from typing import List, Tuple, Set

# 配置参数
MIRROR_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
MAX_WORKERS = 8
TIMEOUT = 300 
LOG_DIR = "pip_test_logs" 

def clean_package_name(pkg: str) -> str:
    return re.split(r'[=<>!~]', pkg)[0].strip()

# def install_package(pkg: str) -> Tuple[str, bool, str]:
def install_package(pkg: str, extra_args: List[str]) -> Tuple[str, bool, str]:
    clean_pkg = clean_package_name(pkg)
    start_time = time.time()
    
    try:
        # cmd = [
        #     'pip', 'install', pkg,
        #     '--no-cache-dir',
        #     '--disable-pip-version-check',
        #     '--prefer-binary',
        #     '-i', MIRROR_URL,
        #     '--progress-bar', 'off' 
        # ]

        cmd = [
            'pip', 'install', pkg,
            '--no-cache-dir',
            '--disable-pip-version-check',
            '--prefer-binary',
            '-i', MIRROR_URL,
            '--progress-bar', 'off'
        ] + extra_args
        
        result = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=TIMEOUT
        )
        
        output = result.stdout
        elapsed = time.time() - start_time
        
        # 判断安装是否成功
        success = (
            "Successfully installed" in output or 
            "Requirement already satisfied" in output or
            "Skipping" in output
        )
        
        # 记录详细日志
        log_content = f"=== 命令: {' '.join(cmd)} ===\n"
        log_content += f"=== 耗时: {elapsed:.2f}s ===\n"
        log_content += f"=== 结果: {'成功' if success else '失败'} ===\n"
        log_content += output
        
        return (clean_pkg, success, log_content)
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        error_msg = f"安装超时({TIMEOUT}s): {pkg}"
        return (clean_pkg, False, f"{error_msg}\n总耗时: {elapsed:.2f}s")
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"安装出错: {pkg}\n错误: {str(e)}"
        return (clean_pkg, False, f"{error_msg}\n总耗时: {elapsed:.2f}s")

def read_requirements(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"未找到文件: {file_path}")
    
    packages = []
    extra_args = []
    
    # with open(file_path) as f:
    #     packages = []
    #     for line in f:
    #         line = line.strip()
    #         if line and not line.startswith('#'):
    #             packages.append(line)
    #     return packages

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('--find-links') or line.startswith('-f'):
                extra_args.extend(line.split())
            else:
                packages.append(line)
    
    return packages, extra_args

def analyze_installation(requirements_file: str = "requirements.txt"):
    """主安装函数"""
    # 准备日志目录
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print(f"开始优化安装依赖包 (使用{MIRROR_URL})...")
    print(f"并行数: {MAX_WORKERS}, 超时: {TIMEOUT}s")
    
    installed = set()
    failed = set()
    stats = defaultdict(int)
    
    try:
        # packages = read_requirements(requirements_file)
        packages, extra_args = read_requirements(requirements_file)
        total_packages = len(packages)
        print(f"共发现 {total_packages} 个需要安装的包")
        
        # 使用线程池并行安装
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # results = executor.map(install_package, packages)
            results = executor.map(lambda p: install_package(p, extra_args), packages)
            
            for i, (pkg, success, log_content) in enumerate(results, 1):
                # 保存日志
                log_type = "success" if success else "failed"
                log_file = os.path.join(LOG_DIR, f"{log_type}_{pkg}.log")
                with open(log_file, 'w') as f:
                    f.write(log_content)
                
                # 更新结果集
                if success:
                    installed.add(pkg)
                    stats['success'] += 1
                    print(f"[{i}/{total_packages}] ✓ {pkg}")
                else:
                    failed.add(pkg)
                    stats['failed'] += 1
                    print(f"[{i}/{total_packages}] ✗ {pkg} (详见 {log_file})")
                
                # 实时显示进度
                stats['processed'] = i
                print(f"进度: {stats['success']}成功, {stats['failed']}失败, {total_packages - i}剩余", end='\r')
    
    except Exception as e:
        print(f"\n初始化失败: {str(e)}")
        return 0, 0
    
    # 保存总结报告
    summary = f"""=== 安装结果总结 ===时间: {time.strftime('%Y-%m-%d %H:%M:%S')}源文件: {requirements_file}镜像源: {MIRROR_URL}并行数: {MAX_WORKERS}超时设置: {TIMEOUT}s总计: {total_packages}成功: {len(installed)}失败: {len(failed)}成功列表:{chr(10).join(sorted(installed))}失败列表:{chr(10).join(sorted(failed))}"""
    
    with open(os.path.join(LOG_DIR, 'summary.txt'), 'w') as f:
        f.write(summary)
    
    # 打印总结
    print("\n\n=== 安装完成 ===")
    print(f"成功: {len(installed)} 个")
    print(f"失败: {len(failed)} 个")
    print(f"详细日志请查看 {LOG_DIR} 目录")
    
    return len(installed), len(failed)

if __name__ == "__main__":
    import sys
    requirements_file = sys.argv[1] if len(sys.argv) > 1 else "test.txt"
    
    try:
        success, failed = analyze_installation(requirements_file)
        exit(failed)
    except KeyboardInterrupt:
        print("\n用户中断安装过程")
        exit(1)
