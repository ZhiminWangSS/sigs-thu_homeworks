#!/usr/bin/env python3
"""
测试不同采样器的效果
"""
import subprocess
import time
import os

def run_sampler_test(sampler_type, n_devices=2):
    """运行特定采样器的测试"""
    print(f"\n{'='*50}")
    print(f"Testing {sampler_type} with {n_devices} devices")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # 使用多进程训练脚本
    cmd = [
        "python", "multiprocess_train.py",
        f"--n_devices={n_devices}",
        f"--sampler_type={sampler_type}",
        "--gpu=0,1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"Exit code: {result.returncode}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        
        if result.returncode == 0:
            print("✓ Test passed")
            # 打印训练输出
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:  # 打印最后10行
                if 'accuracy' in line.lower() or 'test' in line.lower():
                    print(f"  {line}")
        else:
            print("✗ Test failed")
            print(f"Error output:\n{result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("✗ Test timed out after 5 minutes")
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")

def main():
    """主测试函数"""
    print("Starting sampler comparison tests...")
    
    # 测试不同的采样器
    samplers = [ 'randomsampler', 'randomsplitsampler']
    
    for sampler in samplers:
        run_sampler_test(sampler, n_devices=2)
    
    print(f"\n{'='*50}")
    print("All tests completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()