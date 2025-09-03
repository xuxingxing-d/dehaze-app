#!/usr/bin/env python3
"""
测试用户记录文件功能的示例脚本
"""
import requests
import os
import json
import io
from PIL import Image
import numpy as np

# 测试服务器地址
BASE_URL = "http://localhost:3000"

def create_test_image():
    """创建一个测试图像"""
    # 创建一个简单的测试图像（模拟有雾图像）
    img = Image.new('RGB', (256, 256), color=(150, 150, 150))
    
    # 保存到内存
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

def test_image_dehaze_with_records():
    """测试图像去雾功能及记录文件生成"""
    print("测试图像去雾功能及记录文件生成...")
    
    # 测试数据
    test_username = "xustr"
    test_image = create_test_image()
    
    # 首先创建用户目录
    response = requests.post(
        f"{BASE_URL}/api/create-user-directories",
        data={"username": test_username}
    )
    print(f"创建用户目录响应: {response.status_code}")
    
    # 执行图像去雾
    files = {'image': ('test_image.png', test_image, 'image/png')}
    data = {
        'dataset': 'thick',
        'network': '1',
        'username': test_username
    }
    
    response = requests.post(f"{BASE_URL}/api/dehaze", files=files, data=data)
    print(f"图像去雾响应状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"去雾成功！记录文件路径: {result.get('record_file', '未提供')}")
        
        # 检查记录文件是否存在
        backend_dir = os.path.dirname(__file__)
        record_file_path = os.path.join(backend_dir, 'userimages', test_username, 'imagedehazed', 'image_records.txt')
        
        if os.path.exists(record_file_path):
            print(f"✓ 图像记录文件已创建: {record_file_path}")
            
            # 读取并显示记录内容
            with open(record_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print("记录文件内容:")
                print("-" * 50)
                print(content)
                print("-" * 50)
        else:
            print(f"✗ 图像记录文件未找到: {record_file_path}")
    else:
        print(f"去雾失败: {response.text}")

def create_test_video_frames():
    """创建测试视频帧"""
    frames = []
    for i in range(3):  # 创建3帧测试视频
        img = Image.new('RGB', (128, 128), color=(100 + i*20, 100 + i*20, 100 + i*20))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        frames.append(('video_frames', (f'frame_{i+1}.png', img_bytes, 'image/png')))
    return frames

def test_video_dehaze_with_records():
    """测试视频去雾功能及记录文件生成"""
    print("\n测试视频去雾功能及记录文件生成...")
    
    # 测试数据
    test_username = "xustr"
    test_video_name = "test_video"
    
    # 创建测试视频帧
    frames = create_test_video_frames()
    
    # 准备数据
    data = {
        'video_name': test_video_name,
        'dataset': 'thin',
        'network': '2',
        'username': test_username
    }
    
    response = requests.post(f"{BASE_URL}/api/video-dehaze", files=frames, data=data)
    print(f"视频去雾响应状态码: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"视频去雾成功！处理了 {result.get('frames_processed', 0)} 帧")
        
        # 检查视频记录文件是否存在
        backend_dir = os.path.dirname(__file__)
        video_record_file_path = os.path.join(backend_dir, 'userimages', test_username, 'videodehazed', test_video_name, 'video_records.txt')
        
        if os.path.exists(video_record_file_path):
            print(f"✓ 视频记录文件已创建: {video_record_file_path}")
            
            # 读取并显示记录内容
            with open(video_record_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print("视频记录文件内容:")
                print("-" * 50)
                print(content)
                print("-" * 50)
        else:
            print(f"✗ 视频记录文件未找到: {video_record_file_path}")
    else:
        print(f"视频去雾失败: {response.text}")

def check_directory_structure():
    """检查目录结构"""
    print("\n检查目录结构...")
    
    test_username = "xustr"
    backend_dir = os.path.dirname(__file__)
    user_base_dir = os.path.join(backend_dir, 'userimages', test_username)
    
    if os.path.exists(user_base_dir):
        print(f"用户目录结构:")
        for root, dirs, files in os.walk(user_base_dir):
            level = root.replace(user_base_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")

if __name__ == "__main__":
    print("开始测试记录文件功能...\n")
    
    try:
        # 测试图像去雾记录
        test_image_dehaze_with_records()
        
        # 测试视频去雾记录
        test_video_dehaze_with_records()
        
        # 检查目录结构
        check_directory_structure()
        
    except requests.exceptions.ConnectionError:
        print("连接错误：请确保服务器正在运行 (python app.py)")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    
    print("\n测试完成！")