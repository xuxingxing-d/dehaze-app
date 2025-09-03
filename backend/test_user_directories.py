#!/usr/bin/env python3
"""
测试用户目录创建功能
"""
import requests
import os
import json

# 测试服务器地址
BASE_URL = "http://localhost:3000"

def test_create_user_directories():
    """测试创建用户目录功能"""
    print("测试创建用户目录功能...")
    
    # 测试数据
    test_username = "xustr"
    
    # 发送请求
    response = requests.post(
        f"{BASE_URL}/api/create-user-directories",
        data={"username": test_username}
    )
    
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    # 验证目录是否创建
    backend_dir = os.path.dirname(__file__)
    user_image_dir = os.path.join(backend_dir, 'userimages', test_username, 'imagedehazed')
    user_video_dir = os.path.join(backend_dir, 'userimages', test_username, 'videodehazed')
    
    print(f"图像目录是否存在: {os.path.exists(user_image_dir)}")
    print(f"视频目录是否存在: {os.path.exists(user_video_dir)}")
    
    return response.status_code == 200

def test_health_check():
    """测试健康检查功能"""
    print("\n测试健康检查功能...")
    
    response = requests.get(f"{BASE_URL}/healthz")
    print(f"响应状态码: {response.status_code}")
    print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    return response.status_code == 200

if __name__ == "__main__":
    print("开始测试修改后的API功能...\n")
    
    try:
        # 测试健康检查
        if test_health_check():
            print("✓ 健康检查测试通过")
        else:
            print("✗ 健康检查测试失败")
        
        # 测试用户目录创建
        if test_create_user_directories():
            print("✓ 用户目录创建测试通过")
        else:
            print("✗ 用户目录创建测试失败")
            
    except requests.exceptions.ConnectionError:
        print("连接错误：请确保服务器正在运行 (python app.py)")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
    
    print("\n测试完成！")