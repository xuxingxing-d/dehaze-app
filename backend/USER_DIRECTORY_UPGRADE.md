# 用户目录管理功能升级说明

## 概述
为了支持多用户的图像和视频去雾功能，对 `app.py` 进行了以下修改，实现用户专属目录管理。

## 主要修改内容

### 1. 修改 `/api/dehaze` 接口

**新增参数：**
- `username` (必填): 用户名，用于创建用户专属目录

**功能增强：**
- 自动创建用户专属目录结构：
  - `backend/userimages/{username}/imagedehazed/`
  - `backend/userimages/{username}/videodehazed/`
- 处理完成后自动复制结果到用户目录
- 使用时间戳生成唯一文件名，避免文件覆盖
- 返回用户结果路径信息

### 2. 新增 `/api/create-user-directories` 接口

**功能：**
- 专门用于在用户注册时创建用户目录
- 与 server.js 协调一致，确保注册成功即创建目录

**参数：**
- `username` (必填): 用户名

**安全特性：**
- 用户名字符过滤，只允许字母、数字、下划线和横线
- 防止路径遍历攻击

### 3. 新增 `/api/video-dehaze` 接口

**功能：**
- 支持视频去雾处理
- 逐帧处理视频并保存到用户专属目录

**参数：**
- `video_frames`: 视频帧文件列表
- `video_name`: 视频名称
- `dataset`: 数据集类型 (thin/moderate/thick)
- `network`: 网络类型 (1/2)
- `username`: 用户名

**输出结构：**
```
backend/userimages/{username}/videodehazed/{video_name}/
├── 1.png
├── 2.png
├── 3.png
└── ...
```

### 4. 保留健康检查接口

**接口：** `/healthz`
- 检查服务状态和 PyTorch 可用性

### 5. 新增记录文件功能

**图像去雾记录：**
- 文件位置：`backend/userimages/{username}/imagedehazed/image_records.txt`
- 记录内容：文件名、生成日期、详细时间、数据集、网络类型
- 格式示例：
  ```
  1693567890123_image1.png | 生成日期: 2025-09-03 | 详细时间: 2025-09-03 14:30:25 | 数据集: thick | 网络: 1
  ```

**视频去雾记录：**
- 文件位置：`backend/userimages/{username}/videodehazed/{video_name}/video_records.txt`
- 记录内容：视频处理开始/结束时间、每帧处理时间、处理统计信息
- 格式示例：
  ```
  === 视频处理开始 ===
  视频名称: test_video | 开始日期: 2025-09-03 | 开始时间: 2025-09-03 14:30:00 | 数据集: thick | 网络: 1
  第1帧: 1.png | 生成日期: 2025-09-03 | 详细时间: 2025-09-03 14:30:15
  第2帧: 2.png | 生成日期: 2025-09-03 | 详细时间: 2025-09-03 14:30:30
  处理完成时间: 2025-09-03 14:30:45 | 成功处理帧数: 2/2
  === 视频处理结束 ===
  ```

## 目录结构

修改后的用户目录结构：
```
backend/
├── userimages/
│   └── {username}/              # 用户名目录
│       ├── imagedehazed/        # 图像去雾结果
│       │   ├── image_records.txt           # 图像处理记录文件
│       │   ├── 1693567890123_image1.png
│       │   ├── 1693567891456_image2.png
│       │   └── ...
│       └── videodehazed/        # 视频去雾结果
│           ├── {video_name}/    # 视频名称目录
│           │   ├── video_records.txt       # 视频处理记录文件
│           │   ├── 1.png
│           │   ├── 2.png
│           │   └── ...
│           └── ...
├── data/                        # 临时处理目录（原有）
├── results/                     # 模型输出目录（原有）
└── ...
```

## API 调用示例

### 1. 创建用户目录
```bash
curl -X POST "http://localhost:3000/api/create-user-directories" \
     -F "username=xustr"
```

### 2. 图像去雾（新增用户名参数）
```bash
curl -X POST "http://localhost:3000/api/dehaze" \
     -F "image=@test.jpg" \
     -F "dataset=thick" \
     -F "network=1" \
     -F "username=xustr"
```

### 3. 视频去雾
```bash
curl -X POST "http://localhost:3000/api/video-dehaze" \
     -F "video_frames=@frame1.jpg" \
     -F "video_frames=@frame2.jpg" \
     -F "video_name=test_video" \
     -F "dataset=thick" \
     -F "network=1" \
     -F "username=xustr"
```

## 与前端集成要点

### 1. 用户注册流程
当用户在前端注册成功后，前端应立即调用 `/api/create-user-directories` 接口创建用户目录。

### 2. 图像去雾流程
- 前端调用去雾接口时必须传递 `username` 参数
- 去雾完成后，图像自动保存到用户专属目录
- 前端可通过返回的 `user_result_path` 获取保存路径

### 3. 视频去雾流程
- 前端将视频分解为帧序列
- 调用 `/api/video-dehaze` 接口处理
- 处理结果按帧序号保存到用户的视频目录

## 注意事项

1. **文件覆盖问题**：图像去雾使用时间戳文件名，避免覆盖
2. **用户名安全**：自动过滤特殊字符，防止安全问题
3. **目录权限**：确保应用有创建目录的权限
4. **存储空间**：定期清理用户目录，避免存储空间不足
5. **并发处理**：视频去雾仍使用固定的临时文件名，高并发时可能需要优化
6. **记录文件编码**：使用 UTF-8 编码，支持中文显示
7. **记录文件大小**：随着处理次数增加，记录文件会逐渐增大，建议定期清理或轮转

## 测试

**基础功能测试：**
```bash
python test_user_directories.py
```

**记录文件功能测试：**
```bash
python test_record_files.py
```

确保服务器运行后再执行测试。

**测试结果验证：**
- 检查用户目录是否创建
- 检查图像/视频处理结果是否保存
- 检查记录文件是否生成并包含正确信息