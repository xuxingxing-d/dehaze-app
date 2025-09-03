腾讯云部署指南（GPU 推理服务，FastAPI）

一、准备
- 开通 GPU 云服务器（CVM），操作系统推荐：Ubuntu 22.04
- 安装 NVIDIA 驱动（镜像通常已预装，验证 `nvidia-smi`）
- 安装 Docker 与（可选）NVIDIA Container Toolkit
  - Docker: https://docs.docker.com/engine/install/ubuntu/
  - nvidia-docker: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

二、拉取代码并构建镜像
```
git clone <你的仓库>
cd backend
docker build -t dehaze-fastapi:latest -f Dockerfile ..
```

三、安装 PyTorch（二选一）
- 方案A：在镜像里安装（编辑 Dockerfile，取消对应 `pip install torch torchvision --index-url ...` 注释，再 build）
- 方案B：在容器启动时安装（适合快速试验）

四、准备模型
- 将你的 `model.pth`（推荐 TorchScript 格式）放到 `backend/models/model.pth`
- 若使用 TorchScript：保持 `MODEL_TYPE=torchscript`
- 若使用普通 state_dict，需要在 `app.py` 中定义网络结构并 `load_state_dict`

五、运行容器
```
# 使用 GPU 运行（推荐）
docker run --gpus all -d --name dehaze \
  -p 3000:3000 \
  -e MODEL_PATH=/app/models/model.pth \
  -e MODEL_TYPE=torchscript \
  dehaze-fastapi:latest

# 如果运行时报错找不到 torch，可进入容器内安装：
docker exec -it dehaze bash -lc "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
```

六、验证
```
curl http://<你的CVM公网IP>:3000/healthz
```
返回 `{"status":"ok"}` 则服务正常。

七、域名与 HTTPS（可选，强烈推荐）
- 在腾讯云域名控制台解析 `api.yourdomain.com` 到 CVM 公网 IP
- 用 Caddy / Nginx / Nginx Proxy Manager 反向代理到 `http://127.0.0.1:3000`，自动申请 Let’s Encrypt 证书

八、前端配置
- 将 APK/前端里的 `API_BASE` 改为 `https://api.yourdomain.com`
- 表单上传：`POST /api/dehaze`，字段 `image`（文件）和 `network`（可选）

九、常见问题
- CUDA/驱动不匹配：更换 PyTorch 安装源（cu118/cu121）或更新驱动
- 显存不足：设置 `USE_HALF=1`（半精度）、限制输入尺寸、分块推理
- 跨域：生产中将 CORS 的 `allow_origins` 改为你的正式域名


