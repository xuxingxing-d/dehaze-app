import io
import os
import base64
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import json

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Optional: simple OpenCV-based enhancement fallback when no model provided
try:
    import cv2  # type: ignore
    OPENCV_AVAILABLE = True
except Exception:
    OPENCV_AVAILABLE = False

# Optional: NIQE metric (no-reference IQA)
try:
    from skimage import metrics as skmetrics  # type: ignore
    SKIMAGE_AVAILABLE = True
except Exception:
    SKIMAGE_AVAILABLE = False


app = FastAPI(title="Dehaze Inference Service", version="1.0.0")

# CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: set to your app domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class DehazeModelWrapper:
    """Thin wrapper to hold a model and run inference.
    You should replace build_model and preprocess/postprocess to match your model.
    """

    def __init__(self) -> None:
        self.model = None
        self.device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        self.model_path = os.getenv("MODEL_PATH", "models/model.pth")
        self.model_type = os.getenv("MODEL_TYPE", "pytorch")  # pytorch|torchscript
        self.half = os.getenv("USE_HALF", "0") == "1"
        self._load_model_if_available()

    def _load_model_if_available(self) -> None:
        if not TORCH_AVAILABLE:
            return
        if not os.path.exists(self.model_path):
            return
        try:
            if self.model_type == "torchscript":
                self.model = torch.jit.load(self.model_path, map_location=self.device)
            else:
                # NOTE: You must define the model architecture to load state_dict.
                # For now, we attempt to load a scripted model; otherwise keep None.
                # Users can switch to torchscript export for zero-code loading.
                self.model = None
            if self.model is not None:
                self.model.to(self.device)
                if self.half and self.device == "cuda":
                    self.model.half()
                self.model.eval()
        except Exception:
            # Fallback to None if loading fails
            self.model = None

    def _preprocess(self, img_pil: Image.Image) -> Optional["torch.Tensor"]:
        if not TORCH_AVAILABLE:
            return None
        arr = np.array(img_pil.convert("RGB"))
        tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0)  # NCHW
        if self.half and self.device == "cuda":
            tensor = tensor.half()
        return tensor.to(self.device)

    def _postprocess(self, tensor: "torch.Tensor") -> Image.Image:
        tensor = tensor.clamp(0, 1).detach().to("cpu").float()
        arr = (tensor.squeeze(0).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(arr)

    def _opencv_enhance(self, img_pil: Image.Image) -> Image.Image:
        # Simple fallback: apply CLAHE on the L channel in LAB space
        if not OPENCV_AVAILABLE:
            return img_pil
        img = np.array(img_pil.convert("RGB"))
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        rgb = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)

    def infer(self, img_pil: Image.Image) -> Image.Image:
        # If a TorchScript model is loaded, use it; otherwise, fallback to simple enhancement
        if self.model is not None and TORCH_AVAILABLE:
            x = self._preprocess(img_pil)
            with torch.no_grad():
                y = self.model(x)
            return self._postprocess(y)
        # Fallback (keeps service functional; replace with your real model)
        return self._opencv_enhance(img_pil)


model_wrapper = DehazeModelWrapper()


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_niqe_from_pil(img: Image.Image) -> Optional[float]:
    if not SKIMAGE_AVAILABLE:
        return None
    arr = np.array(img.convert("RGB"))
    try:
        return float(skmetrics.niqe(arr, channel_axis=2))  # type: ignore
    except TypeError:
        return float(skmetrics.niqe(arr, multichannel=True))  # type: ignore
    except Exception:
        return None


@app.post("/api/dehaze")
async def dehaze_endpoint(
    image: UploadFile = File(...),
    dataset: Optional[str] = Form(None),  # 新增: thin, moderate, thick
    network: Optional[str] = Form(None),  # 新增: 1, 2
    username: Optional[str] = Form(None),  # 新增: 用户名
):
    try:
        # 1) 读取上传图片
        content = await image.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")

        # 2) 验证参数
        if dataset not in ['thin', 'moderate', 'thick']:
            return JSONResponse(status_code=400, content={"message": "dataset must be one of: thin, moderate, thick"})
        if network not in ['1', '2']:
            return JSONResponse(status_code=400, content={"message": "network must be one of: 1, 2"})
        if not username:
            return JSONResponse(status_code=400, content={"message": "username is required"})
        
        # 3) 根据参数选择对应的数据集目录和测试脚本
        dataset_name = f'Haze1k_{dataset}'
        test_script = f'test-{dataset}{network}.py'

        # 4) 将图片保存到指定数据集目录 backend/data/{dataset_name}/test/hazy/(1).png
        try:
            dataset_hazy_dir = os.path.join(os.path.dirname(__file__), 'data', dataset_name, 'test', 'hazy')
            os.makedirs(dataset_hazy_dir, exist_ok=True)
            hazy_save_path = os.path.join(dataset_hazy_dir, '(1).png')
            img.save(hazy_save_path)
            print(f"Saved input hazy image to: {hazy_save_path}")
        except Exception as save_err:
            print(f"Failed to save hazy image: {save_err}")
            return JSONResponse(status_code=400, content={"message": f"Save hazy failed: {save_err}"})

        # 5) 运行对应的 test-{dataset}{network}.py（以当前 backend 为工作目录）
        try:
            import subprocess, sys
            backend_dir = os.path.dirname(__file__)
            print(f"Running {test_script} to generate dehazed image...")
            completed = subprocess.run(
                [sys.executable, test_script],
                cwd=backend_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print(completed.stdout)
            if completed.stderr:
                print(completed.stderr)
        except subprocess.CalledProcessError as run_err:
            print(f"{test_script} failed: {run_err.stderr}")
            return JSONResponse(status_code=500, content={"message": f"{test_script} execution failed"})
        except Exception as e:
            print(f"Unexpected error running {test_script}: {e}")
            return JSONResponse(status_code=500, content={"message": f"Unexpected error: {e}"})

        # 6) 读取去雾结果 backend/results/{dataset_name}/{model_name}/imgs/(1).png 并返回
        # 根据不同的网络选择不同的模型名称
        model_name = 'dehazeformer-t' if network == '1' else 'dehazeformer-s'
        result_path = os.path.join(os.path.dirname(__file__), 'results', dataset_name, model_name, 'imgs', '(1).png')
        
        # 7) 确保用户目录存在并复制结果到用户专属目录
        try:
            # 创建用户专属目录
            user_image_dir = os.path.join(os.path.dirname(__file__), 'userimages', username, 'imagedehazed')
            user_video_dir = os.path.join(os.path.dirname(__file__), 'userimages', username, 'videodehazed')
            os.makedirs(user_image_dir, exist_ok=True)
            os.makedirs(user_video_dir, exist_ok=True)
            
            # 生成唯一的文件名（使用时间戳）
            import time
            from datetime import datetime
            timestamp = str(int(time.time() * 1000))  # 毫秒级时间戳
            user_result_filename = f"{timestamp}_{image.filename or 'dehazed.png'}"
            user_result_path = os.path.join(user_image_dir, user_result_filename)
            
            # 复制去雾结果到用户目录
            import shutil
            if os.path.exists(result_path):
                shutil.copy2(result_path, user_result_path)
                print(f"Copied dehazed image to user directory: {user_result_path}")
                
                # 记录图像生成时间到.txt文件
                record_file_path = os.path.join(user_image_dir, 'image_records.txt')
                current_date = datetime.now().strftime('%Y-%m-%d')
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                with open(record_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"{user_result_filename} | 生成日期: {current_date} | 详细时间: {current_time} | 数据集: {dataset} | 网络: {network}\n")
                print(f"Recorded image generation time to: {record_file_path}")
            else:
                print(f"Warning: Result file not found at {result_path}")
                
        except Exception as copy_err:
            print(f"Failed to copy to user directory: {copy_err}")
            # 不影响主流程，继续返回结果
        if not os.path.exists(result_path):
            return JSONResponse(status_code=404, content={"message": f"Dehazed image not found after {test_script}"})

        out_img = Image.open(result_path).convert("RGB")

        b64 = image_to_base64(out_img)
        return JSONResponse({
            "image_base64": f"data:image/png;base64,{b64}",
            "dataset": dataset,
            "network": network,
            "test_script_used": test_script,
            "username": username,
            "user_result_path": user_result_path if 'user_result_path' in locals() else None,
            "record_file": os.path.join(user_image_dir, 'image_records.txt') if 'user_image_dir' in locals() else None,
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Dehaze failed: {e}"})


@app.post("/api/create-user-directories")
async def create_user_directories(
    username: str = Form(...),
):
    """
    为新注册用户创建专属目录结构
    """
    try:
        if not username or not username.strip():
            return JSONResponse(status_code=400, content={"message": "username is required and cannot be empty"})
        
        # 清理用户名，确保安全（移除特殊字符）
        import re
        safe_username = re.sub(r'[^a-zA-Z0-9_-]', '', username.strip())
        if not safe_username:
            return JSONResponse(status_code=400, content={"message": "username contains invalid characters"})
        
        # 创建用户专属目录
        user_base_dir = os.path.join(os.path.dirname(__file__), 'userimages', safe_username)
        user_image_dir = os.path.join(user_base_dir, 'imagedehazed')
        user_video_dir = os.path.join(user_base_dir, 'videodehazed')
        
        os.makedirs(user_image_dir, exist_ok=True)
        os.makedirs(user_video_dir, exist_ok=True)
        
        return JSONResponse({
            "status": "success",
            "message": f"User directories created for {safe_username}",
            "username": safe_username,
            "directories_created": [
                user_image_dir,
                user_video_dir
            ]
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Failed to create user directories: {e}"})


@app.post("/api/video-dehaze")
async def video_dehaze_endpoint(
    video_frames: list[UploadFile] = File(...),  # 视频帧列表
    video_name: str = Form(...),  # 视频名称
    dataset: Optional[str] = Form(None),  # thin, moderate, thick
    network: Optional[str] = Form(None),  # 1, 2
    username: Optional[str] = Form(None),  # 用户名
):
    """
    视频去雾处理接口
    接收视频帧序列，逐帧处理后保存到用户的videodehazed目录
    """
    try:
        # 验证参数
        if dataset not in ['thin', 'moderate', 'thick']:
            return JSONResponse(status_code=400, content={"message": "dataset must be one of: thin, moderate, thick"})
        if network not in ['1', '2']:
            return JSONResponse(status_code=400, content={"message": "network must be one of: 1, 2"})
        if not username:
            return JSONResponse(status_code=400, content={"message": "username is required"})
        if not video_name:
            return JSONResponse(status_code=400, content={"message": "video_name is required"})
        
        # 清理视频名称，确保安全
        import re
        safe_video_name = re.sub(r'[^a-zA-Z0-9_-]', '', video_name.strip())
        if not safe_video_name:
            return JSONResponse(status_code=400, content={"message": "video_name contains invalid characters"})
        
        # 创建视频专属目录
        user_video_dir = os.path.join(os.path.dirname(__file__), 'userimages', username, 'videodehazed', safe_video_name)
        os.makedirs(user_video_dir, exist_ok=True)
        
        processed_frames = []
        dataset_name = f'Haze1k_{dataset}'
        test_script = f'test-{dataset}{network}.py'
        model_name = 'dehazeformer-t' if network == '1' else 'dehazeformer-s'
        
        # 创建视频记录文件
        from datetime import datetime
        video_record_file_path = os.path.join(user_video_dir, 'video_records.txt')
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 在记录文件中添加视频处理开始信息
        with open(video_record_file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n=== 视频处理开始 ===\n")
            f.write(f"视频名称: {safe_video_name} | 开始日期: {current_date} | 开始时间: {current_time} | 数据集: {dataset} | 网络: {network}\n")
        
        # 逐帧处理
        for idx, frame_file in enumerate(video_frames):
            try:
                # 读取帧图像
                content = await frame_file.read()
                img = Image.open(io.BytesIO(content)).convert("RGB")
                
                # 保存到临时处理目录
                dataset_hazy_dir = os.path.join(os.path.dirname(__file__), 'data', dataset_name, 'test', 'hazy')
                os.makedirs(dataset_hazy_dir, exist_ok=True)
                hazy_save_path = os.path.join(dataset_hazy_dir, '(1).png')  # 仍使用固定名称进行处理
                img.save(hazy_save_path)
                
                # 运行去雾脚本
                import subprocess, sys
                backend_dir = os.path.dirname(__file__)
                completed = subprocess.run(
                    [sys.executable, test_script],
                    cwd=backend_dir,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # 读取处理结果并复制到用户目录
                result_path = os.path.join(os.path.dirname(__file__), 'results', dataset_name, model_name, 'imgs', '(1).png')
                if os.path.exists(result_path):
                    frame_output_path = os.path.join(user_video_dir, f'{idx+1}.png')
                    import shutil
                    shutil.copy2(result_path, frame_output_path)
                    processed_frames.append(frame_output_path)
                    
                    # 记录每帧的处理时间
                    frame_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    frame_date = datetime.now().strftime('%Y-%m-%d')
                    with open(video_record_file_path, 'a', encoding='utf-8') as f:
                        f.write(f"第{idx+1}帧: {idx+1}.png | 生成日期: {frame_date} | 详细时间: {frame_time}\n")
                else:
                    print(f"Warning: Frame {idx+1} result not found")
                    
            except Exception as frame_err:
                print(f"Error processing frame {idx+1}: {frame_err}")
                continue
        
        # 在记录文件中添加视频处理完成信息
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(video_record_file_path, 'a', encoding='utf-8') as f:
            f.write(f"处理完成时间: {end_time} | 成功处理帧数: {len(processed_frames)}/{len(video_frames)}\n")
            f.write(f"=== 视频处理结束 ===\n\n")
        
        return JSONResponse({
            "status": "success",
            "message": f"Video dehaze completed for {len(processed_frames)} frames",
            "video_name": safe_video_name,
            "username": username,
            "dataset": dataset,
            "network": network,
            "frames_processed": len(processed_frames),
            "total_frames": len(video_frames),
            "output_directory": user_video_dir,
            "processed_frame_files": [os.path.basename(f) for f in processed_frames],
            "record_file": video_record_file_path
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Video dehaze failed: {e}"})


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "torch": TORCH_AVAILABLE}


if __name__ == "__main__":
    import uvicorn
    
    host = "0.0.0.0"
    port = int(os.getenv("PORT", "3000"))  # 3000端口用于去雾功能
    
    # 启动服务器
    uvicorn.run(app, host=host, port=port, log_level="info")


