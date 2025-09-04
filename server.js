const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3001; // 使用3001端口

// 创建图片服务器应用
const imageApp = express();
const imagePort = 3002; // 图片服务端口

// 启动 Python 服务
const pythonProcess = spawn('python', ['backend/app.py'], {
  stdio: 'inherit', // 让 Python 的输出显示在 Node 控制台
  shell: true       // 兼容性更好
});

pythonProcess.on('close', (code) => {
  console.log(`Python process exited with code ${code}`);
});

// 允许跨域请求
app.use(cors());
app.use(bodyParser.json());

// 图片服务器跨域配置
imageApp.use(cors());
imageApp.use(bodyParser.json());
imageApp.use(bodyParser.urlencoded({ extended: true }));

// 配置静态文件服务（为用户图片提供HTTP访问）
const userimagesPath = path.join(__dirname, 'backend', 'userimages');
if (fs.existsSync(userimagesPath)) {
  imageApp.use('/userimages', express.static(userimagesPath));
  console.log(`静态文件服务已启动: /userimages -> ${userimagesPath}`);
} else {
  console.log(`警告: 用户图片目录不存在: ${userimagesPath}`);
}

// 创建数据库连接池
const pool = mysql.createPool({
  host: 'localhost',
  port: 3306,
  user: 'dehaze_user',
  password: '20010223Xu@',
  database: 'dehaze_db',
});

// 密码哈希函数
const crypto = require('crypto');
function hashPassword(password) {
  return crypto.createHash('sha256').update(password).digest('hex');
}

// 创建用户文件目录的函数
function createUserDirectories(username) {
  const userDir = path.join(__dirname, 'backend', 'userimages', username);
  const imageDehazeDir = path.join(userDir, 'imagedehazed');
  const videoDehazeDir = path.join(userDir, 'videodehazed');
  
  try {
    // 创建用户根目录
    if (!fs.existsSync(userDir)) {
      fs.mkdirSync(userDir, { recursive: true });
      console.log(`Created user directory: ${userDir}`);
    }
    
    // 创建图像去雾目录
    if (!fs.existsSync(imageDehazeDir)) {
      fs.mkdirSync(imageDehazeDir, { recursive: true });
      console.log(`Created image dehaze directory: ${imageDehazeDir}`);
    }
    
    // 创建视频去雾目录
    if (!fs.existsSync(videoDehazeDir)) {
      fs.mkdirSync(videoDehazeDir, { recursive: true });
      console.log(`Created video dehaze directory: ${videoDehazeDir}`);
    }
    
    return true;
  } catch (error) {
    console.error(`Failed to create directories for user ${username}:`, error);
    return false;
  }
}

// 登录接口
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ message: '用户名和密码不能为空' });
  }
  const passwordHash = hashPassword(password);
  pool.query(
    'SELECT * FROM users WHERE username = ? AND password_hash = ?',
    [username, passwordHash],
    (err, results) => {
      if (err) {
        return res.status(500).json({ message: '数据库错误', error: err });
      }
      if (results.length > 0) {
        res.json({ message: '登录成功' });
      } else {
        res.status(401).json({ message: '用户名或密码错误' });
      }
    }
  );
});

// 注册接口
app.post('/register', (req, res) => {
  const { username, password } = req.body;
  if (!username || !password) {
    return res.status(400).json({ message: '用户名和密码不能为空' });
  }
  const passwordHash = hashPassword(password);
  // 检查用户名是否已存在
  pool.query(
    'SELECT * FROM users WHERE username = ?',
    [username],
    (err, results) => {
      if (err) {
        return res.status(500).json({ message: '数据库错误', error: err });
      }
      if (results.length > 0) {
        return res.status(409).json({ message: '用户名已存在' });
      }
      // 插入新用户
      pool.query(
        'INSERT INTO users (username, password_hash) VALUES (?, ?)',
        [username, passwordHash],
        (err2, results2) => {
          if (err2) {
            return res.status(500).json({ message: '注册失败', error: err2 });
          }
          
          // 注册成功后创建用户文件目录
          const dirCreated = createUserDirectories(username);
          if (!dirCreated) {
            console.warn(`Warning: Failed to create directories for user ${username}`);
          }
          
          res.json({ message: '注册成功' });
        }
      );
    }
  );
});

// 健康检查接口
app.get('/healthz', (req, res) => {
  res.json({ status: 'ok', server: 'express' });
});

// 图片服务器健康检查
imageApp.get('/healthz', (req, res) => {
  res.json({ status: 'ok', server: 'image-server', port: imagePort });
});

// 图片上传接口
imageApp.post('/api/user/:username/upload', (req, res) => {
  const username = req.params.username;
  const userDir = path.join(__dirname, 'backend', 'userimages', username, 'imagedehazed');
  
  // 确保用户目录存在
  if (!fs.existsSync(userDir)) {
    fs.mkdirSync(userDir, { recursive: true });
  }
  
  // 这里可以添加文件上传逻辑
  // 目前先返回成功状态
  res.json({ message: '上传接口已准备就绪', uploadPath: userDir });
});

// 图片服务器添加目录浏览API
imageApp.get('/api/user/:username/files', (req, res) => {
  const username = req.params.username;
  const userDir = path.join(__dirname, 'backend', 'userimages', username);
  
  if (!fs.existsSync(userDir)) {
    return res.status(404).json({ message: '用户文件目录不存在' });
  }
  
  const files = {
    imageFiles: [],
    videoFiles: []
  };
  
  try {
    // 读取图像去雾文件
    const imageDir = path.join(userDir, 'imagedehazed');
    if (fs.existsSync(imageDir)) {
      const imageFiles = [];
      fs.readdirSync(imageDir).forEach(file => {
        if (file.toLowerCase().endsWith('.png') || file.toLowerCase().endsWith('.jpg') || file.toLowerCase().endsWith('.jpeg')) {
          const filePath = path.join(imageDir, file);
          if (fs.statSync(filePath).isFile()) {
            const stats = fs.statSync(filePath);
            imageFiles.push({
              name: file,
              path: `/userimages/${username}/imagedehazed/${file}`,
              createdTime: stats.mtime.toISOString(),
              size: stats.size,
              type: 'image'
            });
          }
        }
      });
      files.imageFiles = imageFiles.sort((a, b) => new Date(b.createdTime) - new Date(a.createdTime));
    }
    
    // 读取视频去雾文件
    const videoDir = path.join(userDir, 'videodehazed');
    if (fs.existsSync(videoDir)) {
      const videoFiles = [];
      fs.readdirSync(videoDir).forEach(videoFolder => {
        const videoFolderPath = path.join(videoDir, videoFolder);
        if (fs.statSync(videoFolderPath).isDirectory()) {
          const stats = fs.statSync(videoFolderPath);
          const images = [];
          
          fs.readdirSync(videoFolderPath).forEach(frameFile => {
            if (frameFile.toLowerCase().endsWith('.png') || frameFile.toLowerCase().endsWith('.jpg') || frameFile.toLowerCase().endsWith('.jpeg')) {
              try {
                const frameNum = parseInt(frameFile.split('.')[0]);
                images.push({
                  name: frameFile,
                  path: `/userimages/${username}/videodehazed/${videoFolder}/${frameFile}`,
                  frameNumber: frameNum
                });
              } catch (e) {
                // 忽略无法解析的文件名
              }
            }
          });
          
          images.sort((a, b) => a.frameNumber - b.frameNumber);
          
          videoFiles.push({
            name: videoFolder,
            path: `/userimages/${username}/videodehazed/${videoFolder}`,
            createdTime: stats.mtime.toISOString(),
            images: images,
            frameCount: images.length,
            type: 'video'
          });
        }
      });
      files.videoFiles = videoFiles.sort((a, b) => new Date(b.createdTime) - new Date(a.createdTime));
    }
    
    res.json(files);
  } catch (error) {
    console.error('读取用户文件错误:', error);
    res.status(500).json({ message: '读取文件列表失败', error: error.message });
  }
});

// 启动服务
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

// 启动图片服务器
imageApp.listen(imagePort, () => {
  console.log(`图片服务器启动: http://localhost:${imagePort}`);
  console.log(`图片访问地址: http://localhost:${imagePort}/userimages/`);
  console.log(`用户文件API: http://localhost:${imagePort}/api/user/{username}/files`);
});
