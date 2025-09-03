const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');
const cors = require('cors');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3001; // 使用3001端口

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

// 获取用户文件列表接口
app.get('/api/user/:username/files', (req, res) => {
  const { username } = req.params;
  const userDir = path.join(__dirname, 'backend', 'userimages', username);
  
  if (!fs.existsSync(userDir)) {
    return res.status(404).json({ message: '用户文件目录不存在' });
  }
  
  try {
    const files = {
      imageFiles: [],
      videoFiles: []
    };
    
    // 读取图像去雾文件
    const imageDehazeDir = path.join(userDir, 'imagedehazed');
    if (fs.existsSync(imageDehazeDir)) {
      const imageFiles = fs.readdirSync(imageDehazeDir)
        .filter(file => file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg'))
        .map(file => {
          const filePath = path.join(imageDehazeDir, file);
          const stats = fs.statSync(filePath);
          return {
            name: file,
            path: `/userimages/${username}/imagedehazed/${file}`,
            createdTime: stats.birthtime,
            size: stats.size,
            type: 'image'
          };
        })
        .sort((a, b) => new Date(b.createdTime) - new Date(a.createdTime));
      
      files.imageFiles = imageFiles;
    }
    
    // 读取视频去雾文件
    const videoDehazeDir = path.join(userDir, 'videodehazed');
    if (fs.existsSync(videoDehazeDir)) {
      const videoDirs = fs.readdirSync(videoDehazeDir)
        .filter(item => {
          const itemPath = path.join(videoDehazeDir, item);
          return fs.statSync(itemPath).isDirectory();
        })
        .map(dirName => {
          const dirPath = path.join(videoDehazeDir, dirName);
          const stats = fs.statSync(dirPath);
          const images = fs.readdirSync(dirPath)
            .filter(file => file.endsWith('.png') || file.endsWith('.jpg') || file.endsWith('.jpeg'))
            .sort((a, b) => {
              const numA = parseInt(a.split('.')[0]);
              const numB = parseInt(b.split('.')[0]);
              return numA - numB;
            })
            .map(file => ({
              name: file,
              path: `/userimages/${username}/videodehazed/${dirName}/${file}`
            }));
          
          return {
            name: dirName,
            path: `/userimages/${username}/videodehazed/${dirName}`,
            createdTime: stats.birthtime,
            images: images,
            frameCount: images.length,
            type: 'video'
          };
        })
        .sort((a, b) => new Date(b.createdTime) - new Date(a.createdTime));
      
      files.videoFiles = videoDirs;
    }
    
    res.json(files);
  } catch (error) {
    console.error('Error reading user files:', error);
    res.status(500).json({ message: '读取文件列表失败', error: error.message });
  }
});

// 服务静态文件接口
app.use('/userimages', express.static(path.join(__dirname, 'backend', 'userimages')));
app.use('/userimage', express.static(path.join(__dirname, 'backend', 'userimages'))); // 兼容旧版本

// 启动服务
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
