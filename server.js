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

// 启动服务
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
