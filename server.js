const express = require('express');
const mysql = require('mysql2');
const bodyParser = require('body-parser');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
const port = 3001; // 使用3001端口

// 创建图片服务器应用
const imageApp = express();
const imagePort = 3002; // 图片服务端口

// 允许跨域请求
app.use(cors());
app.use(bodyParser.json());

// 图片服务器跨域配置
imageApp.use(cors());
imageApp.use(bodyParser.json());
imageApp.use(bodyParser.urlencoded({ extended: true }));

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

// 图片上传接口（保留供将来扩展使用）
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

// 用户文件列表API - 使用最简单直接的方法
imageApp.get('/api/user/:username/files', (req, res) => {
  console.log('=== 文件列表API被调用 ===');
  console.log('请求URL:', req.url);
  console.log('请求参数 username:', req.params.username);
  
  const username = req.params.username;
  
  // 构建用户目录路径 - 使用最直接的路径
  const userBaseDir = path.join(__dirname, 'backend', 'userimages', username);
  const imageDir = path.join(userBaseDir, 'imagedehazed');
  const videoDir = path.join(userBaseDir, 'videodehazed');
  
  console.log(`获取用户 ${username} 的文件列表`);
  console.log(`用户基础目录: ${userBaseDir}`);
  console.log(`图片目录: ${imageDir}`);
  console.log(`视频目录: ${videoDir}`);
  console.log(`用户目录是否存在: ${fs.existsSync(userBaseDir)}`);
  console.log(`图片目录是否存在: ${fs.existsSync(imageDir)}`);
  console.log(`视频目录是否存在: ${fs.existsSync(videoDir)}`);

  const result = {
    imageFiles: [],
    videoFiles: []
  };
  
  try {
    // 1. 处理图像文件 - 最简单的方法
    if (fs.existsSync(imageDir)) {
      console.log('图片目录存在，开始读取文件...');
      const imageFiles = fs.readdirSync(imageDir);
      console.log(`找到 ${imageFiles.length} 个文件:`, imageFiles);
      
      imageFiles.forEach(file => {
        const ext = path.extname(file).toLowerCase();
        if (ext === '.png' || ext === '.jpg' || ext === '.jpeg') {
          const filePath = path.join(imageDir, file);
          try {
            const stats = fs.statSync(filePath);
            if (stats.isFile()) {
              // 直接读取文件并转换为base64
              const fileBuffer = fs.readFileSync(filePath);
              const base64Data = fileBuffer.toString('base64');
              
              result.imageFiles.push({
                name: file,
                path: `/userimages/${username}/imagedehazed/${file}`,
                createdTime: stats.mtime.toISOString(),
                size: stats.size,
                type: 'image',
                base64: `data:image/${ext.replace('.', '')};base64,${base64Data}`
              });
              console.log(`添加图片文件: ${file}`);
            }
          } catch (fileErr) {
            console.error(`读取文件 ${file} 失败:`, fileErr.message);
          }
        }
      });
      
      // 按创建时间排序
      result.imageFiles.sort((a, b) => new Date(b.createdTime) - new Date(a.createdTime));
    } else {
      console.log('图片目录不存在:', imageDir);
    }
    
    // 2. 处理视频文件 - 最简单的方法
    if (fs.existsSync(videoDir)) {
      console.log('视频目录存在，开始读取文件夹...');
      const videoFolders = fs.readdirSync(videoDir);
      console.log(`找到 ${videoFolders.length} 个视频文件夹:`, videoFolders);
      
      videoFolders.forEach(folder => {
        const folderPath = path.join(videoDir, folder);
        try {
          const stats = fs.statSync(folderPath);
          if (stats.isDirectory()) {
            const frameFiles = fs.readdirSync(folderPath);
            const images = [];
            
            frameFiles.forEach(frameFile => {
              const ext = path.extname(frameFile).toLowerCase();
              if (ext === '.png' || ext === '.jpg' || ext === '.jpeg') {
                try {
                  // 提取帧号
                  const frameNum = parseInt(path.basename(frameFile, ext));
                  if (!isNaN(frameNum)) {
                    const frameFilePath = path.join(folderPath, frameFile);
                    const frameBuffer = fs.readFileSync(frameFilePath);
                    const base64Data = frameBuffer.toString('base64');
                    
                    images.push({
                      name: frameFile,
                      path: `/userimages/${username}/videodehazed/${folder}/${frameFile}`,
                      frameNumber: frameNum,
                      base64: `data:image/${ext.replace('.', '')};base64,${base64Data}`
                    });
                  }
                } catch (frameErr) {
                  console.error(`读取帧文件 ${frameFile} 失败:`, frameErr.message);
                }
              }
            });
            
            // 按帧号排序
            images.sort((a, b) => a.frameNumber - b.frameNumber);
            
            result.videoFiles.push({
              name: folder,
              path: `/userimages/${username}/videodehazed/${folder}`,
              createdTime: stats.mtime.toISOString(),
              images: images,
              frameCount: images.length,
              type: 'video'
            });
            console.log(`添加视频文件夹: ${folder}，包含 ${images.length} 帧`);
          }
        } catch (folderErr) {
          console.error(`读取文件夹 ${folder} 失败:`, folderErr.message);
        }
      });
      
      // 按创建时间排序
      result.videoFiles.sort((a, b) => new Date(b.createdTime) - new Date(a.createdTime));
    } else {
      console.log('视频目录不存在:', videoDir);
    }
    
    console.log(`返回结果: ${result.imageFiles.length} 个图片文件, ${result.videoFiles.length} 个视频文件`);
    res.json(result);
    
  } catch (error) {
    console.error('读取用户文件错误:', error);
    res.status(500).json({ 
      message: '读取文件列表失败', 
      error: error.message,
      stack: error.stack 
    });
  }
});

// 注意：此API现在直接在3002端口提供，使用base64返回图片数据
// 注意：静态文件由Python服务(3000端口)提供，文件列表API由本服务(3002端口)提供

// 启动服务
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});

// 启动图片服务器
imageApp.listen(imagePort, () => {
  console.log(`图片服务器启动成功: http://localhost:${imagePort}`);
  console.log(`提供服务：图片上传接口（为将来扩展预留）`);
  console.log(`注意：静态文件由Python服务(3000端口)提供，文件列表API由本服务(3002端口)提供`);
});
