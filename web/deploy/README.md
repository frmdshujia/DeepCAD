# DeepCAD 部署手册

> 冠心病眼底图像 AI 筛查系统 · 完整部署指南

---

## 目录结构

```
web/
├── deploy/
│   ├── README.md            ← 本文件（部署手册）
│   ├── requirements.txt     ← Python 依赖（后端）
│   ├── start.sh             ← 一键启动脚本
│   └── export_env.sh        ← conda 环境导出脚本
├── backend.py               ← Flask 后端服务
├── benchmark_cpu.py         ← CPU 推理算力测试脚本
├── index.html               ← 落地页
├── upload.html              ← 上传预测页
├── app.js                   ← 前端逻辑
├── styles.css               ← 前端样式
└── API.md                   ← 后端 API 接口文档
```

---

## 一、环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.8 ~ 3.10 |
| PyTorch | ≥ 1.8 |
| 内存 | ≥ 16 GB（CPU 推理），≥ 8 GB（GPU 推理，VRAM ≥ 4 GB） |
| CPU 核心 | ≥ 4 核（建议 ≥ 8 核以保证响应速度） |
| 磁盘 | ≥ 20 GB（模型权重约 3.4 GB × N 个权重） |

---

## 二、服务器部署步骤

### 步骤 1：上传文件

将以下文件上传到远程服务器：

```
必须上传的文件：
  web/                              ← 整个前端+后端目录
  RETFound_MAE-main 中的核心 .py：
    models_vit.py
    models_mae.py
  模型权重：
    UKB_SDPP_focal05_seed42_ep60checkpoint-best.pth  (约 3.4 GB)

推荐目录结构（远程服务器）：
  /opt/deepcad/
  ├── models_vit.py
  ├── models_mae.py
  ├── web/
  │   ├── backend.py
  │   ├── index.html
  │   ├── upload.html
  │   └── ...
  └── checkpoint-best.pth
```

---

### 步骤 2：安装依赖

```bash
# 方式 A：使用 conda 导出的环境（推荐，见步骤 3-A）
conda env create -f retfound_env.yml

# 方式 B：仅安装 pip 依赖（最小化）
pip install -r web/deploy/requirements.txt
```

---

### 步骤 3-A：导出当前 conda 环境（在当前服务器执行）

```bash
# 在当前服务器（node5）上执行：
bash web/deploy/export_env.sh

# 会生成 retfound_env.yml，上传到目标服务器后执行：
conda env create -f retfound_env.yml
```

---

### 步骤 3-B：仅安装 pip 最小依赖（目标服务器无 GPU 时）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask flask-cors pillow timm==0.3.2 psutil
```

---

### 步骤 4：启动后端服务

```bash
# 方式 A：使用一键启动脚本
bash web/deploy/start.sh

# 方式 B：手动启动（CPU 模式）
conda activate retfound
cd /opt/deepcad
python web/backend.py \
  --host 0.0.0.0 \
  --port 8000 \
  --device cpu \
  --checkpoint ./checkpoint-best.pth

# 方式 C：GPU 模式（若服务器有 GPU）
python web/backend.py --device cuda --port 8000 --checkpoint ./checkpoint-best.pth
```

---

### 步骤 5：提供前端静态文件

```bash
# 在 web/ 目录下启动静态文件服务器（前端）
cd web
python3 -m http.server 8080
# 或使用 nginx 代理（生产环境推荐）
```

---

### 步骤 6：修改前端 API 地址

编辑 `web/app.js` 第 7 行：

```javascript
// 改为实际服务器 IP 和端口
const API_BASE = 'http://<服务器IP>:8000';
```

---

### 步骤 7：访问验证

```bash
# 后端健康检查
curl http://<服务器IP>:8000/health
# 预期返回: {"status": "ok", "device": "cpu"}

# 预测接口测试
curl -X POST http://<服务器IP>:8000/api/predict \
  -F "image=@/path/to/test.jpg"
# 预期返回: {"probability": 0.xxxx}
```

前端访问：`http://<服务器IP>:8080`

---

## 三、防火墙 / 端口配置

| 端口 | 用途 |
|------|------|
| 8000 | Flask 后端 API |
| 8080 | 前端静态页面 |

```bash
# Ubuntu / Debian
sudo ufw allow 8000/tcp
sudo ufw allow 8080/tcp

# CentOS / RHEL
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

---

## 四、SSH 隧道访问（本地电脑 → 服务器）

若服务器不能直接暴露公网端口，可通过 SSH 隧道：

```bash
# 在本地电脑执行（将服务器的 8000 端口映射到本地 18000）
ssh -L 18000:localhost:8000 user@<服务器IP>

# 然后在本地浏览器访问：
# http://localhost:18000/health  (后端 API)
# http://localhost:8080           (前端页面，需另外开一条隧道)
```

修改 `app.js` 中 `API_BASE = 'http://localhost:18000'` 即可。

---

## 五、后台持久化运行（断开 SSH 后继续运行）

```bash
# 方式 A：tmux（推荐）
tmux new-session -d -s deepcad "conda activate retfound && python web/backend.py --host 0.0.0.0 --port 8000 --checkpoint ./checkpoint-best.pth"
tmux attach -t deepcad   # 查看日志

# 方式 B：nohup
nohup python web/backend.py --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
tail -f backend.log
```

---

## 六、当前服务器信息（供参考）

| 项目 | 值 |
|------|----|
| 主机名 | node5 |
| 内网 IP | 192.168.3.115 |
| GPU | 4× NVIDIA RTX 3090 (24 GB 各) |
| Python 环境 | conda `retfound`，PyTorch 1.8.1+cu111 |
| 最佳权重路径 | `/data/home/shujia/CHD/result_retfound/basic_classify/finetune_train_UKB_SDPP_focal05_seed42_ep60/UKB_SDPP_focal05_seed42_ep60checkpoint-best.pth` |
| 模型结构 | ViT-Large / patch16 / 307 M 参数 |

---

## 七、算力测试（运行 benchmark_cpu.py 获取实际数值）

```bash
conda activate retfound
cd /data/home/shujia/CHD/model_train/RETFound_MAE-main
python web/benchmark_cpu.py \
  --checkpoint /data/home/shujia/CHD/result_retfound/basic_classify/finetune_train_UKB_SDPP_focal05_seed42_ep60/UKB_SDPP_focal05_seed42_ep60checkpoint-best.pth \
  --runs 10
```

结果会输出完整的推理耗时清单，用于服务器选型决策。
