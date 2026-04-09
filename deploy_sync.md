# DeepCAD 部署同步指南

当前服务器无法被公网直接访问时，需要在本机打包、上传到远程服务器后部署。

---

## 一、需要同步的内容

| 路径 | 说明 | 大小约 |
|------|------|--------|
| `web/` | 前端 + 后端 (HTML/JS/CSS/Python) | ~7MB |
| `fundus_gate/` | 眼底门控模型及推理代码 | ~43MB |
| `infer_dist/dist_data.json` | 风险分布数据（已内嵌到 app.js，可选） | 1KB |
| `models_vit.py` | CHD 模型定义 | 2KB |
| `util/` | 模型依赖 | 小 |
| CHD 权重 | 需在远程准备或一并上传 | ~3.4GB |

---

## 二、在本机打包（当前开发服务器）

在项目根目录执行：

```bash
cd /data/home/shujia/CHD/model_train/RETFound_MAE-main

# 创建打包目录
mkdir -p _deploy_pkg

# 复制 web（排除 .specstory 等无关目录）
rsync -a web/ _deploy_pkg/web/ --exclude='.specstory' --exclude='*.pyc'

# 复制 fundus_gate（仅保留必要文件，排除缓存）
rsync -a fundus_gate/checkpoints/ _deploy_pkg/fundus_gate_checkpoints/
rsync -a fundus_gate/inference.py fundus_gate/__init__.py _deploy_pkg/fundus_gate_code/

# 复制模型相关
cp models_vit.py _deploy_pkg/
cp -r util _deploy_pkg/

# 打包
tar -czvf deepcad_deploy_$(date +%Y%m%d).tar.gz -C _deploy_pkg .

# 如需包含 CHD 权重（体积大）
# tar -czvf deepcad_full_$(date +%Y%m%d).tar.gz _deploy_pkg ...
```

**简化版：直接打包整个项目（排除大文件和缓存）**

```bash
cd /data/home/shujia/CHD/model_train/RETFound_MAE-main

tar -czvf deepcad_deploy.tar.gz \
  --exclude='*.pyc' \
  --exclude='.git' \
  --exclude='.specstory' \
  --exclude='fundus_gate/.cifar_cache' \
  --exclude='fundus_gate/*.csv' \
  --exclude='infer_dist/results' \
  web/ fundus_gate/ models_vit.py util/ infer_dist/dist_data.json
```

---

## 三、上传到远程服务器

### 方式 A：本机能 SSH 到远程

```bash
# 假设远程用户为 user，主机为 remote-host（内网 IP 或跳板机）
scp -P 22 deepcad_deploy.tar.gz user@remote-host:/path/to/RETFound_MAE-main/
```

### 方式 B：本机不能直接连远程

1. 在本机下载 `deepcad_deploy.tar.gz` 到本地电脑
2. 从本地电脑上传到远程（如通过跳板机、VPN、内网电脑等）

```bash
# 在本地电脑执行
scp deepcad_deploy.tar.gz user@jump-host:/tmp/
# 再从 jump-host scp 到目标服务器
```

### 方式 C：使用 rsync 增量同步

```bash
rsync -avz -e "ssh -p 22" \
  --exclude='*.pyc' --exclude='.git' --exclude='.specstory' \
  --exclude='infer_dist/results' \
  web/ fundus_gate/ models_vit.py models_mae.py util/ \
  user@remote-host:/path/to/RETFound_MAE-main/
```

---

## 四、在远程服务器解压并启动

```bash
# SSH 登录远程
ssh user@remote-host

# 进入目标目录
cd /path/to/RETFound_MAE-main

# 解压（若使用 tar 打包）
tar -xzvf deepcad_deploy.tar.gz

# 或合并目录结构（若用 rsync 同步了多个目录，需还原 fundus_gate 结构）
mkdir -p fundus_gate
mv fundus_gate_checkpoints fundus_gate/checkpoints 2>/dev/null || true
mv fundus_gate_code/* fundus_gate/ 2>/dev/null || true

# 确认 CHD 权重路径
# 若有权重文件，放到例如：/path/to/checkpoint-best.pth
# 或修改 web/deploy/start.sh 中的 CHECKPOINT_CANDIDATES

# 安装依赖（若未安装）
pip install flask flask-cors torch torchvision pillow timm

# 启动服务
cd web
bash deploy/start.sh cpu 8000
```

---

## 五、一键打包脚本

可执行项目根目录下的 `pack_for_deploy.sh`：

```bash
./pack_for_deploy.sh
```

会生成 `deepcad_deploy_YYYYMMDD.tar.gz`，包含 web、fundus_gate、模型代码等必要文件。
