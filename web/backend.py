"""
DeepCAD 后端服务
接口: POST /api/predict  -> { "probability": 0.xx }
启动: python backend.py [--host 0.0.0.0] [--port 8000] [--checkpoint <path>] [--device cpu|cuda]
"""
import argparse
import os
import sys
import io
import time
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import torch
from torchvision import transforms

# ── 项目路径 ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import models_vit
from util.tta import DEFAULT_TTA_MODES, forward_logits_tta, parse_tta_modes

# ── 参数 ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--host',       default='0.0.0.0')
parser.add_argument('--port',       default=8000, type=int)
parser.add_argument('--device',     default='cpu', choices=['cpu', 'cuda'])
parser.add_argument('--checkpoint', default=
    '/data/home/shujia/CHD/result_retfound/basic_classify/'
    'finetune_train_UKB_SDPP_focal05_seed42_ep60/'
    'UKB_SDPP_focal05_seed42_ep60checkpoint-best.pth')
parser.add_argument('--threshold',  default=0.5, type=float,
    help='分类阈值（仅用于日志显示，前端使用概率值自行判断）')
parser.add_argument('--gate_checkpoint', default=None,
    help='眼底门控模型权重路径，默认自动查找 fundus_gate/checkpoints/best_fundus_gate.pth')
parser.add_argument('--gate_threshold', default=0.92, type=float,
    help='眼底门控阈值，低于此值视为非眼底图像')
parser.add_argument('--tta', action='store_true',
    help='CHD 预测使用 TTA（多视角 logits 平均，延迟约按视角数倍增加）')
parser.add_argument('--tta_modes', type=str, default=DEFAULT_TTA_MODES,
    help='TTA 视角（默认与对比学习几何策略一致，7 视角）')
args = parser.parse_args()

# ── 日志 ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('deepcad')

# ── 眼底门控模型加载 ──────────────────────────────────────────────────────────
gate_checker = None
_gate_ckpt_default = os.path.join(
    BASE_DIR, 'fundus_gate', 'checkpoints', 'best_fundus_gate.pth'
)
_gate_ckpt_path = args.gate_checkpoint or _gate_ckpt_default

if os.path.exists(_gate_ckpt_path):
    try:
        sys.path.insert(0, BASE_DIR)
        from fundus_gate.inference import FundusGateChecker
        gate_checker = FundusGateChecker(
            ckpt_path=_gate_ckpt_path,
            threshold=args.gate_threshold,
            device=args.device,
        )
        log.info(f"眼底门控模型加载完成: {_gate_ckpt_path}")
    except Exception as e:
        log.warning(f"眼底门控模型加载失败（将跳过门控检查）: {e}")
else:
    log.warning(
        f"未找到眼底门控模型权重: {_gate_ckpt_path}\n"
        "  将跳过眼底检测，所有图像均会进入CHD预测。\n"
        "  请先运行 fundus_gate/prepare_data.py 和 fundus_gate/train.py 完成训练。"
    )

# ── CHD模型加载（全局单例）──────────────────────────────────────────────────────
log.info(f"加载CHD模型: {args.checkpoint}")
log.info(f"推理设备: {args.device}")

if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device(args.device)

model = models_vit.vit_large_patch16(
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=False,
)
ckpt = torch.load(args.checkpoint, map_location='cpu')
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict)
model.to(device)
model.eval()
log.info("模型加载完成，服务就绪")
_tta_modes = parse_tta_modes(args.tta_modes) if args.tta else None
if args.tta:
    log.info(f"TTA 已启用 | modes={_tta_modes}")

# ── 图像预处理 ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # 允许跨域，前端本地开发可直接访问

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}

def _allowed(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({'status': 'ok', 'device': args.device})


@app.route('/info', methods=['GET'])
def info():
    """返回当前加载的模型信息，供前端验证"""
    ckpt_name = os.path.basename(args.checkpoint)
    ckpt_size_gb = os.path.getsize(args.checkpoint) / 1024**3
    n_params = sum(p.numel() for p in model.parameters())
    return jsonify({
        'model':      'ViT-Large / patch16 (RETFound)',
        'checkpoint': ckpt_name,
        'checkpoint_size_gb': round(ckpt_size_gb, 2),
        'params_M':   round(n_params / 1e6, 1),
        'device':     args.device,
        'task':       'CHD binary classification (fundus image)',
        'threshold':  args.threshold,
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    前端 POST /api/predict  multipart/form-data  字段名: image
    返回:
      正常: { "probability": 0.xx }
      非眼底: { "probability": 0, "is_fundus": false, "message": "未检测到眼底图像" }
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    if not _allowed(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img = ImageOps.exif_transpose(img)  # 根据 EXIF 方向信息自动旋转，消除手机/PC 差异
        img = img.convert('RGB')

        # ── 步骤1：眼底门控检测 ──────────────────────────────────────────────
        if gate_checker is not None:
            t_gate = time.perf_counter()
            is_fundus, gate_prob = gate_checker.check(img)
            gate_ms = (time.perf_counter() - t_gate) * 1000
            log.info(
                f"门控检测 | is_fundus={is_fundus} | gate_prob={gate_prob:.4f} "
                f"| 耗时={gate_ms:.1f}ms | 文件={file.filename}"
            )
            if not is_fundus:
                return jsonify({
                    'probability': 0,
                    'is_fundus': False,
                    'gate_prob': round(gate_prob, 6),
                    'message': 'Wrong image, please re-upload!',
                })

        # ── 步骤2：CHD预测 ──────────────────────────────────────────────────
        tensor = transform(img).unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            if _tta_modes is not None:
                output = forward_logits_tta(model, tensor, _tta_modes)
            else:
                output = model(tensor)
        latency_ms = (time.perf_counter() - t0) * 1000

        prob = torch.softmax(output, dim=1)[0, 1].item()
        label = 'CHD' if prob >= args.threshold else 'Non-CHD'

        log.info(f"预测完成 | prob={prob:.4f} | label={label} | 耗时={latency_ms:.1f}ms | 文件={file.filename}")

        return jsonify({
            'probability': round(prob, 6),
            'is_fundus': True,
        })

    except Exception as e:
        log.error(f"推理出错: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    log.info(f"启动后端服务: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=False)
