"""
根据 dist_data.json 更新 web/app.js 中的概率分布数据。

替换策略:
  1. 删除 generateDistribution() 函数体
  2. 替换 const BINS / dataEurope / dataAsia 三行声明为真实直方图数组

用法:
  python update_web_dist.py --dist infer_dist/dist_data.json
  python update_web_dist.py --dist infer_dist/dist_data.json --dry-run
"""

import json, re, argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", required=True, help="dist_data.json 路径")
    parser.add_argument(
        "--appjs",
        default=str(Path(__file__).parent.parent / "web" / "app.js"),
        help="app.js 路径",
    )
    parser.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    args = parser.parse_args()

    with open(args.dist) as f:
        dist = json.load(f)

    ukb_hist = dist["ukb"]
    cn_hist  = dist["cn"]
    bins     = dist["bins"]
    ukb_n    = dist["ukb_n"]
    cn_n     = dist["cn_n"]

    with open(args.appjs, encoding="utf-8") as f:
        src = f.read()

    original = src  # 备份用于 dry-run diff

    # ── Step 1: 移除 generateDistribution 函数 ────────────────────────────────
    # 匹配从注释行到最后的 }
    gen_fn = re.compile(
        r"// Generate distribution.*?\nfunction generateDistribution.*?\n\}\n",
        re.DOTALL,
    )
    if gen_fn.search(src):
        src = gen_fn.sub("", src)
        print("✓ 已移除 generateDistribution 函数")
    else:
        print("⚠ 未找到 generateDistribution 函数（可能已移除）")

    # ── Step 2: 替换 BINS / dataEurope / dataAsia 声明 ────────────────────────
    # 匹配注释行 + 3 行 const 声明
    const_block = re.compile(
        r"// Europe:.*?\n// Asia:.*?\nconst BINS\s*=.*?;\n"
        r"const dataEurope\s*=.*?;\n"
        r"const dataAsia\s*=.*?;\n",
        re.DOTALL,
    )

    real_consts = (
        f"// ── Real distributions (n={ukb_n} UKB, n={cn_n} 中国队列) ──\n"
        f"const BINS = {bins};\n"
        f"const dataEurope = {json.dumps(ukb_hist)};  // UKB\n"
        f"const dataAsia   = {json.dumps(cn_hist)};  // SDPP+SHCC+WHTM+PUDM\n"
    )

    if const_block.search(src):
        src = const_block.sub(real_consts, src)
        print("✓ 已替换 BINS / dataEurope / dataAsia 声明")
    else:
        # 尝试仅替换三行 const（兼容已删除注释的版本）
        const_only = re.compile(
            r"const BINS\s*=\s*\d+;\n"
            r"const dataEurope\s*=.*?;\n"
            r"const dataAsia\s*=.*?;\n",
            re.DOTALL,
        )
        if const_only.search(src):
            src = const_only.sub(real_consts, src)
            print("✓ 已替换 BINS / dataEurope / dataAsia（无注释版）")
        else:
            print("✗ 未找到 BINS/dataEurope/dataAsia 声明，请手动检查 app.js！")
            return

    if args.dry_run:
        print("\n── dry-run 输出 ─────────────────────────────────")
        # 只打印变更附近的内容
        for i, (a, b) in enumerate(zip(original.splitlines(), src.splitlines())):
            if a != b:
                print(f"  L{i+1:4d} - {a[:100]}")
                print(f"  L{i+1:4d} + {b[:100]}")
        print("── 未写入文件 ──")
        return

    with open(args.appjs, "w", encoding="utf-8") as f:
        f.write(src)

    print(f"\n✓ app.js 已更新: {args.appjs}")
    print(f"  UKB:   {bins} bins, n={ukb_n:,}")
    print(f"  中国:   {bins} bins, n={cn_n:,}")


if __name__ == "__main__":
    main()
