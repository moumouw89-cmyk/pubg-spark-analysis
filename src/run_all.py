# -*- coding: utf-8 -*-
"""
PUBG数据分析 - 主运行脚本
"""
import subprocess, sys, os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(name, desc):
    print("\n" + "=" * 60 + f"\n▶ {desc}\n" + "=" * 60)
    r = subprocess.run([sys.executable, os.path.join(SRC_DIR, name)], capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(f"⚠ 警告: {name} 错误码 {r.returncode}", r.stderr[-500:])
    else:
        print(f"✓ {desc} 完成")
    return r.returncode

if __name__ == "__main__":
    print("="*60 + "\n  PUBG游戏数据分析平台 - 全流程运行\n" + "="*60)
    run_script("analysis.py", "PySpark数据分析")
    run_script("train_model.py", "ML排名预测模型")
    print("\n✓ 全部完成! 启动 Web: python web/app.py")
