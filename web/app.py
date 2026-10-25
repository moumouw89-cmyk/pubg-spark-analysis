# -*- coding: utf-8 -*-
"""
PUBG数据分析 - Flask Web可视化应用
所有可视化结果用中文展示。

说明：/api/predict 是为了让看板可交互而实现的“特征重要性加权估算”，
不是 Spark 模型在线推理；训练与评估请运行 src/train_model.py。
"""
from flask import Flask, render_template, jsonify, request
import json
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 加载分析结果
def load_json(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analysis')
def api_analysis():
    data = load_json('analysis_results.json')
    stats = load_json('extra_stats.json')
    return jsonify({'analysis': data, 'stats': stats})

@app.route('/api/ml_results')
def api_ml_results():
    data = load_json('ml_results.json')
    return jsonify(data)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """交互式估算：用预计算的特征重要性做加权，不是模型推理。"""
    import numpy as np
    try:
        params = request.get_json()
        ml = load_json('ml_results.json')
        fi = {item['feature']: item['importance'] for item in ml.get('feature_importance', [])}

        # 简单的加权预测（基于特征重要性的线性组合）
        feature_vals = {
            'kills': float(params.get('kills', 0)),
            'damageDealt': float(params.get('damageDealt', 0)) / 500,
            'walkDistance': float(params.get('walkDistance', 0)) / 3000,
            'rideDistance': float(params.get('rideDistance', 0)) / 2000,
            'heals': float(params.get('heals', 0)) / 10,
            'boosts': float(params.get('boosts', 0)) / 5,
            'weaponsAcquired': float(params.get('weaponsAcquired', 0)) / 10,
            'killPlace': 1 - float(params.get('killPlace', 50)) / 100,
            'matchDuration': float(params.get('matchDuration', 1500)) / 2000,
            'DBNOs': float(params.get('DBNOs', 0)) / 5,
            'headshotKills': float(params.get('headshotKills', 0)) / 3,
            'assists': float(params.get('assists', 0)) / 5,
            'revives': float(params.get('revives', 0)) / 3,
            'longestKill': float(params.get('longestKill', 0)) / 500,
        }

        pred = sum(feature_vals.get(k, 0) * fi.get(k, 0) for k in feature_vals)
        total_imp = sum(fi.get(k, 0) for k in feature_vals)
        pred = pred / total_imp if total_imp > 0 else 0
        pred = max(0, min(1, pred))

        return jsonify({
            'predicted_rank': round(pred, 4),
            'confidence': '基于特征重要性的启发式估算（Demo，非模型推理）'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,
            debug=os.environ.get('FLASK_DEBUG', '0') == '1')
