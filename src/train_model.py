# -*- coding: utf-8 -*-
"""
PUBG玩家最终百分位排名预测 - 机器学习模型
使用 Spark MLlib 对比 RandomForest、GBT、LinearRegression
"""
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import col, mean, count, rand, when
from pyspark.sql.types import IntegerType, FloatType, DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.environ.get(
    'PUBG_DATA_PATH',
    os.path.join(BASE_DIR, 'data', 'smaller_train.csv')
)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sc = SparkContext('local[*]', 'PUBG-Prediction')
sc.setLogLevel('WARN')
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# 读取数据：默认读取 data/smaller_train.csv，也可用 PUBG_DATA_PATH 指向 HDFS/其他路径
print("数据路径:", DATA_PATH)
df = spark.read.format("csv").option("header", "true").load(DATA_PATH)
print("数据总数:", df.count())

# ====== 数据预处理 ======
# 转换数据类型
numeric_cols = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills',
                'heals', 'killPlace', 'kills', 'killStreaks', 'longestKill',
                'matchDuration', 'maxPlace', 'numGroups', 'rankPoints',
                'revives', 'rideDistance', 'swimDistance', 'teamKills',
                'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']

for c in numeric_cols:
    df = df.withColumn(c, col(c).cast(DoubleType()))

df = df.withColumn("winPlacePerc", col("winPlacePerc").cast(DoubleType()))

# 过滤异常值
df_clean = df.filter(
    (col("kills") >= 0) & (col("kills") < 30) &
    (col("damageDealt") >= 0) & (col("damageDealt") < 2000) &
    (col("walkDistance") >= 0) & (col("walkDistance") < 10000) &
    (col("rideDistance") >= 0) & (col("rideDistance") < 10000) &
    (col("heals") >= 0) & (col("heals") < 30) &
    (col("boosts") >= 0) & (col("boosts") < 20) &
    (col("winPlacePerc") >= 0) & (col("winPlacePerc") <= 1)
)

# 消除空值
df_clean = df_clean.dropna()
print("清洗后数据量:", df_clean.count())

# ====== 特征工程 ======
feature_cols = ['kills', 'damageDealt', 'walkDistance', 'rideDistance',
                'heals', 'boosts', 'weaponsAcquired', 'killPlace',
                'matchDuration', 'numGroups', 'DBNOs', 'headshotKills',
                'assists', 'revives', 'longestKill', 'swimDistance',
                'killStreaks']

print("特征列:", feature_cols)

# 特征向量化
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_unscaled")
data = assembler.transform(df_clean)

# 特征标准化
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features",
                        withStd=True, withMean=True)
scaler_model = scaler.fit(data)
data_scaled = scaler_model.transform(data)

# ====== 划分训练集/测试集 ======
train, test = data_scaled.randomSplit([0.8, 0.2], seed=42)
print(f"训练集: {train.count()}, 测试集: {test.count()}")

# ====== 随机森林回归模型 ======
print("\n========== 训练随机森林回归模型 ==========")
rf = RandomForestRegressor(featuresCol="features", labelCol="winPlacePerc",
                           numTrees=50, maxDepth=10, seed=42)

rf_model = rf.fit(train)
rf_pred = rf_model.transform(test)

# 评估
evaluator_rmse = RegressionEvaluator(labelCol="winPlacePerc", predictionCol="prediction",
                                     metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="winPlacePerc", predictionCol="prediction",
                                    metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="winPlacePerc", predictionCol="prediction",
                                   metricName="r2")

rf_rmse = evaluator_rmse.evaluate(rf_pred)
rf_mae = evaluator_mae.evaluate(rf_pred)
rf_r2 = evaluator_r2.evaluate(rf_pred)

print(f"随机森林 - RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")

# ====== 梯度提升树回归 ======
print("\n========== 训练梯度提升树回归模型 ==========")
gbt = GBTRegressor(featuresCol="features", labelCol="winPlacePerc",
                    maxIter=50, maxDepth=5, seed=42)
gbt_model = gbt.fit(train)
gbt_pred = gbt_model.transform(test)

gbt_rmse = evaluator_rmse.evaluate(gbt_pred)
gbt_mae = evaluator_mae.evaluate(gbt_pred)
gbt_r2 = evaluator_r2.evaluate(gbt_pred)

print(f"梯度提升树 - RMSE: {gbt_rmse:.4f}, MAE: {gbt_mae:.4f}, R²: {gbt_r2:.4f}")

# ====== 线性回归（基线） ======
print("\n========== 训练线性回归模型（基线） ==========")
lr = LinearRegression(featuresCol="features", labelCol="winPlacePerc")
lr_model = lr.fit(train)
lr_pred = lr_model.transform(test)

lr_rmse = evaluator_rmse.evaluate(lr_pred)
lr_mae = evaluator_mae.evaluate(lr_pred)
lr_r2 = evaluator_r2.evaluate(lr_pred)

print(f"线性回归 - RMSE: {lr_rmse:.4f}, MAE: {lr_mae:.4f}, R²: {lr_r2:.4f}")

# ====== 预测结果采样（用于可视化） ======
sample_pred = rf_pred.select("winPlacePerc", "prediction") \
    .sample(False, 0.05, seed=42).toPandas()
pred_samples = [{'actual': round(float(r['winPlacePerc']), 4),
                 'predicted': round(float(r['prediction']), 4)}
                for _, r in sample_pred.iterrows()]

# ====== 特征重要性 ======
rf_importance = rf_model.featureImportances.toArray().tolist()
feature_importance = [
    {'feature': feature_cols[i], 'importance': round(rf_importance[i], 6)}
    for i in range(len(feature_cols))
]
feature_importance.sort(key=lambda x: x['importance'], reverse=True)

print("\n=== 特征重要性排名 ===")
for i, fi in enumerate(feature_importance):
    print(f"{i+1}. {fi['feature']}: {fi['importance']:.4f}")

# 按测试集 R² 选择最佳模型（当前数据下 GBT 优于随机森林）
best_name = 'GBT' if gbt_r2 >= rf_r2 else 'RandomForest'
best_model = gbt_model if gbt_r2 >= rf_r2 else rf_model
best_r2 = max(gbt_r2, rf_r2)
best_rmse = gbt_rmse if gbt_r2 >= rf_r2 else rf_rmse
best_mae = gbt_mae if gbt_r2 >= rf_r2 else rf_mae

best_model_path = os.path.join(MODEL_DIR, 'best_model')
best_model.write().overwrite().save(best_model_path)
print(f"\n最佳模型（{best_name}）已保存至: {best_model_path}")

# 保存标量器
scaler_path = os.path.join(MODEL_DIR, 'scaler_model')
scaler_model.write().overwrite().save(scaler_path)

# 保存评估结果
eval_results = {
    'random_forest': {
        'rmse': round(float(rf_rmse), 4),
        'mae': round(float(rf_mae), 4),
        'r2': round(float(rf_r2), 4)
    },
    'gbt': {
        'rmse': round(float(gbt_rmse), 4),
        'mae': round(float(gbt_mae), 4),
        'r2': round(float(gbt_r2), 4)
    },
    'linear_regression': {
        'rmse': round(float(lr_rmse), 4),
        'mae': round(float(lr_mae), 4),
        'r2': round(float(lr_r2), 4)
    },
    'best_model': best_name,
    'feature_importance': feature_importance,
    'prediction_samples': pred_samples[:500],
    'num_features': len(feature_cols),
    'feature_names': feature_cols,
    'training_samples': train.count(),
    'test_samples': test.count(),
}

with open(os.path.join(OUTPUT_DIR, 'ml_results.json'), 'w', encoding='utf-8') as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)
print("ML评估结果已保存到 outputs/ml_results.json")

sc.stop()
print("\n========== 模型训练与评估完成! ==========")
print(f"最佳模型: {best_name} (R²={best_r2:.4f})")
print(f"RMSE={best_rmse:.4f}, MAE={best_mae:.4f}")
print("这意味着模型预测的玩家排名与实际排名的平均误差约为 {:.2f}%".format(best_mae * 100))
