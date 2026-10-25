# -*- coding: utf-8 -*-
"""
基于Spark的PUBG游戏统计数据分析 - 复现脚本
源自厦门大学林子雨团队实验
新增：中文标签 + JSON结果输出供Web使用
"""
from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql.functions import percentile_approx, mean, max as spark_max, col, count, when
from pyspark.sql.types import IntegerType, FloatType
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
DATA_PATH = os.environ.get(
    'PUBG_DATA_PATH',
    os.path.join(BASE_DIR, 'data', 'smaller_train.csv')
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化Spark
sc = SparkContext('local[*]', 'PUBG-Analysis')
sc.setLogLevel('WARN')
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# 读取数据：默认读取 data/smaller_train.csv，也可用 PUBG_DATA_PATH 指向 HDFS/其他路径
print("数据路径:", DATA_PATH)
df = spark.read.format("csv").option("header", "true").load(DATA_PATH)
total = df.count()
print("数据总数:", total)

results = {}

# ====== 模块1：击杀数统计分析 ======
print("\n========== 击杀数统计分析 ==========")
df = df.withColumn("kills", df["kills"].cast(IntegerType()))
avg_kills = df.select(mean("kills")).collect()[0][0]
max_kills = df.select(spark_max("kills")).collect()[0][0]
kills_99 = df.selectExpr("percentile_approx(kills, 0.99)").collect()[0][0]
print(f"平均击杀: {avg_kills:.4f}, 99%玩家击杀≤{kills_99}, 最高击杀: {max_kills}")

# 击杀分布
kills_dist = df.groupBy("kills").count().orderBy("kills").toPandas()
kills_dist_json = kills_dist.to_dict(orient='records')
results['kills'] = {
    'avg': float(avg_kills), 'max': int(max_kills), 'p99': int(kills_99),
    'distribution': [{'kills': int(r['kills']), 'count': int(r['count'])}
                     for r in kills_dist_json if int(r['kills']) <= int(kills_99)]
}

# ====== 模块1.1：零击杀玩家伤害分布 ======
print("\n========== 零击杀玩家伤害分布 ==========")
df = df.withColumn("damageDealt", df["damageDealt"].cast(FloatType()))
zero_kills = df.filter(df["kills"] == 0)
zero_kills_cnt = zero_kills.count()
print(f"零击杀玩家数: {zero_kills_cnt}, 占比: {zero_kills_cnt/total*100:.2f}%")

# 伤害分桶统计
dmg_buckets = zero_kills.select(
    when(col("damageDealt") == 0, "0伤害")
    .when(col("damageDealt") <= 50, "1-50")
    .when(col("damageDealt") <= 100, "51-100")
    .when(col("damageDealt") <= 200, "101-200")
    .otherwise("200+").alias("damage_range")
).groupBy("damage_range").count().orderBy("damage_range").toPandas()

dmg_order = ["0伤害","1-50","51-100","101-200","200+"]
results['zero_kills_damage'] = {
    'count': int(zero_kills_cnt),
    'ratio': round(zero_kills_cnt/total*100, 2),
    'distribution': [
        {'range': d, 'count': int(dmg_buckets[dmg_buckets['damage_range']==d]['count'].iloc[0])}
        if d in dmg_buckets['damage_range'].values else {'range': d, 'count': 0}
        for d in dmg_order
    ]
}

# ====== 模块2：玩家移动行为分析（步行+载具） ======
print("\n========== 玩家移动行为分析 ==========")
df = df.withColumn("walkDistance", df["walkDistance"].cast(FloatType()))
df = df.withColumn("rideDistance", df["rideDistance"].cast(FloatType()))
df = df.withColumn("swimDistance", df["swimDistance"].cast(FloatType()))

avg_walk = df.select(mean("walkDistance")).collect()[0][0]
walk_99 = df.selectExpr("percentile_approx(walkDistance, 0.99)").collect()[0][0]
max_walk = df.select(spark_max("walkDistance")).collect()[0][0]
print(f"平均步行距离: {avg_walk:.1f}m, 99%≤{walk_99:.0f}m, 最大: {max_walk:.0f}m")

# 步行距离分段统计
walk_buckets = df.filter(col("walkDistance") < walk_99).select(
    when(col("walkDistance") == 0, "0m")
    .when(col("walkDistance") <= 500, "1-500m")
    .when(col("walkDistance") <= 1000, "501-1000m")
    .when(col("walkDistance") <= 2000, "1001-2000m")
    .when(col("walkDistance") <= 3000, "2001-3000m")
    .otherwise("3000m+").alias("walk_range")
).groupBy("walk_range").count().orderBy("walk_range").toPandas()

walk_order = ["0m","1-500m","501-1000m","1001-2000m","2001-3000m","3000m+"]
results['walk_distance'] = {
    'avg': round(float(avg_walk), 1),
    'p99': round(float(walk_99), 0),
    'max': round(float(max_walk), 0),
    'distribution': [
        {'range': d, 'count': int(walk_buckets[walk_buckets['walk_range']==d]['count'].iloc[0])}
        if d in walk_buckets['walk_range'].values else {'range': d, 'count': 0}
        for d in walk_order
    ]
}

# 载具行驶距离
avg_ride = df.select(mean("rideDistance")).collect()[0][0]
ride_99 = df.selectExpr("percentile_approx(rideDistance, 0.99)").collect()[0][0]
max_ride = df.select(spark_max("rideDistance")).collect()[0][0]
zero_ride = df.filter(df["rideDistance"] == 0).count()
print(f"平均行驶距离: {avg_ride:.1f}m, 99%≤{ride_99:.0f}m, 最大: {max_ride:.0f}m")
print(f"无载具玩家: {zero_ride}人({zero_ride/total*100:.2f}%)")

ride_buckets = df.filter(col("rideDistance") < ride_99).select(
    when(col("rideDistance") == 0, "0m")
    .when(col("rideDistance") <= 500, "1-500m")
    .when(col("rideDistance") <= 2000, "501-2000m")
    .when(col("rideDistance") <= 5000, "2001-5000m")
    .otherwise("5000m+").alias("ride_range")
).groupBy("ride_range").count().orderBy("ride_range").toPandas()

ride_order = ["0m","1-500m","501-2000m","2001-5000m","5000m+"]
results['ride_distance'] = {
    'avg': round(float(avg_ride), 1),
    'p99': round(float(ride_99), 0),
    'max': round(float(max_ride), 0),
    'zero_pct': round(zero_ride/total*100, 2),
    'distribution': [
        {'range': d, 'count': int(ride_buckets[ride_buckets['ride_range']==d]['count'].iloc[0])}
        if d in ride_buckets['ride_range'].values else {'range': d, 'count': 0}
        for d in ride_order
    ]
}

# ====== 模块3：药品/增益物品与胜率关系 ======
print("\n========== 药品/增益物品与胜率关系 ==========")
df = df.withColumn("heals", df["heals"].cast(IntegerType()))
df = df.withColumn("boosts", df["boosts"].cast(IntegerType()))
df = df.withColumn("winPlacePerc", df["winPlacePerc"].cast(FloatType()))

avg_heal = df.select(mean("heals")).collect()[0][0]
heal_99 = df.selectExpr("percentile_approx(heals, 0.99)").collect()[0][0]
avg_boost = df.select(mean("boosts")).collect()[0][0]
boost_99 = df.selectExpr("percentile_approx(boosts, 0.99)").collect()[0][0]

print(f"平均治疗品: {avg_heal:.1f}个, 99%≤{heal_99}")
print(f"平均增益品: {avg_boost:.1f}个, 99%≤{boost_99}")

# 胜率与物品数量关系
items_df = df.filter((col("heals") < heal_99) & (col("boosts") < boost_99))
heal_win = items_df.groupBy("heals").agg(mean("winPlacePerc").alias("win_rate")) \
    .orderBy("heals").toPandas()
boost_win = items_df.groupBy("boosts").agg(mean("winPlacePerc").alias("win_rate")) \
    .orderBy("boosts").toPandas()

results['items_winrate'] = {
    'avg_heal': round(float(avg_heal), 1),
    'avg_boost': round(float(avg_boost), 1),
    'heal_win': [{'count': int(r['heals']), 'win_rate': round(float(r['win_rate']), 4)}
                 for _, r in heal_win.iterrows()],
    'boost_win': [{'count': int(r['boosts']), 'win_rate': round(float(r['win_rate']), 4)}
                  for _, r in boost_win.iterrows()]
}

# ====== 模块4：组队模式分析 ======
print("\n========== 组队模式分析 ==========")
df = df.withColumn("numGroups", df["numGroups"].cast(IntegerType()))
solos = df.filter(col("numGroups") > 50)
duos = df.filter((col("numGroups") > 25) & (col("numGroups") <= 50))
squads = df.filter(col("numGroups") <= 25)

solo_cnt = solos.count()
duo_cnt = duos.count()
squad_cnt = squads.count()
print(f"单人: {solo_cnt}({solo_cnt/total*100:.1f}%), "
      f"双人: {duo_cnt}({duo_cnt/total*100:.1f}%), "
      f"四人: {squad_cnt}({squad_cnt/total*100:.1f}%)")

# 各模式击杀-胜率关系
def mode_kill_winrate(mode_df, label):
    pdf = mode_df.groupBy("kills").agg(mean("winPlacePerc").alias("win_rate")) \
        .filter(col("kills") <= 15).orderBy("kills").toPandas()
    return [{'kills': int(r['kills']), 'win_rate': round(float(r['win_rate']), 4)}
            for _, r in pdf.iterrows()]

results['game_modes'] = {
    'solo': {'count': solo_cnt, 'ratio': round(solo_cnt/total*100, 1),
             'kill_winrate': mode_kill_winrate(solos, '单人')},
    'duo': {'count': duo_cnt, 'ratio': round(duo_cnt/total*100, 1),
            'kill_winrate': mode_kill_winrate(duos, '双人')},
    'squad': {'count': squad_cnt, 'ratio': round(squad_cnt/total*100, 1),
              'kill_winrate': mode_kill_winrate(squads, '四人')}
}

# 保存JSON结果
with open(os.path.join(OUTPUT_DIR, 'analysis_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n分析结果已保存到 outputs/analysis_results.json")

# 额外统计信息
stats = {
    'total_players': total,
    'avg_assists': float(df.select(mean("assists")).collect()[0][0]),
    'avg_headshotKills': float(df.select(mean("headshotKills")).collect()[0][0]),
    'avg_DBNOs': float(df.select(mean("DBNOs")).collect()[0][0]),
    'avg_weaponsAcquired': float(df.select(mean("weaponsAcquired")).collect()[0][0]),
    'avg_longestKill': float(df.select(mean("longestKill")).collect()[0][0]),
    'avg_matchDuration': float(df.select(mean("matchDuration")).collect()[0][0]),
    'avg_revives': float(df.select(mean("revives")).collect()[0][0]),
}
print("\n=== 额外统计 ===")
for k, v in stats.items():
    print(f"{k}: {v:.4f}")

with open(os.path.join(OUTPUT_DIR, 'extra_stats.json'), 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

sc.stop()
print("\n========== 分析完成! ==========")
