# 数据目录说明

本仓库已包含实验数据：

```text
data/smaller_train.csv
```

- 约 **50MB**
- 约 **37 万行**
- 字段与 Kaggle `train_V2.csv` 一致，共 29 列
- 数据来源：[PUBG Finish Placement Prediction](https://www.kaggle.com/c/pubg-finish-placement-prediction/data)

包含的字段有 `kills`、`damageDealt`、`walkDistance`、`winPlacePerc` 等。数据使用请遵循 Kaggle 竞赛的数据条款。

## 直接运行

克隆仓库后不需要再下载数据，直接执行：

```powershell
python src/analysis.py
python src/train_model.py
```

## 替换成自己的数据

如果文件名或路径不同，可以设置环境变量：

```powershell
$env:PUBG_DATA_PATH = "D:\data\smaller_train.csv"
python src/analysis.py
python src/train_model.py
```

也支持 HDFS 路径：

```powershell
$env:PUBG_DATA_PATH = "hdfs://hadoop01:9000/data/data_train.csv"
```

> 只看 Web 看板时不需要数据：`outputs/` 中已保留预计算的分析与模型结果。
