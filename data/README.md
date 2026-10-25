# 数据目录说明

仓库**不包含**原始数据文件，避免把 50MB 的 CSV 放进 Git 历史。

本项目使用的数据来自 Kaggle 竞赛：

- [PUBG Finish Placement Prediction](https://www.kaggle.com/c/pubg-finish-placement-prediction/data)
- 字段包括 `kills`、`damageDealt`、`walkDistance`、`winPlacePerc` 等 29 列。

## 本地运行

将数据文件放到：

```text
data/smaller_train.csv
```

该文件应包含与 Kaggle `train_V2.csv` 相同的表头。仓库实验使用约 **37 万行** 的抽样数据。

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

> 没有数据也可以直接启动 Web 看板：`outputs/` 中已保留预计算的分析与模型结果。
