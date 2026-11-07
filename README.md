# LGBMRanker 电影推荐示例

用 MovieLens 100K 数据集训练 `LGBMRanker`，为每个用户的候选电影排序，并与热度基线对比 NDCG@10。

## 运行

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
python rank_movielens.py
```

macOS 上如果 `import lightgbm` 报 `Library not loaded: @rpath/libomp.dylib`，先执行 `brew install libomp`。

## 文件

- `download_data.py`：下载并解压数据到 `ml-100k/`
- `rank_movielens.py`：构造候选集和特征、训练、评估、输出用户 1 的 Top5 推荐
