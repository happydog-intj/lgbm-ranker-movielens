import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score

rng = np.random.default_rng(42)

# 1. 读取 MovieLens 100K
ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user", "item", "rating", "ts"])
users = pd.read_csv("ml-100k/u.user", sep="|", names=["user", "age", "gender", "occupation", "zip"])
genres = [f"g{i}" for i in range(19)]
items = pd.read_csv("ml-100k/u.item", sep="|", encoding="latin-1",
                    names=["item", "title", "date", "video", "url"] + genres)
items["year"] = pd.to_datetime(items["date"], format="%d-%b-%Y").dt.year

# 2. 按时间切分：每个用户最早 80% 的评分做训练，最后 20% 做测试
#    这样模拟“用过去的行为预测未来”，避免随机切分时用未来数据训练、过去数据测试造成的泄漏。
#    切分在每个用户内部进行，所以每个用户在训练集和测试集里都有数据。

# 先按用户排序，同一用户内部再按时间戳 ts 从早到晚排序。
# 后面的 cumcount 依赖这个顺序，才能得到“第几次评分”。
ratings = ratings.sort_values(["user", "ts"])

# pos：每条评分在该用户所有评分中的序号，从 0 开始。
# 例如某用户有 5 条评分，按时间顺序依次得到 0, 1, 2, 3, 4。
# 结果是与 ratings 行一一对齐的 Series（索引相同）。
pos = ratings.groupby("user").cumcount()

# cnt：该用户的评分总数，用 transform 广播回每一行。
# 上例中这 5 行的 cnt 都是 5。
# 和 agg 不同，transform 返回与原表等长的结果，可以直接和 pos 逐行比较。
cnt = ratings.groupby("user")["item"].transform("size")

# pos < cnt * 0.8：序号落在前 80% 的评分进训练集，其余进测试集。
# 上例中 cnt * 0.8 = 4，序号 0~3 进训练集，序号 4（最新一条）进测试集。
# 两个布尔条件互补，所以每条评分恰好落在一边，不重不漏。
# 评分数很少的用户也至少有 1 条在训练集（pos=0 时 0 < cnt*0.8 恒成立）；
# 但如果只有 1 条评分，测试集里就没有这个用户。
train_r, test_r = ratings[pos < cnt * 0.8], ratings[pos >= cnt * 0.8]

# 3. 构造候选集：用户看过的电影 + 随机抽的没看过的电影（负样本）
#    label：没看过 = 0，看过 = 评分 1~5
all_items = items["item"].to_numpy()
seen = ratings.groupby("user")["item"].apply(set)

def candidates(r, n_neg=50):
    neg = []
    for u in r["user"].unique():
        pool = np.setdiff1d(all_items, list(seen[u]))
        neg.append(pd.DataFrame({"user": u, "item": rng.choice(pool, n_neg, replace=False)}))
    neg = pd.concat(neg).assign(rating=0)
    return pd.concat([r[["user", "item", "rating"]], neg])

# 4. 特征：用户画像、电影属性、训练期统计量
#    所有统计量只用训练集 train_r 计算，训练和测试共用同一份。
#    如果用全部评分计算，测试期的评分会混进特征里，评估结果会虚高（数据泄漏）。

# 电影统计量，每部电影一行，索引是 item：
#   item_mean：训练期平均评分，反映电影口碑
#   item_cnt ：训练期评分人数，反映电影热度（也是热度基线用的排序依据）
item_stats = train_r.groupby("item")["rating"].agg(item_mean="mean", item_cnt="size")

# 用户统计量，每个用户一行，索引是 user：
#   user_mean：用户平均打分，区分打分偏松和偏严的用户
#   user_cnt ：用户评分数，反映用户活跃度
user_stats = train_r.groupby("user")["rating"].agg(user_mean="mean", user_cnt="size")

# 用户对各类型的偏好：训练期看过的电影中每个类型的占比。
# items 的 g0~g18 是 0/1 列，表示电影是否属于某个类型（一部电影可属于多个类型）。
# 先把评分记录拼上电影类型，再按用户求均值，0/1 列的均值就是占比。
# 例如 u_g5=0.3 表示该用户看过的电影里 30% 属于类型 g5。
# add_prefix("u_") 给列名加前缀，避免和电影自己的 g0~g18 列重名。
user_genre = (train_r.merge(items[["item"] + genres], on="item")
              .groupby("user")[genres].mean().add_prefix("u_"))

def build(r):
    """把评分记录 r 变成模型输入：每行是一个（用户, 候选电影）对，带特征和 label。"""
    # 先生成候选集（正样本 + 随机负样本），再依次拼上各类特征：
    #   users     ：用户画像，年龄、性别、职业
    #   items     ：电影属性，上映年份和 19 个类型列
    #   item_stats：电影统计量。用 how="left" 是因为有些电影训练期没人评分，
    #               这些电影仍要保留，统计量记为 NaN
    #   user_stats、user_genre：用户统计量和类型偏好
    df = (candidates(r)
          .merge(users, on="user")
          .merge(items[["item", "year"] + genres], on="item")
          .merge(item_stats, on="item", how="left")
          .merge(user_stats, on="user", how="left")
          .merge(user_genre, on="user", how="left"))

    # 训练期没人评分的电影，评分人数就是 0，用 0 填充。
    # item_mean 保留 NaN："没有评分"和"平均 0 分"含义不同，LightGBM 能直接处理缺失值。
    df["item_cnt"] = df["item_cnt"].fillna(0)

    # 用户类型偏好与电影类型的匹配度，即用户偏好向量和电影类型向量的点积：
    #   genre_match = Σ_i  电影是否属于类型 i × 用户看过的电影中类型 i 的占比
    # 电影越符合用户常看的类型，这个值越大。
    # 例如用户 60% 看剧情片、40% 看喜剧，一部"剧情+喜剧"电影得 1.0，纯动作片得 0。
    # to_numpy() 把两组列转成同形状的矩阵，逐元素相乘后按行求和（sum(1)）。
    df["genre_match"] = (df[genres].to_numpy() * df["u_" + pd.Index(genres)].to_numpy()).sum(1)

    # 性别转成 0/1 数值：男 = 1，女 = 0。
    df["gender"] = (df["gender"] == "M").astype(int)

    # 职业是字符串（student、engineer 等），转成 pandas 的 category 类型。
    # LightGBM 会自动把 category 列当类别特征处理，不需要手动 one-hot。
    df["occupation"] = df["occupation"].astype("category")

    # LGBMRanker 要求同一 query（用户）的行连续排列，group 参数才能对得上。
    # kind="stable" 是稳定排序，保持同一用户内部原来的行顺序。
    return df.sort_values("user", kind="stable")

train, test = build(train_r), build(test_r)

# 模型使用的特征列：
#   用户画像：age、gender、occupation
#   电影属性：year、g0~g18（19 个类型）
#   统计量  ：item_mean、item_cnt、user_mean、user_cnt
#   交叉特征：genre_match（用户偏好与电影类型的匹配度）
# user、item 这两个 ID 不放进特征，否则模型会去记具体 ID，泛化差。
features = ["age", "gender", "occupation", "year", "item_mean", "item_cnt",
            "user_mean", "user_cnt", "genre_match"] + genres

# 5. 训练：每个用户是一个 query（一组候选电影），模型只比较同一用户内部的电影顺序

# group：每个用户的候选行数，按 train 中用户出现的顺序排列。
# 例如 [120, 85, ...] 表示前 120 行属于第 1 个用户，接下来 85 行属于第 2 个用户。
# sort=False 保持 train 里的行顺序，确保 group 和数据行一一对应；
# group 的总和必须等于 train 的行数。
group = train.groupby("user", sort=False).size().to_numpy()

# 模型参数：
#   objective="lambdarank"：pairwise 排序目标，直接优化 NDCG
#   n_estimators=300      ：树的数量（提升轮数）
#   learning_rate=0.05    ：每棵树的步长，越小越稳，通常需要更多树
#   num_leaves=31         ：每棵树最多的叶子数，控制模型复杂度
#   verbose=-1            ：关闭训练日志
model = LGBMRanker(objective="lambdarank", n_estimators=300,
                   learning_rate=0.05, num_leaves=31, verbose=-1)

# label 是 rating：没看过 = 0，看过 = 1~5。lambdarank 要求 label 是非负整数，
# 数值越大越相关。默认收益 2^label - 1，所以 5 分电影排在前面的收益远大于 1 分电影。
model.fit(train[features], train["rating"], group=group)

# 6. 评估：和“按热度（评分人数）推荐”的基线对比

# predict 输出排序分数，没有具体含义，只在同一用户内部比较大小才有意义。
test["score"] = model.predict(test[features])

def mean_ndcg(df, col, k=10):
    """对每个用户按 col 列排序，计算 NDCG@k，再对所有用户取平均。

    NDCG@k 衡量前 k 个位置的排序质量，取值 0~1，1 表示和按真实评分排序完全一致。
    ndcg_score 要求二维输入（每行一个 query），所以把单个用户的数据包成 [列表]。
    """
    return np.mean([ndcg_score([g["rating"]], [g[col]], k=k)
                    for _, g in df.groupby("user")])

# 基线：直接按电影训练期的评分人数 item_cnt 排序，即“推荐最热门的电影”。
# 模型要比这个基线高，才说明特征和训练带来了价值。

print(f"热度基线   NDCG@10: {mean_ndcg(test, 'item_cnt'):.4f}")
print(f"LGBMRanker NDCG@10: {mean_ndcg(test, 'score'):.4f}")

# 7. 给用户 1 出 Top5 推荐
#    取该用户候选中分数最高的 5 部电影，再拼上片名方便查看。
#    rating 列是真实结果：1~5 表示用户测试期真看了并打了这个分，0 表示用户其实没看过。
top = test[test.user == 1].nlargest(5, "score").merge(items[["item", "title"]], on="item")
print(top[["title", "score", "rating"]].to_string(index=False))
