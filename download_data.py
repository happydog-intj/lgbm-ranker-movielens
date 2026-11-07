"""下载 MovieLens 100K 数据集到当前目录的 ml-100k/。"""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

if Path("ml-100k/u.data").exists():
    print("ml-100k 已存在，跳过下载")
else:
    print(f"下载 {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(".")
    print("完成：ml-100k/")
