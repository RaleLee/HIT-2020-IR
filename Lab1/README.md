# HIT-2020-IR-Lab1
这是第一个实验，网页文本的预处理。
## Requirements
需要的库有 json bs4 re requests

还有pyltp，这个库的安装比较麻烦，这里选择的是py3.6的环境，比较好安装一些。还有静态模型，需要自己去网盘下载。

## 3.1 网页的抓取和正文提取
目标网页是哈工大深圳新闻网，有很多符合实验要求的网页，带文本文档的网页多。

直接运行craw.py，json格式的结果保存在results/full_data.json中。爬取到的文本文档文件保存在files/doc文件夹中。

## 3.2 分词处理、去停用词处理
停用词表为stopwords.txt，pyltp使用的静态模型路径请自行修改变量LTP_DATA_DIR

直接运行segment.py，json格式的结果保存在results/preprocessed.json中