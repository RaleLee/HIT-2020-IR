import json
import os

import requests
import re

from bs4 import BeautifulSoup

HEADER = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/81.0.4044.129 Safari/537.36"
}


def get_urls():
    """
    指定urls集合 1000个 从今日哈工大上进行下载
    :return: urls 列表
    """

    aps = []
    for i in range(51):
        target_url = "http://www.hitsz.edu.cn/article/id-74.html?maxPageItems=20&keywords=&pager.offset=" + str(i * 20)
        r = requests.get(target_url, timeout=100)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        soup = BeautifulSoup(r.text, "html.parser")
        li = soup.find_all('a')
        for url in li:
            if "view" in url.attrs["href"]:
                # print(url)
                # assert isinstance(url, object)
                aps.append(url)
    urls = []

    for url in aps:
        page = {"url": "http://www.hitsz.edu.cn" + url.attrs["href"], "title": str(url.string).strip()}
        urls.append(page)
    urls = urls[1:]
    with open("results/data.json", "w", encoding="utf-8") as j:
        for url in urls:
            j.write(json.dumps(url, ensure_ascii=False) + "\n")

    return urls


def craw(url):
    """
    提取网页正文及附件，返回一个字典对象
    :param url:将要爬取的url列表
    :return: results 字典对象
    """
    results = {}

    return results


def build_path():
    """
    建立文件存放路径
    :return: no returns
    """
    if not os.path.exists("files"):
        os.mkdir("files")
        os.mkdir("files/img")
        os.mkdir("files/doc")
        os.mkdir("results")
        print("Finish making paths")
        return
    print("Path already exists!")


if __name__ == "__main__":
    build_path()
    # need_urls = get_urls()
    # for url in need_urls:
    #     craw(url)
    url = "http://www.hitsz.edu.cn/article/view/id-87929.html"
    r = requests.get(url, timeout=100)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text, "html.parser")
    li = soup.find('div', class_='detail')

    # <class 'bs4.element.Tag'>
    page = str(li)
    # print(page)
    # print(type(page))
    pattern = re.compile(r"<[^>]+>", re.S)
    res = pattern.sub("\t", page)
    files = li.select("a[href]")
    # print(page)
    for file in files:
        href = file.get("href")
        title = file.get("title")
        print(href + " " + title)
        if "http://" not in href:
            href = "http://www.hitsz.edu.cn" + href
        path = os.path.join("files/doc/", title)
        response = requests.get(href, stream=True)
        with open(path, "wb") as f:
            f.write(response.content)
    #     sou = BeautifulSoup(li[0], "html.parser")
    #     docs = sou.find_all('a')
    #     for doc in docs:
    #         print(doc.attr["href"])

    print("need to build")
