import json
import os
from queue import Queue
from threading import Thread
import requests
import re

from bs4 import BeautifulSoup


def build_path():
    """
    建立文件存放路径
    :return: no returns
    """
    if not os.path.exists("files"):
        os.mkdir("files")
        os.mkdir("files/doc")
        os.mkdir("results")
        print("Finish making paths")
        return
    print("Path already exists!")


class Spider:
    """
    爬虫类
    """
    def __init__(self):
        self.queue = Queue()
        self.results = []
        self.thread_num = 10

    def get_urls(self):
        """
        指定urls集合 1000个 从哈工大深圳新闻网上进行下载
        :return: urls 列表
        """
        # 是否已经有缓存好的url
        if os.path.exists("results/data.json"):
            with open("results/data.json", "r", encoding="utf-8") as f:
                urls = f.readlines()
                if len(urls) > 1000:
                    for url in urls:
                        url_dict = json.loads(url)
                        self.queue.put(url_dict)
                    return

        aps = []
        for i in range(51):
            target_url = "http://www.hitsz.edu.cn/article/id-74.html?maxPageItems=20&keywords=&pager.offset=" + str(
                i * 20)
            r = requests.get(target_url, timeout=100)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, "html.parser")
            li = soup.find_all('a')
            for url in li:
                if "view" in url.attrs["href"]:
                    aps.append(url)
        urls = []
        i = 0
        for url in aps:
            page = {"index": str(i), "url": "http://www.hitsz.edu.cn" + url.attrs["href"],
                    "title": str(url.string).strip()}
            i += 1
            urls.append(page)
        urls = urls[1:]
        with open("results/data.json", "w", encoding="utf-8") as j:
            for url in urls:
                self.queue.put(url)
                j.write(json.dumps(url, ensure_ascii=False) + "\n")
        return

    def craw(self):
        """
        提取网页正文及附件，json格式的字符串保存在results中
        :return:
        """
        while not self.queue.empty():
            page = self.queue.get()
            url = page.get("url")
            r = requests.get(url, timeout=100)
            r.raise_for_status()
            r.encoding = r.apparent_encoding
            soup = BeautifulSoup(r.text, "html.parser")
            li = soup.find('div', class_='detail')

            # <class 'bs4.element.Tag'>
            str_page = str(li)
            pattern = re.compile(r"<[^>]+>", re.S)
            res = pattern.sub("\t", str_page)
            page["paragraphs"] = res
            page["file_name"] = []
            files = li.select("a[href]")
            for file in files:
                href = file.get("href")
                title = file.get("title")
                if title is None:
                    title = file.get("download")
                if title is None:
                    print(file)
                    continue
                print(href + " " + title)
                if ".rar" in title or ".zip" in title or ".pdf" in title:
                    continue
                if "http://" not in href:
                    href = "http://www.hitsz.edu.cn" + href
                title = page["index"] + "_" + title
                path = os.path.join("files/doc/", title)
                page["file_name"].append(path)
                response = requests.get(href, stream=True)
                with open(path, "wb") as f:
                    f.write(response.content)

            self.results.append(page)

    def write_result(self):
        """
        将最终结果写入到文件中
        :return:
        """
        with open("results/full_data.json", "w", encoding="utf-8") as f:
            for page in self.results:
                page.pop("index")
                f.write(json.dumps(page, ensure_ascii=False) + "\n")

    def run(self):
        """
        执行程序，提供多线程支持
        :return:
        """
        self.get_urls()

        ths = []
        for _ in range(self.thread_num):
            th = Thread(target=self.craw)
            th.start()
            ths.append(th)
        for th in ths:
            th.join()
        self.write_result()


if __name__ == "__main__":
    build_path()
    sp = Spider()
    sp.run()
