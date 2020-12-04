from os import makedirs
import os
import threading
import re

IN_PATH = "G:\\export\\"
OUT_PATH = "E:\\文档\\作业\\隧道知识图谱\\txt\\txt2"
KWDS = ["摘要", "关键词", "中图分类号", "文献标志码", "文章编号", "引言"]


def TextProcess(ifilename, ipath, opath):
    with open(os.path.join(ipath, ifilename), encoding="utf-8") as ifile:
        txt: str = ifile.read()

        txt = re.sub(r"[^\u4e00-\u9fff"
                     r"\u3002\uFF1F\uFF01\u3010\u3011\uFF0C\u3001\uFF1B"
                     r"\uFF1A\u300C\u300D\u300E\u300F\u2019\u201C\u201D"
                     r"\u2018\uFF08\uFF09\u3014\u3015\u2026\u2013\uFF0E"
                     r"\u2014\u300A\u300B\u3008\u3009]", "", txt)
        start = txt.find("摘要")
        end = txt.find("参考文献")
        txt = txt[start:end]
        for kwd in KWDS:
            txt = txt.replace(kwd, "")
        if len(txt):
            ofile = open(os.path.join(opath, ifilename), "w", encoding="utf-8")
            ofile.write(txt)


def main():
    ths = []
    for folder in os.listdir(IN_PATH):
        ipath = os.path.join(IN_PATH, folder)
        opath = os.path.join(OUT_PATH, folder)
        if not os.path.exists(opath):
            makedirs(opath)
        for filename in os.listdir(ipath):
            th = threading.Thread(target=TextProcess,
                                  args=(filename, ipath, opath))
            ths.append(th)
            th.start()
    for th in ths:
        th.join()


if __name__ == "__main__":
    main()
