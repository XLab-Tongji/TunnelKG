import pdfplumber as pp
import _io
import os 
import threading
import re

IN_PATH = "G:\\export\\CRS\\"
OUT_PATH = "G:\\txt\\CRS\\"
ERR_PATH = "G:\\txt\\"
KWDS = ["摘要", "关键词", "中图分类号", "文献标志码", "文章编号", "引言"]

def TextProcess(ifilename, errfile, eflock):
    ifile :_io.TextIOWrapper = open(IN_PATH + ifilename, encoding="utf-8")
    txt :str = ifile.read()
    
    txt = re.sub('[^\u4e00-\u9fff]','',txt)
    start = txt.find("摘要")
    end = txt.find("参考文献")
    txt = txt[start:end]
    for kwd in KWDS:
        txt = txt.replace(kwd, "")
    if len(txt):
        ofile = open(OUT_PATH + ifilename, "w", encoding="utf-8")
        ofile.write(txt)
        print("[FIN]" + ifilename)
    else:
        eflock.acquire()
        errfile.write(ifilename+ "\n")
        print("[ERROR]" + ifilename)
        eflock.release()

def main():
    efile = open(ERR_PATH + "errors.txt", "a", encoding="utf-8")
    ef_lock = threading.Lock()
    ths = []
    for filename in os.listdir(IN_PATH):
        th = threading.Thread(target=TextProcess,
                             args=(filename, efile, ef_lock))
        ths.append(th)
        th.start()
    for th in ths:
        th.join()

if __name__ == "__main__":
    main()