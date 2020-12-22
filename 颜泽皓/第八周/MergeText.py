import os


IN_SDIR = "E:/文档/作业/隧道知识图谱/txt/source/TC"
IN_TDIR = "E:/文档/作业/隧道知识图谱/txt/target/TC"
OUT_NAME = "TC"
OUT_DIR = "E:/文档/作业/隧道知识图谱/txt/"


def main() -> None:
    o_spath = os.path.join(OUT_DIR, str().join(["source_", OUT_NAME, ".txt"]))
    o_tpath = os.path.join(OUT_DIR, str().join(["target_", OUT_NAME, ".txt"]))
    with open(o_spath, mode="w", encoding="utf-8") as o_source:
        with open(o_tpath, mode="w", encoding="utf-8") as o_target:
            for filename in os.listdir(IN_SDIR):
                ifile = open(os.path.join(IN_SDIR, filename), encoding="utf-8")
                o_source.write(ifile.read())
                ifile = open(os.path.join(IN_TDIR, filename), encoding="utf-8")
                o_target.write(ifile.read())
            
if __name__ == "__main__":
    main()
