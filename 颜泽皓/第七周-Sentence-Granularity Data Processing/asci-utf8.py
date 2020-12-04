import os

IN_PATH = "G:/export/JHTRD"
OUT_PATH = "G:/export/JHTRD2"
for filename in os.listdir(IN_PATH):
    with open(os.path.join(IN_PATH, filename), encoding="ANSI") as ifile:
        txt = ifile.read()
        with open(os.path.join(OUT_PATH, filename), encoding="utf-8", mode="w") as ofile:
            ofile.write(txt)
