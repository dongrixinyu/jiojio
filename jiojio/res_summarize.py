import numpy as np
from .config import Config
import os


def tomatrix(s):
    lines = s.split(Config.lineEnd)
    lst = []
    for line in lines:
        if line == "":
            continue
        if not line.startswith("%"):
            tmp = []
            for i in line.split(Config.comma):
                tmp.append(float(i))
            lst.append(tmp)
    return np.array(lst)


def summarize(config):
    with open(os.path.join(config.outDir, config.fResRaw), encoding="utf-8") as sr:
        txt = sr.read()
    txt = txt.replace("\r", "")
    regions = txt.split(config.triLineEnd)

    with open(os.path.join(config.outDir, config.fResSum), "w", encoding="utf-8") as sw:
        for region in regions:
            if region == "":
                continue

            blocks = region.split(config.biLineEnd)
            mList = []
            for im in blocks:
                mList.append(tomatrix(im))

            avgM = np.zeros_like(mList[0])
            for m in mList:
                avgM = avgM + m
            avgM = avgM / len(mList)

            sqravgM = np.zeros_like(mList[0])
            for m in mList:
                sqravgM += m * m
            sqravgM = sqravgM / len(mList)

            deviM = (sqravgM - avgM * avgM) ** 0.5

            sw.write("#averaged values:\n")
            for i in range(avgM.shape[0]):
                for j in range(avgM.shape[1]):
                    sw.write("{:.2f},".format(avgM[i, j]))
                sw.write("\n")

            sw.write("\n#deviations:\n")
            for i in range(deviM.shape[0]):
                for j in range(deviM.shape[1]):
                    sw.write("{:.2f},".format(deviM[i, j]))
                    # sw.write(("%.2f" % deviM[i, j]) + ",")
                sw.write("\n")

            sw.write("\n#avg & devi:\n")
            for i in range(avgM.shape[0]):
                for j in range(avgM.shape[1]):
                    sw.write("{:.2f}+-{:,2f},".format(avgM[i, j], deviM[i, j]))
                sw.write("\n")

            sw.write("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n")
