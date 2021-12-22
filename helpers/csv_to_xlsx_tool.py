import os
import pandas as pd

# src = "C:\\Users\\geiss\\OneDrive\\Desktop\\logs\\cluster_16_8_20"

# src = "C:\\Users\\geiss\\OneDrive\\Desktop\\15.9.20\\reports"
from helpers.csvTools import csv_to_xlsx

src = "my_evals"
# src = "C:\\Users\\geiss\\OneDrive\\Desktop\\\sonogram_ocv"
fns = os.listdir(src)
N = len(fns)
print("Number of files total: ", N)

for i in range(N):
    f = fns[i]
    if f.endswith(".csv"):
        fp = src + os.sep + f

        df = pd.read_csv(fp, header=0, delimiter=";")
        df = df.drop_duplicates()
        df.to_csv(fp, sep=";", index=False)

        csv_to_xlsx(fp)

        print("Converting file " + str(i) + ": ", f, " done")
