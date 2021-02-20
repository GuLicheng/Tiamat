import csv
from os import read

with open("./a.csv", "r", newline="") as cvsfile:
    reader = csv.DictReader(cvsfile)
    print(reader)