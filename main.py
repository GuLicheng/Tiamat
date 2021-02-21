import csv


with open("a.csv", "r", newline="") as cvsfile:
    reader = csv.DictReader(cvsfile)
    print(reader.reader.line_num)