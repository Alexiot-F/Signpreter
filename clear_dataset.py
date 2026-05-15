import os

path = "dataset/data.csv"

if os.path.exists(path):
    open(path, "w").close()

print("Dataset cleared")