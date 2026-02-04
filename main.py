import pandas as pd

detection = pd.read_csv("Data/Detection.csv")
identification = pd.read_csv("Data/Identification.csv")

print("First 3 rows from the DETECTION DATASET")
print(detection.head(3))

print()

print("First 3 rows from the IDENTIFICATION DATASET")
print(identification.head(3))
