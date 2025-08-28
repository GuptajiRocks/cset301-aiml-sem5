import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Building a self Decision tree Classifier from the start

df = pd.read_excel("aug28//rd.xlsx")
coll = list(df.columns)

lr = LabelEncoder()
df["Class"] = lr.fit_transform(df["Class"])

# Kecimen = 1; Besni = 0

kdf = df[df["Class"] == 1]
bdf = df[df["Class"] == 0]

b_gini_score = (bdf["Class"].count()/df["Class"].count())
