import dill
import pandas as pd

with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = dill.load(f)

df = pd.read_csv("artifacts/test.csv").drop(columns=["math_score"])
print(preprocessor.transform(df.head()).shape)