import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

#Task 1
#iris dataset from sklearn
try:
  iris_data=load_iris()
  df=pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
  df['species']=df['species'].map(dict(zip(range(3),iris_data.target_names)))
  print("Dataset loaded successfully.")
except Exception as e:
  print("Error loading dataset:",e)

Display the first 5 rows
df.head()

Check data types and missing values
print(df.info())
print("\nMissing values:\n",df.isnull().sum())
