import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

print(sns.get_dataset_names())
df = sns.load_dataset("penguins")

print(df.head(8))
print(df.shape)
print(df.tail())
print(df.isnull().sum())
print(df.describe())
print(df.dtypes)
print(df.info())
print(df.describe(include="all"))

numeric_df = df.select_dtypes(include=[np.number])

print(numeric_df.corr())
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()

numeric_df.hist(figsize=(12, 8))
plt.show()

numeric_df.plot(kind='box', subplots=True, layout=(1, 3), figsize=(10, 6))
plt.show()

print(df['sex'].value_counts())
print(df['island'].value_counts())
print(df['species'].value_counts())

sns.countplot(data=df, x='sex', palette='Set2')
plt.show()

sns.countplot(data=df, x='island', palette='Set2')
plt.show()

sns.countplot(data=df, x='species', palette='Set2')
plt.show()

sns.countplot(data=df, x='sex', hue='species', palette='pink')
plt.show()

sns.countplot(data=df, x='island', hue='sex', palette='lavender')
plt.show()

sns.pairplot(data=df, hue='species', palette='Set2')
plt.show()