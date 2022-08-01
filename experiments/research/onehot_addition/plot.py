import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sizes = ["31", "63", "94", "127", "158", "190", "221", "255", "286", "318", "349", "382", "413", "445",
         "476", "511", "542", "574", "605", "638", "669", "701", "732", "766", "797", "829", "860", "893", "924"]

data = np.load("results_mvg/q_values.npy")
data_2 = np.load("results_mvg3/q_values.npy")
data_3 = np.load("results_mvg4/q_values.npy")
data_4 = np.load("results_none/q_values.npy")
data_5 = np.load("results_none3/q_values.npy")
data_6 = np.load("results_none4/q_values.npy")

delta = np.mean(data, axis=1) - np.mean(data_2, axis=1)

means = np.mean(data, axis=1)
import pandas as pd

annotated_data = []
indexes = []

for i, s in enumerate(sizes):
    for q in data[i]:
        indexes.append(i)
        annotated_data.append([s, q, "MVG"])

for i, s in enumerate(sizes):
    for q in data_2[i]:
        indexes.append(i)
        annotated_data.append([s, q, "MVG"])

for i, s in enumerate(sizes):
    for q in data_3[i]:
        indexes.append(i)
        annotated_data.append([s, q, "MVG"])

for i, s in enumerate(sizes):
    for q in data_4[i]:
        indexes.append(i)
        annotated_data.append([s, q, "No MVG"])

for i, s in enumerate(sizes):
    for q in data_5[i]:
        indexes.append(i)
        annotated_data.append([s, q, "No MVG"])

for i, s in enumerate(sizes):
    for q in data_6[i]:
        indexes.append(i)
        annotated_data.append([s, q, "No MVG"])

df = pd.DataFrame(
    columns=["neurons", "Q", "Method"],
    data=annotated_data
)
regressor = LinearRegression()
regressor.fit(np.array(indexes).reshape(-1, 1), df["Q"])

plt.figure(figsize=(15, 10))
plt.suptitle("Modularity vs Model size", size='20')
sns.lineplot(data=df, x="neurons", y="Q", ci="sd", hue="Method")
# sns.lineplot(x=np.linspace(0, len(delta)-1, num=len(delta)), y=delta)
plt.xlabel('Model size in neurons', size=16)
plt.ylabel('Modularity Q', size=16)
plt.show()
