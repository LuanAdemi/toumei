import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

df = pd.DataFrame(data={
    'Q with MVG': [0.218, 0.227, 0.19, 0.156, 0.141, 0.129],
    'Q without MVG': [0.096, 0.114, 0.076, 0.08, 0.062, 0.097],
    'ΔQ': [0.122, 0.113, 0.114, 0.076, 0.079, 0.032]
     }, index=["31", "59", "121", "241", "481", "961"])

sns.lineplot(data=df)
plt.xlabel('Model size in neurons', size=12)
plt.ylabel('Modularity Q', size=12)
plt.show()