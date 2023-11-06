import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = {
    'Máquina': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10'],
    'Temperatura (°C)': [70.2, 65.1, 75.5, 80.3, 68.7, 72.9, 78.6, 66.4, 73.1, 69.5],
    'Vibração': [12.5, 8.2, 15.6, 10.2, 11.8, 14.3, 9.8, 8.9, 13.7, 12.1],
    'Corrente': [4.7, 3.9, 5.1, 4.5, 4.2, 5.3, 4.8, 4.0, 5.0, 4.3]
}

data = pd.DataFrame(data)
X = data[['Temperatura (°C)', 'Vibração', 'Corrente']].values

kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
data['Cluster'] = kmeans.labels_

for i in ['Temperatura (°C)', 'Vibração', 'Corrente']:
    print(i)
    data.boxplot(column=i, by='Cluster')
    plt.title(f'Boxplot de {i} por Cluster')
    plt.suptitle('')
    plt.show()
