import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = {
    'dia_semana': [1, 2, 3, 4, 5, 6, 7],
    'nº de clientes': [15, 20, 30, 100, 350, 500, 700]
}

data = pd.DataFrame(data)

X = data[['dia_semana','nº de clientes']].values


kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

data['Cluster'] = kmeans.labels_

print(data)
