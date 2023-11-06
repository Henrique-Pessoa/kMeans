import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

data = {
    'Setores': [1, 2, 3, 4, 5, 6, 7, 8],
    'nº de produtos fabricados': [100, 50, 15, 200, 500, 1000, 375, 450]
}

data = pd.DataFrame(data)

X = data[['nº de produtos fabricados']].values


kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

data['Cluster'] = kmeans.labels_

print(data)
