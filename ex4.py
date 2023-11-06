import pandas as pd
from sklearn.cluster import KMeans

data = {
    'Substância': ['Álcool', 'Gasolina', 'Leite', 'Querosene', 'Óleo', 'Vinho'],
    'Concentração (%)': [12.5, 0.1, 4.0, 1.2, 0.5, 15.0],
    'Teor Alcoólico (%)': [50, 0.05, 0.01, 0.02, 0.01, 12.5]
}

data = pd.DataFrame(data)

X = data[['Concentração (%)', 'Teor Alcoólico (%)']].values


kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

data['Cluster'] = kmeans.labels_

print(data)
