import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
data = {
    'Teor Alcoólico': [3, 4, 5, 6],
    'Acidez': ['muito', 'pouco', 'médio', 'baixo'],
    'pH': [4.3, 2.8, 4.2, 3.9]
}

data = pd.DataFrame(data)

matriz = data[['Teor Alcoólico', 'pH']].values

print(matriz)
