
# coding: utf-8
# Lê dados do dataset diamonds em salva em formato apropriado para a análise


import pandas as pd
from pandas.api.types import CategoricalDtype

traindata = pd.read_csv('diamonds.csv', header=0)

# ### Converter coluna 'clarity' em categorias
clarity_types  = CategoricalDtype(categories=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'], ordered=True)
traindata.clarity = traindata.clarity.astype(clarity_types)
traindata['clarity_code'] = traindata.clarity.cat.codes

# ### Converter coluna 'cut' em categorias 
cut_types  = CategoricalDtype(categories=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'], ordered=True)
traindata.cut = traindata.cut.astype(cut_types)
traindata['cut_code'] = traindata.cut.cat.codes

# ### Converter coluna 'color' em categorias
color_types  = CategoricalDtype(categories=['J', 'I', 'H', 'G', 'F', 'E', 'D'], ordered=True)
traindata.color = traindata.color.astype(color_types)
traindata['color_code'] = traindata.color.cat.codes

# ## Gerar dataset para o modelo
new_dataset = traindata[['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_code', 'color_code', 'clarity_code', 'price']]
new_dataset.to_csv('diamond_prices.csv', index=False)

