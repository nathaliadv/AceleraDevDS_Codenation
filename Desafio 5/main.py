#!/usr/bin/env python
# coding: utf-8

# # Desafio 5
# 
# Neste desafio, vamos praticar sobre redução de dimensionalidade com PCA e seleção de variáveis com RFE. Utilizaremos o _data set_ [Fifa 2019](https://www.kaggle.com/karangadiya/fifa19), contendo originalmente 89 variáveis de mais de 18 mil jogadores do _game_ FIFA 2019.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[66]:


from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats as st
from sklearn.decomposition import PCA

from loguru import logger


# In[67]:


# Algumas configurações para o matplotlib.
#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[68]:


fifa = pd.read_csv("fifa.csv")


# In[69]:


columns_to_drop = ["Unnamed: 0", "ID", "Name", "Photo", "Nationality", "Flag",
                   "Club", "Club Logo", "Value", "Wage", "Special", "Preferred Foot",
                   "International Reputation", "Weak Foot", "Skill Moves", "Work Rate",
                   "Body Type", "Real Face", "Position", "Jersey Number", "Joined",
                   "Loaned From", "Contract Valid Until", "Height", "Weight", "LS",
                   "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM", "CAM", "RAM", "LM",
                   "LCM", "CM", "RCM", "RM", "LWB", "LDM", "CDM", "RDM", "RWB", "LB", "LCB",
                   "CB", "RCB", "RB", "Release Clause"
]

try:
    fifa.drop(columns_to_drop, axis=1, inplace=True)
except KeyError:
    logger.warning(f"Columns already dropped")


# ## Inicia sua análise a partir daqui

# In[70]:


# Sua análise começa aqui.

fifa.head()


# In[71]:


fifa.shape


# In[72]:


fifa.info()


# In[73]:


#Criando um dataframe auxiliar para analisar a consistencia das variaveis
aux = pd.DataFrame({'colunas' : fifa.columns,
                    'tipo': fifa.dtypes,
                    'missing' : fifa.isna().sum(),
                    'size' : fifa.shape[0],
                    'unicos': fifa.nunique()})
aux


# In[74]:


fifa.dropna(inplace= True)


# In[75]:


fifa.isna().sum()


# In[76]:


fifa.corr()


# In[77]:


pca = PCA()
pca.fit(fifa) 
print(np.matrix(pca.components_))


# In[78]:


evr = pca.explained_variance_ratio_
evr


# In[79]:


g = sns.lineplot(np.arange(len(evr)), np.cumsum(evr))
g.axes.axhline(0.95, ls="--", color="red")
plt.xlabel('Numero de componentes')
plt.ylabel('Variância explicada acumulada');


# ## Questão 1
# 
# Qual fração da variância consegue ser explicada pelo primeiro componente principal de `fifa`? Responda como um único float (entre 0 e 1) arredondado para três casas decimais.

# In[80]:


def q1():
    pca = PCA() 
    pca.fit_transform(fifa) # fit_transform para centralizar os dados de entrada
    evr = pca.explained_variance_ratio_
    return float(round(evr[0],3))


# In[81]:


q1()


# ## Questão 2
# 
# Quantos componentes principais precisamos para explicar 95% da variância total? Responda como un único escalar inteiro.

# In[82]:


def q2():
    pca_095 = PCA(n_components=0.95)
    var_095 = pca_095.fit_transform(fifa)
    return var_095.shape[1]


# In[83]:


q2()


# ## Questão 3
# 
# Qual são as coordenadas (primeiro e segundo componentes principais) do ponto `x` abaixo? O vetor abaixo já está centralizado. Cuidado para __não__ centralizar o vetor novamente (por exemplo, invocando `PCA.transform()` nele). Responda como uma tupla de float arredondados para três casas decimais.

# In[84]:


x = [0.87747123,  -1.24990363,  -1.3191255, -36.7341814,
     -35.55091139, -37.29814417, -28.68671182, -30.90902583,
     -42.37100061, -32.17082438, -28.86315326, -22.71193348,
     -38.36945867, -20.61407566, -22.72696734, -25.50360703,
     2.16339005, -27.96657305, -33.46004736,  -5.08943224,
     -30.21994603,   3.68803348, -36.10997302, -30.86899058,
     -22.69827634, -37.95847789, -22.40090313, -30.54859849,
     -26.64827358, -19.28162344, -34.69783578, -34.6614351,
     48.38377664,  47.60840355,  45.76793876,  44.61110193,
     49.28911284
]


# In[85]:


def q3():
    pca_2comp  = PCA(n_components=2) # reduzindo para dois componentes principais
    pca_2comp .fit(fifa)  # Passo os dados para serem reduzidos 
    # Tendo a variância de cada coluna para os dois principais componentes eu aplico uma multiplicação de matrizes 
    # para achar os principais componentes de x
    pc = pca.components_.dot(x)
    
    return (round(pc[0],3),round(pc[1],3))


# In[86]:


q3()


# ## Questão 4
# 
# Realiza RFE com estimador de regressão linear para selecionar cinco variáveis, eliminando uma a uma. Quais são as variáveis selecionadas? Responda como uma lista de nomes de variáveis.

# In[87]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE


# A feature a ser considerada como target é a 'Overall' que incorpora todos os atributos com ênfase em cada posição

# In[92]:


def q4():
    X_train = fifa.drop(columns='Overall')
    y = fifa['Overall']
    reg = LinearRegression()
    rfe = RFE(reg, n_features_to_select=5)
    selector = rfe.fit(X_train, y)
    return list(X_train.columns[selector.support_])


# In[93]:


q4()

