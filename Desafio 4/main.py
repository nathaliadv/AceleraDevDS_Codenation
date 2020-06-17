#!/usr/bin/env python
# coding: utf-8

# # Desafio 4
# 
# Neste desafio, vamos praticar um pouco sobre testes de hipóteses. Utilizaremos o _data set_ [2016 Olympics in Rio de Janeiro](https://www.kaggle.com/rio2016/olympic-games/), que contém dados sobre os atletas das Olimpíadas de 2016 no Rio de Janeiro.
# 
# Esse _data set_ conta com informações gerais sobre 11538 atletas como nome, nacionalidade, altura, peso e esporte praticado. Estaremos especialmente interessados nas variáveis numéricas altura (`height`) e peso (`weight`). As análises feitas aqui são parte de uma Análise Exploratória de Dados (EDA).
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
import statsmodels.api as sm


# In[2]:


#%matplotlib inline

from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# In[3]:


athletes = pd.read_csv("athletes.csv")


# In[4]:


athletes.info()


# In[5]:


athletes.head()


# In[6]:


athletes[['height','weight']].describe()


# In[7]:


athletes[['height','weight']].hist()


# In[8]:


def get_sample(df, col_name, n=100, seed=42):
    """Get a sample from a column of a dataframe.
    
    It drops any numpy.nan entries before sampling. The sampling
    is performed without replacement.
    
    Example of numpydoc for those who haven't seen yet.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Source dataframe.
    col_name : str
        Name of the column to be sampled.
    n : int
        Sample size. Default is 100.
    seed : int
        Random seed. Default is 42.
    
    Returns
    -------
    pandas.Series
        Sample of size n from dataframe's column.
    """
    np.random.seed(seed)
    
    random_idx = np.random.choice(df[col_name].dropna().index, size=n, replace=False) #retorna uma array com index das colunas 
    
    return df.loc[random_idx, col_name] #retorna uma series com index e valor da coluna


# ## Inicia sua análise a partir daqui

# In[9]:


# Sua análise começa aqui.


# ## Questão 1
# 
# Considerando uma amostra de tamanho 3000 da coluna `height` obtida com a função `get_sample()`, execute o teste de normalidade de Shapiro-Wilk com a função `scipy.stats.shapiro()`. Podemos afirmar que as alturas são normalmente distribuídas com base nesse teste (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[10]:


def q1():
    amostra_q1 = get_sample(athletes,'height', n=3000, seed=42)
    stat, p = sct.shapiro(amostra_q1)
    print('stat= {}, p={}'.format(stat,p))
    return bool(p> 0.05)


# In[11]:


q1()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Plote o qq-plot para essa variável e a analise.
# * Existe algum nível de significância razoável que nos dê outro resultado no teste? (Não faça isso na prática. Isso é chamado _p-value hacking_, e não é legal).

# In[12]:


amostra_q1 = get_sample(athletes,'height', n=3000, seed=42)


# In[13]:


sns.distplot(amostra_q1, bins=25, hist_kws={"density": True})
plt.show ()


# In[14]:


sm.qqplot(amostra_q1, fit=True, line="45")
plt.show ()


# In[15]:


amostra_q1 = get_sample(athletes,'height', n=3000, seed=42)
stat, p = sct.shapiro(amostra_q1)
p > 0.0000001


# ## Questão 2
# 
# Repita o mesmo procedimento acima, mas agora utilizando o teste de normalidade de Jarque-Bera através da função `scipy.stats.jarque_bera()`. Agora podemos afirmar que as alturas são normalmente distribuídas (ao nível de significância de 5%)? Responda com um boolean (`True` ou `False`).

# In[16]:


def q2():
    amostra_q2 = get_sample(athletes,'height', n=3000, seed=42)
    stat, p = sct.jarque_bera(amostra_q2)
    print('stat= {}, p={}'.format(stat,p))
    return bool(p> 0.05)


# In[17]:


q2()


# __Para refletir__:
# 
# * Esse resultado faz sentido?

# In[18]:


amostra_q2 = get_sample(athletes,'height', n=3000, seed=42)
sm.qqplot(amostra_q2, fit=True, line="45")
plt.show ()


# ## Questão 3
# 
# Considerando agora uma amostra de tamanho 3000 da coluna `weight` obtida com a função `get_sample()`. Faça o teste de normalidade de D'Agostino-Pearson utilizando a função `scipy.stats.normaltest()`. Podemos afirmar que os pesos vêm de uma distribuição normal ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[19]:


def q3():
    amostra_q3 = get_sample(athletes,'weight', n=3000, seed=42)
    stat, p = sct.normaltest(amostra_q3)
    print('stat= {}, p={}'.format(stat,p))
    return bool(p> 0.05)


# In[20]:


q3()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Um _box plot_ também poderia ajudar a entender a resposta.

# In[21]:


amostra_q3 = get_sample(athletes,'weight', n=3000, seed=42)
sns.distplot(amostra_q3, bins=25, hist_kws={"density": True})
plt.show ()


# In[22]:


sns.boxplot(data = amostra_q3)


# ## Questão 4
# 
# Realize uma transformação logarítmica em na amostra de `weight` da questão 3 e repita o mesmo procedimento. Podemos afirmar a normalidade da variável transformada ao nível de significância de 5%? Responda com um boolean (`True` ou `False`).

# In[23]:


def q4():
    amostra_q4 = get_sample(athletes,'weight', n=3000, seed=42)
    amostra_q4_transformada = np.log(amostra_q4)
    stat, p = sct.normaltest(amostra_q4_transformada)
    print('stat= {}, p={}'.format(stat,p))
    return bool(p> 0.05)    


# In[24]:


q4()


# __Para refletir__:
# 
# * Plote o histograma dessa variável (com, por exemplo, `bins=25`). A forma do gráfico e o resultado do teste são condizentes? Por que?
# * Você esperava um resultado diferente agora?

# In[25]:


amostra_q4 = get_sample(athletes,'weight', n=3000, seed=42)
amostra_q4_transformada = np.log(amostra_q4)
sns.distplot(amostra_q4_transformada, bins=25, hist_kws={"density": True})
plt.show ()


# In[26]:


sns.boxplot(data = amostra_q4_transformada)


# > __Para as questão 5 6 e 7 a seguir considere todos testes efetuados ao nível de significância de 5%__.

# ## Questão 5
# 
# Obtenha todos atletas brasileiros, norte-americanos e canadenses em `DataFrame`s chamados `bra`, `usa` e `can`,respectivamente. Realize um teste de hipóteses para comparação das médias das alturas (`height`) para amostras independentes e variâncias diferentes com a função `scipy.stats.ttest_ind()` entre `bra` e `usa`. Podemos afirmar que as médias são estatisticamente iguais? Responda com um boolean (`True` ou `False`).

# In[27]:


athletes.columns


# In[45]:


athletes[(athletes.nationality == 'BRA') | (athletes.nationality == 'USA') | (athletes.nationality == 'CAN')]


# In[28]:


bra = athletes[athletes.nationality == 'BRA']
usa = athletes[athletes.nationality == 'USA']
can = athletes[athletes.nationality == 'CAN']


# In[29]:


bra['height'].describe()


# In[30]:


bra.isna().sum()


# In[31]:


usa['height'].describe()


# In[32]:


usa.isna().sum()


# In[46]:


can['height'].describe()


# In[47]:


can.isna().sum()


# In[33]:


def q5():
    stat, p = sct.ttest_ind(bra['height'], usa['height'],  equal_var = False, nan_policy = 'omit') #False: se falso, execute o teste t de Welch, que não assume igual variação populaciona
    print('stat= {}, p={}'.format(stat,p))
    return bool(p> 0.05)


# In[34]:


q5()


# In[35]:


sns.distplot(bra['height'], bins=25, hist=False, rug=True, label='BRA')
sns.distplot(usa['height'], bins=25, hist=False, rug=True, label='USA')


# ## Questão 6
# 
# Repita o procedimento da questão 5, mas agora entre as alturas de `bra` e `can`. Podemos afimar agora que as médias são estatisticamente iguais? Reponda com um boolean (`True` ou `False`).

# In[48]:


def q6():
    stat, p = sct.ttest_ind(bra['height'], can['height'],  equal_var = False, nan_policy = 'omit') #False: se falso, execute o teste t de Welch, que não assume igual variação populaciona
    print('stat= {}, p={}'.format(stat,p))
    return bool(p> 0.05)


# In[49]:


q6()


# In[50]:


sns.distplot(bra['height'], bins=25, hist=False, rug=True, label='BRA')
sns.distplot(can['height'], bins=25, hist=False, rug=True, label='CAN')


# ## Questão 7
# 
# Repita o procedimento da questão 6, mas agora entre as alturas de `usa` e `can`. Qual o valor do p-valor retornado? Responda como um único escalar arredondado para oito casas decimais.

# In[87]:


def q7():
    stat, p = sct.ttest_ind(usa['height'], can['height'],  equal_var = False, nan_policy = 'omit') #False: se falso, execute o teste t de Welch, que não assume igual variação populaciona
    print('stat= {}, p={}'.format(stat,p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')
    return float(np.round(p, 8)) 


# In[88]:


q7()


# __Para refletir__:
# 
# * O resultado faz sentido?
# * Você consegue interpretar esse p-valor?
# * Você consegue chegar a esse valor de p-valor a partir da variável de estatística?

# In[72]:


stat, p = sct.ttest_ind(usa['height'], can['height'],  equal_var = True, nan_policy = 'omit')
print('stat= {}, p={}'.format(stat,p))


# In[69]:


#grau de liberdade para o teste t independente com variancias semelhantes: df = n1 + n2 - 2
gl = len(usa) + len(can) - 2
print(f"Graus de liberdade: {gl}")
q7_sf = sct.t.sf(stat, gl)*2 #Para Hipótese Bicaudal
print(q7_sf)


# In[77]:


sns.distplot(usa['height'], bins=25, hist=False, rug=True, label='USA')
sns.distplot(can['height'], bins=25, hist=False, rug=True, label='CAN')

