#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[3]:



from IPython.core.pylabtools import figsize


figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[4]:


#A geração aleatória de números não é verdadeiramente "aleatória".
#É determinístico, e a sequência que gera é ditada pelo valor inicial que você passa random.seed.

np.random.seed(42)
    
#rvs: Random variates.
#norm.rvs(loc, scale, size)
#binom.rvs(n, p, size) ==> n: n° de experimentos; p: probabilidade de sucesso

dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# In[5]:


dataframe.head()


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[5]:


# Sua análise da parte 1 começa aqui.


# In[6]:


sns.distplot(dataframe['normal'])


# In[7]:


sns.distplot(dataframe['binomial'],bins=range(4, 18), kde=False)


# In[12]:


normal_media = dataframe['normal'].mean()
normal_var = dataframe['normal'].var()
normal_desv = np.sqrt(normal_var)
print('Normal:')
print('\nmédia: ',normal_media,'\nvariância: ', normal_var,'\ndesvio padrão: ', normal_desv)


# In[13]:


bin_media = dataframe['binomial'].mean()
bin_var = dataframe['binomial'].var()
bin_desv = np.sqrt(bin_var)
print('Binomial:')
print('\nmédia: ',bin_media,'\nvariância: ', bin_var,'\ndesvio padrão: ', bin_desv)


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[55]:


#Quantis da variável normal

q1_norm = dataframe.normal.quantile(0.25)
q2_norm = dataframe.normal.quantile(0.5) 
q3_norm = dataframe.normal.quantile(0.75)

dataframe.normal.quantile([0.25,0.5,0.75])


# In[56]:


#Quantis da variável binomial

q1_binom = dataframe.binomial.quantile(0.25)
q2_binom = dataframe.binomial.quantile(0.5) 
q3_binom = dataframe.binomial.quantile(0.75)

dataframe.binomial.quantile([0.25,0.5,0.75])


# In[57]:


def q1():
    resposta = ((q1_norm - q1_binom).round(3), (q2_norm - q2_binom).round(3), (q3_norm - q3_binom).round(3))
    return resposta


# In[58]:


q1()


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[41]:


def q2():
    normal_media = dataframe['normal'].mean()
    normal_var = dataframe['normal'].var()
    normal_desv = np.sqrt(normal_var)
    
    ecdf = ECDF(dataframe.normal) #Retorna o CDF empírico de uma matriz 
    
    mais = ecdf((normal_media + normal_desv))
    menos = ecdf((normal_media - normal_desv))
    resultado = float((mais - menos).round(3))
    
    return resultado


# In[42]:


q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[13]:


def q3():
    m_norm = dataframe['normal'].mean()
    v_norm = dataframe['normal'].var()
    m_binom = dataframe['binomial'].mean()
    v_binom = dataframe['binomial'].var()
    
    return ((m_binom - m_norm).round(3), (v_binom - v_norm).round(3))


# In[14]:


q3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[15]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[16]:


# Sua análise da parte 2 começa aqui.


# In[17]:


stars.head()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[67]:


def q4():
    mean_profile_filter = stars[stars['target'] == False]['mean_profile']
    media = mean_profile_filter.mean()
    desvio_padrao = mean_profile_filter.std()
    z = (mean_profile_filter - media) / desvio_padrao
    false_pulsar_mean_profile_standardized = z
    
    quantil_80 = sct.norm.ppf(0.80, loc = 0, scale = 1)
    quantil_90 = sct.norm.ppf(0.90, loc = 0, scale = 1)
    quantil_95 = sct.norm.ppf(0.95, loc = 0, scale = 1)
    
    ecdf = ECDF(false_pulsar_mean_profile_standardized) #Retorna o CDF empírico de uma matriz como uma função de etapa
    
    prob_quantil_80 = ecdf(quantil_80).round(3)
    prob_quantil_90 = ecdf(quantil_90).round(3)
    prob_quantil_95 = ecdf(quantil_95).round(3)
    
    return (prob_quantil_80, prob_quantil_90, prob_quantil_95)


# In[68]:


q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[20]:


def q5():
    mean_profile_filter = stars[stars['target'] == 0]['mean_profile']
    media = mean_profile_filter.mean()
    desvio_padrao = mean_profile_filter.std()
    z = (mean_profile_filter - media) / desvio_padrao
    false_pulsar_mean_profile_standardized = z
    
    Q1_false_pulsar_mean_profile_standardized = np.percentile(false_pulsar_mean_profile_standardized, 25)
    Q2_false_pulsar_mean_profile_standardized = np.percentile(false_pulsar_mean_profile_standardized, 50)
    Q3_false_pulsar_mean_profile_standardized = np.percentile(false_pulsar_mean_profile_standardized, 75)

    Q1_normal = sct.norm.ppf(0.25, loc=0, scale=1)
    Q2_normal = sct.norm.ppf(0.50, loc=0, scale=1)
    Q3_normal = sct.norm.ppf(0.75, loc=0, scale=1)
    
    resposta = ((Q1_false_pulsar_mean_profile_standardized-Q1_normal).round(3), 
                (Q2_false_pulsar_mean_profile_standardized-Q2_normal).round(3),
                (Q3_false_pulsar_mean_profile_standardized-Q3_normal).round(3))    
   
    return resposta


# In[21]:


q5()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
