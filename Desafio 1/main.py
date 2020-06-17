#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[4]:


import pandas as pd
import numpy as np


# In[5]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[6]:


df = black_friday


# In[7]:


df.columns


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.dtypes.value_counts()


# In[9]:


df[['Product_Category_1','Product_Category_2','Product_Category_3']].head()


# In[10]:


valores_faltantes = df.isnull().sum()
valores_faltantes


# In[11]:


#dataframe auxiliar com tipos e quantidade de valores faltantes
aux = pd.DataFrame({'colunas': df.columns, 'tipo': df.dtypes, 'dados_faltantes': df.isnull().sum(), '%_faltantes':((df.isnull().sum()/df.shape[0])*100)})
aux


# In[ ]:


#Acredita-se que categoria de produto 1,2,3 representa todas as categorias às quais o produto pertence. Podendo um produto
#pertencer a uma categoria ou a todas. Então, conclui-se que os NaN significam que o produto não pertence a essa categoria.
#df = df.fillna(0)
#valores_faltantes = df.isnull().sum()
#valores_faltantes


# In[12]:


# Número de clientes 
clientes = df['User_ID'].nunique()
clientes


# In[26]:


# Produtos únicos 
produtos = df['Product_ID'].unique()
lista_produtos = produtos
quantidade_produtos_unicos = len(produtos)


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[12]:


def q1():
    quant = df.shape
    return quant


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[2]:


def q2():
    q2 = df[(df['Age'] == '26-35') & (df['Gender'] == 'F')] #dataframe com mulheres com idade entre 26-35
    #q2 = q2.drop_duplicates(subset ="User_ID", keep = 'first') #remover os usuários que estão duplicados
    mulheres = len(q2) #número de linhas do dataframe
    return mulheres


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[29]:


def q3():
    clientes = df['User_ID'].nunique()
    return clientes


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[30]:


def q4():
    tipos_dados = df.dtypes.nunique()
    return tipos_dados


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[31]:


def q5():
    porc_nulos = (len(df) - len(df.dropna()))/len(df)
    return porc_nulos


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[23]:


def q6():
    maior_nulos = df.isnull().sum().max()
    return int(maior_nulos)


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[33]:


def q7():
    df_n = df.dropna()
    moda = df_n['Product_Category_3'].mode()[0]
    return moda


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[34]:


def q8():
    purchase_normalizado = (df['Purchase'] - df['Purchase'].min()) / (df['Purchase'].max() - df['Purchase'].min())
    return float(np.mean(purchase_normalizado))


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[35]:


def q9():
    purchase_padronizado = (df['Purchase'] - np.mean(df['Purchase'])) / np.std(df['Purchase'])
    return len([i for i in purchase_padronizado if i > -1 and i < 1])


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[36]:


df[['Product_Category_2', 'Product_Category_3']].isnull().sum()


# In[37]:


def q10():
    df_aux = black_friday[['Product_Category_2','Product_Category_3']] #DataFrame com as colunas Product_Category_2 Product_Category_3
    df_aux = df_aux[df_aux['Product_Category_2'].isna()] #Filtro apenas o Product_Category_2 nulos e reatribuo a DataFrame anterior 
    equal = df_aux['Product_Category_2'].equals(df_aux['Product_Category_3']) #Comparo se o Product_Category_2 é igual a  Product_Category_3
    return equal

