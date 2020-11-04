
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


base = pd.read_csv('house_prices.csv')


# In[3]:


base.head()


# In[4]:


# Variáveis preditoras (previsores)
X = base.iloc[:, 3:19].values


# In[5]:


# Variável target (classe)
y = base.iloc[:, 2].values


# In[6]:


# Dividindo dataset entre treino e teste com 30% para teste
from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[8]:


# Instanciando objeto polinomial para transformar nossas variáveis preditoras
from sklearn.preprocessing import PolynomialFeatures


# In[9]:


poly = PolynomialFeatures(degree=4)


# In[11]:


# Transformando os dados de treino no formato polinomial de grau máximo igual a 4
X_train_poly = poly.fit_transform(X_train)


# In[12]:


# Transformando os dados de teste no formato polinomial de grau máximo igual a 4
X_test_poly = poly.transform(X_test)


# In[13]:


# Modelo de regressão linear com a variável polinomial
from sklearn.linear_model import LinearRegression


# In[15]:


regressor = LinearRegression()


# In[16]:


regressor.fit(X_train_poly, y_train)


# In[23]:


# Calculando precisão do modelo treinado
score = regressor.score(X_train_poly, y_train)
score


# In[24]:


previsoes = regressor.predict(X_test_poly)


# In[25]:


# Calculando erro absoluto médio para o modelo
from sklearn.metrics import mean_absolute_error


# In[26]:


mae = mean_absolute_error(y_test, previsoes)
mae

