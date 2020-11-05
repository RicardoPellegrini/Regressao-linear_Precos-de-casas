
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
y = base.iloc[:, 2:3].values


# In[6]:


# Fazendo escalonamento, pois o SVR necessita desse preprocessamento
from sklearn.preprocessing import StandardScaler


# In[7]:


scaler_x = StandardScaler()
scaler_y = StandardScaler()


# In[8]:


X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)


# In[9]:


# Dividindo dataset entre treino e teste com 30% para teste
from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[11]:


# Modelo de svm.SVR
from sklearn.svm import SVR


# In[12]:


regressor = SVR(kernel='rbf', C=2.0)


# In[13]:


regressor.fit(X_train, y_train)


# In[14]:


# Calculando precisão do modelo treinado para o treinamento
score = regressor.score(X_train, y_train)
score


# In[15]:


previsoes = regressor.predict(X_test)


# In[16]:


# Calculando precisão do modelo treinado para os testes
score = regressor.score(X_test, y_test)
score


# In[17]:


# Voltando os valores de teste para a escala original
y_test = scaler_y.inverse_transform(y_test)
y_test


# In[18]:


# Voltando os valores de previsão para a escala original
previsoes = scaler_y.inverse_transform(previsoes)
previsoes


# In[19]:


# Calculando erro absoluto médio para o modelo
from sklearn.metrics import mean_absolute_error


# In[20]:


mae = mean_absolute_error(y_test, previsoes)
mae

