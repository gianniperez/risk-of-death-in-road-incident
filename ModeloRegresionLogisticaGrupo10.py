#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv('datasetfinal.csv')

# Lista de variables categóricas a codificar
categorical_columns = ['momento_hecho', 'sexo', 'clase_victima', 'vehiculo_victima', 'tipo_via', 'tipo_incidente', 'clima']

# Realiza la codificación one-hot para todas las variables categóricas
encoded_data = pd.get_dummies(data, columns=categorical_columns, prefix=categorical_columns, drop_first=False)

# Reemplaza False por 0 y True por 1 en todo el DataFrame
encoded_data = encoded_data.astype(int)
dataframe = encoded_data
dataframe.head()


# In[55]:


dataframe.describe()


# In[56]:


print(dataframe.groupby('fatal').size())


# In[57]:


dataframe.drop(['fatal'],axis=1).hist()
plt.show()


# In[60]:


sb.pairplot(dataframe.dropna(), hue='fatal',height=12,vars=["edad","momento_hecho_Diurno","momento_hecho_Nocturno","sexo_Femenino","sexo_Masculino","clase_victima_Conductor","clase_victima_Pasajero","clase_victima_Peaton","vehiculo_victima_Automovil","tipo_incidente_Colision vehiculo/Persona","tipo_incidente_Colision vehiculo/Vehiculo","clima_Bueno"],kind='reg')


# In[61]:


X = np.array(dataframe.drop(['fatal'],axis=1))
y = np.array(dataframe['fatal'])
X.shape


# In[63]:


model = linear_model.LogisticRegression(max_iter=2000)
model.fit(X,y)


# In[64]:


predictions = model.predict(X)
print(predictions[:30])


# In[66]:


model.score(X,y)

