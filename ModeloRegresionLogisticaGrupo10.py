#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
from imblearn.over_sampling import SMOTE
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


# In[35]:


dataframe.describe()


# In[36]:


print(dataframe.groupby('fatal').size())


# In[4]:


dataframe.drop(['fatal'],axis=1).hist()
plt.show()


# In[ ]:


sb.pairplot(dataframe.dropna(), hue='fatal',height=12,vars=["edad","momento_hecho_Diurno","momento_hecho_Nocturno","sexo_Femenino","sexo_Masculino","clase_victima_Conductor","clase_victima_Pasajero","clase_victima_Peaton","vehiculo_victima_Automovil","tipo_incidente_Colision vehiculo/Persona","tipo_incidente_Colision vehiculo/Vehiculo","clima_Bueno"],kind='reg')


# In[37]:


X = np.array(dataframe.drop(['fatal'],axis=1))
y = np.array(dataframe['fatal'])
# Aplicar SMOTE para balancear los datos
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_resampled.shape


# In[38]:


# Crear y entrenar el modelo de regresión logística
model = linear_model.LogisticRegression(max_iter=9000, C=1.0, penalty='l2', solver='lbfgs')
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')

report = classification_report(y_test, y_pred)
print('Reporte de clasificación:\n', report)


# In[ ]:




