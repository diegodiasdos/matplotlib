
# coding: utf-8

# In[13]:


from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target


# In[14]:

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X, y)
logreg.predict(X)


# In[15]:


y_pred = logreg.predict(X)

len(y_pred)


# In[16]:

from sklearn import metrics
print(metrics.accuracy_score(y, y_pred))


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[18]:

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X, y)
y_pred = knn.predict(X)
print(metrics.accuracy_score(y, y_pred))


# In[19]:

#imprima as formas dos novos objetos x 
print(X.shape)
print(y.shape)


# In[21]:

#:dividir  x e y em conjuntos  de treinamento  e teste 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)


# In[22]:

#iimprima as formas dos novos objetos x 
print(X_train.shape)
print(X_test.shape)


# In[23]:

#imprima  as formas  dos novos  objetos y 

print(y_train.shape)
print(y_test.shape)


# In[25]:

#imprima as formas  dos novos objetos y 
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[26]:

#faca  previsões no conjunto  de testes 
y_pred = logreg.predict(X_test)
#Compare os valores  de  resposta reais (y_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[27]:

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[28]:

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[29]:

#tente  k=1 a k=25  e registre a precisão  do teste 
k_range = list(range(1,26))
scores =[]
for k in k_range:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  scores.append(metrics.accuracy_score(y_test,y_pred))


# In[31]:

#import matplotlib(biblioteca de traçabilidade)
import matplotlib.pyplot as plt

#permitir que parcelas  apareçam  dentro do notebook
get_ipython().magic(u'matplotlib inline')

#traçar  o relacionamento  entre  k e verificar  a precisao 
plt.plot(k_range, scores)
plt.xlabel('Value  of K for KNN')
plt.ylabel('Testing Accuracy')


# In[32]:

#instanciar   o modelo  com os  parâmetros mais conhecidos 

knn = KNeighborsClassifier(n_neighbors=11)

#treine o modelo com x e y (não x_train e y_train)
knn.fit(X, y)

#faça  uma previsão  para  uma  observação fora 
#a amostra 

knn.predict([[3,5,4,2]])


# In[ ]:



