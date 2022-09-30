from matplotlib.pyplot import cla
import pandas as pd
import numpy
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import GridSearchCV

#Lectura del dataset
dfreview = pd.read_csv('IMDB Dataset.csv')
#print(dfreview)

#slices o particiones para crear un set desbalanceado
dfpositivos = dfreview[dfreview['sentiment']=='positive'][:9000]
dfnegativos = dfreview[dfreview['sentiment']=='negative'][:1000]
dfreviewdes = pd.concat([dfpositivos,dfnegativos])
#print(dfreviewdes)
#print(dfreview.value_counts('sentiment'))
#print(dfreviewdes.value_counts('sentiment'))

#Se hace un undersampling para balancear el dataset partido
rus = RandomUnderSampler()
dfreviewbal, dfreviewbal['sentiment'] = rus.fit_resample(dfreviewdes[['review']],dfreviewdes['sentiment'])
#print(dfreviewbal.value_counts('sentiment'))

#se divide en un conjunto de entrenamiento y otro de prueba
train, test=train_test_split(dfreviewbal, test_size=0.33, random_state=42)
#print(train)
#print(test)

#Se llenan los conjuntos de entrenamiento y prueba en valor(review) y output(sentiment)
trainX, trainY=train['review'], train['sentiment']
testX, testY=test['review'], test['sentiment']

#ejemplo de manejo de texto a representacion numerica
text = ["I love writing code in Python. I love Python code",
        "I hate writing code in Java. I hate Java code"]
#Ejemplo para crear una matriz de conteo de frecuencias
df = pd.DataFrame({'review': ['review1', 'review2'], 'text':text})
cv = CountVectorizer(stop_words='english')
cv_matrix = cv.fit_transform(df['text'])
df_dtm = pd.DataFrame(cv_matrix.toarray(), index=df['review'].values, columns=cv.get_feature_names_out())
#print(df_dtm)

#ejemplo para crear una matriz con valores tfidf
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])
df_dtm = pd.DataFrame(tfidf_matrix.toarray(), index=df['review'].values, columns=tfidf.get_feature_names_out())
#print(df_dtm)

#creamos y llenamos los conjuntos de valores que vamos a usar para los modelos y creamos nuestra bolsa de palabras en base al conjunto de entrenamiento
tfidf = TfidfVectorizer(stop_words='english')
trainXVector = tfidf.fit_transform(trainX)
testXVector = tfidf.transform(testX)

#SVM
svc = SVC(kernel='linear')
svc.fit(trainXVector,trainY)
#prueba de svm
#print(svc.predict(tfidf.transform(['A good movie'])))
#print(svc.predict(tfidf.transform(['An excellent movie'])))
#print(svc.predict(tfidf.transform(['"I did not like this movie at all I gave this movie away"'])))

#Decision tree
decTree = DecisionTreeClassifier()
decTree.fit(trainXVector,trainY)
#test de arbol
#print(decTree.predict(tfidf.transform(['A good movie'])))
#print(decTree.predict(tfidf.transform(['An excellent movie'])))
#print(decTree.predict(tfidf.transform(['"I did not like this movie at all I gave this movie away"'])))

#Naive bayes
nBayes = GaussianNB()
nBayes.fit(trainXVector.toarray(),trainY)
#test de bayes
#print(nBayes.predict(tfidf.transform(['A good movie']).toarray()))
#print(nBayes.predict(tfidf.transform(['An excellent movie']).toarray()))
#print(nBayes.predict(tfidf.transform(['"I did not like this movie at all I gave this movie away"']).toarray()))

#Logistic regression
lr = LogisticRegression()
lr.fit(trainXVector,trainY)
#test de regresion
#print(lr.predict(tfidf.transform(['A good movie'])))
#print(lr.predict(tfidf.transform(['An excellent movie'])))
#print(lr.predict(tfidf.transform(['"I did not like this movie at all I gave this movie away"'])))

#Evaluacion
#score - accuracy
#print(svc.score(testXVector,testY))
#print(decTree.score(testXVector,testY))
#print(nBayes.score(testXVector.toarray(),testY))
#print(lr.score(testXVector,testY))

#F1 score
#F1 Score = 2(Recall Precision) / (Recall + Precision)
#Precision - de lo que se predijo correctamente cuanto es correcto
#Recall - de lo que se predijo en genera que era de una clase, cuanto realmente pertenece a esa clase
#print(f1_score(testY,svc.predict(testXVector),labels=['positive','negative'],average=None))
#print(f1_score(testY,decTree.predict(testXVector),labels=['positive','negative'],average=None))
#print(f1_score(testY,nBayes.predict(testXVector.toarray()),labels=['positive','negative'],average=None))
#print(f1_score(testY,lr.predict(testXVector),labels=['positive','negative'],average=None))

#Clasification report
#print(classification_report(testY,svc.predict(testXVector),labels=['positive','negative']))
#print(classification_report(testY,decTree.predict(testXVector),labels=['positive','negative']))
#print(classification_report(testY,nBayes.predict(testXVector.toarray()),labels=['positive','negative']))
#print(classification_report(testY,lr.predict(testXVector),labels=['positive','negative']))

#Matriz de confusion
#filas - predecidos
#columnas - reales
#print(confusion_matrix(testY,svc.predict(testXVector),labels=['positive','negative']))
#print(confusion_matrix(testY,decTree.predict(testXVector),labels=['positive','negative']))
#print(confusion_matrix(testY,nBayes.predict(testXVector.toarray()),labels=['positive','negative']))
#print(confusion_matrix(testY,lr.predict(testXVector),labels=['positive','negative']))

#Optimizacion de modelo
#GridSearchCV
#Parametros
#C - parametro de penalizacion o error soportable
#kernel - parte del sistema que especifica las funciones a utilizar
#params = {'C': [1,4,8,16,32] ,'kernel':['linear', 'rbf']}
#svc = SVC()
#llamada a la optimizacion con el modelo, los parametros y el numero de validaciones cruzadas
#svcGrid = GridSearchCV(svc, params, cv=5)
#svcGrid.fit(trainXVector,trainY)
#print(svcGrid.best_params_)
#print(svcGrid.best_estimator_)
#print(svcGrid.best_score_)

#Post tutorial
#Analisis de metricas:
print("SVC:")
print(svc.score(testXVector,testY))
print(classification_report(testY,svc.predict(testXVector),labels=['positive','negative']))
#print("decision tree:")
#print(decTree.score(testXVector,testY))
#print(classification_report(testY,decTree.predict(testXVector),labels=['positive','negative']))
#print("naive bayes:")
#print(nBayes.score(testXVector.toarray(),testY))
#print(classification_report(testY,nBayes.predict(testXVector.toarray()),labels=['positive','negative']))
#print("logistic regression:")
#print(lr.score(testXVector,testY))
#print(classification_report(testY,lr.predict(testXVector),labels=['positive','negative']))
#resultados de consola:
"""
SVC:
0.8393939393939394
              precision    recall  f1-score   support

    positive       0.83      0.86      0.84       335
    negative       0.85      0.82      0.83       325

    accuracy                           0.84       660
   macro avg       0.84      0.84      0.84       660
weighted avg       0.84      0.84      0.84       660

decision tree:
0.6757575757575758
              precision    recall  f1-score   support

    positive       0.68      0.69      0.68       335
    negative       0.67      0.66      0.67       325

    accuracy                           0.68       660
   macro avg       0.68      0.68      0.68       660
weighted avg       0.68      0.68      0.68       660

naive bayes:
0.5954545454545455
              precision    recall  f1-score   support

    positive       0.59      0.64      0.62       335
    negative       0.60      0.54      0.57       325

    accuracy                           0.60       660
   macro avg       0.60      0.59      0.59       660
weighted avg       0.60      0.60      0.59       660

logistic regression:
0.8333333333333334
              precision    recall  f1-score   support

    positive       0.82      0.86      0.84       335
    negative       0.85      0.80      0.83       325

    accuracy                           0.83       660
   macro avg       0.83      0.83      0.83       660
weighted avg       0.83      0.83      0.83       660
"""
#Por lo tanto consideramos el mejor modelo es el scv al tener mayor accuracy, preecion y recall
#Matriz de confusion:
confmtrSCV = confusion_matrix(testY,svc.predict(testXVector),labels=['positive','negative'])
print(confmtrSCV)
"""
          Positive    Negative
Positive     288         47
Negative      61        264
"""

#Vamos a calcular el recall de peliculas buenas (positivas), ya que al ser los gustos en peliculas
#algo subjetivo, queremos tener la mayor cantidad posible de peliculas buenas
print(recall_score(testY,svc.predict(testXVector),average=None,labels='positive'))