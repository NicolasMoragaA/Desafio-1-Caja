# Desafio-1

El archivo "PruebaDesafio1Definitivo.ipynb" contiene el modelo realizado para la clasificación binaria utilizada. Se usó la técnica de Gradient Boosting Classifier. Las métricas obtenidas para AUC (train) fue de 1.0, AUC (test) 0.883, F1 score (train) 0.995 y F1 score (test) 0.702. Debido a lo desequilibrado del DataSet, estos valores presentan cierta variación a considerar, dependiendo de la semilla que se tome al hacer el split de los datos. En el split de datos se utilizó la estratificación para "y" y se utilizó average = "weighted" para obtener F1 score ponderado debido a la destribución desigual de la clase objetivo.

En el archivo "Untitled2.ipynb" se intentó modelo muy simple solo considerando 2 variables que eran las que presentaban la correlación más fuerte con el target. Se trabajó tanto con Gradient Boosting Classifier como Logistic Regression.

En el archivo "Untitled3.ipynb" se intentó reducir la dimensionalidad del dataset usando PCA sobre las columnas de variable continua y luego transformarla a variable categórica debido a la distribución que presentaba cuando se realizo un histograma, pero no se llegó a buen puerto. 
