# Desafio-1

El archivo "PruebaDesafio1Definitivo.ipynb" contiene el modelo realizado para la clasificación binaria utilizada. Se usó la técnica de Gradient Boosting Classifier. Las métricas obtenidas para AUC (train) fue de 1.0, AUC (test) 0.883, F1 score (train) 0.995 y F1 score (test) 0.702. Debido a lo desequilibrado del DataSet, estos valores presentan cierta variación a considerar, dependiendo de la semilla que se tome al hacer el split de los datos. En el split de datos se utilizó la estratificación para "y" y se utilizó average = "weighted" para obtener F1 score ponderado debido a la destribución desigual de la clase objetivo.


EL archivo "ParaDesafio2.ipynb" es una manera muy introductoria intentando explorar modelos ya entrenados para el análisis de sentimientos. Es necesario antes trabajar el dataset. Y ahondar más en el paquete pysentimiento https://github.com/pysentimiento/pysentimiento
