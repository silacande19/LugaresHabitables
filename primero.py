import pandas as pd
import numpy as np
df = pd.read_csv("/home/sila/Escritorio/machinelearning1/primero/persons_dataset.csv", index_col=0)
# Imprimir directamente los resultados
print(df.describe())
print(df.info())

set_ape = set(df["last_name"].to_list())
set_ema = set(df["email"].to_list())
set_cum = set(df["birthday"].to_list())
set_tel = set(df["phone"].to_list())
set_ciu = set(df["city"].to_list())
set_pai = set(df["country"].to_list())

print(f"Apellidos: {len(set_ape)}")
print(f"Emails: {len(set_ema)}")
print(f"Fechas de nacimiento: {len(set_cum)}")
print(f"Teléfonos: {len(set_tel)}")
print(f"Ciudades: {len(set_ciu)}")
print(f"Países: {len(set_pai)}")

#ordenarlos en edades, los nacidos entre :1980-1990 y los nacidos entre :1930-1970
# Convertir la columna "birthday" a objetos de fecha y hora
df["birthday"] = pd.to_datetime(df["birthday"])
print(df["birthday"])

# Crear una nueva columna "year" que contenga solo el año de nacimiento
df["year"] = df["birthday"].dt.year
print(df["year"])

# Filtrar los datos para incluir solo a las personas nacidas entre 1980 y 1990
df_1980_1990 = df[(df["year"] >= 1980) & (df["year"] <= 1990)]
print(df_1980_1990)

# Filtrar los datos para incluir solo a las personas nacidas entre 1930 y 1970
df_1930_1970 = df[(df["year"] >= 1930) & (df["year"] <= 1970)]

print(df_1930_1970)

from datetime import datetime

# Convertir la columna "birthday" a objetos de fecha y hora
df["birthday"] = pd.to_datetime(df["birthday"])
print(df["birthday"])

# Calcular la edad de cada persona
df["age"] = datetime.now().year - df["birthday"].dt.year
print(df["age"])

# Calcular la edad mínima, la edad media y la edad máxima
min_age = df["age"].min()
mean_age = df["age"].mean()
max_age = df["age"].max()

print(f"Edad mínima: {min_age}")
print(f"Edad media: {mean_age}")
print(f"Edad máxima: {max_age}")

import matplotlib.pyplot as plt
data = np.random.randn(1000)+ 100 - 50
plt.plot(data)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def min_max_scaler(data):
    data_min = np.min(data)
    data_max = np.max(data)
    return (data - data_min) / (data_max - data_min)


data = np.array([1, 2, 3, 4, 5])

norm_data = min_max_scaler(data)
plt.plot(norm_data)
plt.show()
      
import numpy as np

def standard_scaler(data):
    data_mean = np.mean(data)
    data_std = np.std(data)
    return (data - data_mean) / data_std
std_data = standard_scaler(data)
plt.plot(std_data)
plt.show()

from sklearn.preprocessing import OneHotEncoder
import pandas as pd

model = OneHotEncoder() #Crear un objeto OneHot Encoder
transformed_data = model.fit_transform(df[["city"]]).toarray()
transformed_columns = model.get_feature_names_out(input_features=["city"])
df_transformed = pd.DataFrame(transformed_data, columns=transformed_columns)
print(df_transformed)

#Reestablaeciendo los indices.
df.reset_index(drop=True, inplace=True)
df_transformed.reset_index(drop=True, inplace=True)
print(df)

#concatenando los Dataframes

df_model = pd.concat([df, df_transformed], axis=1)
print(df_model)



# Calcula la edad media
edad_media = df_model['age'].mean()
print(edad_media)

# Columna mayor_que_media
df_model['mayor_que_media'] = (df_model['age'] > edad_media).astype(int)
print(df_model)

#columna menor que media
df_model['menor_que_media'] = (df_model['age'] < edad_media).astype(int)
print(df_model)
df_model.shape

#importamos de sklearn
from sklearn.preprocessing import LabelEncoder
model = LabelEncoder()
df_LE = df_model.copy()
df_LE["country"] = model.fit_transform(df_LE["country"])
df_LE["last_name"] = model.fit_transform(df_LE["last_name"])
df_LE["email"] = model.fit_transform(df_LE["email"])
df_LE["birthday"] = model.fit_transform(df_LE["birthday"])
df_LE["city"] = model.fit_transform(df_LE["city"])
print(df_LE)

#Etiquetando las columnas de manera personalizada
from sklearn.preprocessing import LabelEncoder

def encode_columns(df, columns):
    model = LabelEncoder()
    for column in columns:
        df[column] = model.fit_transform(df[column])
    return df

df_LE = encode_columns(df_LE, ["email", "birthday", "city"])
print(df_LE)

df_label_custom = encode_columns(df, ["country", "last_name", "email", "birthday", "city"])
print(df_label_custom)

import pandas as pd

df = pd.read_csv("/home/sila/Escritorio/machinelearning1/primero/persons_dataset.csv")


def predict_randomly(x_size, num_classes = 3, seed = 42):
    import numpy as np
    np.random.seed(seed=seed)
    return np.random.randint(num_classes, size=x_size)

def faith_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

    return cm

import pandas as pd

df = pd.read_csv("/home/sila/Escritorio/machinelearning1/primero/persons_dataset.csv")

# Supongamos que 'x' corresponde a todas las columnas excepto la última,
# y 'y' corresponde a la última columna
x = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(x)
print(y)

y_pred = predict_randomly(x_size=x.shape[0])
print(y, y_pred)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


x = pd.get_dummies(x)

# Primero, dividimos los datos en un conjunto de entrenamiento y un conjunto de prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Luego, normalizamos los datos para que tengan una media de 0 y una desviación estándar de 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Ahora, creamos y entrenamos el modelo
model = LogisticRegression()
model.fit(x_train, y_train)

# Finalmente, usamos el modelo para hacer predicciones en el conjunto de prueba
y_pred = model.predict(x_test)

print(y_pred)

from sklearn.preprocessing import OrdinalEncoder

# Crear el codificador
encoder = OrdinalEncoder()

# Ajustar el codificador y transformar los datos
y_train_encoded = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.to_numpy().reshape(-1, 1))

# Ahora, creamos y entrenamos el modelo
model = LogisticRegression()
model.fit(x_train, y_train_encoded.ravel())

# Finalmente, usamos el modelo para hacer predicciones en el conjunto de prueba
y_pred = model.predict(x_test)

print(y_pred)

from sklearn.metrics import precision_score

# Primero, necesitamos convertir y_test_encoded a una matriz unidimensional
y_test_encoded = y_test_encoded.ravel()

# Luego, calculamos la precisión
precision = precision_score(y_test_encoded, y_pred, average='weighted')

print('Precision: ', precision)

from sklearn.metrics import recall_score

# Calculamos el recall
recall = recall_score(y_test_encoded, y_pred, average='weighted')

print('Recall: ', recall)

from sklearn.metrics import f1_score

# Calculamos el F1-score
f1 = f1_score(y_test_encoded, y_pred, average='weighted')

print('F1-score: ', f1)


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculamos la matriz de confusión
cm = confusion_matrix(y_test_encoded, y_pred)

# Visualizamos la matriz de confusión
sns.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Verdadero')
plt.xlabel('Predicho')
plt.savefig('/home/sila/Escritorio/machinelearning1/primero//grafico_confusion.png')

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Creamos y entrenamos un clasificador dummy
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(x_train, y_train)

# Hacemos predicciones con el clasificador dummy
y_dummy_pred = dummy.predict(x_test)

# Calculamos las métricas para el clasificador dummy
dummy_accuracy = accuracy_score(y_test, y_dummy_pred)
dummy_recall = recall_score(y_test, y_dummy_pred, average='weighted')
dummy_f1 = f1_score(y_test, y_dummy_pred, average='weighted')

print('Dummy Accuracy: ', dummy_accuracy)
print('Dummy Recall: ', dummy_recall)
print('Dummy F1-score: ', dummy_f1)

#validacion cruzada
from sklearn.model_selection import cross_val_score


scores = cross_val_score(model, x, y, cv=5)

print("Puntuaciones de la validación cruzada: ", scores)
print("Media de las puntuaciones de la validación cruzada: ", scores.mean())