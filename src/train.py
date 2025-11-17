
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_squared_error

# Cargar datos
df = pd.read_csv('data/data_cleaned.csv', sep=",")

# Acoto el dataset a un solo tipo de transacción basado en las conclusiones del notebook 02_Model_comparisson
t = 'Alquiler'
p =  'Oficina'
df = df[(df['operation_type'] == t) & (df['property_type'] == p)]
df = df.drop(columns=['operation_type', 'property_type'])

# Convierto los strings de neighborhood con LabelEncoder
le_neighborhood = LabelEncoder()
df['neighborhood'] = le_neighborhood.fit_transform(df['neighborhood'])

# Guardo las columnas para usar en producción
with open('models/le_neighborhood.pkl', 'wb') as f:
  pickle.dump(le_neighborhood, f)

# Transformo las variables numéricas con StandardScaler
numericas = ['rooms', 'bathrooms', 'surface_covered']
scalers = {}
for col in numericas:
    scaler = StandardScaler()
    df[col] = scaler.fit_transform(df[[col]])
    scalers[col] = scaler

# Guardo el scaler para usar en producción
with open("models/scalers.pkl", "wb") as f:
    pickle.dump(scalers, f)

# Elimina columna target y columna relacionada
X_reg = df.drop(columns=['price_usd', 'price_per_m2'])
y_reg = df['price_per_m2']


# Preparación de train y test
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Separo en train y test
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Definición del modelo base

# mejores hiperparámetros aplicados) 
rf_params = {
    'max_depth': 20, 
    'max_features': 'None', 
    'min_samples_leaf': 1, 
    'min_samples_split': 2, 
    'n_estimators': 1000
    }

model = RandomForestRegressor(**rf_params)


# Entrenamiento en el target log-transformado
model.fit(X_train, y_train)

# Predicciones (aún en escala logarítmica)
y_pred = model.predict(X_test)

# Evaluación de métricas
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred),
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R2: {r2:.4f}")
print(f"MSE USD: {mse:.4f}")
print(f"RMSE USD: {rmse[0]:.4f}")
print(f"MAE USD: {mae:.4f}")


# Guardo modelo en el disco
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Modelo guardado en 'models/model.pkl'")

