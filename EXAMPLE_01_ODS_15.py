# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:48:52 2024

@author: Jaime
"""
#==============================================================================
# EXAMPLE 01 ODS 15
#==============================================================================
"""
Vinuesa et al. (2020) e Isabelle & Westerlund (2022) enfatizan la importancia 
de la inteligencia artificial (IA) en la conservación ambiental, crucial para 
alcanzar el Objetivo de Desarrollo Sostenible 15 de la ONU sobre Vida 
Terrestre. La IA permite monitorear la biodiversidad y gestionar amenazas como 
incendios y extinción de especies con gran precisión. También facilita la toma 
de decisiones en tiempo real y promueve la ciencia ciudadana, democratizando 
la conservación. Subrayan la necesidad de regulación para asegurar un 
desarrollo sostenible y mantener la transparencia y ética.

Vinuesa, R., Azizpour, H., Leite, I., Balaam, M., Dignum, V., Domisch, S., 
Felländer, A., Langhans, S. D., Tegmark, M., & Nerini, F. F. (2020). The role 
of artificial intelligence in achieving the Sustainable Development Goals. 
Nature Communications, 11(1). https://doi.org/10.1038/s41467-019-14108-y.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import folium

# Simulating wildfire data
data = {
    'Temperature': np.random.normal(30, 5, 1000),  # Temperatures in degrees Celsius
    'Humidity': np.random.normal(45, 15, 1000),     # Humidity in %
    'WindSpeed': np.random.normal(10, 3, 1000),  # Wind speed in km/h
    'Precipitation': np.random.uniform(0, 20, 1000),   # Precipitation in mm
    'Fire': np.random.choice([0, 1], 1000, p=[0.7, 0.3])  # 1 indicates fire, 0 no fire
}
df = pd.DataFrame(data)

# Data preprocessing
X = df.drop('Fire', axis=1)
y = df['Fire']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construction of the neural network model
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(X_train_scaled, y_train, epochs=50, batch_size=10, verbose=0)

# Model evaluation
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# Data visualization on a map
map = folium.Map(location=[40.7128, -74.0060], zoom_start=8)
for _, row in df.iterrows():
    color = 'green' if row['Fire'] == 0 else 'red'
    folium.CircleMarker(location=[row['Temperature'], row['Humidity']],
                        radius=5, color=color).add_to(map)
map.save('fire_map.html')

print("Fire map saved as 'fire_map.html'.")
