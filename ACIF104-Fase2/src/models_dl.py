
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm(input_shape):
    """
    Construye modelo LSTM (Long Short-Term Memory).
    Ideal para capturar dependencias a largo plazo en series de tiempo.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(64, return_sequences=True)(inputs) # Retorna secuencia completa para la siguiente capa LSTM
    x = layers.Dropout(0.2)(x) # Dropout para evitar Overfitting
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x) # Salida regresión (1 valor)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape):
    """
    Construye modelo GRU (Gated Recurrent Unit).
    Similar a LSTM pero computacionalmente más eficiente (menos parámetros).
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.GRU(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.GRU(32)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_transformer(input_shape):
    """
    Transformer Encoder para Series de Tiempo.
    Utiliza mecanismo de Atención (MultiHeadAttention) para ponderar qué momentos pasados
    son más relevantes para la predicción actual.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Bloque Transformer Encoder
    # MultiHeadAttention: El modelo aprende a 'prestar atención' a diferentes partes de la serie
    x = inputs
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Dropout(0.1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + inputs) # Conexión residual y Normalización
    
    # Red Feed Forward interna del Transformer
    # Usamos Conv1D como una capa densa aplicada a cada paso de tiempo
    ff = layers.Conv1D(filters=64, kernel_size=1, activation="relu")(x)
    ff = layers.Dropout(0.1)(ff)
    ff = layers.Conv1D(filters=input_shape[-1], kernel_size=1)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    
    # Global Average Pooling para aplanar la secuencia temporal a un vector fijo
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model
