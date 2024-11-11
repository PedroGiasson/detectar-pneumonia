import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np

# Carregando os dados pré-processados
train_images = np.load("train_images.npy")
train_labels = np.load("train_labels.npy")
val_images = np.load("val_images.npy")
val_labels = np.load("val_labels.npy")
test_images = np.load("test_images.npy")
test_labels = np.load("test_labels.npy")


# Definindo o modelo CNN
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Ajuda a evitar o overfitting
        Dense(1, activation='sigmoid')  # Saída binária (0 para NORMAL, 1 para PNEUMONIA)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Criação do modelo
model = create_model()
model.summary()

# Configuração do Early Stopping para interromper o treinamento caso o modelo pare de melhorar
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Treinamento do modelo
history = model.fit(train_images, train_labels,
                    validation_data=(val_images, val_labels),
                    epochs=20,
                    batch_size=32,
                    callbacks=[early_stopping])

# Avaliação do modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# Salvando o modelo após o treinamento
model.save("modelo_pneumonia.h5")

print("Acurácia no conjunto de teste:", test_accuracy)
