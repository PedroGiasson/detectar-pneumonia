import os
import glob
import cv2
import numpy as np

# Caminhos para os diretórios das imagens
base_path = "C:/Users/Sinsoft/Downloads/chest_xray/chest_xray"
train_dir = os.path.join(base_path, "train")
test_dir = os.path.join(base_path, "test")
val_dir = os.path.join(base_path, "validate")

# Função para carregar e pré-processar as imagens
def load_and_preprocess_images(directory):
    images = []
    labels = []

    for label in ["NORMAL", "PNEUMONIA"]:
        # Caminho para as imagens com a etiqueta específica
        path = os.path.join(directory, label, "*.jpeg")
        
        for img_path in glob.glob(path):
            # Carregar a imagem em escala de cinza
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            # Redimensionar a imagem para um tamanho fixo, como 128x128 pixels
            image = cv2.resize(image, (128, 128))
            
            # Aplicar um filtro de desfoque Gaussiano para suavizar a imagem e reduzir ruídos
            image = cv2.GaussianBlur(image, (5, 5), 0)
            
            # Aplicação de segmentação usando uma técnica de limiarização
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            
            # Normalizar a imagem para valores entre 0 e 1
            image = image / 255.0

            # Adicionar a imagem e a etiqueta (0 para NORMAL, 1 para PNEUMONIA) às listas
            images.append(image)
            labels.append(0 if label == "NORMAL" else 1)

    # Converter as listas em arrays NumPy para uso no modelo
    images = np.array(images).reshape(-1, 128, 128, 1)  # Dimensões: número de imagens, altura, largura, 1 (canal)
    labels = np.array(labels)

    return images, labels

# Carregar e pré-processar o conjunto de treinamento, validação e teste
train_images, train_labels = load_and_preprocess_images(train_dir)
test_images, test_labels = load_and_preprocess_images(test_dir)
val_images, val_labels = load_and_preprocess_images(val_dir)

# Salvando os dados pré-processados
np.save("train_images.npy", train_images)
np.save("train_labels.npy", train_labels)
np.save("val_images.npy", val_images)
np.save("val_labels.npy", val_labels)
np.save("test_images.npy", test_images)
np.save("test_labels.npy", test_labels)


print("Conjunto de treinamento:", train_images.shape, train_labels.shape)
print("Conjunto de teste:", test_images.shape, test_labels.shape)
print("Conjunto de validação:", val_images.shape, val_labels.shape)
