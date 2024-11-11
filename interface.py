import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf

# Carregar o modelo treinado
model = tf.keras.models.load_model("modelo_pneumonia.h5") 

# Função para carregar e processar a imagem
def process_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Redimensionar para o tamanho do modelo
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    image = image / 255.0  # Normalizar
    return image.reshape(1, 128, 128, 1)

# Função para carregar uma imagem
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpeg;*.jpg;*.png")])
    if file_path:
        global loaded_image_path
        loaded_image_path = file_path
        
        # Carregar e exibir a imagem na interface
        img = Image.open(file_path)
        img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img)
        img_label.configure(image=img_tk)
        img_label.image = img_tk
    else:
        messagebox.showwarning("Aviso", "Nenhuma imagem selecionada.")

# Função para realizar o diagnóstico
def predict_diagnosis():
    if loaded_image_path:
        processed_image = process_image(loaded_image_path)
        prediction = model.predict(processed_image)
        
        # Interpretação da predição
        if prediction[0][0] > 0.5:
            result = "Diagnóstico: Pneumonia"
        else:
            result = "Diagnóstico: Normal"
        
        result_label.config(text=result)
    else:
        messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

# Configuração da interface Tkinter
root = tk.Tk()
root.title("Detecção de Pneumonia em Imagens de Raio-X")
root.geometry("400x400")

# Botão para carregar a imagem
load_button = tk.Button(root, text="Carregar Imagem", command=load_image)
load_button.pack(pady=10)

# Rótulo para exibir a imagem carregada
img_label = tk.Label(root)
img_label.pack(pady=10)

# Botão para realizar o diagnóstico
predict_button = tk.Button(root, text="Diagnosticar", command=predict_diagnosis)
predict_button.pack(pady=10)

# Rótulo para exibir o resultado do diagnóstico
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

# Variável global para armazenar o caminho da imagem carregada
loaded_image_path = None

# Iniciar a interface
root.mainloop()
