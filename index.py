import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


# Função para refinar segmentações com contornos ativos
def draw_contour_on_image(original_image, predicted_mask):
    # Converta a máscara predita para uint8 (escala de 0 a 255)
    predicted_mask_uint8 = (predicted_mask * 255).astype(np.uint8)

    # Encontre os contornos na máscara predita
    contours, _ = cv2.findContours(predicted_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crie uma versão colorida da imagem original (grayscale para RGB)
    original_image_colored = cv2.cvtColor((original_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Desenhe os contornos em vermelho (BGR: (0, 0, 255)) na imagem original
    cv2.drawContours(original_image_colored, contours, -1, (0, 0, 255), 1)

    return original_image_colored


# Carregar o modelo treinado
model_path = './unet_model3.h5'
model = load_model(model_path, compile=False)

# Configurações da página
st.title("Segmentação de Fígado")
st.write("Permite segmentar o fígado a partir de imagens de TC utilizando um modelo de segmentação UNet.")

# Carregar a imagem do usuário
uploaded_file = st.file_uploader("Escolha uma imagem de CT", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Ler a imagem carregada
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (128, 128)) / 255.0  # Redimensionar e normalizar

    # Mostrar a imagem original
    st.image(image, caption="Imagem Original", use_column_width=True, clamp=True, channels="GRAY")

    # Fazer a predição
    image_input = np.expand_dims(image_resized, axis=(0, -1))
    predicted_mask = model.predict(image_input)[0].squeeze()

    # Refinar a máscara predita
    refined_mask = draw_contour_on_image(image_resized, predicted_mask)

    # Mostrar resultados
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(predicted_mask, cmap="gray")
    ax1.set_title("Máscara Predita")
    ax2.imshow(refined_mask, cmap="gray")
    ax2.set_title("Máscara Refinada")
    st.pyplot(fig)
