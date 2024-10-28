import numpy as np
import cv2
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os


def build_unet(input_shape):
    inputs = layers.Input(input_shape)

    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)

    up8 = layers.UpSampling2D((2, 2))(conv7)
    up8 = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(up8)
    merge8 = layers.concatenate([conv1, up8], axis=3)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge8)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv8)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def load_data(data_dir, mask_dir, img_size=(128, 128)):
    images = []
    masks = []

    for img_file in os.listdir(data_dir):
        img = cv2.imread(os.path.join(data_dir, img_file), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, img_size) / 255.0
        images.append(img)

        mask_file = img_file.replace('.png', '_mask.png')
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size) / 255.0
        masks.append(mask)

    return np.expand_dims(np.array(images), axis=-1), np.expand_dims(np.array(masks), axis=-1)


def refine_segmentation_with_active_contours(predicted_mask):
    predicted_mask_uint8 = (predicted_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(predicted_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    init = np.zeros_like(predicted_mask, dtype=np.uint8)
    cv2.drawContours(init, contours, -1, (255, 255, 255), 1)

    refined_contour = cv2.GaussianBlur(init, (5, 5), 0)
    return refined_contour


def main():
    data_dir = './images/'
    mask_dir = './liver_masks/'
    img_size = (128, 128)
    batch_size = 64
    epochs = 20
    model_save_path = './unet_model3.h5'

    X, Y = load_data(data_dir, mask_dir, img_size)
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    unet_model = build_unet((*img_size, 1))
    unet_model.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=[Accuracy()])

    unet_model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val))

    unet_model.save(model_save_path)

    test_img = X_val[0]
    test_mask = Y_val[0]
    predicted_mask = unet_model.predict(np.expand_dims(test_img, axis=0))[0]

    refined_mask = refine_segmentation_with_active_contours(predicted_mask)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title('Imagem Original')
    plt.imshow(test_img.squeeze(), cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Máscara Predita')
    plt.imshow(predicted_mask.squeeze(), cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Máscara Refinada')
    plt.imshow(refined_mask.squeeze(), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
