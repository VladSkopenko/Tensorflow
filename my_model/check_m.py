import tensorflow as tf
from PIL import Image
import numpy as np
loaded_model = tf.keras.models.load_model('mnist_model97.h5')

image = Image.open('0.png').convert('L')
image = image.resize((28, 28))

image_array = np.array(image)

image_array = image_array / 255.0

# Добавьте дополнительное измерение, чтобы оно соответствовало входу модели (batch_size, height, width, channels)
image_array = np.expand_dims(image_array, axis=0)
image_array = np.expand_dims(image_array, axis=-1)

# Сделайте предсказание
predictions = loaded_model.predict(image_array)

# Получите класс с наивысшей вероятностью
predicted_class = tf.argmax(predictions, axis=1).numpy()[0]

print(f'Predicted class: {predicted_class}')
print("Нічого не працює)")