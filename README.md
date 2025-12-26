# Deepedge-assessment
# ML Assignment - Supervised Regression :
## NAME : MAMTHA I
## PROBLEM STATEMENT :
Using Deep Learning techniques, predict the coordinates (x,y) of a pixel which has a value of
255 for 1 pixel in a given 50x50 pixel grayscale image and all other pixels are 0. The pixel with a
value of 255 is randomly assigned. You may generate a dataset as required for solving the
problem. Please explain your rationale behind dataset choices.

## CODE :
```Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

def generate_dataset(num_samples=5000, img_size=50):
    """
    Generate images with one pixel set to 255 and
    corresponding (x, y) labels.
    """
    images = []
    labels = []

    for _ in range(num_samples):
        img = np.zeros((img_size, img_size), dtype=np.float32)

        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        img[y, x] = 255.0

        images.append(img)
        labels.append([x, y])

    images = np.array(images) / 255.0
    labels = np.array(labels) / (img_size - 1)

    return images, labels

X, y = generate_dataset(num_samples=8000)

X = X[..., np.newaxis] 

X_train, X_val, y_train, y_val = train_test_split( X, y, test_size=0.2, random_state=42)

def build_model():
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(50, 50, 1)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)  # x, y coordinates
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model

model = build_model()
model.summary()

history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_data=(X_val, y_val)
)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training Loss Curve')
plt.show()

preds = model.predict(X_val)

preds_pixels = preds * 49
gt_pixels = y_val * 49

plt.figure(figsize=(6, 6))
plt.scatter(gt_pixels[:, 0], gt_pixels[:, 1],
            label='Ground Truth', alpha=0.5)
plt.scatter(preds_pixels[:, 0], preds_pixels[:, 1],
            label='Predicted', alpha=0.5)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.title('Ground Truth vs Predicted Pixel Coordinates')
plt.show()

idx = np.random.randint(0, len(X_val))
img = X_val[idx].squeeze()

plt.imshow(img, cmap='gray')
plt.title(
    f"GT: {gt_pixels[idx].astype(int)} | "
    f"Pred: {preds_pixels[idx].astype(int)}"
)
plt.show()

```
## MODEL IMAGE :
<img width="590" height="363" alt="image" src="https://github.com/user-attachments/assets/22752e8e-180f-4786-912b-0260eef83e4f" />


## GRAPH :
<img width="592" height="460" alt="image" src="https://github.com/user-attachments/assets/ff35498f-5629-461b-bb0a-0f9c39c288ee" />


## GROUND TRUTH vs PREDICTED PIXEL COORDINATES :

<img width="534" height="546" alt="image" src="https://github.com/user-attachments/assets/9b6b929f-4c0c-44f1-9366-6f3d49bee32f" />


## OUTPUT IMAGE :
<img width="419" height="435" alt="image" src="https://github.com/user-attachments/assets/78921413-3fa1-4221-99a3-c2dcc551b6a9" />




















