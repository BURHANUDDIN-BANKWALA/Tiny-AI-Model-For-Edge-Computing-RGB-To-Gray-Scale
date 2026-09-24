# Grayscale Image Preprocessing and Learning(16 Bytes,4 Params) 

This notebook implements a simple image-to-grayscale learning pipeline using a **single 1×1 convolution layer** in TensorFlow/Keras.

The notebook has two main stages:

1. Prepare and normalize images from the `Normal_Images` directory.
2. Train a neural network to learn a grayscale representation and apply it to images from the `dip` directory.

---

## Project Structure

```text
project/
│
├── preprocessing_2.ipynb
├── Normal_Images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── dip/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── Grey.h5
└── grey predicted.png
```

---

## Requirements

The notebook uses:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV
- Pillow
- TensorFlow / Keras

Install the main dependencies with:

```bash
pip install numpy pandas matplotlib seaborn opencv-python pillow tensorflow
```

---

## 1. Loading the Training Images

Images are read from:

```text
Normal_Images/
```

The notebook supports:

- `.jpg`
- `.jpeg`
- `.png`
- `.bmp`

OpenCV is used to read the images.

```python
images.append(cv2.imread(str(i)))
```

---

## 2. Image Resizing

Each input image is resized to:

```text
320 × 320
```

using OpenCV:

```python
cv2.resize(img, (320, 320))
```

The notebook therefore expects the model input to have the shape:

```text
320 × 320 × 3
```

where the three channels correspond to the color channels loaded by OpenCV.

---

## 3. Pixel Normalization

Pixel values originally lie approximately in:

```text
0 – 255
```

They are converted to floating-point values in approximately:

```text
0 – 1
```

using:

```python
image / 255.0
```

The implementation then rounds the values to two decimal places.

---

## 4. Creating the Grayscale Target

The target (`y_true`) is generated manually from every normalized image.

For each pixel, the three channel values are averaged:

```python
grey_pixel = np.mean(image[i, j, :])
```

This produces a single grayscale value for every pixel.

Therefore:

```text
Input:
320 × 320 × 3

Target:
320 × 320
```

Conceptually:

```text
RGB pixel
   │
   ├── Channel 1
   ├── Channel 2
   └── Channel 3
          │
          ▼
     Channel Mean
          │
          ▼
    Grayscale Pixel
```

---

## 5. Train/Test Split

The images are divided using an 80/20 split based on their sorted order:

```python
train_images = resized_images[:int(len(resized_images) * 0.8)]
test_images = resized_images[int(len(resized_images) * 0.8):]
```

The same split is applied to the generated grayscale targets.

> **Note:** This is an order-based split rather than a randomized split. If the dataset has any ordering pattern, the train/test sets may not be statistically independent.

---

## 6. Model Architecture

The notebook intentionally uses a very small neural network:

```python
model = Sequential([
    Conv2D(
        1,
        1,
        input_shape=(320, 320, 3),
        activation='relu'
    )
])
```

Architecture:

```text
Input Image
320 × 320 × 3
       │
       ▼
1 × 1 Conv2D
1 filter
       │
       ▼
320 × 320 × 1
       │
       ▼
Grayscale Image
```

A **1×1 convolution** processes each pixel independently across its input channels.

It can therefore learn a weighted combination of the three input channels:

```text
Output ≈ w1 × C1 + w2 × C2 + w3 × C3 + bias
```

This makes the model particularly simple and interpretable compared with a conventional CNN.

---

## 7. Training Configuration

The model is trained using:

- Optimizer: Adam
- Learning rate object: `0.1`
- Loss: Mean Absolute Error (`mae`)
- Metric: Mean Squared Error (`mse`)
- Epochs: `200`
- Batch size: `4`

The notebook compiles the model with:

```python
model.compile(
    optimizer='adam',
    loss='mae',
    metrics=['mse']
)
```

### Implementation Note

An `Adam` optimizer with `learning_rate=0.1` is created immediately before compilation, but the `compile()` call uses the string:

```python
optimizer='adam'
```

Therefore, the explicitly created optimizer object and its `0.1` learning rate are **not actually passed to the model**.

If the intention is to use the configured learning rate, the compilation should instead be:

```python
model.compile(
    optimizer=optimizer,
    loss='mae',
    metrics=['mse']
)
```

---

## 8. Training Monitoring

Two plots are generated:

### MAE Loss

Training and validation loss are plotted against epochs.

```text
Epoch → Loss
```

### MSE

Training and validation MSE are also plotted.

These plots can be used to observe convergence and the difference between training and validation behavior.

---

## 9. Prediction

After training, predictions are generated for the test images:

```python
pred_images = model.predict(test_images)
```

The prediction initially contains normalized values.

The notebook converts them back to approximately 0–255 image intensity levels:

```python
pred_images = pred_images * 255.0
pred_images = pred_images.astype(int)
```

The predicted grayscale image can then be displayed using Pillow.

---

## 10. Saving the Model

The trained model is saved as:

```text
Grey.h5
```

using:

```python
model.save('Grey.h5')
```

This allows the trained model to be reused without retraining.

---

# Applying the Model to `dip`

The second part of the notebook loads images from:

```text
dip/
```

These images go through the same preprocessing:

```text
Read image
   ↓
Resize to 320 × 320
   ↓
Normalize to 0–1
   ↓
Model prediction
   ↓
Convert prediction to 0–255
   ↓
Save/display grayscale result
```

The trained model is then used to predict grayscale representations for these images.

One resulting image is saved as:

```text
grey predicted.png
```

---

## End-to-End Pipeline

```text
                 TRAINING
                    │
                    ▼
          Normal_Images Dataset
                    │
                    ▼
             Resize 320×320
                    │
                    ▼
              Normalize
                    │
                    ▼
       Generate Grayscale Target
                    │
                    ▼
             80/20 Split
                    │
                    ▼
            1×1 Conv2D Model
                    │
                    ▼
                Training
                    │
                    ▼
                Grey.h5
                    │
                    │
                    ▼
                 INFERENCE
                    │
                    ▼
               dip Dataset
                    │
                    ▼
             Resize + Normalize
                    │
                    ▼
              Trained Model
                    │
                    ▼
          Predicted Grayscale
                    │
                    ▼
         grey predicted.png
```

---

## Important Technical Observations

### Why a 1×1 convolution?

A 1×1 convolution does not use neighboring pixels. It only combines information from the channels at the same pixel location.

For this task, that is appropriate because the target is created from the average of the input channels.

### Why is the model so small?

The target itself is generated using a simple per-pixel channel operation. A large CNN is unnecessary for reproducing such a transformation.

### Dataset limitation

The grayscale targets are generated directly from the input images. Consequently, this experiment is primarily demonstrating whether a neural network can learn a deterministic pixel-level transformation rather than learning a complex visual concept.

---

## Possible Improvements

For a production-quality preprocessing pipeline, consider:

1. Randomizing the train/test split.
2. Avoiding unnecessary rounding during normalization.
3. Explicitly passing the configured optimizer to `compile()`.
4. Using a fixed random seed for reproducibility.
5. Adding quantitative evaluation such as MAE, MSE, and PSNR.
6. Saving the preprocessing configuration together with the model.
7. Using the modern `.keras` model format instead of legacy `.h5` where appropriate.
8. Creating reusable preprocessing and inference functions.
9. Adding input validation for unreadable/corrupt images.
10. Keeping training, validation, and inference pipelines consistent.

---

## Output Files

| File | Purpose |
|---|---|
| `Grey.h5` | Trained grayscale prediction model |
| `grey predicted.png` | Example grayscale prediction |

---

## Summary

`preprocessing_2.ipynb` demonstrates a compact deep-learning approach for learning grayscale image conversion.

The core idea is:

```text
3-channel image
      ↓
1×1 convolution
      ↓
1-channel grayscale representation
```

Because the transformation is pixel-wise, the experiment provides a useful demonstration of how a convolutional layer can learn channel-wise transformations with very few parameters.
