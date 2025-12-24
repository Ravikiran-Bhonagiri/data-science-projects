# Autoencoders: Neural Compression for Anomaly Detection

**PayPal Fraud Detection (Deep Learning Era):**

PayPal processes 1B+ transactions/quarter with complex fraud patterns.

**Traditional Isolation Forest:**
- Hand-engineered features: amount, merchant, location
- Catches 72% of fraud
- **Problem:** Misses sophisticated fraud (account takeovers, coordinated attacks)

**Autoencoder approach:**
```python
# Train on normal transactions only
autoencoder = Sequential([
    Dense(128, activation='relu'),  # Encoder
    Dense(32, activation='relu'),   # Bottleneck
    Dense(128, activation='relu'),  # Decoder
    Dense(256, activation='sigmoid')
])

autoencoder.fit(normal_transactions, normal_transactions)  # Learn to reconstruct
```

**Anomaly detection logic:**
```python
reconstructed = autoencoder.predict(new_transaction)
reconstruction_error = mse(new_transaction, reconstructed)

if reconstruction_error > threshold:
    flag_as_fraud()  # Model can't reconstruct it → anomaly
```

**Why it works:**
- Model learns what "normal" looks like
- Fraud transactions have high reconstruction error (model fails to recreate them)
- Captures complex, non-linear patterns traditional methods miss

**Results:**
- Fraud detection: 72% → 89%
- **Additional fraud caught:** $340M/year
- False positive rate: Maintained at <0.1%

---

## Production Scenarios

**1. Manufacturing defect detection:**
- Train autoencoder on images of perfect products
- High reconstruction error on defective items → Flag for QA

**2. Network intrusion detection:**
- Learn normal network traffic patterns
- Unusual patterns (DDoS, hacking) have high reconstruction error

**3. Credit card transaction monitoring:**
- Each user has unique spending pattern
- Autoencoder learns individual's "normal"
- Flags deviations (stolen card usage)

---

## 1. The Architecture

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Load MNIST (Digits)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# 1. The Encoder (Compress 784 pixels -> 32 numbers)
input_img = layers.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(32, activation='relu')(encoded) # Bottleneck

# 2. The Decoder (Decompress 32 -> 784 pixels)
decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

# 3. The Model
autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 4. Train (X is both Input AND Target!)
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
```

---

## 2. Anomaly Detection with Reconstruction Error

```python
# Get reconstruction error for test set
reconstructions = autoencoder.predict(x_test)
loss = tf.keras.losses.mae(reconstructions, x_test)

# Plot histogram of errors
plt.hist(loss[None,:], bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

# Set Threshold
threshold = np.mean(loss) + np.std(loss)
print(f"Anomaly Threshold: {threshold}")

# Detect
anomalies = loss > threshold
print(f"Found {np.sum(anomalies)} anomalies in test set.")
```

---

## 3. Variational Autoencoders (VAE) - The "Generative" Twist

Standard Autoencoders learn a distinct code for each image.
**VAEs** learn a **Probability Distribution** (Mean + Variance) for the latent space.

*   **AE:** "This point is a 7."
*   **VAE:** "This region of space contains 7-ish things."

**Why VAE?**
You can sample from the latent space to **generate new, unique digits** (Generative AI). VAEs are the grandmother of Stable Diffusion. This is where Unsupervised Learning meets GenAI.
