# MNIST ANN — Optimized Deep Learning Model



A lightweight Artificial Neural Network (ANN) for **handwritten digit classification**, optimized using **quantization and pruning** for efficient deployment.



## Features



* Classifies handwritten digits (`0–9`)

* Trained on the **MNIST dataset**

* Post-training quantization for reduced model size

* Magnitude-based pruning for weight sparsity

* TensorFlow Lite model for lightweight inference



## Models



| File                     | Description                       |

| ------------------------ | --------------------------------- |

| `baseline\_ann.h5`        | Original trained ANN              |

| `pruned\_model.h5`        | ANN with \~50% weight sparsity     |

| `quantized\_model.tflite` | Lightweight TensorFlow Lite model |



## Usage



### 1. Install dependencies



```bash

pip install tensorflow numpy pillow

```



### 2. Load the model



```python

from tensorflow.keras.models import load\_model



model = load\_model("baseline\_ann.h5")

```



### 3. Predict a digit



Provide a **28×28 grayscale image**, normalize pixel values to `\[0,1]`, and run:



```python

prediction = model.predict(image)

digit = prediction.argmax()

print("Predicted digit:", digit)

```



For deployment on resource-constrained devices, use `quantized\_model.tflite` with the \*\*TensorFlow Lite Interpreter\*\*.



## Performance



| Model     |        Size | Accuracy |

| --------- | ----------: | -------: |

| Baseline  |     ~400 KB |  ~97–98% |

| Quantized | ~100–110 KB |     ~97% |

| Pruned    | ~390–400 KB |     ~97% |



## Tech Stack



`Python` · `TensorFlow/Keras` · `TensorFlow Lite` · `NumPy` · `Pillow`



## Use Cases



This model can be used as a lightweight digit-recognition component in:



* OCR applications

* Handwritten form processing

* Educational applications

* Edge/mobile ML prototypes

* Digit recognition systems



