**Tensorflow**

**Keras API**: https://keras.io/

**Data Preprocessing**: https://keras.io/api/layers/preprocessing_layers/

**Basics**: https://github.com/sahug/ds-tensorflow/blob/master/Tensorflow%20-%20Basics.ipynb

**Beginners**: https://github.com/sahug/ds-tensorflow/blob/master/Tensorflow%20-%20Beginners.ipynb

**Models**: https://github.com/sahug/ds-tensorflow/blob/master/Tensorflow%20-%20Complete%20ML%20Tutorial%20with%20TF%20and%20Keras.ipynb


**Import**: Tensorflow: `import tensorflow as tf` or Keras: `from tensorflow import keras`

 **Dataset**: 

 There are different ways you can feed the data to a keras model. A keras model takes tensors or Numpy as an input so you can either have a datset as tensors or numpy.

 You can use pandas, numpy, sklearn for data analysis. Once the data is preprocessed and ready to feed to the keras model. You can directly feed the data as numpy or as tensors.

If you are using the dataset provided by Tensorflow you can use some of the below methods.

> `from tensorflow.keras.datasets import mnist`
> 
> `(x_train, y_train), (x_test, y_test) = mnist.load_data()`

- `tfds.load()`: Use this for inbuilt dataset available in Tensorflow. As the dataset is processed already and you don't need much control on dataset.

**One Hot Encoding**
> `y_train = tf.keras.utils.to_categorical(y_train)`

**Model**

**Sequential API**

> `model = tf.keras.models.Sequential([`
> 
  > `tf.keras.layers.Flatten(input_shape=(28, 28)),`
  > 
  > `tf.keras.layers.Dense(128, activation="relu"),`
  > 
  > `tf.keras.layers.Dense(10)                                   `
  > 
> `])`

or

> `model = keras.models.Sequential()`
> 
> `model.add(keras.layers.Flatten(input_shape=(28,28))`
> 
> `model.add(keras.layers.Dense(128, activation='relu'))`
> 
> `model.add(keras.layers.Dense(10))`

**Functional API**

> `inputs = keras.Input(shape=(input_shape))`
>
> `x = layers.Rescaling(1.0/255.0)(inputs)`
>
> `x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation=activations.relu)(x)`
>
> `x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation=activations.relu)(x)`
>
> `x = layers.MaxPool2D(pool_size=(2, 2))(x)`
>
> `outputs = layers.Dense(10, activation=activations.softmax)(x)`
>
> `model = keras.Model(inputs=inputs, outputs=[outputs], name="mnist_model")`

**Loss and Optimizers**

> `loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)`
> 
> `optim = keras.optimizers.Adam(lr=0.001)`
> 
> `metrics = [keras.metrics.SparseCategoricalAccuracy()]`
> 
> `model.compile(loss=losses, optimizer=optim, metrics=metrics)`

**Eager Tensors**


**Hyperparameter Tunning**

> `import keras_tuner as kt`
> 
> `tuner = kt.Hyperband(model, objective='val_accuracy', max_epochs=10, factor=3, directory='my_dir', project_name='intro_to_kt')`
> 
> `tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])`
> 
> `best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]`
> 
> `model = tuner.hypermodel.build(best_hps)`
> 
> `history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)`

**Table**
|Project|Coverage|
|-------|--------|
|Tensorflow - 2D CNN - MNIST Digit Recognition|tfds.load, Data Pipeline, Functional API, Rescaling, Loss, Optimizer, Metrics, Tensorboard|
|Tensorflow - 2D CNN CIFAC10 Image Classifier|tfds.load, Data Pipeline, Functional API, Rescaling, Loss, Optimizer, Metrics, Tensorboard|
|Tensorflow - Bank Customer Satisfaction Using CNN and Feature Selection|Normalization, Functional API, Loss, Optimizer, Metrics, KerasClassifier, DataframeFunctionTransformer, Pipeline|
|Tensorflow - Breast Cancer Detection Using CNN|Normalization, Functional API, Loss, Optimizer, Metrics, KerasClassifier, DataframeFunctionTransformer, Pipeline|
|Tensorflow - Backpropagation With Tensorflow|Backpropogation using Tensorflow Core|
|Tensorflow - Basic Image Classification MNIST Dataset|Plain simple Tensorflow Model|
|Tensorflow - Credit Card Fraud Detection Using CNN|Normalization, Functional API, Loss, Optimizer, Metrics, KerasClassifier, DataframeFunctionTransformer, Pipeline|
|Tensorflow - VGG16 - Classification - Dog vs Cat|VGG16, Sequential API, Loss, Optimizer, Metrics, ImageDataGenerator|
|Tensorflow - Google Stock Price Prediction Using RNN-LSTM|Time Series, Functional API|
|Tensorflow - IMDB Sentiments Classification Using RNN-LSTM|IMDB, Pad Sequences, Sequential API|
|Tensorflow - Keras - Deep Learning (DL) and Artificial Neural Network (ANN)|ANN, Sequential API|
|Tensorflow - Malaria Parasite Detection Using CNN|Functional API, Loss, Optimizer, Metrics, ImageDataGenerator|
|Tensorflow - Multi-Label Image Classification Using CNN|Multi Label, Functional API, Loss, Optimizer, Metrics, Image, TQDM|
|Tensorflow - Power Consumption - Multi-Step Predictions Using RNN-LSTM|RNN - LSTM|
|Tensorflow - Save Best Model - Checkpoint and Callbacks|Sequential API, ModelCheckpoint, Callbacks|
|Tensorflow - Text Classification using Tensorflow Hub|Tensorflow HUB, Tensorflow Dataset, Sequential API, KerasLayer Embedding, Batch Training and Validation|
|Tensorflow - Tune Hyperparameter with Keras Tuner|Sequential API, EarlyStopping, KerasTuner|
|Tensorflow - Using Pre Trained Models - VGG16|VGG16, Preprocessing, Sequential API, EarlyStopping, KerasTuner|
|Tensorflow - Word Embedding in NLP On Twitter Sentiment Data|Sequential API, Preprocessing, Tokenizer, Embedding|
