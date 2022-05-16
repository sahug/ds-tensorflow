**Tensorflow**

**Basics**: https://github.com/sahug/ds-tensorflow/blob/master/Tensorflow%20-%20Basics.ipynb
**Beginners**: https://github.com/sahug/ds-tensorflow/blob/master/Tensorflow%20-%20Beginners.ipynb
**Models**: https://github.com/sahug/ds-tensorflow/blob/master/Tensorflow%20-%20Complete%20ML%20Tutorial%20with%20TF%20and%20Keras.ipynb


**Import**: Tensorflow: `import tensorflow as tf` or Keras: `from tensorflow import keras`

 **Dataset**: 
> `from tensorflow.keras.datasets import mnist`
> 
> `(x_train, y_train), (x_test, y_test) = mnist.load_data()`

**One Hot Encoding**
> `y_train = tf.keras.utils.to_categorical(y_train)`

**Model**

**Sequential API**
`from tensorflow import keras`
`from tensorflow.keras import Sequential`
`from tensorflow.keras.layers import Dense, Dropout`
`model = tf.keras.models.Sequential([`
  `tf.keras.layers.Flatten(input_shape=(28, 28)),`
  `tf.keras.layers.Dense(128, activation="relu"),`
  `tf.keras.layers.Dropout(0.2),`
  `tf.keras.layers.Dense(10)                                   `
`])`

**Funvtional API**
`import tensorflow as tf`
`from tensorflow import keras`
`inputs = keras.Input(shape=(28,28))`
`flatten = keras.layers.Flatten()`
`dense1 = keras.layers.Dense(128, activation='relu')`

`dense2 = keras.layers.Dense(10, activation='softmax', name="category_output")`
`dense3 = keras.layers.Dense(1, activation='sigmoid', name="leftright_output")`
`x = flatten(inputs)`
`x = dense1(x)`
`outputs1 = dense2(x)`
`outputs2 = dense3(x)`

`model = keras.Model(inputs=inputs, outputs=[outputs1, outputs2], name="mnist_model")`

**Loss and Optimizers**
`loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)`
`optim = keras.optimizers.Adam(lr=0.001)`
`metrics = ["accuracy"]`
`model.compile(loss=losses, optimizer=optim, metrics=metrics)`

**Eager Tensors**
