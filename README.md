# tensorflow-capi-java

A **self-contained Java machine-learning library** built directly on the
TensorFlow C API through a hand-written JNI bridge. No official TensorFlow
Java dependency, no Python, no Keras &mdash; one JAR carries everything,
including the 909 MB native TensorFlow runtime.

```
com.example.tfcapi
├── TensorFlow        native loader + C API version
├── TfModel           load & run exported SavedModels
├── SimpleNeuralNet   build & train an MLP from scratch (SGD + gradients)
├── Sequential        Keras-style layer API: Conv2D, MaxPool2D, Dense, ...
├── IdxDataset        gzip IDX reader (MNIST / Fashion-MNIST)
└── Metrics           DL4J-style evaluation reports + confusion matrices
```

## Using it in your project

first read how_to_run_manually.txt
build locally and after can use:

Install into your local Maven repository (`mvn install`), then:

```xml
<dependency>
  <groupId>com.example</groupId>
  <artifactId>tensorflow-capi-java</artifactId>
  <version>1.3.0</version>
</dependency>
```

Natives (`tensorflow.dll`, `tf_capi_jni.dll`) are packed under
`/native/windows-x86_64/` inside the JAR and extracted on first use to
`%USERPROFILE%\.tfcapi\cache\windows-x86_64\`. No PATH or environment setup is
needed. Set `TF_CAPI_NATIVE_DIR` to override with your own DLL folder.

## Inference from a SavedModel

```java
try (TfModel model = new TfModel("fashion_model")) {
    float[] probs = model.run(pixels, new long[]{batch, 28, 28});
}
```

## Training from scratch

### Layer API (Sequential)

Compose any architecture from `Input`/`Input4D`, `Dense`, `Flatten`, `ReLU`,
`Conv2D` and `MaxPool2D`. The graph is built through generic C API bindings
from Java &mdash; new layer types need no native changes.

```java
try (Sequential net = new Sequential(0.01f,
        new Input4D(28, 28, 1),
        new Conv2D(8, 3, 3, 1, 1, Padding.SAME), new ReLU(),
        new MaxPool2D(2, 2, 2, 2, Padding.VALID),
        new Flatten(),
        new Dense(128), new ReLU(),
        new Dense(10))) {
    var h = net.fit(trainX, trainY, 5, 128, testX, testY, true);
    // Epoch 5/5 - loss: 0.3121 - val_loss: 0.3548 - val_acc: 0.8712
    net.saveTo(Path.of("cnn.bin"));
}
Sequential restored = Sequential.loadFrom(Path.of("cnn.bin"), 0.01f);
```

The model must start with an Input/Input4D and end with a Dense (its units
are the class count). Training minimizes sparse-softmax cross-entropy;
gradients come from `TF_AddGradients`.

### Optimizers

Pass any optimizer instead of a learning rate; `Sgd` is the default.

```java
new Sequential(0.01f, layers...)                 // plain SGD
new Sequential(Optimizer.momentum(0.05f, 0.9f), layers...)
new Sequential(Optimizer.adam(0.001f), layers...)

net.setOptimizer(Optimizer.adam(0.003f));        // swap mid-training
Sequential.loadFrom(path, Optimizer.adam(1e-3f)); // restore with optimizer
```

Optimizer state (momentum/second moments) lives in native tensors and is
updated without copying parameters into Java.

### Fixed MLP (SimpleNeuralNet)

One-call Keras-style training (shuffling, batching, optional validation):

```java
try (SimpleNeuralNet net = new SimpleNeuralNet(784, 128, 10, 0.15f)) {
    var h = net.fit(trainX, trainY, 10, 128, testX, testY, true);
    // Epoch 10/10 - loss: 0.2831 - val_loss: 0.3102 - val_acc: 0.8891
    net.saveTo(Path.of("model.bin"));               // checkpoint
}
```

Or step manually:

```java
try (SimpleNeuralNet net = new SimpleNeuralNet(784, 128, 10, 0.15f)) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (each batch) loss = net.trainBatch(batchX, batchY);
    }
    net.saveTo(Path.of("model.bin"));               // checkpoint
}
SimpleNeuralNet restored = SimpleNeuralNet.loadFrom(Path.of("model.bin"), 0.15f);
float[] probs = restored.predict(testBatch);        // rows of class probabilities
```

The network graph (`Relu(x·W1+b1) → logits → LogSoftmax cross-entropy`) is
constructed natively, differentiated with `TF_AddGradients`, and SGD updates
are applied on native parameter tensors.

## Evaluation

```java
Metrics.printReport(loss, accuracy, confusionMatrix, Metrics.FASHION_CLASS_NAMES);
Metrics.printDl4jMatrix(confusionMatrix);
```

## Building from source

Requirements: JDK 17+, Maven 3.9+, MSYS2 with UCRT64 gcc (`pacman -S
mingw-w64-ucrt-x86_64-gcc unzip`). The TensorFlow C distribution is vendored
under `native/dist/` &mdash; builds are fully offline.

```powershell
mvn clean install          # builds JNI DLL, compiles, packages fat jar
mvn compile exec:java      # runs the SavedModel demo (needs ../classification/fashion_model)
```

A complete consumer example lives in `..\fashion-mnist-java`: it trains a
Fashion-MNIST classifier from scratch (~86% test accuracy) and shows
predictions in a Swing/JFreeChart viewer.
