package com.example.tfcapi;

import java.io.IOException;
import java.nio.file.Files;

/**
 * A small fully-connected neural network (one hidden ReLU layer + softmax
 * output) that is <em>created and trained from scratch</em> through the
 * TensorFlow C API &mdash; no SavedModel, no Python.
 *
 * <p>The graph is built natively with Placeholder/MatMul/Relu ops, the loss is
 * {@code sparse_softmax_cross_entropy_with_logits}, gradients come from
 * {@code TF_AddGradients} and SGD updates are applied on native parameter
 * tensors.</p>
 *
 * <pre>{@code
 * try (SimpleNeuralNet net = new SimpleNeuralNet(784, 128, 10, 0.15f)) {
 *     History h = net.fit(trainX, trainY, 10, 128);  // full training loop
 *     float loss = net.trainBatch(batchX, batchY);   // or one manual SGD step
 *     float[] probs = net.predict(batchX);           // rows of 10 probabilities
 * }
 * }</pre>
 *
 * <p>For multi-layer architectures see {@link Sequential}.</p>
 */
public final class SimpleNeuralNet extends AbstractClassifier {

    private static final int FORMAT_MAGIC = 0x54464D4C; // "TFML"

    private volatile long netAddr;
    private final int inputDim;
    private final int hiddenUnits;
    private final int numClasses;

    public SimpleNeuralNet(int inputDim, int hiddenUnits, int numClasses, float learningRate) {
        TensorFlow.load();
        if (inputDim <= 0 || hiddenUnits <= 0 || numClasses <= 0 || learningRate <= 0) {
            throw new IllegalArgumentException("all constructor arguments must be positive");
        }
        this.inputDim = inputDim;
        this.hiddenUnits = hiddenUnits;
        this.numClasses = numClasses;
        this.netAddr = nativeCreate(inputDim, hiddenUnits, numClasses, learningRate);
    }

    /**
     * Restores a network previously written by {@link #saveTo(Path)}.
     * The learning rate of the returned network is {@code learningRate}
     * (training can continue from the restored weights).
     */
    public static SimpleNeuralNet loadFrom(java.nio.file.Path file,
                                           float learningRate) throws IOException {
        try (java.io.DataInputStream in = new java.io.DataInputStream(
                new java.io.BufferedInputStream(Files.newInputStream(file)))) {
            int magic = in.readInt();
            if (magic != FORMAT_MAGIC) {
                throw new IOException("not a SimpleNeuralNet checkpoint: " + file);
            }
            int inputDim = in.readInt();
            int hidden = in.readInt();
            int classes = in.readInt();
            float[] params = new float[in.readInt()];
            for (int i = 0; i < params.length; i++) {
                params[i] = in.readFloat();
            }
            SimpleNeuralNet net = new SimpleNeuralNet(inputDim, hidden, classes, learningRate);
            net.nativeSetParams(net.netAddr, params);
            return net;
        }
    }

    /** Writes architecture + weights to {@code file} (little-endian, portable). */
    public void saveTo(java.nio.file.Path file) throws IOException {
        ensureOpen();
        float[] params = nativeGetParams(netAddr);
        Files.createDirectories(file.toAbsolutePath().getParent());
        try (java.io.DataOutputStream out = new java.io.DataOutputStream(
                new java.io.BufferedOutputStream(Files.newOutputStream(file)))) {
            out.writeInt(FORMAT_MAGIC);
            out.writeInt(inputDim);
            out.writeInt(hiddenUnits);
            out.writeInt(numClasses);
            out.writeInt(params.length);
            for (float p : params) {
                out.writeFloat(p);
            }
        }
    }

    /** One SGD step over a batch; returns the mean training loss. */
    @Override
    public synchronized float trainBatch(float[] batchX, int[] batchY) {
        ensureOpen();
        if (batchY == null || batchY.length == 0 ||
                batchX == null || batchX.length != batchY.length * inputDim) {
            throw new IllegalArgumentException(
                    "batchX length must equal batchY.length * inputDim (" + inputDim + ")");
        }
        return nativeTrainBatch(netAddr, batchX, batchY);
    }

    @Override
    public synchronized float[] predict(float[] batchX) {
        ensureOpen();
        if (batchX == null || batchX.length == 0 || batchX.length % inputDim != 0) {
            throw new IllegalArgumentException(
                    "batchX length must be a positive multiple of inputDim (" + inputDim + ")");
        }
        return nativePredict(netAddr, batchX);
    }

    @Override
    public int inputElems() { return inputDim; }

    @Override
    public int numClasses() { return numClasses; }

    public int hiddenUnits() { return hiddenUnits; }

    @Override
    public synchronized void close() {
        if (netAddr != 0) {
            nativeClose(netAddr);
            netAddr = 0;
        }
    }

    private void ensureOpen() {
        if (netAddr == 0) {
            throw new IllegalStateException("network is closed");
        }
    }

    private static native long nativeCreate(int inputDim, int hiddenUnits,
                                            int numClasses, float learningRate);
    private native float nativeTrainBatch(long handle, float[] batchX, int[] batchY);
    private native float[] nativePredict(long handle, float[] batchX);
    private native float[] nativeGetParams(long handle);
    private native void nativeSetParams(long handle, float[] params);
    private native void nativeClose(long handle);
}
