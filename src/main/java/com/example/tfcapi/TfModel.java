package com.example.tfcapi;

/**
 * A SavedModel loaded through the TensorFlow C API.
 *
 * <pre>{@code
 * try (TfModel model = new TfModel("fashion_model")) {
 *     float[] probs = model.run(pixelData, new long[]{500, 28, 28});
 *     long[] shape  = model.outputShape();   // e.g. [500, 10]
 * }
 * }</pre>
 */
public class TfModel implements AutoCloseable {

    // Native peer addresses, set by nativeLoadModel.
    private volatile long graphAddr;
    private volatile long sessionAddr;
    private volatile long statusAddr;
    private volatile long optsAddr;
    private volatile long inputOpAddr;
    private volatile long outputOpAddr;

    private volatile long[] outputShape = new long[0];
    private volatile boolean closed;

    public TfModel(String exportDir, String... tags) {
        TensorFlow.load();
        if (tags == null || tags.length == 0) {
            tags = new String[] {"serve"};
        }
        nativeLoadModel(exportDir, tags);
    }

    /**
     * Runs the model with one batch. The product of {@code inShape} must equal
     * {@code input.length}. Returns the flattened output tensor; the shape of
     * that tensor is available afterwards via {@link #outputShape()}.
     */
    public synchronized float[] run(float[] input, long[] inShape) {
        ensureOpen();
        if (inShape == null || inShape.length == 0 || inShape.length > 8) {
            throw new IllegalArgumentException("inShape must have 1..8 dimensions");
        }
        long expected = 1;
        for (long d : inShape) {
            expected *= d;
        }
        if (expected != input.length) {
            throw new IllegalArgumentException(
                    "input length " + input.length + " does not match shape (expected " + expected + ")");
        }
        return nativeRun(input, inShape);
    }

    /** Shape of the most recent run's output tensor. */
    public synchronized long[] outputShape() {
        return outputShape.clone();
    }

    @Override
    public synchronized void close() {
        if (!closed && sessionAddr != 0) {
            nativeClose();
        }
        closed = true;
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("model is closed");
        }
    }

    private native void nativeLoadModel(String exportDir, String[] tags);
    private native float[] nativeRun(float[] data, long[] inShape);
    private native void nativeClose();
}
