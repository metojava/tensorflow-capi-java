package com.example.tfcapi;

/**
 * Strategy for turning gradients into parameter updates. An optimizer is
 * bound to a model's parameters via {@link #init} and stepped once per
 * batch by {@code Sequential.trainBatch}. State lives in native tensors
 * &mdash; nothing is copied into Java.
 *
 * <p>Use the factories {@link #sgd}, {@link #momentum} and {@link #adam},
 * or extend this class with your own update rule using the generic
 * tensor kernels in {@code NativeTF}.</p>
 */
public abstract class Optimizer implements AutoCloseable {

    long[][] shapes;

    /** Fraction of the configured learning rate (LR-schedule hook). */
    private float lrMultiplier = 1f;
    /** Decoupled weight decay: grad += decay * param before the update rule. */
    private float weightDecay;
    /** Global-norm gradient clipping threshold; 0 disables. */
    private float clipNorm;

    /**
     * Entry point used by models. Applies optional gradient clipping and
     * decoupled weight decay, then delegates to the concrete rule.
     */
    public final void apply(long[] params, long[] grads) {
        long[] g = grads;
        long[] clipCopy = null;
        long[] decayCopy = null;
        try {
            if (clipNorm > 0) {
                float norm = globalNorm(grads);
                if (norm > clipNorm) {
                    float scale = clipNorm / (norm + 1e-6f);
                    clipCopy = scaledCopy(grads, scale);
                    g = clipCopy;
                }
            }
            if (weightDecay > 0) {
                decayCopy = addWeightDecay(params, g);
                g = decayCopy;
            }
            step(params, g);
        } finally {
            deleteAll(clipCopy);
            deleteAll(decayCopy);
        }
    }

    private float globalNorm(long[] grads) {
        double sum = 0;
        for (long grad : grads) {
            for (float v : NativeTF.tensorDataF32(grad)) {
                sum += (double) v * v;
            }
        }
        return (float) Math.sqrt(sum);
    }

    private long[] scaledCopy(long[] grads, float scale) {
        long[] out = new long[grads.length];
        for (int i = 0; i < grads.length; i++) {
            float[] data = NativeTF.tensorDataF32(grads[i]);
            for (int j = 0; j < data.length; j++) {
                data[j] *= scale;
            }
            out[i] = NativeTF.allocTensorF32(shapes[i], data);
        }
        return out;
    }

    private long[] addWeightDecay(long[] params, long[] grads) {
        long[] out = new long[grads.length];
        for (int i = 0; i < grads.length; i++) {
            float[] p = NativeTF.tensorDataF32(params[i]);
            float[] g = NativeTF.tensorDataF32(grads[i]);
            for (int j = 0; j < g.length; j++) {
                g[j] += weightDecay * p[j];
            }
            out[i] = NativeTF.allocTensorF32(shapes[i], g);
        }
        return out;
    }

    private static void deleteAll(long[] handles) {
        if (handles == null) return;
        for (long h : handles) NativeTF.deleteTensor(h);
    }

    /** Sets the fraction of the configured learning rate used from now on. */
    public synchronized void setLrMultiplier(float m) {
        if (m <= 0 || m > 1.5f) {
            throw new IllegalArgumentException("lr multiplier out of range: " + m);
        }
        lrMultiplier = m;
    }

    public synchronized float lrMultiplier() { return lrMultiplier; }

    /** Decoupled weight decay (AdamW-style); 0 disables. */
    public void setWeightDecay(float decay) { this.weightDecay = Math.max(0f, decay); }

    /** Global-norm gradient clipping; e.g. 5.0. 0 disables. */
    public void setGradClipNorm(float maxNorm) { this.clipNorm = Math.max(0f, maxNorm); }

    /** Effective per-step learning rate for subclasses. */
    protected final float effLr(float configuredLr) { return configuredLr * lrMultiplier; }

    /** Binds this optimizer to parameters of the given shapes (zeros state). */
    public void init(long[][] paramShapes) {
        close();
        this.shapes = new long[paramShapes.length][];
        for (int i = 0; i < paramShapes.length; i++) {
            this.shapes[i] = paramShapes[i].clone();
        }
        initState();
    }

    /** Allocates optimizer state; called after {@link #init}. */
    protected abstract void initState();

    /**
     * Applies one update. Called from a synchronized context.
     *
     * @param params current parameter tensors (may be replaced between steps)
     * @param grads  gradient tensors from the last training step
     */
    protected abstract void step(long[] params, long[] grads);

    long zeros(long[] shape) {
        int len = 1;
        for (long d : shape) len *= d;
        return NativeTF.allocTensorF32(shape, new float[len]);
    }

    static long product(long[] shape) {
        long p = 1;
        for (long d : shape) p *= d;
        return p;
    }

    @Override
    public abstract void close();

    /* ----------------------------- factories ------------------------------ */

    public static Sgd sgd(float learningRate) {
        return new Sgd(learningRate);
    }

    public static Momentum momentum(float learningRate, float momentum) {
        return new Momentum(learningRate, momentum);
    }

    public static Adam adam(float learningRate) {
        return new Adam(learningRate);
    }
}
