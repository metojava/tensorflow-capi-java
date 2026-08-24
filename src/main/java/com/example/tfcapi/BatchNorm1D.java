package com.example.tfcapi;

/**
 * Batch normalization over the batch axis for 2-D flows:
 * {@code y = gamma · (x - mu) / sqrt(var + eps) + beta}.
 *
 * <p>Statistics are placeholder-fed auxiliary tensors, so one graph serves
 * both modes: during {@code fit()} each batch feeds its own mean/variance
 * (and the Java-side running averages are updated with momentum 0.9);
 * {@code predict()} feeds the running averages. Gradients w.r.t. gamma/beta
 * and the input flow through TF's autodiff; the statistics themselves are
 * treated as constants (the standard frozen-stats approximation).</p>
 *
 * <p>Running averages live in Java memory only — they are re-estimated on
 * every {@code fit()} and not written to checkpoints.</p>
 */
public final class BatchNorm1D extends Layer {

    private static final float EPS = 1e-5f;
    private static final float MOMENTUM = 0.9f;

    private float[] runningMean = new float[0]; // resized lazily
    private float[] runningVar = new float[0];
    private boolean initialized;

    public BatchNorm1D() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 2,
                "BatchNorm1D needs a 2-D flow (got " + in.shape + ")");
        int f = (int) in.shape.last();
        long[] fs = new long[]{f};

        long gamma = gb.onesParam("bn_gamma", fs);
        long beta = gb.zerosParam("bn_beta", fs);
        long muPh = aux(gb, "bn_mu", fs);
        long varPh = aux(gb, "bn_var", fs);

        // xhat = (x - mu) * rsqrt(var + eps)
        long centered = gb.binary("Sub", "bn_sub", in.op, muPh);
        long varEps = gb.binary("AddV2", "bn_ve", varPh,
                gb.constF32("bn_eps", EPS));
        long invStd = gb.unary("Rsqrt", "bn_rsqrt", varEps);
        long xhat = gb.binary("Mul", "bn_norm", centered, invStd);
        long out = gb.binary("AddV2", "bn_out",
                gb.binary("Mul", "bn_scale", gamma, xhat), beta);
        return new Flow(out, in.shape);
    }

    @Override
    void supplyAux(float[][] buffers, int n, boolean training, float[] batchX) {
        float[] mu = buffers[0];
        float[] var = buffers[1];
        int f = mu.length;
        ensureRunning(f);

        if (training && batchX != null && batchX.length >= n * f) {
            computeStats(batchX, n, f, mu, var);
            for (int j = 0; j < f; j++) {
                runningMean[j] = MOMENTUM * runningMean[j] + (1 - MOMENTUM) * mu[j];
                runningVar[j] = MOMENTUM * runningVar[j] + (1 - MOMENTUM) * var[j];
            }
            initialized = true;
        } else {
            for (int j = 0; j < f; j++) {
                mu[j] = initialized ? runningMean[j] : 0f;
                var[j] = initialized ? runningVar[j] : 1f;
            }
        }
    }

    private void ensureRunning(int f) {
        if (runningMean.length != f) {
            runningMean = new float[f];   // mean starts at 0
            runningVar = new float[f];    // variance starts at 1
            java.util.Arrays.fill(runningVar, 1f);
        }
    }

    private static void computeStats(float[] x, int n, int f, float[] mu, float[] var) {
        java.util.Arrays.fill(mu, 0f);
        java.util.Arrays.fill(var, 0f);
        for (int i = 0; i < n; i++) {
            int off = i * f;
            for (int j = 0; j < f; j++) {
                mu[j] += x[off + j];
            }
        }
        for (int j = 0; j < f; j++) {
            mu[j] /= n;
        }
        for (int i = 0; i < n; i++) {
            int off = i * f;
            for (int j = 0; j < f; j++) {
                float d = x[off + j] - mu[j];
                var[j] += d * d;
            }
        }
        for (int j = 0; j < f; j++) {
            var[j] /= n;
        }
    }

    @Override
    public int typeCode() { return T_BATCHNORM; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "BatchNorm1D()"; }
}
