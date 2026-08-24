package com.example.tfcapi;

/**
 * Batch normalization for 4-D NCHW flows {@code [batch, C, H, W]}: one mean/
 * variance pair per channel, computed over the batch and both spatial axes.
 * Otherwise identical to {@link BatchNorm1D} - placeholder-fed statistics,
 * Java-side running averages (momentum 0.9) fed back at inference time.
 *
 * <p>Insert directly after a {@link Conv2D}/{@link DepthwiseConv2D} layer,
 * before the activation.</p>
 */
public final class BatchNorm2D extends Layer {

    private static final float EPS = 1e-5f;
    private static final float MOMENTUM = 0.9f;

    private float[] runningMean = new float[0];
    private float[] runningVar = new float[0];
    private boolean initialized;

    public BatchNorm2D() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 4,
                "BatchNorm2D needs a 4-D NCHW flow (got " + in.shape + ")");
        int c = (int) in.shape.dims[1];
        long[] cs = new long[]{c};

        long gamma = gb.onesParam("bn2_gamma", cs);
        long beta = gb.zerosParam("bn2_beta", cs);
        long muPh = aux(gb, "bn2_mu", cs);
        long varPh = aux(gb, "bn2_var", cs);

        // broadcast helpers: [C] -> [1,C,1,1]
        long shape4 = gb.constInt32Tensor("bn2_bshape", new int[]{1, c, 1, 1}, new long[]{4});
        long muB = reshape(gb, muPh, shape4);
        long varB = reshape(gb, varPh, shape4);
        long gammaB = reshape(gb, gamma, shape4);
        long betaB = reshape(gb, beta, shape4);

        // xhat = (x - mu) * rsqrt(var + eps)
        long centered = gb.binary("Sub", "bn2_sub", in.op, muB);
        long varEps = gb.binary("AddV2", "bn2_ve", varB, gb.constF32("bn2_eps", EPS));
        long invStd = gb.unary("Rsqrt", "bn2_rsqrt", varEps);
        long xhat = gb.binary("Mul", "bn2_norm", centered, invStd);
        long out = gb.binary("AddV2", "bn2_out",
                gb.binary("Mul", "bn2_scale", gammaB, xhat), betaB);
        return new Flow(out, in.shape);
    }

    private static long reshape(GraphBuilder gb, long t, long shapeConst) {
        return gb.graph.op("Reshape", gb.unique("bn2_rs"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tshape", NativeTF.DT_INT32)
                .input(t)
                .input(shapeConst)
                .build();
    }

    @Override
    void supplyAux(float[][] buffers, int n, boolean training, float[] batchX) {
        float[] mu = buffers[0];
        float[] var = buffers[1];
        int c = mu.length;
        ensureRunning(c);
        int spatial = batchX != null ? (batchX.length / (n * c)) : 0;

        if (training && batchX != null && spatial > 0) {
            computeStats(batchX, n, c, spatial, mu, var);
            for (int j = 0; j < c; j++) {
                runningMean[j] = MOMENTUM * runningMean[j] + (1 - MOMENTUM) * mu[j];
                runningVar[j] = MOMENTUM * runningVar[j] + (1 - MOMENTUM) * var[j];
            }
            initialized = true;
        } else {
            for (int j = 0; j < c; j++) {
                mu[j] = initialized ? runningMean[j] : 0f;
                var[j] = initialized ? runningVar[j] : 1f;
            }
        }
    }

    private void ensureRunning(int c) {
        if (runningMean.length != c) {
            runningMean = new float[c];
            runningVar = new float[c];
            java.util.Arrays.fill(runningVar, 1f);
        }
    }

    /** Channel means/variances over batch and spatial axes of NCHW data. */
    private static void computeStats(float[] x, int n, int c, int spatial,
                                     float[] mu, float[] var) {
        int plane = c * spatial;
        java.util.Arrays.fill(mu, 0f);
        java.util.Arrays.fill(var, 0f);
        for (int i = 0; i < n; i++) {
            int base = i * plane;
            for (int j = 0; j < c; j++) {
                int off = base + j * spatial;
                for (int s = 0; s < spatial; s++) {
                    mu[j] += x[off + s];
                }
            }
        }
        for (int j = 0; j < c; j++) {
            mu[j] /= n * spatial;
        }
        for (int i = 0; i < n; i++) {
            int base = i * plane;
            for (int j = 0; j < c; j++) {
                int off = base + j * spatial;
                for (int s = 0; s < spatial; s++) {
                    float d = x[off + s] - mu[j];
                    var[j] += d * d;
                }
            }
        }
        for (int j = 0; j < c; j++) {
            var[j] /= (float) n * spatial;
        }
    }

    @Override
    public int typeCode() { return T_BATCHNORM2D; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "BatchNorm2D()"; }
}
