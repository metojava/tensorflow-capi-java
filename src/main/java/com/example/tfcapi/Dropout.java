package com.example.tfcapi;

/**
 * Inverted dropout on 2-D flows. The binary mask is a placeholder-fed
 * auxiliary tensor: training feeds a fresh random mask scaled by
 * {@code 1/(1-p)} (so the expected activation is unchanged), inference feeds
 * all ones — the same graph serves both modes.
 *
 * <p>Mode control: {@code fit()} always trains with dropout active;
 * {@code predict()} uses inference mode unless
 * {@link Sequential#setTraining(boolean)} was called with {@code true}.
 */
public final class Dropout extends Layer {

    private final float p;

    public Dropout() { this(0.5f); }

    public Dropout(float p) {
        require(p >= 0f && p < 1f, "p must be in [0,1)");
        this.p = p;
    }

    public float p() { return p; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 2,
                "Dropout needs a 2-D flow (got " + in.shape + ")");
        long maskPh = aux(gb, "dropout_mask", in.shape.dims);
        long out = gb.binary("Mul", "dropout_mul", in.op, maskPh);
        return new Flow(out, in.shape);
    }

    @Override
    void supplyAux(float[][] buffers, int n, boolean training, float[] batchX) {
        float[] mask = buffers[0];
        if (!training || p == 0f) {
            java.util.Arrays.fill(mask, 1f);
            return;
        }
        float keep = 1f / (1f - p);
        java.util.concurrent.ThreadLocalRandom rnd =
                java.util.concurrent.ThreadLocalRandom.current();
        for (int i = 0; i < mask.length; i++) {
            mask[i] = rnd.nextFloat() < p ? 0f : keep;
        }
    }

    @Override
    public int typeCode() { return T_DROPOUT; }

    @Override
    int[] specArgs() { return new int[]{Float.floatToIntBits(p)}; }

    @Override
    public String toString() { return "Dropout(p=" + p + ")"; }
}
