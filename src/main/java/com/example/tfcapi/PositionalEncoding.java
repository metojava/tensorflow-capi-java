package com.example.tfcapi;

/**
 * Sinusoidal positional encoding for sequence flows {@code [batch, T, D]}
 * (parameter-free, added to the input):
 * <pre>
 *   PE[pos, 2i]   = sin(pos / 10000^(2i/D))
 *   PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
 * </pre>
 * Attention is order-blind; this table injects position information so
 * pure-attention stacks can distinguish "early" from "late". T and D are
 * fixed at build time, so the whole table is baked into a constant tensor -
 * zero runtime cost, no trainable parameters, nothing new at checkpoints.
 */
public final class PositionalEncoding extends Layer {

    public PositionalEncoding() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 3,
                "PositionalEncoding needs an Input3D(T, D) flow (got "
                        + in.shape + ")");
        int t = (int) in.shape.dims[1];
        int d = (int) in.shape.last();

        float[] table = new float[t * d];
        for (int pos = 0; pos < t; pos++) {
            for (int i = 0; i < d; i++) {
                int k = i / 2;
                double angle = pos / Math.pow(10000.0, 2.0 * k / d);
                table[pos * d + i] = (float) (i % 2 == 0 ? Math.sin(angle)
                                                         : Math.cos(angle));
            }
        }
        long tab = gb.constFloat32Tensor("pe_table", table, new long[]{t, d});
        long out = gb.binary("AddV2", "pe_add", in.op, tab);
        return new Flow(out, in.shape);
    }

    @Override
    public int typeCode() { return T_POSITIONAL; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "PositionalEncoding()"; }
}
