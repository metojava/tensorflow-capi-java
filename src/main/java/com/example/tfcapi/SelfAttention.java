package com.example.tfcapi;

/**
 * Single-head scaled dot-product self-attention over a rank-3 sequence flow
 * {@code [batch, T, D]}:
 * <pre>
 *   scores  = Q&middot;K&#7481; / &radic;D          [B, T, T]
 *   weights = softmax(scores)     over the last axis
 *   output  = weights&middot;V + x      (residual), shape preserved [B, T, D]
 * </pre>
 * with trainable projections {@code Wq/Wk/Wv} ({@code [D,D]} each,
 * Glorot-uniform). Built entirely from gradient-registered TF kernels
 * ({@code Transpose}, {@code BatchMatMul}, {@code Softmax}, {@code Mul},
 * {@code AddV2}), so backprop works through the generic
 * {@code TF_AddGradients} path with no native changes.
 *
 * <p>Placement: on any rank-3 flow - typically {@code Input3D(T, D)} one-hot
 * planes BEFORE a recurrent layer (recurrent layers flatten their output to
 * {@code [B, T*H]}, and attention needs the sequence structure intact).
 * Stacking several instances deepens the pairwise interactions.</p>
 */
public final class SelfAttention extends Layer {

    public SelfAttention() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 3,
                "SelfAttention needs an Input3D(T, D) flow (got "
                        + in.shape + ")");
        int d = (int) in.shape.last();
        float scale = 1f / (float) Math.sqrt(d);

        long wq = gb.glorotParam("at_wq", new long[]{d, d}, d, d);
        long wk = gb.glorotParam("at_wk", new long[]{d, d}, d, d);
        long wv = gb.glorotParam("at_wv", new long[]{d, d}, d, d);

        long q = gb.binary("BatchMatMulV2", "at_q", in.op, wq);   // [B,T,D]
        long k = gb.binary("BatchMatMulV2", "at_k", in.op, wk);   // [B,T,D]
        long v = gb.binary("BatchMatMulV2", "at_v", in.op, wv);   // [B,T,D]

        // K^T: [B,D,T] via transpose perm {0,2,1}
        long perm = gb.constInt32Tensor("at_perm",
                new int[]{0, 2, 1}, new long[]{3});
        long kt = gb.graph.op("Transpose", gb.unique("at_t"))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(k)
                .input(perm)
                .build();

        long scores = gb.binary("BatchMatMulV2", "at_scores", q, kt); // [B,T,T]
        long scaled = gb.binary("Mul", "at_scale", scores,
                gb.constF32("at_inv_sqrt_d", scale));
        long weights = gb.unary("Softmax", "at_weights", scaled);   // rows sum to 1

        long ctx = gb.binary("BatchMatMulV2", "at_ctx", weights, v);  // [B,T,D]
        long out = gb.binary("AddV2", "at_out", ctx, in.op);        // residual

        return new Flow(out, in.shape);
    }

    @Override
    public int typeCode() { return T_SELF_ATTENTION; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "SelfAttention()"; }
}
