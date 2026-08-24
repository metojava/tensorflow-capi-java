package com.example.tfcapi;

/**
 * Multi-head scaled dot-product self-attention over {@code [batch, T, D]}
 * with an optional causal mask:
 * <pre>
 *   per head h (dh = D/heads):
 *     scores_h = Q_h&middot;K_h&#7481;/&radic;dh          [B, T, T]
 *     (optional) scores += causalMask      upper triangle -1e9
 *     ctx_h    = softmax(scores_h)&middot;V_h    [B, T, dh]
 *   concat heads --&gt; WO --&gt; + residual     [B, T, D]
 * </pre>
 * Heads are realized as column slices of the shared {@code [D,D]} Q/K/V
 * projections, so everything stays rank-3 (no rank-4 reshapes). With
 * {@code causal=true}, position t may only attend to positions &le; t -
 * the decoder-style mask for generation-flavored models.
 */
public final class MultiHeadAttention extends Layer {

    private final int heads;
    private final boolean causal;

    public MultiHeadAttention(int heads) {
        this(heads, false);
    }

    public MultiHeadAttention(int heads, boolean causal) {
        require(heads > 0, "heads must be positive");
        this.heads = heads;
        this.causal = causal;
    }

    public int heads() { return heads; }
    public boolean causal() { return causal; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 3,
                "MultiHeadAttention needs an Input3D(T, D) flow (got "
                        + in.shape + ")");
        int t = (int) in.shape.dims[1];
        int d = (int) in.shape.last();
        Layer.require(d % heads == 0,
                "D=" + d + " must divide evenly by heads=" + heads);
        int dh = d / heads;

        long wq = gb.glorotParam("mha_wq", new long[]{d, d}, d, d);
        long wk = gb.glorotParam("mha_wk", new long[]{d, d}, d, d);
        long wv = gb.glorotParam("mha_wv", new long[]{d, d}, d, d);
        long wo = gb.glorotParam("mha_wo", new long[]{d, d}, d, d);

        long q = gb.binary("BatchMatMulV2", "mha_q", in.op, wq);
        long k = gb.binary("BatchMatMulV2", "mha_k", in.op, wk);
        long v = gb.binary("BatchMatMulV2", "mha_v", in.op, wv);

        long scale = gb.constF32("mha_scale", 1f / (float) Math.sqrt(dh));

        // Per-head context, projected straight back to [B,T,D] by its own
        // slice of WO (concat . WO == sum of per-head slices), then summed.
        // Avoids ConcatV2, whose list inputs the JNI binding cannot express.
        long sum = 0;
        boolean firstHead = true;
        for (int h = 0; h < heads; h++) {
            long qh = sliceLast(gb, q, h * dh, dh);
            long kh = sliceLast(gb, k, h * dh, dh);
            long vh = sliceLast(gb, v, h * dh, dh);

            long kt = gb.graph.op("Transpose", gb.unique("mha_t"))
                    .attrType("T", NativeTF.DT_FLOAT)
                    .input(kh)
                    .input(gb.constInt32Tensor("mha_perm",
                            new int[]{0, 2, 1}, new long[]{3}))
                    .build();

            long scores = gb.binary("BatchMatMulV2", "mha_scores", qh, kt);
            scores = gb.binary("Mul", "mha_scaled", scores, scale);
            if (causal) {
                float[] m = new float[t * t];
                for (int i = 0; i < t; i++)
                    for (int j = i + 1; j < t; j++)
                        m[i * t + j] = -1e9f;
                scores = gb.binary("AddV2", "mha_causal", scores,
                        gb.constFloat32Tensor("mha_mask",
                                m, new long[]{t, t}));
            }
            long weights = gb.unary("Softmax", "mha_w", scores);
            long ctx = gb.binary("BatchMatMulV2", "mha_ctx", weights, vh);

            long woRows = gb.graph.op("Slice", gb.unique("mha_wos"))
                    .attrType("T", NativeTF.DT_FLOAT)
                    .attrType("Index", NativeTF.DT_INT32)
                    .input(wo)
                    .input(gb.constI32Vec("mha_wobegin", 0, h * dh))
                    .input(gb.constI32Vec("mha_wosize", -1, dh))
                    .build();
            // lift to rank-3 [1,dh,D] so BatchMatMulV2 broadcasts over batch
            long woRows3 = gb.graph.op("Reshape", gb.unique("mha_wor"))
                    .attrType("T", NativeTF.DT_FLOAT)
                    .attrType("Tshape", NativeTF.DT_INT32)
                    .input(woRows)
                    .input(gb.constInt32Tensor("mha_worshape",
                            new int[]{1, dh, d}, new long[]{3}))
                    .build();

            long proj = gb.binary("BatchMatMulV2", "mha_proj", ctx, woRows3);
            sum = firstHead ? proj
                    : gb.binary("AddV2", "mha_sum", sum, proj);
            firstHead = false;
        }

        long result = gb.binary("AddV2", "mha_out", sum, in.op);   // residual
        return new Flow(result, in.shape);
    }

    private static long sliceLast(GraphBuilder gb, long x, int off, int len) {
        return gb.graph.op("Slice", gb.unique("mha_slice"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Index", NativeTF.DT_INT32)
                .input(x)
                .input(gb.constI32Vec("mha_begin", 0, 0, off))
                .input(gb.constI32Vec("mha_size", -1, -1, len))
                .build();
    }

    @Override
    public int typeCode() { return T_MULTIHEAD; }

    @Override
    int[] specArgs() { return new int[]{heads, causal ? 1 : 0}; }

    @Override
    public String toString() {
        return "MultiHeadAttention(h=" + heads
                + (causal ? ", causal)" : ")");
    }
}
