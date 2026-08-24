package com.example.tfcapi;

/**
 * Vanilla recurrent layer over a rank-3 sequence flow {@code [batch, T, D]}:
 * <pre>
 *   h[t] = tanh(x[t]·Wx + h[t-1]·Wh + b),   h[-1] = 0
 * </pre>
 * The graph is unrolled over the fixed {@code T} declared by
 * {@link Input3D}; backpropagation through time is TensorFlow's autodiff
 * across that unroll — no manual BPTT code. Output: {@code [batch, T*H]}
 * (all hidden states concatenated, last timestep last).
 */
public final class RNN extends Layer {

    private final int hidden;

    public RNN(int hidden) {
        require(hidden > 0, "hidden must be positive");
        this.hidden = hidden;
    }

    public int hidden() { return hidden; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 3,
                "RNN needs an Input3D(T, D) flow (got " + in.shape + ")");
        int t = (int) in.shape.dims[1];
        int d = (int) in.shape.last();

        long wx = gb.glorotParam("rnn_wx", new long[]{d, hidden}, d, hidden);
        long wh = gb.glorotParam("rnn_wh", new long[]{hidden, hidden},
                hidden, hidden);
        long b = gb.zerosParam("rnn_b", new long[]{hidden});

        // zero state [B,H]: slice of x times 0, tiled to H columns
        long zeroCol = gb.binary("Mul", "rnn_zero",
                timeSlice(gb, in.op, 0, d), gb.constF32("rnn_zero_f", 0f));
        long hPrev = gb.graph.op("Tile", gb.unique("rnn_tile"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tmultiples", NativeTF.DT_INT32)
                .input(zeroCol)
                .input(gb.constI32Vec("rnn_mult", 1, hidden))
                .build();

        long[] steps = new long[t];
        for (int s = 0; s < t; s++) {
            long xt = timeSlice(gb, in.op, s, d);
            long a = gb.binary("Add", "rnn_a1",
                    gb.binary("MatMul", "rnn_mm_x", xt, wx),
                    gb.binary("MatMul", "rnn_mm_h", hPrev, wh));
            steps[s] = gb.unary("Tanh", "rnn_tanh",
                    gb.binary("Add", "rnn_a2", a, b));
            hPrev = steps[s];
        }
        return new Flow(mergeSteps(gb, steps, t * hidden),
                Shape.of(-1, t * hidden));
    }

    /** Extracts timestep {@code s} as a 2-D [B, D] tensor. */
    static long timeSlice(GraphBuilder gb, long x, int s, int d) {
        // end_mask bit 0 keeps the FULL batch range; without it end[0]=0
        // slices an empty [0,...] batch dimension.
        long sliced = gb.graph.op("StridedSlice", gb.unique("seq_slice"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Index", NativeTF.DT_INT32)
                .attrInt("end_mask", 1)
                .input(x)
                .input(gb.constI32Vec("ss_begin", 0, s, 0))
                .input(gb.constI32Vec("ss_end", 0, s + 1, d))
                .input(gb.constI32Vec("ss_strides", 1, 1, 1))
                .build();
        // Squeeze ONLY the window axis: a bare Squeeze would also drop the
        // batch (and feature) dims whenever they equal 1, producing a rank-1
        // tensor that downstream MatMuls reject.
        return gb.graph.op("Squeeze", gb.unique("seq_squeeze"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrIntList("squeeze_dims", 1)
                .input(sliced)
                .build();                            // [B,1,D] -> [B,D]
    }

    /** Initial recurrent state: a zero [B, H] tensor derived from the input. */
    static long zeroState(GraphBuilder gb, long x, int d, int hidden) {
        float[] ones = new float[d * hidden];
        java.util.Arrays.fill(ones, 1f);
        return gb.binary("Mul", "seq_zero",
                gb.binary("MatMul", "seq_zero_mm", timeSlice(gb, x, 0, d),
                        gb.constFloat32Tensor("seq_zero_ones", ones,
                                new long[]{d, hidden})),
                gb.constF32("seq_zero_f", 0f));      // [B,H]
    }

    /**
     * Concatenates per-timestep [B,H] tensors into [B, T*H] where
     * {@code outCols = T*H}.
     *
     * Implemented as {@code sum_s h_s @ E_s} with constant one-hot-block
     * selection matrices E_s instead of Pack+Reshape: the TF C API only
     * accepts list inputs through TF_AddInputList, so feeding Pack via
     * repeated TF_AddInput fails NodeDef validation ("Single tensor passed
     * to 'values', expected list"). The MatMul/Add form is mathematically
     * identical and fully differentiable.
     */
    static long mergeSteps(GraphBuilder gb, long[] steps, int outCols) {
        int t = steps.length;
        int h = outCols / t;
        long acc = 0;
        for (int s = 0; s < t; s++) {
            float[] e = new float[h * outCols];
            for (int r = 0; r < h; r++) e[r * outCols + s * h + r] = 1f;
            long sel = gb.constFloat32Tensor("seq_sel", e, new long[]{h, outCols});
            long term = gb.binary("MatMul", "seq_mm", steps[s], sel);
            acc = (s == 0) ? term : gb.binary("Add", "seq_add", acc, term);
        }
        return acc;
    }

    @Override
    public int typeCode() { return T_RNN; }

    @Override
    int[] specArgs() { return new int[]{hidden}; }

    @Override
    public String toString() { return "RNN(H=" + hidden + ")"; }
}
