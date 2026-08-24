package com.example.tfcapi;

/**
 * Long Short-Term Memory layer over a rank-3 sequence flow
 * {@code [batch, T, D]}:
 * <pre>
 *   a = x·W + h'·U + b            (W: [D,4H], U: [H,4H], b: [4H])
 *   f,i,o = sigmoid(quarters)     g = tanh(quarter)
 *   c = f ⊙ c' + i ⊙ g            cell state
 *   h = o ⊙ tanh(c)               exposed state
 * </pre>
 * One big 4H matmul per step; the four gate pre-activations are carved out
 * with {@code Slice}. Unrolled over the fixed T of {@link Input3D}; BPTT is
 * TF autodiff across the unroll. Output: {@code [batch, T*H]}.
 */
public final class LSTM extends Layer {

    private final int hidden;

    public LSTM(int hidden) {
        require(hidden > 0, "hidden must be positive");
        this.hidden = hidden;
    }

    public int hidden() { return hidden; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 3,
                "LSTM needs an Input3D(T, D) flow (got " + in.shape + ")");
        int t = (int) in.shape.dims[1];
        int d = (int) in.shape.last();
        int h = hidden;

        long w = gb.glorotParam("lstm_w", new long[]{d, 4 * h}, d, 4 * h);
        long u = gb.glorotParam("lstm_u", new long[]{h, 4 * h}, h, 4 * h);
        long b = gb.zerosParam("lstm_b", new long[]{4 * h});

        // zero states [B,H] for h and c
        long hp = RNN.zeroState(gb, in.op, d, h);
        long cp = hp;

        long[] steps = new long[t];
        for (int s = 0; s < t; s++) {
            long xt = RNN.timeSlice(gb, in.op, s, d);

            // a = x·W + h'·U + b  -> [B, 4H]
            long a = gb.binary("Add", "lstm_a_b",
                    gb.binary("Add", "lstm_a",
                            gb.binary("MatMul", "lstm_mm_x", xt, w),
                            gb.binary("MatMul", "lstm_mm_h", hp, u)),
                    b);

            long f = gb.unary("Sigmoid", "lstm_f", gate(gb, a, 0, h));
            long i = gb.unary("Sigmoid", "lstm_i", gate(gb, a, 1, h));
            long o = gb.unary("Sigmoid", "lstm_o", gate(gb, a, 2, h));
            long g = gb.unary("Tanh", "lstm_g", gate(gb, a, 3, h));

            // c = f ⊙ c' + i ⊙ g
            cp = gb.binary("AddV2", "lstm_c",
                    gb.binary("Mul", "lstm_keep", f, cp),
                    gb.binary("Mul", "lstm_write", i, g));
            steps[s] = gb.binary("Mul", "lstm_h", o,
                    gb.unary("Tanh", "lstm_tanhc", cp));
            hp = steps[s];
        }
        return new Flow(RNN.mergeSteps(gb, steps, t * h),
                Shape.of(-1, t * h));
    }

    /** Column block q of the stacked gate pre-activations [B, H]. */
    private static long gate(GraphBuilder gb, long a, int q, int h) {
        return gb.graph.op("Slice", gb.unique("lstm_gate" + q))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Index", NativeTF.DT_INT32)
                .input(a)
                .input(gb.constI32Vec("gate_begin", 0, q * h))
                .input(gb.constI32Vec("gate_size", -1, h))
                .build();
    }

    @Override
    public int typeCode() { return T_LSTM; }

    @Override
    int[] specArgs() { return new int[]{hidden}; }

    @Override
    public String toString() { return "LSTM(H=" + hidden + ")"; }
}
