package com.example.tfcapi;

/**
 * Gated Recurrent Unit over a rank-3 sequence flow {@code [batch, T, D]}:
 * <pre>
 *   z[t] = sigmoid(x·Wz + h'·Uz + bz)          update gate
 *   r[t] = sigmoid(x·Wr + h'·Ur + br)          reset gate
 *   c[t] = tanh(x·Wc + (r ⊙ h')·Uc + bc)       candidate
 *   h[t] = (1 - z) ⊙ c + z ⊙ h'
 * </pre>
 * Unrolled over the fixed T of {@link Input3D}; BPTT is TF autodiff across
 * the unroll. Output: {@code [batch, T*H]}.
 */
public final class GRU extends Layer {

    private final int hidden;

    public GRU(int hidden) {
        require(hidden > 0, "hidden must be positive");
        this.hidden = hidden;
    }

    public int hidden() { return hidden; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 3,
                "GRU needs an Input3D(T, D) flow (got " + in.shape + ")");
        int t = (int) in.shape.dims[1];
        int d = (int) in.shape.last();
        int h = hidden;

        long wz = gb.glorotParam("gru_wz", new long[]{d, h}, d, h);
        long uz = gb.glorotParam("gru_uz", new long[]{h, h}, h, h);
        long wr = gb.glorotParam("gru_wr", new long[]{d, h}, d, h);
        long ur = gb.glorotParam("gru_ur", new long[]{h, h}, h, h);
        long wc = gb.glorotParam("gru_wc", new long[]{d, h}, d, h);
        long uc = gb.glorotParam("gru_uc", new long[]{h, h}, h, h);
        long bz = gb.zerosParam("gru_bz", new long[]{h});
        long br = gb.zerosParam("gru_br", new long[]{h});
        long bc = gb.zerosParam("gru_bc", new long[]{h});

        // zero state [B,H]
        long hp = RNN.zeroState(gb, in.op, d, h);

        long[] steps = new long[t];
        for (int s = 0; s < t; s++) {
            long xt = RNN.timeSlice(gb, in.op, s, d);

            long z = gb.unary("Sigmoid", "gru_z",
                    affine(gb, xt, hp, wz, uz, bz));
            long r = gb.unary("Sigmoid", "gru_r",
                    affine(gb, xt, hp, wr, ur, br));
            long rh = gb.binary("Mul", "gru_rh", r, hp);      // r ⊙ h'
            long cand = gb.unary("Tanh", "gru_cand",
                    affine(gb, xt, rh, wc, uc, bc));

            steps[s] = gb.binary("AddV2", "gru_h",
                    gb.binary("Mul", "gru_new",
                            gb.binary("Sub", "gru_one_z",
                                    gb.constF32("gru_one", 1f), z), cand),
                    gb.binary("Mul", "gru_old", z, hp));
            hp = steps[s];
        }
        return new Flow(RNN.mergeSteps(gb, steps, t * h),
                Shape.of(-1, t * h));
    }

    /** x·W + s·U + b for a gate. */
    private static long affine(GraphBuilder gb, long x, long state,
                               long w, long u, long b) {
        return gb.binary("Add", "gru_aff_b",
                gb.binary("Add", "gru_aff_a",
                        gb.binary("MatMul", "gru_mm_x", x, w),
                        gb.binary("MatMul", "gru_mm_s", state, u)),
                b);
    }

    @Override
    public int typeCode() { return T_GRU; }

    @Override
    int[] specArgs() { return new int[]{hidden}; }

    @Override
    public String toString() { return "GRU(H=" + hidden + ")"; }
}
