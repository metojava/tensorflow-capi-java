package com.example.tfcapi;

/**
 * Position-wise fully-connected layer for flows of ANY rank (typically the
 * sequence flow {@code [batch, T, D]}):
 * <pre>
 *   out[b, t, :] = x[b, t, :] &middot; W + b      // same W and b at every position
 * </pre>
 * Implemented as reshape-to-2D &rarr; MatMul+bias &rarr; reshape-back, so
 * weights are <em>shared across positions</em> and the sequence structure
 * survives - unlike {@code Flatten -> Dense}, which learns separate weights
 * per position and destroys the T axis. This is the feed-forward sub-layer
 * of Transformer blocks and a convenient per-token projection before or
 * after attention/recurrence.
 */
public final class DenseND extends Layer {

    private final int units;

    public DenseND(int units) {
        require(units > 0, "units must be positive");
        this.units = units;
    }

    public int units() { return units; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() >= 2,
                "DenseND needs rank >= 2 input (got " + in.shape + ")");
        int rank = in.shape.rank();
        int d = (int) in.shape.last();

        // [lead..., D] -> [-1, D]
        long toFlat = gb.constInt32Tensor("densend_flat",
                new int[]{-1, d}, new long[]{2});
        long x2 = gb.graph.op("Reshape", gb.unique("densend_r1"))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(in.op)
                .input(toFlat)
                .build();

        long w = gb.glorotParam("densend_w", new long[]{d, units}, d, units);
        long bias = gb.zerosParam("densend_b", new long[]{units});
        long y2 = gb.binary("Add", "densend_bias",
                gb.binary("MatMul", "densend_mm", x2, w), bias);

        // [-1, units] -> [lead..., units]
        int[] back = new int[rank];
        back[0] = -1;
        for (int i = 1; i < rank - 1; i++) back[i] = (int) in.shape.dims[i];
        back[rank - 1] = units;
        long toOrig = gb.constInt32Tensor("densend_back", back, new long[]{rank});
        long out = gb.graph.op("Reshape", gb.unique("densend_r2"))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(y2)
                .input(toOrig)
                .build();

        long[] restored = in.shape.dims.clone();
        restored[rank - 1] = units;
        return new Flow(out, Shape.of(restored));
    }

    @Override
    public int typeCode() { return T_DENSE_ND; }

    @Override
    int[] specArgs() { return new int[]{units}; }

    @Override
    public String toString() { return "DenseND(" + units + ")"; }
}
