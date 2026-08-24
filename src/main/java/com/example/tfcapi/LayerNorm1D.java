package com.example.tfcapi;

/**
 * Layer normalization over the FEATURE axis, applied per position:
 * <pre>
 *   xhat = (x - mean) * rsqrt(var + eps)      // mean/var over last axis
 *   out  = gamma &middot; xhat + beta               // gamma, beta: [D], trainable
 * </pre>
 * Unlike {@link BatchNorm1D} (batch-axis statistics, needs train/inference
 * modes), LayerNorm statistics depend only on the current sample's feature
 * vector - identical behavior at train and inference time, which is why it is
 * the normalization of choice inside Transformer blocks.
 */
public final class LayerNorm1D extends Layer {

    private static final float EPS = 1e-5f;

    public LayerNorm1D() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() >= 2,
                "LayerNorm1D needs rank >= 2 input (got " + in.shape + ")");
        int rank = in.shape.rank();
        int d = (int) in.shape.last();

        long gamma = gb.onesParam("ln_gamma", new long[]{d});
        long beta = gb.zerosParam("ln_beta", new long[]{d});

        // reduce over the LAST axis only; axes constant must carry the value
        long axes = gb.constInt32Tensor("ln_axes",
                new int[]{rank - 1}, new long[]{1});
        long mean = reduceMean(gb, in.op, axes);

        // restore reduced dim for broadcasting: [..., 1]
        int[] midShape = new int[rank];
        midShape[0] = -1;
        for (int i = 1; i < rank - 1; i++) midShape[i] = (int) in.shape.dims[i];
        midShape[rank - 1] = 1;
        long midShapeC = gb.constInt32Tensor("ln_mshape", midShape, new long[]{rank});
        long meanB = reshape(gb, mean, midShapeC);

        long diff = gb.binary("Sub", "ln_diff", in.op, meanB);
        long sq = gb.binary("Mul", "ln_sq", diff, diff);
        long var = reduceMean(gb, sq, axes);
        long varB = reshape(gb, var, midShapeC);

        long invStd = gb.unary("Rsqrt", "ln_rsqrt",
                gb.binary("AddV2", "ln_ve", varB, gb.constF32("ln_eps", EPS)));
        long xhat = gb.binary("Mul", "ln_xhat", diff, invStd);

        long out = gb.binary("AddV2", "ln_out",
                gb.binary("Mul", "ln_scale", gamma, xhat), beta);
        return new Flow(out, in.shape);
    }

    private static long reduceMean(GraphBuilder gb, long x, long axes) {
        return gb.graph.op("Mean", gb.unique("ln_mean"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tidx", NativeTF.DT_INT32)
                .input(x)
                .input(axes)
                .build();
    }

    private static long reshape(GraphBuilder gb, long t, long shapeConst) {
        return gb.graph.op("Reshape", gb.unique("ln_rs"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tshape", NativeTF.DT_INT32)
                .input(t)
                .input(shapeConst)
                .build();
    }

    @Override
    public int typeCode() { return T_LAYERNORM; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "LayerNorm1D()"; }
}
