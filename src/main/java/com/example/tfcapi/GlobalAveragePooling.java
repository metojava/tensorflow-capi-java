package com.example.tfcapi;

public final class GlobalAveragePooling extends Layer {

    public GlobalAveragePooling() {
        // Constructor
    }

    @Override
    public int typeCode() {
        return T_GLOBAL_AVERAGE_POOLING;
    }

    @Override
    int[] specArgs() {
        return new int[0];
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() >= 3,
                "GlobalAveragePooling needs rank >= 3, e.g. [B,T,D] -> [B,D] "
                        + "(got " + in.shape + ")");
        // average over all SPATIAL axes (1..rank-2), keep batch and channels:
        // [B,T,D] -> [B,D]
        int r = in.shape.rank();
        int[] axes = new int[r - 2];
        for (int i = 1; i <= r - 2; i++) axes[i - 1] = i;
        long out = gb.graph.op("Mean", gb.unique("gap_mean"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tidx", NativeTF.DT_INT32)
                .input(in.op)
                .input(gb.constInt32Tensor("gap_axes",
                        axes, new long[]{axes.length}))
                .build();
        return new Flow(out, Shape.of(-1, in.shape.last()));
    }

    @Override
    public String toString() {
        return "GlobalAveragePooling()";
    }

    // Define constant for type code
    static final int T_GLOBAL_AVERAGE_POOLING = 23;
}