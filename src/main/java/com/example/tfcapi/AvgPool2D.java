package com.example.tfcapi;

/** 2-D average pooling (NHWC) with bias. */
public final class AvgPool2D extends Layer {

    private final int ph, pw, sh, sw;
    private final Padding padding;

    public AvgPool2D(int poolH, int poolW, int strideH, int strideW, Padding padding) {
        require(poolH > 0 && poolW > 0 && strideH > 0 && strideW > 0 && padding != null,
                "AvgPool2D arguments must be positive and padding set");
        this.ph = poolH;
        this.pw = poolW;
        this.sh = strideH;
        this.sw = strideW;
        this.padding = padding;
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 4,
                "AvgPool2D needs 4-D NHWC input; start the model with Input4D (got " + in.shape + ")");
        int cin = (int) in.shape.dims[3];
        long out = gb.graph.op("AvgPool", gb.unique("avgpool"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrIntList("ksize", 1, ph, pw, 1)
                .attrIntList("strides", 1, sh, sw, 1)
                .attrString("padding", padding.tfName)
                .input(in.op)
                .build();
        return new Flow(out, Shape.of(-1,
                outDim(in.shape.dims[1], ph, sh), outDim(in.shape.dims[2], pw, sw),
                cin));
    }

    private long outDim(long inDim, int k, int s) {
        if (padding == Padding.SAME) {
            return (inDim + s - 1) / s;
        }
        return (inDim - k) / s + 1;
    }

    @Override
    public int typeCode() { return T_AVGPOOL; }

    @Override
    int[] specArgs() {
        return new int[]{ph, pw, sh, sw, padding.code};
    }

    @Override
    public String toString() {
        return "AvgPool2D(" + ph + "x" + pw + ",s" + sh + "x" + sw + "," + padding + ")";
    }
}