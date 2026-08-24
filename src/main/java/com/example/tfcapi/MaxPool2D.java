package com.example.tfcapi;

/** Max pooling over NHWC input with kernel {@code [ph, pw]} and stride {@code [sh, sw]}. */
public final class MaxPool2D extends Layer {

    private final int ph, pw, sh, sw;
    private final Padding padding;

    public MaxPool2D(int poolH, int poolW, int strideH, int strideW, Padding padding) {
        require(poolH > 0 && poolW > 0 && strideH > 0 && strideW > 0 && padding != null,
                "MaxPool2D arguments must be positive and padding set");
        this.ph = poolH;
        this.pw = poolW;
        this.sh = strideH;
        this.sw = strideW;
        this.padding = padding;
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 4,
                "MaxPool2D needs 4-D NHWC input (got " + in.shape + ")");
        long out = gb.graph.op("MaxPool", gb.unique("maxpool"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrIntList("ksize", 1, ph, pw, 1)
                .attrIntList("strides", 1, sh, sw, 1)
                .attrString("padding", padding.tfName)
                .input(in.op)
                .build();
        return new Flow(out, Shape.of(-1,
                outDim(in.shape.dims[1], ph, sh), outDim(in.shape.dims[2], pw, sw),
                in.shape.dims[3]));
    }

    private long outDim(long inDim, int k, int s) {
        if (padding == Padding.SAME) {
            return (inDim + s - 1) / s;
        }
        return (inDim - k) / s + 1;
    }

    @Override
    public int typeCode() { return T_MAXPOOL; }

    @Override
    int[] specArgs() {
        return new int[]{ph, pw, sh, sw, padding.code};
    }

    @Override
    public String toString() {
        return "MaxPool2D(" + ph + "x" + pw + ",s" + sh + "x" + sw + "," + padding + ")";
    }
}
