package com.example.tfcapi;

/**
 * 2-D convolution (NHWC) with bias: filter {@code [kh, kw, cin, filters]},
 * stride {@code [sh, sw]}, Glorot-uniform init.
 */
public final class Conv2D extends Layer {

    private final int filters, kh, kw, sh, sw;
    private final Padding padding;

    public Conv2D(int filters, int kernelH, int kernelW,
                  int strideH, int strideW, Padding padding) {
        require(filters > 0 && kernelH > 0 && kernelW > 0
                && strideH > 0 && strideW > 0 && padding != null,
                "Conv2D arguments must be positive and padding set");
        this.filters = filters;
        this.kh = kernelH;
        this.kw = kernelW;
        this.sh = strideH;
        this.sw = strideW;
        this.padding = padding;
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 4,
                "Conv2D needs 4-D NHWC input; start the model with Input4D (got "
                        + in.shape + ")");
        int cin = (int) in.shape.dims[3];
        long w = gb.glorotParam("conv_w",
                new long[]{kh, kw, cin, filters}, kh * kw * cin, kh * kw * filters);
        long b = gb.zerosParam("conv_b", new long[]{filters});
        long conv = gb.graph.op("Conv2D", gb.unique("conv2d"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrIntList("strides", 1, sh, sw, 1)
                .attrString("padding", padding.tfName)
                .input(in.op)
                .input(w)
                .build();
        long out = gb.binary("Add", "conv_bias", conv, b);
        return new Flow(out, Shape.of(-1, outDim(in.shape.dims[1]), outDim(in.shape.dims[2]), filters));
    }

    private long outDim(long inDim) {
        if (padding == Padding.SAME) {
            return (inDim + sh - 1) / sh;
        }
        return (inDim - kh) / sh + 1;
    }

    @Override
    public int typeCode() { return T_CONV2D; }

    @Override
    int[] specArgs() {
        return new int[]{filters, kh, kw, sh, sw, padding.code};
    }

    @Override
    public String toString() {
        return "Conv2D(" + filters + "," + kh + "x" + kw + ",s" + sh + "x" + sw
                + "," + padding + ")";
    }
}
