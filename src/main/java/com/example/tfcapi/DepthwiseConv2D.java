package com.example.tfcapi;

/** Depthwise 2-D convolution (NHWC) with per‑channel filters. */
public final class DepthwiseConv2D extends Layer {

    private final int filters, kh, kw, sh, sw;
    private final Padding padding;

    public DepthwiseConv2D(int filters, int kernelH, int kernelW,
                   int strideH, int strideW, Padding padding) {
        require(filters > 0 && kernelH > 0 && kernelW > 0
                && strideH > 0 && strideW > 0 && padding != null,
                "DepthwiseConv2D arguments must be positive and padding set");
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
                "DepthwiseConv2D needs 4-D NHWC input; start the model with Input4D (got " + in.shape + ")");
        int cin = (int) in.shape.dims[3];
        Layer.require(filters == cin,
                "DepthwiseConv2D filters (" + filters + ") must equal input channels (" + cin + ")");
        long w = gb.glorotParam("dw_conv_w",
                new long[]{kh, kw, cin, 1}, kh * kw * cin, kh * kw * cin);
        long out = gb.graph.op("DepthwiseConv2dNative", gb.unique("dwconv2d"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrIntList("strides", 1, sh, sw, 1)
                .attrString("padding", padding.tfName)
                .input(in.op)
                .input(w)
                .build();
        return new Flow(out, Shape.of(-1,
                outDim(in.shape.dims[1], kh, sh), outDim(in.shape.dims[2], kw, sw),
                cin));
    }

    private long outDim(long inDim, int k, int s) {
        if (padding == Padding.SAME) {
            return (inDim + s - 1) / s;
        }
        return (inDim - k) / s + 1;
    }

    @Override
    public int typeCode() { return T_DEPTHWISE_CONV; }

    @Override
    int[] specArgs() {
        return new int[]{filters, kh, kw, sh, sw, padding.code};
    }

    @Override
    public String toString() {
        return "DepthwiseConv2D(" + filters + "," + kh + "x" + kw + ",s" + sh + "x" + sw
                + "," + padding + ")";
    }
}