package com.example.tfcapi;

/**
 * Nearest-neighbor 2x upsampling for NCHW flows - the decoder-side twin of
 * {@link MaxPool2D}, useful in autoencoders between transpose-free decoder
 * stages. Implemented with TF's {@code ResizeNearestNeighbor} (gradient
 * registered, so end-to-end training works).
 *
 * <pre>{@code
 * new Conv2D(8,3,3,1,1,Padding.SAME), new ReLU(),
 * new Upsample2D(2),                       // [B,C,H,W] -> [B,C,2H,2W]
 * new Conv2D(1,3,3,1,1,Padding.SAME)
 * }</pre>
 */
public final class Upsample2D extends Layer {

    private final int scale;

    public Upsample2D(int scale) {
        require(scale > 1, "scale must be > 1");
        this.scale = scale;
    }

    public int scale() { return scale; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 4,
                "Upsample2D needs a 4-D NCHW flow (got " + in.shape + ")");
        int h = (int) in.shape.dims[2];
        int w = (int) in.shape.last();
        long size = gb.constInt32Tensor("upsample_size",
                new int[]{h * scale, w * scale}, new long[]{2});
        // TF resize ops are channels-last: NCHW -> NHWC -> resize -> NCHW
        long toNhwc = gb.constInt32Tensor("upsample_to_nhwc",
                new int[]{0, 2, 3, 1}, new long[]{4});
        long toNchw = gb.constInt32Tensor("upsample_to_nchw",
                new int[]{0, 3, 1, 2}, new long[]{4});
        long nhwc = gb.graph.op("Transpose", gb.unique("upsample_t1"))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(in.op)
                .input(toNhwc)
                .build();
        long resized = gb.graph.op("ResizeNearestNeighbor", gb.unique("upsample"))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(nhwc)
                .input(size)
                .build();
        long out = gb.graph.op("Transpose", gb.unique("upsample_t2"))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(resized)
                .input(toNchw)
                .build();
        return new Flow(out, Shape.of(-1,
                (int) in.shape.dims[1], h * scale, w * scale));
    }

    @Override
    public int typeCode() { return T_UPSAMPLE; }

    @Override
    int[] specArgs() { return new int[]{scale}; }

    @Override
    public String toString() { return "Upsample2D(x" + scale + ")"; }
}
