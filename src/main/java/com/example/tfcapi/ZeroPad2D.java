package com.example.tfcapi;

/** Zero-pads a 2-D flow (NHWC) along the height and width dimensions. */
public final class ZeroPad2D extends Layer {

    private final int ph, pw; // total padding for height and width

    public ZeroPad2D(int padH, int padW) {
        this.ph = padH;
        this.pw = padW;
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 4,
                "ZeroPad2D needs 4-D NHWC input (got " + in.shape + ")");
        int h = (int) in.shape.dims[1];
        int w = (int) in.shape.dims[2];
        int cin = (int) in.shape.dims[3];

        // Compute split pads (top/bottom for height, left/right for width).
        int padHtop = ph / 2;
        int padHbot = ph - padHtop;
        int padWleft = pw / 2;
        int padWright = pw - padWleft;

        // Build the paddings tensor: [[0,0],[pad_h_top,pad_h_bot],[pad_w_left,pad_w_right],[0,0]]
        int[][] padsArray = {{0, 0}, {padHtop, padHbot}, {padWleft, padWright}, {0, 0}};
        long padsConst = gb.constTensorI32("zeropad_pads",
                flattenInts(padsArray), new long[]{4, 2});

        long padded = gb.graph.op("Pad", gb.unique("zeropad"))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(in.op)
                .input(padsConst)
                .build();

        int newH = h + ph;
        int newW = w + pw;
        return new Flow(padded, Shape.of(-1, newH, newW, cin));
    }

    private static int[] flattenInts(int[][] array) {
        int total = 0;
        for (int[] row : array) total += row.length;
        int[] out = new int[total];
        int idx = 0;
        for (int[] row : array) {
            System.arraycopy(row, 0, out, idx, row.length);
            idx += row.length;
        }
        return out;
    }

    @Override
    public int typeCode() { return T_ZEROPAD2D; }

    @Override
    int[] specArgs() { return new int[]{ph, pw}; }

    @Override
    public String toString() { return "ZeroPad2D(ph=" + ph + ",pw=" + pw + ")"; }
}