package com.example.tfcapi;

/** Residual (skip-connection) layer: output = input + block(input). */
public final class Residual extends Layer {

    private final Layer[] block;

    public Residual(Layer... block) {
        this.block = block;
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(block.length != 0, "Residual block must not be empty");
        // Build the sub-graph starting from the residual input.
        Flow f = in;
        for (Layer l : block) {
            f = l.build(gb, f);
        }
        // Add the original input to the block's output.
        long added = gb.binary("Add", "residual_add", in.op, f.op);
        return new Flow(added, in.shape);
    }

    @Override
    public int typeCode() { return T_RESIDUAL; }

    @Override
    int[] specArgs() {
        // encode: number of layers, then each layer's spec ints consecutively
        int total = 1; // the count itself
        for (Layer l : block) {
            total += l.spec().length;
        }
        int[] args = new int[total];
        args[0] = block.length;
        int offset = 1;
        for (Layer l : block) {
            int[] s = l.spec();
            System.arraycopy(s, 0, args, offset, s.length);
            offset += s.length;
        }
        return args;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Residual(");
        for (int i = 0; i < block.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(block[i].toString());
        }
        sb.append(")");
        return sb.toString();
    }
}