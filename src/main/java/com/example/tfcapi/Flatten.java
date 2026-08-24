package com.example.tfcapi;

/** Flattens any input to 2-D {@code [batch, features]} via Reshape. */
public final class Flatten extends Layer {

    public Flatten() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        if (in.shape.rank() == 2) {
            return in;
        }
        long flat = in.shape.elemsPerSample();
        long shapeConst = gb.constI32Vec("flatten_shape", -1, (int) flat);
        long out = gb.graph.op("Reshape", gb.unique("flatten"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tshape", NativeTF.DT_INT32)
                .input(in.op)
                .input(shapeConst)
                .build();
        return new Flow(out, in.shape.flattened());
    }

    @Override
    public int typeCode() { return T_FLATTEN; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "Flatten()"; }
}
