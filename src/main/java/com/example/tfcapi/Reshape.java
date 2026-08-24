package com.example.tfcapi;

/** Reshapes the flow's output to a new last dimension size. */
public final class Reshape extends Layer {

    private final int outputSize;

    public Reshape(int outputSize) {
        this.outputSize = outputSize;
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() >= 2,
                "Reshape needs at least 2-D input (got " + in.shape + ")");
        // Create a constant shape tensor [−1, outputSize] for the Reshape op.
        long shapeConst = gb.constTensorI32("reshape_shape",
                new int[]{-1, outputSize}, new long[]{2});
        long reshaped = gb.graph.op("Reshape", gb.unique("reshape"))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(in.op)
                .input(shapeConst)
                .build();
        return new Flow(reshaped, Shape.of(-1, outputSize));
    }

    @Override
    public int typeCode() { return T_RESHAPE; }

    @Override
    int[] specArgs() { return new int[]{outputSize}; }

    @Override
    public String toString() { return "Reshape(" + outputSize + ")"; }
}