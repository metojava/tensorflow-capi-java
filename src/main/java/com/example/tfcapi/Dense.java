package com.example.tfcapi;

/**
 * Fully-connected layer on 2-D input: {@code out = x . W + b}.
 * Weights use Glorot-uniform init, bias zero init.
 */
public final class Dense extends Layer {

    private final int units;

    public Dense(int units) {
        require(units > 0, "units must be positive");
        this.units = units;
    }

    public int units() { return units; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 2,
                "Dense needs 2-D input; insert Flatten before it (input shape "
                        + in.shape + ")");
        int fanIn = (int) in.shape.last();
        long w = gb.glorotParam("dense_w", new long[]{fanIn, units}, fanIn, units);
        long b = gb.zerosParam("dense_b", new long[]{units});
        long mm = gb.binary("MatMul", "dense_matmul", in.op, w);
        long out = gb.binary("Add", "dense_bias", mm, b);
        return new Flow(out, Shape.of(-1, units));
    }

    @Override
    public int typeCode() { return T_DENSE; }

    @Override
    int[] specArgs() { return new int[]{units}; }

    @Override
    public String toString() { return "Dense(" + units + ")"; }
}
