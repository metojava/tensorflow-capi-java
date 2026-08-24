package com.example.tfcapi;

/**
 * Leaky ReLU: {@code y = x if x > 0 else alpha * x}, native {@code LeakyRelu}
 * op with configurable negative slope (default 0.01).
 */
public final class LeakyReLU extends Layer {

    private final float alpha;

    public LeakyReLU() { this(0.01f); }

    public LeakyReLU(float alpha) {
        require(alpha >= 0 && alpha < 1, "alpha must be in [0,1)");
        this.alpha = alpha;
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        long op = gb.graph.op("LeakyRelu", gb.unique("leaky_relu"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrFloat("alpha", alpha)
                .input(in.op)
                .build();
        return new Flow(op, in.shape);
    }

    @Override
    public int typeCode() { return T_LEAKY_RELU; }

    @Override
    int[] specArgs() {
        // exact roundtrip of the float through the int-based checkpoint format
        return new int[]{Float.floatToIntBits(alpha)};
    }

    @Override
    public String toString() { return "LeakyReLU(alpha=" + alpha + ")"; }
}
