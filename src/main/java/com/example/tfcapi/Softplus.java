package com.example.tfcapi;

/** {@code y = ln(1 + e^x)}, native {@code Softplus} op — smooth ReLU. */
public final class Softplus extends Layer {

    public Softplus() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        return new Flow(gb.unary("Softplus", "softplus", in.op), in.shape);
    }

    @Override
    public int typeCode() { return T_SOFTPLUS; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "Softplus()"; }
}
