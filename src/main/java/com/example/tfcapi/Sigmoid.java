package com.example.tfcapi;

/** Logistic sigmoid activation, native {@code Sigmoid} op. */
public final class Sigmoid extends Layer {

    public Sigmoid() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        return new Flow(gb.unary("Sigmoid", "sigmoid", in.op), in.shape);
    }

    @Override
    public int typeCode() { return T_SIGMOID; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "Sigmoid()"; }
}
