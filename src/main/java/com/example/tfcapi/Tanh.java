package com.example.tfcapi;

/** Hyperbolic tangent activation, native {@code Tanh} op. */
public final class Tanh extends Layer {

    public Tanh() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        return new Flow(gb.unary("Tanh", "tanh", in.op), in.shape);
    }

    @Override
    public int typeCode() { return T_TANH; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "Tanh()"; }
}
