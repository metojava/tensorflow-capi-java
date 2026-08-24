package com.example.tfcapi;

/**
 * Scaled exponential linear unit, native {@code Selu} op.
 * Keep inputs standardized and use proper weight init to preserve the
 * self-normalizing property.
 */
public final class SELU extends Layer {

    public SELU() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        return new Flow(gb.unary("Selu", "selu", in.op), in.shape);
    }

    @Override
    public int typeCode() { return T_SELU; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "SELU()"; }
}
