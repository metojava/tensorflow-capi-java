package com.example.tfcapi;

/**
 * Row-wise softmax over the last axis, native {@code Softmax} op.
 * For classification this is normally fused into the loss; use as an
 * explicit layer only when you need probabilities from {@code predict()}.
 */
public final class Softmax extends Layer {

    public Softmax() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        // softmax over the last axis (implicit in TF >= 2.x: no dim attr)
        return new Flow(gb.unary("Softmax", "softmax", in.op), in.shape);
    }

    @Override
    public int typeCode() { return T_SOFTMAX; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "Softmax()"; }
}
