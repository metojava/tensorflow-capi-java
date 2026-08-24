package com.example.tfcapi;

/** Rectified linear activation. */
public final class ReLU extends Layer {

    public ReLU() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        return new Flow(gb.unary("Relu", "relu", in.op), in.shape);
    }

    @Override
    public int typeCode() { return T_RELU; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "ReLU()"; }
}
