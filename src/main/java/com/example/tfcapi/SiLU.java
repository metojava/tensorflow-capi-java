package com.example.tfcapi;

/**
 * SiLU / Swish: {@code y = x · sigmoid(x)} — composed from primitive ops
 * so it works on any TF build (no dedicated kernel needed).
 */
public final class SiLU extends Layer {

    public SiLU() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        long sig = gb.unary("Sigmoid", "silu_sig", in.op);
        long y = gb.binary("Mul", "silu_out", in.op, sig);
        return new Flow(y, in.shape);
    }

    @Override
    public int typeCode() { return T_SILU; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "SiLU()"; }
}
