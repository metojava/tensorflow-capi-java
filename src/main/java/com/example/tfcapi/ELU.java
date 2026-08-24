package com.example.tfcapi;

/**
 * Exponential linear unit: {@code y = x if x > 0 else e^x - 1},
 * native {@code Elu} op. Smooth negative branch; mean activation closer
 * to zero than ReLU.
 */
public final class ELU extends Layer {

    public ELU() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        return new Flow(gb.unary("Elu", "elu", in.op), in.shape);
    }

    @Override
    public int typeCode() { return T_ELU; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "ELU()"; }
}
