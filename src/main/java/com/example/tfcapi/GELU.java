package com.example.tfcapi;

/**
 * GELU (tanh approximation, as used by BERT/GPT):
 * <pre>
 *   y = 0.5·x·(1 + tanh( sqrt(2/pi)·(x + 0.044715·x^3) ))
 * </pre>
 * Composed from primitive ops (Mul/AddV2/Tanh/Const); TF differentiates
 * through the whole expression automatically.
 */
public final class GELU extends Layer {

    private static final float C = 0.7978845608f;   // sqrt(2/pi)
    private static final float A = 0.044715f;

    public GELU() {}

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        long x = in.op;
        long half = gb.constF32("gelu_half", 0.5f);
        long one = gb.constF32("gelu_one", 1.0f);

        long x2 = gb.binary("Mul", "gelu_x2", x, x);
        long x3 = gb.binary("Mul", "gelu_x3", x2, x);
        long inner = gb.binary("AddV2", "gelu_add", x,
                gb.binary("Mul", "gelu_ax3", gb.constF32("gelu_a", A), x3));
        long scaled = gb.binary("Mul", "gelu_scale",
                gb.constF32("gelu_c", C), inner);
        long tanh = gb.unary("Tanh", "gelu_tanh", scaled);
        long onePlus = gb.binary("AddV2", "gelu_1p", one, tanh);
        long y = gb.binary("Mul", "gelu_out",
                gb.binary("Mul", "gelu_hx", half, x), onePlus);
        return new Flow(y, in.shape);
    }

    @Override
    public int typeCode() { return T_GELU; }

    @Override
    int[] specArgs() { return new int[0]; }

    @Override
    public String toString() { return "GELU()"; }
}
