package com.example.tfcapi;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class of all layers. A layer is a configuration object that appends
 * its ops to a graph while it is wired into a {@link Sequential} model, and
 * can serialize itself to an int spec for checkpoints.
 */
public abstract class Layer {

    static final int T_INPUT = 0;
    static final int T_INPUT4D = 1;
    static final int T_FLATTEN = 2;
    static final int T_RELU = 3;
    static final int T_DENSE = 4;
    static final int T_CONV2D = 5;
    static final int T_MAXPOOL = 6;
    static final int T_SIGMOID = 7;
    static final int T_TANH = 8;
    static final int T_SOFTMAX = 9;
    static final int T_LEAKY_RELU = 10;
    static final int T_ELU = 11;
    static final int T_SELU = 12;
    static final int T_SOFTPLUS = 13;
    static final int T_SILU = 14;
    static final int T_GELU = 15;
    static final int T_INPUT3D = 16;
    static final int T_DROPOUT = 17;
    static final int T_BATCHNORM = 18;
    static final int T_EMBEDDING = 19;
    static final int T_RNN = 20;
    static final int T_GRU = 21;
    static final int T_LSTM = 22;
    static final int T_GLOBAL_AVERAGE_POOLING = 23;
    static final int T_AVGPOOL = 24;
    static final int T_DEPTHWISE_CONV = 25;
    static final int T_RESHAPE = 26;
    static final int T_RESIDUAL = 27;
    static final int T_ZEROPAD2D = 28;
    static final int T_BATCHNORM2D = 29;
    static final int T_UPSAMPLE = 30;
    static final int T_SELF_ATTENTION = 31;
    static final int T_DENSE_ND = 32;
    static final int T_LAYERNORM = 33;
    static final int T_POSITIONAL = 34;
    static final int T_EMBEDDING_ND = 35;
    static final int T_MULTIHEAD = 36;
    static final int T_CROSS_ATTENTION = 37;

    /**
     * Auxiliary feed placeholders registered by layers that need per-call
     * runtime inputs besides parameters: Dropout masks, BatchNorm statistics.
     * Order is stable: registration order during graph build.
     */
    private final java.util.List<long[]> auxPhList = new java.util.ArrayList<>();
    private final java.util.List<long[]> auxShapeList = new java.util.ArrayList<>();

    /** Registers an auxiliary placeholder; call only from {@link #build}. */
    final long aux(GraphBuilder gb, String name, long[] shape) {
        long ph = gb.placeholder(name, NativeTF.DT_FLOAT, Shape.of(shape));
        auxPhList.add(new long[]{ph});
        auxShapeList.add(shape.clone());
        return ph;
    }

    final int auxCount() { return auxPhList.size(); }

    final long[] auxPhs() {
        long[] out = new long[auxPhList.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = auxPhList.get(i)[0];
        }
        return out;
    }

    final long[][] auxShapes() {
        return auxShapeList.toArray(new long[0][]);
    }

    /**
     * Fills this layer's auxiliary feed buffers (one flat buffer per
     * placeholder, in registration order). Default: no aux feeds.
     *
     * @param n         batch size of the current run
     * @param training  true during fit(), false during predict()
     * @param batchX    the raw input batch (row-major), for BatchNorm stats
     */
    void supplyAux(float[][] buffers, int n, boolean training, float[] batchX) {
    }

    /** Appends this layer's ops; returns the new data flow. */
    abstract Flow build(GraphBuilder gb, Flow in);

    public abstract int typeCode();

    /** Spec ints following the type code (checkpoint format). */
    abstract int[] specArgs();

    final int[] spec() {
        int[] args = specArgs();
        int[] out = new int[1 + args.length];
        out[0] = typeCode();
        System.arraycopy(args, 0, out, 1, args.length);
        return out;
    }

    /** Decodes one layer from {@code spec}, reading from {@code pos[0]}. */
    static Layer decode(int[] spec, int[] pos) {
        int type = spec[pos[0]++];
        switch (type) {
            case T_INPUT:   return new Input(spec[pos[0]++]);
            case T_INPUT4D: return new Input4D(spec[pos[0]++], spec[pos[0]++], spec[pos[0]++]);
            case T_FLATTEN: return new Flatten();
            case T_RELU:    return new ReLU();
            case T_SIGMOID: return new Sigmoid();
            case T_TANH:    return new Tanh();
            case T_SOFTMAX: return new Softmax();
            case T_LEAKY_RELU: return new LeakyReLU(Float.intBitsToFloat(spec[pos[0]++]));
            case T_ELU:     return new ELU();
            case T_SELU:    return new SELU();
            case T_SOFTPLUS: return new Softplus();
            case T_SILU:    return new SiLU();
            case T_GELU:    return new GELU();
            case T_INPUT3D: return new Input3D(spec[pos[0]++], spec[pos[0]++]);
            case T_DROPOUT: return new Dropout(Float.intBitsToFloat(spec[pos[0]++]));
            case T_BATCHNORM: return new BatchNorm1D();
            case T_BATCHNORM2D: return new BatchNorm2D();
            case T_UPSAMPLE: return new Upsample2D(spec[pos[0]++]);
            case T_SELF_ATTENTION: return new SelfAttention();
            case T_DENSE_ND: return new DenseND(spec[pos[0]++]);
            case T_LAYERNORM: return new LayerNorm1D();
            case T_POSITIONAL: return new PositionalEncoding();
            case T_EMBEDDING_ND: return new EmbeddingND(spec[pos[0]++], spec[pos[0]++]);
            case T_MULTIHEAD: return new MultiHeadAttention(spec[pos[0]++], spec[pos[0]++] == 1);
            case T_CROSS_ATTENTION: return new CrossAttention(spec[pos[0]++]);
            case T_EMBEDDING: return new Embedding(spec[pos[0]++], spec[pos[0]++]);
            case T_RNN:     return new RNN(spec[pos[0]++]);
            case T_GRU:     return new GRU(spec[pos[0]++]);
            case T_LSTM:    return new LSTM(spec[pos[0]++]);
            case T_DENSE:   return new Dense(spec[pos[0]++]);
            case T_CONV2D: {
                int filters = spec[pos[0]++];
                int kh = spec[pos[0]++];
                int kw = spec[pos[0]++];
                int sh = spec[pos[0]++];
                int sw = spec[pos[0]++];
                Padding pad = Padding.fromCode(spec[pos[0]++]);
                return new Conv2D(filters, kh, kw, sh, sw, pad);
            }
            case T_MAXPOOL: {
                int ph = spec[pos[0]++];
                int pw = spec[pos[0]++];
                int sh = spec[pos[0]++];
                int sw = spec[pos[0]++];
                Padding pad = Padding.fromCode(spec[pos[0]++]);
                return new MaxPool2D(ph, pw, sh, sw, pad);
            }
            case T_AVGPOOL: {
                int ph = spec[pos[0]++];
                int pw = spec[pos[0]++];
                int sh = spec[pos[0]++];
                int sw = spec[pos[0]++];
                Padding pad = Padding.fromCode(spec[pos[0]++]);
                return new AvgPool2D(ph, pw, sh, sw, pad);
            }
            case T_DEPTHWISE_CONV: {
                int filters = spec[pos[0]++];
                int kh = spec[pos[0]++];
                int kw = spec[pos[0]++];
                int sh = spec[pos[0]++];
                int sw = spec[pos[0]++];
                Padding pad = Padding.fromCode(spec[pos[0]++]);
                return new DepthwiseConv2D(filters, kh, kw, sh, sw, pad);
            }
            case T_RESHAPE: {
                int outputSize = spec[pos[0]++];
                return new Reshape(outputSize);
            }
            case T_RESIDUAL: {
                int blockSpecLen = spec[pos[0]++];
                int[] blockSpec = new int[blockSpecLen];
                for (int i = 0; i < blockSpecLen; i++) blockSpec[i] = spec[pos[0]++];
                // decode block layers recursively
                List<Layer> block = new ArrayList<>();
                int[] bpos = {0};
                while (bpos[0] < blockSpec.length) {
                    block.add(Layer.decode(blockSpec, bpos));
                }
                return new Residual(block.toArray(new Layer[0]));
            }
            case T_ZEROPAD2D: {
                int padH = spec[pos[0]++];
                int padW = spec[pos[0]++];
                return new ZeroPad2D(padH, padW);
            }
            default:
                throw new IllegalArgumentException("unknown layer type code " + type);
        }
    }

    static void require(boolean cond, String msg) {
        if (!cond) {
            throw new IllegalArgumentException(msg);
        }
    }
}
