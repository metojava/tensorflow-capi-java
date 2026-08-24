package com.example.tfcapi;

/**
 * Sequence-preserving Embedding: maps integer token ids to dense vectors via
 * a lookup table {@code E[vocab, embDim]} and KEEPS the time axis - output is
 * rank-3 {@code [batch, T, embDim]}, ready for RNN/LSTM/GRU/SelfAttention.
 *
 * <p>This fixes the composition gap of the original {@link Embedding}, whose
 * output is flattened {@code [batch, T*embDim]} and therefore can only feed
 * Dense heads. Same one-hot MatMul realization as {@link Embedding} (dense
 * gradients, no IndexedSlices), same input contract: rank-3 token flow from
 * {@code Input3D(T, 1)} with ids as floats.</p>
 */
public final class EmbeddingND extends Layer {

    private final int vocabSize, embDim;

    public EmbeddingND(int vocabSize, int embDim) {
        require(vocabSize > 0 && embDim > 0,
                "vocabSize and embDim must be positive");
        this.vocabSize = vocabSize;
        this.embDim = embDim;
    }

    public int vocabSize() { return vocabSize; }
    public int embDim() { return embDim; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 3 && in.shape.last() == 1,
                "EmbeddingND needs an Input3D(T, 1) token flow (got "
                        + in.shape + ")");
        int t = (int) in.shape.dims[1];

        long e = gb.glorotParam("embnd_table",
                new long[]{vocabSize, embDim}, vocabSize, embDim);

        // ids: [B,T,1] float -> [B*T] int32
        long cast = gb.graph.op("Cast", gb.unique("embnd_cast"))
                .attrType("SrcT", NativeTF.DT_FLOAT)
                .attrType("DstT", NativeTF.DT_INT32)
                .input(in.op)
                .build();
        long flatIds = gb.graph.op("Reshape", gb.unique("embnd_flat"))
                .attrType("T", NativeTF.DT_INT32)
                .attrType("Tshape", NativeTF.DT_INT32)
                .input(cast)
                .input(gb.constI32Vec1("embnd_shape1", -1))
                .build();

        // onehot(ids) [B*T, V] . E [V, embDim] -> [B*T, embDim]
        long onehot = gb.graph.op("OneHot", gb.unique("embnd_onehot"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("TI", NativeTF.DT_INT32)
                .attrInt("axis", -1)
                .input(flatIds)
                .input(gb.constI32Vec("embnd_vocab", vocabSize))
                .input(gb.constF32("embnd_on", 1.0f))
                .input(gb.constF32("embnd_off", 0.0f))
                .build();
        long rows = gb.binary("MatMul", "embnd_mm", onehot, e);

        long out = gb.graph.op("Reshape", gb.unique("embnd_out"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tshape", NativeTF.DT_INT32)
                .input(rows)
                .input(gb.constInt32Tensor("embnd_out_shape",
                        new int[]{-1, t, embDim}, new long[]{3}))
                .build();
        return new Flow(out, Shape.of(-1, t, embDim));
    }

    @Override
    public int typeCode() { return T_EMBEDDING_ND; }

    @Override
    int[] specArgs() { return new int[]{vocabSize, embDim}; }

    @Override
    public String toString() {
        return "EmbeddingND(vocab=" + vocabSize + ", dim=" + embDim + ")";
    }
}
