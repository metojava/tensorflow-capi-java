package com.example.tfcapi;

/**
 * Embedding: maps integer token IDs to dense vectors via a lookup table
 * {@code E[vocab, embDim]}.
 *
 * <p>Input is a rank-3 flow {@code [batch, T, 1]} (from
 * {@link Input3D#Input3D(int, int) Input3D(T, 1)}) holding token IDs as
 * floats. The graph casts them to int32 and realizes the lookup as a
 * one-hot MatMul — chosen over {@code GatherV2} because its gradient is a
 * plain dense tensor, which the generic {@code TF_AddGradients} fetch path
 * handles without IndexedSlices conversion.</p>
 *
 * <p>Output: {@code [batch, T*embDim]} (2-D), ready for Dense layers.</p>
 */
public final class Embedding extends Layer {

    private final int vocabSize, embDim;

    public Embedding(int vocabSize, int embDim) {
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
                "Embedding needs an Input3D(T, 1) token flow (got "
                        + in.shape + ")");
        int t = (int) in.shape.dims[1];

        long e = gb.glorotParam("emb_table", new long[]{vocabSize, embDim},
                vocabSize, embDim);

        // ids: [B,T,1] float -> [B*T] int32
        long cast = gb.graph.op("Cast", gb.unique("emb_cast"))
                .attrType("SrcT", NativeTF.DT_FLOAT)
                .attrType("DstT", NativeTF.DT_INT32)
                .input(in.op)
                .build();
        long flatIds = gb.graph.op("Reshape", gb.unique("emb_flat_ids"))
                .attrType("T", NativeTF.DT_INT32)
                .attrType("Tshape", NativeTF.DT_INT32)
                .input(cast)
                .input(gb.constI32Vec1("emb_flat_shape", -1))
                .build();

        // onehot(ids) [B*T, V] . E [V, embDim] -> [B*T, embDim]
        long onehot = gb.graph.op("OneHot", gb.unique("emb_onehot"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("TI", NativeTF.DT_INT32)
                .attrInt("axis", -1)
                .input(flatIds)
                .input(gb.constI32Vec("emb_vocab", vocabSize))
                .input(gb.constF32("emb_on", 1.0f))
                .input(gb.constF32("emb_off", 0.0f))
                .build();
        long rows = gb.binary("MatMul", "emb_mm", onehot, e);

        // -> [B, T*embDim]
        long out = gb.graph.op("Reshape", gb.unique("emb_out"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tshape", NativeTF.DT_INT32)
                .input(rows)
                .input(gb.constI32Vec("emb_out_shape", -1, t * embDim))
                .build();
        return new Flow(out, Shape.of(-1, t * embDim));
    }

    @Override
    public int typeCode() { return T_EMBEDDING; }

    @Override
    int[] specArgs() { return new int[]{vocabSize, embDim}; }

    @Override
    public String toString() {
        return "Embedding(vocab=" + vocabSize + ", dim=" + embDim + ")";
    }
}
