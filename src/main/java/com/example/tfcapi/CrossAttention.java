package com.example.tfcapi;

/**
 * Cross-attention inside a single sequence: the FIRST {@code queryLen}
 * positions act as queries; the REMAINING positions supply keys and values.
 * <pre>
 *   q = x[0:Tq]&middot;Wq                       [B, Tq, D]
 *   k = x[Tq:]&middot;Wk,  v = x[Tq:]&middot;Wv        [B, Tkv, D]
 *   scores = softmax(q&middot;k&#7481;/&radic;D)            [B, Tq, Tkv]
 *   out    = scores&middot;v + q                [B, Tq, D]  (residual on queries)
 * </pre>
 * This brings encoder&rarr;decoder style attention to {@code Sequential}
 * models despite the single-flow layer contract: lay your two "streams" end
 * to end in one input (e.g. a probe/question segment followed by a document
 * segment) and the layer computes directed attention across the boundary.
 * The output length is {@code queryLen} - downstream layers see only the
 * updated query segment.
 */
public final class CrossAttention extends Layer {

    private final int queryLen;

    public CrossAttention(int queryLen) {
        require(queryLen > 0, "queryLen must be positive");
        this.queryLen = queryLen;
    }

    public int queryLen() { return queryLen; }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        Layer.require(in.shape.rank() == 3,
                "CrossAttention needs an Input3D(T, D) flow (got "
                        + in.shape + ")");
        int t = (int) in.shape.dims[1];
        int d = (int) in.shape.last();
        Layer.require(t > queryLen,
                "sequence T=" + t + " must exceed queryLen=" + queryLen);

        long wq = gb.glorotParam("xa_wq", new long[]{d, d}, d, d);
        long wk = gb.glorotParam("xa_wk", new long[]{d, d}, d, d);
        long wv = gb.glorotParam("xa_wv", new long[]{d, d}, d, d);
        long wo = gb.glorotParam("xa_wo", new long[]{d, d}, d, d);

        // query segment: first queryLen positions
        long qs = gb.graph.op("Slice", gb.unique("xa_qslice"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Index", NativeTF.DT_INT32)
                .input(in.op)
                .input(gb.constI32Vec("xa_qbegin", 0, 0, 0))
                .input(gb.constI32Vec("xa_qsize", -1, queryLen, d))
                .build();
        // key/value segment: everything after
        long kv = gb.graph.op("Slice", gb.unique("xa_kvslice"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Index", NativeTF.DT_INT32)
                .input(in.op)
                .input(gb.constI32Vec("xa_kvbegin", 0, queryLen, 0))
                .input(gb.constI32Vec("xa_kvsize", -1, -1, d))
                .build();

        long q = gb.binary("BatchMatMulV2", "xa_q", qs, wq);
        long k = gb.binary("BatchMatMulV2", "xa_k", kv, wk);
        long v = gb.binary("BatchMatMulV2", "xa_v", kv, wv);

        long kt = gb.graph.op("Transpose", gb.unique("xa_t"))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(k)
                .input(gb.constInt32Tensor("xa_perm",
                        new int[]{0, 2, 1}, new long[]{3}))
                .build();

        long scores = gb.binary("Mul", "xa_scaled",
                gb.binary("BatchMatMulV2", "xa_scores", q, kt),
                gb.constF32("xa_scale", 1f / (float) Math.sqrt(d)));
        long weights = gb.unary("Softmax", "xa_weights", scores);
        long ctx = gb.binary("BatchMatMulV2", "xa_ctx", weights, v);

        long proj = gb.binary("BatchMatMulV2", "xa_proj", ctx, wo);
        long out = gb.binary("AddV2", "xa_out", proj, qs);   // residual on queries
        return new Flow(out, Shape.of(-1, queryLen, d));
    }

    @Override
    public int typeCode() { return T_CROSS_ATTENTION; }

    @Override
    int[] specArgs() { return new int[]{queryLen}; }

    @Override
    public String toString() { return "CrossAttention(queryLen=" + queryLen + ")"; }
}
