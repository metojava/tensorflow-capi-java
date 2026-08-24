package com.example.tfcapi;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Accumulates ops, trainable parameters and unique names while a
 * {@link Sequential} graph is being built. Parameters are placeholder-fed
 * native tensors owned by the model (same scheme as SimpleNeuralNet).
 */
public final class GraphBuilder {

    final NativeGraph graph;
    private final Random rnd;
    private final List<Long> paramOps = new ArrayList<>();
    private final List<Long> paramTensors = new ArrayList<>();
    private final List<long[]> paramShapes = new ArrayList<>();
    private int uniq;

    public GraphBuilder(NativeGraph graph) {
        this(graph, 42L);
    }

    /** Seeded construction makes weight initialization reproducible. */
    public GraphBuilder(NativeGraph graph, long seed) {
        this.graph = graph;
        this.rnd = new Random(seed);
    }

    public String unique(String base) {
        return base + "_" + (uniq++);
    }

    public long placeholder(String name, int dtype, Shape shape) {
        return graph.op("Placeholder", unique(name))
                .attrType("dtype", dtype)
                .attrShape("shape", shape.dims)
                .build();
    }

    /** Registers a trainable parameter: placeholder + initialized native tensor. */
    private long param(String name, long[] shape, float[] initData) {
        long ph = placeholder(name, NativeTF.DT_FLOAT, Shape.of(shape));
        long t = NativeTF.allocTensorF32(shape, initData);
        paramOps.add(ph);
        paramTensors.add(t);
        paramShapes.add(shape.clone());
        return ph;
    }

    /** Glorot-uniform weight of any rank; fanIn/fanOut control the limit. */
    public long glorotParam(String name, long[] shape, int fanIn, int fanOut) {
        float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        float[] data = new float[(int) product(shape)];
        for (int i = 0; i < data.length; i++) {
            data[i] = (rnd.nextFloat() * 2f - 1f) * limit;
        }
        return param(name, shape, data);
    }

    public long zerosParam(String name, long[] shape) {
        return param(name, shape, new float[(int) product(shape)]);
    }

    /** Row of ones, e.g. BatchNorm gamma initialization. */
    public long onesParam(String name, long[] shape) {
        float[] data = new float[(int) product(shape)];
        java.util.Arrays.fill(data, 1f);
        return param(name, shape, data);
    }

    public long unary(String type, String name, long in) {
        return graph.op(type, unique(name))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(in)
                .build();
    }

    public long binary(String type, String name, long a, long b) {
        return graph.op(type, unique(name))
                .attrType("T", NativeTF.DT_FLOAT)
                .input(a)
                .input(b)
                .build();
    }

    public long constF32(String name, float v) {
        return graph.op("Const", unique(name))
                .attrTensorF32("value", new float[]{v})
                .attrType("dtype", NativeTF.DT_FLOAT)
                .build();
    }

    public long constI32Vec(String name, int... vals) {
        long[] shape = vals.length == 1 ? new long[0] : new long[]{vals.length};
        return graph.op("Const", unique(name))
                .attrTensorI32("value", vals, shape)
                .attrType("dtype", NativeTF.DT_INT32)
                .build();
    }

    /** Single-element INT32 constant with EXPLICIT rank-1 shape [1]. */
    long constI32Vec1(String name, int v) {
        return graph.op("Const", unique(name))
                .attrTensorI32("value", new int[]{v}, new long[]{1})
                .attrType("dtype", NativeTF.DT_INT32)
                .build();
    }

    /** Create a Const op with an arbitrary int32 tensor value and explicit shape. */
    public long constInt32Tensor(String name, int[] values, long[] shape) {
        return graph.op("Const", unique(name))
                .attrTensorI32("value", values, shape)
                .attrType("dtype", NativeTF.DT_INT32)
                .build();
    }

    /** Create a Const op with an arbitrary float tensor value and explicit shape. */
    long constFloat32Tensor(String name, float[] values, long[] shape) {
        return graph.op("Const", unique(name))
                .attrTensorF32("value", values, shape)
                .attrType("dtype", NativeTF.DT_FLOAT)
                .build();
    }

    /** Create a Const op with an arbitrary int32 tensor value and explicit shape. */
    long constTensorI32(String name, int[] values, long[] shape) {
        return graph.op("Const", unique(name))
                .attrTensorI32("value", values, shape)
                .attrType("dtype", NativeTF.DT_INT32)
                .build();
    }

    public long[] paramOps() { return toArray(paramOps); }
    /** Number of parameters registered so far (for per-layer attribution). */
    public int paramCount() { return paramOps.size(); }

    /** The underlying graph, for building custom ops fluently. */
    public NativeGraph graph() { return graph; }
    public long[] paramTensors() { return toArray(paramTensors); }
    public long[][] paramShapes() {
        return paramShapes.toArray(new long[0][]);
    }

    private static long product(long[] shape) {
        long p = 1;
        for (long d : shape) p *= d;
        return p;
    }

    private static long[] toArray(List<Long> list) {
        long[] out = new long[list.size()];
        for (int i = 0; i < out.length; i++) out[i] = list.get(i);
        return out;
    }
}
