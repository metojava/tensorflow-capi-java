package com.example.tfcapi;

/**
 * Owns a native {@code TF_Graph} and offers a fluent builder for ops.
 * Op/tensor handles are raw native addresses; the graph must stay open
 * while any of them are used.
 */
public final class NativeGraph implements AutoCloseable {

    private long addr;

    public NativeGraph() {
        TensorFlow.load();
        this.addr = NativeTF.newGraph();
    }

    long addr() {
        ensureOpen();
        return addr;
    }

    public OpBuilder op(String opType, String name) {
        return new OpBuilder(opType, name);
    }

    /** Registers d(loss)/d(params) in the graph; returns interleaved [op, idx] pairs. */
    public long[] addGradients(long lossOp, int lossIdx, long[] params) {
        return NativeTF.addGradients(addr(), lossOp, lossIdx, params);
    }

    public NativeSession newSession() {
        return new NativeSession(this);
    }

    @Override
    public synchronized void close() {
        if (addr != 0) {
            NativeTF.deleteGraph(addr);
            addr = 0;
        }
    }

    private void ensureOpen() {
        if (addr == 0) {
            throw new IllegalStateException("graph is closed");
        }
    }

    /** Fluent builder for one operation. */
    public final class OpBuilder {
        private final long desc;

        private OpBuilder(String opType, String name) {
            this.desc = NativeTF.newOp(NativeGraph.this.addr(), opType, name);
        }

        public OpBuilder attrType(String attr, int dataType) {
            NativeTF.setAttrType(desc, attr, dataType);
            return this;
        }

        public OpBuilder attrInt(String attr, long value) {
            NativeTF.setAttrInt(desc, attr, value);
            return this;
        }

        public OpBuilder attrIntList(String attr, long... values) {
            NativeTF.setAttrIntList(desc, attr, values);
            return this;
        }

        public OpBuilder attrFloat(String attr, float value) {
            NativeTF.setAttrFloat(desc, attr, value);
            return this;
        }

        public OpBuilder attrString(String attr, String value) {
            NativeTF.setAttrString(desc, attr, value);
            return this;
        }

        public OpBuilder attrShape(String attr, long... dims) {
            NativeTF.setAttrShape(desc, attr, dims);
            return this;
        }

        public OpBuilder attrTensorF32(String attr, float[] data, long... shape) {
            NativeTF.setAttrTensorF32(desc, attr, data, shape);
            return this;
        }

        public OpBuilder attrTensorI32(String attr, int[] data, long... shape) {
            NativeTF.setAttrTensorI32(desc, attr, data, shape);
            return this;
        }

        public OpBuilder input(long op) {
            NativeTF.addInput(desc, op, 0);
            return this;
        }

        public OpBuilder input(long op, int outputIndex) {
            NativeTF.addInput(desc, op, outputIndex);
            return this;
        }

        public long build() {
            return NativeTF.finishOp(NativeGraph.this.addr(), desc);
        }
    }
}
