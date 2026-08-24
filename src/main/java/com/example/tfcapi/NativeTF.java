package com.example.tfcapi;

/**
 * Raw JNI surface of the TensorFlow C API: graph construction, tensors,
 * gradients and session execution. Generic primitives only &mdash; all layer
 * logic lives in Java. Not thread-safe; guard with the owning model.
 */
final class NativeTF {

    static final int DT_FLOAT = 1;
    static final int DT_INT32 = 3;

    private NativeTF() {}

    static native long newGraph();
    static native void deleteGraph(long graph);

    static native long newOp(long graph, String opType, String name);
    static native long finishOp(long graph, long opDesc);
    static native void setAttrType(long opDesc, String attr, int dataType);
    static native void setAttrInt(long opDesc, String attr, long value);
    static native void setAttrIntList(long opDesc, String attr, long[] values);
    static native void setAttrFloat(long opDesc, String attr, float value);
    static native void setAttrString(long opDesc, String attr, String value);
    static native void setAttrShape(long opDesc, String attr, long[] dims);
    static native void setAttrTensorF32(long opDesc, String attr,
                                        float[] data, long[] shape);
    static native void setAttrTensorI32(long opDesc, String attr,
                                        int[] data, long[] shape);
    static native void addInput(long opDesc, long op, int outputIndex);

    /** Returns interleaved [op0, idx0, op1, idx1, ...] gradient outputs. */
    static native long[] addGradients(long graph, long lossOp, int lossIdx,
                                      long[] params);

    static native long allocTensorF32(long[] shape, float[] data);
    static native long allocTensorI32(long[] shape, int[] data);
    static native float[] tensorDataF32(long tensor);
    static native long[] tensorShape(long tensor);
    static native void deleteTensor(long tensor);

    /** dst -= lr * src, element-wise. */
    static native void tensorAxpy(long dst, long src, float lr);

    static native void tensorCopy(long dst, long src);
    static native void tensorScale(long t, float alpha);
    static native void tensorAxpyAcc(long dst, long src, float alpha);
    static native void tensorAxpyAccSquare(long dst, long src, float alpha);
    static native void tensorAdamApply(long w, long m, long v, float lrT, float eps);

    static native long newSession(long graph);
    static native long[] run(long graph, long session,
                             long[] feedOps, int[] feedIdx, long[] feedTensors,
                             long[] fetchOps, int[] fetchIdx);
    static native void deleteSession(long graph, long session);
}
