package com.example.tfcapi;

/**
 * Owns a native {@code TF_Session} bound to a {@link NativeGraph}.
 * {@link #run} executes one step: feeds in, fetched tensors out (caller
 * deletes each returned tensor handle).
 */
public final class NativeSession implements AutoCloseable {

    private final NativeGraph graph;
    private long addr;

    NativeSession(NativeGraph graph) {
        this.graph = graph;
        this.addr = NativeTF.newSession(graph.addr());
    }

    public long[] run(long[] feedOps, int[] feedIdx, long[] feedTensors,
                      long[] fetchOps, int[] fetchIdx) {
        return NativeTF.run(graph.addr(), addr, feedOps, feedIdx, feedTensors,
                            fetchOps, fetchIdx);
    }

    @Override
    public synchronized void close() {
        if (addr != 0) {
            NativeTF.deleteSession(graph.addr(), addr);
            addr = 0;
        }
    }
}
