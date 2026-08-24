package com.example.tfcapi;

import java.util.Arrays;

/** Symbolic tensor shape; dims[0] is the batch dimension, {@code -1} = dynamic. */
public final class Shape {

    final long[] dims;

    private Shape(long[] dims) {
        this.dims = dims;
    }

    public static Shape of(long... dims) {
        return new Shape(dims.clone());
    }

    int rank() {
        return dims.length;
    }

    long last() {
        return dims[dims.length - 1];
    }

    /** Product of all non-batch dimensions. */
    long elemsPerSample() {
        long p = 1;
        for (int i = 1; i < dims.length; i++) {
            p *= dims[i];
        }
        return p;
    }

    Shape flattened() {
        return of(-1, elemsPerSample());
    }

    @Override
    public String toString() {
        return Arrays.toString(dims);
    }
}
