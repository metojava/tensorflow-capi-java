package com.example.tfcapi;

/** Declares a flat 1-D input of {@code features} values per sample. */
public final class Input extends Layer {

    private final int features;

    public Input(int features) {
        require(features > 0, "features must be positive");
        this.features = features;
    }

    long[] dims() {
        return new long[]{features};
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        throw new IllegalStateException("Input must be the first layer");
    }

    @Override
    public int typeCode() { return T_INPUT; }

    @Override
    int[] specArgs() { return new int[]{features}; }

    @Override
    public String toString() { return "Input(" + features + ")"; }
}
