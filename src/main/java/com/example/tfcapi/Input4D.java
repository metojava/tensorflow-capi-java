package com.example.tfcapi;

/** Declares a 4-D image input of {@code height x width x channels} per sample. */
public final class Input4D extends Layer {

    private final int height, width, channels;

    public Input4D(int height, int width, int channels) {
        require(height > 0 && width > 0 && channels > 0,
                "height, width and channels must be positive");
        this.height = height;
        this.width = width;
        this.channels = channels;
    }

    long[] dims() {
        return new long[]{height, width, channels};
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        throw new IllegalStateException("Input4D must be the first layer");
    }

    @Override
    public int typeCode() { return T_INPUT4D; }

    @Override
    int[] specArgs() { return new int[]{height, width, channels}; }

    @Override
    public String toString() {
        return "Input4D(" + height + "," + width + "," + channels + ")";
    }
}
