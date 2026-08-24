package com.example.tfcapi;

/**
 * Rank-3 input for sequence models: {@code [batch, T, D]} per sample
 * ({@code T} timesteps of {@code D} features). Values are floats; for
 * {@link Embedding} they are token IDs stored as floats and cast in-graph.
 */
public final class Input3D extends Layer {

    private final int timesteps, features;

    public Input3D(int timesteps, int features) {
        require(timesteps > 0 && features > 0,
                "timesteps and features must be positive");
        this.timesteps = timesteps;
        this.features = features;
    }

    long[] dims() {
        return new long[]{timesteps, features};
    }

    @Override
    Flow build(GraphBuilder gb, Flow in) {
        throw new IllegalStateException("Input3D must be the first layer");
    }

    @Override
    public int typeCode() { return T_INPUT3D; }

    @Override
    int[] specArgs() { return new int[]{timesteps, features}; }

    @Override
    public String toString() {
        return "Input3D(T=" + timesteps + ", D=" + features + ")";
    }
}
