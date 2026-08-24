package com.example.tfcapi;

/** Vanilla gradient descent: {@code w -= lr * grad}. */
public final class Sgd extends Optimizer {

    private final float lr;

    public Sgd(float learningRate) {
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive");
        }
        this.lr = learningRate;
    }

    @Override
    protected void initState() {
    }

    @Override
    protected void step(long[] params, long[] grads) {
        float stepLr = effLr(lr);
        for (int i = 0; i < params.length; i++) {
            NativeTF.tensorAxpy(params[i], grads[i], stepLr);
        }
    }

    @Override
    public void close() {
    }

    @Override
    public String toString() {
        return "Sgd(lr=" + lr + ")";
    }
}
