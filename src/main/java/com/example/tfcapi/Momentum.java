package com.example.tfcapi;

/**
 * SGD with classical momentum: {@code v = mu*v + grad; w -= lr * v}.
 * Typical momentum is 0.9.
 */
public final class Momentum extends Optimizer {

    private final float lr, mu;
    private long[] velocity;

    public Momentum(float learningRate, float momentum) {
        if (learningRate <= 0 || momentum < 0 || momentum >= 1) {
            throw new IllegalArgumentException(
                    "learningRate must be positive and momentum in [0,1)");
        }
        this.lr = learningRate;
        this.mu = momentum;
    }

    @Override
    protected void initState() {
        velocity = new long[shapes.length];
        for (int i = 0; i < shapes.length; i++) {
            velocity[i] = zeros(shapes[i]);
        }
    }

    @Override
    protected void step(long[] params, long[] grads) {
        float stepLr = effLr(lr);
        for (int i = 0; i < params.length; i++) {
            NativeTF.tensorScale(velocity[i], mu);
            NativeTF.tensorAxpyAcc(velocity[i], grads[i], 1f);
            NativeTF.tensorAxpy(params[i], velocity[i], stepLr);
        }
    }

    @Override
    public void close() {
        if (velocity != null) {
            for (long v : velocity) {
                NativeTF.deleteTensor(v);
            }
            velocity = null;
        }
    }

    @Override
    public String toString() {
        return "Momentum(lr=" + lr + ", mu=" + mu + ")";
    }
}
