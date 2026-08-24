package com.example.tfcapi;

/**
 * Adam (Kingma &amp; Ba, 2015). Keeps exponential moving averages of the
 * gradient and its square, with bias correction folded into the effective
 * learning rate:
 * <pre>
 *   m   = b1*m + (1-b1)*g
 *   v   = b2*v + (1-b2)*g^2
 *   lrT = lr * sqrt(1-b2^t) / (1-b1^t)
 *   w  -= lrT * m / (sqrt(v) + eps)
 * </pre>
 * Defaults: lr 0.001, b1 0.9, b2 0.999, eps 1e-8.
 */
public final class Adam extends Optimizer {

    private final float lr, beta1, beta2, eps;
    private long[] m, v;
    private int t;

    public Adam(float learningRate) {
        this(learningRate, 0.9f, 0.999f, 1e-8f);
    }

    public Adam(float learningRate, float beta1, float beta2, float eps) {
        if (learningRate <= 0 || beta1 <= 0 || beta1 >= 1
                || beta2 <= 0 || beta2 >= 1 || eps <= 0) {
            throw new IllegalArgumentException(
                    "need lr > 0, betas in (0,1), eps > 0");
        }
        this.lr = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
    }

    @Override
    protected void initState() {
        m = new long[shapes.length];
        v = new long[shapes.length];
        for (int i = 0; i < shapes.length; i++) {
            m[i] = zeros(shapes[i]);
            v[i] = zeros(shapes[i]);
        }
        t = 0;
    }

    @Override
    protected void step(long[] params, long[] grads) {
        t++;
        double bc1 = 1.0 - Math.pow(beta1, t);
        double bc2 = 1.0 - Math.pow(beta2, t);
        float lrT = (float) (effLr(lr) * Math.sqrt(bc2) / bc1);

        for (int i = 0; i < params.length; i++) {
            NativeTF.tensorScale(m[i], beta1);
            NativeTF.tensorAxpyAcc(m[i], grads[i], 1f - beta1);
            NativeTF.tensorScale(v[i], beta2);
            NativeTF.tensorAxpyAccSquare(v[i], grads[i], 1f - beta2);
            NativeTF.tensorAdamApply(params[i], m[i], v[i], lrT, eps);
        }
    }

    @Override
    public void close() {
        if (m != null) {
            for (int i = 0; i < m.length; i++) {
                NativeTF.deleteTensor(m[i]);
                NativeTF.deleteTensor(v[i]);
            }
            m = null;
            v = null;
        }
    }

    @Override
    public String toString() {
        return "Adam(lr=" + lr + ", b1=" + beta1 + ", b2=" + beta2 + ")";
    }
}
