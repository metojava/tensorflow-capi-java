package com.example.tfcapi;

import java.util.Arrays;

public final class RMSProp extends Optimizer {
    private float learningRate;
    private float decay;
    private float momentum;
    private float epsilon;
    private long[] state;
    
    public RMSProp(float learningRate, float decay, float momentum, float epsilon) {
        this.learningRate = learningRate;
        this.decay = decay;
        this.momentum = momentum;
        this.epsilon = epsilon;
    }
    
    public RMSProp(float learningRate) {
        this(learningRate, 0.9f, 0.0f, 1e-8f);
    }
    
    @Override
    public void init(long[][] paramShapes) {
        super.init(paramShapes);
        // Initialize RMSProp state (cache of squared gradients)
        state = new long[paramShapes.length];
        for (int i = 0; i < paramShapes.length; i++) {
            state[i] = NativeTF.allocTensorF32(paramShapes[i], new float[paramShapes[i].length]);
        }
    }
    
    @Override
    protected void initState() {
        // Initialize RMSProp state
    }
    
    @Override
    protected void step(long[] params, long[] grads) {
        // Apply RMSProp update rule
        // For now, we'll just print a message
        System.out.println("Applying RMSProp update with lr=" + learningRate);
    }
    
    @Override
    public void close() {
        // Clean up resources if needed
        // Delete state tensors
        if (state != null) {
            for (long s : state) {
                NativeTF.deleteTensor(s);
            }
            state = null;
        }
    }
    
    @Override
    public String toString() {
        return "RMSProp(lr=" + learningRate + ", decay=" + decay + ", momentum=" + momentum + ", epsilon=" + epsilon + ")";
    }
}