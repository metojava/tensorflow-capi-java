package com.example.tfcapi;

/**
 * Keras-style early stopping: configuration passed to the {@code fit(...)}
 * overloads that accept it, doubling as the result holder afterwards.
 *
 * <p>Training stops when the monitored validation metric fails to improve
 * for {@code patience} consecutive epochs. With
 * {@code restoreBestWeights=true} the model's parameters are snapshotted at
 * every new best epoch and written back when training ends, so the returned
 * model is the best one seen - not merely the last one.</p>
 *
 * <pre>{@code
 * EarlyStopping es = EarlyStopping.valLoss(5, true);
 * History h = net.fit(trainX, trainY, 300, 32, testX, testY, true, es);
 * System.out.println(es.stoppedEarly + " best=" + es.bestEpoch);
 * }</pre>
 */
public final class EarlyStopping {

    /** Metric watched for improvement. */
    public enum Monitor { VAL_LOSS, VAL_ACC }

    public final Monitor monitor;
    public final int patience;
    public final boolean restoreBestWeights;
    public final float minDelta;

    /* --------------------- results (filled by fit) ---------------------- */
    /** Epochs actually executed (may be less than requested). */
    public int epochsRun;
    /** Epoch index (0-based) of the best monitored score, -1 if none. */
    public int bestEpoch = -1;
    /** Best monitored score observed. */
    public float bestScore;
    /** True if training was cut short by this callback. */
    public boolean stoppedEarly;

    private EarlyStopping(Monitor monitor, int patience,
                          boolean restoreBestWeights, float minDelta) {
        if (patience < 1) {
            throw new IllegalArgumentException("patience must be >= 1");
        }
        this.monitor = monitor;
        this.patience = patience;
        this.restoreBestWeights = restoreBestWeights;
        this.minDelta = minDelta;
    }

    /** Stop when validation loss stops improving (no weight restore). */
    public static EarlyStopping valLoss(int patience) {
        return valLoss(patience, false);
    }

    /** Stop on validation loss plateau; optionally restore best weights. */
    public static EarlyStopping valLoss(int patience, boolean restoreBestWeights) {
        return new EarlyStopping(Monitor.VAL_LOSS, patience, restoreBestWeights, 0f);
    }

    /** Stop when validation accuracy plateaus; optionally restore best weights. */
    public static EarlyStopping valAccuracy(int patience, boolean restoreBestWeights) {
        return new EarlyStopping(Monitor.VAL_ACC, patience, restoreBestWeights, 0f);
    }

    boolean improves(float score, float best) {
        return monitor == Monitor.VAL_LOSS
                ? score < best - minDelta
                : score > best + minDelta;
    }

    float initialBest() {
        return monitor == Monitor.VAL_LOSS ? Float.POSITIVE_INFINITY : Float.NEGATIVE_INFINITY;
    }
}
