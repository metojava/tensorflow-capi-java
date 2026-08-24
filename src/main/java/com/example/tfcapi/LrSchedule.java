package com.example.tfcapi;

/**
 * Learning-rate schedule expressed as a <em>fraction</em> of the optimizer's
 * configured learning rate. Attach to {@link Sequential} or {@link Regressor}
 * with {@code net.setLrSchedule(...)}; before every epoch the model sets the
 * effective rate to {@code baseLr * fractionAt(epoch)} by adjusting the
 * optimizer multiplier - momentum/Adam state is preserved (no re-init).
 *
 * <pre>{@code
 * net.setLrSchedule(LrSchedule.cosine(40, 0.01f)); // decay over 40 epochs
 * History h = net.fit(trainX, trainY, 40, 64, testX, testY, true);
 * }</pre>
 */
public interface LrSchedule {

    /** Fraction of the configured learning rate for the given 0-based epoch. */
    float fractionAt(int epoch);

    /**
     * Cosine annealing from 1.0 down to {@code minFraction} over
     * {@code totalEpochs}, then flat.
     */
    static LrSchedule cosine(int totalEpochs, float minFraction) {
        if (totalEpochs <= 0 || minFraction < 0 || minFraction >= 1) {
            throw new IllegalArgumentException("bad cosine schedule args");
        }
        return epoch -> {
            if (epoch >= totalEpochs) return minFraction;
            double t = Math.PI * epoch / totalEpochs;
            return (float) (minFraction + (1 - minFraction) * (0.5 + 0.5 * Math.cos(t)));
        };
    }

    /** Step decay: multiply the previous fraction by {@code factor} every {@code every} epochs. */
    static LrSchedule step(int every, float factor) {
        if (every <= 0 || factor <= 0 || factor >= 1) {
            throw new IllegalArgumentException("bad step schedule args");
        }
        return epoch -> (float) Math.pow(factor, epoch / every);
    }

    /** Exponential decay: {@code factor^epoch} (factor slightly below 1). */
    static LrSchedule exponential(float factor) {
        if (factor <= 0 || factor >= 1) {
            throw new IllegalArgumentException("factor must be in (0,1)");
        }
        return epoch -> (float) Math.pow(factor, epoch);
    }

    /** Constant schedule (identity hook; useful to toggle schedules off). */
    static LrSchedule constant() {
        return epoch -> 1f;
    }
}
