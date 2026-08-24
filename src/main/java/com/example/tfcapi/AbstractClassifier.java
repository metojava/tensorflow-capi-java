package com.example.tfcapi;

import java.util.Random;

/**
 * Shared Keras-style training loop for classifiers that expose
 * {@link #trainBatch}, {@link #predict} and fixed input/class counts.
 * Both {@link SimpleNeuralNet} and {@link Sequential} inherit
 * {@link #fit} from here.
 */
public abstract class AbstractClassifier implements AutoCloseable {

    /** Epoch-shuffle seed; fix via {@link #setShuffleSeed(long)} for reproducible runs. */
    private long shuffleSeed = 42L;

    /** Fixes the RNG seed used to shuffle batch order every epoch. */
    public synchronized void setShuffleSeed(long seed) { this.shuffleSeed = seed; }

    /** One SGD step over a batch; returns the mean training loss. */
    public abstract float trainBatch(float[] batchX, int[] batchY);

    /** Probabilities for a batch; returns {@code rows * numClasses} floats. */
    public abstract float[] predict(float[] batchX);

    /** Input values per sample (e.g. 784 for 28x28x1 images). */
    public abstract int inputElems();

    /** Number of output classes. */
    public abstract int numClasses();

    /**
     * Keras-style training loop: runs {@code epochs} full passes over the
     * training set, shuffling the batch order every epoch. Returns a
     * {@link History} with the per-epoch mean losses.
     */
    public History fit(float[] trainX, int[] trainY, int epochs, int batchSize) {
        return fit(trainX, trainY, epochs, batchSize, null, null, true);
    }

    /** As {@link #fit(float[], int[], int, int, float[], int[], boolean)} without early stopping. */
    public synchronized History fit(float[] trainX, int[] trainY, int epochs, int batchSize,
                                    float[] valX, int[] valY, boolean verbose) {
        return fit(trainX, trainY, epochs, batchSize, valX, valY, verbose, null);
    }

    /**
     * Full training loop with optional {@link EarlyStopping}. When stopping
     * fires and {@code restoreBestWeights} is on, parameters roll back to the
     * best epoch before returning.
     */
    public synchronized History fit(float[] trainX, int[] trainY, int epochs, int batchSize,
                                    float[] valX, int[] valY, boolean verbose,
                                    EarlyStopping es) {
        validateDataset("train", trainX, trainY);
        if (epochs <= 0 || batchSize <= 0) {
            throw new IllegalArgumentException("epochs and batchSize must be positive");
        }
        boolean hasVal = valY != null || valX != null;
        if (hasVal) {
            validateDataset("val", valX, valY);
        } else if (es != null) {
            throw new IllegalArgumentException(
                    "EarlyStopping requires a validation set to monitor");
        }
        if (es != null && es.restoreBestWeights && snapshotParams() == null
                && es.monitor == EarlyStopping.Monitor.VAL_ACC) {
            // snapshot supported check happens lazily; VAL_ACC fine either way
        }

        int n = trainY.length;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Random shuffleRnd = new Random(shuffleSeed);
        float[] batch = new float[Math.min(batchSize, n) * inputElems()];
        int[] labels = new int[batch.length / inputElems()];

        History history = new History(epochs);
        float bestScore = es != null ? es.initialBest() : 0f;
        int bad = 0;
        float[] bestWeights = null;
        int ran = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            beforeEpoch(epoch);
            shuffle(shuffleRnd, order);
            double lossSum = 0;
            int steps = 0;
            for (int start = 0; start < n; start += labels.length) {
                int len = Math.min(labels.length, n - start);
                for (int i = 0; i < len; i++) {
                    int src = order[start + i] * inputElems();
                    System.arraycopy(trainX, src, batch, i * inputElems(), inputElems());
                    labels[i] = trainY[order[start + i]];
                }
                if (len == labels.length) {
                    lossSum += trainBatch(batch, labels) * len;
                } else {
                    // trim stale rows from previous iterations of the scratch
                    // buffers so the final short batch trains only on fresh data
                    float[] bx = java.util.Arrays.copyOf(batch, len * inputElems());
                    int[] ly = java.util.Arrays.copyOf(labels, len);
                    lossSum += trainBatch(bx, ly) * len;
                }
                steps += len;
            }
            float trainLoss = (float) (lossSum / steps);
            history.trainLoss[epoch] = trainLoss;

            if (hasVal) {
                float[] eval = evaluate(valX, valY);
                history.valLoss[epoch] = eval[0];
                history.valAcc[epoch] = eval[1];
            }
            if (verbose) {
                StringBuilder line = new StringBuilder(String.format(
                        "Epoch %d/%d - loss: %.4f", epoch + 1, epochs, trainLoss));
                if (hasVal) {
                    line.append(String.format(" - val_loss: %.4f - val_acc: %.4f",
                            history.valLoss[epoch], history.valAcc[epoch]));
                }
                System.out.println(line);
            }
            ran = epoch + 1;

            if (es != null) {
                float score = es.monitor == EarlyStopping.Monitor.VAL_LOSS
                        ? history.valLoss[epoch] : history.valAcc[epoch];
                if (es.improves(score, bestScore)) {
                    bestScore = score;
                    es.bestScore = score;
                    es.bestEpoch = epoch;
                    bad = 0;
                    if (es.restoreBestWeights) {
                        bestWeights = snapshotParams();
                    }
                } else {
                    bad++;
                    if (bad >= es.patience) {
                        es.stoppedEarly = true;
                        if (verbose) {
                            System.out.printf("Early stopping at epoch %d - best %s %.4f at epoch %d%n",
                                    epoch + 1, es.monitor, bestScore, es.bestEpoch + 1);
                        }
                        break;
                    }
                }
            }
        }
        history.epochsRun = ran;
        if (bestWeights != null) {
            restoreParams(bestWeights);
        }
        return history;
    }

    /** Called at the start of each epoch (LR schedules hook in here). */
    protected void beforeEpoch(int epoch) {
    }

    /** Flat parameter snapshot for early-stopping restore; null if unsupported. */
    protected float[] snapshotParams() {
        return null;
    }

    /** Writes back a snapshot taken by {@link #snapshotParams()}. */
    protected void restoreParams(float[] params) {
        throw new UnsupportedOperationException("restore not supported");
    }

    private float[] evaluate(float[] x, int[] y) {
        int n = y.length;
        float[] probs = predict(x);
        double lossSum = 0;
        int correct = 0;
        for (int i = 0; i < n; i++) {
            int truth = y[i];
            lossSum += -Math.log(probs[i * numClasses() + truth] + 1e-12);
            if (argmax(probs, i * numClasses(), numClasses()) == truth) {
                correct++;
            }
        }
        return new float[]{(float) (lossSum / n), (float) correct / n};
    }

    static int argmax(float[] row, int off, int len) {
        int best = 0;
        for (int i = 1; i < len; i++) {
            if (row[off + i] > row[off + best]) {
                best = i;
            }
        }
        return best;
    }

    private static void shuffle(Random rnd, int[] a) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    private void validateDataset(String name, float[] x, int[] y) {
        if (y == null || y.length == 0 || x == null || x.length != y.length * inputElems()) {
            throw new IllegalArgumentException(name + "X length must equal "
                    + name + "Y.length * inputElems (" + inputElems() + ")");
        }
        for (int label : y) {
            if (label < 0 || label >= numClasses()) {
                throw new IllegalArgumentException(
                        name + "Y contains label outside [0," + numClasses() + ")");
            }
        }
    }

    /** Per-epoch metrics recorded by {@link #fit}. */
    public static final class History {
        public final float[] trainLoss;
        public final float[] valLoss;
        public final float[] valAcc;
        private int epochsRun;

        History(int epochs) {
            trainLoss = new float[epochs];
            valLoss = new float[epochs];
            valAcc = new float[epochs];
            java.util.Arrays.fill(valLoss, Float.NaN);
            java.util.Arrays.fill(valAcc, Float.NaN);
            this.epochsRun = epochs;
        }

        public int epochs() { return trainLoss.length; }

        /** Epochs actually executed (smaller than {@link #epochs()} on early stop). */
        public int epochsRun() { return epochsRun; }
        public float finalTrainLoss() { return trainLoss[epochsRun - 1]; }
        public float finalValLoss() { return valLoss[epochsRun - 1]; }
        public float finalValAcc() { return valAcc[epochsRun - 1]; }
    }
}
