package com.example.tfcapi;

import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Keras-style <em>regression</em> model: predicts continuous float targets
 * instead of class ids. Architecturally identical to {@link Sequential}
 * (same composable {@link Layer} set) but the label placeholder is
 * {@code DT_FLOAT}, the training objective is mean-squared error and the
 * prediction op returns raw network outputs - no softmax.
 *
 * <p>The last layer should be a single-unit {@code Dense} (multi-output
 * regression also works; targets are then flat {@code rows * outputs}).
 * Typical use:</p>
 *
 * <pre>{@code
 * try (Regressor net = new Regressor(Optimizer.adam(0.01f), 42L,
 *         new Input(4),
 *         new Dense(24), new ReLU(),
 *         new Dense(1))) {
 *     net.setShuffleSeed(42L);
 *     RegressionHistory h = net.fit(trainX, trainY, 200, 32, testX, testY, true);
 *     float[] preds = net.predict(testX);   // one float per row
 *     net.saveTo(Path.of("model.bin"));
 * }
 * }</pre>
 */
public final class Regressor implements AutoCloseable {

    private static final int FORMAT_MAGIC = 0x54465247; // "TFRG"
    private static final int FORMAT_VERSION = 1;

    private final NativeGraph graph;
    private final NativeSession session;

    private final long xPh, yPh, lossOp, predOp;
    private final long[] paramOps;
    private final long[] paramTensors;
    private final long[][] paramShapes;
    private final long[] gradFetchOps;
    private final int[] gradFetchIdx;

    /** Auxiliary feeds (dropout masks, BN stats), in registration order. */
    private final long[] auxOps;
    private final long[][] auxShapes;
    private boolean training = false;

    private final long[] inputDims;
    private final int outputs;
    private final Layer[] layerList;
    private final int[] layerParamStart;
    private final int[] layerParamCount;
    private boolean[] paramFrozen;
    private long[] zeroGrads;
    private LrSchedule lrSchedule;
    private Optimizer optimizer;
    private long shuffleSeed = 42L;
    private boolean open = true;

    /** Creates a regressor trained with the given optimizer (seeded init). */
    public Regressor(Optimizer optimizer, Layer... layers) {
        this(optimizer, 42L, layers);
    }

    /** Creates a regressor with reproducible weight initialization. */
    public Regressor(Optimizer optimizer, long seed, Layer... layers) {
        TensorFlow.load();
        validate(layers);
        this.optimizer = java.util.Objects.requireNonNull(optimizer, "optimizer");
        this.layerList = layers.clone();
        this.graph = new NativeGraph();
        try {
            GraphBuilder gb = new GraphBuilder(graph, seed);

            long[] inDims = inputDimsOf(layers[0]);
            this.inputDims = inDims.clone();
            this.xPh = gb.placeholder("x", NativeTF.DT_FLOAT,
                    Shape.of(prepend(-1, inDims)));

            Flow f = new Flow(xPh, Shape.of(prepend(-1, inDims)));
            this.layerParamStart = new int[layers.length];
            this.layerParamCount = new int[layers.length];
            for (int i = 1; i < layers.length; i++) {
                layerParamStart[i] = gb.paramCount();
                f = layers[i].build(gb, f);
                layerParamCount[i] = gb.paramCount() - layerParamStart[i];
            }
            this.outputs = (int) f.shape.last();

            this.yPh = gb.placeholder("y", NativeTF.DT_FLOAT,
                    Shape.of(-1, outputs));
            // loss = mean((pred - y)^2)
            long diff = gb.binary("Sub", "loss_diff", f.op, yPh);
            long sq = gb.binary("Mul", "loss_sq", diff, diff);
            this.lossOp = reduceAll(gb, "Mean", sq);
            this.predOp = f.op;

            this.paramOps = gb.paramOps();
            this.paramTensors = gb.paramTensors();
            this.paramShapes = gb.paramShapes();

            int auxTotal = 0;
            for (Layer l : this.layerList) {
                auxTotal += l.auxCount();
            }
            this.auxOps = new long[auxTotal];
            this.auxShapes = new long[auxTotal][];
            int ai = 0;
            for (Layer l : this.layerList) {
                for (long ph : l.auxPhs()) {
                    auxOps[ai] = ph;
                    ai++;
                }
            }
            ai = 0;
            for (Layer l : this.layerList) {
                for (long[] sh : l.auxShapes()) {
                    auxShapes[ai] = sh;
                    ai++;
                }
            }

            long[] grads = graph.addGradients(lossOp, 0, paramOps);
            this.gradFetchOps = new long[paramOps.length];
            this.gradFetchIdx = new int[paramOps.length];
            for (int i = 0; i < paramOps.length; i++) {
                gradFetchOps[i] = grads[2 * i];
                gradFetchIdx[i] = (int) grads[2 * i + 1];
            }

            this.session = graph.newSession();
            this.optimizer.init(this.paramShapes);
        } catch (RuntimeException | Error e) {
            graph.close();
            optimizer.close();
            throw e;
        }
    }

    /** Restores a checkpoint written by {@link #saveTo(java.nio.file.Path)}. */
    public static Regressor loadFrom(java.nio.file.Path file,
                                     Optimizer optimizer) throws IOException {
        try (java.io.DataInputStream in = new java.io.DataInputStream(
                new java.io.BufferedInputStream(Files.newInputStream(file)))) {
            if (in.readInt() != FORMAT_MAGIC) {
                throw new IOException("not a Regressor checkpoint: " + file);
            }
            if (in.readInt() != FORMAT_VERSION) {
                throw new IOException("unsupported Regressor checkpoint version");
            }
            int[] spec = new int[in.readInt()];
            for (int i = 0; i < spec.length; i++) spec[i] = in.readInt();
            float[] params = new float[in.readInt()];
            for (int i = 0; i < params.length; i++) params[i] = in.readFloat();
            Regressor net = new Regressor(optimizer, decodeSpec(spec));
            net.setParams(params);
            return net;
        }
    }

    /** Writes layer architecture + weights to {@code file}. */
    public void saveTo(java.nio.file.Path file) throws IOException {
        checkOpen();
        int[] spec = spec();
        float[] params = getParams();
        Files.createDirectories(file.toAbsolutePath().getParent());
        try (java.io.DataOutputStream out = new java.io.DataOutputStream(
                new java.io.BufferedOutputStream(Files.newOutputStream(file)))) {
            out.writeInt(FORMAT_MAGIC);
            out.writeInt(FORMAT_VERSION);
            out.writeInt(spec.length);
            for (int s : spec) out.writeInt(s);
            out.writeInt(params.length);
            for (float p : params) out.writeFloat(p);
        }
    }

    /**
     * Keras-style training loop over flat float targets. Pass
     * {@code null}/{@code null} validation data to skip evaluation.
     * Deterministic when {@link #setShuffleSeed(long)} is fixed.
     */
    public synchronized RegressionHistory fit(float[] trainX, float[] trainY,
                                              int epochs, int batchSize,
                                              float[] valX, float[] valY,
                                              boolean verbose) {
        return fit(trainX, trainY, epochs, batchSize, valX, valY, verbose, null);
    }

    /**
     * Full loop with optional {@link EarlyStopping} (VAL_LOSS only) and
     * optional best-weight restore.
     */
    public synchronized RegressionHistory fit(float[] trainX, float[] trainY,
                                              int epochs, int batchSize,
                                              float[] valX, float[] valY,
                                              boolean verbose,
                                              EarlyStopping es) {
        checkOpen();
        requireData("train", trainX, trainY);
        if (epochs <= 0 || batchSize <= 0) {
            throw new IllegalArgumentException("epochs and batchSize must be positive");
        }
        boolean hasVal = valX != null || valY != null;
        if (hasVal) {
            requireData("val", valX, valY);
        } else if (es != null) {
            throw new IllegalArgumentException(
                    "EarlyStopping requires a validation set to monitor");
        }

        int n = trainY.length / outputs;
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Random rnd = new Random(shuffleSeed);

        int batchRows = Math.min(batchSize, n);
        float[] bx = new float[batchRows * inputElems()];
        float[] by = new float[batchRows * outputs];

        RegressionHistory history = new RegressionHistory(epochs);
        float bestScore = es != null ? Float.POSITIVE_INFINITY : 0f;
        int bad = 0;
        float[] bestWeights = null;
        boolean trackBest = false;
        int ran = 0;

        for (int epoch = 0; epoch < epochs; epoch++) {
            beforeEpoch(epoch);
            shuffle(rnd, order);
            double lossSum = 0;
            int seen = 0;
            for (int start = 0; start < n; start += batchRows) {
                int len = Math.min(batchRows, n - start);
                for (int i = 0; i < len; i++) {
                    int src = order[start + i];
                    System.arraycopy(trainX, src * inputElems(), bx, i * inputElems(), inputElems());
                    System.arraycopy(trainY, src * outputs, by, i * outputs, outputs);
                }
                if (len == batchRows) {
                    lossSum += trainBatch(bx, by, len) * len;
                } else {
                    // exact-size copies: trainBatch validates buffer shapes
                    float[] bx2 = java.util.Arrays.copyOf(bx, len * inputElems());
                    float[] by2 = java.util.Arrays.copyOf(by, len * outputs);
                    lossSum += trainBatch(bx2, by2, len) * len;
                }
                seen += len;
            }
            history.trainLoss[epoch] = (float) (lossSum / seen);
            if (hasVal) {
                history.valLoss[epoch] = mse(valX, valY);
            }
            if (verbose) {
                StringBuilder line = new StringBuilder(String.format(
                        "Epoch %d/%d - loss: %.5f", epoch + 1, epochs, history.trainLoss[epoch]));
                if (hasVal) {
                    line.append(String.format(" - val_loss: %.5f", history.valLoss[epoch]));
                }
                System.out.println(line);
            }
            ran = epoch + 1;

            if (es != null) {
                if (es.monitor != EarlyStopping.Monitor.VAL_LOSS) {
                    throw new IllegalArgumentException(
                            "Regressor only supports EarlyStopping.valLoss");
                }
                float score = history.valLoss[epoch];
                if (score < bestScore - es.minDelta) {
                    bestScore = score;
                    es.bestScore = score;
                    es.bestEpoch = epoch;
                    bad = 0;
                    if (es.restoreBestWeights) {
                        bestWeights = snapshotParams();
                        trackBest = true;
                    }
                } else {
                    bad++;
                    if (bad >= es.patience) {
                        es.stoppedEarly = true;
                        if (verbose) {
                            System.out.printf("Early stopping at epoch %d - best val_loss %.5f at epoch %d%n",
                                    epoch + 1, bestScore, es.bestEpoch + 1);
                        }
                        break;
                    }
                }
            }
        }
        history.epochsRun = ran;
        if (trackBest && bestWeights != null) {
            restoreParams(bestWeights);
        }
        return history;
    }

    private void beforeEpoch(int epoch) {
        if (lrSchedule != null && optimizer != null) {
            optimizer.setLrMultiplier(lrSchedule.fractionAt(epoch));
        }
    }

    private float[] snapshotParams() { return getParams(); }

    private void restoreParams(float[] params) { setParams(params); }

    /** Scales the learning rate per epoch; optimizer state is preserved. */
    public void setLrSchedule(LrSchedule schedule) { this.lrSchedule = schedule; }

    /**
     * Freezes/unfreezes all parameters of one layer (index into the
     * constructor list). Frozen layers receive zero gradients.
     */
    public synchronized void setTrainable(int layerIndex, boolean trainable) {
        checkOpen();
        if (layerIndex <= 0 || layerIndex >= layerParamStart.length) {
            throw new IllegalArgumentException("layerIndex out of range: " + layerIndex);
        }
        if (paramFrozen == null) {
            paramFrozen = new boolean[paramOps.length];
        }
        int start = layerParamStart[layerIndex];
        for (int i = 0; i < layerParamCount[layerIndex]; i++) {
            paramFrozen[start + i] = !trainable;
        }
    }

    /** Unfreezes every layer. */
    public synchronized void unfreezeAll() { paramFrozen = null; }

    private long[] effectiveGrads(long[] fetched) {
        long[] grads = java.util.Arrays.copyOfRange(fetched, 1, fetched.length);
        if (paramFrozen == null) {
            return grads;
        }
        if (zeroGrads == null || zeroGrads.length != paramShapes.length) {
            zeroGrads = new long[paramShapes.length];
        }
        for (int i = 0; i < grads.length; i++) {
            if (paramFrozen[i]) {
                if (zeroGrads[i] == 0) {
                    zeroGrads[i] = NativeTF.allocTensorF32(paramShapes[i],
                            new float[(int) product(paramShapes[i])]);
                }
                grads[i] = zeroGrads[i];
            }
        }
        return grads;
    }

    /** Convenience overload without a validation split. */
    public RegressionHistory fit(float[] trainX, float[] trainY,
                                 int epochs, int batchSize) {
        return fit(trainX, trainY, epochs, batchSize, null, null, true);
    }

    /** One SGD step over a batch; returns the mean squared error. */
    public synchronized float trainBatch(float[] batchX, float[] batchY, int rows) {
        checkOpen();
        if (rows <= 0 || batchX.length != rows * inputElems()
                || batchY.length != rows * outputs) {
            throw new IllegalArgumentException("batch shapes must be [" + rows
                    + " x " + inputElems() + "] and [" + rows + " x " + outputs + "]");
        }
        long xT = NativeTF.allocTensorF32(withBatch(rows, inputDims), batchX);
        long yT = NativeTF.allocTensorF32(new long[]{rows, outputs}, batchY);
        long[] auxT = allocAux(rows, true, batchX);
        long[] fetched = runStep(xT, yT, true);
        NativeTF.deleteTensor(xT);
        NativeTF.deleteTensor(yT);
        deleteAll(auxT);

        float loss = NativeTF.tensorDataF32(fetched[0])[0];
        optimizer.apply(paramTensors, effectiveGrads(fetched));
        deleteAll(fetched);
        return loss;
    }

    /** Raw predictions; returns {@code rows * outputs} floats (no softmax). */
    public synchronized float[] predict(float[] batchX) {
        checkOpen();
        int n = requireBatchX(batchX);
        long xT = NativeTF.allocTensorF32(withBatch(n, inputDims), batchX);
        long[] auxT = allocAux(n, training, batchX);
        long[] fetched = runStep(xT, 0, false);
        NativeTF.deleteTensor(xT);
        deleteAll(auxT);
        float[] out = NativeTF.tensorDataF32(fetched[0]);
        deleteAll(fetched);
        return out;
    }

    /** Mean squared error of predictions vs targets on a whole dataset. */
    public synchronized float mse(float[] x, float[] y) {
        requireData("eval", x, y);
        float[] p = predict(x);
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            double d = p[i] - y[i];
            sum += d * d;
        }
        return (float) (sum / p.length);
    }

    /** Inference mode for Dropout/BatchNorm layers (default false). */
    public synchronized void setTraining(boolean t) { this.training = t; }

    /** Fixes the epoch-shuffle RNG for fully reproducible runs. */
    public synchronized void setShuffleSeed(long seed) { this.shuffleSeed = seed; }

    /** Replaces the optimizer (state re-initialized; weights kept). */
    public synchronized void setOptimizer(Optimizer opt) {
        checkOpen();
        java.util.Objects.requireNonNull(opt, "optimizer");
        optimizer.close();
        opt.init(paramShapes);
        optimizer = opt;
    }

    /** Concatenated values of all trainable parameters. */
    public synchronized float[] getParams() {
        checkOpen();
        int total = 0;
        for (long[] shape : paramShapes) total += product(shape);
        float[] out = new float[total];
        int off = 0;
        for (int i = 0; i < paramTensors.length; i++) {
            float[] data = NativeTF.tensorDataF32(paramTensors[i]);
            System.arraycopy(data, 0, out, off, data.length);
            off += data.length;
        }
        return out;
    }

    /** Replaces all trainable parameters (length must match). */
    public synchronized void setParams(float[] params) {
        checkOpen();
        int total = 0;
        for (long[] shape : paramShapes) total += product(shape);
        if (params == null || params.length != total) {
            throw new IllegalArgumentException("expected " + total + " params, got "
                    + (params == null ? 0 : params.length));
        }
        int off = 0;
        for (int i = 0; i < paramTensors.length; i++) {
            int len = (int) product(paramShapes[i]);
            NativeTF.deleteTensor(paramTensors[i]);
            paramTensors[i] = NativeTF.allocTensorF32(paramShapes[i],
                    java.util.Arrays.copyOfRange(params, off, off + len));
            off += len;
        }
    }

    public int[] spec() {
        List<Integer> list = new ArrayList<>();
        for (Layer l : layerList) {
            for (int s : l.spec()) list.add(s);
        }
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) out[i] = list.get(i);
        return out;
    }

    /** Input values per sample. */
    public int inputElems() { return (int) product(inputDims); }

    /** Target/prediction values per sample (1 for scalar regression). */
    public int outputs() { return outputs; }

    @Override
    public synchronized void close() {
        if (!open) return;
        open = false;
        optimizer.close();
        for (long t : paramTensors) NativeTF.deleteTensor(t);
        if (zeroGrads != null) {
            for (long z : zeroGrads) {
                if (z != 0) NativeTF.deleteTensor(z);
            }
            zeroGrads = null;
        }
        session.close();
        graph.close();
    }

    /* ------------------------------ internals ----------------------------- */

    private synchronized long[] runStep(long xT, long yT, boolean withGrads) {
        int base = (withGrads ? 2 : 1) + paramOps.length + auxOps.length;
        long[] feedOps = new long[base];
        int[] feedIdx = new int[base];
        long[] feedTs = new long[base];
        int fi = 0;
        feedOps[fi] = xPh; feedTs[fi] = xT; fi++;
        if (withGrads) { feedOps[fi] = yPh; feedTs[fi] = yT; fi++; }
        System.arraycopy(paramOps, 0, feedOps, fi, paramOps.length);
        System.arraycopy(paramTensors, 0, feedTs, fi, paramTensors.length);
        fi += paramOps.length;
        if (auxOps.length > 0) {
            System.arraycopy(auxOps, 0, feedOps, fi, auxOps.length);
            System.arraycopy(auxTensorsSnapshot(), 0, feedTs, fi, auxOps.length);
        }
        int nFetch = 1 + (withGrads ? paramOps.length : 0);
        long[] fetchOps = new long[nFetch];
        int[] fetchIdx = new int[nFetch];
        fetchOps[0] = withGrads ? lossOp : predOp;
        if (withGrads) {
            System.arraycopy(gradFetchOps, 0, fetchOps, 1, gradFetchOps.length);
            System.arraycopy(gradFetchIdx, 0, fetchIdx, 1, gradFetchIdx.length);
        }
        return session.run(feedOps, feedIdx, feedTs, fetchOps, fetchIdx);
    }

    private long[] auxTensorBuffer;   // set by allocAux before runStep(predict)

    private long[] auxTensorsSnapshot() { return auxTensorBuffer; }

    /** Allocates aux tensors and remembers them for the next predict run. */
    private long[] allocAux(int n, boolean trainingMode, float[] batchX) {
        long[] handles = new long[auxOps.length];
        if (auxOps.length == 0) return handles;
        float[][] bufs = new float[auxOps.length][];
        int li = 0;
        for (Layer l : layerList) {
            if (l.auxCount() == 0) continue;
            float[][] mine = new float[l.auxCount()][];
            for (int j = 0; j < l.auxCount(); j++) {
                mine[j] = new float[auxLen(auxShapes[li + j], n)];
            }
            l.supplyAux(mine, n, trainingMode, batchX);
            System.arraycopy(mine, 0, bufs, li, mine.length);
            li += l.auxCount();
        }
        for (int i = 0; i < bufs.length; i++) {
            handles[i] = NativeTF.allocTensorF32(
                    auxConcreteShape(auxShapes[i], n), bufs[i]);
        }
        auxTensorBuffer = handles;
        return handles;
    }

    private void requireData(String name, float[] x, float[] y) {
        if (x == null || y == null || y.length == 0
                || x.length != (y.length / outputs) * inputElems()) {
            throw new IllegalArgumentException(name + ": X length must equal "
                    + "(Y.length / outputs) * inputElems (" + inputElems() + ")");
        }
    }

    private int requireBatchX(float[] batchX) {
        int elems = inputElems();
        if (batchX == null || batchX.length == 0 || batchX.length % elems != 0) {
            throw new IllegalArgumentException(
                    "batchX length must be a positive multiple of " + elems);
        }
        return batchX.length / elems;
    }

    private static long[] withBatch(int n, long[] dims) {
        long[] shape = new long[1 + dims.length];
        shape[0] = n;
        System.arraycopy(dims, 0, shape, 1, dims.length);
        return shape;
    }

    private void deleteAll(long[] handles) {
        if (handles == null) return;
        for (long h : handles) NativeTF.deleteTensor(h);
    }

    private void checkOpen() {
        if (!open) throw new IllegalStateException("model is closed");
    }

    private static int auxLen(long[] shape, int n) {
        boolean hasBatch = false;
        int p = 1;
        for (long d : shape) {
            if (d < 0) hasBatch = true; else p *= d;
        }
        return hasBatch ? p * n : p;
    }

    private static long[] auxConcreteShape(long[] shape, int n) {
        long[] out = shape.clone();
        for (int i = 0; i < out.length; i++) {
            if (out[i] < 0) out[i] = n;
        }
        return out;
    }

    private static long product(long[] shape) {
        long p = 1;
        for (long d : shape) p *= d;
        return p;
    }

    private static long[] prepend(long v, long[] rest) {
        long[] out = new long[1 + rest.length];
        out[0] = v;
        System.arraycopy(rest, 0, out, 1, rest.length);
        return out;
    }

    private static long[] inputDimsOf(Layer first) {
        if (first instanceof Input) return ((Input) first).dims();
        if (first instanceof Input3D) return ((Input3D) first).dims();
        return ((Input4D) first).dims();
    }

    private static void validate(Layer... layers) {
        if (layers == null || layers.length < 2) {
            throw new IllegalArgumentException(
                    "Regressor needs at least an Input and a Dense layer");
        }
        if (!(layers[0] instanceof Input) && !(layers[0] instanceof Input4D)
                && !(layers[0] instanceof Input3D)) {
            throw new IllegalArgumentException(
                    "first layer must be Input, Input3D or Input4D, got " + layers[0]);
        }
        if (!(layers[layers.length - 1] instanceof Dense)) {
            throw new IllegalArgumentException(
                    "last layer must be Dense(outputs), got " + layers[layers.length - 1]);
        }
        for (int i = 1; i < layers.length; i++) {
            if (layers[i] instanceof Input || layers[i] instanceof Input4D
                    || layers[i] instanceof Input3D) {
                throw new IllegalArgumentException("only the first layer may be an Input");
            }
        }
    }

    private static void shuffle(Random rnd, int[] a) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = a[i]; a[i] = a[j]; a[j] = t;
        }
    }

    private static long reduceAll(GraphBuilder gb, String type, long in) {
        return gb.graph.op(type, gb.unique("reduceall_" + type))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tidx", NativeTF.DT_INT32)
                .input(in)
                .input(gb.constInt32Tensor("reduce_axes0", new int[]{0}, new long[]{}))
                .build();
    }

    private static Layer[] decodeSpec(int[] spec) {
        List<Layer> layers = new ArrayList<>();
        int[] pos = {0};
        while (pos[0] < spec.length) {
            layers.add(Layer.decode(spec, pos));
        }
        return layers.toArray(new Layer[0]);
    }

    /** Per-epoch losses recorded by {@link #fit}. */
    public static final class RegressionHistory {
        public final float[] trainLoss;
        public final float[] valLoss;
        private int epochsRun;

        RegressionHistory(int epochs) {
            trainLoss = new float[epochs];
            valLoss = new float[epochs];
            java.util.Arrays.fill(valLoss, Float.NaN);
            this.epochsRun = epochs;
        }

        public int epochs() { return trainLoss.length; }

        /** Epochs actually executed (smaller than {@link #epochs()} on early stop). */
        public int epochsRun() { return epochsRun; }
        public float finalTrainLoss() { return trainLoss[epochsRun - 1]; }
        public float finalValLoss() { return valLoss[epochsRun - 1]; }
    }
}
