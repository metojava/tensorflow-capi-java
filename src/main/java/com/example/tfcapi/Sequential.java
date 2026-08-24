package com.example.tfcapi;

import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

/**
 * A Keras-style sequential classifier built from composable layers
 * ({@link Input}/{@link Input4D}, {@link Dense}, {@link Flatten},
 * {@link ReLU}, {@link Conv2D}, {@link MaxPool2D}). The graph is constructed
 * through the generic TensorFlow C API bindings in Java &mdash; adding a new
 * layer type requires no native changes.
 *
 * <p>The model must start with an {@link Input} or {@link Input4D} and end
 * with a {@link Dense} (its units are the class count); training minimizes
 * sparse-softmax cross-entropy with plain SGD, gradients via
 * {@code TF_AddGradients}.</p>
 *
 * <pre>{@code
 * try (Sequential net = new Sequential(0.01f,
 *         new Input4D(28, 28, 1),
 *         new Conv2D(8, 3, 3, 1, 1, Padding.SAME), new ReLU(),
 *         new MaxPool2D(2, 2, 2, 2, Padding.VALID),
 *         new Flatten(),
 *         new Dense(128), new ReLU(),
 *         new Dense(10))) {
 *     History h = net.fit(trainX, trainY, 5, 128, testX, testY, true);
 *     net.saveTo(Path.of("cnn.bin"));
 * }
 * }</pre>
 */
public final class Sequential extends AbstractClassifier {

    private static final int FORMAT_MAGIC = 0x5446534C; // "TFSL"
    private static final int FORMAT_VERSION = 2;

    /** Loss strategies supported by {@link Sequential}. */
    public enum Loss {
        /** Mean of sparse softmax cross-entropy between logits and class ids. */
        SPARSE_CROSS_ENTROPY,
        /** Mean squared error between softmax probabilities and one-hot targets. */
        SOFTMAX_MSE,
        /** Per-class sigmoid binary cross-entropy (multi-label style). */
        SIGMOID_CROSS_ENTROPY
    }

    private final NativeGraph graph;
    private final NativeSession session;

    private final long xPh, yPh, lossOp, probsOp;
    private final long[] paramOps;
    private final long[] paramTensors;
    private final long[][] paramShapes;
    private final long[] gradFetchOps;
    private final int[] gradFetchIdx;

    /** Auxiliary feeds (dropout masks, BN stats), in registration order. */
    private final long[] auxOps;
    private final long[][] auxShapes;
    private boolean training = false;   // predict-time mode for Dropout etc.

    private final long[] inputDims;   // per-sample dims from the Input layer
    private final int numClasses;
    private final Layer[] layerList;
    private final int[] layerParamStart;   // first param index owned by layer i
    private final int[] layerParamCount;
    private boolean[] paramFrozen;         // null until setTrainable is used
    private long[] zeroGrads;              // cached zero tensors for frozen params
    private LrSchedule lrSchedule;
    private Optimizer optimizer;
    private long shuffleSeedUnused;        // kept for layout stability
    private boolean open = true;

/** Creates a model trained with plain SGD at the given learning rate. */
    public Sequential(float learningRate, Layer... layers) {
        this(Optimizer.sgd(learningRate), layers);
    }

    /** Creates a model trained with the given optimizer (default loss = SPARSE_CROSS_ENTROPY). */
    public Sequential(Optimizer optimizer, Layer... layers) {
        this(optimizer, 42L, layers);
    }

    /** As {@link #Sequential(Optimizer, Layer...)} with reproducible weight init. */
    public Sequential(Optimizer optimizer, long initSeed, Layer... layers) {
        this(optimizer, Loss.SPARSE_CROSS_ENTROPY, 0f, initSeed, layers);
    }

    /** Creates a model with a loss strategy and optional label smoothing. */
    public Sequential(Optimizer optimizer, Loss loss, float labelSmoothing, Layer... layers) {
        this(optimizer, loss, labelSmoothing, 42L, layers);
    }

    /**
     * Full configuration: optimizer, loss strategy, label smoothing and the
     * weight-initialization seed (same seed + same data = identical training).
     */
    public Sequential(Optimizer optimizer, Loss loss, float labelSmoothing,
                      long initSeed, Layer... layers) {
        TensorFlow.load();
        validate(layers);
        this.optimizer = java.util.Objects.requireNonNull(optimizer, "optimizer");
        this.layerList = layers.clone();
        this.graph = new NativeGraph();
        try {
            GraphBuilder gb = new GraphBuilder(graph, initSeed);

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
            this.numClasses = (int) f.shape.last();

            this.yPh = gb.placeholder("y", NativeTF.DT_INT32, Shape.of(-1));
            long[] lp = buildLossAndProbs(gb, f, this.yPh, loss, labelSmoothing);
            this.lossOp = lp[0];
            this.probsOp = lp[1];

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

    /** Restores a checkpoint written by {@link #saveTo(Path)} (SGD). */
    public static Sequential loadFrom(java.nio.file.Path file,
                                      float learningRate) throws IOException {
        return loadFrom(file, Optimizer.sgd(learningRate));
    }

    /** Restores a checkpoint written by {@link #saveTo(Path)}. */
    public static Sequential loadFrom(java.nio.file.Path file,
                                      Optimizer optimizer) throws IOException {
        try (java.io.DataInputStream in = new java.io.DataInputStream(
                new java.io.BufferedInputStream(Files.newInputStream(file)))) {
            if (in.readInt() != FORMAT_MAGIC) {
                throw new IOException("not a Sequential checkpoint: " + file);
            }
            if (in.readInt() != FORMAT_VERSION) {
                throw new IOException("unsupported Sequential checkpoint version");
            }
            int[] spec = new int[in.readInt()];
            for (int i = 0; i < spec.length; i++) {
                spec[i] = in.readInt();
            }
            float[] params = new float[in.readInt()];
            for (int i = 0; i < params.length; i++) {
                params[i] = in.readFloat();
            }
            Sequential net = new Sequential(optimizer, decodeSpec(spec));
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
            for (int s : spec) {
                out.writeInt(s);
            }
            out.writeInt(params.length);
            for (float p : params) {
                out.writeFloat(p);
            }
        }
    }

    @Override
    public synchronized float trainBatch(float[] batchX, int[] batchY) {
        checkOpen();
        int n = requireBatch(batchX, batchY);

        long xT = NativeTF.allocTensorF32(withBatch(n), batchX);
        long yT = NativeTF.allocTensorI32(new long[]{n}, batchY);
        long[] auxT = allocAux(n, true, batchX);
        long[] fetched = runStep(xT, yT, true, auxT);
        NativeTF.deleteTensor(xT);
        NativeTF.deleteTensor(yT);
        deleteAll(auxT);

        float loss = NativeTF.tensorDataF32(fetched[0])[0];
        optimizer.apply(paramTensors, effectiveGrads(fetched));
        deleteAll(fetched);
        return loss;
    }

    /**
     * Freezes/unfreezes all parameters of one layer (index into the
     * constructor list; index 0 is the Input and is ignored). Frozen layers
     * receive zero gradients - their weights never change. Use for
     * fine-tuning: train with everything frozen except the head, then unfreeze.
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
    public synchronized void unfreezeAll() {
        paramFrozen = null;
    }

    /** True if no parameter is frozen. */
    public synchronized boolean fullyTrainable() { return paramFrozen == null; }

    /** Current LR multiplier (schedule diagnostics). */
    public synchronized float currentLrMultiplier() {
        return optimizer != null ? optimizer.lrMultiplier() : 1f;
    }

    /** Snapshot of one layer's parameters (index into constructor list). */
    public synchronized float[] layerParams(int layerIndex) {
        checkOpen();
        if (layerIndex <= 0 || layerIndex >= layerParamStart.length) {
            throw new IllegalArgumentException("layerIndex out of range: " + layerIndex);
        }
        int start = layerParamStart[layerIndex];
        int total = 0;
        for (int i = start; i < start + layerParamCount[layerIndex]; i++) {
            total += (int) product(paramShapes[i]);
        }
        float[] out = new float[total];
        int off = 0;
        for (int i = start; i < start + layerParamCount[layerIndex]; i++) {
            float[] d = NativeTF.tensorDataF32(paramTensors[i]);
            System.arraycopy(d, 0, out, off, d.length);
            off += d.length;
        }
        return out;
    }

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

    @Override
    protected void beforeEpoch(int epoch) {
        if (lrSchedule != null && optimizer != null) {
            optimizer.setLrMultiplier(lrSchedule.fractionAt(epoch));
        }
    }

    @Override
    protected float[] snapshotParams() { return getParams(); }

    @Override
    protected void restoreParams(float[] params) { setParams(params); }

    /**
     * Scales the learning rate per epoch by {@code schedule.fractionAt(epoch)}
     * without touching optimizer state (momentum/Adam moments survive).
     */
    public void setLrSchedule(LrSchedule schedule) { this.lrSchedule = schedule; }

    /**
     * Inference mode for {@link #predict}: by default {@code false}
     * (dropout disabled, BatchNorm uses running averages). Set to
     * {@code true} to evaluate with training-time behavior.
     */
    public synchronized void setTraining(boolean t) {
        this.training = t;
    }

    @Override
    public synchronized float[] predict(float[] batchX) {
        checkOpen();
        int n = requireBatch(batchX, null);

        long xT = NativeTF.allocTensorF32(withBatch(n), batchX);
        long[] auxT = allocAux(n, training, batchX);
        long[] fetched = runStep(xT, 0, false, auxT);
        NativeTF.deleteTensor(xT);
        deleteAll(auxT);
        float[] probs = NativeTF.tensorDataF32(fetched[0]);
        deleteAll(fetched);
        return probs;
    }

    /** Concatenated values of all trainable parameters (checkpoint payload). */
    public synchronized float[] getParams() {
        checkOpen();
        int total = 0;
        for (long[] shape : paramShapes) {
            total += product(shape);
        }
        float[] out = new float[total];
        int off = 0;
        for (int i = 0; i < paramTensors.length; i++) {
            float[] data = NativeTF.tensorDataF32(paramTensors[i]);
            System.arraycopy(data, 0, out, off, data.length);
            off += data.length;
        }
        return out;
    }

    /** Replaces all trainable parameters (length must match {@link #getParams()}). */
    public synchronized void setParams(float[] params) {
        checkOpen();
        int total = 0;
        for (long[] shape : paramShapes) {
            total += product(shape);
        }
        if (params == null || params.length != total) {
            throw new IllegalArgumentException(
                    "expected " + total + " parameter values, got "
                            + (params == null ? 0 : params.length));
        }
        int off = 0;
        for (int i = 0; i < paramTensors.length; i++) {
            int len = (int) product(paramShapes[i]);
            NativeTF.deleteTensor(paramTensors[i]);
            paramTensors[i] = NativeTF.allocTensorF32(
                    paramShapes[i], java.util.Arrays.copyOfRange(params, off, off + len));
            off += len;
        }
    }

    public int[] spec() {
        List<Integer> list = new ArrayList<>();
        for (Layer l : layerList) {
            for (int s : l.spec()) {
                list.add(s);
            }
        }
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = list.get(i);
        }
        return out;
    }

    /** Replaces the optimizer (state is re-initialized; weights are kept). */
    public synchronized void setOptimizer(Optimizer opt) {
        checkOpen();
        java.util.Objects.requireNonNull(opt, "optimizer");
        optimizer.close();
        opt.init(paramShapes);
        optimizer = opt;
    }

    @Override
    public int inputElems() {
        return (int) product(inputDims);
    }

    @Override
    public int numClasses() { return numClasses; }

    public long[] inputDims() { return inputDims.clone(); }

    @Override
    public synchronized void close() {
        if (!open) {
            return;
        }
        open = false;
        optimizer.close();
        for (long t : paramTensors) {
            NativeTF.deleteTensor(t);
        }
        if (zeroGrads != null) {
            for (long z : zeroGrads) {
                if (z != 0) NativeTF.deleteTensor(z);
            }
            zeroGrads = null;
        }
        session.close();
        graph.close();
    }

    /** Builds the loss op and the matching predict-time probability op. */
    private long[] buildLossAndProbs(GraphBuilder gb, Flow f, long yPh,
                                     Loss loss, float labelSmoothing) {
        long oneHot = graph.op("OneHot", gb.unique("onehot"))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("TI", NativeTF.DT_INT32)
                .attrInt("axis", -1)
                .input(yPh)
                .input(gb.constI32Vec("num_classes", numClasses))
                .input(gb.constF32("on_value", 1.0f))
                .input(gb.constF32("off_value", 0.0f))
                .build();
        oneHot = gb.unary("StopGradient", "onehot_sg", oneHot);
        // Label smoothing: oneHot <- oneHot * (1 - eps) + eps / K
        if (labelSmoothing > 0f && labelSmoothing < 1f) {
            oneHot = gb.binary("Mul", "ls_mul", oneHot, gb.constF32("ls_scale", 1f - labelSmoothing));
            oneHot = gb.binary("Add", "ls_add", oneHot,
                    gb.constF32("ls_shift", labelSmoothing / numClasses));
        }
        switch (loss) {
            case SPARSE_CROSS_ENTROPY: {
                long p = gb.unary("Softmax", "loss_softmax", f.op);
                long logP = gb.unary("Log", "loss_log",
                        gb.binary("Add", "loss_eps", p, gb.constF32("loss_eps_c", 1e-7f)));
                long nll = gb.unary("Neg", "loss_neg",
                        reduce(gb, "Sum", gb.binary("Mul", "loss_ce", oneHot, logP), 1));
                return new long[]{reduceAll(gb, "Mean", nll),
                        gb.unary("Softmax", "probs", f.op)};
            }
            case SOFTMAX_MSE: {
                long p = gb.unary("Softmax", "loss_softmax", f.op);
                long diff = gb.binary("Sub", "loss_diff", p, oneHot);
                long sq = gb.binary("Mul", "loss_sq", diff, diff);
                return new long[]{reduceAll(gb, "Mean", sq),
                        gb.unary("Softmax", "probs", f.op)};
            }
            case SIGMOID_CROSS_ENTROPY: {
                long p = gb.unary("Sigmoid", "loss_sigmoid", f.op);
                float eps = 1e-7f;
                long c1 = gb.constF32("loss_one", 1f);
                long t1 = gb.binary("Mul", "loss_bce_t1", oneHot,
                        gb.unary("Log", "loss_logp",
                                gb.binary("Add", "loss_epsp", p, gb.constF32("loss_eps1", eps))));
                long t2 = gb.binary("Mul", "loss_bce_t2",
                        gb.binary("Sub", "loss_1my", c1, oneHot),
                        gb.unary("Log", "loss_log1p",
                                gb.binary("Add", "loss_eps1p",
                                        gb.binary("Sub", "loss_1mp", c1, p),
                                        gb.constF32("loss_eps2", eps))));
                long nll = gb.unary("Neg", "loss_neg",
                        reduce(gb, "Sum", gb.binary("Add", "loss_bce", t1, t2), 1));
                return new long[]{reduceAll(gb, "Mean", nll),
                        gb.unary("Sigmoid", "probs", f.op)};
            }
            default:
                throw new IllegalStateException("unhandled loss " + loss);
        }
    }

    /** Reduction over one axis (keep_dims=false). */
    private long reduce(GraphBuilder gb, String type, long in, int axis) {
        return gb.graph.op(type, gb.unique("reduce_" + type))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tidx", NativeTF.DT_INT32)
                .input(in)
                .input(gb.constInt32Tensor("reduce_axes", new int[]{axis}, new long[]{1}))
                .build();
    }

    /** Reduction over all axes, producing a scalar (keep_dims=false). */
    private long reduceAll(GraphBuilder gb, String type, long in) {
        return gb.graph.op(type, gb.unique("reduceall_" + type))
                .attrType("T", NativeTF.DT_FLOAT)
                .attrType("Tidx", NativeTF.DT_INT32)
                .input(in)
                .input(gb.constInt32Tensor("reduce_axes0", new int[]{0}, new long[]{}))
                .build();
    }

    /* ------------------------------ internals ----------------------------- */
    private static long[] inputDimsOf(Layer first) {
        if (first instanceof Input) {
            return ((Input) first).dims();
        }
        if (first instanceof Input3D) {
            return ((Input3D) first).dims();
        }
        return ((Input4D) first).dims();
    }

    private static void validate(Layer... layers) {
        if (layers == null || layers.length < 2) {
            throw new IllegalArgumentException(
                    "Sequential needs at least an Input and a Dense layer");
        }
        if (!(layers[0] instanceof Input) && !(layers[0] instanceof Input4D)
                && !(layers[0] instanceof Input3D)) {
            throw new IllegalArgumentException(
                    "first layer must be Input, Input3D or Input4D, got "
                            + layers[0]);
        }
        if (!(layers[layers.length - 1] instanceof Dense)) {
            throw new IllegalArgumentException(
                    "last layer must be Dense(classes), got "
                            + layers[layers.length - 1]);
        }
        for (int i = 1; i < layers.length; i++) {
            if (layers[i] instanceof Input || layers[i] instanceof Input4D
                    || layers[i] instanceof Input3D) {
                throw new IllegalArgumentException(
                        "only the first layer may be an Input: " + layers[i]);
            }
        }
    }

    private synchronized long[] runStep(long xT, long yT, boolean withGrads,
                                        long[] auxTensors) {
        int base = (withGrads ? 2 : 1) + paramOps.length + auxOps.length;
        long[] feedOps = new long[base];
        int[] feedIdx = new int[base];
        long[] feedTs = new long[base];
        int f = 0;
        feedOps[f] = xPh;
        feedTs[f] = xT;
        f++;
        if (withGrads) {
            feedOps[f] = yPh;
            feedTs[f] = yT;
            f++;
        }
        System.arraycopy(paramOps, 0, feedOps, f, paramOps.length);
        System.arraycopy(paramTensors, 0, feedTs, f, paramTensors.length);
        f += paramOps.length;
        if (auxOps.length > 0) {
            System.arraycopy(auxOps, 0, feedOps, f, auxOps.length);
            System.arraycopy(auxTensors, 0, feedTs, f, auxTensors.length);
        }

        int nFetch = 1 + (withGrads ? paramOps.length : 0);
        long[] fetchOps = new long[nFetch];
        int[] fetchIdx = new int[nFetch];
        fetchOps[0] = withGrads ? lossOp : probsOp;
        if (withGrads) {
            System.arraycopy(gradFetchOps, 0, fetchOps, 1, gradFetchOps.length);
            System.arraycopy(gradFetchIdx, 0, fetchIdx, 1, gradFetchIdx.length);
        }
        return session.run(feedOps, feedIdx, feedTs, fetchOps, fetchIdx);
    }

    /** Asks each layer to fill its aux buffers and allocates the tensors. */
    private long[] allocAux(int n, boolean trainingMode, float[] batchX) {
        long[] handles = new long[auxOps.length];
        if (auxOps.length == 0) {
            return handles;
        }
        float[][] bufs = new float[auxOps.length][];
        int li = 0;
        for (Layer l : layerList) {
            if (l.auxCount() == 0) {
                continue;
            }
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
        return handles;
    }

    /**
     * Aux buffer length. Shapes containing -1 are per-row (batch dim scales
     * with n); shapes without -1 are global tensors (e.g. BatchNorm [F]
     * statistics shared across the batch).
     */
    private static int auxLen(long[] shape, int n) {
        boolean hasBatch = false;
        int p = 1;
        for (long d : shape) {
            if (d < 0) {
                hasBatch = true;
            } else {
                p *= d;
            }
        }
        return hasBatch ? p * n : p;
    }

    /** Replaces any -1 batch dim with the concrete {@code n}. */
    private static long[] auxConcreteShape(long[] shape, int n) {
        long[] out = shape.clone();
        for (int i = 0; i < out.length; i++) {
            if (out[i] < 0) {
                out[i] = n;
            }
        }
        return out;
    }

    private int requireBatch(float[] batchX, int[] batchY) {
        int elems = inputElems();
        if (batchX == null || batchX.length == 0 || batchX.length % elems != 0) {
            throw new IllegalArgumentException(
                    "batchX length must be a positive multiple of " + elems);
        }
        int n = batchX.length / elems;
        if (batchY != null && batchY.length != n) {
            throw new IllegalArgumentException(
                    "batchY length must equal the number of samples (" + n + ")");
        }
        return n;
    }

    private long[] withBatch(int n) {
        long[] shape = new long[1 + inputDims.length];
        shape[0] = n;
        System.arraycopy(inputDims, 0, shape, 1, inputDims.length);
        return shape;
    }

    private void deleteAll(long[] handles) {
        for (long h : handles) {
            NativeTF.deleteTensor(h);
        }
    }

    private void checkOpen() {
        if (!open) {
            throw new IllegalStateException("model is closed");
        }
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

    private static Layer[] decodeSpec(int[] spec) {
        List<Layer> layers = new ArrayList<>();
        int[] pos = {0};
        while (pos[0] < spec.length) {
            layers.add(Layer.decode(spec, pos));
        }
        return layers.toArray(new Layer[0]);
    }
}
