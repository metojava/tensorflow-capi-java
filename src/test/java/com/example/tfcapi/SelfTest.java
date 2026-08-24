package com.example.tfcapi;

import java.util.Arrays;
import java.util.Random;

/**
 * Framework-free regression suite for the library. Run with:
 * <pre>
 *   javac -cp target/classes -d target/test-classes src/test/java/com/example/tfcapi/SelfTest.java
 *   java -cp target/classes;target/test-classes com.example.tfcapi.SelfTest
 * </pre>
 * (requires TF_CAPI_NATIVE_DIR pointing at the TensorFlow DLL folder)
 */
public final class SelfTest {

    private static int passed = 0, failed = 0;

    public static void main(String[] args) throws Exception {
        lossesLearnXor();
        regressorConverges();
        datasetsDeterministicSplit();
        earlyStoppingRestoresBest();
        freezeKeepsWeights();
        lrScheduleApplies();
        batchNorm2dClassifies();
        upsampleShape();
        selfAttentionClassifies();
        selfAttentionSaveLoad();
        denseNdClassifies();
        layerNormAttentionClassifies();
        embeddingNdFeedsLstm();
        positionalEncodingOrder();
        multiHeadAttentionClassifies();
        causalMaskTrains();
        crossAttentionRetrieves();
        saveLoadRoundTrips();
        System.out.printf("%n%d passed, %d failed%n", passed, failed);
        if (failed > 0) System.exit(1);
    }

    static void check(boolean ok, String name) {
        if (ok) { passed++; System.out.println("PASS " + name); }
        else { failed++; System.out.println("FAIL " + name); }
    }

    /** All three Sequential.Loss values must learn XOR. */
    static void lossesLearnXor() {
        float[] x = flat(new float[][]{{0,0},{0,1},{1,0},{1,1}});
        int[] y = {0,1,1,0};
        boolean all = true;
        for (Sequential.Loss loss : Sequential.Loss.values()) {
            try (Sequential net = new Sequential(Optimizer.adam(0.05f), loss, 0f,
                    new Input(2), new Dense(8), new ReLU(), new Dense(2))) {
                for (int i = 0; i < 500; i++) net.trainBatch(x, y);
                float[] p = net.predict(x);
                int correct = 0;
                for (int i = 0; i < 4; i++) {
                    if (Metrics.argmax(p, i * 2, 2) == y[i]) correct++;
                }
                all &= correct >= 3;
            }
        }
        check(all, "losses learn XOR");
    }

    static void regressorConverges() {
        Random rnd = new Random(7);
        int n = 256;
        float[] X = new float[n * 3], Y = new float[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 3; j++) X[i*3+j] = rnd.nextFloat()*4-2;
            Y[i] = 2f*X[i*3] - X[i*3+1] + (float)rnd.nextGaussian()*0.05f;
        }
        try (Regressor net = new Regressor(Optimizer.adam(0.02f), 42L,
                new Input(3), new Dense(32), new ReLU(), new Dense(1))) {
            var h = net.fit(X, Y, 400, 32);
            check(h.finalTrainLoss() < 0.05 && Math.abs(net.predict(new float[]{1,-1,0})[0] - 3f) < 0.3,
                    "regressor converges");
        }
    }

    static void datasetsDeterministicSplit() {
        int[][] a = Datasets.trainTestSplit(100, 0.3, 99L);
        int[][] b = Datasets.trainTestSplit(100, 0.3, 99L);
        check(Arrays.equals(a[0], b[0]) && Arrays.equals(a[1], b[1])
                        && a[0].length == 70 && a[1].length == 30,
                "datasets split deterministic");
    }

    static void earlyStoppingRestoresBest() {
        Random rnd = new Random(3);
        float[] tx = new float[600]; int[] ty = new int[300];
        float[] vx = new float[300]; int[] vy = new int[150];
        for (int i = 0; i < 300; i++) gen(rnd, tx, ty, i);
        for (int i = 0; i < 150; i++) gen(rnd, vx, vy, i);

        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 7L,
                new Input(2), new Dense(64), new ReLU(), new Dense(2))) {
            EarlyStopping es = EarlyStopping.valLoss(3, true);
            var h = net.fit(tx, ty, 200, 32, vx, vy, false, es);
            double now = valLoss(net, vx, vy);
            check(es.stoppedEarly && h.epochsRun() < h.epochs()
                            && Math.abs(now - es.bestScore) < 0.05,
                    "early stopping restores best");
        }
    }

    static void gen(Random rnd, float[] x, int[] y, int i) {
        int cls = i % 2;
        float c = cls == 0 ? -1.5f : 1.5f;
        x[i*2] = c + (float)rnd.nextGaussian()*0.8f;
        x[i*2+1] = c + (float)rnd.nextGaussian()*0.8f;
        y[i] = cls;
    }

    static double valLoss(Sequential net, float[] x, int[] y) {
        float[] p = net.predict(x);
        double sum = 0;
        for (int i = 0; i < y.length; i++)
            sum += -Math.log(Math.max(p[i*2+y[i]], 1e-12));
        return sum / y.length;
    }

    static void freezeKeepsWeights() {
        try (Sequential net = new Sequential(Optimizer.sgd(0.1f), 42L,
                new Input(2), new Dense(8), new Tanh(), new Dense(2))) {
            float[] x = flat(new float[][]{{0,0},{0,1},{1,0},{1,1}});
            int[] y = {0,1,1,0};
            for (int i = 0; i < 20; i++) net.trainBatch(x, y);
            net.setTrainable(1, false);
            float[] before = net.layerParams(1);
            for (int i = 0; i < 50; i++) net.trainBatch(x, y);
            check(Arrays.equals(before, net.layerParams(1)), "freeze keeps weights");
        }
    }

    static void lrScheduleApplies() {
        try (Sequential net = new Sequential(Optimizer.adam(0.02f), 42L,
                new Input(2), new Dense(8), new ReLU(), new Dense(2))) {
            net.setLrSchedule(LrSchedule.cosine(20, 0.05f));
            AbstractClassifier.History h = net.fit(
                    flat(new float[][]{{0,0},{0,1},{1,0},{1,1}}),
                    new int[]{0,1,1,0}, 20, 4, null, null, false);
            float m = net.currentLrMultiplier();
            check(h.epochsRun() == 20 && m <= 0.06f && m >= 0.04f,
                    "lr schedule applied");
        }
    }

    static void batchNorm2dClassifies() {
        Random rnd = new Random(23);
        int n = 200, sz = 8;
        float[] X = new float[n*sz*sz];
        int[] Y = new int[n];
        for (int i = 0; i < n; i++) {
            float lvl = i % 2 == 0 ? 0.75f : 0.15f;
            for (int p = 0; p < sz*sz; p++)
                X[i*sz*sz+p] = (float)Math.min(1, Math.max(0, lvl + rnd.nextGaussian()*0.15));
            Y[i] = i % 2;
        }
        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 5L,
                new Input4D(sz, sz, 1),
                new Conv2D(4, 3, 3, 1, 1, Padding.SAME),
                new BatchNorm2D(), new ReLU(),
                new MaxPool2D(2,2,2,2,Padding.VALID),
                new Flatten(), new Dense(2))) {
            net.fit(X, Y, 40, 32, null, null, false);
            float[] p = net.predict(X);
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (Metrics.argmax(p, i*2, 2) == Y[i]) correct++;
            check(correct / (double) n > 0.9, "BatchNorm2D CNN classifies");
        }
    }

    static void upsampleShape() {
        try (NativeGraph g = new NativeGraph()) {
            GraphBuilder gb = new GraphBuilder(g);
            long xPh = gb.placeholder("x", NativeTF.DT_FLOAT,
                    Shape.of(-1, 1, 4, 4));
            Flow f = new Upsample2D(2).build(gb, new Flow(xPh, Shape.of(-1, 1, 4, 4)));
            check(f.shape.dims[2] == 8 && f.shape.dims[3] == 8,
                    "Upsample2D doubles spatial dims");
            // forward run to prove the native op executes
            NativeSession s = g.newSession();
            float[] px = new float[16];
            long xt = NativeTF.allocTensorF32(new long[]{1,1,4,4}, px);
            long[] got = s.run(new long[]{xPh}, new int[]{0}, new long[]{xt},
                    new long[]{f.op}, new int[]{0});
            float[] out = got.length > 0 ? NativeTF.tensorDataF32(got[0]) : new float[0];
            System.out.println("DEBUG upsample: got=" + got.length + " len=" + out.length);
            check(out.length == 64,
                    "Upsample2D forward runs");
            NativeTF.deleteTensor(xt);
            NativeTF.deleteTensor(got[0]);
            s.close();
        } catch (Exception e) {
            check(false, "Upsample2D forward runs (" + e + ")");
        }
    }

    static void selfAttentionClassifies() {
        Random rnd = new Random(31);
        int n = 200, t = 8, d = 4;
        float[] X = new float[n * t * d];
        int[] Y = new int[n];
        for (int i = 0; i < n; i++) {
            float lvl = i % 2 == 0 ? 0.75f : 0.15f;
            for (int p = 0; p < t * d; p++)
                X[i * t * d + p] =
                        (float) Math.min(1, Math.max(0,
                                lvl + rnd.nextGaussian() * 0.15));
            Y[i] = i % 2;
        }
        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 7L,
                new Input3D(t, d),
                new SelfAttention(),
                new SelfAttention(),
                new Flatten(), new Dense(2))) {
            net.fit(X, Y, 40, 32, null, null, false);
            float[] p = net.predict(X);
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (Metrics.argmax(p, i * 2, 2) == Y[i]) correct++;
            check(correct / (double) n > 0.9, "SelfAttention classifies");
        }
    }

    static void selfAttentionSaveLoad() throws Exception {
        java.nio.file.Path dir = java.nio.file.Path.of("target", "selftest");
        try (Sequential s = new Sequential(Optimizer.adam(0.01f), 3L,
                new Input3D(6, 5),
                new SelfAttention(),
                new LSTM(8),
                new Dense(2))) {
            s.saveTo(dir.resolve("attn.bin"));
            try (Sequential r = Sequential.loadFrom(dir.resolve("attn.bin"),
                    Optimizer.adam(0.01f))) {
                check(Arrays.equals(s.getParams(), r.getParams()),
                        "SelfAttention save/load round-trips");
            }
        }
    }

    static void denseNdClassifies() throws Exception {
        Random rnd = new Random(41);
        int n = 200, t = 8, d = 4;
        float[] X = new float[n * t * d];
        int[] Y = new int[n];
        for (int i = 0; i < n; i++) {
            float lvl = i % 2 == 0 ? 0.75f : 0.15f;
            for (int p = 0; p < t * d; p++)
                X[i * t * d + p] =
                        (float) Math.min(1, Math.max(0,
                                lvl + rnd.nextGaussian() * 0.15));
            Y[i] = i % 2;
        }
        java.nio.file.Path dir = java.nio.file.Path.of("target", "selftest");
        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 11L,
                new Input3D(t, d),
                new DenseND(16), new ReLU(),
                new DenseND(8),
                new Flatten(), new Dense(2))) {
            net.fit(X, Y, 40, 32, null, null, false);
            float[] p = net.predict(X);
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (Metrics.argmax(p, i * 2, 2) == Y[i]) correct++;
            check(correct / (double) n > 0.9, "DenseND classifies");

            // rank-3 flow flows through DenseND: [B,T,units] -> flatten -> head
            try (Sequential shapeNet = new Sequential(Optimizer.adam(0.01f),
                    new Input3D(t, d),
                    new DenseND(16),
                    new Flatten(),
                    new Dense(4))) {
                float[] probe = new float[2 * t * d];
                float[] out = shapeNet.predict(probe);
                check(out.length == 2 * 4, "DenseND feeds [B,T,16] into head");
            }

            net.saveTo(dir.resolve("densend.bin"));
            try (Sequential r = Sequential.loadFrom(dir.resolve("densend.bin"),
                    Optimizer.adam(0.01f))) {
                check(Arrays.equals(net.getParams(), r.getParams()),
                        "DenseND save/load round-trips");
            }
        }
    }

    static void layerNormAttentionClassifies() throws Exception {
        // NOTE: LayerNorm normalizes each position's FEATURE VECTOR, so the
        // signal must live WITHIN a position: is channel 0 louder than
        // channel 1 (averaged over time)? That relative difference survives
        // per-position normalization.
        Random rnd = new Random(53);
        int n = 200, t = 8, d = 4;
        float[] X = new float[n * t * d];
        int[] Y = new int[n];
        for (int i = 0; i < n; i++) {
            boolean ch0High = rnd.nextBoolean();
            for (int p = 0; p < t; p++) {
                float c0 = (ch0High ? 0.75f : 0.25f) + (float) rnd.nextGaussian() * 0.15f;
                float c1 = (ch0High ? 0.25f : 0.75f) + (float) rnd.nextGaussian() * 0.15f;
                X[i * t * d + p * d]     = Math.max(0, Math.min(1, c0));
                X[i * t * d + p * d + 1] = Math.max(0, Math.min(1, c1));
                X[i * t * d + p * d + 2] = 0.5f;
                X[i * t * d + p * d + 3] = 0.5f;
            }
            Y[i] = ch0High ? 1 : 0;
        }
        java.nio.file.Path dir = java.nio.file.Path.of("target", "selftest");
        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 13L,
                new Input3D(t, d),
                new LayerNorm1D(),
                new SelfAttention(),
                new LayerNorm1D(),
                new Flatten(), new Dense(2))) {
            net.fit(X, Y, 120, 32, null, null, false);
            float[] p = net.predict(X);
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (Metrics.argmax(p, i * 2, 2) == Y[i]) correct++;
            check(correct / (double) n > 0.9, "LayerNorm1D + attention classifies");

            net.saveTo(dir.resolve("ln.bin"));
            try (Sequential r = Sequential.loadFrom(dir.resolve("ln.bin"),
                    Optimizer.adam(0.01f))) {
                check(Arrays.equals(net.getParams(), r.getParams()),
                        "LayerNorm1D save/load round-trips");
            }
        }
    }

    static void embeddingNdFeedsLstm() {
        // the composition the old flattened Embedding could not express:
        // EmbeddingND -> LSTM. Task: does token id 3 appear in the window?
        Random rnd = new Random(61);
        int n = 300, t = 6, vocab = 10;
        float[] X = new float[n * t];
        int[] Y = new int[n];
        for (int i = 0; i < n; i++) {
            boolean has = rnd.nextInt(100) < 50;
            boolean placed = false;
            for (int p = 0; p < t; p++) {
                int tok = rnd.nextInt(vocab);
                if (!has && tok == 3) tok = 4;
                if (has && p == t - 1 && !placed) tok = 3;
                if (tok == 3) placed = true;
                X[i * t + p] = tok;
            }
            Y[i] = has ? 1 : 0;
        }
        try (Sequential net = new Sequential(Optimizer.adam(0.02f), 17L,
                new Input3D(t, 1),
                new EmbeddingND(vocab, 12),
                new LSTM(16),
                new Dense(24), new ReLU(),
                new Dense(2))) {
            net.fit(X, Y, 60, 32, null, null, false);
            float[] p = net.predict(X);
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (Metrics.argmax(p, i * 2, 2) == Y[i]) correct++;
            check(correct / (double) n > 0.95,
                    "EmbeddingND -> LSTM composition trains");
        }
    }

    static void positionalEncodingOrder() {
        // position-aware task: label depends on the FIRST symbol only -
        // impossible without position information.
        Random rnd = new Random(67);
        int n = 300, t = 6, vocab = 6;
        float[] X = new float[n * t * vocab];
        int[] Y = new int[n];
        for (int i = 0; i < n; i++) {
            int first = rnd.nextInt(vocab);
            for (int p = 0; p < t; p++)
                X[i * t * vocab + p * vocab + (p == 0 ? first : rnd.nextInt(vocab))] = 1f;
            Y[i] = first % 2;
        }
        try (Sequential net = new Sequential(Optimizer.adam(0.02f), 19L,
                new Input3D(t, vocab),
                new PositionalEncoding(),
                new SelfAttention(),
                new Flatten(),
                new Dense(12), new ReLU(),
                new Dense(2))) {
            net.fit(X, Y, 80, 32, null, null, false);
            float[] p = net.predict(X);
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (Metrics.argmax(p, i * 2, 2) == Y[i]) correct++;
            check(correct / (double) n > 0.85,
                    "PositionalEncoding enables order task");
        }
    }

    static void multiHeadAttentionClassifies() throws Exception {
        // same channel-contrast task as the LayerNorm test - signal lives
        // WITHIN positions, so it survives attention of any head count.
        Random rnd = new Random(71);
        int n = 200, t = 8, d = 4;
        float[] X = new float[n * t * d];
        int[] Y = new int[n];
        for (int i = 0; i < n; i++) {
            boolean ch0High = rnd.nextBoolean();
            for (int p = 0; p < t; p++) {
                float c0 = (ch0High ? 0.75f : 0.25f) + (float) rnd.nextGaussian() * 0.15f;
                float c1 = (ch0High ? 0.25f : 0.75f) + (float) rnd.nextGaussian() * 0.15f;
                X[i * t * d + p * d]     = Math.max(0, Math.min(1, c0));
                X[i * t * d + p * d + 1] = Math.max(0, Math.min(1, c1));
                X[i * t * d + p * d + 2] = 0.5f;
                X[i * t * d + p * d + 3] = 0.5f;
            }
            Y[i] = ch0High ? 1 : 0;
        }
        java.nio.file.Path dir = java.nio.file.Path.of("target", "selftest");
        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 23L,
                new Input3D(t, d),
                new MultiHeadAttention(2),
                new LayerNorm1D(),
                new Flatten(), new Dense(2))) {
            net.fit(X, Y, 60, 32, null, null, false);
            float[] p = net.predict(X);
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (Metrics.argmax(p, i * 2, 2) == Y[i]) correct++;
            check(correct / (double) n > 0.9, "MultiHeadAttention classifies");

            net.saveTo(dir.resolve("mha.bin"));
            try (Sequential r = Sequential.loadFrom(dir.resolve("mha.bin"),
                    Optimizer.adam(0.01f))) {
                check(Arrays.equals(net.getParams(), r.getParams()),
                        "MultiHeadAttention save/load round-trips");
            }
        }
    }

    static void causalMaskTrains() {
        // causal variant: every position still sees its own channel contrast
        // (position t attends to <= t), so the task stays learnable.
        Random rnd = new Random(73);
        int n = 200, t = 8, d = 4;
        float[] X = new float[n * t * d];
        int[] Y = new int[n];
        for (int i = 0; i < n; i++) {
            boolean ch0High = rnd.nextBoolean();
            for (int p = 0; p < t; p++) {
                float c0 = (ch0High ? 0.75f : 0.25f) + (float) rnd.nextGaussian() * 0.15f;
                float c1 = (ch0High ? 0.25f : 0.75f) + (float) rnd.nextGaussian() * 0.15f;
                X[i * t * d + p * d]     = Math.max(0, Math.min(1, c0));
                X[i * t * d + p * d + 1] = Math.max(0, Math.min(1, c1));
                X[i * t * d + p * d + 2] = 0.5f;
                X[i * t * d + p * d + 3] = 0.5f;
            }
            Y[i] = ch0High ? 1 : 0;
        }
        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 29L,
                new Input3D(t, d),
                new MultiHeadAttention(2, true),
                new Flatten(), new Dense(2))) {
            net.fit(X, Y, 80, 32, null, null, false);
            float[] p = net.predict(X);
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (Metrics.argmax(p, i * 2, 2) == Y[i]) correct++;
            check(correct / (double) n > 0.9,
                    "causal MultiHeadAttention trains through mask");
        }
    }

    static void crossAttentionRetrieves() {
        // queries: first 3 slots carry probe digits; payload: next 8 slots.
        // label: does ANY query digit appear in the payload? Cross-attention
        // matches across the segment boundary in one hop.
        Random rnd = new Random(79);
        int n = 400, tq = 3, tp = 8, t = tq + tp, v = 12;
        float[] X = new float[n * t * v];
        int[] Y = new int[n];
        for (int i = 0; i < n; i++) {
            boolean present = rnd.nextBoolean();
            Integer[] probes = pickProbes(rnd, tq, v);
            Integer[] payload = pickPayload(rnd, tp, v, probes, present ? probes[0] : -1);
            for (int j = 0; j < tq; j++)
                X[i * t * v + j * v + probes[j]] = 1f;
            for (int j = 0; j < tp; j++)
                X[i * t * v + (tq + j) * v + payload[j]] = 1f;
            // ensure at least one match when present
            if (present) X[i * t * v + (tq + rnd.nextInt(tp)) * v + probes[0]] = 1f;
            Y[i] = present ? 1 : 0;
        }
        try (Sequential net = new Sequential(Optimizer.adam(0.02f), 31L,
                new Input3D(t, v),
                new CrossAttention(tq),
                new LayerNorm1D(),
                new Flatten(),
                new Dense(16), new ReLU(),
                new Dense(2))) {
            net.fit(X, Y, 120, 32, null, null, false);
            float[] p = net.predict(X);
            int correct = 0;
            for (int i = 0; i < n; i++)
                if (Metrics.argmax(p, i * 2, 2) == Y[i]) correct++;
            check(correct / (double) n > 0.9,
                    "CrossAttention retrieves across segments");
        }
    }

    static Integer[] pickProbes(Random rnd, int count, int limit) {
        java.util.List<Integer> cand = new java.util.ArrayList<>();
        for (int i = 0; i < limit; i++) cand.add(i);
        java.util.Collections.shuffle(cand, rnd);
        return cand.subList(0, count).toArray(new Integer[0]);
    }

    /** count distinct ids from 0..limit-1 excluding everything in 'exclude';
     *  'include' (>= 0) is always part of the selection. */
    static Integer[] pickPayload(Random rnd, int count, int limit,
                                 Integer[] exclude, int include) {
        java.util.List<Integer> cand = new java.util.ArrayList<>();
        for (int i = 0; i < limit; i++) {
            boolean skip = false;
            for (int e : exclude) if (e == i) { skip = true; break; }
            if (!skip && i != include) cand.add(i);
        }
        java.util.Collections.shuffle(cand, rnd);
        java.util.List<Integer> out = new java.util.ArrayList<>();
        if (include >= 0 && count > 0) out.add(include);
        for (int c : cand) {
            if (out.size() == count) break;
            out.add(c);
        }
        return out.toArray(new Integer[0]);
    }

    static void saveLoadRoundTrips() throws Exception {
        java.nio.file.Path dir = java.nio.file.Path.of("target", "selftest");
        try (Sequential s = new Sequential(Optimizer.adam(0.01f), 9L,
                new Input(2), new Dense(8), new ReLU(), new Dense(2))) {
            s.saveTo(dir.resolve("seq.bin"));
            try (Sequential r = Sequential.loadFrom(dir.resolve("seq.bin"), Optimizer.adam(0.01f))) {
                check(Arrays.equals(s.getParams(), r.getParams()), "Sequential save/load");
            }
        }
        try (Regressor g = new Regressor(Optimizer.adam(0.01f), 9L,
                new Input(2), new Dense(8), new ReLU(), new Dense(1))) {
            g.saveTo(dir.resolve("reg.bin"));
            try (Regressor r = Regressor.loadFrom(dir.resolve("reg.bin"), Optimizer.adam(0.01f))) {
                check(Arrays.equals(g.getParams(), r.getParams()), "Regressor save/load");
            }
        }
    }

    static float[] flat(float[][] arr) {
        float[] out = new float[arr.length * arr[0].length];
        for (int i = 0; i < arr.length; i++)
            System.arraycopy(arr[i], 0, out, i * arr[0].length, arr[0].length);
        return out;
    }

    private SelfTest() {}
}
