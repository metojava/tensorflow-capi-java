package com.example.tfcapi;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;

/**
 * Small helpers for the unglamorous half of machine learning: loading CSVs,
 * splitting rows, standardizing features and encoding labels. All methods are
 * static and allocation-happy on purpose - they run once per program, not per
 * batch.
 *
 * <pre>{@code
 * List<String[]> rows = Datasets.readCsv("/data/housing.csv", ",");
 * float[] X = Datasets.toFloatFeatures(rows.subList(1, rows.size()), 0, 1, 2);
 * int[][] split = Datasets.trainTestSplit(numRows, 0.3, 42L);
 * float[][] norm = Datasets.fitStandardizer(trainX, numFeatures);
 * Datasets.applyStandardizer(testX, norm[0], norm[1]);
 * }</pre>
 */
public final class Datasets {

    /**
     * Reads delimited text lines from a classpath resource (when
     * {@code location} starts with {@code "/"}) or from the file system.
     * Blank lines are skipped; cells are trimmed. The header (if any) is
     * returned as the first row - callers typically drop it with
     * {@code rows.subList(1, rows.size())}.
     */
    public static List<String[]> readCsv(String location, String delimiter) throws IOException {
        InputStream in = location.startsWith("/")
                ? Datasets.class.getResourceAsStream(location)
                : new FileInputStream(location);
        if (in == null) {
            throw new IOException("resource or file not found: " + location);
        }
        List<String[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] cells = line.split(Pattern.quote(delimiter), -1);
                for (int i = 0; i < cells.length; i++) {
                    cells[i] = cells[i].trim();
                }
                rows.add(cells);
            }
        }
        return rows;
    }

    /** Selected columns of {@code rows} as a flat row-major float matrix. */
    public static float[] toFloatFeatures(List<String[]> rows, int... cols) {
        float[] out = new float[rows.size() * cols.length];
        for (int r = 0; r < rows.size(); r++) {
            for (int c = 0; c < cols.length; c++) {
                out[r * cols.length + c] = Float.parseFloat(rows.get(r)[cols[c]]);
            }
        }
        return out;
    }

    /** One numeric column of {@code rows} as a float array. */
    public static float[] toFloatColumn(List<String[]> rows, int col) {
        float[] out = new float[rows.size()];
        for (int r = 0; r < rows.size(); r++) {
            out[r] = Float.parseFloat(rows.get(r)[col]);
        }
        return out;
    }

    /** One integer column of {@code rows} (class ids, quality scores...). */
    public static int[] toIntColumn(List<String[]> rows, int col) {
        int[] out = new int[rows.size()];
        for (int r = 0; r < rows.size(); r++) {
            out[r] = Integer.parseInt(rows.get(r)[col]);
        }
        return out;
    }

    /**
     * Deterministic shuffled index split. Returns
     * {@code {trainIndices, testIndices}}; sizes are
     * {@code n - round(n * testFraction)} and {@code round(n * testFraction)}.
     * Same {@code n}/{@code testFraction}/{@code seed} always yields the same
     * split.
     */
    public static int[][] trainTestSplit(int n, double testFraction, long seed) {
        if (n <= 0 || testFraction < 0 || testFraction >= 1) {
            throw new IllegalArgumentException("bad split: n=" + n + " frac=" + testFraction);
        }
        int[] order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Random rnd = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int t = order[i]; order[i] = order[j]; order[j] = t;
        }
        int nTest = (int) Math.round(n * testFraction);
        int[] test = new int[nTest];
        int[] train = new int[n - nTest];
        System.arraycopy(order, 0, train, 0, train.length);
        System.arraycopy(order, train.length, test, 0, test.length);
        return new int[][]{train, test};
    }

    /**
     * Computes per-feature mean and population std of a flat row-major
     * matrix; returns {@code {mean[features], std[features]}}. Fit on the
     * TRAINING split only, then apply to both splits.
     */
    public static float[][] fitStandardizer(float[] flat, int features) {
        int rows = flat.length / features;
        float[] mean = new float[features];
        float[] std = new float[features];
        for (int r = 0; r < rows; r++) {
            for (int f = 0; f < features; f++) {
                mean[f] += flat[r * features + f];
            }
        }
        for (int f = 0; f < features; f++) {
            mean[f] /= Math.max(rows, 1);
        }
        for (int r = 0; r < rows; r++) {
            for (int f = 0; f < features; f++) {
                float d = flat[r * features + f] - mean[f];
                std[f] += d * d;
            }
        }
        for (int f = 0; f < features; f++) {
            std[f] = (float) Math.sqrt(std[f] / Math.max(rows - 1, 1));
        }
        return new float[][]{mean, std};
    }

    /** In-place z-score transform {@code (x - mean) / (std + 1e-8)}. */
    public static void applyStandardizer(float[] flat, float[] mean, float[] std) {
        int features = mean.length;
        for (int r = 0; r < flat.length / features; r++) {
            for (int f = 0; f < features; f++) {
                flat[r * features + f] =
                        (flat[r * features + f] - mean[f]) / (std[f] + 1e-8f);
            }
        }
    }

    /** Copies the given rows of a flat row-major matrix into a new array. */
    public static float[] gatherRows(float[] flat, int[] indices, int rowElems) {
        float[] out = new float[indices.length * rowElems];
        for (int i = 0; i < indices.length; i++) {
            System.arraycopy(flat, indices[i] * rowElems, out, i * rowElems, rowElems);
        }
        return out;
    }

    /** Copies values at the given indices ({@code out[i] = values[idx[i]]}). */
    public static int[] gatherValues(int[] values, int[] indices) {
        int[] out = new int[indices.length];
        for (int i = 0; i < indices.length; i++) {
            out[i] = values[indices[i]];
        }
        return out;
    }

    /** Unit-vector encoding of one class id: {@code oneHot(2, 4) -> [0,0,1,0]}. */
    public static float[] oneHot(int label, int numClasses) {
        float[] out = new float[numClasses];
        out[label] = 1f;
        return out;
    }

    private Datasets() {}
}
