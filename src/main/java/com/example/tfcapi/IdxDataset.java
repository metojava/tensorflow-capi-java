package com.example.tfcapi;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.zip.GZIPInputStream;

/** Reader for gzip-compressed IDX files (MNIST / Fashion-MNIST format). */
public final class IdxDataset {

    public record Images(int count, byte[] pixels) {}
    public record Labels(int count, byte[] labels) {}

    public static Images loadImages(String path) throws IOException {
        try (DataInputStream in = open(path)) {
            if (in.readInt() != 2051) throw new IOException("bad magic in " + path);
            int count = in.readInt(), rows = in.readInt(), cols = in.readInt();
            if (rows != 28 || cols != 28) throw new IOException("expected 28x28 images");
            byte[] px = new byte[count * rows * cols];
            in.readFully(px);
            return new Images(count, px);
        }
    }

    public static Labels loadLabels(String path) throws IOException {
        try (DataInputStream in = open(path)) {
            if (in.readInt() != 2049) throw new IOException("bad magic in " + path);
            int count = in.readInt();
            byte[] lb = new byte[count];
            in.readFully(lb);
            return new Labels(count, lb);
        }
    }

    /** Converts raw bytes to floats scaled to [0,1]. */
    public static float[] normalize(byte[] pixels) {
        float[] out = new float[pixels.length];
        for (int i = 0; i < pixels.length; i++) {
            out[i] = (pixels[i] & 0xFF) / 255.0f;
        }
        return out;
    }

    private static DataInputStream open(String path) throws IOException {
        return new DataInputStream(
                new BufferedInputStream(new GZIPInputStream(new FileInputStream(path))));
    }

    private IdxDataset() {}
}
