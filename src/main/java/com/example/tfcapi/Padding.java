package com.example.tfcapi;

/** Spatial padding mode for Conv2D / MaxPool2D. */
public enum Padding {
    VALID("VALID", 0),
    SAME("SAME", 1);

    final String tfName;
    final int code;

    Padding(String tfName, int code) {
        this.tfName = tfName;
        this.code = code;
    }

    static Padding fromCode(int code) {
        for (Padding p : values()) {
            if (p.code == code) return p;
        }
        throw new IllegalArgumentException("unknown padding code " + code);
    }
}
