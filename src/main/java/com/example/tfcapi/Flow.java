package com.example.tfcapi;

/** Data flowing between layers: an op handle plus its symbolic shape. */
final class Flow {

    final long op;
    final Shape shape;

    Flow(long op, Shape shape) {
        this.op = op;
        this.shape = shape;
    }
}
