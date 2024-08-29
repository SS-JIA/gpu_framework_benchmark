## Context

The aim of this repository is to benchmark different mobile GPU ML frameworks at
the operator level, in order to compare and contrast between them.

Currently, the following frameworks are included:

* MNN

The following operators are supported:

* Matrix Multiplication

## Build and Run

To build, run the `./build.sh` script from repository root.

To run, run the `./run.sh` script from repository root. The binary will be
pushed to your device and be executed via `adb shell`.
