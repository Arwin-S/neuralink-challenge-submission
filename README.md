# Neuralink Challenge Submission

## Domain-Specific Autoencoder for Neuralink Compression Challenge

This project provides an implementation of a domain-specific autoencoder designed for the Neuralink Compression Challenge.

## Setup and Build

To set up and build the project, run the following command:

```bash
make
```

If you need to clear all build files (e.g., if files are messed up), use:

```bash
make clean
```

## Executables

The following executables will be compiled into the root project directory:

- `encode`
- `decode`

These executables are built to run with `eval.sh` and have the following arguments:

### Encode

```bash
./encode <.wav file path> <.brainwire file path>
```

### Decode

```bash
./decode <.brainwire file path> <.wav file path>
```