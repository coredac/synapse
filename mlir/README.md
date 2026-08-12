# SYNAPSE MLIR Dependencies

This directory contains the compiler projects used by SYNAPSE after its Python
frontend has captured a program.

## Dependency layout

```text
mlir/
  amoeba/                 # git submodule: https://github.com/coredac/amoeba
    thirdparty/neura/     # nested submodule managed by Amoeba
```

SYNAPSE owns the programming model, Python frontend, and lowering of a captured
program into compiler IR. Amoeba provides the backend-neutral Taskflow dialect
and backend integration. Neura is Amoeba's CGRA backend and provides the
single-CGRA dialect, mapping, routing, register allocation, and code generation.

SYNAPSE therefore depends on Amoeba rather than carrying a second, independent
Neura checkout. This keeps the Taskflow and Neura interfaces on the revisions
tested together by Amoeba.

## Initial single-task compilation boundary

The first CGRA path intentionally handles one task only. The frontend treats the
whole captured program as an implicit task and lowers it to the following
container hierarchy:

```text
taskflow.task
  neura.kernel
    mapped Neura operations
```

The operations inside `neura.kernel` are expected to carry post-mapping
information such as tile coordinates, time steps, links, and registers. The
single-task path does not yet define user-facing task syntax or perform
inter-task allocation, placement, scheduling, replication, or communication.

## Checkout

Initialize Amoeba and its Neura dependency recursively:

```sh
git submodule update --init --recursive
```
