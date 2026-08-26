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
    placed Neura operations with ordinary MLIR value types
```

The frontend fixes the spatial operation placement selected by the TileArray
program, but it does not construct Neura's predicated value type or final
mapping metadata. The backend compilation flow performs:

```text
--leverage-predicated-value
  -> --insert-data-mov
  -> --map-to-accelerator="mapping-strategy=template mapping-mode=spatial-only"
```

The final mapped Neura IR contains tile coordinates, time steps, links, and
register information. The single-task path does not yet define user-facing
task syntax or perform inter-task allocation, placement, scheduling,
replication, or communication.

## Checkout

Initialize Amoeba and its Neura dependency recursively:

```sh
git submodule update --init --recursive
```
