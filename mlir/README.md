# SYNAPSE MLIR Submodules

This directory is reserved for MLIR/compiler submodules used by SYNAPSE.

Current layout:

```text
mlir/
  neura/   # git submodule: https://github.com/coredac/neura
```

SYNAPSE itself only defines the programming model, frontend, internal graph IR,
and lowering to Taskflow IR. Compilation from Taskflow IR to NEURA or lower
hardware/compiler targets belongs to the NEURA/Taskflow compiler stack in this
directory.

The submodule is registered in `.gitmodules`:

```text
[submodule "mlir/neura"]
	path = mlir/neura
	url = https://github.com/coredac/neura
```
