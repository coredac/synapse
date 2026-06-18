# SYNAPSE 代码阅读指南

这份文档按当前 scope 写：SYNAPSE 只负责 programming model、AST frontend、内部 Graph IR，以及 lowering 到 Taskflow IR。Taskflow IR 之后如何编译到 NEURA / CGRA / hardware IR，不在 SYNAPSE frontend 展示路径里；那部分应该由 `mlir/neura` submodule 负责。

## 0. 当前边界

```text
SYNAPSE
  language      用户怎么写程序
  frontend      Python source/AST -> SYNAPSE IR
  ir            Graph / Task / Channel / Buffer 等内部表示
  compiler      SYNAPSE IR -> Taskflow IR
  mlir/neura    外部 NEURA/Taskflow compiler submodule
```

现在代码里 `ir/` 目录还没有拆出来，`Graph/Task/Buffer/Channel/Layout/HWIntent` 仍在 `language/core.py`。这是下一步应该整理的地方。

## 1. 先看 Examples

先读：

1. `synapse/examples/matmul_relu.py`
2. `synapse/examples/hierarchical_attention.py`

这两个文件代表用户侧 programming model，而不是 IR builder。

`matmul_relu.py` 展示：

```python
@syn.program
def matmul_relu(
    A: syn.Tensor((64, 64), "f32", layout=syn.tile(16, 16)),
    B: syn.Tensor((64, 64), "f32", layout=syn.tile(16, 16)),
):
    C = syn.matmul(...)
    return syn.relu(...)
```

你审这里主要看：

- API 是否像编程语言，而不是手写 IR；
- `target/layout/hw_intent` 放在 op 参数里是否自然；
- 中间 tensor `C` 是否应该由 compiler 管理。

当前行为：

- `A/B` 是 input；
- `Out` 是 return value，所以是 output；
- `C` 是 temp，Taskflow compiler 会生成内部 `memref.alloc()`；
- `syn.matmul` / `syn.relu` 经 AST frontend 变成 Graph/Task/Buffer。

`hierarchical_attention.py` 展示：

```python
with syn.region("attention", target=["cgra", "gpu"]):
    Q = syn.matmul(...)
    K = syn.matmul(...)
    Kt = syn.transpose(...)
    return syn.matmul(Q, Kt, ...)
```

这里的 `syn.region` 表示 composite task / nested dataflow region。内部 task 会带 `synapse.parent_task = "attention"` metadata。

## 2. 看 Language

读：

1. `synapse/python/synapse/language/tensor.py`
2. `synapse/python/synapse/language/__init__.py`
3. `synapse/python/synapse/__init__.py`

`language/tensor.py` 定义用户可见 API：

- `@syn.program`
- `syn.Tensor`
- `syn.Scalar`
- `syn.matmul`
- `syn.relu`
- `syn.transpose`
- `syn.region`
- `syn.tile / syn.row / syn.pipeline / syn.systolic`

重点：

- `Program.build_graph()` 不执行用户函数体，而是调用 frontend pipeline；
- `TensorSpec` / `ScalarSpec` 是用户 annotation；
- `TensorValue` 是 frontend 内部 symbolic value；
- `_BuildContext` 现在是 AST builder 复用的低层 graph construction 工具。

这个文件以后应该更薄。真正的 language surface 留在这里，IR construction 逐渐迁到 `frontend/` 和 `ir/`。

## 3. 看 Frontend

读：

1. `synapse/python/synapse/frontend/parser.py`
2. `synapse/python/synapse/frontend/infer.py`
3. `synapse/python/synapse/frontend/builder.py`

这是最像 Allo 的部分。

`parser.py`：

- 用 `inspect.getsourcelines()` 拿 Python source；
- 用 `ast.parse()` 得到 Python AST；
- 捕获 globals / closure；
- 返回 `ProgramSource`。

`infer.py`：

- 遍历 Python AST；
- 根据 function annotations 建 input symbol；
- 根据 `syn.matmul/relu/transpose` 推导 output shape、dtype、layout；
- 记录 return value。

`builder.py`：

- 再遍历 AST；
- 把 input annotation 建成 `Buffer`；
- 把 op call 建成 `Task`；
- 把中间值建成 temp buffer；
- 把 return tensor 标记成 output；
- 支持 `with syn.region(...)` 生成 nested task scope。

重点审：

- `parser / infer / builder` 的边界是否清楚；
- 是否应该引入一个独立 SYNAPSE AST，而不是直接 Python AST -> Graph；
- error reporting 是否需要单独 `diagnostics.py`；
- shape/layout/resource inference 是否应该越来越像 Allo 的 `TypeInferer`。

## 4. 看内部 IR

现在读：

1. `synapse/python/synapse/language/core.py`

但从架构上，它应该迁到：

```text
synapse/python/synapse/ir/
  graph.py
  values.py
  attrs.py
  verifier.py
```

当前 `core.py` 定义：

- `Graph`
- `Task`
- `Kernel`
- `Channel`
- `Buffer`
- `Stream`
- `Token`
- `Layout`
- `HWIntent`

重点审：

- `Task` 是否应该支持 nested children；
- `Kernel` 抽象是否过早；
- `Buffer.role = input/output/temp/io` 是否应该变成 enum；
- `Layout/HWIntent` 是否要和 Taskflow MLIR attr schema 对齐；
- `Graph.task(...)` 是否应该作为低层 escape hatch 隐藏起来。

## 5. 看 Compiler

读：

1. `synapse/python/synapse/compiler/base.py`
2. `synapse/python/synapse/compiler/driver.py`
3. `synapse/python/synapse/compiler/taskflow.py`
4. `synapse/python/synapse/compiler/__init__.py`

当前 compiler 只暴露一个 target：

```text
taskflow
```

`driver.py`：

- `syn.compile(program_or_graph, target="taskflow")`
- 如果输入是 `@syn.program`，先 `build_graph()`；
- 然后调用 `TaskflowCompiler`。

`taskflow.py`：

- 把 Graph/Task/Buffer 转成 textual Taskflow MLIR；
- 保留 `synapse.target_candidates`、layout、resource policy、hw_intent；
- flatten nested task，同时保留 `synapse.parent_task`；
- 为静态 temp buffer 生成 `memref.alloc()`。

重点审：

- textual MLIR 作为第一版 bridge 是否接受；
- TaskflowDialect 是否应该正式加入 channel/region/buffer role；
- `synapse.*` metadata 是否应改成 `taskflow.*`；
- kernel body 现在大多还是 placeholder 注释，这是当前技术债。

## 6. 看 MLIR Submodule 边界

读：

1. `synapse/mlir/README.md`

计划：

```text
synapse/mlir/neura
```

作为 git submodule 指向：

```text
https://github.com/coredac/neura
```

这个 submodule 负责 Taskflow IR 之后的编译，包括 TaskflowDialect / NeuraDialect / passes / codegen。SYNAPSE frontend 不展示 `neura` target。

当前 `mlir/neura` 已经注册为 git submodule，见 `.gitmodules`。

## 7. 看 CLI

读：

1. `synapse/python/synapse/tools/synapse_compile.py`

现在只需要支持：

```bash
python -m synapse.tools.synapse_compile SOURCE.py --target taskflow
```

重点审：

- CLI 是否应该直接识别 `@syn.program`；
- 是否继续要求 source 里有 `build_graph()`；
- `--target` 是否还需要保留，还是默认只输出 Taskflow。

## 8. 看 Tests

读：

1. `synapse/test/Synapse/frontend/test_python_smoke.py`
2. `synapse/test/Synapse/taskflow/matmul_relu_taskflow.mlir`

`test_python_smoke.py` 现在覆盖：

- low-level Graph -> Taskflow；
- nested task parent metadata；
- high-level `@syn.program` -> Taskflow；
- AST frontend 不执行用户函数体。

## 9. 当前待拍板问题

1. `language/core.py` 是否立即迁到 `ir/`？
2. 是否保留 `Graph/Task/Kernel` 的 public export，还是藏到 low-level namespace？
3. `syn.region` 是否是 hierarchical dataflow 的正确用户 API？
4. `target/layout/hw_intent` 应该写在 program 里，还是后面引入 Allo-style schedule/customize？
5. Taskflow IR 是否应该新增正式 channel/region/buffer role op/attr？
6. CLI 是否应该只输出 Taskflow，去掉 target registry？

## 10. 当前技术债

已经做到：

- AST frontend，而不是 Python side-effect builder；
- 用户写 tensor program；
- frontend 生成 Graph/Task/Buffer/Channel；
- compiler 只 lower 到 Taskflow IR；
- examples 不再手写 `Graph.task(...)`。

还没做到：

- 独立 `ir/` 包；
- 完整 AST 支持；
- PyTorch FX / Allo frontend；
- operator overloading；
- real linalg/affine body generation；
- formal Taskflow channel/region ops；
- `mlir/neura` 只作为下游 compiler stack 存在，SYNAPSE frontend 不调用它。
