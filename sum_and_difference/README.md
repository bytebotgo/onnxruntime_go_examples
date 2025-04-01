Sum and Difference `onnxruntime_go` Example
===========================================

这是一个基本的、注释丰富的命令行程序，使用 `onnxruntime_go` 库加载和运行一个 ONNX 格式的神经网络。

用法
-----

使用 `go build` 构建程序。之后，它应该在大多数系统上不带参数运行：`./sum_and_difference`。 如果你遇到错误，可能需要使用 `-onnxruntime_lib` 命令行标志指定不同的 `onnxruntime` 共享库版本。 （运行程序 `-help` 查看用法信息。）

```bash
go build .
./sum_and_difference
```

如果成功，应该输出以下内容：
```
The network ran without errors.
  Input data: [0.2 0.3 0.6 0.9]
  Approximate sum of inputs: 1.999988
  Approximate max difference between any two inputs: 0.607343
The network seemed to run OK!
```

