`onnxruntime_go`: Float16 手写数字识别 
=======================================

这个例子几乎与这个仓库中的普通 `mnist` 例子相同，但使用了一个已经转换为使用 16 位浮点数的模型。这个例子旨在说明如何使用 `github.com/x448/float16` 包和 `onnxruntime_go` 的 `CustomDataTensor` 类型将输入转换为 16 位浮点值。

代码几乎是从 `../mnist` 例子复制和粘贴的。它只在几个地方有所不同：
  - `ProcessedImage.GetNetworkInput` 函数现在将每个输入像素从 float32 灰度值转换为 float16，并将 float16 数据写入一个字节切片。
  - `input` 和 `output` 张量在 `classifyDigit` 函数中创建，现在都是 `CustomDataTensor`s，由字节切片支持。
  - `convertFloat16Data` 函数已被添加，用于将 `float16.Float16` 数据的输出张量的字节转换为 `float32` 的切片。

包含的 `mnist_float16.onnx` 网络是通过使用 `onnxconverter-common` python 包在 `../mnist/mnist.onnx` 网络上创建的，使用的是 [这个页面](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html) 中描述的过程。

Example Usage
-------------

这个程序的使用方式与 `../mnist` 完全相同。使用 `go build` 构建它，并使用 `-help` 查看所有命令行标志。它从当前目录加载 `mnist_float16.onnx` 网络。

例如，
```bash
go build .
./mnist_float16 -image_path ../mnist/eight.png
```

将产生以下输出：
```
Saved postprocessed input image to ./postprocessed_input_image.png.
  0: 1.350586
  1: 1.148438
  2: 2.232422
  3: 0.827148
  4: -3.474609
  5: 1.199219
  6: -1.187500
  7: -5.960938
  8: 4.765625
  9: -2.345703
../mnist/eight.png 可能是 8，概率为 4.765625
一切都运行正常！
```

