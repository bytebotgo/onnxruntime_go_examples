`onnxruntime_go`: 手写数字识别
===============================

这个例子使用了预训练的 MNIST 网络，从 ONNX 官方模型库(https://github.com/onnx/models/tree/ddbbd1274c8387e3745778705810c340dea3d8c7/validated/vision/classification/mnist) 。具体地说，包含的 mnist.onnx 是来自上述链接的 MNIST-12。

这个例子使用网络分析单个图像文件，这些文件在命令行中指定。

Example Usage
-------------

运行程序时使用 `-help` 查看所有命令行标志。一般来说，你需要提供一个输入图像。

```bash
./mnist -image_path ./eight.png
./mnist -image_path ./tiny_5.png

# 如果你想要反转图像颜色，可以使用以下标志。网络是在黑色背景上训练的，所以你可能想要反转白色背景的图像。
./mnist -image_path ./seven.png -invert_image
```

注意，程序还会在当前目录中创建 `postprocessed_input_image.png`，显示传递给神经网络的图像，经过调整大小和转换为灰度。
