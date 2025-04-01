Getting ONNX Input and Output Information
=========================================

这个示例项目定义了一个命令行实用程序，用于打印用户指定 .onnx 文件的输入和输出信息到标准输出。


示例用法：
```
go build .

./onnx_list_inputs_and_outputs -onnx_file ../image_object_detect/yolov8n.onnx
```

上述命令应该输出类似以下的内容：

```
1 inputs to ../image_object_detect/yolov8n.onnx:
  Index 0: "images": [1 3 640 640], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
1 outputs from ../image_object_detect/yolov8n.onnx:
  Index 0: "output0": [1 84 8400], ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
```

(yolov8 网络只有一个输入和一个输出：一个 1x3x640x640 input,
名为 "images", 和一个 1x84x8400 output, 名为 "output0".)

