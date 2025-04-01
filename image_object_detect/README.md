基于 Yolo 的图像目标检测
=================================

本示例使用包含的 yolov8n.onnx 网络来检测图像中的图像。目前，该示例是硬编码的，以处理包含的 car.png 图像。它多次执行检测以计算定时统计。

可以通过将 USE_COREML 环境变量设置为 真的（尽管这会导致程序在不支持 CoreML 的系统上失败。

使用 CoreML
-------------------
```bash
$ go build .
$ USE_COREML=true ./image_object_detect

Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Object: car Confidence: 0.50 Coordinates: (392.156250, 286.328125), (692.111755, 655.371094)
Min Time: 17.401875ms, Max Time: 21.7065ms, Avg Time: 19.258691ms, Count: 5
50th: 18.485666ms, 90th: 21.7065ms, 99th: 21.7065ms
```

仅在 CPU 上运行，不使用 CoreML
-----------------------------------
```bash
$ go build .
$ ./image_object_detect

Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Object: car Confidence: 0.50 Coordinates: (392.655396, 285.742920), (691.901306, 656.455566)
Min Time: 41.5205ms, Max Time: 58.348084ms, Avg Time: 46.154341ms, Count: 5
50th: 43.471958ms, 90th: 58.348084ms, 99th: 58.348084ms
```

(Note the slower execution times.)
