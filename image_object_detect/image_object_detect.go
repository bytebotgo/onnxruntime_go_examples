package main

import (
	"fmt"
	"image"
	_ "image/gif"  // 支持 GIF 格式图片
	_ "image/jpeg" // 支持 JPEG 格式图片
	_ "image/png"  // 支持 PNG 格式图片
	"os"
	"runtime"
	"sort"

	"github.com/8ff/prettyTimer"
	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

// 模型和图片路径配置
var modelPath = "./yolov8n.onnx"                  // YOLOv8 模型文件路径
var imagePath = "/Users/wxz/Downloads/person.png" // 待检测图片路径
var useCoreML = false                             // 是否使用 CoreML 加速

// ModelSession 结构体用于管理 ONNX 运行时会话
type ModelSession struct {
	Session *ort.AdvancedSession // ONNX 运行时会话
	Input   *ort.Tensor[float32] // 输入张量
	Output  *ort.Tensor[float32] // 输出张量
}

func main() {
	os.Exit(run())
}

func run() int {

	// 计时器
	timingStats := prettyTimer.NewTimingStats()

	if os.Getenv("USE_COREML") == "true" {
		useCoreML = true
	}
	// 读取输入图像到 image.Image 对象
	pic, e := loadImageFile(imagePath)
	if e != nil {
		fmt.Printf("Error loading input image: %s\n", e)
		return 1
	}
	// 获取图片的宽度和高度
	originalWidth := pic.Bounds().Canon().Dx()
	originalHeight := pic.Bounds().Canon().Dy()

	// 初始化模型会话
	modelSession, e := initSession()
	if e != nil {
		fmt.Printf("Error creating session and tensors: %s\n", e)
		return 1
	}
	defer modelSession.Destroy()

	// 运行检测 5 次
	for i := 0; i < 5; i++ {
		// 准备输入
		e = prepareInput(pic, modelSession.Input)
		if e != nil {
			fmt.Printf("Error converting image to network input: %s\n", e)
			return 1
		}

		timingStats.Start()
		// 运行模型
		e = modelSession.Session.Run()
		if e != nil {
			fmt.Printf("Error running ORT session: %s\n", e)
			return 1
		}
		timingStats.Finish()
		// 处理输出
		boxes := processOutput(modelSession.Output.GetData(), originalWidth,
			originalHeight)
		for i, box := range boxes {
			fmt.Printf("Box %d: %s\n", i, &box)
		}
	}
	timingStats.PrintStats()
	return 0
}

// loadImageFile 加载并解码图片文件
func loadImageFile(filePath string) (image.Image, error) {
	// 打开图片文件
	f, e := os.Open(filePath)
	if e != nil {
		return nil, fmt.Errorf("Error opening %s: %w", filePath, e)
	}
	defer f.Close()
	// 解码图片文件
	pic, _, e := image.Decode(f)
	if e != nil {
		return nil, fmt.Errorf("Error decoding %s: %w", filePath, e)
	}
	return pic, nil
}

// prepareInput 将输入图像预处理并填充到 YOLOv8 输入张量中
// 1. 将图像调整为 640x640 大小
// 2. 将像素值归一化到 [0,1] 范围
// 3. 分离 RGB 通道并填充到对应的张量通道中
func prepareInput(pic image.Image, dst *ort.Tensor[float32]) error {
	// 获取数据
	data := dst.GetData()
	// 计算通道大小
	channelSize := 640 * 640
	// 检查数据是否足够
	if len(data) < (channelSize * 3) {
		return fmt.Errorf("Destination tensor only holds %d floats, needs "+
			"%d (make sure it's the right shape!)", len(data), channelSize*3)
	}
	redChannel := data[0:channelSize]
	greenChannel := data[channelSize : channelSize*2]
	blueChannel := data[channelSize*2 : channelSize*3]

	// 使用 Lanczos3 算法将图像调整为 640x640
	pic = resize.Resize(640, 640, pic, resize.Lanczos3)
	// 遍历图像
	i := 0
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			// 获取像素值
			r, g, b, _ := pic.At(x, y).RGBA()
			// 归一化像素值
			redChannel[i] = float32(r>>8) / 255.0
			// 归一化像素值
			greenChannel[i] = float32(g>>8) / 255.0
			// 归一化像素值
			blueChannel[i] = float32(b>>8) / 255.0
			// 增加索引
			i++
		}
	}

	return nil
}

// getSharedLibPath 根据操作系统和架构返回对应的 ONNX Runtime 动态库路径
func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.dylib"
		}
		if runtime.GOARCH == "amd64" {
			return "../third_party/onnxruntime_amd64.dylib"
		}

	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "../third_party/onnxruntime_arm64.so"
		}
		return "../third_party/onnxruntime.so"
	}
	panic("Unable to find a version of the onnxruntime library supporting this system.")
}

// initSession 初始化 ONNX Runtime 会话
// 1. 设置动态库路径
// 2. 初始化运行环境
// 3. 创建输入输出张量
// 4. 配置会话选项（如 CoreML 加速）
// 5. 创建会话
func initSession() (*ModelSession, error) {
	// 设置动态库路径
	ort.SetSharedLibraryPath(getSharedLibPath())
	// 初始化运行环境
	err := ort.InitializeEnvironment()
	if err != nil {
		return nil, fmt.Errorf("Error initializing ORT environment: %w", err)
	}
	// 创建输入输出张量
	inputShape := ort.NewShape(1, 3, 640, 640)
	inputTensor, err := ort.NewEmptyTensor[float32](inputShape)
	if err != nil {
		return nil, fmt.Errorf("Error creating input tensor: %w", err)
	}
	// 创建输出张量
	outputShape := ort.NewShape(1, 84, 8400)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("Error creating output tensor: %w", err)
	}
	// 创建会话选项
	options, err := ort.NewSessionOptions()
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("Error creating ORT session options: %w", err)
	}
	defer options.Destroy()

	// 如果启用了 CoreML，则附加 CoreML 执行提供者
	if useCoreML {
		err = options.AppendExecutionProviderCoreML(0)
		if err != nil {
			inputTensor.Destroy()
			outputTensor.Destroy()
			return nil, fmt.Errorf("Error enabling CoreML: %w", err)
		}
	}
	// 创建会话
	session, err := ort.NewAdvancedSession(modelPath,
		[]string{"images"}, []string{"output0"},
		[]ort.ArbitraryTensor{inputTensor},  // 输入张量
		[]ort.ArbitraryTensor{outputTensor}, // 输出张量
		options)                             // 会话选项
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("Error creating ORT session: %w", err)
	}

	return &ModelSession{
		Session: session,
		Input:   inputTensor,
		Output:  outputTensor,
	}, nil
}

func (m *ModelSession) Destroy() {
	m.Session.Destroy()
	m.Input.Destroy()
	m.Output.Destroy()
}

// boundingBox 结构体表示检测到的目标边界框
type boundingBox struct {
	label      string  // 目标类别标签
	confidence float32 // 检测置信度
	x1, y1     float32 // 左上角坐标
	x2, y2     float32 // 右下角坐标
}

func (b *boundingBox) String() string {
	return fmt.Sprintf("Object %s (confidence %f): (%f, %f), (%f, %f)",
		b.label, b.confidence, b.x1, b.y1, b.x2, b.y2)
}

// 这会丢失精度，但请记住，boundingBox 已经缩放到原始图像的维度。因此，它只会失去边缘附近的分数像素。
func (b *boundingBox) toRect() image.Rectangle {
	return image.Rect(int(b.x1), int(b.y1), int(b.x2), int(b.y2)).Canon()
}

// 返回 b 的面积（以像素为单位），在转换为 image.Rectangle 之后。
func (b *boundingBox) rectArea() int {
	size := b.toRect().Size()
	return size.X * size.Y
}

func (b *boundingBox) intersection(other *boundingBox) float32 {
	r1 := b.toRect()
	r2 := other.toRect()
	intersected := r1.Intersect(r2).Canon().Size()
	return float32(intersected.X * intersected.Y)
}

func (b *boundingBox) union(other *boundingBox) float32 {
	intersectArea := b.intersection(other)
	totalArea := float32(b.rectArea() + other.rectArea())
	return totalArea - intersectArea
}

// 由于转换为 image.Image 库的积分矩形，这不会完全精确，但我们只使用它来估计哪些框重叠太多，所以一些不精确应该是可以的。
func (b *boundingBox) iou(other *boundingBox) float32 {
	return b.intersection(other) / b.union(other)
}

// processOutput 处理模型输出，生成目标检测结果
// 1. 遍历所有可能的检测框（8400个）
// 2. 对每个框计算80个类别的概率
// 3. 筛选置信度大于0.5的检测框
// 4. 将坐标转换回原始图像尺寸
// 5. 使用非极大值抑制(NMS)去除重叠框
func processOutput(output []float32, originalWidth,
	originalHeight int) []boundingBox {
	// 创建一个切片来保存所有检测框
	boundingBoxes := make([]boundingBox, 0, 8400)

	// 定义变量来保存当前类ID和概率
	var classID int
	// 定义变量来保存当前概率
	var probability float32

	// 遍历输出数组，考虑 8400 个索引
	for idx := 0; idx < 8400; idx++ {
		// 遍历 80 个类，找到概率最高的类
		probability = -1e9
		for col := 0; col < 80; col++ {
			// 获取当前概率
			currentProb := output[8400*(col+4)+idx]
			// 如果当前概率大于当前概率，则更新当前概率和类ID
			if currentProb > probability {
				// 更新当前概率
				probability = currentProb
				// 更新当前类ID
				classID = col
			}
		}

		// 如果概率小于 0.5，继续到下一个索引
		if probability < 0.5 {
			continue
		}

		// 提取边界框的坐标和维度
		xc, yc := output[idx], output[8400+idx]
		w, h := output[2*8400+idx], output[3*8400+idx]
		x1 := (xc - w/2) / 640 * float32(originalWidth)
		y1 := (yc - h/2) / 640 * float32(originalHeight)
		x2 := (xc + w/2) / 640 * float32(originalWidth)
		y2 := (yc + h/2) / 640 * float32(originalHeight)

		// 将边界框附加到结果
		boundingBoxes = append(boundingBoxes, boundingBox{
			label:      yoloClasses[classID],
			confidence: probability,
			x1:         x1,
			y1:         y1,
			x2:         x2,
			y2:         y2,
		})
	}

	// 按概率排序边界框
	sort.Slice(boundingBoxes, func(i, j int) bool {
		return boundingBoxes[i].confidence < boundingBoxes[j].confidence
	})

	// 定义一个切片来保存最终结果
	mergedResults := make([]boundingBox, 0, len(boundingBoxes))

	// 遍历排序的边界框，移除重叠
	for _, candidateBox := range boundingBoxes {
		overlapsExistingBox := false
		for _, existingBox := range mergedResults {
			if (&candidateBox).iou(&existingBox) > 0.7 {
				overlapsExistingBox = true
				break
			}
		}
		if !overlapsExistingBox {
			mergedResults = append(mergedResults, candidateBox)
		}
	}

	// 这仍然会按置信度排序
	return mergedResults
}

// yoloClasses 定义 COCO 数据集的80个类别标签
var yoloClasses = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}
