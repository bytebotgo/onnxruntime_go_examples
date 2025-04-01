package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/yalue/onnxruntime_go"
)

// Timestamp 时间戳结构
type Timestamp struct {
	Start int // 开始时间（采样点）
	End   int // 结束时间（采样点）
}

// VadIterator 语音活动检测迭代器
type VadIterator struct {
	session             *onnxruntime_go.AdvancedSession
	input               *onnxruntime_go.Tensor[float32]
	output              *onnxruntime_go.Tensor[float32]
	stateN              *onnxruntime_go.Tensor[float32]
	inputData           []float32
	stateData           []float32
	srData              []int64
	inputNodeDims       []int64
	stateNodeDims       []int64
	srNodeDims          []int64
	effectiveWindowSize int
	windowSize          int
	sampleRate          int
	speechPadMs         int
	speechPadSamples    int
	threshold           float32
	minSpeechDuration   int
	minSilenceDuration  int
	maxSpeechDuration   int
	srPerMs             int
	windowSizeSamples   int
	contextSamples      int
	context             []float32
	triggered           bool
	tempEnd             int
	currentSample       int
	prevEnd             int
	nextStart           int
	speeches            []Timestamp
	currentSpeech       Timestamp
	sr                  *onnxruntime_go.Tensor[int64]
}

// NewVadIterator 创建新的语音活动检测迭代器
func NewVadIterator(modelPath string, sampleRate int, threshold float32, windowSizeMs int, speechPadMs int, minSpeechMs int, minSilenceMs int, maxSpeechSec float32) (*VadIterator, error) {
	vad := &VadIterator{
		sampleRate:         sampleRate,
		threshold:          threshold,
		speechPadMs:        speechPadMs,
		minSpeechDuration:  minSpeechMs * sampleRate / 1000,
		minSilenceDuration: minSilenceMs * sampleRate / 1000,
		maxSpeechDuration:  int(float32(sampleRate)*maxSpeechSec) - windowSizeMs*sampleRate/1000 - 2*speechPadMs*sampleRate/1000,
	}

	// 计算采样率相关参数
	vad.srPerMs = sampleRate / 1000
	vad.windowSizeSamples = windowSizeMs * vad.srPerMs
	vad.contextSamples = 64
	vad.effectiveWindowSize = vad.windowSizeSamples + vad.contextSamples

	// 初始化输入输出维度
	vad.inputNodeDims = []int64{1, int64(vad.effectiveWindowSize)}
	vad.stateNodeDims = []int64{2, 1, 128}
	vad.srNodeDims = []int64{1}

	// 初始化状态和上下文
	vad.stateData = make([]float32, 2*1*128)
	vad.context = make([]float32, vad.contextSamples)

	// 初始化采样率张量
	vad.srData = make([]int64, 1)
	vad.srData[0] = int64(vad.sampleRate)
	var err error
	vad.sr, err = onnxruntime_go.NewTensor(vad.srNodeDims, vad.srData)
	if err != nil {
		return nil, fmt.Errorf("创建采样率张量失败: %w", err)
	}

	// 初始化 ONNX Runtime 会话
	err = vad.initSession(modelPath)
	if err != nil {
		return nil, err
	}

	return vad, nil
}

// initSession 初始化 ONNX Runtime 会话
func (v *VadIterator) initSession(modelPath string) error {
	// 创建输入张量
	var err error
	inputShape := []int64{1, int64(v.effectiveWindowSize)}
	inputData := make([]float32, v.effectiveWindowSize)
	v.input, err = onnxruntime_go.NewTensor[float32](inputShape, inputData)
	if err != nil {
		return fmt.Errorf("创建输入张量失败: %w", err)
	}

	// 创建状态张量
	stateShape := []int64{2, 1, 128}
	stateData := make([]float32, 2*1*128)
	v.stateN, err = onnxruntime_go.NewTensor[float32](stateShape, stateData)
	if err != nil {
		return fmt.Errorf("创建状态张量失败: %w", err)
	}

	// 创建输出张量
	outputShape := []int64{1, 1}
	outputData := make([]float32, 1)
	v.output, err = onnxruntime_go.NewTensor[float32](outputShape, outputData)
	if err != nil {
		return fmt.Errorf("创建输出张量失败: %w", err)
	}

	// 创建采样率张量
	srShape := []int64{1}
	srData := make([]int64, 1)
	srData[0] = int64(v.sampleRate)
	v.sr, err = onnxruntime_go.NewTensor[int64](srShape, srData)
	if err != nil {
		return fmt.Errorf("创建采样率张量失败: %w", err)
	}

	// 创建会话选项
	options, err := onnxruntime_go.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("创建会话选项失败: %w", err)
	}
	defer options.Destroy()

	// 创建 ONNX Runtime 会话
	v.session, err = onnxruntime_go.NewAdvancedSession(
		modelPath,
		[]string{"input", "state", "sr"},
		[]string{"output", "stateN"},
		[]onnxruntime_go.ArbitraryTensor{v.input, v.stateN, v.sr},
		[]onnxruntime_go.ArbitraryTensor{v.output, v.stateN},
		options,
	)
	if err != nil {
		return fmt.Errorf("创建会话失败: %w", err)
	}

	return nil
}

// predict 执行一次推理
func (v *VadIterator) predict(dataChunk []float32) (float32, error) {
	// 构建新的输入数据：前contextSamples个样本来自context，后面是当前块
	newData := make([]float32, v.effectiveWindowSize)
	copy(newData[:v.contextSamples], v.context)
	copy(newData[v.contextSamples:], dataChunk)
	v.inputData = newData

	// 复制输入数据到输入张量
	copy(v.input.GetData(), v.inputData)

	// 运行推理
	err := v.session.Run()
	if err != nil {
		return 0, fmt.Errorf("运行推理失败: %w", err)
	}

	// 获取输出结果
	outputData := v.output.GetData()
	if len(outputData) == 0 {
		return 0, fmt.Errorf("输出数据为空")
	}

	// 更新状态
	stateNData := v.stateN.GetData()
	if len(stateNData) != len(v.stateData) {
		return 0, fmt.Errorf("状态数据长度不匹配: 期望 %d, 实际 %d", len(stateNData), len(v.stateData))
	}
	copy(v.stateData, stateNData)

	// 更新当前采样点
	v.currentSample += v.windowSizeSamples

	// 处理检测结果
	speechProb := outputData[0]
	if speechProb >= v.threshold {
		if v.tempEnd != 0 {
			v.tempEnd = 0
			if v.nextStart < v.prevEnd {
				v.nextStart = v.currentSample - v.windowSizeSamples
			}
		}
		if !v.triggered {
			v.triggered = true
			v.currentSpeech.Start = v.currentSample - v.windowSizeSamples
		}
		// 更新上下文
		copy(v.context, newData[len(newData)-v.contextSamples:])
		return speechProb, nil
	}

	// 如果语音段太长
	if v.triggered && ((v.currentSample - v.currentSpeech.Start) > v.maxSpeechDuration) {
		if v.prevEnd > 0 {
			v.currentSpeech.End = v.prevEnd
			v.speeches = append(v.speeches, v.currentSpeech)
			v.currentSpeech = Timestamp{}
			if v.nextStart < v.prevEnd {
				v.triggered = false
			} else {
				v.currentSpeech.Start = v.nextStart
			}
			v.prevEnd = 0
			v.nextStart = 0
			v.tempEnd = 0
		} else {
			v.currentSpeech.End = v.currentSample
			v.speeches = append(v.speeches, v.currentSpeech)
			v.currentSpeech = Timestamp{}
			v.prevEnd = 0
			v.nextStart = 0
			v.tempEnd = 0
			v.triggered = false
		}
		// 更新上下文
		copy(v.context, newData[len(newData)-v.contextSamples:])
		return speechProb, nil
	}

	if (speechProb >= (v.threshold - 0.15)) && (speechProb < v.threshold) {
		// 当语音概率暂时下降但仍然在语音中时，只更新上下文
		copy(v.context, newData[len(newData)-v.contextSamples:])
		return speechProb, nil
	}

	if speechProb < (v.threshold - 0.15) {
		if v.triggered {
			if v.tempEnd == 0 {
				v.tempEnd = v.currentSample
			}
			if v.currentSample-v.tempEnd > v.minSilenceDuration {
				v.prevEnd = v.tempEnd
			}
			if (v.currentSample - v.tempEnd) >= v.minSilenceDuration {
				v.currentSpeech.End = v.tempEnd
				if v.currentSpeech.End-v.currentSpeech.Start > v.minSpeechDuration {
					v.speeches = append(v.speeches, v.currentSpeech)
					v.currentSpeech = Timestamp{}
					v.prevEnd = 0
					v.nextStart = 0
					v.tempEnd = 0
					v.triggered = false
				}
			}
		}
		// 更新上下文
		copy(v.context, newData[len(newData)-v.contextSamples:])
		return speechProb, nil
	}

	return speechProb, nil
}

// Process 处理整个音频输入
func (v *VadIterator) Process(inputWav []float32) error {
	v.resetStates()
	audioLengthSamples := len(inputWav)

	// 按窗口大小处理音频
	for j := 0; j < audioLengthSamples; j += v.windowSizeSamples {
		if j+v.windowSizeSamples > audioLengthSamples {
			break
		}
		chunk := inputWav[j : j+v.windowSizeSamples]
		_, err := v.predict(chunk)
		if err != nil {
			return err
		}
	}

	// 处理最后一个语音段
	if v.currentSpeech.Start >= 0 {
		v.currentSpeech.End = audioLengthSamples
		v.speeches = append(v.speeches, v.currentSpeech)
		v.currentSpeech = Timestamp{}
		v.prevEnd = 0
		v.nextStart = 0
		v.tempEnd = 0
		v.triggered = false
	}

	return nil
}

// GetSpeechTimestamps 获取检测到的语音时间戳
func (v *VadIterator) GetSpeechTimestamps() []Timestamp {
	return v.speeches
}

// resetStates 重置内部状态
func (v *VadIterator) resetStates() {
	for i := range v.stateData {
		v.stateData[i] = 0
	}
	v.triggered = false
	v.tempEnd = 0
	v.currentSample = 0
	v.prevEnd = 0
	v.nextStart = 0
	v.speeches = v.speeches[:0]
	v.currentSpeech = Timestamp{}
	for i := range v.context {
		v.context[i] = 0
	}
}

// getSharedLibPath 根据操作系统和架构返回对应的 ONNX Runtime 动态库路径
func getDefaultSharedLibPath() string {
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
	fmt.Printf("Unable to determine a path to the onnxruntime shared library"+
		" for OS \"%s\" and architecture \"%s\".\n", runtime.GOOS,
		runtime.GOARCH)
	return ""
}

func main() {
	// 设置动态库路径
	onnxruntime_go.SetSharedLibraryPath(getDefaultSharedLibPath())

	// 初始化 ONNX Runtime
	if err := onnxruntime_go.InitializeEnvironment(); err != nil {
		log.Fatalf("初始化 ONNX Runtime 失败: %v", err)
	}
	defer onnxruntime_go.DestroyEnvironment()

	// 读取 WAV 文件
	wavReader := &WavReader{}
	if err := wavReader.Open("./audio/files_ru.wav"); err != nil {
		log.Fatalf("打开音频文件失败: %v", err)
	}

	// 检查音频格式
	if wavReader.SampleRate() != 16000 || wavReader.NumChannels() != 1 {
		log.Fatal("音频格式不正确，需要 16000Hz 单声道 WAV 文件")
	}

	// 创建 VAD 迭代器
	vad, err := NewVadIterator(
		"./model/silero_vad.onnx",
		16000, // 采样率
		0.5,   // 阈值
		32,    // 窗口大小（毫秒）
		30,    // 语音填充（毫秒）
		250,   // 最小语音持续时间（毫秒）
		100,   // 最小静音持续时间（毫秒）
		30.0,  // 最大语音持续时间（秒）
	)
	if err != nil {
		log.Fatalf("创建 VAD 迭代器失败: %v", err)
	}

	// 处理音频
	if err := vad.Process(wavReader.Data()); err != nil {
		log.Fatalf("处理音频失败: %v", err)
	}

	// 获取语音时间戳
	stamps := vad.GetSpeechTimestamps()

	// 将时间戳转换为秒并四舍五入到一位小数
	sampleRateFloat := 16000.0
	for _, stamp := range stamps {
		startSec := float64(stamp.Start) / sampleRateFloat
		endSec := float64(stamp.End) / sampleRateFloat
		fmt.Printf("检测到语音从 %.1f 秒到 %.1f 秒\n", startSec, endSec)
	}

	// 重置内部状态
	vad.resetStates()
}
