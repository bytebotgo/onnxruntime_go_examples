package main

import (
	"encoding/binary"
	"fmt"
	"os"
)

// WavHeader WAV 文件头结构
type WavHeader struct {
	Riff           [4]byte // "RIFF"
	Size           uint32  // 文件大小
	Wave           [4]byte // "WAVE"
	Fmt            [4]byte // "fmt "
	FmtSize        uint32  // fmt 块大小
	Format         uint16  // 音频格式
	Channels       uint16  // 声道数
	SampleRate     uint32  // 采样率
	BytesPerSecond uint32  // 每秒字节数
	BlockSize      uint16  // 数据块大小
	BitsPerSample  uint16  // 采样位数
	Data           [4]byte // "data"
	DataSize       uint32  // 数据大小
}

// WavReader WAV 文件读取器
type WavReader struct {
	header        WavHeader
	data          []float32
	numChannels   int
	sampleRate    int
	bitsPerSample int
	numSamples    int
}

// Open 打开 WAV 文件
func (w *WavReader) Open(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("打开文件失败: %w", err)
	}
	defer file.Close()

	// 读取文件头
	if err := binary.Read(file, binary.LittleEndian, &w.header); err != nil {
		return fmt.Errorf("读取文件头失败: %w", err)
	}

	// 检查 fmt 块大小
	if w.header.FmtSize < 16 {
		return fmt.Errorf("WAV 文件格式错误：fmt 块大小小于 16")
	}

	// 如果 fmt 块大小大于 16，跳过额外的字节
	if w.header.FmtSize > 16 {
		offset := 44 - 8 + w.header.FmtSize - 16
		file.Seek(int64(offset), 0)
	}

	// 跳过 fmt 和 data 块之间的子块
	for string(w.header.Data[:]) != "data" {
		file.Seek(int64(w.header.DataSize), 1)
		if _, err := file.Read(w.header.Data[:]); err != nil {
			return fmt.Errorf("读取数据块标记失败: %w", err)
		}
	}

	// 如果数据大小为 0，使用文件剩余部分
	if w.header.DataSize == 0 {
		offset, _ := file.Seek(0, 1)
		file.Seek(0, 2)
		end, _ := file.Seek(0, 1)
		w.header.DataSize = uint32(end - offset)
		file.Seek(offset, 0)
	}

	// 设置音频参数
	w.numChannels = int(w.header.Channels)
	w.sampleRate = int(w.header.SampleRate)
	w.bitsPerSample = int(w.header.BitsPerSample)
	numData := int(w.header.DataSize) / (w.bitsPerSample / 8)
	w.numSamples = numData / w.numChannels

	// 读取音频数据
	w.data = make([]float32, numData)
	switch w.bitsPerSample {
	case 8:
		for i := 0; i < numData; i++ {
			var sample uint8
			if err := binary.Read(file, binary.LittleEndian, &sample); err != nil {
				return fmt.Errorf("读取 8 位采样数据失败: %w", err)
			}
			w.data[i] = float32(sample) / 32768.0
		}
	case 16:
		for i := 0; i < numData; i++ {
			var sample int16
			if err := binary.Read(file, binary.LittleEndian, &sample); err != nil {
				return fmt.Errorf("读取 16 位采样数据失败: %w", err)
			}
			w.data[i] = float32(sample) / 32768.0
		}
	case 32:
		if w.header.Format == 1 { // S32
			for i := 0; i < numData; i++ {
				var sample int32
				if err := binary.Read(file, binary.LittleEndian, &sample); err != nil {
					return fmt.Errorf("读取 32 位采样数据失败: %w", err)
				}
				w.data[i] = float32(sample) / 32768.0
			}
		} else if w.header.Format == 3 { // IEEE-float
			for i := 0; i < numData; i++ {
				var sample float32
				if err := binary.Read(file, binary.LittleEndian, &sample); err != nil {
					return fmt.Errorf("读取 32 位浮点采样数据失败: %w", err)
				}
				w.data[i] = sample
			}
		} else {
			return fmt.Errorf("不支持的量化位数: %d", w.bitsPerSample)
		}
	default:
		return fmt.Errorf("不支持的量化位数: %d", w.bitsPerSample)
	}

	return nil
}

// NumChannels 返回声道数
func (w *WavReader) NumChannels() int {
	return w.numChannels
}

// SampleRate 返回采样率
func (w *WavReader) SampleRate() int {
	return w.sampleRate
}

// BitsPerSample 返回采样位数
func (w *WavReader) BitsPerSample() int {
	return w.bitsPerSample
}

// NumSamples 返回采样点数
func (w *WavReader) NumSamples() int {
	return w.numSamples
}

// Data 返回音频数据
func (w *WavReader) Data() []float32 {
	return w.data
}
