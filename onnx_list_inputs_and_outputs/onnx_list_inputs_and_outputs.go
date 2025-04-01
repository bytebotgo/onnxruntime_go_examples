// This is a simple command-line utility that takes a single .onnx file and
// lists the inputs and outputs to it.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

// 有关更多评论，请参见 sum_and_difference 示例。

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

// 打印 onnx 格式网络的输入和输出到 stdout。
func showNetworkInputsAndOutputs(libPath, networkPath string) error {
	ort.SetSharedLibraryPath(libPath)
	e := ort.InitializeEnvironment()
	if e != nil {
		return fmt.Errorf("Error initializing onnxruntime library: %w", e)
	}
	inputs, outputs, e := ort.GetInputOutputInfo(networkPath)
	if e != nil {
		return fmt.Errorf("Error getting input and output info for %s: %w",
			networkPath, e)
	}
	fmt.Printf("%d inputs to %s:\n", len(inputs), networkPath)
	for i, v := range inputs {
		fmt.Printf("  Index %d: %s\n", i, &v)
	}
	fmt.Printf("%d outputs from %s:\n", len(outputs), networkPath)
	for i, v := range outputs {
		fmt.Printf("  Index %d: %s\n", i, &v)
	}
	return nil
}

func run() int {
	var onnxruntimeLibPath string
	var onnxNetworkPath string
	flag.StringVar(&onnxruntimeLibPath, "onnxruntime_lib",
		getDefaultSharedLibPath(),
		"The path to the onnxruntime shared library for your system.")
	flag.StringVar(&onnxNetworkPath, "onnx_file", "",
		"The path to the .onnx file to load.")
	flag.Parse()
	if onnxruntimeLibPath == "" {
		fmt.Println("You must specify a path to the onnxruntime shared " +
			"on your system. Run with -help for more information.")
		return 1
	}
	if onnxNetworkPath == "" {
		fmt.Println("You must specify a .onnx network to list the inputs and" +
			" outputs for. Run with -help for more information.")
	}
	e := showNetworkInputsAndOutputs(onnxruntimeLibPath, onnxNetworkPath)
	if e != nil {
		fmt.Printf("Error getting network inputs and outputs: %s\n", e)
		return 1
	}
	return 0
}

func main() {
	os.Exit(run())
}
