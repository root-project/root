
# TMVA SOFIE

ROOT/TMVA SOFIE (___System for Optimized Fast Inference code Emit___) generates C++ functions easily invokable for the fast inference of trained neural network models. It takes ONNX model files as inputs and produces C++ header files that can be included and utilized in a “plug-and-go” style.

This is a new development in TMVA and is currently in early experimental stage. Bug reports and suggestions for improvements are [warmly welcomed](mailto:Lorenzo.Moneta@cern.ch).


## Prerequisite
- BLAS or Eigen (for execution of the generated code for inference)

## Installation

SOFIE is built as part of TMVA, so it is enabled whenever ROOT is built with
the `tmva` cmake option (which is ON by default).

```bash
cmake ../root -Dtmva=ON
make -j8
```

## Usage
SOFIE works in a parser-generator working architecture. With SOFIE, the user gets an [ONNX](https://github.com/root-project/root/tree/master/tmva/sofie_parsers), [Keras](https://github.com/root-project/root/blob/master/tmva/pymva/src/RModelParser_Keras.cxx) and a [PyTorch](https://github.com/root-project/root/blob/master/tmva/pymva/src/RModelParser_PyTorch.cxx) parser for translating models in respective formats into SOFIE's internal representation.

From ROOT command line, or in a ROOT macro, we can proceed with an ONNX model:

```c++
using namespace TMVA::Experimental;
SOFIE::RModelParser_ONNX parser;
SOFIE::RModel model = parser.Parse(“./example_model.onnx”);
model.Generate();
model.OutputGenerated(“./example_output.hxx”);
```

And an C++ header file and a `.dat` file containing the model weights will be generated. You can also use

```c++
model.PrintRequiredInputTensors();
```

to check the required size and type of input tensor for that particular model, and use

```c++
model.PrintInitializedTensors();
```

to check the tensors (weights) already included in the model.

To use the generated inference code:

```c++
#include "example_output.hxx"
float input[INPUT_SIZE];
std::vector<float> out = TMVA_SOFIE_example_model::infer(input);

// Generated header file shall contain a Session class which requires initialization to load the corresponding weights.
TMVA_SOFIE_example_model::Session s("example_model.dat")

// Once instantiated the session object's infer method can be used
std::vector<float> out = s.infer(input);
```

With the default settings, the weights are contained in a separate binary file, but if the user instead wants them to be in the generated header file itself, they can use approproiate generation options.

```c++
model.Generate(Options::kNoWeightFile);
```

By default the separate weight file uses a simple text format (`*.dat`). A
binary alternative is the [safetensors](https://huggingface.co/docs/safetensors)
format (`*.safetensors`), which stores the weights as raw little-endian data
behind a small JSON header. It loads faster, round-trips the values bit-exactly,
and can be inspected with the standard Python and Rust safetensors tooling:

```c++
model.Generate(Options::kSafetensorsWeightFile);
```

SOFIE also supports generating inference code with RDataFrame as inputs, refer to the tutorials below for examples.

## Supported ONNX operators

Here is the updated list of supported ONNX operators. You can obtain this list by doing
```cpp
using namespace TMVA::Experimental;
SOFIE::RModelParser_ONNX parser;
std::vector<std::string> supportedOperators = parser.GetRegisteredOperators();
```

- [x] Abs
- [x] Add
- [x] AveragePool
- [x] BatchNormalization
- [x] Cast
- [x] Concat
- [x] Constant
- [x] ConstantOfShape
- [x] Conv
- [x] ConvTranspose
- [x] Cos
- [x] Div
- [x] Einsum
- [x] Elu
- [x] Equal
- [x] Erf
- [x] Exp
- [x] Expand
- [x] EyeLike
- [x] Flatten
- [x] GRU
- [x] Gather
- [x] Gemm
- [x] GlobalAveragePool
- [x] Greater
- [x] GreaterOrEqual
- [x] GRU
- [x] Identity
- [x] If
- [x] LSTM
- [x] LayerNormalization
- [x] LeakyRelu
- [x] Less
- [x] LessOrEqual
- [x] Log
- [x] MatMul
- [x] Max
- [x] MaxPool
- [x] Mean
- [x] Min
- [x] Mul
- [x] Neg
- [x] Pad
- [x] Pow
- [x] RNN
- [x] RandomNormal
- [x] RandomNormalLike
- [x] RandomUniform
- [x] RandomUniformLike
- [x] Range
- [x] Reciprocal
- [x] ReduceMean
- [x] ReduceProd
- [x] ReduceSum
- [x] ReduceSumSquare
- [x] Relu
- [x] Reshape
- [x] ScatterElements
- [x] Selu
- [x] Shape
- [x] Sigmoid
- [x] Sin
- [x] Slice
- [x] Softmax
- [x] Split
- [x] Sqrt
- [x] Squeeze
- [x] Sub
- [x] Sum
- [x] Tanh
- [x] Tile
- [x] TopK
- [x] Transpose
- [x] Unsqueeze
- [x] Where

The above operators are supported for tensors of the following types:

- [x] float
- [x] double
- [x] int32
- [x] int64
- [x] bool (for comparison operators)

You can also check your model whether all operators are implemented by doing the following:
```c++
using namespace TMVA::Experimental;
SOFIE::RModelParser_ONNX parser;
parser.CheckModel("example_model.ONNX");
```



## Additional Links

- **Tutorials**
    - [TMVA_SOFIE_Inference](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_Inference.py)
    - [TMVA_SOFIE_ONNX](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_ONNX.C)
    - [TMVA_SOFIE_PyTorch_HiggsModel](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_PyTorch_HiggsModel.py)
    - [TMVA_SOFIE_RDataFrame](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_RDataFrame.C)
    - [TMVA_SOFIE_RDataFrame](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_RDataFrame.py)
    - [TMVA_SOFIE_RDataFrame_JIT](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_RDataFrame_JIT.C)
    - [TMVA_SOFIE_RSofieReader](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_RSofieReader.C)


## Experimental CUDA/HIP dense inference

GPU parsing uses SOFIE's existing operator registry and builds an `RModel`.
`RModel::MakeGPUModel()` runs the existing operator initialization and shape
inference, then dispatches `ROperator::LowerGPU()`. The existing Gemm and Relu
operators implement the lowering; other operators reject it explicitly.
The GPU execution plan currently remains a sequential dense-layer plan.
FP16 constants are promoted exactly to FP32 for the shared operator representation;
the execution plan retains FP16 input, output and intermediate tensor precision.
Direct conversion of an existing FP32 `RModel` is also supported.


`RModelParser_ONNX::ParseGPU()` parses sequential, rank-two `Gemm`/`Relu`
networks with embedded FP32 or FP16 weights. `Gemm` supports constant matrix
weights, optional vector bias, alpha/beta, transB=0/1 and transA=0. Other graphs
are rejected rather than falling back to CPU. FP16 uses FP32 accumulation and
rounds every layer output to FP16. This is an initial scalar-kernel implementation,
not a tuned BLAS/tensor-core backend.

```cpp
#include <TMVA/RModelParser_ONNX.hxx>

auto model = TMVA::Experimental::SOFIE::RModelParser_ONNX{}.ParseGPU("network.onnx");
using Model = TMVA::Experimental::SOFIE::RGPUModel;
model.Compile({Model::Backend::HIP, "/path/to/hipcc", "gfx90a"});
auto session = model.CreateSession(nativeStream, deviceId, maxBatch);
// The framework allocates an aligned device arena of session->WorkspaceSize() bytes.
session->SetWorkspace(deviceArena, arenaBytes);
session->Infer(deviceInput, deviceOutput, batch);
```

Use `Backend::CUDA`, `nvcc`, and the device architecture (e.g. `sm_80`) for CUDA.
`Compile()` runs the compiler once using an argument vector, loads a private
shared library, and removes its temporary files. The runtime compiler must be
available on the execution host; ROOT itself needs no GPU toolkit to build this
host API. Compilation currently requires a POSIX host. Compiler failures include
the compiler output. No compilation takes place in `Infer()`.

Streams are passed as opaque native CUDA/HIP stream handles. The current device
must match the supplied device ID. The arena is 256-byte aligned and contains
weights plus two intermediate buffers sized for `maxBatch`. Input, output and
arena must be disjoint. Their precision must match the model. No CUDA/HIP device
allocation is emitted. `SetWorkspace()` uploads weights on the saved stream;
`Infer()` enqueues work without synchronizing. GPU faults are observed by the
caller's normal stream/event synchronization. The framework must finish queued
work before destroying a session or reclaiming its storage. Separate concurrent
streams need separate sessions and workspaces; they may share a compiled model.

Tests (not run as part of implementation): with ROOT testing enabled,
`TestSofieGPU` checks parsing, rejection and capacity handling without a GPU.
Set `SOFIE_GPU_TEST_BACKEND=CUDA` or `HIP` to additionally build
`TestSofieGPUDevice`, which needs the corresponding toolkit and compiler. It
compiles models when the test runs and checks six-layer FP32/FP16 inference,
external nonblocking streams, several batch sizes, and invalid workspaces.
The device test skips when no GPU is available.
