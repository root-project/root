
# TMVA SOFIE

ROOT/TMVA SOFIE (___System for Optimized Fast Inference code Emit___) generates C++ functions easily invokable for the fast inference of trained neural network models. It takes ONNX model files as inputs and produces C++ header files that can be included and utilized in a “plug-and-go” style.

This is a new development in TMVA and is currently in early experimental stage. Bug reports and suggestions for improvements are [warmly welcomed](mailto:Lorenzo.Moneta@cern.ch).


## Prerequisite
- Protobuf 3.0 or higher (for input of ONNX model files)
- BLAS or Eigen (for execution of the generated code for inference)

## Installation

Build ROOT with the cmake option tmva-sofie enabled.

```bash
cmake ../root -Dtmva-sofie=ON
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

Other such options includes `Options::kNoSession` (for not generating the Session class, and instead keeping the infer function independent).
SOFIE also supports generating inference code with RDataFrame as inputs, refer to the tutorials below for examples.

## Supported ONNX operators

Here is the updated list of supported ONNX operators

- [x] Add
- [x] AveragePool
- [x] BatchNormalization
- [x] Cast
- [x] Concat
- [x] Constant
- [x] ConstantOfShape
- [x] Conv
- [x] ConvTranspose
- [x] Elu
- [x] Equal
- [x] Erf
- [x] Exp
- [x] Expand
- [x] EyeLike
- [x] Flatten
- [x] Gather
- [x] Gemm
- [x] GlobalAveragePool
- [x] Greater
- [x] GreaterOrEqual
- [x] GRU
- [x] Identity
- [x] If
- [x] LayerNormalization
- [x] LeakyRelu
- [x] Less
- [x] LessOrEqual
- [x] Log
- [x] LSTM
- [x] MatMul
- [x] Max
- [x] MaxPool
- [x] Mean
- [x] Min
- [x] Mul
- [x] Neg
- [x] Pool
- [x] Pow
- [x] Range
- [x] Reciprocal
- [x] ReduceMean
- [x] ReduceProd
- [x] ReduceSum
- [x] ReduceSumSquare
- [x] Relu
- [x] Reshape
- [x] RNN
- [x] Selu
- [x] Sigmoid
- [x] Slice
- [x] Softmax
- [x] Split
- [x] Sqrt
- [x] Squeeze
- [x] Tanh
- [x] Tile
- [x] TopK
- [x] Transpose
- [x] Unsqueeze

The above operators are supported for tensors of the following types:

- [x] float
- [x] double
- [x] int32
- [x] int64
- [x] bool (for comparison operators)




## Additional Links

- **Tutorials**
    - [TMVA_SOFIE_Inference](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_Inference.py)
    - [TMVA_SOFIE_Keras](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_Keras.C)
    - [TMVA_SOFIE_Keras_HiggsModel](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_Keras_HiggsModel.C)
    - [TMVA_SOFIE_ONNX](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_ONNX.C)
    - [TMVA_SOFIE_PyTorch](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_PyTorch.C)
    - [TMVA_SOFIE_RDataFrame](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_RDataFrame.C)
    - [TMVA_SOFIE_RDataFrame](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_RDataFrame.py)
    - [TMVA_SOFIE_RDataFrame_JIT](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_RDataFrame_JIT.C)
    - [TMVA_SOFIE_RSofieReader](https://github.com/root-project/root/blob/master/tutorials/machine_learning/TMVA_SOFIE_RSofieReader.C)

