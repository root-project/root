
## About

ROOT/TMVA SOFIE (“System for Optimized Fast Inference code Emit”) generates C++ functions easily invokable for the fast inference of trained neural network models. It takes ONNX model files as inputs and produces C++ header files that can be included and utilized in a “plug-and-go” style.

This is a new development in TMVA and is currently in early experimental stage. Bug reports and suggestions for improvements are [warmly welcomed](mailto:s.an@cern.ch).


## Prerequisite

- Protobuf 3.0 or higher (for input of ONNX model files)
- BLAS or Eigen (for execution of the generated code for inference)

## Installation

Build ROOT with the cmake option tmva-sofie enabled.

    $ cmake ../root -Dtmva-sofie=ON
    $ make -j8


## Usage


From ROOT command line, or in a ROOT macro:
```C++
using namespace TMVA::Experimental;
SOFIE::RModelParser_ONNX parser;
SOFIE::RModel model = parser.Parse(“./example_model.onnx”);
model.Generate();
model.OutputGenerated(“./example_output.hxx”);
```
And an C++ header file will be generated. You can also use
```C++
model.PrintRequiredInputTensors();
```
to check the required size and type of input tensor for that particular model, and use
```C++
model.PrintInitializedTensors();
```
to check the tensors (weights) already included in the model.

To use the generated inference code:

```C++
#include "example_output.hxx"
float input[INPUT_SIZE];
std::vector<float> out = TMVA_SOFIE_example_model::infer(input);
```

## Profiling
For profiling purposes, SOFIE can also generate inferece models that keeps track of the 
time elapsed to execute each one of the operators. This is done thought _code instrumentation_.

```C++
// Assuming 'model' is a pre-created SOFIE::RModel
// SOFIE::RModel model = ...
SOFIE::RModelProfiler profiler(model);
profiler.Generate();
model.OutputGenerated(“./example_output_prof.hxx”);
```
Once the model has been created, we can include it in our code and start profiling the inference.
By running `infer` at least once, the profiler will collect the microseconds that each operator took.

The use will also be able to use utility functions like `GetOpAvgTime()` and `GetOpAvgTime()`.
```C++
#include "example_output_prof.hxx"

for (int i = 0; i < 1000; ++i)
    TMVA_SOFIE_example_model::infer(input);

auto res = TMVA_SOFIE_example_model::profiler_results;
res["Gemm_0"][0];		// Gemm_0 in the first run of 'infer'
res["Relu_5"][2];		// Relu_5 in the third run of 'infer'

auto avg = TMVA_SOFIE_example_model::GetOpAvgTime();
avg["Gemm_14"];			// Average time [us] of operator Gemm_14

auto var = TMVA_SOFIE_example_model::GetOpVariance();
var["Gemm_14"];         // Time variance of operator Gemm_14

```