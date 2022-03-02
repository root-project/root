
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

	using namespace TMVA::Experimental;
	SOFIE::RModelParser_ONNX parser;
	SOFIE::RModel model = parser.Parse(“./example_model.onnx”);
	model.Generate();
	model.OutputGenerated(“./example_output.hxx”);

And an C++ header file will be generated. You can also use

	model.PrintRequiredInputTensors();

to check the required size and type of input tensor for that particular model, and use

	model.PrintInitializedTensors();

to check the tensors (weights) already included in the model.

To use the generated inference code:

	#include "example_output.hxx"
	float input[INPUT_SIZE];
	std::vector<float> out = TMVA_SOFIE_example_model::infer(input);
