
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
SOFIE works in a parser-generator working architecture. With SOFIE, the user gets an ONNX, [Keras](https://github.com/root-project/root/blob/master/tmva/pymva/src/RModelParser_Keras.cxx) and a [PyTorch](https://github.com/root-project/root/blob/master/tmva/pymva/src/RModelParser_PyTorch.cxx) parser for translating models in respective formats into SOFIE's internal representation.

From ROOT command line, or in a ROOT macro, we can proceed with an ONNX model:

	using namespace TMVA::Experimental;
	SOFIE::RModelParser_ONNX parser;
	SOFIE::RModel model = parser.Parse(“./example_model.onnx”);
	model.Generate();
	model.OutputGenerated(“./example_output.hxx”);

And an C++ header file and a `.dat` file containing the model weights will be generated. You can also use

	model.PrintRequiredInputTensors();

to check the required size and type of input tensor for that particular model, and use

	model.PrintInitializedTensors();

to check the tensors (weights) already included in the model.

To use the generated inference code:

	#include "example_output.hxx"
	float input[INPUT_SIZE];

    // Generated header file shall contain a Session class which requires initialization to load the corresponding weights.
    TMVA_SOFIE_example_model::Session s("example_model.dat")

    // Once instantiated the session object's infer method can be used
	std::vector<float> out = s.infer(input);

With the default settings, the weights are contained in a separate binary file, but if the user instead wants them to be in the generated header file itself, they can use approproiate generation options. 
    
    model.Generate(Options::kNoWeightFile);

Other such options includes `Options::kNoSession` (for not generating the Session class, and instead keeping the infer function independent).
