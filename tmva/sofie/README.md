
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
SOFIE also supports generating inference code with RDataFrame as inputs, refer to the tutorials below for examples.

  
## Additional Links

- **Tutorials**
	- [TMVA_SOFIE_Inference](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_Inference.py)
	- [TMVA_SOFIE_Keras](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_Keras.C)
	- [TMVA_SOFIE_Keras_HiggsModel](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_Keras_HiggsModel.C)
	- [TMVA_SOFIE_ONNX](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_ONNX.C)
	- [TMVA_SOFIE_PyTorch](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_PyTorch.C)
	- [TMVA_SOFIE_RDataFrame](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_RDataFrame.C)
	- [TMVA_SOFIE_RDataFrame](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_RDataFrame.py)
	- [TMVA_SOFIE_RDataFrame_JIT](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_RDataFrame_JIT.C)
	- [TMVA_SOFIE_RSofieReader](https://github.com/root-project/root/blob/master/tutorials/tmva/TMVA_SOFIE_RSofieReader.C)

# TMVA SOFIE-SYCL
SOFIE SYCL extends the current functionality of SYCL to generating SYCL inference code that can be run on Intel GPUs using Intel oneAPI MKL libraries and on Intel/NVIDIA/AMD GPUs using portBLAS libraries. 

## Installation
The user must specify the following variables during build:
| Variable | Valid Values | Default Value | Description |
| -------- | ------------ | ------------- | ----------- | 
| tmva-sofie | On, Off     | Off           | Build TMVA with support for sofie - fast inference code generation (requires protobuf 3)   |
| sofie-sycl | On, Off		| Off			| Build TMVA with support for sofie SYCL code generation for inference, tmva-sofie must be on |
| SYCL_IMPLEMENTATION | IntelSYCL | IntelSYCL | SYCL implementation for the tests to be compiled with. Currently, the only SYCL implementation supported is DPC++ (icpx, clang++, dpcpp) | 
| TARGET_GPU | Intel, NVIDIA, AMD | Intel | Specify GPU architecture for which the SOFIE SYCL tests will be built. |
| GPU_BLAS 	 | MKLBLAS, portBLAS    | MKLBLAS | Specify which BLAS libraries will be used for building the tests. |

You must also set CMAKE_CXX_COMPILER to an IntelSYCL compatible compiler (icpx, clang++, dpcpp).

## Prerequisites (only if -Dtesting and -Dsofie-sycl are enabled)
- Intel® oneAPI DPC++/C++ Compiler 
- If GPU_BLAS=MKLBLAS:
	- Intel oneAPI Math Kernel Library (oneMKL)
- If GPU_BLAS=portBLAS
	- [portBLAS](https://github.com/codeplaysoftware/portBLAS) (formerly known as SYCL-BLAS)

NVIDIA and AMD devices are not currently supported by MKLBLAS, so GPU_BLAS=MKLBLAS will be ignored and portBLAS will be used instead.

## Installation
SYCL Inference code can be generated by SOFIE without any additional flags (only the -Dtmva-sofie flag). If you wish to run the GPU tests for tmva-sofie, then flags -Dsofie-sycl and -Dtesting must be on and the above dependencies must be met.
```bash
cmake ../root -Dtmva-sofie=ON -Dsofie-sycl=ON -Dtesting=On
make -j8
```

## Usage
To generate SYCL inference code, the process is similar to generating C++ inference code.

From ROOT command line, or in a ROOT macro, we can proceed with an ONNX model:

	using namespace TMVA::Experimental;
	SOFIE::RModelParser_ONNX parser;
	SOFIE::RModel model = parser.Parse(“./example_model.onnx”);
	model.GenerateGPU();
	model.OutputGeneratedGPU(“./example_output.hxx”);

To use the generated inference code: 

	#include "example_output.hxx"
	std::vector<float> input(INPUT_SIZE);

    // Generated header file shall contain a Session class which requires initialization to load the corresponding weights.
    TMVA_SOFIE_example_model::Session s("example_model.dat")

    // Once instantiated the session object's infer method can be used
	// Note that input is now an std::vector
	std::vector<float> out = s.infer(input);

The inference code must now be compiled with the DPC++ compiler and linked against the oneAPI MKL or portBLAS libraries, as shown in the Makefiles below.

### Sample Makefile (when GPU_BLAS=MKLBLAS)

	CC = icx
	CXX = icpx

	ROOTINCLUDE = $(shell root-config --incdir)
	ROOTLIBS = $(shell root-config --libs)
	CFLAGS = $(shell root-config --cflags) -I$(ROOTINCLUDE)

	LDFLAGS = $(ROOTLIBS) -lROOTTMVASofie 
	LDFLAGS += -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 
	LDFLAGS += -lmkl_sequential -lmkl_core -lsycl -lpthread -lm -ldl
	CFLAGS += -fsycl -DMKL_ILP64  -I"${MKLROOT}/include" -O3

	EXECUTABLE = example_inference
	SRC = $(EXECUTABLE).cxx

	$(EXECUTABLE): $(SRC)
		$(CXX) $(CFLAGS) $(LDFLAGS) $(SRC) -o $@
		
	clean: 
	rm -rf $(EXECUTABLE)

### Sample Makefile (when GPU_BLAS=portBLAS and TARGET_GPU=NVIDIA)

	CC = icx
	CXX = icpx

	ROOTINCLUDE = $(shell root-config --incdir)
	ROOTLIBS = $(shell root-config --libs)
	CFLAGS = $(shell root-config --cflags) -I$(ROOTINCLUDE)

	LDFLAGS = $(ROOTLIBS) -lROOTTMVASofie 
	LDFLAGS +=-fsycl -fsycl-targets=nvptx64-nvidia-cuda 
	CFLAGS += -I${PORTBLAS_INCLUDE_DIR} -I${PORTBLAS_SRC_DIR} -fsycl-targets=nvptx64-nvidia-cuda -O3

	EXECUTABLE = example_inference
	SRC = $(EXECUTABLE).cxx

	$(EXECUTABLE): $(SRC)
		$(CXX) $(CFLAGS) $(LDFLAGS) $(SRC) -o $@
		
	clean: 
	rm -rf $(EXECUTABLE)

## Supported Operators
All ONNX operators supported by SOFIE are also supported by SOFIE-SYCL apart from LSTM and GRU. For portBLAS, RNN operator is not currently supported.

## Troubleshooting
If you encounter errors during build, try defining the following CMake variables
- IntelSYCL_DIR 
- PORTBLAS_DIR (the installation directory)
- PORTBLAS_INCLUDE_DIR (the directory that includes the portblas.hpp file)
- PORTBLAS_SRC_DIR (the directory that includes the portblas.h file)

Also, make sure that you have set the Intel environment with:
```bash
source /opt/intel/oneapi/setvars.sh --include-intel-llvm
```

If you are using an NVIDIA GPU with Intel DPC++, follow [this](https://developer.codeplay.com/products/oneapi/nvidia/2023.0.0/guides/get-started-guide-nvidia) guide to setup your environment.

If you are using an AMD GPU with Intel DPC++, follow [this](https://developer.codeplay.com/products/oneapi/amd/2023.0.0/guides/get-started-guide-amd) guide to setup your environment.

Make sure that your portBLAS installation matches your target GPU.

