# TMVA SOFIE Parsers

The ROOT/TMVA `sofie_parsers` directory contains the parser implementations used by TMVA SOFIE to translate external machine learning model formats into SOFIE’s internal intermediate representation (`RModel`).

The parsers are responsible for reading, validating, and translating model graphs, but do not perform code generation or runtime inference.

## Purpose and Scope

The `sofie_parsers` module provides frontend parsers that convert external machine learning model formats into SOFIE’s internal representation.

The responsibilities of this module include:
- Reading and interpreting model graphs
- Validating tensor shapes, data types, and connectivity
- Mapping supported operators to SOFIE internal operators
- Performing basic shape and type inference
- Rejecting unsupported or incompatible constructs at parse time

This module does not execute models or perform inference. Code generation, optimization, and runtime execution are handled by other components of TMVA SOFIE.

## Supported Model Formats

The SOFIE parsers currently support the following model formats:

- **ONNX** 
  The primary and most actively supported format.

- **Keras**
  Supported through a dedicated parser implementation, primarily for
  compatibility with existing TMVA workflows.

- **PyTorch**
  Supported via TorchScript-based parsing mechanisms.

Each supported format is translated into a common internal representation (`RModel`) used by the SOFIE backend.

## Model Format Compatibility

Compatibility of external models with SOFIE is determined primarily by operator support rather than by strict guarantees on model format or version.

For ONNX models:
- Compatibility depends on whether all operators used in the model are implemented and registered in the SOFIE ONNX parser.
- Models produced with different ONNX opset versions may be supported, provided they only use operators with compatible semantics.
- There is no fixed or guaranteed range of supported ONNX opset versions.

Users are encouraged to validate models using the parser APIs before attempting code generation.

## Basic Parser Usage

The SOFIE parsers are typically invoked through the TMVA SOFIE API to translate external model files into SOFIE’s internal representation.

A minimal example of parsing an ONNX model in C++ is shown below:

```c++
#include "TMVA/Experimental/SOFIE/RModelParser_ONNX.hxx"
using namespace TMVA::Experimental;
SOFIE::RModelParser_ONNX parser;
SOFIE::RModel model = parser.Parse("model.onnx");
```

## ONNX Operator Support

ONNX operator support in TMVA SOFIE is determined by operators registered via `RegisterOperator(...)` in `root/tmva/sofie_parsers/RModelParser_ONNX.cxx` in the current ROOT source tree.

Only tensor-based ONNX operators are considered; sequence-based ONNX operators are out of scope.

### Supported ONNX operators

<details>
<summary>Click to see full list</summary>

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
- [x] LeakyRelu
- [x] Less
- [x] LessOrEqual
- [x] Log
- [x] LSTM
- [x] MatMul
- [x] Max
- [x] MaxPool
- [x] Mean
- [x] Mul
- [x] Neg
- [x] Pad
- [x] Pow
- [x] Range
- [x] RandomNormal
- [x] RandomNormalLike
- [x] RandomUniform
- [x] RandomUniformLike
- [x] Reciprocal
- [x] ReduceMean
- [x] ReduceProd
- [x] ReduceSum
- [x] ReduceSumSquare
- [x] Relu
- [x] Reshape
- [x] RNN
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

</details>

## Limitations and Design Notes

- SOFIE parsers support tensor-based operators; other ONNX constructs may not be supported.
- Support for dynamic tensor shapes is limited and depends on the specific operator.
- Operator implementations may impose constraints on supported data types.

## Model Validation and Diagnostics

Model compatibility with SOFIE can be checked using the ONNX parser validation API:

```cpp
using namespace TMVA::Experimental;
SOFIE::RModelParser_ONNX parser;
parser.CheckModel("model.onnx");
```

# Developer Documentation

This section describes the internal structure of the TMVA SOFIE parser infrastructure and is intended for contributors.

## Architecture and Directory Structure

The `sofie_parsers` directory contains the parser implementations used by TMVA SOFIE to translate external model formats into the internal `RModel` representation.

Each supported model format provides a dedicated parser responsible for:
- Reading the model description
- Validating the model graph
- Translating supported operators into SOFIE internal operators

The resulting `RModel` object is subsequently consumed by the SOFIE components responsible for code generation and runtime inference.

## Parser Responsibilities

Each SOFIE parser is responsible for:

- Loading the external model description
- Performing structural and semantic validation
- Resolving tensor shapes and data types
- Constructing the internal `RModel` representation

Parsers are intentionally kept independent of code generation and runtime execution logic.

## How Operator Support Works

Operator support in the SOFIE parsers is explicit.

For ONNX models, an operator is considered supported only if it is registered in the ONNX parser implementation. Operator registration associates an ONNX operator with a corresponding SOFIE internal operator and defines the required input, output, and shape constraints.

During parsing, the model graph is traversed and each operator node is translated into its SOFIE representation. If an operator is not registered or does not satisfy the required constraints, parsing fails.

## Adding Support for New Operators

Support for new operators can be added by extending the corresponding
parser implementation.

In general, this involves:
- Implementing the SOFIE internal operator logic
- Registering the operator in the parser
- Defining shape and type validation rules
- Adding minimal tests exercising the new operator

New implementations should follow existing patterns to ensure consistency with the rest of the parser infrastructure.

## Relation to TMVA SOFIE Core

The SOFIE parser infrastructure is responsible only for parsing and validation of external model formats.

Code generation, optimization, and runtime inference are handled by other components of TMVA SOFIE.
