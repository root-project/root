#ifndef RooFit_SofieLoader_h
#define RooFit_SofieLoader_h

#include <string>
#include <string_view>

namespace RooFit::Detail {

// SOFIE generates the C++ code to evaluate ONNX models. RooFit has no
// build-time dependency on it: the SOFIE library is loaded at runtime the first
// time it is needed, and its C interface is called by looking up its symbols.

/// Whether the SOFIE library can be loaded.
bool IsSofieAvailable();

/// Generate the C++ code to evaluate an ONNX model, with the weights of the
/// model embedded in it. The code is placed in the namespace SOFIE_<modelName>.
/// \param[in] onnx The content of the ONNX file.
/// \param[in] modelName The name of the model, which must be a valid C++ identifier.
/// \throws std::runtime_error If the SOFIE library can't be loaded, or if the code can't be generated.
std::string GenerateCodeFromONNX(std::string_view onnx, std::string const &modelName);

} // namespace RooFit::Detail

#endif
