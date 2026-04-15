/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN  04/2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooONNXFunction.h>

#include <TInterpreter.h>
#include <TSystem.h>

#include <fstream>
#include <mutex>

/**
 \file RooONNXFunction.cxx
 \class RooONNXFunction
 \ingroup Roofit

 RooONNXFunction wraps an ONNX model as a RooAbsReal, allowing it to be used as
 a building block in likelihoods, fits, and statistical analyses without
 additional boilerplate code. The class supports models with **one or more
 statically-shaped input tensors** and a **single scalar output**. The class
 was designed to share workspaces with neural functions for combined fits in
 RooFit-based frameworks written in C++. Therefore, the RooONNXFunction doesn't
 depend on any Python packages and fully supports ROOT IO,

 The ONNX model is evaluated through compiled C++ code generated at runtime
 using **TMVA SOFIE**. Automatic differentiation is supported via **Clad**,
 allowing RooFit to access analytical gradients for fast minimization with
 Minuit 2.

 The ONNX model is stored internally as a byte payload and serialized together
 with the RooONNXFunction object using ROOT I/O. Upon reading from a file or
 workspace, the runtime backend is rebuilt automatically.

 ### Input handling

 The model inputs are provided as a list of tensors, where each tensor is
 represented by a RooArgList of RooAbsReal objects. The order of the inputs
 defines the feature ordering passed to the ONNX model.
 Optionally, users can validate that the ONNX model has the expected input

 ### Example (C++)

 \code
 // Define input variables
 RooRealVar x{"x", "x", 0.0};
 RooRealVar y{"y", "y", 0.0};
 RooRealVar z{"z", "z", 0.0};

 // Construct ONNX function, building the std::vector<RooArgList> in-place
 RooONNXFunction func{
     "func", "func",
     {{x, y}, {z}},
     "model.onnx"
 };

 // Evaluate
 double val = func.getVal();
 std::cout << "Model output: " << val << std::endl;
 \endcode

 ### Example (Python)

 \code{.py}
 import ROOT

 # Define variables
 x = ROOT.RooRealVar("x", "x", 0.0)
 y = ROOT.RooRealVar("y", "y", 0.0)
 z = ROOT.RooRealVar("z", "z", 0.0)

 # Create ONNX function
 func = ROOT.RooONNXFunction(
     "func", "func",
     [[x, y], [z]],
     "model.onnx"
 )

 # Evaluate
 print("Model output:", func.getVal())
 \endcode

 */

namespace {

std::vector<std::uint8_t> fileToBytes(std::string const &filePath)
{
   // Read file into byte vector
   std::ifstream file(filePath, std::ios::binary);
   if (!file) {
      std::ostringstream os;
      os << "failed to open file '" << filePath << "'";
      throw std::runtime_error(os.str());
   }

   file.seekg(0, std::ios::end);
   const std::streamsize size = file.tellg();
   file.seekg(0, std::ios::beg);

   if (size <= 0) {
      std::ostringstream os;
      os << "file '" << filePath << "' is empty";
      throw std::runtime_error(os.str());
   }

   std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
   file.read(reinterpret_cast<char *>(bytes.data()), size);

   if (!file) {
      std::ostringstream os;
      os << "error while reading file '" << filePath << "'";
      throw std::runtime_error(os.str());
   }

   return bytes;
}

template <typename Fn>
Fn resolveLazy(std::string const &name, const char *code)
{
   static Fn fn = nullptr;
   static std::once_flag flag;

   std::call_once(flag, [&] {
      // Try to declare the code
      if (!gInterpreter->Declare(code)) {
         throw std::runtime_error(std::string("ROOT JIT Declare failed for code defining ") + name);
      }

      // Try to resolve the symbol
      void *symbol = reinterpret_cast<void *>(gInterpreter->ProcessLine((name + ";").c_str()));

      if (!symbol) {
         throw std::runtime_error(std::string("ROOT JIT failed to resolve symbol: ") + name);
      }

      fn = reinterpret_cast<Fn>(symbol);

      if (!fn) {
         throw std::runtime_error(std::string("ROOT JIT produced null function pointer for: ") + name);
      }
   });

   return fn;
}

template <typename T>
std::string toPtrString(T *ptr, std::string const &castType)
{
   return TString::Format("reinterpret_cast<%s>(0x%zx)", (castType + "*").c_str(), reinterpret_cast<std::size_t>(ptr))
      .Data();
}

} // namespace

void RooFit::Detail::AnyWithVoidPtr::emplace(std::string const &typeName)
{
   auto anyPtrSession = toPtrString(this, "RooFit::Detail::AnyWithVoidPtr");
   gInterpreter->ProcessLine((anyPtrSession + "->emplace<" + typeName + ">();").c_str());
}

struct RooONNXFunction::RuntimeCache {
   using Func = void (*)(void *, float const *, float *);

   RooFit::Detail::AnyWithVoidPtr _session;
   RooFit::Detail::AnyWithVoidPtr _d_session;
   Func _func;
};

/**
 Construct a RooONNXFunction from an ONNX model file.

 \param name Name of the RooFit object
 \param title Title of the RooFit object
 \param inputTensors Vector of RooArgList, each representing one input tensor.
        The variables in each RooArgList match to each flattened input tensor.
 \param onnxFile Path to the ONNX model file. The file is read and stored
        internally as a byte payload for persistence with RooWorkspace.
 \param inputNames Optional list of ONNX input node names. If provided, these
        are used to validate that the ONNX model has the structure expected by
        your RooFit code.
 \param inputShapes Optional list of tensor shapes corresponding to each input
        tensor. If provided, these are used to validate that the ONNX models
        input tensors have the shape that you expect. If omitted, only the
        total size of each tensor is checked.
 */
RooONNXFunction::RooONNXFunction(const char *name, const char *title, const std::vector<RooArgList> &inputTensors,
                                 const std::string &onnxFile, const std::vector<std::string> & /*inputNames*/,
                                 const std::vector<std::vector<int>> & /*inputShapes*/)
   : RooAbsReal{name, title}, _onnxBytes{fileToBytes(onnxFile)}
{
   for (std::size_t i = 0; i < inputTensors.size(); ++i) {
      std::string istr = std::to_string(i);
      _inputTensors.emplace_back(
         std::make_unique<RooListProxy>(("!inputs_" + istr).c_str(), ("Input tensor " + istr).c_str(), this));
      _inputTensors.back()->addTyped<RooAbsReal>(inputTensors[i]);
   }
}

RooONNXFunction::RooONNXFunction(const RooONNXFunction &other, const char *newName)
   : RooAbsReal{other, newName}, _onnxBytes{other._onnxBytes}, _runtime{other._runtime}
{
   for (std::size_t i = 0; i < other._inputTensors.size(); ++i) {
      _inputTensors.emplace_back(std::make_unique<RooListProxy>("!inputs", this, *other._inputTensors[i]));
   }
}

void RooONNXFunction::fillInputBuffer() const
{
   _inputBuffer.clear();
   _inputBuffer.reserve(_inputTensors.size());

   for (auto const &tensorList : _inputTensors) {
      for (auto const *real : static_range_cast<RooAbsReal const *>(*tensorList)) {
         _inputBuffer.push_back(static_cast<float>(real->getVal(tensorList->nset())));
      }
   }
}

void RooONNXFunction::initialize() const
{
   if (_runtime) {
      return;
   }

   _runtime = std::make_unique<RuntimeCache>();

   // We are jitting the SOFIE invocation lazily at runtime, to avoid the
   // link-time dependency to the SOFIE parser library.
   if (gSystem->Load("libROOTTMVASofieParser") < 0) {
      throw std::runtime_error("RooONNXFunction: cannot load ONNX file since SOFIE ONNX parser is missing."
                               " Please build ROOT with tmva-sofie=ON.");
   }
   using OnnxToCpp = std::string (*)(std::uint8_t const *, std::size_t, const char *);
   auto onnxToCppWithSofie = resolveLazy<OnnxToCpp>("_RooONNXFunction_onnxToCppWithSofie",
                                                    R"(
#include "TMVA/RModelParser_ONNX.hxx"

std::string _RooONNXFunction_onnxToCppWithSofie(std::uint8_t const *onnxBytes, std::size_t onnxBytesSize, const char *outputName)
{
   namespace SOFIE = TMVA::Experimental::SOFIE;

   std::string buffer{reinterpret_cast<const char *>(onnxBytes), onnxBytesSize};
   std::istringstream stream{buffer};

   SOFIE::RModel rmodel = SOFIE::RModelParser_ONNX{}.Parse(stream, outputName);
   rmodel.SetOptimizationLevel(SOFIE::OptimizationLevel::kBasic);
   rmodel.Generate(SOFIE::Options::kNoWeightFile);

   std::stringstream ss{};
   rmodel.PrintGenerated(ss);
   return ss.str();
}
)");

   static int counter = 0;
   _funcName = "roo_onnx_func_" + std::to_string(counter);
   std::string namespaceName = "TMVA_SOFIE_" + _funcName + "";
   counter++;

   std::string modelCode = onnxToCppWithSofie(_onnxBytes.data(), _onnxBytes.size(), _funcName.c_str());
   gInterpreter->Declare(modelCode.c_str());

   // Declare string to the interpreter, where the %%NAMESPACE%% placeholder
   // will first be replaced by the namespace for the emitted code.
   auto declareWithNamespace = [&](std::string codeTemplate) {
      const std::string placeholder = "%%NAMESPACE%%";
      size_t pos = 0;

      while ((pos = codeTemplate.find(placeholder, pos)) != std::string::npos) {
         codeTemplate.replace(pos, placeholder.length(), namespaceName);
         pos += namespaceName.length();
      }

      gInterpreter->Declare(codeTemplate.c_str());
   };

   declareWithNamespace(R"(

namespace %%NAMESPACE%% {

float roo_inner_wrapper(Session const &session, float const *input)
{
   float out = 0.;
   doInfer(session, input, &out);
   return out;
}

float roo_wrapper(Session const &session, float const *input)
{
   return roo_inner_wrapper(session, input);
}

} // namespace %%NAMESPACE%%

)");

   std::string sessionName = "::TMVA_SOFIE_" + _funcName + "::Session";

   _runtime->_session.emplace(sessionName);
   auto ptrSession = toPtrString(_runtime->_session.ptr, sessionName);

   std::stringstream ss2;
   ss2 << "static_cast<void (*)(void *, float const *, float *)>(RooFit::Detail::doInferWithSessionVoidPtr<"
       << sessionName << ">" << ");";
   _runtime->_func = reinterpret_cast<RuntimeCache::Func>(gInterpreter->ProcessLine(ss2.str().c_str()));

   // hardcode the gradient for now
   _runtime->_d_session.emplace(sessionName);
   auto ptrDSession = toPtrString(_runtime->_d_session.ptr, sessionName);

   gInterpreter->Declare("#include <Math/CladDerivator.h>");

   gInterpreter->ProcessLine(("clad::gradient(" + namespaceName + "::roo_wrapper, \"input\");").c_str());

   declareWithNamespace(R"(
namespace %%NAMESPACE%% {

double roo_outer_wrapper(double const *input) {
    auto &session = *)" +
                        ptrSession + R"(;
    float inputFlt[inputTensorDims[0].total_size()];
    for (std::size_t i = 0; i < std::size(inputFlt); ++i) {
       inputFlt[i] = input[i];
    }
    return roo_inner_wrapper(session, inputFlt);
}

} // namespace %%NAMESPACE%%

namespace clad::custom_derivatives {

namespace %%NAMESPACE%% {

void roo_outer_wrapper_pullback(double const *input, double d_y, double *d_input) {

    using namespace ::%%NAMESPACE%%;

    float inputFlt[inputTensorDims[0].total_size()];
    float d_inputFlt[::std::size(inputFlt)];
    for (::std::size_t i = 0; i < ::std::size(inputFlt); ++i) {
       inputFlt[i] = input[i];
       d_inputFlt[i] = d_input[i];
    }
    auto *session = )" + ptrSession +
                        R"(;
    auto *d_session = )" +
                        ptrDSession + R"(;
    roo_inner_wrapper_pullback(*session, inputFlt, d_y, d_session, d_inputFlt);
    for (::std::size_t i = 0; i < ::std::size(inputFlt); ++i) {
       d_input[i] += d_inputFlt[i];
    }
}

} // namespace %%NAMESPACE%%

} // namespace clad::custom_derivatives

)");
}

double RooONNXFunction::evaluate() const
{
   initialize();
   fillInputBuffer();

   float out = 0.f;
   _runtime->_func(_runtime->_session.ptr, _inputBuffer.data(), &out);
   return static_cast<double>(out);
}
