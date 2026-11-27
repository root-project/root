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

#include <RooONNXFunc.h>

#include <RooMsgService.h>

#include <TBuffer.h>
#include <TInterpreter.h>
#include <TSystem.h>

#include <fstream>
#include <mutex>

/**
 \file RooONNXFunc.cxx
 \class RooONNXFunc
 \ingroup Roofit

 RooONNXFunc wraps an ONNX model as a RooAbsReal, allowing it to be used as
 a building block in likelihoods, fits, and statistical analyses without
 additional boilerplate code. The class supports models with **one or more
 statically-shaped input tensors** and a **single scalar output**. The class
 was designed to share workspaces with neural functions for combined fits in
 RooFit-based frameworks written in C++. Therefore, the RooONNXFunc doesn't
 depend on any Python packages and fully supports ROOT IO,

 The ONNX model is evaluated through compiled C++ code generated at runtime
 using **TMVA SOFIE**. Automatic differentiation is supported via
 [Clad](https://github.com/vgvassilev/clad), allowing RooFit to access
 analytical gradients for fast minimization with Minuit 2.

 The ONNX model is stored internally as a byte payload and serialized together
 with the RooONNXFunc object using ROOT I/O. Upon reading from a file or
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
 RooONNXFunc func{
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
 func = ROOT.RooONNXFunc(
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

// Expression for the offset into a flat input buffer to the i-th tensor:
//   i=0: "0"; i=1: "inputTensorDims[0].total_size()";
//   i=2: "inputTensorDims[0].total_size() + inputTensorDims[1].total_size()"
std::string flatOffsetExpr(std::size_t i)
{
   if (i == 0)
      return "0";
   std::string out;
   for (std::size_t j = 0; j < i; ++j) {
      if (j > 0)
         out += " + ";
      out += "inputTensorDims[" + std::to_string(j) + "].total_size()";
   }
   return out;
}

} // namespace

void RooFit::Detail::AnyWithVoidPtr::emplace(std::string const &typeName)
{
   auto anyPtrSession = toPtrString(this, "RooFit::Detail::AnyWithVoidPtr");
   gInterpreter->ProcessLine((anyPtrSession + "->emplace<" + typeName + ">();").c_str());
}

struct RooONNXFunc::RuntimeCache {
   /// Uniform thunk signature regardless of input-tensor count.
   /// Args: (Session*, output, flat input buffer).
   using Func = void (*)(void *, float *, float const *);

   RooFit::Detail::AnyWithVoidPtr _session;
   RooFit::Detail::AnyWithVoidPtr _d_session;
   RooFit::Detail::AnyWithVoidPtr _zero_session; ///< Zero-weight session, the tangent for forward-mode derivatives.
   Func _func = nullptr;
   std::size_t _nInputTensors = 0;
   bool _codegenDeclared = false;
};

/**
 Construct a RooONNXFunc from an ONNX model file.

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
RooONNXFunc::RooONNXFunc(const char *name, const char *title, const std::vector<RooArgList> &inputTensors,
                         const std::string &onnxFile, const std::vector<std::string> & /*inputNames*/,
                         const std::vector<std::vector<int>> & /*inputShapes*/)
   : RooAbsReal{name, title}, _onnxBytes{fileToBytes(onnxFile)}
{
   initialize();

   for (std::size_t i = 0; i < inputTensors.size(); ++i) {
      std::string istr = std::to_string(i);
      _inputTensors.emplace_back(
         std::make_unique<RooListProxy>(("!inputs_" + istr).c_str(), ("Input tensor " + istr).c_str(), this));
      _inputTensors.back()->addTyped<RooAbsReal>(inputTensors[i]);
   }
}

RooONNXFunc::RooONNXFunc(const RooONNXFunc &other, const char *newName)
   : RooAbsReal{other, newName}, _onnxBytes{other._onnxBytes}, _runtime{other._runtime}, _funcName{other._funcName}
{
   for (std::size_t i = 0; i < other._inputTensors.size(); ++i) {
      _inputTensors.emplace_back(std::make_unique<RooListProxy>("!inputs", this, *other._inputTensors[i]));
   }
}

void RooONNXFunc::fillInputBuffer() const
{
   _inputBuffer.clear();
   _inputBuffer.reserve(_inputTensors.size());

   for (auto const &tensorList : _inputTensors) {
      for (auto const *real : static_range_cast<RooAbsReal const *>(*tensorList)) {
         _inputBuffer.push_back(static_cast<float>(real->getVal(tensorList->nset())));
      }
   }
}

void RooONNXFunc::initialize()
{
   if (_runtime) {
      return;
   }

   _runtime = std::make_unique<RuntimeCache>();

   // We are jitting the SOFIE invocation lazily at runtime, to avoid the
   // link-time dependency to the SOFIE parser library.
   if (gSystem->Load("libROOTTMVASofieParser") < 0) {
      throw std::runtime_error("RooONNXFunc: cannot load ONNX file since SOFIE ONNX parser is missing."
                               " Please build ROOT with tmva=ON.");
   }
   using OnnxToCpp = std::string (*)(std::uint8_t const *, std::size_t, const char *);
   auto onnxToCppWithSofie = resolveLazy<OnnxToCpp>("_RooONNXFunc_onnxToCppWithSofie",
                                                    R"(
#include "TMVA/RModelParser_ONNX.hxx"

std::string _RooONNXFunc_onnxToCppWithSofie(std::uint8_t const *onnxBytes, std::size_t onnxBytesSize, const char *outputName)
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

   auto nInputTensors = static_cast<unsigned long>(
      gInterpreter->ProcessLine(("std::size(" + namespaceName + "::inputTensorDims);").c_str()));
   _runtime->_nInputTensors = nInputTensors;

   // Evaluation thunk with a uniform signature regardless of the input-tensor count:
   // takes a flat float buffer and splits it into the per-tensor pointers expected by
   // SOFIE's doInfer.
   {
      std::ostringstream ss;
      ss << "namespace " << namespaceName << " {\n"
         << "void roo_eval_thunk(void *session_void, float *out, float const *flat_input) {\n"
         << "   auto *session = reinterpret_cast<Session *>(session_void);\n"
         << "   doInfer(*session";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << ", flat_input + (" << flatOffsetExpr(i) << ")";
      }
      ss << ", out);\n"
         << "}\n"
         << "} // namespace " << namespaceName << "\n";
      gInterpreter->Declare(ss.str().c_str());
   }

   std::string sessionName = "::TMVA_SOFIE_" + _funcName + "::Session";

   _runtime->_session.emplace(sessionName);

   _runtime->_func = reinterpret_cast<RuntimeCache::Func>(gInterpreter->ProcessLine(
      ("static_cast<void(*)(void *, float *, float const *)>(" + namespaceName + "::roo_eval_thunk);").c_str()));
}

/// Name of the function that RooFit codegen calls. Declares it on first use,
/// so that only codegen users pay for the Clad differentiation.
std::string RooONNXFunc::outerWrapperName() const
{
   declareCodegenFunctions();
   return "TMVA_SOFIE_" + funcName() + "::roo_outer_wrapper";
}

void RooONNXFunc::declareCodegenFunctions() const
{
   if (_runtime->_codegenDeclared) {
      return;
   }
   _runtime->_codegenDeclared = true;

   const std::size_t nInputTensors = _runtime->_nInputTensors;
   std::string namespaceName = "TMVA_SOFIE_" + _funcName;
   std::string sessionName = "::" + namespaceName + "::Session";
   auto ptrSession = toPtrString(_runtime->_session.ptr, sessionName);

   // Comma-separated list "<prefix>0, <prefix>1, ..." over the input tensors.
   auto perTensor = [&](std::string const &prefix) {
      std::string out;
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         out += (i > 0 ? ", " : "") + prefix + std::to_string(i);
      }
      return out;
   };

   // Per-input-tensor parameter / argument lists used by the JIT'd code below.
   std::string innerParams = perTensor("float const *input");
   std::string innerArgs = perTensor("input");
   std::string innerTangentParams = perTensor("float const *d_input");
   std::string innerTangentArgs = perTensor("d_input");
   std::string outerDoubleParams = perTensor("double const *input");
   std::string tangentDoubleParams = perTensor("double const *d_input");
   std::string cladInputs = perTensor("input"); // for clad::gradient param spec

   // Non-template inner / wrapper functions, generated with the right number of inputs.
   {
      std::ostringstream ss;
      ss << "namespace " << namespaceName << " {\n\n"
         << "float roo_inner_wrapper(Session const &session, " << innerParams << ") {\n"
         << "   float out = 0.;\n"
         << "   doInfer(session, " << innerArgs << ", &out);\n"
         << "   return out;\n"
         << "}\n\n"
         << "float roo_wrapper(Session const &session, " << innerParams << ") {\n"
         << "   return roo_inner_wrapper(session, " << innerArgs << ");\n"
         << "}\n\n"
         << "} // namespace " << namespaceName << "\n";
      gInterpreter->Declare(ss.str().c_str());
   }

   // hardcode the gradient for now
   _runtime->_d_session.emplace(sessionName);
   auto ptrDSession = toPtrString(_runtime->_d_session.ptr, sessionName);

   gInterpreter->Declare("#include <Math/CladDerivator.h>");

   gInterpreter->ProcessLine(("clad::gradient(" + namespaceName + "::roo_wrapper, \"" + cladInputs + "\");").c_str());

   // The codegen call site (CodegenImpl::codegenImpl(RooONNXFunc)) passes one
   // double-array argument per input tensor. Emit roo_outer_wrapper and the matching
   // custom-derivative pullback with the corresponding number of parameters.
   {
      std::ostringstream ss;
      ss << "namespace " << namespaceName << " {\n\n"
         << "double roo_outer_wrapper(" << outerDoubleParams << ") {\n"
         << "    auto &session = *" << ptrSession << ";\n";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << "    float inputFlt" << i << "[inputTensorDims[" << i << "].total_size()];\n"
            << "    for (std::size_t i = 0; i < std::size(inputFlt" << i << "); ++i) {\n"
            << "       inputFlt" << i << "[i] = input" << i << "[i];\n"
            << "    }\n";
      }
      ss << "    return roo_inner_wrapper(session";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << ", inputFlt" << i;
      }
      ss << ");\n"
         << "}\n\n"
         << "} // namespace " << namespaceName << "\n\n"
         << "namespace clad::custom_derivatives {\n"
         << "namespace " << namespaceName << " {\n\n"
         << "void roo_outer_wrapper_pullback(" << outerDoubleParams << ", double d_y";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << ", double *d_input" << i;
      }
      ss << ") {\n"
         << "    using namespace ::" << namespaceName << ";\n";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << "    float inputFlt" << i << "[inputTensorDims[" << i << "].total_size()];\n"
            << "    float d_inputFlt" << i << "[::std::size(inputFlt" << i << ")];\n"
            << "    for (::std::size_t i = 0; i < ::std::size(inputFlt" << i << "); ++i) {\n"
            << "       inputFlt" << i << "[i] = input" << i << "[i];\n"
            << "       d_inputFlt" << i << "[i] = 0;\n"
            << "    }\n";
      }
      ss << "    auto *session = " << ptrSession << ";\n"
         << "    auto *d_session = " << ptrDSession << ";\n"
         << "    roo_inner_wrapper_pullback(*session";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << ", inputFlt" << i;
      }
      ss << ", d_y, d_session";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << ", d_inputFlt" << i;
      }
      ss << ");\n";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << "    for (::std::size_t i = 0; i < ::std::size(inputFlt" << i << "); ++i) {\n"
            << "       d_input" << i << "[i] += d_inputFlt" << i << "[i];\n"
            << "    }\n";
      }
      ss << "}\n\n"
         << "} // namespace " << namespaceName << "\n"
         << "} // namespace clad::custom_derivatives\n";
      if (!gInterpreter->Declare(ss.str().c_str())) {
         throw std::runtime_error("RooONNXFunc: failed to declare the code for RooFit codegen and Clad");
      }
   }

   // Clad Hessians are reverse-over-forward, so the opaque roo_outer_wrapper also needs a
   // custom pushforward and pushforward_pullback. If declaring them fails, only Hessians
   // are unavailable.

   // Generates roo_inner_wrapper_pushforward as a by-product, like the gradient above
   // does for roo_inner_wrapper_pullback.
   gInterpreter->ProcessLine(("clad::differentiate(" + namespaceName + "::roo_wrapper, \"input0[0]\");").c_str());

   // Weights are constant, so their tangent is zero. The intermediate tensors of
   // this session serve as tangent scratch space.
   _runtime->_zero_session.emplace(sessionName);
   auto ptrZeroSession = toPtrString(_runtime->_zero_session.ptr, sessionName);
   gInterpreter->ProcessLine((ptrZeroSession + "->SetWeightsToZero();").c_str());

   // roo_dir = grad . d_input. Its pullback, generated by differentiating roo_dir_outer,
   // gives all that the pushforward_pullback needs: H . d_input in the input adjoints and
   // the gradient in the d_input adjoints. Differentiating w.r.t. d_input also avoids a
   // clad bug: for a non-varied pointer argument, clad finds no custom helper pullback
   // with the reduced signature and silently differentiates the BLAS calls, yielding zero.
   {
      std::ostringstream ss;
      ss << "namespace " << namespaceName << " {\n\n"
         << "float roo_dir(Session const &session, Session const &zeroSession, " << innerParams << ", "
         << innerTangentParams << ") {\n"
         << "   return roo_inner_wrapper_pushforward(session, " << innerArgs << ", zeroSession, " << innerTangentArgs
         << ").pushforward;\n"
         << "}\n\n"
         << "float roo_dir_outer(Session const &session, Session const &zeroSession, " << innerParams << ", "
         << innerTangentParams << ") {\n"
         << "   return roo_dir(session, zeroSession, " << innerArgs << ", " << innerTangentArgs << ");\n"
         << "}\n\n"
         << "} // namespace " << namespaceName << "\n";
      gInterpreter->Declare(ss.str().c_str());
   }
   gInterpreter->ProcessLine(
      ("clad::gradient(" + namespaceName + "::roo_dir_outer, \"" + cladInputs + ", " + innerTangentArgs + "\");")
         .c_str());

   {
      std::ostringstream ss;

      // Emits the sessions and float copies of the inputs and the tangents.
      auto emitPrologue = [&]() {
         ss << "    using namespace ::" << namespaceName << ";\n"
            << "    auto &session = *" << ptrSession << ";\n"
            << "    auto &zeroSession = *" << ptrZeroSession << ";\n";
         for (std::size_t i = 0; i < nInputTensors; ++i) {
            ss << "    float inputFlt" << i << "[inputTensorDims[" << i << "].total_size()];\n"
               << "    float dInputFlt" << i << "[inputTensorDims[" << i << "].total_size()];\n"
               << "    for (::std::size_t i = 0; i < ::std::size(inputFlt" << i << "); ++i) {\n"
               << "       inputFlt" << i << "[i] = input" << i << "[i];\n"
               << "       dInputFlt" << i << "[i] = d_input" << i << "[i];\n"
               << "    }\n";
         }
      };

      ss << "namespace clad::custom_derivatives {\n"
         << "namespace " << namespaceName << " {\n\n"
         << "clad::ValueAndPushforward<double, double> roo_outer_wrapper_pushforward(" << outerDoubleParams << ", "
         << tangentDoubleParams << ") {\n";
      emitPrologue();
      ss << "    auto vp = roo_inner_wrapper_pushforward(session, " << perTensor("inputFlt") << ", zeroSession, "
         << perTensor("dInputFlt") << ");\n"
         << "    return {vp.value, vp.pushforward};\n"
         << "}\n\n";

      ss << "void roo_outer_wrapper_pushforward_pullback(" << outerDoubleParams << ", " << tangentDoubleParams
         << ", clad::ValueAndPushforward<double, double> d_y, " << perTensor("double *d_out_input") << ", "
         << perTensor("double *d_out_d_input") << ") {\n";
      emitPrologue();
      // Fresh adjoint sessions per call: clad's second-order pullback doesn't reset
      // their adjoint state, so reused ones would accumulate it across calls.
      ss << "    Session dSession1;\n"
         << "    Session dSession2;\n";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << "    float hvpFlt" << i << "[inputTensorDims[" << i << "].total_size()] = {};\n"
            << "    float gradFlt" << i << "[inputTensorDims[" << i << "].total_size()] = {};\n";
      }
      ss << "    roo_dir_pullback(session, zeroSession, " << perTensor("inputFlt") << ", " << perTensor("dInputFlt")
         << ", 1.F, &dSession1, &dSession2, " << perTensor("hvpFlt") << ", " << perTensor("gradFlt") << ");\n";
      for (std::size_t i = 0; i < nInputTensors; ++i) {
         ss << "    for (::std::size_t i = 0; i < ::std::size(inputFlt" << i << "); ++i) {\n"
            << "       d_out_input" << i << "[i] += d_y.value * gradFlt" << i << "[i] + d_y.pushforward * hvpFlt" << i
            << "[i];\n"
            << "       d_out_d_input" << i << "[i] += d_y.pushforward * gradFlt" << i << "[i];\n"
            << "    }\n";
      }
      ss << "}\n\n"
         << "} // namespace " << namespaceName << "\n"
         << "} // namespace clad::custom_derivatives\n";
      if (!gInterpreter->Declare(ss.str().c_str())) {
         coutW(Fitting) << "RooONNXFunc::declareCodegenFunctions(" << GetName()
                        << "): failed to declare the custom derivatives for Clad Hessians" << std::endl;
      }
   }
}

double RooONNXFunc::evaluate() const
{
   fillInputBuffer();

   float out = 0.f;
   _runtime->_func(_runtime->_session.ptr, &out, _inputBuffer.data());
   return static_cast<double>(out);
}

void RooONNXFunc::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RooONNXFunc::Class(), this);
      this->initialize();
   } else {
      R__b.WriteClassBuffer(RooONNXFunc::Class(), this);
   }
}
