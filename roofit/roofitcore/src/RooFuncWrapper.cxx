/*
 * Project: RooFit
 * Authors:
 *   Garima Singh, CERN 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFuncWrapper.h>

#include <RooAbsData.h>
#include <RooFit/Detail/BatchModeDataHelpers.h>
#include <RooFit/Detail/CodeSquashContext.h>
#include <RooFit/Evaluator.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include "RooEvaluatorWrapper.h"

#include <TROOT.h>
#include <TSystem.h>

#include <fstream>

namespace RooFit {

namespace Experimental {

RooFuncWrapper::RooFuncWrapper(const char *name, const char *title, RooAbsReal &obj, const RooAbsData *data,
                               RooSimultaneous const *simPdf, bool useEvaluator)
   : RooAbsReal{name, title}, _params{"!params", "List of parameters", this}, _useEvaluator{useEvaluator}
{
   if (_useEvaluator) {
      _absReal = std::make_unique<RooEvaluatorWrapper>(obj, const_cast<RooAbsData *>(data), false, "", simPdf, false);
   }

   std::string func;

   // Get the parameters.
   RooArgSet paramSet;
   obj.getParameters(data ? data->get() : nullptr, paramSet);
   RooArgSet floatingParamSet;
   for (RooAbsArg *param : paramSet) {
      if (!param->isConstant()) {
         floatingParamSet.add(*param);
      }
   }

   // Load the parameters and observables.
   loadParamsAndData(&obj, floatingParamSet, data, simPdf);

   func = buildCode(obj);

   declareToInterpreter("#pragma cling optimize(2)");

   // Declare the function and create its derivative.
   _funcName = declareFunction(func);
   _func = reinterpret_cast<Func>(gInterpreter->ProcessLine((_funcName + ";").c_str()));
}

RooFuncWrapper::RooFuncWrapper(const RooFuncWrapper &other, const char *name)
   : RooAbsReal(other, name),
     _params("!params", this, other._params),
     _funcName(other._funcName),
     _func(other._func),
     _grad(other._grad),
     _hasGradient(other._hasGradient),
     _gradientVarBuffer(other._gradientVarBuffer),
     _observables(other._observables)
{
}

void RooFuncWrapper::loadParamsAndData(RooAbsArg const *head, RooArgSet const &paramSet, const RooAbsData *data,
                                       RooSimultaneous const *simPdf)
{
   // Extract observables
   std::stack<std::vector<double>> vectorBuffers; // for data loading
   std::map<RooFit::Detail::DataKey, std::span<const double>> spans;

   if (data) {
      spans = RooFit::Detail::BatchModeDataHelpers::getDataSpans(*data, "", simPdf, true, false, vectorBuffers);
   }

   std::size_t idx = 0;
   for (auto const &item : spans) {
      std::size_t n = item.second.size();
      _obsInfos.emplace(item.first, ObsInfo{idx, n});
      _observables.reserve(_observables.size() + n);
      for (std::size_t i = 0; i < n; ++i) {
         _observables.push_back(item.second[i]);
      }
      idx += n;
   }

   // Extract parameters
   for (auto *param : paramSet) {
      if (!dynamic_cast<RooAbsReal *>(param)) {
         std::stringstream errorMsg;
         errorMsg << "In creation of function " << GetName()
                  << " wrapper: input param expected to be of type RooAbsReal.";
         coutE(InputArguments) << errorMsg.str() << std::endl;
         throw std::runtime_error(errorMsg.str().c_str());
      }
      if (spans.find(param) == spans.end()) {
         _params.add(*param);
      }
   }
   _gradientVarBuffer.resize(_params.size());

   if (head) {
      _nodeOutputSizes = RooFit::Detail::BatchModeDataHelpers::determineOutputSizes(
         *head, [&spans](RooFit::Detail::DataKey key) -> int {
            auto found = spans.find(key);
            return found != spans.end() ? found->second.size() : -1;
         });
   }
}

std::string RooFuncWrapper::declareFunction(std::string const &funcBody)
{
   static int iFuncWrapper = 0;
   auto funcName = "roo_func_wrapper_" + std::to_string(iFuncWrapper++);

   // Declare the function
   std::stringstream bodyWithSigStrm;
   bodyWithSigStrm << "double " << funcName << "(double* params, double const* obs, double const* xlArr) {\n"
                   << funcBody << "\n}";
   if (!declareToInterpreter(bodyWithSigStrm.str())) {
      std::stringstream errorMsg;
      errorMsg << "Function " << funcName << " could not be compiled. See above for details.";
      oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }
   return funcName;
}

void RooFuncWrapper::createGradient()
{
   std::string gradName = _funcName + "_grad_0";
   std::string requestName = _funcName + "_req";

   // Calculate gradient
   declareToInterpreter("#include <Math/CladDerivator.h>\n");
   // disable clang-format for making the following code unreadable.
   // clang-format off
   std::stringstream requestFuncStrm;
   requestFuncStrm << "#pragma clad ON\n"
                      "void " << requestName << "() {\n"
                      "  clad::gradient(" << _funcName << ", \"params\");\n"
                      "}\n"
                      "#pragma clad OFF";
   // clang-format on
   if (!declareToInterpreter(requestFuncStrm.str())) {
      std::stringstream errorMsg;
      errorMsg << "Function " << GetName() << " could not be differentiated. See above for details.";
      oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }

   _grad = reinterpret_cast<Grad>(gInterpreter->ProcessLine((gradName + ";").c_str()));
   _hasGradient = true;
}

void RooFuncWrapper::gradient(double *out) const
{
   updateGradientVarBuffer();
   std::fill(out, out + _params.size(), 0.0);

   _grad(_gradientVarBuffer.data(), _observables.data(), _xlArr.data(), out);
}

void RooFuncWrapper::updateGradientVarBuffer() const
{
   std::transform(_params.begin(), _params.end(), _gradientVarBuffer.begin(),
                  [](RooAbsArg *obj) { return static_cast<RooAbsReal *>(obj)->getVal(); });
}

double RooFuncWrapper::evaluate() const
{
   if (_useEvaluator)
      return _absReal->getVal();
   updateGradientVarBuffer();

   return _func(_gradientVarBuffer.data(), _observables.data(), _xlArr.data());
}

void RooFuncWrapper::gradient(const double *x, double *g) const
{
   std::fill(g, g + _params.size(), 0.0);

   _grad(const_cast<double *>(x), _observables.data(), _xlArr.data(), g);
}

std::string RooFuncWrapper::buildCode(RooAbsReal const &head)
{
   RooFit::Detail::CodeSquashContext ctx(_nodeOutputSizes, _xlArr);

   // First update the result variable of params in the compute graph to in[<position>].
   int idx = 0;
   for (RooAbsArg *param : _params) {
      ctx.addResult(param, "params[" + std::to_string(idx) + "]");
      idx++;
   }

   for (auto const &item : _obsInfos) {
      const char *name = item.first->GetName();
      // If the observable is scalar, set name to the start idx. else, store
      // the start idx and later set the the name to obs[start_idx + curr_idx],
      // here curr_idx is defined by a loop producing parent node.
      if (item.second.size == 1) {
         ctx.addResult(name, "obs[" + std::to_string(item.second.idx) + "]");
      } else {
         ctx.addResult(name, "obs");
         ctx.addVecObs(name, item.second.idx);
      }
   }

   return ctx.assembleCode(ctx.getResult(head));
}

/// @brief Declare code to the interpreter and keep track of all declared code in this RooFuncWrapper.
bool RooFuncWrapper::declareToInterpreter(std::string const &code)
{
   _allCode << code << std::endl;
   return gInterpreter->Declare(code.c_str());
}

/// @brief Dumps a macro "filename.C" that can be used to test and debug the generated code and gradient.
void RooFuncWrapper::writeDebugMacro(std::string const &filename) const
{
   std::ofstream outFile;
   outFile.open(filename + ".C");
   outFile << "#include <RooFit/Detail/MathFuncs.h>" << std::endl;
   outFile << std::endl;
   outFile << _allCode.str();
   outFile << std::endl;

   updateGradientVarBuffer();

   auto writeVector = [&](std::string const &name, std::span<const double> vec) {
      outFile << "std::vector<double> " << name << " = {";
      for (std::size_t i = 0; i < vec.size(); ++i) {
         if (i % 10 == 0)
            outFile << "\n    ";
         outFile << vec[i];
         if (i < vec.size() - 1)
            outFile << ", ";
      }
      outFile << "\n};\n";
   };

   outFile << "// clang-format off\n" << std::endl;
   writeVector("parametersVec", _gradientVarBuffer);
   outFile << std::endl;
   writeVector("observablesVec", _observables);
   outFile << std::endl;
   writeVector("auxConstantsVec", _xlArr);
   outFile << std::endl;
   outFile << "// clang-format on\n" << std::endl;

   outFile << R"(
// To run as a ROOT macro
void )" << filename
           << R"(()
{
   std::vector<double> gradientVec(parametersVec.size());

   )" << _funcName
           << R"((parametersVec.data(), observablesVec.data(), auxConstantsVec.data());
   )" << _funcName
           << R"(_grad_0(parametersVec.data(), observablesVec.data(), auxConstantsVec.data(), gradientVec.data());
}
)";
}

} // namespace Experimental

} // namespace RooFit
