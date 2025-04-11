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
#include <RooFit/CodegenContext.h>
#include <RooFit/Evaluator.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include "RooEvaluatorWrapper.h"
#include "RooFit/BatchModeDataHelpers.h"
#include "RooFitImplHelpers.h"

#include <TROOT.h>
#include <TSystem.h>

#include <Math/Util.h>

#include <fstream>
#include <set>

namespace {

void replaceAll(std::string &str, const std::string &from, const std::string &to)
{
   if (from.empty())
      return;
   size_t start_pos = 0;
   while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
      str.replace(start_pos, from.length(), to);
      start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
   }
}

} // namespace

namespace RooFit {

namespace Experimental {

RooFuncWrapper::RooFuncWrapper(const char *name, const char *title, RooAbsReal &obj, const RooAbsData *data,
                               RooSimultaneous const *simPdf, bool useEvaluator)
   : RooAbsReal{name, title}, _params{"!params", "List of parameters", this}, _useEvaluator{useEvaluator}
{
   if (_useEvaluator) {
      _absReal = std::make_unique<RooEvaluatorWrapper>(obj, const_cast<RooAbsData *>(data), false, "", simPdf, false);
   }

   // Get the parameters.
   RooArgSet paramSet;
   obj.getParameters(data ? data->get() : nullptr, paramSet);

   // Load the parameters and observables.
   auto spans = loadParamsAndData(paramSet, data, simPdf);

   // Set up the code generation context
   std::map<RooFit::Detail::DataKey, std::size_t> nodeOutputSizes =
      RooFit::BatchModeDataHelpers::determineOutputSizes(obj, [&spans](RooFit::Detail::DataKey key) -> int {
         auto found = spans.find(key);
         return found != spans.end() ? found->second.size() : -1;
      });

   RooFit::Experimental::CodegenContext ctx;

   // First update the result variable of params in the compute graph to in[<position>].
   int idx = 0;
   for (RooAbsArg *param : _params) {
      ctx.addResult(param, "params[" + std::to_string(idx) + "]");
      idx++;
   }

   for (auto const &item : _obsInfos) {
      const char *obsName = item.first->GetName();
      // If the observable is scalar, set name to the start idx. else, store
      // the start idx and later set the the name to obs[start_idx + curr_idx],
      // here curr_idx is defined by a loop producing parent node.
      if (item.second.size == 1) {
         ctx.addResult(obsName, "obs[" + std::to_string(item.second.idx) + "]");
      } else {
         ctx.addResult(obsName, "obs");
         ctx.addVecObs(obsName, item.second.idx);
      }
   }

   gInterpreter->Declare("#pragma cling optimize(2)");

   // Declare the function and create its derivative.
   auto print = [](std::string const &msg) { oocoutI(nullptr, Fitting) << msg << std::endl; };
   ROOT::Math::Util::TimingScope timingScope(print, "Function JIT time:");
   _funcName = ctx.buildFunction(obj, nodeOutputSizes);
   _func = reinterpret_cast<Func>(gInterpreter->ProcessLine((_funcName + ";").c_str()));

   _xlArr = ctx.xlArr();
   _collectedFunctions = ctx.collectedFunctions();
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

std::map<RooFit::Detail::DataKey, std::span<const double>>
RooFuncWrapper::loadParamsAndData(RooArgSet const &paramSet, const RooAbsData *data, RooSimultaneous const *simPdf)
{
   // Extract observables
   std::stack<std::vector<double>> vectorBuffers; // for data loading
   std::map<RooFit::Detail::DataKey, std::span<const double>> spans;

   if (data) {
      spans = RooFit::BatchModeDataHelpers::getDataSpans(*data, "", simPdf, true, false, vectorBuffers);
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

   return spans;
}

void RooFuncWrapper::createGradient()
{
#ifdef ROOFIT_CLAD
   std::string gradName = _funcName + "_grad_0";
   std::string requestName = _funcName + "_req";

   // Calculate gradient
   gInterpreter->Declare("#include <Math/CladDerivator.h>\n");
   // disable clang-format for making the following code unreadable.
   // clang-format off
   std::stringstream requestFuncStrm;
   requestFuncStrm << "#pragma clad ON\n"
                      "void " << requestName << "() {\n"
                      "  clad::gradient(" << _funcName << ", \"params\");\n"
                      "}\n"
                      "#pragma clad OFF";
   // clang-format on
   auto print = [](std::string const &msg) { oocoutI(nullptr, Fitting) << msg << std::endl; };

   bool cladSuccess = false;
   {
      ROOT::Math::Util::TimingScope timingScope(print, "Gradient generation time:");
      cladSuccess = !gInterpreter->Declare(requestFuncStrm.str().c_str());
   }
   if (cladSuccess) {
      std::stringstream errorMsg;
      errorMsg << "Function " << GetName() << " could not be differentiated. See above for details.";
      oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }

   // Clad provides different overloads for the gradient, and we need to
   // resolve to the one that we want. Without the static_cast, getting the
   // function pointer would be ambiguous.
   std::stringstream ss;
   ROOT::Math::Util::TimingScope timingScope(print, "Gradient IR to machine code time:");
   ss << "static_cast<void (*)(double *, double const *, double const *, double *)>(" << gradName << ");";
   _grad = reinterpret_cast<Grad>(gInterpreter->ProcessLine(ss.str().c_str()));
   _hasGradient = true;
#else
   _hasGradient = false;
   std::stringstream errorMsg;
   errorMsg << "Function " << GetName() << " could not be differentiated since ROOT was built without Clad support.";
   oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
   throw std::runtime_error(errorMsg.str().c_str());
#endif
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

/// @brief Dumps a macro "filename.C" that can be used to test and debug the generated code and gradient.
void RooFuncWrapper::writeDebugMacro(std::string const &filename) const
{
   std::stringstream allCode;
   std::set<std::string> seenFunctions;

   // Remove duplicated declared functions
   for (std::string const &name : _collectedFunctions) {
      if (seenFunctions.count(name) > 0) {
         continue;
      }
      seenFunctions.insert(name);
      std::unique_ptr<TInterpreterValue> v = gInterpreter->MakeInterpreterValue();
      gInterpreter->Evaluate(name.c_str(), *v);
      std::string s = v->ToString();
      for (int i = 0; i < 2; ++i) {
         s = s.erase(0, s.find("\n") + 1);
      }
      allCode << s << std::endl;
   }

   std::ofstream outFile;
   outFile.open(filename + ".C");
   outFile << R"(//auto-generated test macro
#include <RooFit/Detail/MathFuncs.h>
#include <Math/CladDerivator.h>

#pragma cling optimize(2)
)" << allCode.str()
           << R"(
#pragma clad ON
void gradient_request() {
  clad::gradient()"
           << _funcName << R"(, "params");
}
#pragma clad OFF
)";

   updateGradientVarBuffer();

   auto writeVector = [&](std::string const &name, std::span<const double> vec) {
      std::stringstream decl;
      decl << "std::vector<double> " << name << " = {";
      for (std::size_t i = 0; i < vec.size(); ++i) {
         if (i % 10 == 0)
            decl << "\n    ";
         decl << vec[i];
         if (i < vec.size() - 1)
            decl << ", ";
      }
      decl << "\n};\n";

      std::string declStr = decl.str();

      replaceAll(declStr, "inf", "std::numeric_limits<double>::infinity()");
      replaceAll(declStr, "nan", "NAN");

      outFile << declStr;
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

   auto func = [&](std::span<double> params) {
      return )"
           << _funcName << R"((params.data(), observablesVec.data(), auxConstantsVec.data());
   };
   auto grad = [&](std::span<double> params, std::span<double> out) {
      return )"
           << _funcName << R"(_grad_0(parametersVec.data(), observablesVec.data(), auxConstantsVec.data(),
                                        out.data());
   };

   grad(parametersVec, gradientVec);

   auto numDiff = [&](int i) {
      const double eps = 1e-6;
      std::vector<double> p{parametersVec};
      p[i] = parametersVec[i] - eps;
      double funcValDown = func(p);
      p[i] = parametersVec[i] + eps;
      double funcValUp = func(p);
      return (funcValUp - funcValDown) / (2 * eps);
   };

   for (std::size_t i = 0; i < parametersVec.size(); ++i) {
      std::cout << i << ":" << std::endl;
      std::cout << "  numr : " << numDiff(i) << std::endl;
      std::cout << "  clad : " << gradientVec[i] << std::endl;
   }
}
)";
}

} // namespace Experimental

} // namespace RooFit
