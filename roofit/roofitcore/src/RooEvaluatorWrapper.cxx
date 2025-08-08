/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
\internal
\file RooEvaluatorWrapper.cxx
\class RooEvaluatorWrapper
\ingroup Roofitcore

Wraps a RooFit::Evaluator that evaluates a RooAbsReal back into a RooAbsReal.
**/

#include "RooEvaluatorWrapper.h"

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooConstVar.h>
#include <RooHelpers.h>
#include <RooMsgService.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include <TInterpreter.h>

#include <fstream>

RooEvaluatorWrapper::RooEvaluatorWrapper(RooAbsReal &topNode, RooAbsData *data, bool useGPU,
                                         std::string const &rangeName, RooAbsPdf const *pdf,
                                         bool takeGlobalObservablesFromData)
   : RooAbsReal{"RooEvaluatorWrapper", "RooEvaluatorWrapper"},
     _evaluator{std::make_unique<RooFit::Evaluator>(topNode, useGPU)},
     _topNode("topNode", "top node", this, topNode, false, false),
     _data{data},
     _paramSet("paramSet", "Set of parameters", this),
     _rangeName{rangeName},
     _pdf{pdf},
     _takeGlobalObservablesFromData{takeGlobalObservablesFromData}
{
   if (data) {
      setData(*data, false);
   }
   _paramSet.add(_evaluator->getParameters());
   for (auto const &item : _dataSpans) {
      _paramSet.remove(*_paramSet.find(item.first->GetName()));
   }
}

RooEvaluatorWrapper::RooEvaluatorWrapper(const RooEvaluatorWrapper &other, const char *name)
   : RooAbsReal{other, name},
     _evaluator{other._evaluator},
     _topNode("topNode", this, other._topNode),
     _data{other._data},
     _paramSet("paramSet", "Set of parameters", this),
     _rangeName{other._rangeName},
     _pdf{other._pdf},
     _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData},
     _dataSpans{other._dataSpans}
{
   _paramSet.add(other._paramSet);
}

RooEvaluatorWrapper::~RooEvaluatorWrapper() = default;

bool RooEvaluatorWrapper::getParameters(const RooArgSet *observables, RooArgSet &outputSet,
                                        bool stripDisconnected) const
{
   outputSet.add(_evaluator->getParameters());
   if (observables) {
      outputSet.remove(*observables, /*silent*/ false, /*matchByNameOnly*/ true);
   }
   // Exclude the data variables from the parameters which are not global observables
   for (auto const &item : _dataSpans) {
      if (_data->getGlobalObservables() && _data->getGlobalObservables()->find(item.first->GetName())) {
         continue;
      }
      RooAbsArg *found = outputSet.find(item.first->GetName());
      if (found) {
         outputSet.remove(*found);
      }
   }
   // If we take the global observables as data, we have to return these as
   // parameters instead of the parameters in the model. Otherwise, the
   // constant parameters in the fit result that are global observables will
   // not have the right values.
   if (_takeGlobalObservablesFromData && _data->getGlobalObservables()) {
      outputSet.replace(*_data->getGlobalObservables());
   }

   // The disconnected parameters are stripped away in
   // RooAbsArg::getParametersHook(), that is only called in the original
   // RooAbsArg::getParameters() implementation. So he have to call it to
   // identify disconnected parameters to remove.
   if (stripDisconnected) {
      RooArgSet paramsStripped;
      _topNode->getParameters(observables, paramsStripped, true);
      RooArgSet toRemove;
      for (RooAbsArg *param : outputSet) {
         if (!paramsStripped.find(param->GetName())) {
            toRemove.add(*param);
         }
      }
      outputSet.remove(toRemove, /*silent*/ false, /*matchByNameOnly*/ true);
   }

   return false;
}

bool RooEvaluatorWrapper::setData(RooAbsData &data, bool /*cloneData*/)
{
   // To make things easiear for RooFit, we only support resetting with
   // datasets that have the same structure, e.g. the same columns and global
   // observables. This is anyway the usecase: resetting same-structured data
   // when iterating over toys.
   constexpr auto errMsg = "Error in RooAbsReal::setData(): only resetting with same-structured data is supported.";

   _data = &data;
   bool isInitializing = _paramSet.empty();
   const std::size_t oldSize = _dataSpans.size();

   std::stack<std::vector<double>>{}.swap(_vectorBuffers);
   bool skipZeroWeights = !_pdf || !_pdf->getAttribute("BinnedLikelihoodActive");
   _dataSpans =
      RooFit::BatchModeDataHelpers::getDataSpans(*_data, _rangeName, dynamic_cast<RooSimultaneous const *>(_pdf),
                                                 skipZeroWeights, _takeGlobalObservablesFromData, _vectorBuffers);
   if (!isInitializing && _dataSpans.size() != oldSize) {
      coutE(DataHandling) << errMsg << std::endl;
      throw std::runtime_error(errMsg);
   }
   for (auto const &item : _dataSpans) {
      const char *name = item.first->GetName();
      _evaluator->setInput(name, item.second, false);
      if (_paramSet.find(name)) {
         coutE(DataHandling) << errMsg << std::endl;
         throw std::runtime_error(errMsg);
      }
   }
   return true;
}

/// @brief  A wrapper class to store a C++ function of type 'double (*)(double*, double*)'.
/// The parameters can be accessed as params[<relative position of param in paramSet>] in the function body.
/// The observables can be accessed as obs[i + j], where i represents the observable position and j
/// represents the data entry.
class RooFuncWrapper {
public:
   RooFuncWrapper(RooAbsReal &obj, const RooAbsData *data, RooSimultaneous const *simPdf, RooArgSet const &paramSet);

   bool hasGradient() const { return _hasGradient; }
   void gradient(double *out) const
   {
      updateGradientVarBuffer();
      std::fill(out, out + _params.size(), 0.0);

      _grad(_gradientVarBuffer.data(), _observables.data(), _xlArr.data(), out);
   }

   void createGradient();

   void writeDebugMacro(std::string const &) const;

   std::vector<std::string> const &collectedFunctions() { return _collectedFunctions; }

   double evaluate() const
   {
      updateGradientVarBuffer();
      return _func(_gradientVarBuffer.data(), _observables.data(), _xlArr.data());
   }

private:
   void updateGradientVarBuffer() const;

   std::map<RooFit::Detail::DataKey, std::span<const double>>
   loadParamsAndData(RooArgSet const &paramSet, const RooAbsData *data, RooSimultaneous const *simPdf);

   void buildFuncAndGradFunctors();

   using Func = double (*)(double *, double const *, double const *);
   using Grad = void (*)(double *, double const *, double const *, double *);

   struct ObsInfo {
      ObsInfo(std::size_t i, std::size_t n) : idx{i}, size{n} {}
      std::size_t idx = 0;
      std::size_t size = 0;
   };

   RooArgList _params;
   std::string _funcName;
   Func _func;
   Grad _grad;
   bool _hasGradient = false;
   mutable std::vector<double> _gradientVarBuffer;
   std::vector<double> _observables;
   std::map<RooFit::Detail::DataKey, ObsInfo> _obsInfos;
   std::vector<double> _xlArr;
   std::vector<std::string> _collectedFunctions;
};

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

RooFuncWrapper::RooFuncWrapper(RooAbsReal &obj, const RooAbsData *data, RooSimultaneous const *simPdf,
                               RooArgSet const &paramSet)
{
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

   for (auto *param : paramSet) {
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
      errorMsg << "Function could not be differentiated. See above for details.";
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
   errorMsg << "Function could not be differentiated since ROOT was built without Clad support.";
   oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
   throw std::runtime_error(errorMsg.str().c_str());
#endif
}

void RooFuncWrapper::updateGradientVarBuffer() const
{
   std::transform(_params.begin(), _params.end(), _gradientVarBuffer.begin(), [](RooAbsArg *obj) {
      return obj->isCategory() ? static_cast<RooAbsCategory *>(obj)->getCurrentIndex()
                               : static_cast<RooAbsReal *>(obj)->getVal();
   });
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

double RooEvaluatorWrapper::evaluate() const
{
   if (_useGeneratedFunctionCode)
      return _funcWrapper->evaluate();

   if (!_evaluator)
      return 0.0;

   _evaluator->setOffsetMode(hideOffset() ? RooFit::EvalContext::OffsetMode::WithoutOffset
                                          : RooFit::EvalContext::OffsetMode::WithOffset);

   return _evaluator->run()[0];
}

void RooEvaluatorWrapper::createFuncWrapper()
{
   // Get the parameters.
   RooArgSet paramSet;
   this->getParameters(_data ? _data->get() : nullptr, paramSet, /*sripDisconnectedParams=*/false);

   _funcWrapper =
      std::make_unique<RooFuncWrapper>(*_topNode, _data, dynamic_cast<RooSimultaneous const *>(_pdf), paramSet);
}

void RooEvaluatorWrapper::generateGradient()
{
   if (!_funcWrapper)
      createFuncWrapper();
   _funcWrapper->createGradient();
}

void RooEvaluatorWrapper::setUseGeneratedFunctionCode(bool flag)
{
   _useGeneratedFunctionCode = flag;
   if (!_funcWrapper && _useGeneratedFunctionCode)
      createFuncWrapper();
}

void RooEvaluatorWrapper::gradient(double *out) const
{
   _funcWrapper->gradient(out);
}

bool RooEvaluatorWrapper::hasGradient() const
{
   if (!_funcWrapper)
      return false;
   return _funcWrapper->hasGradient();
}

/// \endcond
