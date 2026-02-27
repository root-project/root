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
#include <RooMsgService.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>

#include "RooFit/BatchModeDataHelpers.h"
#include "RooFitImplHelpers.h"

#include <TInterpreter.h>

#include <fstream>

namespace RooFit::Experimental {

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

/// @brief  A wrapper class to store a C++ function of type 'double (*)(double*, double*)'.
/// The parameters can be accessed as params[<relative position of param in paramSet>] in the function body.
/// The observables can be accessed as obs[i + j], where i represents the observable position and j
/// represents the data entry.
class RooFuncWrapper {
public:
   RooFuncWrapper(RooAbsReal &obj, const RooAbsData *data, RooSimultaneous const *simPdf, RooArgSet const &paramSet);

   bool hasGradient() const { return _hasGradient; }
   bool hasHessian() const { return _hasHessian; }
   void gradient(double *out) const
   {
      updateGradientVarBuffer();
      std::fill(out, out + _params.size(), 0.0);
      _grad(_varBuffer.data(), _observables.data(), _xlArr.data(), out);
   }
   void hessian(double *out) const
   {
      updateGradientVarBuffer();
      std::fill(out, out + _params.size() * _params.size(), 0.0);
      _hessian(_varBuffer.data(), _observables.data(), _xlArr.data(), out);
   }

   void createGradient();
   void createHessian();

   void writeDebugMacro(std::string const &) const;

   std::vector<std::string> const &collectedFunctions() { return _collectedFunctions; }

   double evaluate() const
   {
      updateGradientVarBuffer();
      return _func(_varBuffer.data(), _observables.data(), _xlArr.data());
   }

   void loadData(RooAbsData const &data, RooSimultaneous const *simPdf);

private:
   void updateGradientVarBuffer() const;

   void buildFuncAndGradFunctors();

   using Func = double (*)(double *, double const *, double const *);
   using Grad = void (*)(double *, double const *, double const *, double *);
   using Hessian = void (*)(double *, double const *, double const *, double *);

   RooArgList _params;
   std::string _funcName;
   Func _func;
   Grad _grad;
   Hessian _hessian;
   bool _hasGradient = false;
   bool _hasHessian = false;
   mutable std::vector<double> _varBuffer;
   std::vector<double> _observables;
   std::unordered_map<RooFit::Detail::DataKey, std::size_t> _obsInfos;
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

auto getDependsOnData(RooAbsReal &obj, RooArgSet const &dataObs)
{
   RooArgSet serverSet;
   RooHelpers::getSortedComputationGraph(obj, serverSet);

   std::unordered_set<RooFit::Detail::DataKey> dependsOnData;
   for (RooAbsArg *arg : dataObs) {
      dependsOnData.insert(arg);
   }

   for (RooAbsArg *arg : serverSet) {
      if (arg->getAttribute("__obs__")) {
         dependsOnData.insert(arg);
      }
      for (RooAbsArg *server : arg->servers()) {
         if (server->isValueServer(*arg)) {
            if (dependsOnData.find(server) != dependsOnData.end() && !arg->isReducerNode()) {
               dependsOnData.insert(arg);
               break;
            }
         }
      }
   }

   return dependsOnData;
}

} // namespace

RooFuncWrapper::RooFuncWrapper(RooAbsReal &obj, const RooAbsData *data, RooSimultaneous const *simPdf,
                               RooArgSet const &paramSet)
{
   // Load the observables from the dataset
   if (data) {
      loadData(*data, simPdf);
   }

   // Define the parameters
   for (auto *param : paramSet) {
      if (_obsInfos.find(param) == _obsInfos.end()) {
         _params.add(*param);
      }
   }
   _varBuffer.resize(_params.size());

   // Figure out which part of the computation graph depends on data
   std::unordered_set<RooFit::Detail::DataKey> dependsOnData;
   if (data) {
      dependsOnData = getDependsOnData(obj, *data->get());
   }

   // Set up the code generation context
   RooFit::Experimental::CodegenContext ctx;

   // First update the result variable of params in the compute graph to in[<position>].
   int idx = 0;
   for (RooAbsArg *param : _params) {
      ctx.addResult(param, "params[" + std::to_string(idx) + "]");
      idx++;
   }

   for (auto const &item : _obsInfos) {
      const char *obsName = item.first->GetName();
      ctx.addResult(obsName, "obs");
      ctx.addVecObs(obsName, item.second);
   }

   // Declare the function and create its derivative.
   auto print = [](std::string const &msg) { oocoutI(nullptr, Fitting) << msg << std::endl; };
   ROOT::Math::Util::TimingScope timingScope(print, "Function JIT time:");
   _funcName = ctx.buildFunction(obj, dependsOnData);

   // Make sure the codegen implementations are known to the interpreter
   gInterpreter->Declare("#include <RooFit/CodegenImpl.h>\n");

   if (!gInterpreter->Declare(ctx.collectedCode().c_str())) {
      std::stringstream errorMsg;
      std::string debugFileName = "_codegen_" + _funcName + ".cxx";
      errorMsg << "Function " << _funcName << " could not be compiled. See above for details. Full code dumped to file "
               << debugFileName << " for debugging";
      {
         std::ofstream outFile;
         outFile.open(debugFileName.c_str());
         outFile << ctx.collectedCode();
      }
      oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }

   _func = reinterpret_cast<Func>(gInterpreter->ProcessLine((_funcName + ";").c_str()));

   _xlArr = ctx.xlArr();
   _collectedFunctions = ctx.collectedFunctions();
}

void RooFuncWrapper::loadData(RooAbsData const &data, RooSimultaneous const *simPdf)
{
   // Extract observables
   std::stack<std::vector<double>> vectorBuffers; // for data loading
   auto spans = RooFit::BatchModeDataHelpers::getDataSpans(data, "", simPdf, true, false, vectorBuffers);

   _observables.clear();
   // The first elements contain the sizes of the packed observable arrays
   std::size_t total = 0;
   _observables.reserve(2 * spans.size());
   std::size_t idx = 0;
   for (auto const &item : spans) {
      _obsInfos.emplace(item.first, idx);
      _observables.push_back(total + 2 * spans.size());
      _observables.push_back(item.second.size());
      total += item.second.size();
      idx += 1;
   }
   idx = 0;
   for (auto const &item : spans) {
      std::size_t n = item.second.size();
      _observables.reserve(_observables.size() + n);
      for (std::size_t i = 0; i < n; ++i) {
         _observables.push_back(item.second[i]);
      }
      idx += n;
   }
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

void RooFuncWrapper::createHessian()
{
#ifdef ROOFIT_CLAD
   std::string hessianName = _funcName + "_hessian_0";
   std::string requestName = _funcName + "_hessian_req";

   // Calculate Hessian
   gInterpreter->Declare("#include <Math/CladDerivator.h>\n");
   // disable clang-format for making the following code unreadable.
   // clang-format off
   std::stringstream requestFuncStrm;
   std::string paramsStr =
      _params.size() == 1 ? "\"params[0]\"" : ("\"params[0:" + std::to_string(_params.size() - 1) + "]\"");
   requestFuncStrm << "#pragma clad ON\n"
                      "void " << requestName << "() {\n"
                      "  clad::hessian(" << _funcName << ", " << paramsStr << ");\n"
                      "}\n"
                      "#pragma clad OFF";
   // clang-format on
   auto print = [](std::string const &msg) { oocoutI(nullptr, Fitting) << msg << std::endl; };

   bool cladSuccess = false;
   {
      ROOT::Math::Util::TimingScope timingScope(print, "Hessian generation time:");
      cladSuccess = !gInterpreter->Declare(requestFuncStrm.str().c_str());
   }
   if (cladSuccess) {
      std::stringstream errorMsg;
      errorMsg << "Function could not be differentiated. See above for details.";
      oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }

   // Clad provides different overloads for the Hessian, and we need to
   // resolve to the one that we want. Without the static_cast, getting the
   // function pointer would be ambiguous.
   std::stringstream ss;
   ROOT::Math::Util::TimingScope timingScope(print, "Hessian IR to machine code time:");
   ss << "static_cast<void (*)(double *, double const *, double const *, double *)>(" << hessianName << ");";
   _hessian = reinterpret_cast<Hessian>(gInterpreter->ProcessLine(ss.str().c_str()));
   _hasHessian = true;
#else
   _hasHessian = false;
   std::stringstream errorMsg;
   errorMsg << "Function could not be differentiated since ROOT was built without Clad support.";
   oocoutE(nullptr, InputArguments) << errorMsg.str() << std::endl;
   throw std::runtime_error(errorMsg.str().c_str());
#endif
}

void RooFuncWrapper::updateGradientVarBuffer() const
{
   std::transform(_params.begin(), _params.end(), _varBuffer.begin(), [](RooAbsArg *obj) {
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
   std::string paramsStr =
      _params.size() == 1 ? "\"params[0]\"" : ("\"params[0:" + std::to_string(_params.size() - 1) + "]\"");
   outFile.open(filename + ".C");
   outFile << R"(//auto-generated test macro
#include <RooFit/Detail/MathFuncs.h>
#include <Math/CladDerivator.h>

//#define DO_HESSIAN

)" << allCode.str()
           << R"(
#pragma clad ON
void gradient_request() {
  clad::gradient()"
           << _funcName << R"(, "params");
#ifdef DO_HESSIAN
   clad::hessian()"
           << _funcName << ", " << paramsStr << R"();
#endif
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
   writeVector("parametersVec", _varBuffer);
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
   const std::size_t n = parametersVec.size();

   std::vector<double> gradientVec(n);

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

#ifdef DO_HESSIAN
   std::cout << "\n";

   auto hess = [&](std::span<double> params, std::span<double> out) {
      return )"
           << _funcName << R"(_hessian_0(params.data(), observablesVec.data(), auxConstantsVec.data(), out.data());
   };

   std::vector<double> hessianVec(n * n);
   hess(parametersVec, hessianVec);

   // ---------- Numerical Hessian ----------
   // Uses central differences:
   // diag: (f(x+ei)-2f(x)+f(x-ei))/eps^2
   // offdiag: (f(++ ) - f(+-) - f(-+) + f(--)) / (4 eps^2)
   auto numHess = [&](std::size_t i, std::size_t j) {
      const double eps = 1e-5; // often needs to be a bit larger than grad eps
      std::vector<double> p(parametersVec.begin(), parametersVec.end());

      if (i == j) {
         const double f0 = func(p);

         p[i] = parametersVec[i] + eps;
         const double fUp = func(p);

         p[i] = parametersVec[i] - eps;
         const double fDown = func(p);

         return (fUp - 2.0 * f0 + fDown) / (eps * eps);
      } else {
         // f(x_i + eps, x_j + eps)
         p[i] = parametersVec[i] + eps;
         p[j] = parametersVec[j] + eps;
         const double fPP = func(p);

         // f(x_i + eps, x_j - eps)
         p[i] = parametersVec[i] + eps;
         p[j] = parametersVec[j] - eps;
         const double fPM = func(p);

         // f(x_i - eps, x_j + eps)
         p[i] = parametersVec[i] - eps;
         p[j] = parametersVec[j] + eps;
         const double fMP = func(p);

         // f(x_i - eps, x_j - eps)
         p[i] = parametersVec[i] - eps;
         p[j] = parametersVec[j] - eps;
         const double fMM = func(p);

         return (fPP - fPM - fMP + fMM) / (4.0 * eps * eps);
      }
   };

   // Compute full numerical Hessian
   std::vector<double> numHessianVec(n * n);
   for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
         numHessianVec[i + n * j] = numHess(i, j); // keep same layout as your print
      }
   }

   // ---------- Compare & print ----------
   std::cout << "Hessian comparison (clad vs numeric vs diff):\n\n";

   for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
         const std::size_t idx = i + n * j; // same indexing you used
         const double cladH = hessianVec[idx];
         const double numH = numHessianVec[idx];
         const double diff = cladH - numH;

         std::cout << "[" << i << "," << j << "] "
                   << "clad=" << cladH << "  num=" << numH << "  diff=" << diff << "\n";
      }
   }

   std::cout << "\nRaw Clad Hessian matrix:\n";
   for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
         std::cout << hessianVec[i + n * j] << "   ";
      }
      std::cout << "\n";
   }

   std::cout << "\nRaw Numerical Hessian matrix:\n";
   for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
         std::cout << numHessianVec[i + n * j] << "   ";
      }
      std::cout << "\n";
   }
#endif
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

bool RooEvaluatorWrapper::setData(RooAbsData &data, bool /*cloneData*/)
{
   // To make things easier for RooFit, we only support resetting with
   // datasets that have the same structure, e.g. the same columns and global
   // observables. This is anyway the usecase: resetting same-structured data
   // when iterating over toys.
   constexpr auto errMsg = "Error in RooAbsReal::setData(): only resetting with same-structured data is supported.";

   _data = &data;
   bool isInitializing = _paramSet.empty();
   const std::size_t oldSize = _dataSpans.size();

   std::stack<std::vector<double>>{}.swap(_vectorBuffers);
   bool skipZeroWeights = !_pdf || !_pdf->getAttribute("BinnedLikelihoodActive");
   auto simPdf = dynamic_cast<RooSimultaneous const *>(_pdf);
   _dataSpans = RooFit::BatchModeDataHelpers::getDataSpans(*_data, _rangeName, simPdf, skipZeroWeights,
                                                           _takeGlobalObservablesFromData, _vectorBuffers);
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
   if (_funcWrapper) {
      _funcWrapper->loadData(*_data, simPdf);
   }
   return true;
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
   if (!_funcWrapper->hasGradient())
      _funcWrapper->createGradient();
}

void RooEvaluatorWrapper::generateHessian()
{
   if (!_funcWrapper)
      createFuncWrapper();
   if (!_funcWrapper->hasHessian())
      _funcWrapper->createHessian();
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

void RooEvaluatorWrapper::hessian(double *out) const
{
   _funcWrapper->hessian(out);
}

bool RooEvaluatorWrapper::hasGradient() const
{
   return _funcWrapper && _funcWrapper->hasGradient();
}

bool RooEvaluatorWrapper::hasHessian() const
{
   return _funcWrapper && _funcWrapper->hasHessian();
}

void RooEvaluatorWrapper::writeDebugMacro(std::string const &filename) const
{
   if (_funcWrapper)
      return _funcWrapper->writeDebugMacro(filename);
}

} // namespace RooFit::Experimental

/// \endcond
