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
#include <RooGlobalFunc.h>
#include <RooMsgService.h>
#include <RooRealVar.h>
#include <RooHelpers.h>
#include <RooFit/Detail/CodeSquashContext.h>
#include "RooFit/BatchModeDataHelpers.h"

#include <TROOT.h>
#include <TSystem.h>

RooFuncWrapper::RooFuncWrapper(const char *name, const char *title, std::string const &funcBody,
                               RooArgSet const &paramSet, const RooAbsData *data /*=nullptr*/)
   : RooAbsReal{name, title}, _params{"!params", "List of parameters", this}
{
   // Declare the function and create its derivative.
   declareAndDiffFunction(name, funcBody);

   // Load the parameters and observables.
   loadParamsAndData(name, nullptr, paramSet, data);
}

RooFuncWrapper::RooFuncWrapper(const char *name, const char *title, RooAbsReal const &obj, RooArgSet const &normSet,
                               const RooAbsData *data /*=nullptr*/)
   : RooAbsReal{name, title}, _params{"!params", "List of parameters", this}
{
   std::string func;

   // Compile the computation graph for the norm set, such that we also get the
   // integrals explicitly in the graph.
   std::unique_ptr<RooAbsReal> pdf{RooFit::Detail::compileForNormSet(obj, normSet)};
   // Get the parameters.
   RooArgSet paramSet;
   obj.getParameters(data ? data->get() : nullptr, paramSet);

   // Load the parameters and observables.
   loadParamsAndData(name, pdf.get(), paramSet, data);

   func = buildCode(*pdf);

   // Declare the function and create its derivative.
   declareAndDiffFunction(name, func);
}

RooFuncWrapper::RooFuncWrapper(const RooFuncWrapper &other, const char *name)
   : RooAbsReal(other, name),
     _params("!params", this, other._params),
     _func(other._func),
     _grad(other._grad),
     _gradientVarBuffer(other._gradientVarBuffer),
     _observables(other._observables)
{
}

void RooFuncWrapper::loadParamsAndData(std::string funcName, RooAbsArg const *head, RooArgSet const &paramSet,
                                       const RooAbsData *data)
{
   // Extract parameters
   for (auto *param : paramSet) {
      if (!dynamic_cast<RooAbsReal *>(param)) {
         std::stringstream errorMsg;
         errorMsg << "In creation of function " << funcName
                  << " wrapper: input param expected to be of type RooAbsReal.";
         coutE(InputArguments) << errorMsg.str() << std::endl;
         throw std::runtime_error(errorMsg.str().c_str());
      }
      _params.add(*param);
   }
   _gradientVarBuffer.reserve(_params.size());

   if (data == nullptr)
      return;

   // Extract observables
   std::stack<std::vector<double>> vectorBuffers; // for data loading
   auto spans = RooFit::BatchModeDataHelpers::getDataSpans(*data, "", "", vectorBuffers, true);

   for (auto const &item : spans) {
      std::size_t n = item.second.size();
      _obsInfos.emplace(item.first, ObsInfo{_observables.size(), n});
      _observables.reserve(_observables.size() + n);
      for (std::size_t i = 0; i < n; ++i) {
         _observables.push_back(item.second[i]);
      }
   }

   if (head) {
      _nodeOutputSizes = RooFit::BatchModeDataHelpers::determineOutputSizes(*head, spans);
   }
}

void RooFuncWrapper::declareAndDiffFunction(std::string funcName, std::string const &funcBody)
{
   std::string gradName = funcName + "_grad_0";
   std::string requestName = funcName + "_req";
   std::string wrapperName = funcName + "_derivativeWrapper";

   gInterpreter->Declare("#pragma cling optimize(2)");

   // Declare the function
   std::stringstream bodyWithSigStrm;
   bodyWithSigStrm << "double " << funcName << "(double* params, double const* obs) {\n" << funcBody << "\n}";
   bool comp = gInterpreter->Declare(bodyWithSigStrm.str().c_str());
   if (!comp) {
      std::stringstream errorMsg;
      errorMsg << "Function " << funcName << " could not be compiled. See above for details.";
      coutE(InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }
   _func = reinterpret_cast<Func>(gInterpreter->ProcessLine((funcName + ";").c_str()));

   // Calculate gradient
   gInterpreter->ProcessLine("#include <Math/CladDerivator.h>");
   // disable clang-format for making the following code unreadable.
   // clang-format off
   std::stringstream requestFuncStrm;
   requestFuncStrm << "#pragma clad ON\n"
                      "void " << requestName << "() {\n"
                      "  clad::gradient(" << funcName << ", \"params\");\n"
                      "}\n"
                      "#pragma clad OFF";
   // clang-format on
   comp = gInterpreter->Declare(requestFuncStrm.str().c_str());
   if (!comp) {
      std::stringstream errorMsg;
      errorMsg << "Function " << funcName << " could not be differentiated. See above for details.";
      coutE(InputArguments) << errorMsg.str() << std::endl;
      throw std::runtime_error(errorMsg.str().c_str());
   }

   // Build a wrapper over the derivative to hide clad specific types such as 'array_ref'.
   // disable clang-format for making the following code unreadable.
   // clang-format off
   std::stringstream dWrapperStrm;
   dWrapperStrm << "void " << wrapperName << "(double* params, double const* obs, double* out) {\n"
                   "  clad::array_ref<double> cladOut(out, " << _params.size() << ");\n"
                   "  " << gradName << "(params, obs, cladOut);\n"
                   "}";
   // clang-format on
   gInterpreter->Declare(dWrapperStrm.str().c_str());
   _grad = reinterpret_cast<Grad>(gInterpreter->ProcessLine((wrapperName + ";").c_str()));
}

void RooFuncWrapper::getGradient(double *out) const
{
   updateGradientVarBuffer();
   std::fill(out, out + _params.size(), 0.0);

   _grad(_gradientVarBuffer.data(), _observables.data(), out);
}

void RooFuncWrapper::updateGradientVarBuffer() const
{
   std::transform(_params.begin(), _params.end(), _gradientVarBuffer.begin(),
                  [](RooAbsArg *obj) { return static_cast<RooAbsReal *>(obj)->getVal(); });
}

double RooFuncWrapper::evaluate() const
{
   updateGradientVarBuffer();

   return _func(_gradientVarBuffer.data(), _observables.data());
}

void RooFuncWrapper::gradient(const double *x, double *g) const
{
   std::fill(g, g + _params.size(), 0.0);

   _grad(const_cast<double *>(x), _observables.data(), g);
}

std::string RooFuncWrapper::buildCode(RooAbsReal const &head)
{
   RooFit::Detail::CodeSquashContext ctx(_nodeOutputSizes);

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
