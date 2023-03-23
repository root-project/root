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

#include "RooFuncWrapper.h"

#include "TROOT.h"
#include "TSystem.h"
#include "RooAbsData.h"
#include "RooGlobalFunc.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooHelpers.h"
#include "RooFit/Detail/CodeSquashContext.h"

RooFuncWrapper::RooFuncWrapper(const char *name, const char *title, std::string const &funcBody,
                               RooArgSet const &paramSet, RooArgSet const &obsSet, const RooAbsData *data /*=nullptr*/)
   : RooAbsReal{name, title}, _params{"!params", "List of parameters", this}
{
   // Declare the function and create its derivative.
   declareAndDiffFunction(name, funcBody);

   // Load the parameters and observables.
   loadParamsAndObs(name, paramSet, obsSet, data);
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
   pdf->getParameters(data ? data->get() : nullptr, paramSet);
   // Get the observable if we have a valid dataset.
   RooArgSet obsSet;
   if (data)
      pdf->getObservables(data->get(), obsSet);

   // Load the parameters and observables.
   loadParamsAndObs(name, paramSet, obsSet, data);

   func = buildCode(*pdf, paramSet, obsSet, data);

   // Declare the function and create its derivative.
   declareAndDiffFunction(name, func);
}

RooFuncWrapper::RooFuncWrapper(const RooFuncWrapper &other, const char *name /*=nullptr*/)
   : RooAbsReal(other, name),
     _params("!params", this, other._params),
     _func(other._func),
     _grad(other._grad),
     _gradientVarBuffer(other._gradientVarBuffer),
     _observables(other._observables)
{
}

void RooFuncWrapper::loadParamsAndObs(std::string funcName, RooArgSet const &paramSet, RooArgSet const &obsSet,
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

   // Extract observables
   if (!obsSet.empty()) {
      auto dataSpans = data->getBatches(0, data->numEntries());
      _observables.reserve(_observables.size() * data->numEntries());
      for (auto *obs : static_range_cast<RooRealVar *>(obsSet)) {
         RooSpan<const double> span{dataSpans.at(obs)};
         for (int i = 0; i < data->numEntries(); ++i) {
            _observables.push_back(span[i]);
         }
      }
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
   bodyWithSigStrm << "double " << funcName << "(double* params, double* obs) {\n" << funcBody << "\n}";
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
   dWrapperStrm << "void " << wrapperName << "(double* params, double* obs, double* out) {\n"
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

std::string RooFuncWrapper::buildCode(RooAbsReal const &head, RooArgSet const & /* paramSet */,
                               RooArgSet const &obsSet, const RooAbsData *data)
{
   RooFit::Detail::CodeSquashContext ctx;

   // First update the result variable of params in the compute graph to in[<position>].
   int idx = 0;
   for (RooAbsArg *param : _params) {
      ctx.addResult(param, "params[" + std::to_string(idx) + "]");
      idx++;
   }

   // Also update observables...
   idx = 0;
   if (!obsSet.empty()) {
      auto dataSpans = data->getBatches(0, data->numEntries());
      _observables.reserve(_observables.size() * data->numEntries());
      for (auto *obs : static_range_cast<RooRealVar *>(obsSet)) {
         RooSpan<const double> span{dataSpans.at(obs)};
         // If the observable is a scalar, set its name to the start index.
         // else, store the start index and later set the the name to obs[start_idx + curr_idx], here curr_idx is
         // defined by a loop producing parent node.
         if (data->numEntries() == 1)
            ctx.addResult(obs, "obs[" + std::to_string(idx) + "]");
         else
            ctx.addVecObs(obs, idx);
         idx += data->numEntries();
      }
   }

   // This will not work for nodes that produce loops as we need to keep track of the subtree of the loop producing
   // node. A better approach is to have 2 stacks and perform an iterative post order traversal on the graph/resolved
   // tree.
   RooArgSet nodes;
   RooHelpers::getSortedComputationGraph(head, nodes);

   for (RooAbsArg *node : nodes) {
      RooAbsReal *curr = dynamic_cast<RooAbsReal *>(node);
      if (!curr) {
         std::stringstream errorMsg;
         errorMsg << "Translate is only supported for RooAbsReal derived objects.";
         oocoutE(nullptr, Minimization) << errorMsg.str() << std::endl;
         throw std::runtime_error(errorMsg.str().c_str());
      }
      curr->translate(ctx);
   }

   return ctx.assembleCode(ctx.getResult(&head));
}
