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

#ifndef RooFit_RooEvaluatorWrapper_h
#define RooFit_RooEvaluatorWrapper_h

#include <RooAbsData.h>
#include <RooFit/EvalContext.h>
#include <RooFit/Evaluator.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooRealProxy.h>
#include <RooSetProxy.h>

#include "RooFit/BatchModeDataHelpers.h"

#include <chrono>
#include <memory>
#include <stack>

class RooAbsArg;
class RooAbsCategory;
class RooAbsPdf;
class RooFuncWrapper;

class RooEvaluatorWrapper final : public RooAbsReal {
public:
   RooEvaluatorWrapper(RooAbsReal &topNode, RooAbsData *data, bool useGPU, std::string const &rangeName,
                       RooAbsPdf const *simPdf, bool takeGlobalObservablesFromData);

   RooEvaluatorWrapper(const RooEvaluatorWrapper &other, const char *name = nullptr);

   ~RooEvaluatorWrapper();

   TObject *clone(const char *newname) const override { return new RooEvaluatorWrapper(*this, newname); }

   double defaultErrorLevel() const override { return _topNode->defaultErrorLevel(); }

   bool getParameters(const RooArgSet *observables, RooArgSet &outputSet, bool stripDisconnected = true) const override;

   bool setData(RooAbsData &data, bool cloneData) override;

   double getValV(const RooArgSet *) const override { return evaluate(); }

   void applyWeightSquared(bool flag) override { _topNode->applyWeightSquared(flag); }

   void printMultiline(std::ostream &os, Int_t /*contents*/, bool /*verbose*/ = false,
                       TString /*indent*/ = "") const override
   {
      _evaluator->print(os);
   }

   /// The RooFit::Evaluator is dealing with constant terms itself.
   void constOptimizeTestStatistic(ConstOpCode /*opcode*/, bool /*doAlsoTrackingOpt*/) override {}

   bool hasGradient() const override;

   void gradient(double *out) const override;

   void generateGradient();

   void setUseGeneratedFunctionCode(bool);

protected:
   double evaluate() const override;

private:
   void createFuncWrapper();

   std::shared_ptr<RooFit::Evaluator> _evaluator;
   std::shared_ptr<RooFuncWrapper> _funcWrapper;
   RooRealProxy _topNode;
   RooAbsData *_data = nullptr;
   RooSetProxy _paramSet;
   std::string _rangeName;
   RooAbsPdf const *_pdf = nullptr;
   const bool _takeGlobalObservablesFromData;
   bool _useGeneratedFunctionCode = false;
   std::stack<std::vector<double>> _vectorBuffers; // used for preserving resources
   std::map<RooFit::Detail::DataKey, std::span<const double>> _dataSpans;
};

#endif

/// \endcond
