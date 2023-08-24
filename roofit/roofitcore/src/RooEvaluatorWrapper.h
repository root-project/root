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
#include "RooFit/Detail/DataMap.h"
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooRealProxy.h>
#include <RooFit/Evaluator.h>

#include "RooFit/Detail/Buffers.h"

#include <chrono>
#include <memory>
#include <stack>

class RooAbsArg;
class RooAbsCategory;
class RooSimultaneous;

class RooEvaluatorWrapper final : public RooAbsReal {
public:
   RooEvaluatorWrapper(RooAbsReal &topNode, std::unique_ptr<RooFit::Evaluator> evaluator, std::string const &rangeName,
                       RooSimultaneous const *simPdf, bool takeGlobalObservablesFromData);

   RooEvaluatorWrapper(const RooEvaluatorWrapper &other, const char *name = nullptr);

   TObject *clone(const char *newname) const override { return new RooEvaluatorWrapper(*this, newname); }

   double defaultErrorLevel() const override { return _topNode->defaultErrorLevel(); }

   bool getParameters(const RooArgSet *observables, RooArgSet &outputSet, bool stripDisconnected) const override;

   bool setData(RooAbsData &data, bool cloneData) override;

   double getValV(const RooArgSet *) const override { return evaluate(); }

   void applyWeightSquared(bool flag) override { _topNode->applyWeightSquared(flag); }

   void printMultiline(std::ostream &os, Int_t /*contents*/, bool /*verbose*/ = false,
                       TString /*indent*/ = "") const override
   {
      _evaluator->print(os);
   }

protected:
   double evaluate() const override { return _evaluator ? _evaluator->run()[0] : 0.0; }

private:
   std::shared_ptr<RooFit::Evaluator> _evaluator;
   RooRealProxy _topNode;
   RooAbsData *_data = nullptr;
   RooArgSet _parameters;
   std::string _rangeName;
   RooSimultaneous const *_simPdf = nullptr;
   const bool _takeGlobalObservablesFromData;
   std::stack<std::vector<double>> _vectorBuffers; // used for preserving resources
};

#endif

/// \endcond
