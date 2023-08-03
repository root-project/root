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

#ifndef RooFit_RooFitDriverWrapper_h
#define RooFit_RooFitDriverWrapper_h

#include <RooAbsData.h>
#include "RooFit/Detail/DataMap.h"
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooRealProxy.h>
#include "RooFit/Detail/Buffers.h"
#include "RooFitDriver.h"

#include <chrono>
#include <memory>
#include <stack>

class RooAbsArg;
class RooAbsCategory;
class RooSimultaneous;

class RooFitDriverWrapper final : public RooAbsReal {
public:
   RooFitDriverWrapper(RooAbsReal &topNode, std::unique_ptr<ROOT::Experimental::RooFitDriver> driver,
                       std::string const &rangeName, RooSimultaneous const *simPdf, bool takeGlobalObservablesFromData);

   RooFitDriverWrapper(const RooFitDriverWrapper &other, const char *name = nullptr);

   TObject *clone(const char *newname) const override { return new RooFitDriverWrapper(*this, newname); }

   double defaultErrorLevel() const override { return _topNode->defaultErrorLevel(); }

   bool getParameters(const RooArgSet *observables, RooArgSet &outputSet, bool stripDisconnected) const override;

   bool setData(RooAbsData &data, bool cloneData) override;

   double getValV(const RooArgSet *) const override { return evaluate(); }

   void applyWeightSquared(bool flag) override { _topNode->applyWeightSquared(flag); }

   void printMultiline(std::ostream &os, Int_t /*contents*/, bool /*verbose*/ = false,
                       TString /*indent*/ = "") const override
   {
      _driver->print(os);
   }

protected:
   double evaluate() const override { return _driver ? _driver->run()[0] : 0.0; }

private:
   std::shared_ptr<ROOT::Experimental::RooFitDriver> _driver;
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
