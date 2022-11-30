/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_BatchModeHelpers_h
#define RooFit_BatchModeHelpers_h

#include <RooGlobalFunc.h>
#include <RooAbsReal.h>
#include <RooRealProxy.h>
#include <RooArgSet.h>

#include <memory>
#include <string>

class RooAbsData;
class RooAbsPdf;
class RooSimultaneous;

namespace ROOT {
namespace Experimental {
class RooFitDriver;
}
} // namespace ROOT

class RooAbsRealWrapper final : public RooAbsReal {
public:
   RooAbsRealWrapper(std::unique_ptr<ROOT::Experimental::RooFitDriver> driver, std::string const &rangeName,
                     RooSimultaneous const *simPdf, bool splitRange, bool takeGlobalObservablesFromData);

   RooAbsRealWrapper(const RooAbsRealWrapper &other, const char *name = nullptr);

   TObject *clone(const char *newname) const override { return new RooAbsRealWrapper(*this, newname); }

   double defaultErrorLevel() const override;

   bool getParameters(const RooArgSet *observables, RooArgSet &outputSet, bool /*stripDisconnected*/) const override;

   bool setData(RooAbsData &data, bool /*cloneData*/) override;

   double getValV(const RooArgSet *) const override { return evaluate(); }

   void applyWeightSquared(bool flag) override;
   void printMultiline(std::ostream &os, Int_t /*contents*/, bool /*verbose*/ = false,
                       TString /*indent*/ = "") const override;
   std::shared_ptr<ROOT::Experimental::RooFitDriver> getRooFitDriverObj() { return _driver; }

protected:
   double evaluate() const override;

private:
   std::shared_ptr<ROOT::Experimental::RooFitDriver> _driver;
   RooRealProxy _topNode;
   RooAbsData *_data = nullptr;
   RooArgSet _parameters;
   std::string _rangeName;
   RooSimultaneous const *_simPdf = nullptr;
   bool _splitRange = false;
   const bool _takeGlobalObservablesFromData;
};

namespace RooFit {
namespace BatchModeHelpers {
std::unique_ptr<RooAbsArg> createSimultaneousNLL(RooSimultaneous const &simPdf, RooArgSet &observables, bool isExtended,
                                                 std::string const &rangeName, bool doOffset, bool splitRange);

std::unique_ptr<RooAbsReal>
createNLL(std::unique_ptr<RooAbsPdf> &&pdf, RooAbsData &data, std::unique_ptr<RooAbsReal> &&constraints,
          std::string const &rangeName, RooArgSet const &projDeps, bool isExtended, double integrateOverBinsPrecision,
          RooFit::BatchModeOption batchMode, bool doOffset, bool splitRange, bool takeGlobalObservablesFromData);

void logArchitectureInfo(RooFit::BatchModeOption batchMode);

} // namespace BatchModeHelpers
} // namespace RooFit

#endif
