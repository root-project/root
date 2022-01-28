/*
 * Project: RooFit
 * Authors:
 *   PB, Patrick Bos, Netherlands eScience Center, p.bos@esciencecenter.nl
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROOT_ROOFIT_TESTSTATISTICS_RooRealL
#define ROOT_ROOFIT_TESTSTATISTICS_RooRealL

#include "RooAbsReal.h"
#include "RooSetProxy.h"

#include "Rtypes.h" // ClassDef, ClassImp

#include <memory> // shared_ptr

namespace RooFit {
namespace TestStatistics {

class RooAbsL;

class RooRealL : public RooAbsReal {
public:
   RooRealL(const char *name, const char *title, std::shared_ptr<RooAbsL> likelihood);
   RooRealL(const RooRealL &other, const char *name = 0);

   inline TObject *clone(const char *newname) const override { return new RooRealL(*this, newname); }

   inline double globalNormalization() const
   {
      // Default value of global normalization factor is 1.0
      return 1.0;
   }

   inline double getCarry() const { return eval_carry; }
   inline Double_t defaultErrorLevel() const override { return 0.5; }

protected:
   Double_t evaluate() const override;

private:
   std::shared_ptr<RooAbsL> likelihood_;
   mutable double eval_carry = 0;
   RooSetProxy vars_proxy_; ///< sets up client-server connections
   RooArgSet vars_obs_;     ///< list of observables

   ClassDefOverride(RooRealL, 0);
};

} // namespace TestStatistics
} // namespace RooFit

#endif // ROOT_ROOFIT_TESTSTATISTICS_RooRealL
