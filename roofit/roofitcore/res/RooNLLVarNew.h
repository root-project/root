/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2021
 *   Emmanouil Michalainas, CERN 2021
 *
 * Copyright (c) 2021, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooNLLVarNew_h
#define RooFit_RooNLLVarNew_h

#include "RooAbsPdf.h"
#include "RooAbsReal.h"
#include "RooTemplateProxy.h"

#include "RooBatchComputeTypes.h"

namespace ROOT {
namespace Experimental {

class RooNLLVarNew : public RooAbsReal {

public:
   // The names for the weight variables that the RooNLLVarNew expects
   static constexpr const char *weightVarName = "_weight";
   static constexpr const char *weightVarNameSumW2 = "_weight_sumW2";

   RooNLLVarNew(){};
   RooNLLVarNew(const char *name, const char *title, RooAbsPdf &pdf, RooArgSet const &observables, bool isExtended,
                std::string const &rangeName);
   RooNLLVarNew(const RooNLLVarNew &other, const char *name = 0);
   TObject *clone(const char *newname) const override { return new RooNLLVarNew(*this, newname); }

   void getParametersHook(const RooArgSet *nset, RooArgSet *list, Bool_t stripDisconnected) const override;

   /// Return default level for MINUIT error analysis.
   double defaultErrorLevel() const override { return 0.5; }

   inline RooAbsPdf *getPdf() const { return &*_pdf; }
   void computeBatch(cudaStream_t *, double *output, size_t nOut, RooFit::Detail::DataMap const&) const override;
   inline bool isReducerNode() const override { return true; }

   RooArgSet prefixObservableAndWeightNames(std::string const &prefix);

   void applyWeightSquared(bool flag) override;

protected:
   void setObservables(RooArgSet const &observables)
   {
      _observables.clear();
      _observables.add(observables);
   }

   RooTemplateProxy<RooAbsPdf> _pdf;
   RooArgSet _observables;
   mutable double _sumWeight = 0.0;         //!
   mutable double _sumWeight2 = 0.0;        //!
   bool _isExtended;
   bool _weightSquared = false;
   std::string _prefix;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _fractionInRange;

   double evaluate() const override;

}; // end class RooNLLVar

} // end namespace Experimental
} // end namespace ROOT

#endif
