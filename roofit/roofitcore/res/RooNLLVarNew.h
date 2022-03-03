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
   RooNLLVarNew(){};
   RooNLLVarNew(const char *name, const char *title, RooAbsPdf &pdf, RooArgSet const &observables, RooAbsReal *weight,
                bool isExtended, std::string const &rangeName);
   RooNLLVarNew(const RooNLLVarNew &other, const char *name = 0);
   TObject *clone(const char *newname) const override { return new RooNLLVarNew(*this, newname); }

   void getParametersHook(const RooArgSet *nset, RooArgSet *list, Bool_t stripDisconnected) const override;

   /// Return default level for MINUIT error analysis.
   double defaultErrorLevel() const override { return 0.5; }

   inline RooAbsPdf *getPdf() const { return &*_pdf; }
   void computeBatch(cudaStream_t *, double *output, size_t nOut, RooBatchCompute::DataMap &) const override;
   inline bool isReducerNode() const override { return true; }

   void setObservables(RooArgSet const &observables)
   {
      _observables.clear();
      _observables.add(observables);
   }

protected:
   RooTemplateProxy<RooAbsPdf> _pdf;
   RooArgSet _observables;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _weight;
   mutable double _sumWeight = 0.0;         //!
   mutable double _sumCorrectionTerm = 0.0; //!
   bool _isExtended;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _rangeNormTerm;

   double evaluate() const override;

}; // end class RooNLLVar

} // end namespace Experimental
} // end namespace ROOT

#endif
