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

#include "RooBatchCompute.h"

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
   inline bool canComputeBatchWithCuda() const override { return true; }
   inline bool isReducerNode() const override { return true; }

protected:
   double reduce(cudaStream_t *, const double *input, size_t nEvents) const;

   RooTemplateProxy<RooAbsPdf> _pdf;
   RooArgSet const *_observables = nullptr;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _weight;
   mutable double _sumWeight = 0.0;         //!
   mutable double _sumCorrectionTerm = 0.0; //!
   bool _isExtended;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _rangeNormTerm;

   double getValV(const RooArgSet *normalisationSet = nullptr) const override;

   double evaluate() const override;

   RooSpan<double> evaluateSpan(RooBatchCompute::RunContext &evalData, const RooArgSet *normSet) const override;

   RooSpan<const double>
   getValues(RooBatchCompute::RunContext &evalData, const RooArgSet *normSet = nullptr) const override;

}; // end class RooNLLVar
} // end namespace Experimental
} // end namespace ROOT

#endif
