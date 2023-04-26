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

#include <RooAbsPdf.h>
#include <RooAbsReal.h>
#include <RooGlobalFunc.h>
#include <RooTemplateProxy.h>

#include <Math/Util.h>

#include "RooBatchComputeTypes.h"

namespace ROOT {
namespace Experimental {

class RooNLLVarNew : public RooAbsReal {

public:
   // The names for the weight variables that the RooNLLVarNew expects
   static constexpr const char *weightVarName = "_weight";
   static constexpr const char *weightVarNameSumW2 = "_weight_sumW2";

   RooNLLVarNew() {}
   RooNLLVarNew(const char *name, const char *title, RooAbsPdf &pdf, RooArgSet const &observables, bool isExtended,
                RooFit::OffsetMode offsetMode);
   RooNLLVarNew(const RooNLLVarNew &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooNLLVarNew(*this, newname); }

   void getParametersHook(const RooArgSet *nset, RooArgSet *list, bool stripDisconnected) const override;

   /// Return default level for MINUIT error analysis.
   double defaultErrorLevel() const override { return 0.5; }

   void computeBatch(cudaStream_t *, double *output, size_t nOut, RooFit::Detail::DataMap const &) const override;
   bool canComputeBatchWithCuda() const override { return !_binnedL; }
   bool isReducerNode() const override { return true; }

   void setPrefix(std::string const &prefix);

   void applyWeightSquared(bool flag) override;

   void enableOffsetting(bool) override;

   void enableBinOffsetting(bool on = true) { _doBinOffset = on; }

   void setSimCount(int simCount) { _simCount = simCount; }

private:
   double evaluate() const override { return _value; }
   void resetWeightVarNames();
   double finalizeResult(ROOT::Math::KahanSum<double> result, double weightSum) const;
   void fillBinWidthsFromPdfBoundaries(RooAbsReal const &pdf);
   double computeBatchBinnedL(RooSpan<const double> preds, RooSpan<const double> weights) const;

   RooTemplateProxy<RooAbsPdf> _pdf;
   RooArgSet _observables;
   mutable double _sumWeight = 0.0;  //!
   mutable double _sumWeight2 = 0.0; //!
   bool _isExtended;
   bool _weightSquared = false;
   bool _binnedL = false;
   bool _doOffset = false;
   bool _doBinOffset = false;
   int _simCount = 1;
   std::string _prefix;
   RooTemplateProxy<RooAbsReal> _weightVar;
   RooTemplateProxy<RooAbsReal> _weightSquaredVar;
   RooTemplateProxy<RooAbsReal> _binVolumeVar;
   std::vector<double> _binw;
   mutable ROOT::Math::KahanSum<double> _offset{0.}; ///<! Offset as KahanSum to avoid loss of precision

}; // end class RooNLLVar

} // end namespace Experimental
} // end namespace ROOT

#endif
