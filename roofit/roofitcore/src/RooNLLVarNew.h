/// \cond ROOFIT_INTERNAL

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

class RooNLLVarNew : public RooAbsReal {

public:
   // The names for the weight variables that the RooNLLVarNew expects
   static constexpr const char *weightVarName = "_weight";
   static constexpr const char *weightVarNameSumW2 = "_weight_sumW2";

   RooNLLVarNew(const char *name, const char *title, RooAbsPdf &pdf, RooArgSet const &observables, bool isExtended,
                RooFit::OffsetMode offsetMode);
   RooNLLVarNew(const RooNLLVarNew &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooNLLVarNew(*this, newname); }

   void getParametersHook(const RooArgSet *nset, RooArgSet *list, bool stripDisconnected) const override;

   /// Return default level for MINUIT error analysis.
   double defaultErrorLevel() const override { return 0.5; }

   void doEval(RooFit::EvalContext &) const override;
   bool canComputeBatchWithCuda() const override { return !_binnedL; }
   bool isReducerNode() const override { return true; }

   void setPrefix(std::string const &prefix);

   void applyWeightSquared(bool flag) override;

   void enableOffsetting(bool) override;

   void enableBinOffsetting(bool on = true) { _doBinOffset = on; }

   void setSimCount(int simCount) { _simCount = simCount; }

   void translate(RooFit::Detail::CodeSquashContext &ctx) const override;

private:
   double evaluate() const override { return _value; }
   void resetWeightVarNames();
   void finalizeResult(RooFit::EvalContext &, ROOT::Math::KahanSum<double> result, double weightSum) const;
   void fillBinWidthsFromPdfBoundaries(RooAbsReal const &pdf, RooArgSet const &observables);
   void doEvalBinnedL(RooFit::EvalContext &, std::span<const double> preds, std::span<const double> weights) const;

   RooTemplateProxy<RooAbsPdf> _pdf;
   RooTemplateProxy<RooAbsReal> _weightVar;
   RooTemplateProxy<RooAbsReal> _weightSquaredVar;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _expectedEvents;
   std::unique_ptr<RooTemplateProxy<RooAbsPdf>> _offsetPdf;
   mutable double _sumWeight = 0.0;  //!
   mutable double _sumWeight2 = 0.0; //!
   bool _weightSquared = false;
   bool _binnedL = false;
   bool _doOffset = false;
   bool _doBinOffset = false;
   int _simCount = 1;
   std::string _prefix;
   std::vector<double> _binw;
   mutable ROOT::Math::KahanSum<double> _offset{0.}; ///<! Offset as KahanSum to avoid loss of precision

}; // end class RooNLLVar

#endif

/// \endcond
