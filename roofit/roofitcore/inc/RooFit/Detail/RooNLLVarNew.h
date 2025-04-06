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

namespace RooFit {
namespace Detail {

class RooNLLVarNew : public RooAbsReal {

public:
   // The names for the weight variables that the RooNLLVarNew expects
   static constexpr const char *weightVarName = "_weight";
   static constexpr const char *weightVarNameSumW2 = "_weight_sumW2";

   RooNLLVarNew(const char *name, const char *title, RooAbsPdf &pdf, RooArgSet const &observables, bool isExtended,
                RooFit::OffsetMode offsetMode);
   RooNLLVarNew(const RooNLLVarNew &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooNLLVarNew(*this, newname); }

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

   RooAbsPdf const &pdf() const { return *_pdf; }
   RooAbsReal const &weightVar() const { return *_weightVar; }
   bool binnedL() const { return _binnedL; }
   int simCount() const { return _simCount; }
   RooAbsReal const *expectedEvents() const { return _expectedEvents ? &**_expectedEvents : nullptr; }

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
   bool _weightSquared = false;
   bool _binnedL = false;
   bool _doOffset = false;
   bool _doBinOffset = false;
   int _simCount = 1;
   std::string _prefix;
   std::vector<double> _binw;
   mutable ROOT::Math::KahanSum<double> _offset{0.}; ///<! Offset as KahanSum to avoid loss of precision

   ClassDefOverride(RooFit::Detail::RooNLLVarNew, 0);
};

} // namespace Detail
} // namespace RooFit

#endif

/// \endcond
