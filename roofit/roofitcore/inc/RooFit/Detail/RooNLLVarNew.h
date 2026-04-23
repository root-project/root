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
#include <RooDataHist.h>
#include <RooGlobalFunc.h>
#include <RooTemplateProxy.h>

#include <Math/Util.h>

namespace RooFit {
namespace Detail {

class RooNLLVarNew : public RooAbsReal {

public:
   // The names for the special variables that the RooNLLVarNew expects
   static constexpr const char *weightVarName = "_weight";
   static constexpr const char *weightVarNameSumW2 = "_weight_sumW2";
   static constexpr const char *binVolumeVarName = "_bin_volume";
   static constexpr const char *weightErrorLoVarName = "_weight_err_lo";
   static constexpr const char *weightErrorHiVarName = "_weight_err_hi";

   enum class Statistic {
      NLL,
      Chi2
   };

   /// Configuration struct for the unified constructor. Note that `offsetMode`
   /// only applies to `Statistic::NLL`, and `chi2ErrorType` only applies to
   /// `Statistic::Chi2`.
   struct Config {
      Statistic statistic = Statistic::NLL;
      bool extended = false;
      RooFit::OffsetMode offsetMode = RooFit::OffsetMode::None;
      RooDataHist::ErrorType chi2ErrorType = RooDataHist::Expected;
   };

   RooNLLVarNew(const char *name, const char *title, RooAbsReal &func, RooArgSet const &observables, Config const &cfg);
   RooNLLVarNew(const RooNLLVarNew &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooNLLVarNew(*this, newname); }

   /// Return default level for MINUIT error analysis.
   double defaultErrorLevel() const override { return _statistic == Statistic::Chi2 ? 1.0 : 0.5; }

   void doEval(RooFit::EvalContext &) const override;
   bool canComputeBatchWithCuda() const override { return _statistic == Statistic::NLL && !_binnedL; }
   bool isReducerNode() const override { return true; }

   void setPrefix(std::string const &prefix);

   void applyWeightSquared(bool flag) override;

   void enableOffsetting(bool) override;

   void enableBinOffsetting(bool on = true) { _doBinOffset = on; }

   void setSimCount(int simCount) { _simCount = simCount; }

   enum class FuncMode {
      Pdf,
      ExtendedPdf,
      Function
   };

   RooAbsReal const &func() const { return *_func; }
   RooAbsReal const &weightVar() const { return *_weightVar; }
   RooAbsReal const &weightSquaredVar() const { return *_weightSquaredVar; }
   bool binnedL() const { return _binnedL; }
   int simCount() const { return _simCount; }
   Statistic statistic() const { return _statistic; }
   FuncMode funcMode() const { return _funcMode; }
   RooDataHist::ErrorType chi2ErrorType() const { return _chi2ErrorType; }
   RooAbsReal const *expectedEvents() const { return _expectedEvents ? &**_expectedEvents : nullptr; }
   RooAbsReal const *binVolumes() const { return _binVolumes ? &**_binVolumes : nullptr; }
   RooAbsReal const *weightErrLo() const { return _weightErrLo ? &**_weightErrLo : nullptr; }
   RooAbsReal const *weightErrHi() const { return _weightErrHi ? &**_weightErrHi : nullptr; }

private:
   double evaluate() const override { return _value; }
   void resetWeightVarNames();
   void finalizeResult(RooFit::EvalContext &, ROOT::Math::KahanSum<double> result, double weightSum) const;
   void fillBinWidthsFromPdfBoundaries(RooAbsReal const &pdf, RooArgSet const &observables);
   void doEvalBinnedL(RooFit::EvalContext &, std::span<const double> preds, std::span<const double> weights) const;
   void doEvalChi2(RooFit::EvalContext &, std::span<const double> preds, std::span<const double> weights,
                   std::span<const double> weightsSumW2) const;

   RooTemplateProxy<RooAbsReal> _func;
   RooTemplateProxy<RooAbsReal> _weightVar;
   RooTemplateProxy<RooAbsReal> _weightSquaredVar;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _expectedEvents;
   std::unique_ptr<RooTemplateProxy<RooAbsPdf>> _offsetPdf;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _binVolumes;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _weightErrLo;
   std::unique_ptr<RooTemplateProxy<RooAbsReal>> _weightErrHi;
   bool _weightSquared = false;
   bool _binnedL = false;
   bool _doOffset = false;
   bool _doBinOffset = false;
   Statistic _statistic = Statistic::NLL;
   FuncMode _funcMode = FuncMode::Pdf;
   RooDataHist::ErrorType _chi2ErrorType = RooDataHist::Expected;
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
