/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROO_NLL_VAR
#define ROO_NLL_VAR

#include "RooAbsOptTestStatistic.h"
#include "RooCmdArg.h"
#include "RooAbsPdf.h"
#include <vector>
#include <utility>

class RooNLLVar : public RooAbsOptTestStatistic {
public:

  // Constructors, assignment etc
  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
            bool extended,
            RooAbsTestStatistic::Configuration const& cfg=RooAbsTestStatistic::Configuration{});

  RooNLLVar(const char *name, const char *title, RooAbsPdf& pdf, RooAbsData& data,
            const RooArgSet& projDeps, bool extended = false,
            RooAbsTestStatistic::Configuration const& cfg=RooAbsTestStatistic::Configuration{});

  RooNLLVar(const RooNLLVar& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooNLLVar(*this,newname); }

  RooAbsTestStatistic* create(const char *name, const char *title, RooAbsReal& pdf, RooAbsData& adata,
                                      const RooArgSet& projDeps, RooAbsTestStatistic::Configuration const& cfg) override;

  ~RooNLLVar() override;

  void applyWeightSquared(bool flag) override;

  double defaultErrorLevel() const override { return 0.5 ; }

  void enableBinOffsetting(bool on = true);

  using ComputeResult = std::pair<ROOT::Math::KahanSum<double>, double>;

  static RooNLLVar::ComputeResult computeScalarFunc(const RooAbsPdf *pdfClone, RooAbsData *dataClone, RooArgSet *normSet,
                                                bool weightSq, std::size_t stepSize, std::size_t firstEvent,
                                                std::size_t lastEvent, RooAbsPdf const* offsetPdf = nullptr);

  bool setDataSlave(RooAbsData& data, bool cloneData=true, bool ownNewDataAnyway=false) override;

protected:

  bool processEmptyDataSets() const override { return _extended ; }
  double evaluatePartition(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const override;

  static RooArgSet _emptySet ; // Supports named argument constructor

private:
  ComputeResult computeScalar(std::size_t stepSize, std::size_t firstEvent, std::size_t lastEvent) const;

  bool _extended{false};
  bool _doBinOffset{false};
  bool _weightSq{false}; ///< Apply weights squared?
  mutable bool _first{true}; ///<!
  ROOT::Math::KahanSum<double> _offsetSaveW2{0.0}; ///<!

  mutable std::vector<double> _binw ; ///<!
  mutable RooAbsPdf* _binnedPdf{nullptr}; ///<!
  std::unique_ptr<RooAbsPdf> _offsetPdf; ///<! An optional per-bin likelihood offset
};

#endif

/// \endcond
