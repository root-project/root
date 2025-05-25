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

#ifndef ROO_CHI2_VAR
#define ROO_CHI2_VAR

#include "RooAbsOptTestStatistic.h"
#include "RooCmdArg.h"
#include "RooDataHist.h"
#include "RooAbsPdf.h"

class RooChi2Var : public RooAbsOptTestStatistic {
public:
  enum FuncMode { Function, Pdf, ExtendedPdf } ;

  // Constructors, assignment etc
  RooChi2Var(const char *name, const char *title, RooAbsReal& func, RooDataHist& data,
             bool extended, RooDataHist::ErrorType etype,
             RooAbsTestStatistic::Configuration const& cfg=RooAbsTestStatistic::Configuration{});

  RooChi2Var(const RooChi2Var& other, const char* name=nullptr);
  TObject* clone(const char* newname=nullptr) const override { return new RooChi2Var(*this,newname); }

  RooAbsTestStatistic* create(const char *name, const char *title, RooAbsReal& pdf, RooAbsData& dhist,
                                      const RooArgSet& projDeps, RooAbsTestStatistic::Configuration const& cfg) override {
    // Virtual constructor
    return new RooChi2Var(name,title,(RooAbsPdf&)pdf,(RooDataHist&)dhist,projDeps,_funcMode,cfg,_etype) ;
  }

  double defaultErrorLevel() const override {
    // The default error level for MINUIT error analysis for a chi^2 is 1.0
    return 1.0 ;
  }

private:

  RooChi2Var(const char *name, const char *title, RooAbsReal& func, RooDataHist& data,
             const RooArgSet& projDeps, FuncMode funcMode,
             RooAbsTestStatistic::Configuration const& cfg,
             RooDataHist::ErrorType etype)
    : RooAbsOptTestStatistic(name,title,func,data,projDeps,cfg), _etype(etype), _funcMode(funcMode) {}

protected:

  double evaluatePartition(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const override ;

  static RooArgSet _emptySet ;        ///< Supports named argument constructor

  RooDataHist::ErrorType _etype ;     ///< Error type store in associated RooDataHist
  FuncMode _funcMode ;                ///< Function, P.d.f. or extended p.d.f?
};


#endif

/// \endcond
