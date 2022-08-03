/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooChi2Var.h,v 1.12 2007/05/11 09:11:30 verkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROO_CHI2_VAR
#define ROO_CHI2_VAR

#include "RooAbsOptTestStatistic.h"
#include "RooCmdArg.h"
#include "RooDataHist.h"
#include "RooAbsPdf.h"

class RooChi2Var : public RooAbsOptTestStatistic {
public:

  // Constructors, assignment etc
  RooChi2Var(const char *name, const char* title, RooAbsReal& func, RooDataHist& data,
        const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),const RooCmdArg& arg3=RooCmdArg::none(),
        const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none(),
        const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none(),const RooCmdArg& arg9=RooCmdArg::none()) ;

  RooChi2Var(const char *name, const char* title, RooAbsPdf& pdf, RooDataHist& data,
        const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),const RooCmdArg& arg3=RooCmdArg::none(),
        const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none(),
        const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none(),const RooCmdArg& arg9=RooCmdArg::none()) ;

  enum FuncMode { Function, Pdf, ExtendedPdf } ;

  RooChi2Var(const RooChi2Var& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooChi2Var(*this,newname); }

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

  ClassDefOverride(RooChi2Var,1) // Chi^2 function of p.d.f w.r.t a binned dataset
};


#endif
