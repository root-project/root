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

// We can't print deprecation warnings when including headers in cling, because
// this will be done automatically anyway.
#ifdef __CLING__
#ifndef ROOFIT_BUILDS_ITSELF
// These warnings should only be suppressed when building ROOT itself!
#warning "Including RooChi2Var.h is deprecated, and this header will be removed in ROOT v6.34: Please use RooAbsReal::createChi2() to create chi-square test statistics objects"
#else
// If we are builting RooFit itself, this will serve as a reminder to actually
// remove this deprecate public header. Here is now this needs to be done:
//    1. Move this header file from inc/ to src/
//    2. Remove the LinkDef entry, ClassDefOverride, and ClassImpl macros for
//       this class
//    3. If there are are tests using this class in the test/ directory, change
//       the include to use a relative path the moved header file in the src/
//       directory, e.g. #include <RemovedInterface.h> becomes #include
//       "../src/RemovedInterface.h"
//    4. Remove this ifndef-else-endif block from the header
//    5. Remove the deprecation warning at the end of the class declaration
#include <RVersion.h>
#if ROOT_VERSION_CODE >= ROOT_VERSION(6, 34, 00)
#error "Please remove this deprecated public interface."
#endif
#endif
#endif

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

  ClassDefOverride(RooChi2Var,0) // Chi^2 function of p.d.f w.r.t a binned dataset

#ifndef ROOFIT_BUILDS_ITSELF
} R__DEPRECATED(6,34, "Please use RooAbsReal::createChi2() to create chi-square test statistics objects.");
#else
};
#endif


#endif
