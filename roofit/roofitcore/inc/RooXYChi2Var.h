/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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

#ifndef ROO_XY_CHI2_VAR
#define ROO_XY_CHI2_VAR

// We can't print deprecation warnings when including headers in cling, because
// this will be done automatically anyway.
#ifdef __CLING__
#ifndef ROOFIT_BUILDS_ITSELF
// These warnings should only be suppressed when building ROOT itself!
#warning "Including RooXYChi2Var.h is deprecated, and this header will be removed in ROOT v6.34: please use RooAbsReal::createChi2(RooAbsData &, ...) to create chi-square test statistics objects on X-Y data"
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
#include "RooDataSet.h"
#include "RooAbsPdf.h"
#include "RooNumIntConfig.h"
#include <list>
class RooAbsIntegrator ;


class RooXYChi2Var : public RooAbsOptTestStatistic {
public:

  // Constructors, assignment etc
  RooXYChi2Var(const char *name, const char* title, RooAbsReal& func, RooDataSet& data, bool integrate=false) ;
  RooXYChi2Var(const char *name, const char* title, RooAbsReal& func, RooDataSet& data, RooRealVar& yvar, bool integrate=false) ;
  /// \cond ROOFIT_INTERNAL
  // For internal use in RooAbsReal::createChi2().
  RooXYChi2Var(const char *name, const char *title, RooAbsReal& func, RooAbsData& data, RooRealVar *yvar, bool integrate,
               RooAbsTestStatistic::Configuration const& cfg);
  /// \endcond ROOFIT_INTERNAL

  RooXYChi2Var(const RooXYChi2Var& other, const char* name=nullptr);
  TObject* clone(const char* newname) const override { return new RooXYChi2Var(*this,newname); }

  RooAbsTestStatistic* create(const char *name, const char *title, RooAbsReal& pdf, RooAbsData& adata,
                                      const RooArgSet&, RooAbsTestStatistic::Configuration const&) override {
    // Virtual constructor
    return new RooXYChi2Var(name,title,pdf,(RooDataSet&)adata) ;
  }

  ~RooXYChi2Var() override;

  double defaultErrorLevel() const override {
    // The default error level for MINUIT error analysis for a chi^2 is 1.0
    return 1.0 ;
  }

  RooNumIntConfig& binIntegratorConfig() { return _intConfig ; }
  const RooNumIntConfig& binIntegratorConfig() const { return _intConfig ; }

protected:

  bool allowFunctionCache() override {
    // Disable function (component) caching if integration is requested as the function
    // will be evaluated at coordinates other than the points in the dataset
    return !_integrate ;
  }

  RooArgSet requiredExtraObservables() const override ;

  double fy() const ;

  bool _extended ; ///< Is the input function and extended p.d.f.
  bool _integrate ; ///< Is integration over the bin volume requested

  RooRealVar* _yvar ; ///< Y variable if so designated
  RooArgSet _rrvArgs ; ///< Set of real-valued observables

  void initialize() ;
  void initIntegrator() ;
  double xErrorContribution(double ydata) const ;

  double evaluatePartition(std::size_t firstEvent, std::size_t lastEvent, std::size_t stepSize) const override ;

  RooNumIntConfig   _intConfig ; ///< Numeric integrator configuration for integration of function over bin
  std::unique_ptr<RooAbsReal>    _funcInt; ///<! Function integral
  std::list<RooAbsBinning*> _binList ; ///<! Bin ranges

  ClassDefOverride(RooXYChi2Var,0) // Chi^2 function of p.d.f w.r.t a unbinned dataset with X and Y values

#ifndef ROOFIT_BUILDS_ITSELF
} R__DEPRECATED(6,34, "Please use RooAbsReal::createChi2(RooAbsData &, ...) to create chi-square test statistics objects on X-Y data");
#else
};
#endif


#endif
