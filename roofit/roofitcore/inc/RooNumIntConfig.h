/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooNumIntConfig.h,v 1.8 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_NUM_INT_CONFIG
#define ROO_NUM_INT_CONFIG

#include "TObject.h"
#include "RooCategory.h"
#include "RooLinkedList.h"
class RooNumIntFactory ;
class RooAbsIntegrator ;

class RooNumIntConfig : public TObject, public RooPrintable {
public:

  RooNumIntConfig();
  RooNumIntConfig(const RooNumIntConfig& other) ;
  RooNumIntConfig& operator=(const RooNumIntConfig& other) ;
  ~RooNumIntConfig() override;

  // Return selected integration techniques for 1,2,N dimensional integrals
  RooCategory& method1D() { return _method1D ; }
  RooCategory& method2D() { return _method2D ; }
  RooCategory& methodND() { return _methodND ; }
  const RooCategory& method1D() const { return _method1D ; }
  const RooCategory& method2D() const { return _method2D ; }
  const RooCategory& methodND() const { return _methodND ; }

  // Return selected integration techniques for 1,2,N dimensional open-ended integrals
  RooCategory& method1DOpen() { return _method1DOpen ; }
  RooCategory& method2DOpen() { return _method2DOpen ; }
  RooCategory& methodNDOpen() { return _methodNDOpen ; }
  const RooCategory& method1DOpen() const { return _method1DOpen ; }
  const RooCategory& method2DOpen() const { return _method2DOpen ; }
  const RooCategory& methodNDOpen() const { return _methodNDOpen ; }

  // Set/get absolute and relative precision convergence criteria
  Double_t epsAbs() const { return _epsAbs ; }
  Double_t epsRel() const { return _epsRel ; }
  void setEpsAbs(Double_t newEpsAbs) ;
  void setEpsRel(Double_t newEpsRel) ;

  // Set/get switch that activates printing of number of required
  // function evaluations for each numeric integration
  bool printEvalCounter() const { return _printEvalCounter ; }
  void setPrintEvalCounter(bool newVal) { _printEvalCounter = newVal ; }

  static RooNumIntConfig& defaultConfig() ;

  bool addConfigSection(const RooAbsIntegrator* proto, const RooArgSet& defaultConfig) ;
  const RooArgSet& getConfigSection(const char* name) const ;
  RooArgSet& getConfigSection(const char* name) ;

  void printMultiline(std::ostream &os, Int_t content, bool verbose, TString indent= "") const override;

  StyleOption defaultPrintStyle(Option_t* opt) const override ;
  inline void Print(Option_t *options= 0) const override {
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }

protected:
  Double_t _epsAbs          ; ///< Absolute precision
  Double_t _epsRel          ; ///< Relative precision
  bool   _printEvalCounter ; ///< Flag to control printing of function evaluation counter

  RooCategory _method1D     ; ///< Selects integration method for 1D integrals
  RooCategory _method2D     ; ///< Selects integration method for 2D integrals
  RooCategory _methodND     ; ///< Selects integration method for ND integrals
  RooCategory _method1DOpen ; ///< Selects integration method for open ended 1D integrals
  RooCategory _method2DOpen ; ///< Selects integration method for open ended 2D integrals
  RooCategory _methodNDOpen ; ///< Selects integration method for open ended ND integrals
  RooLinkedList _configSets ; ///< List of configuration sets for individual integration methods

  ClassDefOverride(RooNumIntConfig,1) // Numeric Integrator configuration
};

#endif


