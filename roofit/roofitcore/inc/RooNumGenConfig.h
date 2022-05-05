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
#ifndef ROO_NUM_GEN_CONFIG
#define ROO_NUM_GEN_CONFIG

#include "TObject.h"
#include "RooCategory.h"
#include "RooLinkedList.h"
class RooNumGenFactory ;
class RooAbsNumGenerator ;

class RooNumGenConfig : public TObject, public RooPrintable {
public:

  RooNumGenConfig();
  RooNumGenConfig(const RooNumGenConfig& other) ;
  RooNumGenConfig& operator=(const RooNumGenConfig& other) ;
  ~RooNumGenConfig() override;

  // Return selected integration techniques for 1,2,N dimensional integrals
  RooCategory& method1D(bool cond, bool cat) ;
  RooCategory& method2D(bool cond, bool cat) ;
  RooCategory& methodND(bool cond, bool cat) ;
  const RooCategory& method1D(bool cond, bool cat) const ;
  const RooCategory& method2D(bool cond, bool cat) const ;
  const RooCategory& methodND(bool cond, bool cat) const ;


  static RooNumGenConfig& defaultConfig() ;

  bool addConfigSection(const RooAbsNumGenerator* proto, const RooArgSet& defaultConfig) ;
  const RooArgSet& getConfigSection(const char* name) const ;
  RooArgSet& getConfigSection(const char* name) ;

  void printMultiline(std::ostream &os, Int_t content, bool verbose, TString indent= "") const override;

  inline void Print(Option_t *options= 0) const override {
    printStream(defaultPrintStream(),defaultPrintContents(options),defaultPrintStyle(options));
  }
  StyleOption defaultPrintStyle(Option_t* opt) const override ;


protected:

  RooCategory _method1D        ; ///< Selects integration method for 1D p.d.f.s
  RooCategory _method1DCat     ; ///< Selects integration method for 1D  p.d.f.s with categories
  RooCategory _method1DCond    ; ///< Selects integration method for 1D conditional p.d.f.s
  RooCategory _method1DCondCat ; ///< Selects integration method for 1D conditional p.d.f.s with categories

  RooCategory _method2D        ; ///< Selects integration method for 2D p.d.f.s
  RooCategory _method2DCat     ; ///< Selects integration method for 2D  p.d.f.s with categories
  RooCategory _method2DCond    ; ///< Selects integration method for 2D conditional p.d.f.s
  RooCategory _method2DCondCat ; ///< Selects integration method for 2D conditional p.d.f.s with categories

  RooCategory _methodND        ; ///< Selects integration method for ND p.d.f.s
  RooCategory _methodNDCat     ; ///< Selects integration method for ND  p.d.f.s with categories
  RooCategory _methodNDCond    ; ///< Selects integration method for ND conditional p.d.f.s
  RooCategory _methodNDCondCat ; ///< Selects integration method for ND conditional p.d.f.s with categories

  RooLinkedList _configSets ; ///< List of configuration sets for individual integration methods

  ClassDefOverride(RooNumGenConfig,1) // Numeric (MC) Event generator configuration
};

#endif


