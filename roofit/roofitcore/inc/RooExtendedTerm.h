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
#ifndef ROO_EXTENDED_TERM
#define ROO_EXTENDED_TERM

#include "RooAbsPdf.h"
#include "RooRealProxy.h"

class RooExtendedTerm : public RooAbsPdf {
public:

  RooExtendedTerm() ;
  RooExtendedTerm(const char *name, const char *title, const RooAbsReal& n) ;
  RooExtendedTerm(const RooExtendedTerm& other, const char* name=0) ;
  TObject* clone(const char* newname) const override { return new RooExtendedTerm(*this,newname) ; }
  ~RooExtendedTerm() override ;

  Double_t evaluate() const override { return 1. ; }

  ExtendMode extendMode() const override { return CanBeExtended ; }
  /// Return number of expected events, in other words the value of the associated n parameter.
  Double_t expectedEvents(const RooArgSet* nset) const override ;

protected:

  RooRealProxy _n ;          ///< Number of expected events

  ClassDefOverride(RooExtendedTerm,1) // Meta-p.d.f flat in all observables introducing only extended ML term
};

#endif
