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
#ifndef ROO_MOMENT
#define ROO_MOMENT

#include "RooAbsMoment.h"
#include "RooRealProxy.h"
#include "RooSetProxy.h"


class RooRealVar;
class RooArgList ;

class RooMoment : public RooAbsMoment {
public:

  RooMoment() ;
  RooMoment(const char *name, const char *title, RooAbsReal& func, RooRealVar& x, Int_t order=1, bool central=false, bool takeRoot=false) ;
  RooMoment(const char *name, const char *title, RooAbsReal& func, RooRealVar& x, const RooArgSet& nset, Int_t order=1, bool central=false, bool takeRoot=false,
       bool intNSet=false) ;
  ~RooMoment() override ;

  RooMoment(const RooMoment& other, const char *name = nullptr);
  TObject* clone(const char* newname) const override { return new RooMoment(*this, newname); }

  const RooAbsReal& xF() { return _xf.arg() ; }
  const RooAbsReal& ixF() { return _ixf.arg() ; }
  const RooAbsReal& iF() { return _if.arg() ; }

protected:

  RooRealProxy _xf ;                     ///< X*F
  RooRealProxy _ixf ;                    ///< Int(X*F(X))dx ;
  RooRealProxy _if ;                     ///< Int(F(x))dx ;
  double evaluate() const override;

  ClassDefOverride(RooMoment,1) // Representation of moment in a RooAbsReal in a given RooRealVar
};

#endif
