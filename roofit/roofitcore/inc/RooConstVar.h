/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooConstVar.h,v 1.9 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_CONST_VAR
#define ROO_CONST_VAR

#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooListProxy.h"

class RooArgSet ;

class RooConstVar final : public RooAbsReal {
public:
  // Constructors, assignment etc
  RooConstVar() { }
  RooConstVar(const char *name, const char *title, Double_t value);
  RooConstVar(const RooConstVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooConstVar(*this,newname); }
  virtual ~RooConstVar();

  ////////////////////////////////////////////////////////////////////////////////
  /// Return (constant) value.
  virtual Double_t getValV(const RooArgSet*) const {
    return _value;
  }

  void writeToStream(std::ostream& os, Bool_t compact) const ;

  virtual Bool_t isDerived() const { 
    // Does value or shape of this arg depend on any other arg?
    return kFALSE ;
  }

protected:

  Double_t evaluate() const {
    return _value;
  }

  Double_t _value{0.}; // Constant value of self

  ClassDef(RooConstVar,1) // Constant RooAbsReal value object
};

#endif
