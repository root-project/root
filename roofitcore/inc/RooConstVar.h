/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$                                                             *
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2002, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_CONST_VAR
#define ROO_CONST_VAR

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooListProxy.hh"

class RooArgSet ;

class RooConstVar : public RooAbsReal {
public:
  // Constructors, assignment etc
  inline RooConstVar() { }
  RooConstVar(const char *name, const char *title, Double_t value);
  RooConstVar(const RooConstVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooConstVar(*this,newname); }
  virtual ~RooConstVar();

  virtual Double_t getVal(const RooArgSet* set=0) const { return _value ; }
  void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  virtual Double_t evaluate() const { return _value ; } ;

  Double_t _value ;

  ClassDef(RooConstVar,1) // Real-valued variable, calculated from a string expression formula 
};

#endif
