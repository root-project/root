/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooStringVar.h,v 1.23 2007/05/11 09:11:30 verkerke Exp $
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
#ifndef ROO_STRING_VAR
#define ROO_STRING_VAR

#include "Riosfwd.h"
#include "TString.h"
#include "RooAbsString.h"
class RooArgSet ;

class RooStringVar : public RooAbsString {
public:
  // Constructors, assignment etc.
  inline RooStringVar() { }
  RooStringVar(const char *name, const char *title, const char* value, Int_t size=1024) ; 
  RooStringVar(const RooStringVar& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooStringVar(*this,newname); }
  virtual ~RooStringVar();
  
  // Parameter value and error accessors
  virtual operator TString() ;
  virtual const char* getVal() const { return _value ; } // overrides RooAbsReal::getVal()
  virtual void setVal(const char* newVal) ;
  virtual RooAbsArg& operator=(const char* newValue);

  // We implement a fundamental type of AbsArg that can be stored in a dataset
  inline virtual Bool_t isFundamental() const { return kTRUE; }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  ClassDef(RooStringVar,1) // String-valued variable 
};

#endif
