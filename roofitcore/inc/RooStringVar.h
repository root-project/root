/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooStringVar.rdl,v 1.12 2001/10/19 06:56:53 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_STRING_VAR
#define ROO_STRING_VAR

#include <iostream.h>
#include "TString.h"
#include "RooFitCore/RooAbsString.hh"
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
  virtual TString getVal() const { return TString(_value) ; } // overrides RooAbsReal::getVal()
  virtual void setVal(TString value);
  virtual RooStringVar& operator=(const char* newValue);

  // We implement a fundamental type of AbsArg that can be stored in a dataset
  inline virtual Bool_t isFundamental() const { return kTRUE; }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

protected:

  ClassDef(RooStringVar,1) // String-valued variable 
};

#endif
