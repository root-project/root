/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooStringVar.rdl,v 1.2 2001/03/29 01:59:09 verkerke Exp $
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
  RooStringVar(const char *name, const char *title, const char* value) ; 
  RooStringVar(const RooStringVar& other);
  RooStringVar(const char* name, const RooStringVar& other);
  virtual ~RooStringVar();
  RooStringVar& operator=(const RooStringVar& other) ;
  
  // Parameter value and error accessors
  virtual operator TString() ;
  virtual TString getVal() const { return TString(_value) ; } // overrides RooAbsReal::getVal()
  virtual void setVal(TString value);
  virtual TString operator=(TString newValue);

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard) const ;

protected:

  virtual RooAbsArg& operator=(const RooAbsArg& other) ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;

  virtual Bool_t isValid() const ;
  virtual Bool_t isValid(TString value, Bool_t verbose=kFALSE) const ;

  ClassDef(RooStringVar,1) // a real-valued variable and its value
};

#endif
