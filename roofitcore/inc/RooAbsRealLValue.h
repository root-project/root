/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#ifndef ROO_ABS_REAL_LVALUE
#define ROO_ABS_REAL_LVALUE

#include <iostream.h>
#include <math.h>
#include <float.h>
#include "TString.h"
#include "RooFitCore/RooAbsReal.hh"

#ifndef INFINITY
 #define INFINITY FLT_MAX
#endif

class RooArgSet ;

class RooAbsRealLValue : public RooAbsReal {
public:
  // Constructors, assignment etc.
  inline RooAbsRealLValue() { }
  RooAbsRealLValue(const char *name, const char *title, const char *unit= "") ;
  RooAbsRealLValue(const RooAbsRealLValue& other, const char* name=0);
  virtual ~RooAbsRealLValue();
  RooAbsRealLValue& operator=(const RooAbsRealLValue& other) ;
  
  // Parameter value and error accessors
  virtual void setVal(Double_t value)=0;
  virtual Double_t operator=(Double_t newValue);

  // Get fit range limits
  virtual Double_t getFitMin() const = 0 ;
  virtual Double_t getFitMax() const = 0 ;
  inline Bool_t hasFitMin() const { return getFitMin() != -INFINITY; }
  inline Bool_t hasFitMax() const { return getFitMax() != +INFINITY; }

  // Test a value against our fit range
  Bool_t inFitRange(Double_t value, Double_t* clippedValue=0) const;

  // Constant and Projected flags 
  inline void setConstant(Bool_t value= kTRUE) { setAttribute("Constant",value); }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;

protected:

  virtual RooAbsArg& operator=(const RooAbsArg& other) ;
  virtual void attachToTree(TTree& t, Int_t bufSize=32000) ;

  virtual Bool_t isValid(Double_t value, Bool_t verbose=kFALSE) const ;

  ClassDef(RooAbsRealLValue,1) // a real-valued variable and its value
};

#endif
