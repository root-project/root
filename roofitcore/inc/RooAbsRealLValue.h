/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsRealLValue.rdl,v 1.10 2001/08/08 23:11:24 david Exp $
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
#include "RooFitCore/RooNumber.hh"
#include "RooFitCore/RooAbsLValue.hh"

class RooArgSet ;

class RooAbsRealLValue : public RooAbsReal, public RooAbsLValue {
public:
  // Constructors, assignment etc.
  inline RooAbsRealLValue() { }
  RooAbsRealLValue(const char *name, const char *title, const char *unit= "") ;
  RooAbsRealLValue(const RooAbsRealLValue& other, const char* name=0);
  virtual ~RooAbsRealLValue();
  
  // Parameter value and error accessors
  virtual void setVal(Double_t value)=0;
  virtual RooAbsRealLValue& operator=(Double_t newValue);
  virtual RooAbsRealLValue& operator=(const RooAbsReal& other) ;
  virtual void setPlotBin(Int_t ibin) ;
  void randomize();

  // Get fit range limits
  virtual Double_t getFitMin() const = 0 ;
  virtual Double_t getFitMax() const = 0 ;
  inline Bool_t hasFitMin() const { return !RooNumber::isInfinite(getFitMin()); }
  inline Bool_t hasFitMax() const { return !RooNumber::isInfinite(getFitMax()); }

  virtual Bool_t isJacobianOK(const RooArgSet& depList) const { return kTRUE ; }
  virtual Double_t jacobian() const { return 1 ; }

  inline virtual Bool_t isLValue() const { return kTRUE; }

  // Test a value against our fit range
  Bool_t inFitRange(Double_t value, Double_t* clippedValue=0) const;
  virtual Bool_t isValidReal(Double_t value, Bool_t printError=kFALSE) const ; 

  // Constant and Projected flags 
  inline void setConstant(Bool_t value= kTRUE) { setAttribute("Constant",value); }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(ostream& stream, PrintOption opt=Standard, TString indent= "") const ;

protected:

  void copyCache(const RooAbsArg* source) ;

  ClassDef(RooAbsRealLValue,1) // Abstract modifiable real-valued variable
};

#endif
