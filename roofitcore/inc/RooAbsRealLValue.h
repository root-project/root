/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsRealLValue.rdl,v 1.15 2001/11/20 03:53:06 verkerke Exp $
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
  virtual RooAbsArg& operator=(const RooAbsReal& other) ;
  virtual RooAbsArg& operator=(Double_t newValue);
  virtual RooAbsArg& operator=(Int_t ival) { return operator=((Double_t)ival) ; }
  virtual RooAbsArg& operator=(const char* cval) { return RooAbsArg::operator=(cval) ; }
  void randomize();

  // Binned fit interface
  virtual void setFitBin(Int_t ibin) ;
  virtual Int_t getFitBin() const ;
  virtual Int_t numFitBins() const { return getFitBins() ; }
  virtual Double_t getFitBinWidth() const { return fitBinWidth() ; }
  virtual RooAbsBinIter* createFitBinIterator() const ;

  virtual Double_t fitBinCenter(Int_t i) const ;
  virtual Double_t fitBinLow(Int_t i) const ;
  virtual Double_t fitBinHigh(Int_t i) const ;
  virtual Double_t fitBinWidth() const ;

  // Get fit range limits
  virtual Double_t getFitMin() const = 0 ;
  virtual Double_t getFitMax() const = 0 ;
  virtual Int_t getFitBins() const = 0 ;
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
  
  // Build 1-dimensional plots
  RooPlot *frame(Double_t lo, Double_t hi, Int_t nbins) const;
  RooPlot *frame(Double_t lo, Double_t hi) const;
  RooPlot *frame(Int_t nbins) const;
  RooPlot *frame() const;

  // Create empty 1,2, and 3D histograms from a list of 1-3 RooAbsReals
  TH1F *createHistogram(const char *name, const char *yAxisLabel= 0) const;
  TH2F *createHistogram(const char *name, const RooAbsReal &yvar, const char *zAxisLabel= 0) const;
  TH3F *createHistogram(const char *name, const RooAbsReal &yvar, const RooAbsReal &zvar,
			const char *tAxisLabel= 0) const;
  static TH1* createHistogram(const char *name, RooArgList &vars, const char *tAxisLabel= 0);

protected:

  void copyCache(const RooAbsArg* source) ;

  ClassDef(RooAbsRealLValue,1) // Abstract modifiable real-valued variable
};

#endif
