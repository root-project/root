/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsRealLValue.rdl,v 1.30 2005/02/24 22:36:06 wverkerke Exp $
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
#ifndef ROO_ABS_REAL_LVALUE
#define ROO_ABS_REAL_LVALUE

#include <iostream>
#include <math.h>
#include <float.h>
#include "TString.h"

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooNumber.hh"
#include "RooFitCore/RooAbsLValue.hh"
#include "RooFitCore/RooAbsBinning.hh"

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
  virtual void randomize();

  // Implementation of RooAbsLValue
  virtual void setBin(Int_t ibin) ;
  virtual Int_t getBin() const { return getBinning().binNumber(getVal()) ; }
  virtual Int_t numBins() const { return getBins() ; }
  virtual Double_t getBinWidth(Int_t i) const { return getBinning().binWidth(i) ; }
  
  // Get fit range limits
  virtual const RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE) const = 0 ;
  virtual RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE) = 0 ;
  virtual Bool_t hasBinning(const char* name) const = 0 ;
  virtual Double_t getMin(const char* name=0) const { return getBinning(name).lowBound() ; }
  virtual Double_t getMax(const char* name=0) const { return getBinning(name).highBound() ; }
  virtual Int_t getBins(const char* name=0) const { return getBinning(name).numBins() ; }
  inline Bool_t hasMin(const char* name=0) const { return !RooNumber::isInfinite(getMin(name)); }
  inline Bool_t hasMax(const char* name=0) const { return !RooNumber::isInfinite(getMax(name)); }
  virtual Bool_t inRange(const char* name) const ;

  virtual Bool_t isJacobianOK(const RooArgSet& depList) const { return kTRUE ; }
  virtual Double_t jacobian() const { return 1 ; }

  inline virtual Bool_t isLValue() const { return kTRUE; }

  // Test a value against our fit range
  Bool_t inRange(Double_t value, Double_t* clippedValue=0) const;
  virtual Bool_t isValidReal(Double_t value, Bool_t printError=kFALSE) const ; 

  // Constant and Projected flags 
  inline void setConstant(Bool_t value= kTRUE) { setAttribute("Constant",value); }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(std::istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(std::ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printToStream(std::ostream& stream, PrintOption opt=Standard, TString indent= "") const ;
  
  // Build 1-dimensional plots
  RooPlot* frame(const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none, 
                 const RooCmdArg& arg3=RooCmdArg::none, const RooCmdArg& arg4=RooCmdArg::none, const RooCmdArg& arg5=RooCmdArg::none, 
                 const RooCmdArg& arg6=RooCmdArg::none, const RooCmdArg& arg7=RooCmdArg::none, const RooCmdArg& arg8=RooCmdArg::none) const ;
  RooPlot *frame(const RooLinkedList& cmdList) const ;
  RooPlot *frame(Double_t lo, Double_t hi, Int_t nbins) const;
  RooPlot *frame(Double_t lo, Double_t hi) const;
  RooPlot *frame(Int_t nbins) const;
  RooPlot *frame() const;

  // Create empty 1,2, and 3D histograms from a list of 1-3 RooAbsReals
  TH1 *createHistogram(const char *name, 
                       const RooCmdArg& arg1=RooCmdArg::none, const RooCmdArg& arg2=RooCmdArg::none, 
                       const RooCmdArg& arg3=RooCmdArg::none, const RooCmdArg& arg4=RooCmdArg::none, 
                       const RooCmdArg& arg5=RooCmdArg::none, const RooCmdArg& arg6=RooCmdArg::none, 
                       const RooCmdArg& arg7=RooCmdArg::none, const RooCmdArg& arg8=RooCmdArg::none) const ;
  TH1 *createHistogram(const char *name, const RooLinkedList& cmdList) const ;
 
  TH1F *createHistogram(const char *name, const char *yAxisLabel) const ;
  TH1F *createHistogram(const char *name, const char *yAxisLabel, Double_t xlo, Double_t xhi, Int_t nBins) const ;
  TH1F *createHistogram(const char *name, const char *yAxisLabel, const RooAbsBinning& bins) const ;

  TH2F *createHistogram(const char *name, const RooAbsRealLValue &yvar, const char *zAxisLabel=0, 
			Double_t* xlo=0, Double_t* xhi=0, Int_t* nBins=0) const ;
  TH2F *createHistogram(const char *name, const RooAbsRealLValue &yvar, const char *zAxisLabel, const RooAbsBinning** bins) const ;
  

  TH3F *createHistogram(const char *name, const RooAbsRealLValue &yvar, const RooAbsRealLValue &zvar,
			const char *tAxisLabel, Double_t* xlo=0, Double_t* xhi=0, Int_t* nBins=0) const ;
  TH3F *createHistogram(const char *name, const RooAbsRealLValue &yvar, const RooAbsRealLValue &zvar, const char* tAxisLabel, const RooAbsBinning** bins) const ;
  
  static TH1* createHistogram(const char *name, RooArgList &vars, const char *tAxisLabel, Double_t* xlo, Double_t* xhi, Int_t* nBins) ;
  static TH1* createHistogram(const char *name, RooArgList &vars, const char *tAxisLabel, const RooAbsBinning** bins) ;

protected:

  Bool_t fitRangeOKForPlotting() const ;

  void copyCache(const RooAbsArg* source) ;

  ClassDef(RooAbsRealLValue,1) // Abstract modifiable real-valued variable
};

#endif
