/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsRealLValue.h,v 1.37 2007/07/13 21:50:24 wouter Exp $
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

#include "Riosfwd.h"
#include <math.h>
#include <float.h>
#include "TString.h"

#include "RooAbsReal.h"
#include "RooNumber.h"
#include "RooAbsLValue.h"
#include "RooAbsBinning.h"

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

  // Implementation of RooAbsLValue
  virtual void setBin(Int_t ibin, const char* rangeName=0) ;
  virtual Int_t getBin(const char* rangeName=0) const { return getBinning(rangeName).binNumber(getVal()) ; }
  virtual Int_t numBins(const char* rangeName=0) const { return getBins(rangeName) ; }
  virtual Double_t getBinWidth(Int_t i, const char* rangeName=0) const { return getBinning(rangeName).binWidth(i) ; }
  virtual Double_t volume(const char* rangeName) const { return getMax(rangeName)-getMin(rangeName) ; }
  virtual void randomize(const char* rangeName=0);

  virtual const RooAbsBinning* getBinningPtr(const char* rangeName) const { return &getBinning(rangeName) ; }
  virtual Int_t getBin(const RooAbsBinning* ptr) const { return ptr->binNumber(getVal()) ; }

  virtual void setBin(Int_t ibin, const RooAbsBinning& binning) ;
  virtual Int_t getBin(const RooAbsBinning& binning) const { return binning.binNumber(getVal()) ; }
  virtual Int_t numBins(const RooAbsBinning& binning) const { return binning.numBins() ; }
  virtual Double_t getBinWidth(Int_t i, const RooAbsBinning& binning) const { return binning.binWidth(i) ; }
  virtual Double_t volume(const RooAbsBinning& binning) const { return binning.highBound() - binning.lowBound() ; }
  virtual void randomize(const RooAbsBinning& binning) ;


  virtual void setBinFast(Int_t ibin, const RooAbsBinning& binning) ;
  
  // Get fit range limits

  virtual const RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE, Bool_t createOnTheFly=kFALSE) const = 0 ;
  virtual RooAbsBinning& getBinning(const char* name=0, Bool_t verbose=kTRUE, Bool_t createOnTheFly=kFALSE) = 0 ;
  virtual Bool_t hasBinning(const char* name) const = 0 ;
  virtual Bool_t inRange(const char* name) const ;
  virtual Int_t getBins(const char* name=0) const { return getBinning(name).numBins() ; }
  virtual Double_t getMin(const char* name=0) const { return getBinning(name).lowBound() ; }
  virtual Double_t getMax(const char* name=0) const { return getBinning(name).highBound() ; }
  inline Bool_t hasMin(const char* name=0) const { return !RooNumber::isInfinite(getMin(name)); }
  inline Bool_t hasMax(const char* name=0) const { return !RooNumber::isInfinite(getMax(name)); }
  virtual Bool_t hasRange(const char* name) const { return hasBinning(name) ; }

  // Jacobian term management
  virtual Bool_t isJacobianOK(const RooArgSet& depList) const ;
  virtual Double_t jacobian() const { return 1 ; }

  inline virtual Bool_t isLValue() const { return kTRUE; }

  // Test a value against our fit range
  Bool_t inRange(Double_t value, const char* rangeName, Double_t* clippedValue=0) const;
  virtual Bool_t isValidReal(Double_t value, Bool_t printError=kFALSE) const ; 

  // Constant and Projected flags 
  inline void setConstant(Bool_t value= kTRUE) { setAttribute("Constant",value); setValueDirty() ; setShapeDirty() ; }

  // I/O streaming interface (machine readable)
  virtual Bool_t readFromStream(std::istream& is, Bool_t compact, Bool_t verbose=kFALSE) ;
  virtual void writeToStream(std::ostream& os, Bool_t compact) const ;

  // Printing interface (human readable)
  virtual void printMultiline(std::ostream& os, Int_t contents, Bool_t verbose=kFALSE, TString indent="") const ;

  
  // Build 1-dimensional plots
  RooPlot* frame(const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(), 
                 const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(), const RooCmdArg& arg5=RooCmdArg::none(), 
                 const RooCmdArg& arg6=RooCmdArg::none(), const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;
  RooPlot *frame(const RooLinkedList& cmdList) const ;
  RooPlot *frame(Double_t lo, Double_t hi, Int_t nbins) const;
  RooPlot *frame(Double_t lo, Double_t hi) const;
  RooPlot *frame(Int_t nbins) const;
  RooPlot *frame() const;

  // Create empty 1,2, and 3D histograms from a list of 1-3 RooAbsReals
  TH1 *createHistogram(const char *name, 
                       const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(), 
                       const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(), 
                       const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none(), 
                       const RooCmdArg& arg7=RooCmdArg::none(), const RooCmdArg& arg8=RooCmdArg::none()) const ;
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

  friend class RooRealBinding ;

  virtual void setValFast(Double_t value) { setVal(value) ; }

  virtual void setVal(Double_t value, const char* /*rangeName*/) { 
    // Set object value to 'value'
    return setVal(value) ; 
  }
  Bool_t fitRangeOKForPlotting() const ;
  void copyCache(const RooAbsArg* source, Bool_t valueOnly=kFALSE, Bool_t setValDirty=kTRUE) ;

  ClassDef(RooAbsRealLValue,1) // Abstract modifiable real-valued object
};

#endif
