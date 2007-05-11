/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsData.rdl,v 1.31 2005/12/08 15:26:16 wverkerke Exp $
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
#ifndef ROO_ABS_DATA
#define ROO_ABS_DATA

#include "TNamed.h"
#include "RooPrintable.h"
#include "RooArgSet.h"
#include "RooFormulaVar.h"

class RooAbsArg;
class RooAbsReal ;
class RooAbsCategory ;
class Roo1DTable ;
class RooPlot;
class RooArgList;
class TH1;
class RooAbsBinning ;

class RooAbsData : public TNamed, public RooPrintable {
public:

  // Constructors, factory methods etc.
  RooAbsData() ; 
  RooAbsData(const char *name, const char *title, const RooArgSet& vars) ;
  RooAbsData(const RooAbsData& other, const char* newname = 0) ;
  virtual ~RooAbsData() ;
  virtual RooAbsData* emptyClone(const char* newName=0, const char* newTitle=0, const RooArgSet* vars=0) const = 0 ;

  // Reduction methods
  RooAbsData* reduce(RooCmdArg arg1,RooCmdArg arg2=RooCmdArg(),RooCmdArg arg3=RooCmdArg(),RooCmdArg arg4=RooCmdArg(),
                     RooCmdArg arg5=RooCmdArg(),RooCmdArg arg6=RooCmdArg(),RooCmdArg arg7=RooCmdArg(),RooCmdArg arg8=RooCmdArg()) ;
  RooAbsData* reduce(const char* cut) ;
  RooAbsData* reduce(const RooFormulaVar& cutVar) ;
  RooAbsData* reduce(const RooArgSet& varSubset, const char* cut=0) ;
  RooAbsData* reduce(const RooArgSet& varSubset, const RooFormulaVar& cutVar) ;

  // Add one ore more rows of data
  virtual void add(const RooArgSet& row, Double_t weight=1) = 0 ;
  virtual void fill() = 0 ;

  // Load a given row of data
  virtual inline const RooArgSet* get() const { return &_vars ; } // last loaded row
  virtual Double_t weight() const = 0 ; 
  enum ErrorType { Poisson, SumW2 } ;
  virtual Double_t weightError(ErrorType etype=Poisson) const ;
  virtual void weightError(Double_t& lo, Double_t& hi, ErrorType etype=Poisson) const ; 
  virtual const RooArgSet* get(Int_t index) const = 0 ;

  virtual Int_t numEntries(Bool_t useWeights=kFALSE) const = 0 ;
  virtual Double_t sumEntries(const char* cutSpec=0, const char* cutRange=0) const = 0 ;
  virtual Bool_t isWeighted() const { return kFALSE ; }
  virtual void reset() = 0 ;

  // Plot the distribution of a real valued arg
  virtual Roo1DTable* table(const RooAbsCategory& cat, const char* cuts="", const char* opts="") const = 0;
  virtual RooPlot* plotOn(RooPlot* frame, 
			  const RooCmdArg& arg1=RooCmdArg::none, const RooCmdArg& arg2=RooCmdArg::none,
			  const RooCmdArg& arg3=RooCmdArg::none, const RooCmdArg& arg4=RooCmdArg::none,
			  const RooCmdArg& arg5=RooCmdArg::none, const RooCmdArg& arg6=RooCmdArg::none,
			  const RooCmdArg& arg7=RooCmdArg::none, const RooCmdArg& arg8=RooCmdArg::none) const ;
  virtual RooPlot* plotOn(RooPlot* frame, const RooLinkedList& cmdList) const = 0 ;

  // Split a dataset by a category
  virtual TList* split(const RooAbsCategory& splitCat) const = 0 ;
 

  // Create 1,2, and 3D histograms from and fill it
  TH1 *createHistogram(const char *name, const RooAbsRealLValue& xvar,
                       const RooCmdArg& arg1=RooCmdArg::none, const RooCmdArg& arg2=RooCmdArg::none, 
                       const RooCmdArg& arg3=RooCmdArg::none, const RooCmdArg& arg4=RooCmdArg::none, 
                       const RooCmdArg& arg5=RooCmdArg::none, const RooCmdArg& arg6=RooCmdArg::none, 
                       const RooCmdArg& arg7=RooCmdArg::none, const RooCmdArg& arg8=RooCmdArg::none) const ;

  // Fill an existing histogram
  virtual TH1 *fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts= "", const char* cutRange=0) const = 0;

  // Printing interface (human readable)
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }

protected:


  // Constant term optimizer interface
  friend class RooAbsReal ;
  friend class RooAbsOptGoodnessOfFit ;

  virtual RooAbsData* cacheClone(const RooArgSet* newCacheVars, const char* newName=0) = 0 ;
  virtual void cacheArgs(RooArgSet& varSet, const RooArgSet* nset=0) = 0 ;
  virtual void resetCache() = 0 ;
  virtual void setArgStatus(const RooArgSet& set, Bool_t active) = 0 ;
  void setDirtyProp(Bool_t flag) { _doDirtyProp = flag ; }

  virtual RooAbsData* reduceEng(const RooArgSet& varSubset, const RooFormulaVar* cutVar, const char* cutRange=0, 
	                        Int_t nStart=0, Int_t nStop=2000000000, Bool_t copyCache=kTRUE) = 0 ;

  // Column structure definition
  RooArgSet _vars;         // Dimensions of this data set
  RooArgSet _cachedVars ;  //! External variables cached with this data set

  TIterator *_iterator;    //! Iterator over dimension variables
  TIterator *_cacheIter ;  //! Iterator over cached variables
  Bool_t _doDirtyProp ;    // Switch do (de)activate dirty state propagation when loading a data point

private:

  ClassDef(RooAbsData,1) // Abstract data collection
};

#endif
