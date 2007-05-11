/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooTreeData.rdl,v 1.39 2005/06/21 19:59:52 wverkerke Exp $
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
#ifndef ROO_TREE_DATA
#define ROO_TREE_DATA

#include "RooAbsData.h"

class TTree;
class TIterator;
class TBranch;
class TH1F;
class TH2F;
class TPaveText;
class RooAbsArg;
class RooAbsReal ;
class RooAbsCategory ;
class RooAbsString ;
class Roo1DTable ;
class RooPlot;
class RooFormulaVar ;
class RooLinkedList ;

class RooTreeData : public RooAbsData {
public:

  // Constructors, factory methods etc.
  RooTreeData() ; 
  RooTreeData(const char *name, const char *title, const RooArgSet& vars) ;
  RooTreeData(const char *name, const char *title, RooTreeData *ntuple, 
	     const RooArgSet& vars, const char *cuts);
  RooTreeData(const char *name, const char *title, RooTreeData *t, 
	     const RooArgSet& vars, const RooFormulaVar& cutVar) ;
  RooTreeData(const char *name, const char *title, TTree *t, 
	     const RooArgSet& vars, const RooFormulaVar& cutVar) ;
  RooTreeData(const char *name, const char *title, TTree *ntuple, 
	     const RooArgSet& vars, const char *cuts);
  RooTreeData(const char *name, const char *filename, const char *treename, 
	     const RooArgSet& vars, const char *cuts);  
  RooTreeData(RooTreeData const & other, const char* newname=0) ;
  virtual ~RooTreeData() ;

  // Load a given row of data
  virtual void fill() { Fill() ; } ;
  virtual const RooArgSet* get(Int_t index) const;
  virtual const RooArgSet* get() const { return &_vars ; } 

  virtual RooAbsArg* addColumn(RooAbsArg& var) ;
  virtual RooArgSet* addColumns(const RooArgList& varList) ;

  virtual Int_t numEntries(Bool_t useWeights=kFALSE) const ;
  virtual void reset() { Reset() ; }

  virtual Roo1DTable* table(const RooAbsCategory& cat, const char* cuts="", const char* opts="") const ;

  virtual RooPlot* statOn(RooPlot* frame, 
                          const RooCmdArg& arg1=RooCmdArg::none, const RooCmdArg& arg2=RooCmdArg::none, 
                          const RooCmdArg& arg3=RooCmdArg::none, const RooCmdArg& arg4=RooCmdArg::none, 
                          const RooCmdArg& arg5=RooCmdArg::none, const RooCmdArg& arg6=RooCmdArg::none, 
                          const RooCmdArg& arg7=RooCmdArg::none, const RooCmdArg& arg8=RooCmdArg::none) ;

  virtual RooPlot* statOn(RooPlot* frame, const char *what, 
			  const char *label= "", Int_t sigDigits= 2,
			  Option_t *options= "NELU", Double_t xmin=0.15, 
			  Double_t xmax= 0.65,Double_t ymax=0.85, 
                          const char* cutSpec=0, const char* cutRange=0, 
                          const RooCmdArg* formatCmd=0);

  TH1 *fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts= "", const char* cutRange=0) const;

  Double_t moment(RooRealVar &var, Double_t order, Double_t offset=0, const char* cutSpec=0, const char* cutRange=0) const ;
  RooRealVar* meanVar(RooRealVar &var, const char* cutSpec=0, const char* cutRange=0) const ;
  RooRealVar* rmsVar(RooRealVar &var, const char* cutSpec=0, const char* cutRange=0) const ;

  Bool_t getRange(RooRealVar& var, Double_t& lowest, Double_t& highest) const ;

  virtual TList* split(const RooAbsCategory& splitCat) const ;

  // Forwarded from TTree
  Int_t Scan(const char* varexp="", const char* selection="", Option_t* option="", 
		    Int_t nentries = 1000000000, Int_t firstentry = 0);
  const TTree& tree() const { return *_tree ; }


  // WVE Debug stuff
  void dump() ;

  void printToStream(ostream& os, PrintOption opt, TString indent) const ;

  using RooAbsData::plotOn ;
  virtual RooPlot* plotOn(RooPlot* frame, const RooLinkedList& cmdList) const ;

  // WVE --- This needs to be public to avoid CINT problems
  struct PlotOpt {
   PlotOpt() : cuts(""), drawOptions("P"), bins(0), etype(RooAbsData::Poisson), cutRange(0), histName(0), histInvisible(kFALSE),
              addToHistName(0),addToWgtSelf(1.),addToWgtOther(1.),xErrorSize(1),refreshFrameNorm(kFALSE) {} ;
   const char* cuts ;
   Option_t* drawOptions ;
   RooAbsBinning* bins ;
   RooAbsData::ErrorType etype ;
   const char* cutRange ;
   const char* histName ;
   Bool_t histInvisible ;
   const char* addToHistName ;
   Double_t addToWgtSelf ;
   Double_t addToWgtOther ;
   Double_t xErrorSize ;
   Bool_t refreshFrameNorm ;
  } ;
	
  // PlotOn implementation
  virtual RooPlot *plotOn(RooPlot *frame, PlotOpt o) const ;
  virtual RooPlot *plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat, PlotOpt o) const ;


protected:

  friend class RooMCStudy ;

  // Cache copy feature is not publicly accessible
  RooTreeData(const char *name, const char *title, RooTreeData *ntuple, 
	      const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange, 
	      Int_t nStart, Int_t nStop, Bool_t copyCache);


  Int_t ScanCache(const char* varexp="", const char* selection="", Option_t* option="", 
			 Int_t nentries = 1000000000, Int_t firstentry = 0);
  const TTree& cacheTree() const { return *_cacheTree ; }

  // Forwarded from TTree
  Stat_t GetEntries() const;
  void Reset(Option_t* option=0);
  Int_t Fill();
  Int_t GetEntry(Int_t entry = 0, Int_t getall = 0);
  void treePrint();

  // Constant term  optimizer interface
  virtual void cacheArgs(RooArgSet& varSet, const RooArgSet* nset=0) ;
  void setArgStatus(const RooArgSet& set, Bool_t active) ;
  virtual void resetCache() ;

  // TTree Branch buffer size contro
  void setBranchBufferSize(Int_t size) { _defTreeBufSize = size ; }
  Int_t getBranchBufferSize() const { return _defTreeBufSize ; }


  void checkInit() const {
    if (_defCtor) {
      ((RooTreeData*)this)->initialize() ;
      _defCtor = kFALSE ;    
    }
  }

  // Load data from another TTree
  void loadValues(const RooTreeData *t, RooFormulaVar* select=0, const char* rangeName=0, Int_t nStart=0, Int_t nStop=2000000000) ; 
  void loadValues(const TTree *t, RooFormulaVar* cutVar=0, const char* rangeName=0, Int_t nStart=0, Int_t nStop=2000000000) ; 
  void loadValues(const char *filename, const char *treename, RooFormulaVar *cutVar=0);

  friend class RooDataSet ;
  void createTree(const char* name, const char* title) ; 
  TTree *_tree ;           // TTree holding the data points
  TTree *_cacheTree ;      //! TTree holding the cached function values
  mutable Bool_t _defCtor ;//! Was object constructed with default ctor?

  // Column structure definition
  RooArgSet _truth;        // Truth variables   
  TString _blindString ;   // Blinding string (optionally read from ASCII files)

  static Int_t _defTreeBufSize ;  

  void initCache(const RooArgSet& cachedVars) ; 
  
private:

  void initialize();

  ClassDef(RooTreeData,1) // Abstract ttree based data collection
};


#endif
