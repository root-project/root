/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooTreeData.rdl,v 1.29 2003/10/06 18:32:59 wverkerke Exp $
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2004, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_TREE_DATA
#define ROO_TREE_DATA

#include "TTree.h"
#include "RooFitCore/RooAbsData.hh"

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

  virtual Int_t numEntries(Bool_t useWeights=kFALSE) const { return (Int_t)GetEntries() ; }
  virtual void reset() { Reset() ; }

  virtual Roo1DTable* table(const RooAbsCategory& cat, const char* cuts="", const char* opts="") const ;

  virtual RooPlot* plotOn(RooPlot* frame, 
			  const RooCmdArg& arg1            , const RooCmdArg& arg2=RooCmdArg(),
			  const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),
			  const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),
			  const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) const ;

  virtual RooPlot *plotOn(RooPlot *frame, const char* cuts="", Option_t* drawOptions="P", 
			  const RooAbsBinning* bins=0, RooAbsData::ErrorType=RooAbsData::Poisson) const;
  virtual RooPlot *plotOn(RooPlot *frame, const RooFormulaVar* cutVar, Option_t* drawOptions="P", 
			  const RooAbsBinning* bins=0, RooAbsData::ErrorType=RooAbsData::Poisson) const;
  virtual RooPlot *plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat, 
			      const char* cut="", Option_t* drawOptions="P", const RooAbsBinning* bins=0) const ;
  virtual RooPlot* statOn(RooPlot* frame, const char *what, 
			  const char *label= "", Int_t sigDigits= 2,
			  Option_t *options= "NELU", Double_t xmin=0.15, 
			  Double_t xmax= 0.65,Double_t ymax=0.85);
  virtual RooPlot* statOn(RooPlot* frame, RooRealVar &var,
			  const char *label= "", Int_t sigDigits= 2,
			  Option_t *options= "NELU", Double_t xmin=0.15, 
			  Double_t xmax= 0.65,Double_t ymax=0.85) { 
    return statOn(frame,"NMR",label,sigDigits,options,xmin,xmax,ymax) ;
  }

  TH1 *fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts= "") const;

  Double_t moment(RooRealVar &var, Double_t order, Double_t offset=0) const ;
  RooRealVar* meanVar(RooRealVar &var) const ;
  RooRealVar* rmsVar(RooRealVar &var) const ;

  virtual TList* split(const RooAbsCategory& splitCat) const ;

  // Forwarded from TTree
  inline Int_t Scan(const char* varexp="", const char* selection="", Option_t* option="", 
		    Int_t nentries = 1000000000, Int_t firstentry = 0) {
    return _tree->Scan(varexp,selection,option,nentries,firstentry) ; 
  }
  const TTree& tree() const { return *_tree ; }


  // WVE Debug stuff
  void dump() ;

  void printToStream(ostream& os, PrintOption opt, TString indent) const ;

  // Cache copy feature is not publicly accessible
  RooTreeData(const char *name, const char *title, RooTreeData *ntuple, 
	     const RooArgSet& vars, const RooFormulaVar* cutVar, Bool_t copyCache);

  //protected:

  inline Int_t ScanCache(const char* varexp="", const char* selection="", Option_t* option="", 
			 Int_t nentries = 1000000000, Int_t firstentry = 0) {
    return _cacheTree->Scan(varexp,selection,option,nentries,firstentry) ; 
  }
  const TTree& cacheTree() const { return *_cacheTree ; }

  // Forwarded from TTree
  inline Stat_t GetEntries() const { return _tree->GetEntries() ; }
  inline void Reset(Option_t* option=0) { _tree->Reset(option) ; }
  inline Int_t Fill() { return _tree->Fill() ; }
  inline Int_t GetEntry(Int_t entry = 0, Int_t getall = 0) { 
    Int_t ret1 = _tree->GetEntry(entry,getall) ; 
    if (!ret1) return 0 ;
    _cacheTree->GetEntry(entry,getall) ; 
    return ret1 ;
  }
  void treePrint() { _tree->Print() ; }

  // Constant term  optimizer interface
  virtual void cacheArgs(RooArgSet& varSet, const RooArgSet* nset=0) ;
  void setArgStatus(const RooArgSet& set, Bool_t active) ;
  virtual void resetCache() ;

  // TTree Branch buffer size contro
  void setBranchBufferSize(Int_t size) { _defTreeBufSize = size ; }
  Int_t getBranchBufferSize() const { return _defTreeBufSize ; }

protected:

  void checkInit() const {
    if (_defCtor) {
      ((RooTreeData*)this)->initialize() ;
      _defCtor = kFALSE ;    
    }
  }

  // Load data from another TTree
  void loadValues(const RooTreeData *t, RooFormulaVar* select=0) ;
  void loadValues(const TTree *t, RooFormulaVar* cutVar=0) ; 
  void loadValues(const char *filename, const char *treename,
		  RooFormulaVar *cutVar=0);


  // PlotOn with command list
  virtual RooPlot* plotOn(RooPlot* frame, RooLinkedList& cmdList) const ;

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

RooCmdArg Cut(const char* cutSpec) ;
RooCmdArg Cut(const RooAbsReal& cutVar) ;
RooCmdArg Binning(const RooAbsBinning& binning) ;
RooCmdArg MarkerStyle(Style_t style) ;
RooCmdArg MarkerSize(Size_t size) ;
RooCmdArg MarkerColor(Color_t color) ;
//RooCmdArg XErrorSize(Double_t width) ;


#endif
