/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTreeData.rdl,v 1.13 2001/11/22 01:07:11 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 * History:
 *   01-Nov-1999 DK Created initial version
 *   30-Nov-2000 WV Add support for multiple file reading with optional common path
 *   09-Mar-2001 WV Migrate from RooFitTools and adapt to RooFitCore
 *   24-Aug-2001 AB Added TH2F * createHistogram method
 *
 * Copyright (C) 1999 Stanford University
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
class RooFitContext ;
class RooFormulaVar ;

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

  RooAbsArg* addColumn(RooAbsArg& var) ;
  RooArgSet* addColumns(const RooArgList& varList) ;

  virtual Int_t numEntries(Bool_t useWeights=kFALSE) const { return (Int_t)GetEntries() ; }
  virtual void reset() { Reset() ; }

  virtual Roo1DTable* table(const RooAbsCategory& cat, const char* cuts="", const char* opts="") const ;
  virtual RooPlot *plotOn(RooPlot *frame, const char* cuts="", Option_t* drawOptions="P") const;
  virtual RooPlot *plotOn(RooPlot *frame, const RooFormulaVar* cutVar, Option_t* drawOptions="P") const;
  virtual RooPlot *plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat, 
			      const char* cut="", Option_t* drawOptions="P") const ;
  virtual RooPlot* statOn(RooPlot* frame, RooRealVar &var,
			  const char *label= "", Int_t sigDigits= 2,
			  Option_t *options= "NELU", Double_t xmin=0.15, 
			  Double_t xmax= 0.65,Double_t ymax=0.85);


  TH1 *fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts= "") const;

  Double_t moment(RooRealVar &var, Double_t order, Double_t offset=0) const ;
  RooRealVar* meanVar(RooRealVar &var) const ;
  RooRealVar* rmsVar(RooRealVar &var) const ;

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

protected:

  // Forwarded from TTree
  inline Stat_t GetEntries() const { return _tree->GetEntries() ; }
  inline void Reset(Option_t* option=0) { _tree->Reset(option) ; }
  inline Int_t Fill() { return _tree->Fill() ; }
  inline Int_t GetEntry(Int_t entry = 0, Int_t getall = 0) { return _tree->GetEntry(entry,getall) ; }
  void treePrint() { _tree->Print() ; }

  // RooFitContext optimizer interface
  friend class RooFitContext ;
  virtual void cacheArgs(RooArgSet& varSet, const RooArgSet* nset=0) ;

protected:

  // Load data from another TTree
  void loadValues(const RooTreeData *t, RooFormulaVar* select=0) ;
  void loadValues(const TTree *t, RooFormulaVar* cutVar=0) ; 
  void loadValues(const char *filename, const char *treename,
		  RooFormulaVar *cutVar=0);


  void createTree(const char* name, const char* title) ; 
  TTree *_tree ;           // TTree holding the data points
  mutable Bool_t _defCtor ;//! Was object constructed with default ctor?

  // Column structure definition
  RooArgSet _truth;        // Truth variables   
  TString _blindString ;   // Blinding string (optionally read from ASCII files)

private:

  void initialize(const RooArgSet& vars);
  void initCache(const RooArgSet& cachedVars) ; 

  enum { bufSize = 8192 };
  ClassDef(RooTreeData,1) // Abstract ttree based data collection
};

#endif
