/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataSet.rdl,v 1.29 2001/09/08 01:49:40 verkerke Exp $
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
#ifndef ROO_DATA_SET
#define ROO_DATA_SET

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

class RooDataSet : public RooAbsData {
public:

  // Constructors, factory methods etc.
  RooDataSet() ; 
  RooDataSet(const char *name, const char *title, const RooArgSet& vars) ;
  RooDataSet(const char *name, const char *title, RooDataSet *ntuple, 
	     const RooArgSet& vars, const char *cuts);
  RooDataSet(const char *name, const char *title, RooDataSet *t, 
	     const RooArgSet& vars, const RooFormulaVar& cutVar) ;
  RooDataSet(const char *name, const char *title, TTree *t, 
	     const RooArgSet& vars, const RooFormulaVar& cutVar) ;
  RooDataSet(const char *name, const char *title, TTree *ntuple, 
	     const RooArgSet& vars, const char *cuts);
  RooDataSet(const char *name, const char *filename, const char *treename, 
	     const RooArgSet& vars, const char *cuts);  
  RooDataSet(RooDataSet const & other, const char* newname=0) ;
  virtual TObject* Clone(const char* newname=0) const { return new RooDataSet(*this,newname?newname:GetName()) ; }
  virtual ~RooDataSet() ;

  // Read data from a text file and create a dataset from it.
  // The possible options are: (D)ebug, (Q)uiet.
  static RooDataSet *read(const char *filename, RooArgSet &variables,
			  const char *opts= "", const char* commonPath="",
			  const char *indexCatName=0) ;

  // Add one ore more rows of data
  virtual void add(const RooArgSet& row, Double_t weight=1.0);
  void append(RooDataSet& data) ;
  RooAbsArg* addColumn(RooAbsArg& var) ;

  // Load a given row of data
  virtual const RooArgSet* get(Int_t index) const;
  virtual const RooArgSet* get() const { return &_vars ; } 

  virtual Roo1DTable* table(RooAbsCategory& cat, const char* cuts="", const char* opts="") const ;

  // Plot the distribution of a real valued arg
  virtual RooPlot *plotOn(RooPlot *frame, const char* cuts="", Option_t* drawOptions="P") const;
  TH1F* createHistogram(const RooAbsReal& var, const char* cuts="", 
			const char *name= "hist") const;	 
  TH2F* createHistogram(const RooAbsReal& var1, const RooAbsReal& var2, const char* cuts="", 
			const char *name= "hist") const;	 
  TH2F* createHistogram(const RooAbsReal& var1, const RooAbsReal& var2, Int_t nx, Int_t ny,
                        const char* cuts="", const char *name="hist") const;

  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt= Standard, 
			     TString indent= "") const;

  virtual Int_t numEntries() const { return (Int_t)GetEntries() ; }
  virtual void reset() { Reset() ; }

  // Forwarded from TTree
  inline Stat_t GetEntries() const { return _tree->GetEntries() ; }
  inline void Reset(Option_t* option=0) { _tree->Reset(option) ; }
  inline Int_t Fill() { return _tree->Fill() ; }
  inline Int_t GetEntry(Int_t entry = 0, Int_t getall = 0) { return _tree->GetEntry(entry,getall) ; }
  inline Int_t Scan(const char* varexp="", const char* selection="", Option_t* option="", 
		    Int_t nentries = 1000000000, Int_t firstentry = 0) {
    _tree->Scan(varexp,selection,option,nentries,firstentry) ; 
  }


  // WVE Debug stuff
  void dump() ;
  void origPrint() { _tree->Print() ; }

  // Cache copy feature is not publicly accessible
  RooDataSet(const char *name, const char *title, RooDataSet *ntuple, 
	     const RooArgSet& vars, Bool_t copyCache);

protected:

  // RooFitContext optimizer interface
  friend class RooFitContext ;
  virtual void cacheArg(RooAbsArg& var) ;
  virtual void cacheArgs(RooArgSet& varSet) ;

  void fillCacheArgs() ;

  // Load data from another TTree
  void loadValues(const TTree *t, RooFormulaVar* cutVar=0) ; 
  void loadValues(const char *filename, const char *treename,
		  RooFormulaVar *cutVar=0);


  TTree *_tree ; 

  // Column structure definition
  RooArgSet _truth;        
  TString _blindString ;

private:

  void initialize(const RooArgSet& vars);
  void initCache(const RooArgSet& cachedVars) ; 

  enum { bufSize = 8192 };
  ClassDef(RooDataSet,1) // Unbinned data set
};

#endif
