/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooDataSet.rdl,v 1.17 2001/05/14 22:54:20 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   01-Nov-1999 DK Created initial version
 *   30-Nov-2000 WV Add support for multiple file reading with optional common path
 *   09-Mar-2001 WV Migrate from RooFitTools and adapt to RooFitCore
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/
#ifndef ROO_DATA_SET
#define ROO_DATA_SET

#include "TTree.h"

#include "RooFitCore/RooPrintable.hh"
#include "RooFitCore/RooArgSet.hh"

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

class RooDataSet : public TTree, public RooPrintable {
public:

  // Constructors, factory methods etc.
  inline RooDataSet() { }
  RooDataSet(const char *name, const char *title, const RooArgSet& vars) ;
  RooDataSet(const char *name, const char *title, RooDataSet *ntuple, 
	     const RooArgSet& vars, const char *cuts);
  RooDataSet(const char *name, const char *title, TTree *ntuple, 
	     const RooArgSet& vars, const char *cuts);
  RooDataSet(const char *name, const char *filename, const char *treename, 
	     const RooArgSet& vars, const char *cuts);  
  RooDataSet(RooDataSet const & other);
  inline virtual ~RooDataSet() ;

  // Read data from a text file and create a dataset from it.
  // The possible options are: (D)ebug, (Q)uiet.
  static RooDataSet *read(const char *filename, RooArgSet &variables,
			  const char *opts= "", const char* commonPath="",
			  const char *indexCatName=0) ;

  // Add one ore more rows of data
  void add(const RooArgSet& row);
  void append(RooDataSet& data) ;
  void addColumn(RooAbsArg& var) ;

  // Load a given row of data
  const RooArgSet* get() const { return &_vars ; } // last loaded row
  const RooArgSet* get(Int_t index) const;

  Roo1DTable* Table(RooAbsCategory& cat, const char* cuts="", const char* opts="") ;

  // Plot the distribution of a real valued arg
  RooPlot *plotOn(RooPlot *frame, const char* cuts="", Option_t* drawOptions="P") const;
  TH1F* createHistogram(const RooAbsReal& var, const char* cuts="", 
			const char *name= "hist") const;	 
 
  // Printing interface (human readable)
  virtual void printToStream(ostream& os, PrintOption opt= Standard, 
			     TString indent= "") const;
  inline virtual void Print(Option_t *options= 0) const {
    printToStream(defaultStream(),parseOptions(options));
  }


  // WVE Debug stuff
  void dump() ;
  void origPrint() { TTree::Print() ; }

protected:

  void cacheArg(RooAbsArg& var) ;
  void cacheArgs(RooArgSet& varSet) ;
  void fillCacheArgs() ;

  // Load data from another TTree
  void loadValues(const TTree *ntuple, const char *cuts);
  void loadValues(const char *filename, const char *treename,
		  const char *cuts);

  // RooFitContext optimizer interface
  friend class RooFitContext ;

  // Column structure definition
  RooArgSet _vars, _truth;
  RooArgSet _cachedVars ;  //! do not clone
  TString _blindString ;

private:

  void initialize(const RooArgSet& vars);
  TIterator *_iterator;    //! don't make this data member persistent
  TIterator *_cacheIter ;  //! don't make this data member persistent
  TBranch *_branch; //! don't make this data member persistent

  enum { bufSize = 8192 };
  ClassDef(RooDataSet,1) // Data set for fitting
};

#endif
