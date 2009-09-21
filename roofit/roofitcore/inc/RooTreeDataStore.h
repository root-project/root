/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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
#ifndef ROO_TREE_DATA_STORE
#define ROO_TREE_DATA_STORE

#include "RooAbsDataStore.h" 
#include "TString.h"

class RooAbsArg ;
class RooArgList ;
class TTree ;
class RooFormulaVar ;
class RooArgSet ;


class RooTreeDataStore : public RooAbsDataStore {
public:

  RooTreeDataStore() ; 
  RooTreeDataStore(TTree* t, const RooArgSet& vars) ; 

  // Empty ctor
  RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars) ;
  virtual RooAbsDataStore* clone(const RooArgSet& vars, const char* newname=0) const { return new RooTreeDataStore(*this,vars,newname) ; }

  // Ctors from TTree
  RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars, TTree& t, const RooFormulaVar& select) ; 
  RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars, TTree& t, const char* selExpr=0) ; 

  // Ctors from DataStore
  RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars, const RooAbsDataStore& tds, const RooFormulaVar& select) ;
  RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars, const RooAbsDataStore& tds, const char* selExpr=0) ;

  RooTreeDataStore(const char *name, const char *title, RooAbsDataStore& tds, 
		   const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
		   Int_t nStart, Int_t nStop, Bool_t /*copyCache*/) ;

  RooTreeDataStore(const RooTreeDataStore& other, const char* newname=0) ;
  RooTreeDataStore(const RooTreeDataStore& other, const RooArgSet& vars, const char* newname=0) ;
  virtual ~RooTreeDataStore() ;


  // Write current row
  virtual Int_t fill() ;

  // Retrieve a row
  using RooAbsDataStore::get ;
  virtual const RooArgSet* get(Int_t index) const ;

  // Change observable name
  virtual Bool_t changeObservableName(const char* from, const char* to) ;
  
  // Add one or more columns
  virtual RooAbsArg* addColumn(RooAbsArg& var, Bool_t adjustRange=kTRUE) ;
  virtual RooArgSet* addColumns(const RooArgList& varList) ;

  // Merge column-wise
  RooAbsDataStore* merge(const RooArgSet& allvars, std::list<RooAbsDataStore*> dstoreList) ;

  // Add rows 
  virtual void append(RooAbsDataStore& other) ;

  // General & bookkeeping methods
  virtual Bool_t valid() const ;
  virtual Int_t numEntries() const ;
  virtual void reset() ;
  
  // Tree access
  TTree& tree() { return *_tree ; }
  virtual const TTree* tree() const { return _tree ; }  

  // Forwarded from TTree
  Stat_t GetEntries() const;
  void Reset(Option_t* option=0);
  Int_t Fill();
  Int_t GetEntry(Int_t entry = 0, Int_t getall = 0);

  void	Draw(Option_t* option = "") ;

  // Constant term  optimizer interface
  virtual void cacheArgs(const RooAbsArg* owner, RooArgSet& varSet, const RooArgSet* nset=0) ;
  virtual const RooAbsArg* cacheOwner() { return _cacheOwner ; }
  virtual void setArgStatus(const RooArgSet& set, Bool_t active) ;
  virtual void resetCache() ;

  void loadValues(const TTree *t, const RooFormulaVar* select=0, const char* rangeName=0, Int_t nStart=0, Int_t nStop=2000000000)  ;
  void loadValues(const RooAbsDataStore *tds, const RooFormulaVar* select=0, const char* rangeName=0, Int_t nStart=0, Int_t nStop=2000000000)  ;

  virtual void checkInit() const;
  
 protected:

  void initialize();
  void attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVars) ;

  // TTree Branch buffer size control
  void setBranchBufferSize(Int_t size) { _defTreeBufSize = size ; }
  Int_t getBranchBufferSize() const { return _defTreeBufSize ; }

  static Int_t _defTreeBufSize ;  

  void createTree(const char* name, const char* title) ; 
  TTree *_tree ;           // TTree holding the data points
  TTree *_cacheTree ;      //! TTree holding the cached function values
  const RooAbsArg* _cacheOwner ; //! Object owning cache contents
  mutable Bool_t _defCtor ;//! Was object constructed with default ctor?


  ClassDef(RooTreeDataStore,1) // TTree-based Data Storage class
};


#endif
