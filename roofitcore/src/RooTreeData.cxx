/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooTreeData.cxx,v 1.73 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [DATA] --
// RooTreeData is the abstract base class for data collection that
// use a TTree as internal storage mechanism

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <math.h>

#include "TRegexp.h"
#include "TTreeFormula.h"
#include "TIterator.h"
#include "TObjArray.h"
#include "TTreeFormula.h"
#include "TIterator.h"
#include "TPaveText.h"
#include "TLeaf.h"
#include "TMath.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TFile.h"
#include "TChain.h"
#include "TROOT.h"

#include "RooTreeData.h"
#include "RooAbsArg.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooRealVar.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "Roo1DTable.h"
#include "RooFormula.h"
#include "RooCategory.h"
#include "RooPlot.h"
#include "RooStringVar.h"
#include "RooHist.h"
#include "RooFormulaVar.h"
#include "RooTrace.h"
#include "RooAbsBinning.h" 
#include "RooCmdConfig.h" 
#include "RooGlobalFunc.h"

ClassImp(RooTreeData)
;

Int_t RooTreeData::_defTreeBufSize = 4096 ;


//________________________________________________________________
//   Interface functions to TTree

//________________________________________________________________
Int_t RooTreeData::Scan(const char* varexp, const char* selection, Option_t* option, 
		    Int_t nentries, Int_t firstentry) 
{
   //   Interface function to TTree::Scan
   return _tree->Scan(varexp,selection,option,nentries,firstentry) ;
}

//________________________________________________________________
Int_t RooTreeData::ScanCache(const char* varexp, const char* selection, Option_t* option, 
		    Int_t nentries, Int_t firstentry) 
{
   // Interface function to TTree::Scan
   return _cacheTree->Scan(varexp,selection,option,nentries,firstentry) ;
}
 
//________________________________________________________________
Stat_t RooTreeData::GetEntries() const
{
   // Interface function to TTree::GetEntries
   return _tree->GetEntries() ;
}
 
//________________________________________________________________
void RooTreeData::Reset(Option_t* option)
{
   // Interface function to TTree::Reset
   _tree->Reset(option) ;
}
 
//________________________________________________________________
void RooTreeData::treePrint()
{
   // Interface function to TTree::Print
   _tree->Print();
}
 
//________________________________________________________________
Int_t RooTreeData::Fill()
{
   // Interface function to TTree::Fill
   return _tree->Fill() ;
}
 
//________________________________________________________________
Int_t RooTreeData::GetEntry(Int_t entry, Int_t getall)
{
   // Interface function to TTree::GetEntry
   Int_t ret1 = _tree->GetEntry(entry,getall) ; 
   if (!ret1) return 0 ;
   _cacheTree->GetEntry(entry,getall) ; 
   return ret1 ;
}

//________________________________________________________________
//   Interface functions to TTree
//________________________________________________________________
//   Interface functions to TTree
//________________________________________________________________
//   Interface functions to TTree

RooTreeData::RooTreeData() 
{
  // Default constructor
  RooTrace::create(this) ; 
  _defCtor = kTRUE ;  
  _cacheTree = 0 ;
  _tree = 0 ;
}


RooTreeData::RooTreeData(const char *name, const char *title, const RooArgSet& vars) :
  RooAbsData(name,title,vars), _defCtor(kFALSE), _truth("Truth")
{
  // Constructor of empty collection with specified dimensions
  RooTrace::create(this) ;
  
  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(name,title) ;

  // Constructor with list of variables
  initialize();
}


RooTreeData::RooTreeData(const char *name, const char *title, RooTreeData *t, 
                       const RooArgSet& vars, const char *cuts) :
 RooAbsData(name,title,vars), _defCtor(kFALSE), _truth("Truth"), 
  _blindString(t->_blindString)
{
  // Constructor from existing data collection with specified dimensions and
  // optional string expression cut

  RooTrace::create(this) ;

  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(name,title) ;

  // Constructor from existing data set with list of variables and cut expression
  initialize();

  if (cuts && *cuts) {
    // Create a RooFormulaVar cut from given cut expression
    RooFormulaVar cutVar(cuts,cuts,t->_vars) ;
    loadValues(t,&cutVar);
  } else {
    loadValues(t,0);
  }
}



RooTreeData::RooTreeData(const char *name, const char *title, RooTreeData *t, 
                       const RooArgSet& vars, const RooFormulaVar& cutVar) :
  RooAbsData(name,title,vars), _defCtor(kFALSE), _truth("Truth"), 
  _blindString(t->_blindString)
{
  // Constructor from existing data collection with specified dimensions and
  // RooFormulaVar cut

  RooTrace::create(this) ;

  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(name,title) ;

  // Constructor from existing data set with list of variables and cut expression
  initialize();

  // Deep clone cutVar and attach clone to this dataset
  RooArgSet* tmp = (RooArgSet*) RooArgSet(cutVar).snapshot() ;
  if (!tmp) {
    cout << "RooTreeData::RooTreeData(" << GetName() << ") Couldn't deep-clone cut variable, abort." << endl ;
    RooErrorHandler::softAbort() ;
  }

  RooFormulaVar* cloneVar = (RooFormulaVar*) tmp->find(cutVar.GetName()) ;
  cloneVar->attachDataSet(*this) ;

  loadValues(t,cloneVar);

  delete tmp ;
}


RooTreeData::RooTreeData(const char *name, const char *title, TTree *t, 
                       const RooArgSet& vars, const RooFormulaVar& cutVar) :
  RooAbsData(name,title,vars), _defCtor(kFALSE), _truth("Truth")
{
  // Constructor from external TTree with specified dimensions and
  // RooFormulaVar cut

  RooTrace::create(this) ;

  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(name,title) ;

  // Constructor from existing data set with list of variables and cut expression
  initialize();

  // Deep clone cutVar and attach clone to this dataset
  RooArgSet* tmp = (RooArgSet*) RooArgSet(cutVar).snapshot() ;
  if (!tmp) {
    cout << "RooTreeData::RooTreeData(" << GetName() << ") Couldn't deep-clone cut variable, abort." << endl ;
    RooErrorHandler::softAbort() ;
  }
  RooFormulaVar* cloneVar = (RooFormulaVar*) tmp->find(cutVar.GetName()) ;
  cloneVar->attachDataSet(*this) ;

  loadValues(t,cloneVar);

  delete tmp ;
}



RooTreeData::RooTreeData(const char *name, const char *title, RooTreeData *t, 
			 const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
			 Int_t nStart, Int_t nStop, Bool_t /*copyCache*/) :
  RooAbsData(name,title,vars), _defCtor(kFALSE), _truth("Truth"), 
  _blindString(t->_blindString)
{
  // Protected constructor for internal use only

  RooTrace::create(this) ;

  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(name,title) ;

  // Deep clone cutVar and attach clone to this dataset
  RooArgSet* cloneVarSet = 0;
  RooFormulaVar* cloneVar = 0;
  if (cutVar) {
    cloneVarSet = (RooArgSet*) RooArgSet(*cutVar).snapshot() ;
    if (!cloneVarSet) {
      cout << "RooTreeData::RooTreeData(" << GetName() << ") Couldn't deep-clone cut variable, abort." << endl ;
      RooErrorHandler::softAbort() ;
    }
   cloneVar = (RooFormulaVar*) cloneVarSet->find(cutVar->GetName()) ;
    cloneVar->attachDataSet(*t) ;
  }

  // Constructor from existing data set with list of variables that preserves the cache
  initialize();
  initCache(t->_cachedVars) ;
  
  loadValues(t,cloneVar,cutRange,nStart,nStop);

  // WVE copy values of cached variables here!!!

  if (cloneVarSet) delete cloneVarSet ;
}




RooTreeData::RooTreeData(const char *name, const char *title, TTree *t, 
                       const RooArgSet& vars, const char *cuts) :
  RooAbsData(name,title,vars), _defCtor(kFALSE), _truth("Truth")
{
  // Constructor from external TTree with specified dimensions and
  // optional string expression cut

  RooTrace::create(this) ;

  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(name,title) ;

  // Constructor from existing TTree with list of variables and cut expression
  initialize();

  if (cuts && *cuts) {
    // Create a RooFormulaVar cut from given cut expression
    RooFormulaVar cutVar(cuts,cuts,_vars) ;
    loadValues(t,&cutVar);
  } else {
    loadValues(t,0);    
  }
}



RooTreeData::RooTreeData(const char *name, const char *filename,
		       const char *treename,
                       const RooArgSet& vars, const char *cuts) :
  RooAbsData(name,name,vars), _defCtor(kFALSE), _truth("Truth")
{
  // Constructor from external TTree with given name in given file
  // with specified dimensions and optional string expression cut

  RooTrace::create(this) ;

  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(name,name) ;

  // Constructor from TTree file with list of variables and cut expression
  initialize();

  // Create a RooFormulaVar cut from given cut expression
  if (cuts && *cuts) {
    RooFormulaVar cutVar(cuts,cuts,_vars) ;    
    loadValues(filename,treename,&cutVar);
  } else {
    loadValues(filename,treename,0);
  }
}


RooTreeData::RooTreeData(RooTreeData const & other, const char* newName) : 
  RooAbsData(other,newName), _defCtor(other._defCtor), _truth("Truth")
{
  // Copy constructor
  RooTrace::create(this) ;

  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(newName,other.GetTitle()) ;

  initialize() ;
  loadValues(&other,0) ;
}


void RooTreeData::createTree(const char* name, const char* title)
{
  // Create TTree object that lives in memory, independent of current
  // location of gDirectory

  TString pwd(gDirectory->GetPath()) ;
  TString memDir(gROOT->GetName()) ;
  memDir.Append(":/") ;
  gDirectory->cd(memDir) ;
  if (!_tree) {
    _tree = new TTree(name,title) ;
  }
  if (!_cacheTree) {
    _cacheTree = new TTree(name,title) ;
  }
  gDirectory->RecursiveRemove(_tree) ;
  gDirectory->RecursiveRemove(_cacheTree) ;
  gDirectory->cd(pwd) ;
  
}


RooTreeData::~RooTreeData()
{
  // Destructor
  RooTrace::destroy(this) ;

  delete _tree ;
  delete _cacheTree ;
}


void RooTreeData::initialize() {
  // Attach variables of internal ArgSet 
  // to the corresponding TTree branches

  // Recreate (empty) cache tree
  createTree(GetName(),GetTitle()) ;

  // Attach each variable to the dataset
  _iterator->Reset() ;  
  RooAbsArg *var;
  while((0 != (var= (RooAbsArg*)_iterator->Next()))) {
    var->attachToTree(*_tree,_defTreeBufSize) ;
  }
}


void RooTreeData::initCache(const RooArgSet& cachedVars) 
{
  // Initialize cache of dataset: attach variables of cache ArgSet
  // to the corresponding TTree branches

  // iterate over the cache variables for this dataset
  _cachedVars.removeAll() ;
  TIterator* iter = cachedVars.createIterator() ;
  RooAbsArg *var;
  while((0 != (var= (RooAbsArg*)iter->Next()))) {    
    var->attachToTree(*_cacheTree,_defTreeBufSize) ;
    _cachedVars.add(*var) ;
  }
  delete iter ;
}


void RooTreeData::loadValues(const char *filename, const char *treename,
			    RooFormulaVar* cutVar) {
  // Load TTree name 'treename' from file 'filename' and pass call
  // to loadValue(TTree*,...)

  TFile *file= (TFile*)gROOT->GetListOfFiles()->FindObject(filename);
  if(!file) file= new TFile(filename);
  if(!file) {
    cout << "RooTreeData::loadValues: unable to open " << filename << endl;
  }
  else {
    TTree* tree= (TTree*)gDirectory->Get(treename);
    loadValues(tree,cutVar);
  }
}



void RooTreeData::loadValues(const RooTreeData *t, RooFormulaVar* select, 
			     const char* rangeName, Int_t nStart, Int_t nStop)  
{
  // Load values from dataset 't' into this data collection, optionally
  // selecting events using 'select' RooFormulaVar
  //

  // Redirect formula servers to source data row
  if (select) {
    select->recursiveRedirectServers(*t->get()) ;

    RooArgSet branchList ;
    select->branchNodeServerList(&branchList) ;
    TIterator* iter = branchList.createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      arg->setOperMode(RooAbsArg::ADirty) ;
    }
    delete iter ;
  }

  // Force RDS internal initialization
  t->get(0) ;

  // Loop over events in source tree   
  RooAbsArg* arg = 0;
  Int_t nevent = nStop < t->numEntries() ? nStop : t->numEntries() ;
  Bool_t allValid ;

  for(Int_t i=nStart; i < nevent ; ++i) {
    t->_tree->GetEntry(i,1) ;
    t->_cacheTree->GetEntry(i,1) ;

    // Does this event pass the cuts?
    if (select && select->getVal()==0) {
      continue ; 
    }
    
    _vars = t->_vars ;
    _iterator->Reset() ;

    // Check that all copied values are valid
    allValid=kTRUE ;
    while((arg=(RooAbsArg*)_iterator->Next())) {
      if (!arg->isValid() || (rangeName && !arg->inRange(rangeName))) {
	allValid=kFALSE ;
	break ;
      }
    }
    if (!allValid) continue ;

    _cachedVars = t->_cachedVars ;
    Fill() ;
   }
  
  SetTitle(t->GetTitle());
}


void RooTreeData::loadValues(const TTree *t, RooFormulaVar* select, const char* /*rangeName*/, Int_t /*nStart*/, Int_t /*nStop*/) 
{
  // Load values from tree 't' into this data collection, optionally
  // selecting events using 'select' RooFormulaVar
  //
  // The source tree 't' is first clone as not disturb its branch
  // structure when retrieving information from it.

  // Clone source tree
  // WVE Clone() crashes on trees, CloneTree() crashes on tchains :-(
  TTree* tClone ;
  if (dynamic_cast<const TChain*>(t)) {
    tClone = (TTree*) t->Clone() ; 
  } else {
    tClone = ((TTree*)t)->CloneTree() ;
  }
    
  // Clone list of variables  
  RooArgSet *sourceArgSet = (RooArgSet*) _vars.snapshot(kFALSE) ;
  
  // Attach args in cloned list to cloned source tree
  TIterator* sourceIter =  sourceArgSet->createIterator() ;
  RooAbsArg* sourceArg = 0;
  while ((sourceArg=(RooAbsArg*)sourceIter->Next())) {
    sourceArg->attachToTree(*tClone,_defTreeBufSize) ;
  }

  // Redirect formula servers to sourceArgSet
  if (select) {
    select->recursiveRedirectServers(*sourceArgSet) ;

    RooArgSet branchList ;
    select->branchNodeServerList(&branchList) ;
    TIterator* iter = branchList.createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      arg->setOperMode(RooAbsArg::ADirty) ;
    }
    delete iter ;
  }

  // Loop over events in source tree   
  RooAbsArg* destArg = 0;
  Int_t numInvalid(0) ;
  Int_t nevent= (Int_t)tClone->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=tClone->GetEntryNumber(i);
    if (entryNumber<0) break;
    tClone->GetEntry(entryNumber,1);
 
    // Copy from source to destination
     _iterator->Reset() ;
     sourceIter->Reset() ;
     Bool_t allOK(kTRUE) ;
     while ((destArg = (RooAbsArg*)_iterator->Next())) {              
       sourceArg = (RooAbsArg*) sourceIter->Next() ;
       destArg->copyCache(sourceArg) ;
       if (!destArg->isValid()) {
	 numInvalid++ ;
	 allOK=kFALSE ;
	 break ;
       }       
     }   

     // Does this event pass the cuts?
     if (!allOK || (select && select->getVal()==0)) {
       continue ; 
     }

     Fill() ;
   }

  if (numInvalid>0) {
    cout << "RooTreeData::loadValues(" << GetName() << ") Ignored " << numInvalid << " out of range events" << endl ;
  }
  
  SetTitle(t->GetTitle());

  delete sourceIter ;
  delete sourceArgSet ;
  delete tClone ;
}


void RooTreeData::dump() {
  // DEBUG: Dump contents

  RooAbsArg* arg ;

  // Header line
  _iterator->Reset() ;
  while ((arg = (RooAbsArg*)_iterator->Next())) {       
    cout << arg->GetName() << "  " ;
  }  
  cout << endl ;
     
  // Dump contents 
  Int_t nevent= (Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=_tree->GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);
     
    _iterator->Reset() ;
    // Copy from source to destination
    while ((arg = (RooAbsArg*)_iterator->Next())) {
     arg->writeToStream(cout,kTRUE) ; 
     cout << " " ;
     }  
    cout << endl ;
  }
}


void RooTreeData::resetCache() 
{
  // Reset the cache 

  // Empty list of cached functions
  _cachedVars.removeAll() ;

  // Delete & recreate cache tree 
  delete _cacheTree ;
  _cacheTree = 0 ;
  createTree(GetName(),GetTitle()) ;

  return ;
}




void RooTreeData::setArgStatus(const RooArgSet& set, Bool_t active) 
{
  TIterator* iter = set.createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    RooAbsArg* depArg = _vars.find(arg->GetName()) ;
    if (!depArg) {
      cout << "RooTreeData::setArgStatus(" << GetName() 
	   << ") dataset doesn't contain variable " << arg->GetName() << endl ;
      continue ;
    }
    depArg->setTreeBranchStatus(*_tree,active) ;
  }
  delete iter ;
}



void RooTreeData::cacheArgs(RooArgSet& newVarSet, const RooArgSet* nset) 
{
  // Cache given RooAbsArgs with this tree: The tree is
  // given direct write access of the args internal cache
  // the args values is pre-calculated for all data points
  // in this data collection. Upon a get() call, the
  // internal cache of 'newVar' will be loaded with the
  // precalculated value and it's dirty flag will be cleared.

  TIterator *iter = newVarSet.createIterator() ;
  RooAbsArg *arg ;

  Bool_t doTreeFill = (_cachedVars.getSize()==0) ;
    
  while ((arg=(RooAbsArg*)iter->Next())) {
    // Attach original newVar to this tree
    arg->attachToTree(*_cacheTree,_defTreeBufSize) ;
    arg->redirectServers(_vars) ;
    _cachedVars.add(*arg) ;
  }
  
  // Refill regular and cached variables of current tree from clone
  for (int i=0 ; i<GetEntries() ; i++) {
    get(i) ;
    
    // Evaluate the cached variables and store the results
    iter->Reset() ;
    while ((arg=(RooAbsArg*)iter->Next())) {
      arg->setValueDirty() ;
      arg->syncCache(nset) ;
      if (!doTreeFill) {
	arg->fillTreeBranch(*_cacheTree) ;
      }
    }

    if (doTreeFill) {
      _cacheTree->Fill() ;
    }
  }

  delete iter ;
}



const RooArgSet* RooTreeData::get(Int_t index) const 
{
  // Load the n-th data point (n='index') in memory
  // and return a pointer to the internal RooArgSet
  // holding its coordinates.

  checkInit() ;

  Int_t ret = ((RooTreeData*)this)->GetEntry(index, 1) ;
  if(!ret) return 0;

  if (_doDirtyProp) {
    // Raise all dirty flags 
    _iterator->Reset() ;
    RooAbsArg* var = 0;
    while ((var=(RooAbsArg*)_iterator->Next())) {
      var->setValueDirty() ; // This triggers recalculation of all clients
    } 
    
    _cacheIter->Reset() ;
    while ((var=(RooAbsArg*)_cacheIter->Next())) {
      var->setValueDirty()  ; // This triggers recalculation of all clients, but doesn't recalculate self
      var->clearValueDirty() ; 
    } 
  }

  return &_vars;
}


RooAbsArg* RooTreeData::addColumn(RooAbsArg& newVar)
{
  // Add a new column to the data set which holds the pre-calculated values
  // of 'newVar'. This operation is only meaningful if 'newVar' is a derived
  // value.
  //
  // The return value points to the added element holding 'newVar's value
  // in the data collection. The element is always the corresponding fundamental
  // type of 'newVar' (e.g. a RooRealVar if 'newVar' is a RooFormulaVar)
  //
  // Note: This function is explicitly NOT intended as a speed optimization
  //       opportunity for the user. Components of complex PDFs that can be
  //       precalculated with the dataset are automatically identified as such
  //       and will be precalculated when fitting to a dataset
  // 
  //       By forcibly precalculating functions with non-trivial Jacobians,
  //       or functions of multiple variables occurring in the data set,
  //       using addColumn(), you may alter the outcome of the fit. 
  //
  //       Only in cases where such a modification of fit behaviour is intentional, 
  //       this function should be used. (E.g collapsing a continuous B0 flavour
  //       probability into a 2-state B0/B0bar category)

  checkInit() ;

  // Create a fundamental object of the right type to hold newVar values
  RooAbsArg* valHolder= newVar.createFundamental();
  // Sanity check that the holder really is fundamental
  if(!valHolder->isFundamental()) {
    cout << GetName() << "::addColumn: holder argument is not fundamental: \""
	 << valHolder->GetName() << "\"" << endl;
    return 0;
  }

  // Clone variable and attach to cloned tree 
  RooArgSet* newVarCloneList = (RooArgSet*) RooArgSet(newVar).snapshot() ;  
  if (!newVarCloneList) {
    cout << "RooTreeData::addColumn(" << GetName() << ") Couldn't deep-clone variable to add, abort." << endl ;
    return 0 ;
  }
  RooAbsArg* newVarClone = newVarCloneList->find(newVar.GetName()) ;
  newVarClone->recursiveRedirectServers(_vars,kFALSE) ;

  // Attach value place holder to this tree
  ((RooAbsArg*)valHolder)->attachToTree(*_tree,_defTreeBufSize) ;
  _vars.addOwned(*valHolder) ;

  // Fill values of of placeholder
  for (int i=0 ; i<GetEntries() ; i++) {
    get(i) ;

    newVarClone->syncCache(&_vars) ;
    valHolder->copyCache(newVarClone) ;
    valHolder->fillTreeBranch(*_tree) ;
  }
  
  delete newVarCloneList;  
  return valHolder ;
}



RooArgSet* RooTreeData::addColumns(const RooArgList& varList)
{
  TIterator* vIter = varList.createIterator() ;
  RooAbsArg* var ;

  checkInit() ;

  TList cloneSetList ;
  RooArgSet cloneSet ;
  RooArgSet* holderSet = new RooArgSet ;

  while((var=(RooAbsArg*)vIter->Next())) {
    // Create a fundamental object of the right type to hold newVar values
    RooAbsArg* valHolder= var->createFundamental();
    holderSet->add(*valHolder) ;

    // Sanity check that the holder really is fundamental
    if(!valHolder->isFundamental()) {
      cout << GetName() << "::addColumn: holder argument is not fundamental: \""
	   << valHolder->GetName() << "\"" << endl;
      return 0;
    }
    
    // Clone variable and attach to cloned tree 
    RooArgSet* newVarCloneList = (RooArgSet*) RooArgSet(*var).snapshot() ;  
    if (!newVarCloneList) {
      cout << "RooTreeData::RooTreeData(" << GetName() << ") Couldn't deep-clone variable " << var->GetName() << ", abort." << endl ;
      return 0 ;
    }
    RooAbsArg* newVarClone = newVarCloneList->find(var->GetName()) ;   
    newVarClone->recursiveRedirectServers(_vars,kFALSE) ;
    newVarClone->recursiveRedirectServers(*holderSet,kFALSE) ;

    cloneSetList.Add(newVarCloneList) ;
    cloneSet.add(*newVarClone) ;

    // Attach value place holder to this tree
    ((RooAbsArg*)valHolder)->attachToTree(*_tree,_defTreeBufSize) ;
    _vars.addOwned(*valHolder) ;
  }
  delete vIter ;


  TIterator* cIter = cloneSet.createIterator() ;
  TIterator* hIter = holderSet->createIterator() ;
  RooAbsArg *clone, *holder ;
  // Fill values of of placeholder
  for (int i=0 ; i<GetEntries() ; i++) {
    get(i) ;

    cIter->Reset() ;
    hIter->Reset() ;
    while((clone=(RooAbsArg*)cIter->Next())) {
      holder = (RooAbsArg*)hIter->Next() ;

      clone->syncCache(&_vars) ;
      holder->copyCache(clone) ;
      holder->fillTreeBranch(*_tree) ;
    }
  }
  
  delete cIter ;
  delete hIter ;

  cloneSetList.Delete() ;
  return holderSet ;
}




TList* RooTreeData::split(const RooAbsCategory& splitCat) const
{
  // Sanity check
  if (!splitCat.dependsOn(*get())) {
    cout << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName() 
	 << " doesn't depend on any variable in this dataset" << endl ;
    return 0 ;
  }

  // Clone splitting category and attach to self
  RooAbsCategory* cloneCat =0;
  RooArgSet* cloneSet = 0;
  if (splitCat.isDerived()) {
    cloneSet = (RooArgSet*) RooArgSet(splitCat).snapshot(kTRUE) ;
    if (!cloneSet) {
      cout << "RooTreeData::split(" << GetName() << ") Couldn't deep-clone splitting category, abort." << endl ;
      return 0 ;
    }
    cloneCat = (RooAbsCategory*) cloneSet->find(splitCat.GetName()) ;
    cloneCat->attachDataSet(*this) ;
  } else {
    cloneCat = dynamic_cast<RooAbsCategory*>(get()->find(splitCat.GetName())) ;
    if (!cloneCat) {
      cout << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName() 
	   << " is fundamental and does not appear in this dataset" << endl ;
      return 0 ;      
    }
  }

  // Split a dataset in a series of subsets, each corresponding
  // to a state of splitCat
  TList* dsetList = new TList ;

  // Construct set of variables to be included in split sets = full set - split category
  RooArgSet subsetVars(*get()) ;
  if (splitCat.isDerived()) {
    RooArgSet* vars = splitCat.getVariables() ;
    subsetVars.remove(*vars,kTRUE,kTRUE) ;
    delete vars ;
  } else {
    subsetVars.remove(splitCat,kTRUE,kTRUE) ;
  }
  
  // Loop over dataset and copy event to matching subset
  Int_t i ;
  for (i=0 ; i<numEntries() ; i++) {
    const RooArgSet* row =  get(i) ;
    RooAbsData* subset = (RooAbsData*) dsetList->FindObject(cloneCat->getLabel()) ;
    if (!subset) {
      subset = emptyClone(cloneCat->getLabel(),cloneCat->getLabel(),&subsetVars) ;
      dsetList->Add((RooAbsArg*)subset) ;
    }
    subset->add(*row,weight()) ;
  }

  delete cloneSet ;
  return dsetList ;
}


RooPlot* RooTreeData::plotOn(RooPlot* frame, const RooLinkedList& argList) const
{
  // Plot dataset on specified frame. By default an unbinned dataset will use the default binning of
  // the target frame. A binned dataset will by default retain its intrinsic binning.
  //
  // The following optional named arguments can be used to modify the default behavior
  //
  // Data representation options
  // ---------------------------
  // Asymmetry(const RooCategory& c) -- Show the asymmetry of the daya in given two-state category [F(+)-F(-)] / [F(+)+F(-)]. 
  //                                    Category must have two states with indices -1 and +1 or three states with indeces -1,0 and +1.
  // ErrorType(RooAbsData::EType)    -- Select the type of error drawn: Poisson (default) draws asymmetric Poisson
  //                                    confidence intervals. SumW2 draws symmetric sum-of-weights error
  // Binning(double xlo, double xhi, -- Use specified binning to draw dataset
  //                      int nbins)
  // Binning(const RooAbsBinning&)   -- Use specified binning to draw dataset
  // Binning(const char* name)       -- Use binning with specified name to draw dataset
  // RefreshNorm(Bool_t flag)        -- Force refreshing for PDF normalization information in frame.
  //                                    If set, any subsequent PDF will normalize to this dataset, even if it is
  //                                    not the first one added to the frame. By default only the 1st dataset
  //                                    added to a frame will update the normalization information
  //
  // Histogram drawing options
  // -------------------------
  // DrawOption(const char* opt)     -- Select ROOT draw option for resulting TGraph object
  // LineStyle(Int_t style)          -- Select line style by ROOT line style code, default is solid
  // LineColor(Int_t color)          -- Select line color by ROOT color code, default is black
  // LineWidth(Int_t width)          -- Select line with in pixels, default is 3
  // MarkerStyle(Int_t style)        -- Select the ROOT marker style, default is 21
  // MarkerColor(Int_t color)        -- Select the ROOT marker color, default is black
  // MarkerSize(Double_t size)       -- Select the ROOT marker size
  // XErrorSize(Double_t frac)       -- Select size of X error bar as fraction of the bin width, default is 1
  //
  //
  // Misc. other options
  // -------------------
  // Name(const chat* name)          -- Give curve specified name in frame. Useful if curve is to be referenced later
  // Invisble(Bool_t flag)           -- Add curve to frame, but do not display. Useful in combination AddTo()
  // AddTo(const char* name,         -- Add constructed histogram to already existing histogram with given name and relative weight factors
  // double_t wgtSelf, double_t wgtOther)
  // 
  //                                    
  //

  // New experimental plotOn() with varargs...

  // Define configuration for this method
  RooCmdConfig pc(Form("RooTreeData::plotOn(%s)",GetName())) ;
  pc.defineString("drawOption","DrawOption",0,"P") ;
  pc.defineString("cutRange","CutRange",0,"",kTRUE) ;
  pc.defineString("cutString","CutSpec",0,"") ;
  pc.defineString("histName","Name",0,"") ;
  pc.defineObject("cutVar","CutVar",0) ;
  pc.defineObject("binning","Binning",0) ;
  pc.defineString("binningName","BinningName",0,"") ;
  pc.defineInt("nbins","BinningSpec",0,100) ;
  pc.defineDouble("xlo","BinningSpec",0,0) ;
  pc.defineDouble("xhi","BinningSpec",1,1) ;
  pc.defineObject("asymCat","Asymmetry",0) ;
  pc.defineInt("lineColor","LineColor",0,-999) ;
  pc.defineInt("lineStyle","LineStyle",0,-999) ;
  pc.defineInt("lineWidth","LineWidth",0,-999) ;
  pc.defineInt("markerColor","MarkerColor",0,-999) ;
  pc.defineInt("markerStyle","MarkerStyle",0,-999) ;
  pc.defineDouble("markerSize","MarkerSize",0,-999) ;
  pc.defineInt("errorType","DataError",0,(Int_t)RooAbsData::Poisson) ;
  pc.defineInt("histInvisible","Invisible",0,0) ;
  pc.defineInt("refreshFrameNorm","RefreshNorm",0,0) ;
  pc.defineString("addToHistName","AddTo",0,"") ;
  pc.defineDouble("addToWgtSelf","AddTo",0,1.) ;
  pc.defineDouble("addToWgtOther","AddTo",1,1.) ;
  pc.defineDouble("xErrorSize","XErrorSize",0,1.) ;
  pc.defineMutex("DataError","Asymmetry") ;
  pc.defineMutex("Binning","BinningName","BinningSpec") ;

  // Process & check varargs 
  pc.process(argList) ;
  if (!pc.ok(kTRUE)) {
    return frame ;
  }

  PlotOpt o ;

  // Extract values from named arguments
  o.drawOptions = pc.getString("drawOption") ;
  o.cuts = pc.getString("cutString") ;
  if (pc.hasProcessed("Binning")) {
    o.bins = (RooAbsBinning*) pc.getObject("binning") ;
  } else if (pc.hasProcessed("BinningName")) {
    o.bins = &frame->getPlotVar()->getBinning(pc.getString("binningName")) ;
  } else if (pc.hasProcessed("BinningSpec")) {
    Double_t xlo = pc.getDouble("xlo") ;
    Double_t xhi = pc.getDouble("xhi") ;
    o.bins = new RooUniformBinning((xlo==xhi)?frame->getPlotVar()->getMin():xlo,
				   (xlo==xhi)?frame->getPlotVar()->getMax():xhi,pc.getInt("nbins")) ;
  }
  const RooAbsCategoryLValue* asymCat = (const RooAbsCategoryLValue*) pc.getObject("asymCat") ;
  o.etype = (RooAbsData::ErrorType) pc.getInt("errorType") ;
  o.histInvisible = pc.getInt("histInvisible") ;
  o.xErrorSize = pc.getDouble("xErrorSize") ;
  o.cutRange = pc.getString("cutRange",0,kTRUE) ;
  o.histName = pc.getString("histName",0,kTRUE) ;
  o.addToHistName = pc.getString("addToHistName",0,kTRUE) ;
  o.addToWgtSelf = pc.getDouble("addToWgtSelf") ;
  o.addToWgtOther = pc.getDouble("addToWgtOther") ;
  o.refreshFrameNorm = pc.getInt("refreshFrameNorm") ;
  
  if (o.addToHistName && !frame->findObject(o.addToHistName,RooHist::Class())) {
    cout << "RooTreeData::plotOn(" << GetName() << ") cannot find existing histogram " << o.addToHistName << " to add to in RooPlot" << endl ;
    return frame ;
  }

  RooPlot* ret ;
  if (!asymCat) {
    ret = plotOn(frame,o) ;
  } else {
    ret = plotAsymOn(frame,*asymCat,o) ;    
  }

  Int_t lineColor   = pc.getInt("lineColor") ;
  Int_t lineStyle   = pc.getInt("lineStyle") ;
  Int_t lineWidth   = pc.getInt("lineWidth") ;
  Int_t markerColor = pc.getInt("markerColor") ;
  Int_t markerStyle = pc.getInt("markerStyle") ;
  Size_t markerSize  = pc.getDouble("markerSize") ;
  if (lineColor!=-999) ret->getAttLine()->SetLineColor(lineColor) ;
  if (lineStyle!=-999) ret->getAttLine()->SetLineStyle(lineStyle) ;
  if (lineWidth!=-999) ret->getAttLine()->SetLineWidth(lineWidth) ;
  if (markerColor!=-999) ret->getAttMarker()->SetMarkerColor(markerColor) ;
  if (markerStyle!=-999) ret->getAttMarker()->SetMarkerStyle(markerStyle) ;
  if (markerSize!=-999) ret->getAttMarker()->SetMarkerSize(markerSize) ;

  if (pc.hasProcessed("BinningSpec")) {
    delete o.bins ;
  }

  return ret ;
}


RooPlot *RooTreeData::plotOn(RooPlot *frame, PlotOpt o) const 
{
  // Create and fill a histogram of the frame's variable and append it to the frame.
  // The frame variable must be one of the data sets dimensions.
  //
  // The plot range and the number of plot bins is determined by the parameters
  // of the plot variable of the frame (RooAbsReal::setPlotRange(), RooAbsReal::setPlotBins())
  // 
  // The optional cut string expression can be used to select the events to be plotted.
  // The cut specification may refer to any variable contained in the data set
  //
  // The drawOptions are passed to the TH1::Draw() method

  if(0 == frame) {
    cout << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return 0;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: frame does not specify a plot variable" << endl;
    return 0;
  }

  // create and fill a temporary histogram of this variable
  TString histName(GetName());
  histName.Append("_plot");
  TH1F *hist ;
  if (o.bins) {
    hist= static_cast<TH1F*>(var->createHistogram(histName.Data(), RooFit::AxisLabel("Events"), RooFit::Binning(*o.bins))) ;
  } else {
    hist= var->createHistogram(histName.Data(), "Events", 
			       frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(), frame->GetNbinsX());
  }

  // Keep track of sum-of-weights error
  hist->Sumw2() ;

  if(0 == fillHistogram(hist,RooArgList(*var),o.cuts,o.cutRange)) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: fillHistogram() failed" << endl;
    return 0;
  }

  // If frame has no predefined bin width (event density) it will be adjusted to 
  // our histograms bin width so we should force that bin width here
  Double_t nomBinWidth ;
  if (frame->getFitRangeNEvt()==0 && o.bins) {
    nomBinWidth = o.bins->averageBinWidth() ;
  } else {
    nomBinWidth = o.bins ? frame->getFitRangeBinW() : 0 ;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist,nomBinWidth,1,o.etype,o.xErrorSize);
  if(0 == graph) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: unable to create a RooHist object" << endl;
    delete hist;
    return 0;
  }  

  // If the dataset variable has a wide range than the plot variable,
  // calculate the number of entries in the dataset in the plot variable fit range
  RooAbsRealLValue* dataVar = (RooAbsRealLValue*) _vars.find(var->GetName()) ;
  Double_t nEnt(sumEntries()) ;
  if (dataVar->getMin()<var->getMin() || dataVar->getMax()>var->getMax()) {
    RooAbsData* tmp = ((RooTreeData*)this)->reduce(*var) ;
    nEnt = tmp->sumEntries() ;
    delete tmp ;
  }

  // Store the number of entries before the cut, if any was made
  if ((o.cuts && strlen(o.cuts)) || o.cutRange) {
    cout << "RooTreeData::plotOn: plotting " << hist->GetSum() << " events out of " << nEnt << " total events" << endl ;
    graph->setRawEntries(nEnt) ;
  }

  // Add self to other hist if requested
  if (o.addToHistName) {
    RooHist* otherGraph = static_cast<RooHist*>(frame->findObject(o.addToHistName,RooHist::Class())) ;

    if (!graph->hasIdenticalBinning(*otherGraph)) {
      cout << "RooTreeData::plotOn: ERROR Histogram to be added to, '" << o.addToHistName << "',has different binning" << endl ;
      delete graph ;
      return frame ;
    }

    RooHist* sumGraph = new RooHist(*graph,*otherGraph,o.addToWgtSelf,o.addToWgtOther,o.etype) ;
    delete graph ;
    graph = sumGraph ;
  }  

  // Rename graph if requested
  if (o.histName) {
    graph->SetName(o.histName) ;
  }

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);


  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);



  // cleanup
  delete hist;

  return frame;  
}




RooPlot* RooTreeData::plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat, PlotOpt o) const 
{
  // Create and fill a histogram with the asymmetry N[+] - N[-] / ( N[+] + N[-] ),
  // where N(+/-) is the number of data points with asymCat=+1 and asymCat=-1 
  // as function of the frames variable. The asymmetry category 'asymCat' must
  // have exactly 2 (or 3) states defined with index values +1,-1 (and 0)
  // 
  // The plot range and the number of plot bins is determined by the parameters
  // of the plot variable of the frame (RooAbsReal::setPlotRange(), RooAbsReal::setPlotBins())
  // 
  // The optional cut string expression can be used to select the events to be plotted.
  // The cut specification may refer to any variable contained in the data set
  //
  // The drawOptions are passed to the TH1::Draw() method

  if(0 == frame) {
    cout << ClassName() << "::" << GetName() << ":plotAsymOn: frame is null" << endl;
    return 0;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    cout << ClassName() << "::" << GetName()
	 << ":plotAsymOn: frame does not specify a plot variable" << endl;
    return 0;
  }

  // create and fill temporary histograms of this variable for each state
  TString hist1Name(GetName()),hist2Name(GetName());
  hist1Name.Append("_plot1");
  TH1F *hist1, *hist2 ;
  hist2Name.Append("_plot2");

  if (o.bins) {
    hist1= var->createHistogram(hist1Name.Data(), "Events", *o.bins) ;
    hist2= var->createHistogram(hist2Name.Data(), "Events", *o.bins) ;
  } else {
    hist1= var->createHistogram(hist1Name.Data(), "Events", 
				frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
				frame->GetNbinsX());
    hist2= var->createHistogram(hist2Name.Data(), "Events", 
				frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
				frame->GetNbinsX());
  }

  assert(0 != hist1 && 0 != hist2);

  TString cuts1,cuts2 ;
  if (o.cuts && strlen(o.cuts)) {
    cuts1 = Form("(%s)&&(%s>0)",o.cuts,asymCat.GetName());
    cuts2 = Form("(%s)&&(%s<0)",o.cuts,asymCat.GetName());
  } else {
    cuts1 = Form("(%s>0)",asymCat.GetName());
    cuts2 = Form("(%s<0)",asymCat.GetName());
  }

  if(0 == fillHistogram(hist1,RooArgList(*var),cuts1.Data()) ||
     0 == fillHistogram(hist2,RooArgList(*var),cuts2.Data())) {
    cout << ClassName() << "::" << GetName()
	 << ":plotAsymOn: createHistogram() failed" << endl;
    return 0;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist1,*hist2,0,1,o.xErrorSize);
  if(0 == graph) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: unable to create a RooHist object" << endl;
    delete hist1;
    delete hist2;
    return 0;
  }

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);

  // Rename graph if requested
  if (o.histName) {
    graph->SetName(o.histName) ;
  }

  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions);

  // cleanup
  delete hist1;
  delete hist2;

  return frame;  
}



TH1 *RooTreeData::fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts, const char* cutRange) const
{
  // Loop over columns of our tree data and fill the input histogram. Returns a pointer to the
  // input histogram, or zero in case of an error. The input histogram can be any TH1 subclass, and
  // therefore of arbitrary dimension. Variables are matched with the (x,y,...) dimensions of the input
  // histogram according to the order in which they appear in the input plotVars list.

  // Do we have a valid histogram to use?
  if(0 == hist) {
    cout << ClassName() << "::" << GetName() << ":fillHistogram: no valid histogram to fill" << endl;
    return 0;
  }

  // Check that the number of plotVars matches the input histogram's dimension
  Int_t hdim= hist->GetDimension();
  if(hdim != plotVars.getSize()) {
    cout << ClassName() << "::" << GetName() << ":fillHistogram: plotVars has the wrong dimension" << endl;
    return 0;
  }

  // Check that the plot variables are all actually RooAbsReal's and print a warning if we do not
  // explicitly depend on one of them. Clone any variables that we do not contain directly and
  // redirect them to use our event data.
  RooArgSet plotClones,localVars;
  for(Int_t index= 0; index < plotVars.getSize(); index++) {
    const RooAbsArg *var= plotVars.at(index);
    const RooAbsReal *realVar= dynamic_cast<const RooAbsReal*>(var);
    if(0 == realVar) {
      cout << ClassName() << "::" << GetName() << ":fillHistogram: cannot plot variable \"" << var->GetName()
	   << "\" of type " << var->ClassName() << endl;
      return 0;
    }
    RooAbsArg *found= _vars.find(realVar->GetName());
    if(!found) {
      RooAbsArg *clone= plotClones.addClone(*realVar,kTRUE); // do not complain about duplicates
      assert(0 != clone);
      if(!clone->dependsOn(_vars)) {
	cout << ClassName() << "::" << GetName()
	     << ":fillHistogram: WARNING: data does not contain variable: " << realVar->GetName() << endl;
      }
      else {
	clone->recursiveRedirectServers(_vars);
      }
      localVars.add(*clone);
    }
    else {
      localVars.add(*found);
    }
  }

  // Create selection formula if selection cuts are specified
  RooFormula* select = 0;
  if(0 != cuts && strlen(cuts)) {
    select=new RooFormula(cuts,cuts,_vars);
    if (!select || !select->ok()) {
      cout << ClassName() << "::" << GetName() << ":fillHistogram: invalid cuts \"" << cuts << "\"" << endl;
      delete select;
      return 0 ;
    }
  }
  
  // Lookup each of the variables we are binning in our tree variables
  const RooAbsReal *xvar = 0;
  const RooAbsReal *yvar = 0;
  const RooAbsReal *zvar = 0;
  switch(hdim) {
  case 3:
    zvar= dynamic_cast<RooAbsReal*>(localVars.find(plotVars.at(2)->GetName()));
    assert(0 != zvar);
    // fall through to next case...
  case 2:
    yvar= dynamic_cast<RooAbsReal*>(localVars.find(plotVars.at(1)->GetName()));
    assert(0 != yvar);
    // fall through to next case...
  case 1:
    xvar= dynamic_cast<RooAbsReal*>(localVars.find(plotVars.at(0)->GetName()));
    assert(0 != xvar);
    break;
  default:
    cout << ClassName() << "::" << GetName() << ":fillHistogram: cannot fill histogram with "
	 << hdim << " dimensions" << endl;
    break;
  }

  // Parse cutRange specification
  vector<string> cutVec ;
  if (cutRange && strlen(cutRange)>0) {
    if (strchr(cutRange,',')==0) {
      cutVec.push_back(cutRange) ;
    } else {
      char* buf = new char[strlen(cutRange)+1] ;
      strcpy(buf,cutRange) ;
      const char* oneRange = strtok(buf,",") ;
      while(oneRange) {
	cutVec.push_back(oneRange) ;
	oneRange = strtok(0,",") ;
      }
      delete[] buf ;
    }
  }

  // Loop over events and fill the histogram
  Int_t nevent= (Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber= _tree->GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);

    // Apply expression based selection criteria
    if (select && select->eval()==0) {
      continue ;
    }


    // Apply range based selection criteria
    Bool_t selectByRange = kTRUE ;
    if (cutRange) {
      _iterator->Reset() ;
      RooAbsArg* arg ;
      while((arg=(RooAbsArg*)_iterator->Next())) {
	Bool_t selectThisArg = kFALSE ;
	UInt_t icut ;
	for (icut=0 ; icut<cutVec.size() ; icut++) {
	  if (arg->inRange(cutVec[icut].c_str())) {
	    selectThisArg = kTRUE ;
	    break ;
	  }
	}
	if (!selectThisArg) {
	  selectByRange = kFALSE ;
	  break ;
	}
      }
    }

    if (!selectByRange) {
      // Go to next event in loop over events
      continue ;
    }

    Int_t bin(0);
    switch(hdim) {
    case 1:
      bin= hist->FindBin(xvar->getVal());
      hist->Fill(xvar->getVal(),weight()) ;
      break;
    case 2:
      bin= hist->FindBin(xvar->getVal(),yvar->getVal());
      static_cast<TH2*>(hist)->Fill(xvar->getVal(),yvar->getVal(),weight()) ;
      break;
    case 3:
      bin= hist->FindBin(xvar->getVal(),yvar->getVal(),zvar->getVal());
      static_cast<TH3*>(hist)->Fill(xvar->getVal(),yvar->getVal(),zvar->getVal(),weight()) ;
      break;
    default:
      assert(hdim < 3);
      break;
    }

    //cout << "hdim = " << hdim << " bin = " << bin << endl ;

    Double_t error2 = TMath::Power(hist->GetBinError(bin),2)-TMath::Power(weight(),2)  ;
    Double_t we = weightError(RooAbsData::SumW2) ;
    if (we==0) we = weight() ;
    error2 += TMath::Power(we,2) ;
    //hist->AddBinContent(bin,weight());
    hist->SetBinError(bin,sqrt(error2)) ;
  }

  if(0 != select) delete select;

  return hist;
}



Roo1DTable* RooTreeData::table(const RooAbsCategory& cat, const char* cuts, const char* /*opts*/) const
{
  // Create and fill a 1-dimensional table for given category column
  // This functions is the equivalent of plotOn() for category dimensions. 
  //
  // The optional cut string expression can be used to select the events to be tabulated
  // The cut specification may refer to any variable contained in the data set
  //
  // The option string is currently not used


  // First see if var is in data set 
  RooAbsCategory* tableVar = (RooAbsCategory*) _vars.find(cat.GetName()) ;
  RooArgSet *tableSet = 0;
  Bool_t ownPlotVar(kFALSE) ;
  if (!tableVar) {
    if (!cat.dependsOn(_vars)) {
      cout << "RooTreeData::Table(" << GetName() << "): Argument " << cat.GetName() 
	   << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    tableSet = (RooArgSet*) RooArgSet(cat).snapshot(kTRUE) ;
    if (!tableSet) {
      cout << "RooTreeData::table(" << GetName() << ") Couldn't deep-clone table category, abort." << endl ;
      return 0 ;
    }
    tableVar = (RooAbsCategory*) tableSet->find(cat.GetName()) ;
    ownPlotVar = kTRUE ;    

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    tableVar->recursiveRedirectServers(_vars) ;
  }

  TString tableName(GetName()) ;
  if (cuts && strlen(cuts)) {
    tableName.Append("(") ;
    tableName.Append(cuts) ;
    tableName.Append(")") ;    
  }
  Roo1DTable* table = tableVar->createTable(tableName) ;

  // Make cut selector if cut is specified
  RooFormulaVar* cutVar = 0;
  if (cuts && strlen(cuts)) {
    cutVar = new RooFormulaVar("cutVar",cuts,_vars) ;
  }
  
  // Dump contents   
  Int_t nevent= (Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=_tree->GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);

    if (cutVar && cutVar->getVal()==0) continue ;
    
    table->fill(*tableVar,weight()) ;
  }

  if (ownPlotVar) delete tableSet ;
  if (cutVar) delete cutVar ;

  return table ;
}



Double_t RooTreeData::moment(RooRealVar &var, Double_t order, Double_t offset, const char* cutSpec, const char* cutRange) const
{
  // Lookup variable in dataset
  RooRealVar *varPtr= (RooRealVar*) _vars.find(var.GetName());
  if(0 == varPtr) {
    cout << "RooDataSet::moment(" << GetName() << ") ERROR: unknown variable: " << var.GetName() << endl ;
    return 0;
  }

  // Check if found variable is of type RooRealVar
  if (!dynamic_cast<RooRealVar*>(varPtr)) {
    cout << "RooDataSet::moment(" << GetName() << ") ERROR: variable " << var.GetName() << " is not of type RooRealVar" << endl ;
    return 0;
  }

  // Check if dataset is not empty
  if(sumEntries() == 0.) {
    cout << "RooDataSet::moment(" << GetName() << ") WARNING: empty dataset" << endl ;
    return 0;
  }

  // Setup RooFormulaVar for cutSpec if it is present
  RooFormula* select = 0 ;
  if (cutSpec) {
    select = new RooFormula("select",cutSpec,*get()) ;
  }


  // Calculate requested moment
  Double_t sum(0);
  const RooArgSet* vars ;
  for(Int_t index= 0; index < numEntries(); index++) {
    vars = get(index) ;
    if (select && select->eval()==0) continue ;
    if (cutRange && vars->allInRange(cutRange)) continue ;
    
    sum+= weight() * TMath::Power(varPtr->getVal() - offset,order);
  }
  return sum/sumEntries();
}



RooRealVar* RooTreeData::meanVar(RooRealVar &var, const char* cutSpec, const char* cutRange) const
{
  // Create a new variable with appropriate strings. The error is calculated as
  // RMS/Sqrt(N) which is generally valid.

  // Create holder variable for mean
  TString name(var.GetName()),title("Mean of ") ;
  name.Append("Mean");
  title.Append(var.GetTitle());
  RooRealVar *mean= new RooRealVar(name,title,0) ;
  mean->setConstant(kFALSE) ;

  // Adjust plot label
  TString label("<") ;
  label.Append(var.getPlotLabel());
  label.Append(">");
  mean->setPlotLabel(label.Data());

  // fill in this variable's value and error
  Double_t meanVal=moment(var,1,0,cutSpec,cutRange) ;
  Double_t N(sumEntries(cutSpec,cutRange)) ;

  Double_t rmsVal= sqrt(moment(var,2,meanVal,cutSpec,cutRange)*N/(N-1));
  mean->setVal(meanVal) ;
  mean->setError(N > 0 ? rmsVal/sqrt(N) : 0);

  return mean;
}



RooRealVar* RooTreeData::rmsVar(RooRealVar &var, const char* cutSpec, const char* cutRange) const
{
  // Create a new variable with appropriate strings. The error is calculated as
  // RMS/(2*Sqrt(N)) which is only valid if the variable has a Gaussian distribution.

  // Create RMS value holder
  TString name(var.GetName()),title("RMS of ") ;
  name.Append("RMS");
  title.Append(var.GetTitle());
  RooRealVar *rms= new RooRealVar(name,title,0) ;
  rms->setConstant(kFALSE) ;

  // Adjust plot label
  TString label(var.getPlotLabel());
  label.Append("_{RMS}");
  rms->setPlotLabel(label);

  // Fill in this variable's value and error
  Double_t meanVal(moment(var,1)) ;
  Double_t N(sumEntries());
  Double_t rmsVal= sqrt(moment(var,2,meanVal,cutSpec,cutRange)*N/(N-1));
  rms->setVal(rmsVal) ;
  rms->setError(rmsVal/sqrt(2*N));

  return rms;
}


Bool_t RooTreeData::getRange(RooRealVar& var, Double_t& lowest, Double_t& highest) const 
{
  // Lookup variable in dataset
  RooRealVar *varPtr= (RooRealVar*) _vars.find(var.GetName());
  if(0 == varPtr) {
    cout << "RooDataSet::getRange(" << GetName() << ") ERROR: unknown variable: " << var.GetName() << endl ;
    return kTRUE;
  }

  // Check if found variable is of type RooRealVar
  if (!dynamic_cast<RooRealVar*>(varPtr)) {
    cout << "RooDataSet::getRange(" << GetName() << ") ERROR: variable " << var.GetName() << " is not of type RooRealVar" << endl ;
    return kTRUE;
  }

  // Check if dataset is not empty
  if(sumEntries() == 0.) {
    cout << "RooDataSet::getRange(" << GetName() << ") WARNING: empty dataset" << endl ;
    return kTRUE;
  }

  // Look for highest and lowest value 
  lowest = RooNumber::infinity ;
  highest = -RooNumber::infinity ;
  for (Int_t i=0 ; i<numEntries() ; i++) {
    get(i) ;
    if (varPtr->getVal()<lowest) {
      lowest = varPtr->getVal() ;
    }
    if (varPtr->getVal()>highest) {
      highest = varPtr->getVal() ;
    }
  }  

  return kFALSE ;
}



RooPlot* RooTreeData::statOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2, 
			    const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5, 
			    const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  // Add a box with statistics information to the specified frame. By default a box with the
  // event count, mean and rms of the plotted variable is added.
  //
  // The following optional named arguments are accepted
  //
  //   What(const char* whatstr)          -- Controls what is printed: "N" = count, "M" is mean, "R" is RMS.
  //   Format(const char* optStr)         -- Classing [arameter formatting options, provided for backward compatibility
  //   Format(const char* what,...)       -- Parameter formatting options, details given below
  //   Label(const chat* label)           -- Add header label to parameter box
  //   Layout(Double_t xmin,              -- Specify relative position of left,right side of box and top of box. Position of 
  //       Double_t xmax, Double_t ymax)     bottom of box is calculated automatically from number lines in box
  //   Cut(const char* expression)        -- Apply given cut expression to data when calculating statistics
  //   CutRange(const char* rangeName)    -- Only consider events within given range when calculating statistics. Multiple
  //                                         CutRange() argument may be specified to combine ranges
  //
  // The Format(const char* what,...) has the following structure
  //
  //   const char* what          -- Controls what is shown. "N" adds name, "E" adds error, 
  //                                "A" shows asymmetric error, "U" shows unit, "H" hides the value
  //   FixedPrecision(int n)     -- Controls precision, set fixed number of digits
  //   AutoPrecision(int n)      -- Controls precision. Number of shown digits is calculated from error 
  //                                + n specified additional digits (1 is sensible default)
  //   VerbatimName(Bool_t flag) -- Put variable name in a \verb+   + clause.
  //

  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;

  // Select the pdf-specific commands 
  RooCmdConfig pc(Form("RooTreeData::statOn(%s)",GetName())) ;
  pc.defineString("what","What",0,"MNR") ;
  pc.defineString("label","Label",0,"") ;
  pc.defineDouble("xmin","Layout",0,0.65) ;
  pc.defineDouble("xmax","Layout",1,0.99) ;
  pc.defineInt("ymaxi","Layout",0,Int_t(0.95*10000)) ;
  pc.defineString("formatStr","Format",0,"NELU") ;
  pc.defineInt("sigDigit","Format",0,2) ;
  pc.defineInt("dummy","FormatArgs",0,0) ;
  pc.defineString("cutRange","CutRange",0,"",kTRUE) ;
  pc.defineString("cutString","CutSpec",0,"") ;
  pc.defineMutex("Format","FormatArgs") ;

  // Process and check varargs 
  pc.process(cmdList) ;
  if (!pc.ok(kTRUE)) {
    return frame ;
  }

  const char* label = pc.getString("label") ;
  Double_t xmin = pc.getDouble("xmin") ;
  Double_t xmax = pc.getDouble("xmax") ;
  Double_t ymax = pc.getInt("ymaxi") / 10000. ;
  const char* formatStr = pc.getString("formatStr") ;
  Int_t sigDigit = pc.getInt("sigDigit") ;  
  const char* what = pc.getString("what") ;

  const char* cutSpec = pc.getString("cutString",0,kTRUE) ;
  const char* cutRange = pc.getString("cutRange",0,kTRUE) ;

  if (pc.hasProcessed("FormatArgs")) {
    RooCmdArg* formatCmd = static_cast<RooCmdArg*>(cmdList.FindObject("FormatArgs")) ;
    return statOn(frame,what,label,0,0,xmin,xmax,ymax,cutSpec,cutRange,formatCmd) ;
  } else {
    return statOn(frame,what,label,sigDigit,formatStr,xmin,xmax,ymax,cutSpec,cutRange) ;
  }
}



RooPlot* RooTreeData::statOn(RooPlot* frame, const char* what, const char *label, Int_t sigDigits,
			     Option_t *options, Double_t xmin, Double_t xmax, Double_t ymax, 
			     const char* cutSpec, const char* cutRange, const RooCmdArg* formatCmd) 
{

  Bool_t showLabel= (label != 0 && strlen(label) > 0);

  TString whatStr(what) ;
  whatStr.ToUpper() ;
  Bool_t showN = whatStr.Contains("N") ;
  Bool_t showR = whatStr.Contains("R") ;
  Bool_t showM = whatStr.Contains("M") ;
  Int_t nPar= 0;
  if (showN) nPar++ ;
  if (showR) nPar++ ;
  if (showM) nPar++ ;

  // calculate the box's size
  Double_t dy(0.06), ymin(ymax-nPar*dy);
  if(showLabel) ymin-= dy;

  // create the box and set its options
  TPaveText *box= new TPaveText(xmin,ymax,xmax,ymin,"BRNDC");
  if(!box) return 0;
  box->SetFillColor(0);
  box->SetBorderSize(1);
  box->SetTextAlign(12);
  box->SetTextSize(0.04F);
  box->SetFillStyle(1001);

  // add formatted text for each statistic
  TText *text = 0;
  RooRealVar N("N","Number of Events",sumEntries(cutSpec,cutRange));
  N.setPlotLabel("Entries") ;
  RooRealVar *mean= meanVar(*(RooRealVar*)frame->getPlotVar(),cutSpec,cutRange);
  mean->setPlotLabel("Mean") ;
  RooRealVar *rms= rmsVar(*(RooRealVar*)frame->getPlotVar(),cutSpec,cutRange);
  rms->setPlotLabel("RMS") ;
  TString *rmsText, *meanText, *NText ;
  if (options) {
    rmsText= rms->format(sigDigits,options);
    meanText= mean->format(sigDigits,options);
    NText= N.format(sigDigits,options);
  } else {
    rmsText= rms->format(*formatCmd);
    meanText= mean->format(*formatCmd);
    NText= N.format(*formatCmd);
  }
  if (showR) text= box->AddText(rmsText->Data());
  if (showM) text= box->AddText(meanText->Data());
  if (showN) text= box->AddText(NText->Data());

  // cleanup heap memory
  delete NText;
  delete meanText;
  delete rmsText;
  delete mean;
  delete rms;

  // add the optional label if specified
  if(showLabel) text= box->AddText(label);

  frame->addObject(box) ;
  return frame ;
}


Int_t RooTreeData::numEntries(Bool_t) const 
{ 
  return (Int_t)GetEntries() ; 
}


void RooTreeData::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this dataset to the specified output stream.
  //
  //   Standard: number of entries
  //      Shape: list of variables we define & were generated with

  oneLinePrint(os,*this);
  if(opt >= Standard) {
    if (isWeighted()) {
      os << indent << "  Contains " << numEntries() << " entries with a total weight of " << sumEntries() << endl;
    } else {
      os << indent << "  Contains " << numEntries() << " entries" << endl;
    }
    if(opt >= Shape) {
      os << indent << "  Defines ";
      TString deeper(indent);
      deeper.Append("  ");
      _vars.printToStream(os,Standard,deeper);
      os << indent << "  Caches ";
      _cachedVars.printToStream(os,Standard,deeper);
      
      if(_truth.getSize() > 0) {
	os << indent << "  Generated with ";
	_truth.printToStream(os,Shape,deeper);
      }
    }
  }
}






