/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
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

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooTreeData is the abstract base class for data collection that
// use a TTree as internal storage mechanism
// END_HTML
//

#include "RooFit.h"
#include "RooMsgService.h"

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

using namespace std ;

ClassImp(RooTreeData)
;

Int_t RooTreeData::_defTreeBufSize = 4096 ;


//________________________________________________________________
//   Interface functions to TTree

//_____________________________________________________________________________
Int_t RooTreeData::Scan(const char* varexp, const char* selection, Option_t* option, 
		    Int_t nentries, Int_t firstentry) 
{
   //   Interface function to TTree::Scan
   return _tree->Scan(varexp,selection,option,nentries,firstentry) ;
}


//_____________________________________________________________________________
Int_t RooTreeData::ScanCache(const char* varexp, const char* selection, Option_t* option, 
		    Int_t nentries, Int_t firstentry) 
{
   // Interface function to TTree::Scan
   return _cacheTree->Scan(varexp,selection,option,nentries,firstentry) ;
}
 

//_____________________________________________________________________________
Stat_t RooTreeData::GetEntries() const
{
   // Interface function to TTree::GetEntries
   return _tree->GetEntries() ;
}
 

//_____________________________________________________________________________
void RooTreeData::Reset(Option_t* option)
{
   // Interface function to TTree::Reset
   _tree->Reset(option) ;
}
 

//_____________________________________________________________________________
void RooTreeData::treePrint()
{
   // Interface function to TTree::Print
   _tree->Print();
}
 

//_____________________________________________________________________________
Int_t RooTreeData::Fill()
{
   // Interface function to TTree::Fill
   return _tree->Fill() ;
}
 

//_____________________________________________________________________________
Int_t RooTreeData::GetEntry(Int_t entry, Int_t getall)
{
   // Interface function to TTree::GetEntry
   Int_t ret1 = _tree->GetEntry(entry,getall) ; 
   if (!ret1) return 0 ;
   _cacheTree->GetEntry(entry,getall) ; 
   return ret1 ;
}


//_____________________________________________________________________________
void RooTreeData::Draw(Option_t* opt) 
{
   // Interface function to TTree::Draw()
  return _tree->Draw(opt) ;
}



//_____________________________________________________________________________
Long64_t RooTreeData::Draw(const char* varexp, const char* selection, Option_t* option, Long64_t nentries, Long64_t firstentry)
{
  // Interface function to TTree::Draw()
  return _tree->Draw(varexp,selection,option,nentries,firstentry) ;
}



//________________________________________________________________
//   Interface functions to TTree


//_____________________________________________________________________________
RooTreeData::RooTreeData() 
{
  // Default constructor
  RooTrace::create(this) ; 
  _defCtor = kTRUE ;  
  _cacheTree = 0 ;
  _tree = 0 ;
}



//_____________________________________________________________________________
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



//_____________________________________________________________________________
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



//_____________________________________________________________________________
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
    coutE(InputArguments) << "RooTreeData::RooTreeData(" << GetName() << ") Couldn't deep-clone cut variable, abort." << endl ;
    RooErrorHandler::softAbort() ;
  }

  RooFormulaVar* cloneVar = (RooFormulaVar*) tmp->find(cutVar.GetName()) ;
  cloneVar->attachDataSet(*this) ;

  loadValues(t,cloneVar);

  delete tmp ;
}



//_____________________________________________________________________________
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
    coutE(InputArguments) << "RooTreeData::RooTreeData(" << GetName() << ") Couldn't deep-clone cut variable, abort." << endl ;
    RooErrorHandler::softAbort() ;
  }
  RooFormulaVar* cloneVar = (RooFormulaVar*) tmp->find(cutVar.GetName()) ;
  cloneVar->attachDataSet(*this) ;

  loadValues(t,cloneVar);

  delete tmp ;
}



//_____________________________________________________________________________
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
      coutE(InputArguments) << "RooTreeData::RooTreeData(" << GetName() << ") Couldn't deep-clone cut variable, abort." << endl ;
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




//_____________________________________________________________________________
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



//_____________________________________________________________________________
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



//_____________________________________________________________________________
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



//_____________________________________________________________________________
void RooTreeData::createTree(const char* name, const char* title)
{
  // Create TTree object that lives in memory, independent of current
  // location of gDirectory

  TString pwd(gDirectory->GetPath()) ;
  TString memDir(gROOT->GetName()) ;
  memDir.Append(":/") ;
  Bool_t notInMemNow= (pwd!=memDir) ;

  // cout << "RooTreeData::createTree pwd=" << pwd << " memDir=" << memDir << " notInMemNow = " << (notInMemNow?"T":"F") << endl ;

  if (notInMemNow) {
    gDirectory->cd(memDir) ;
  }

  if (!_tree) {
    _tree = new TTree(name,title) ;
    _tree->SetDirectory(0) ;
  }
  if (!_cacheTree) {
    _cacheTree = new TTree(name,title) ;
    _cacheTree->SetDirectory(0) ;
  }
  gDirectory->RecursiveRemove(_tree) ;
  gDirectory->RecursiveRemove(_cacheTree) ;

  if (notInMemNow) {
    gDirectory->cd(pwd) ;
  }
  
}



//_____________________________________________________________________________
RooTreeData::~RooTreeData()
{
  // Destructor

  RooTrace::destroy(this) ;

  delete _tree ;
  delete _cacheTree ;
}



//_____________________________________________________________________________
void RooTreeData::initialize() 
{
  // One-time initialization common to all constructor forms.  Attach
  // variables of internal ArgSet to the corresponding TTree branches

  // Recreate (empty) cache tree
  createTree(GetName(),GetTitle()) ;

  // Attach each variable to the dataset
  _iterator->Reset() ;  
  RooAbsArg *var;
  while((0 != (var= (RooAbsArg*)_iterator->Next()))) {
    var->attachToTree(*_tree,_defTreeBufSize) ;
  }
}



//_____________________________________________________________________________
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



//_____________________________________________________________________________
void RooTreeData::loadValues(const char *filename, const char *treename,
			    RooFormulaVar* cutVar) 
{
  // Load TTree name 'treename' from file 'filename' and pass call
  // to loadValue(TTree*,...)

  TFile *file= (TFile*)gROOT->GetListOfFiles()->FindObject(filename);
  if(!file) file= new TFile(filename);
  if(!file) {
    coutE(InputArguments) << "RooTreeData::loadValues: unable to open " << filename << endl;
  }
  else {
    TTree* tree2= (TTree*)gDirectory->Get(treename);
    loadValues(tree2,cutVar);
  }
}



//_____________________________________________________________________________
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
	//cout << "arg " << arg->GetName() << " is not valid" << endl ;
	//arg->Print("v") ;
	allValid=kFALSE ;
	break ;
      }
    }
    //cout << "RooTreeData::loadValues(" << GetName() << ") allValid = " << (allValid?"T":"F") << endl ;
    if (!allValid) continue ;

    _cachedVars = t->_cachedVars ;
    Fill() ;
   }
  
  SetTitle(t->GetTitle());
}



//_____________________________________________________________________________
void RooTreeData::loadValues(const TTree *t, RooFormulaVar* select, const char* /*rangeName*/, Int_t /*nStart*/, Int_t /*nStop*/) 
{
  // Load values from tree 't' into this data collection, optionally
  // selecting events using 'select' RooFormulaVar
  //
  // The source tree 't' is first clone as not disturb its branch
  // structure when retrieving information from it.

  // Clone source tree
  // WVE Clone() crashes on trees, CloneTree() crashes on tchains :-(

  // Change directory to memory dir before cloning tree to avoid ROOT errors
  TString pwd(gDirectory->GetPath()) ;
  TString memDir(gROOT->GetName()) ;
  memDir.Append(":/") ;
  Bool_t notInMemNow= (pwd!=memDir) ;

  if (notInMemNow) {
    gDirectory->cd(memDir) ;
  }

  TTree* tClone ;
  if (dynamic_cast<const TChain*>(t)) {
    tClone = (TTree*) t->Clone() ; 
  } else {
    tClone = ((TTree*)t)->CloneTree() ;
  }

  // Change directory back to original directory
  tClone->SetDirectory(0) ;

  if (notInMemNow) {
    gDirectory->cd(pwd) ;
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
    coutI(Eval) << "RooTreeData::loadValues(" << GetName() << ") Ignored " << numInvalid << " out of range events" << endl ;
  }
  
  SetTitle(t->GetTitle());

  delete sourceIter ;
  delete sourceArgSet ;
  delete tClone ;
}



//_____________________________________________________________________________
void RooTreeData::dump() 
{
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



//_____________________________________________________________________________
void RooTreeData::resetCache() 
{
  // Remove tree with values of cached observables
  // and clear list of cached observables

  // Empty list of cached functions
  _cachedVars.removeAll() ;

  // Delete & recreate cache tree 
  delete _cacheTree ;
  _cacheTree = 0 ;
  createTree(GetName(),GetTitle()) ;

  return ;
}



//_____________________________________________________________________________
Bool_t RooTreeData::changeObservableName(const char* from, const char* to) 
{
  // Change name of internal observable named 'from' into 'to'

  // Find observable to be changed
  RooAbsArg* var = _vars.find(from) ;

  // Check that we found it
  if (!var) {
    coutE(InputArguments) << "RooTreeData::changeObservableName(" << GetName() << " no observable " << from << " in this dataset" << endl ;
    return kTRUE ;
  }

  // Process name change
  TString oldBranchName = var->cleanBranchName() ;
  var->SetName(to) ;  

  // Change the branch name as well 
  if (_tree->GetBranch(oldBranchName.Data())) {

    // Simple case varName = branchName
    _tree->GetBranch(oldBranchName.Data())->SetName(var->cleanBranchName().Data()) ;

    // Process any error branch if existing
    if (_tree->GetBranch(Form("%s_err",oldBranchName.Data()))) {
      _tree->GetBranch(Form("%s_err",oldBranchName.Data()))->SetName(Form("%s_err",var->cleanBranchName().Data())) ;
    }
    if (_tree->GetBranch(Form("%s_aerr_lo",oldBranchName.Data()))) {
      _tree->GetBranch(Form("%s_aerr_lo",oldBranchName.Data()))->SetName(Form("%s_aerr_lo",var->cleanBranchName().Data())) ;
    }
    if (_tree->GetBranch(Form("%s_aerr_hi",oldBranchName.Data()))) {
      _tree->GetBranch(Form("%s_aerr_hi",oldBranchName.Data()))->SetName(Form("%s_aerr_hi",var->cleanBranchName().Data())) ;
    }

  } else {

    // Native category case branchNames = varName_idx and varName_lbl
    if (_tree->GetBranch(Form("%s_idx",oldBranchName.Data()))) {
      _tree->GetBranch(Form("%s_idx",oldBranchName.Data()))->SetName(Form("%s_idx",var->cleanBranchName().Data())) ;
    }
    if (_tree->GetBranch(Form("%s_lbl",oldBranchName.Data()))) {
      _tree->GetBranch(Form("%s_lbl",oldBranchName.Data()))->SetName(Form("%s_lb",var->cleanBranchName().Data())) ;
    }
    
  }
  
  // If variable have default tree branch association (varName=vranchName) now make
  // branch association explicit as default will no longer hold
//   if (!var->getStringAttribute("BranchName")) {
//     var->setStringAttribute("BranchName",from) ;
//   }

  return kFALSE ;
}



//_____________________________________________________________________________
void RooTreeData::setArgStatus(const RooArgSet& set, Bool_t active) 
{
  // Activate or deactivate the branch status of the TTree branch associated
  // with the given set of dataset observables

  TIterator* iter = set.createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    RooAbsArg* depArg = _vars.find(arg->GetName()) ;
    if (!depArg) {
      coutE(InputArguments) << "RooTreeData::setArgStatus(" << GetName() 
			    << ") dataset doesn't contain variable " << arg->GetName() << endl ;
      continue ;
    }
    depArg->setTreeBranchStatus(*_tree,active) ;
  }
  delete iter ;
}



//_____________________________________________________________________________
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



//_____________________________________________________________________________
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



//_____________________________________________________________________________
RooAbsArg* RooTreeData::addColumn(RooAbsArg& newVar, Bool_t adjustRange)
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
  //       this function should be used. 

  checkInit() ;

  // Create a fundamental object of the right type to hold newVar values
  RooAbsArg* valHolder= newVar.createFundamental();
  // Sanity check that the holder really is fundamental
  if(!valHolder->isFundamental()) {
    coutE(InputArguments) << GetName() << "::addColumn: holder argument is not fundamental: \""
	 << valHolder->GetName() << "\"" << endl;
    return 0;
  }

  // Clone variable and attach to cloned tree 
  RooArgSet* newVarCloneList = (RooArgSet*) RooArgSet(newVar).snapshot() ;  
  if (!newVarCloneList) {
    coutE(InputArguments) << "RooTreeData::addColumn(" << GetName() << ") Couldn't deep-clone variable to add, abort." << endl ;
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


  if (adjustRange) {
    // Set range of valHolder to (just) bracket all values stored in the dataset
    Double_t vlo,vhi ;
    RooRealVar* rrvVal = dynamic_cast<RooRealVar*>(valHolder) ;
    if (rrvVal) {
      getRange(*rrvVal,vlo,vhi,0.05) ;
      rrvVal->setRange(vlo,vhi) ;  
    }
  }

  delete newVarCloneList;  
  return valHolder ;
}



//_____________________________________________________________________________
RooArgSet* RooTreeData::addColumns(const RooArgList& varList)
{
  // Utility function to add multiple columns in one call
  // See addColumn() for details

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
      coutE(InputArguments) << GetName() << "::addColumn: holder argument is not fundamental: \""
	   << valHolder->GetName() << "\"" << endl;
      return 0;
    }
    
    // Clone variable and attach to cloned tree 
    RooArgSet* newVarCloneList = (RooArgSet*) RooArgSet(*var).snapshot() ;  
    if (!newVarCloneList) {
      coutE(InputArguments) << "RooTreeData::RooTreeData(" << GetName() << ") Couldn't deep-clone variable " << var->GetName() << ", abort." << endl ;
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




//_____________________________________________________________________________
TList* RooTreeData::split(const RooAbsCategory& splitCat) const
{
  // Split dataset into subsets based on states of given splitCat in this dataset.
  // A TList of RooDataSets is returned in which each RooDataSet is named
  // after the state name of splitCat of which it contains the dataset subset.
  // The observables splitCat itself is no longer present in the sub datasets.
  // This method only creates datasets for states which have at least one entry
  // The caller takes ownership of the returned list and its contents

  // Sanity check
  if (!splitCat.dependsOn(*get())) {
    coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName() 
	 << " doesn't depend on any variable in this dataset" << endl ;
    return 0 ;
  }

  // Clone splitting category and attach to self
  RooAbsCategory* cloneCat =0;
  RooArgSet* cloneSet = 0;
  if (splitCat.isDerived()) {
    cloneSet = (RooArgSet*) RooArgSet(splitCat).snapshot(kTRUE) ;
    if (!cloneSet) {
      coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") Couldn't deep-clone splitting category, abort." << endl ;
      return 0 ;
    }
    cloneCat = (RooAbsCategory*) cloneSet->find(splitCat.GetName()) ;
    cloneCat->attachDataSet(*this) ;
  } else {
    cloneCat = dynamic_cast<RooAbsCategory*>(get()->find(splitCat.GetName())) ;
    if (!cloneCat) {
      coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName() 
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



//_____________________________________________________________________________
RooPlot* RooTreeData::plotOn(RooPlot* frame, const RooLinkedList& argList) const
{
  // Plot dataset on specified frame. By default an unbinned dataset will use the default binning of
  // the target frame. A binned dataset will by default retain its intrinsic binning.
  //
  // The following optional named arguments can be used to modify the default behavior
  //
  // Data representation options
  // ---------------------------
  // Asymmetry(const RooCategory& c) -- Show the asymmetry of the data in given two-state category [F(+)-F(-)] / [F(+)+F(-)]. 
  //                                    Category must have two states with indices -1 and +1 or three states with indeces -1,0 and +1.
  // Efficiency(const RooCategory& c)-- Show the efficiency F(acc)/[F(acc)+F(rej)]. Category must have two states with indices 0 and 1
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
  // FillStyle(Int_t style)          -- Select fill style, default is filled. 
  // FillColor(Int_t color)          -- Select fill color by ROOT color code
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
  pc.defineObject("effCat","Efficiency",0) ;
  pc.defineInt("lineColor","LineColor",0,-999) ;
  pc.defineInt("lineStyle","LineStyle",0,-999) ;
  pc.defineInt("lineWidth","LineWidth",0,-999) ;
  pc.defineInt("markerColor","MarkerColor",0,-999) ;
  pc.defineInt("markerStyle","MarkerStyle",0,-999) ;
  pc.defineDouble("markerSize","MarkerSize",0,-999) ;
  pc.defineInt("fillColor","FillColor",0,-999) ;
  pc.defineInt("fillStyle","FillStyle",0,-999) ;
  pc.defineInt("errorType","DataError",0,(Int_t)RooAbsData::Poisson) ;
  pc.defineInt("histInvisible","Invisible",0,0) ;
  pc.defineInt("refreshFrameNorm","RefreshNorm",0,1) ;
  pc.defineString("addToHistName","AddTo",0,"") ;
  pc.defineDouble("addToWgtSelf","AddTo",0,1.) ;
  pc.defineDouble("addToWgtOther","AddTo",1,1.) ;
  pc.defineDouble("xErrorSize","XErrorSize",0,1.) ;
  pc.defineMutex("DataError","Asymmetry","Efficiency") ;
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
  const RooAbsCategoryLValue* effCat = (const RooAbsCategoryLValue*) pc.getObject("effCat") ;
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
    coutE(InputArguments) << "RooTreeData::plotOn(" << GetName() << ") cannot find existing histogram " << o.addToHistName << " to add to in RooPlot" << endl ;
    return frame ;
  }

  RooPlot* ret ;
  if (!asymCat && !effCat) {
    ret = plotOn(frame,o) ;
  } else if (asymCat) {
    ret = plotAsymOn(frame,*asymCat,o) ;    
  } else {
    ret = plotEffOn(frame,*effCat,o) ;    
  }

  Int_t lineColor   = pc.getInt("lineColor") ;
  Int_t lineStyle   = pc.getInt("lineStyle") ;
  Int_t lineWidth   = pc.getInt("lineWidth") ;
  Int_t markerColor = pc.getInt("markerColor") ;
  Int_t markerStyle = pc.getInt("markerStyle") ;
  Size_t markerSize  = pc.getDouble("markerSize") ;
  Int_t fillColor = pc.getInt("fillColor") ;
  Int_t fillStyle = pc.getInt("fillStyle") ;
  if (lineColor!=-999) ret->getAttLine()->SetLineColor(lineColor) ;
  if (lineStyle!=-999) ret->getAttLine()->SetLineStyle(lineStyle) ;
  if (lineWidth!=-999) ret->getAttLine()->SetLineWidth(lineWidth) ;
  if (markerColor!=-999) ret->getAttMarker()->SetMarkerColor(markerColor) ;
  if (markerStyle!=-999) ret->getAttMarker()->SetMarkerStyle(markerStyle) ;
  if (markerSize!=-999) ret->getAttMarker()->SetMarkerSize(markerSize) ;
  if (fillColor!=-999) ret->getAttFill()->SetFillColor(fillColor) ;
  if (fillStyle!=-999) ret->getAttFill()->SetFillStyle(fillStyle) ;

  if (pc.hasProcessed("BinningSpec")) {
    delete o.bins ;
  }

  return ret ;
}



//_____________________________________________________________________________
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
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return 0;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
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
    coutE(Plotting) << ClassName() << "::" << GetName()
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
  RooHist *graph= new RooHist(*hist,nomBinWidth,1,o.etype,o.xErrorSize,o.correctForBinWidth); 
  if(0 == graph) {
    coutE(Plotting) << ClassName() << "::" << GetName()
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
    coutI(Plotting) << "RooTreeData::plotOn: plotting " << hist->GetSum() << " events out of " << nEnt << " total events" << endl ;
    graph->setRawEntries(nEnt) ;
  }

  // Add self to other hist if requested
  if (o.addToHistName) {
    RooHist* otherGraph = static_cast<RooHist*>(frame->findObject(o.addToHistName,RooHist::Class())) ;

    if (!graph->hasIdenticalBinning(*otherGraph)) {
      coutE(Plotting) << "RooTreeData::plotOn: ERROR Histogram to be added to, '" << o.addToHistName << "',has different binning" << endl ;
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
  } else {
    TString hname(Form("h_%s",GetName())) ;
    if (o.cutRange && strlen(o.cutRange)>0) {
      hname.Append(Form("_CutRange[%s]",o.cutRange)) ;
    } 
    if (o.cuts && strlen(o.cuts)>0) {
      hname.Append(Form("_Cut[%s]",o.cuts)) ;
    } 
    graph->SetName(hname.Data()) ;
  }
  
  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);


  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);



  // cleanup
  delete hist;

  return frame;  
}




//_____________________________________________________________________________
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
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotAsymOn: frame is null" << endl;
    return 0;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
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

  if(0 == fillHistogram(hist1,RooArgList(*var),cuts1.Data(),o.cutRange) ||
     0 == fillHistogram(hist2,RooArgList(*var),cuts2.Data(),o.cutRange)) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotAsymOn: createHistogram() failed" << endl;
    return 0;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist1,*hist2,0,1,o.xErrorSize);
  graph->setYAxisLabel(Form("Asymmetry in %s",asymCat.GetName())) ;

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);

  // Rename graph if requested
  if (o.histName) {
    graph->SetName(o.histName) ;
  } else {
    TString hname(Form("h_%s_Asym[%s]",GetName(),asymCat.GetName())) ;
    if (o.cutRange && strlen(o.cutRange)>0) {
      hname.Append(Form("_CutRange[%s]",o.cutRange)) ;
    } 
    if (o.cuts && strlen(o.cuts)>0) {
      hname.Append(Form("_Cut[%s]",o.cuts)) ;
    } 
    graph->SetName(hname.Data()) ;
  }

  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);

  // cleanup
  delete hist1;
  delete hist2;

  return frame;  
}



//_____________________________________________________________________________
RooPlot* RooTreeData::plotEffOn(RooPlot* frame, const RooAbsCategoryLValue& effCat, PlotOpt o) const 
{
  // Create and fill a histogram with the effiency N[1] / ( N[1] + N[0] ),
  // where N(1/0) is the number of data points with effCat=1 and effCat=0
  // as function of the frames variable. The efficiency category 'effCat' must
  // have exactly 2 +1 and 0.
  // 
  // The plot range and the number of plot bins is determined by the parameters
  // of the plot variable of the frame (RooAbsReal::setPlotRange(), RooAbsReal::setPlotBins())
  // 
  // The optional cut string expression can be used to select the events to be plotted.
  // The cut specification may refer to any variable contained in the data set
  //
  // The drawOptions are passed to the TH1::Draw() method

  if(0 == frame) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotEffOn: frame is null" << endl;
    return 0;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotEffOn: frame does not specify a plot variable" << endl;
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
    cuts1 = Form("(%s)&&(%s==1)",o.cuts,effCat.GetName());
    cuts2 = Form("(%s)&&(%s==0)",o.cuts,effCat.GetName());
  } else {
    cuts1 = Form("(%s==1)",effCat.GetName());
    cuts2 = Form("(%s==0)",effCat.GetName());
  }

  if(0 == fillHistogram(hist1,RooArgList(*var),cuts1.Data(),o.cutRange) ||
     0 == fillHistogram(hist2,RooArgList(*var),cuts2.Data(),o.cutRange)) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotEffOn: createHistogram() failed" << endl;
    return 0;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist1,*hist2,0,1,o.xErrorSize,kTRUE);
  graph->setYAxisLabel(Form("Efficiency of %s=%s",effCat.GetName(),effCat.lookupType(1)->GetName())) ;

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);

  // Rename graph if requested
  if (o.histName) {
    graph->SetName(o.histName) ;
  } else {
    TString hname(Form("h_%s_Eff[%s]",GetName(),effCat.GetName())) ;
    if (o.cutRange && strlen(o.cutRange)>0) {
      hname.Append(Form("_CutRange[%s]",o.cutRange)) ;
    } 
    if (o.cuts && strlen(o.cuts)>0) {
      hname.Append(Form("_Cut[%s]",o.cuts)) ;
    } 
    graph->SetName(hname.Data()) ;
  }

  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);

  // cleanup
  delete hist1;
  delete hist2;

  return frame;  
}



//_____________________________________________________________________________
TH1 *RooTreeData::fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts, const char* cutRange) const
{
  // Loop over columns of our tree data and fill the input histogram. Returns a pointer to the
  // input histogram, or zero in case of an error. The input histogram can be any TH1 subclass, and
  // therefore of arbitrary dimension. Variables are matched with the (x,y,...) dimensions of the input
  // histogram according to the order in which they appear in the input plotVars list.

  // Do we have a valid histogram to use?
  if(0 == hist) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: no valid histogram to fill" << endl;
    return 0;
  }

  // Check that the number of plotVars matches the input histogram's dimension
  Int_t hdim= hist->GetDimension();
  if(hdim != plotVars.getSize()) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: plotVars has the wrong dimension" << endl;
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
      coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: cannot plot variable \"" << var->GetName()
	   << "\" of type " << var->ClassName() << endl;
      return 0;
    }
    RooAbsArg *found= _vars.find(realVar->GetName());
    if(!found) {
      RooAbsArg *clone= plotClones.addClone(*realVar,kTRUE); // do not complain about duplicates
      assert(0 != clone);
      if(!clone->dependsOn(_vars)) {
	coutW(InputArguments) << ClassName() << "::" << GetName()
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
      coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: invalid cuts \"" << cuts << "\"" << endl;
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
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: cannot fill histogram with "
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



//_____________________________________________________________________________
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
      coutE(Plotting) << "RooTreeData::Table(" << GetName() << "): Argument " << cat.GetName() 
	   << " is not in dataset and is also not dependent on data set" << endl ;
      return 0 ; 
    }

    // Clone derived variable 
    tableSet = (RooArgSet*) RooArgSet(cat).snapshot(kTRUE) ;
    if (!tableSet) {
      coutE(Plotting) << "RooTreeData::table(" << GetName() << ") Couldn't deep-clone table category, abort." << endl ;
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
  Roo1DTable* table2 = tableVar->createTable(tableName) ;

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
    
    table2->fill(*tableVar,weight()) ;
  }

  if (ownPlotVar) delete tableSet ;
  if (cutVar) delete cutVar ;

  return table2 ;
}



//_____________________________________________________________________________
Double_t RooTreeData::standMoment(RooRealVar &var, Double_t order, const char* cutSpec, const char* cutRange) const 
{
  // Calculate standardized moment < (X - <X>)^n > / sigma^n,  where n = order.
  // 
  // If cutSpec and/or cutRange are specified
  // the moment is calculated on the subset of the data which pass the C++ cut specification expression 'cutSpec'
  // and/or are inside the range named 'cutRange'

  // Hardwire invariant answer for first and second moment
  if (order==1) return 0 ;
  if (order==2) return 1 ;

  return moment(var,order,cutSpec,cutRange) / TMath::Power(sigma(var,cutSpec,cutRange),order) ; 
}



//_____________________________________________________________________________
Double_t RooTreeData::moment(RooRealVar &var, Double_t order, const char* cutSpec, const char* cutRange) const 
{
  // Calculate moment < (X - <X>)^n > where n = order.
  // 
  // If cutSpec and/or cutRange are specified
  // the moment is calculated on the subset of the data which pass the C++ cut specification expression 'cutSpec'
  // and/or are inside the range named 'cutRange'

  Double_t offset = order>0 ? moment(var,0,cutSpec,cutRange) : 0 ;
  return moment(var,offset,order,cutSpec,cutRange) ;

}





//_____________________________________________________________________________
Double_t RooTreeData::moment(RooRealVar &var, Double_t order, Double_t offset, const char* cutSpec, const char* cutRange) const
{
  // Return the 'order'-ed moment of observable 'var' in this dataset. If offset is non-zero it is subtracted
  // from the values of 'var' prior to the moment calculation. If cutSpec and/or cutRange are specified
  // the moment is calculated on the subset of the data which pass the C++ cut specification expression 'cutSpec'
  // and/or are inside the range named 'cutRange'

  // Lookup variable in dataset
  RooRealVar *varPtr= (RooRealVar*) _vars.find(var.GetName());
  if(0 == varPtr) {
    coutE(InputArguments) << "RooDataSet::moment(" << GetName() << ") ERROR: unknown variable: " << var.GetName() << endl ;
    return 0;
  }

  // Check if found variable is of type RooRealVar
  if (!dynamic_cast<RooRealVar*>(varPtr)) {
    coutE(InputArguments) << "RooDataSet::moment(" << GetName() << ") ERROR: variable " << var.GetName() << " is not of type RooRealVar" << endl ;
    return 0;
  }

  // Check if dataset is not empty
  if(sumEntries() == 0.) {
    coutE(InputArguments) << "RooDataSet::moment(" << GetName() << ") WARNING: empty dataset" << endl ;
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



//_____________________________________________________________________________
RooRealVar* RooTreeData::meanVar(RooRealVar &var, const char* cutSpec, const char* cutRange) const
{
  // Create a RooRealVar containing the mean of observable 'var' in
  // this dataset.  If cutSpec and/or cutRange are specified the
  // moment is calculated on the subset of the data which pass the C++
  // cut specification expression 'cutSpec' and/or are inside the
  // range named 'cutRange'
  
  // Create a new variable with appropriate strings. The error is calculated as
  // RMS/Sqrt(N) which is generally valid.

  // Create holder variable for mean
  TString name(var.GetName()),title("Mean of ") ;
  name.Append("Mean");
  title.Append(var.GetTitle());
  RooRealVar *meanv= new RooRealVar(name,title,0) ;
  meanv->setConstant(kFALSE) ;

  // Adjust plot label
  TString label("<") ;
  label.Append(var.getPlotLabel());
  label.Append(">");
  meanv->setPlotLabel(label.Data());

  // fill in this variable's value and error
  Double_t meanVal=moment(var,1,0,cutSpec,cutRange) ;
  Double_t N(sumEntries(cutSpec,cutRange)) ;

  Double_t rmsVal= sqrt(moment(var,2,meanVal,cutSpec,cutRange)*N/(N-1));
  meanv->setVal(meanVal) ;
  meanv->setError(N > 0 ? rmsVal/sqrt(N) : 0);

  return meanv;
}



//_____________________________________________________________________________
RooRealVar* RooTreeData::rmsVar(RooRealVar &var, const char* cutSpec, const char* cutRange) const
{
  // Create a RooRealVar containing the RMS of observable 'var' in
  // this dataset.  If cutSpec and/or cutRange are specified the
  // moment is calculated on the subset of the data which pass the C++
  // cut specification expression 'cutSpec' and/or are inside the
  // range named 'cutRange'

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



//_____________________________________________________________________________
Bool_t RooTreeData::getRange(RooRealVar& var, Double_t& lowest, Double_t& highest, Double_t marginFrac, Bool_t symMode) const 
{
  // Fill Doubles 'lowest' and 'highest' with the lowest and highest value of
  // observable 'var' in this dataset. If the return value is kTRUE and error
  // occurred

  // Lookup variable in dataset
  RooRealVar *varPtr= (RooRealVar*) _vars.find(var.GetName());
  if(0 == varPtr) {
    coutE(InputArguments) << "RooDataSet::getRange(" << GetName() << ") ERROR: unknown variable: " << var.GetName() << endl ;
    return kTRUE;
  }

  // Check if found variable is of type RooRealVar
  if (!dynamic_cast<RooRealVar*>(varPtr)) {
    coutE(InputArguments) << "RooDataSet::getRange(" << GetName() << ") ERROR: variable " << var.GetName() << " is not of type RooRealVar" << endl ;
    return kTRUE;
  }

  // Check if dataset is not empty
  if(sumEntries() == 0.) {
    coutE(InputArguments) << "RooDataSet::getRange(" << GetName() << ") WARNING: empty dataset" << endl ;
    return kTRUE;
  }

  // Look for highest and lowest value 
  lowest = RooNumber::infinity() ;
  highest = -RooNumber::infinity() ;
  for (Int_t i=0 ; i<numEntries() ; i++) {
    get(i) ;
    if (varPtr->getVal()<lowest) {
      lowest = varPtr->getVal() ;
    }
    if (varPtr->getVal()>highest) {
      highest = varPtr->getVal() ;
    }
  }  

  if (marginFrac>0) {
    if (symMode==kFALSE) {

      Double_t margin = marginFrac*(highest-lowest) ;    
      lowest -= margin ;
      highest += margin ; 
      if (lowest<var.getMin()) lowest = var.getMin() ;
      if (highest>var.getMax()) highest = var.getMax() ;

    } else {

      Double_t mom1 = moment(var,1) ;
      Double_t delta = ((highest-mom1)>(mom1-lowest)?(highest-mom1):(mom1-lowest))*(1+marginFrac) ;
      lowest = mom1-delta ;
      highest = mom1+delta ;
      if (lowest<var.getMin()) lowest = var.getMin() ;
      if (highest>var.getMax()) highest = var.getMax() ;

    }
  }
  
  return kFALSE ;
}



//_____________________________________________________________________________
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



//_____________________________________________________________________________
RooPlot* RooTreeData::statOn(RooPlot* frame, const char* what, const char *label, Int_t sigDigits,
			     Option_t *options, Double_t xmin, Double_t xmax, Double_t ymax, 
			     const char* cutSpec, const char* cutRange, const RooCmdArg* formatCmd) 
{
  // Implementation back-end of statOn() mehtod with named arguments

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
  box->SetName(Form("%s_statBox",GetName())) ;
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
  RooRealVar *meanv= meanVar(*(RooRealVar*)frame->getPlotVar(),cutSpec,cutRange);
  meanv->setPlotLabel("Mean") ;
  RooRealVar *rms= rmsVar(*(RooRealVar*)frame->getPlotVar(),cutSpec,cutRange);
  rms->setPlotLabel("RMS") ;
  TString *rmsText, *meanText, *NText ;
  if (options) {
    rmsText= rms->format(sigDigits,options);
    meanText= meanv->format(sigDigits,options);
    NText= N.format(sigDigits,options);
  } else {
    rmsText= rms->format(*formatCmd);
    meanText= meanv->format(*formatCmd);
    NText= N.format(*formatCmd);
  }
  if (showR) text= box->AddText(rmsText->Data());
  if (showM) text= box->AddText(meanText->Data());
  if (showN) text= box->AddText(NText->Data());

  // cleanup heap memory
  delete NText;
  delete meanText;
  delete rmsText;
  delete meanv;
  delete rms;

  // add the optional label if specified
  if(showLabel) text= box->AddText(label);

  frame->addObject(box) ;
  return frame ;
}



//_____________________________________________________________________________
Int_t RooTreeData::numEntries(Bool_t) const 
{ 
  // Return the number of entries in this dataset
  return (Int_t)GetEntries() ; 
}



//_____________________________________________________________________________
void RooTreeData::printMultiline(ostream& os, Int_t content, Bool_t verbose, TString indent) const 
{
  // Detailed printing interface

  RooAbsData::printMultiline(os,content,verbose,indent) ;  

  os << indent << "Dataset " << GetName() << " (" << GetTitle() << ")" << endl ;
  if (isWeighted()) {
    os << indent << "  Contains " << numEntries() << " entries with a total weight of " << sumEntries() << endl;
  } else {
    os << indent << "  Contains " << numEntries() << " entries" << endl;
  }

  if (!verbose) {
    os << indent << "  Observables " << _vars << endl ;
  } else {
    os << indent << "  Observables: " ;
    _vars.printStream(os,kName|kValue|kExtras|kTitle,kVerbose,indent+"  ") ;
  }

  if(verbose) {
    if (_cachedVars.getSize()>0) {
      os << indent << "  Caches " << _cachedVars << endl ;
    }
    if(_truth.getSize() > 0) {
      os << indent << "  Generated with ";
      TString deeper(indent) ;
      deeper += "   " ;
      _truth.printStream(os,kName|kValue,kStandard,deeper) ;
    }
  }
}



//_____________________________________________________________________________
void RooTreeData::optimizeReadingWithCaching(RooAbsArg& arg, const RooArgSet& cacheList)
{
  // Prepare dataset for use with cached constant terms listed in
  // 'cacheList' of expression 'arg'. Deactivate tree branches
  // for any dataset observable that is either not used at all,
  // or is used exclusively by cached branch nodes.

  RooArgSet pruneSet ;

  // Add unused observables in this dataset to pruneSet
  pruneSet.add(*get()) ;
  RooArgSet* usedObs = arg.getObservables(*this) ;
  pruneSet.remove(*usedObs,kTRUE,kTRUE) ;

  // Add observables exclusively used to calculate cached observables to pruneSet
  TIterator* vIter = get()->createIterator() ;
  RooAbsArg *var ;
  while ((var=(RooAbsArg*) vIter->Next())) {
    if (allClientsCached(var,cacheList)) {
      pruneSet.add(*var) ;
    }
  }
  delete vIter ;


  if (pruneSet.getSize()!=0) {

    // Go over all used observables and check if any of them have parameterized
    // ranges in terms of pruned observables. If so, remove those observable
    // from the pruning list
    TIterator* uIter = usedObs->createIterator() ;
    RooAbsArg* obs ;
    while((obs=(RooAbsArg*)uIter->Next())) {
      RooRealVar* rrv = dynamic_cast<RooRealVar*>(obs) ;
      if (rrv && !rrv->getBinning().isShareable()) {
	RooArgSet depObs ;
	RooAbsReal* loFunc = rrv->getBinning().lowBoundFunc() ;
	RooAbsReal* hiFunc = rrv->getBinning().highBoundFunc() ;
	if (loFunc) {
	  loFunc->leafNodeServerList(&depObs,0,kTRUE) ;
	}
	if (hiFunc) {
	  hiFunc->leafNodeServerList(&depObs,0,kTRUE) ;
	}
	if (depObs.getSize()>0) {
	  pruneSet.remove(depObs,kTRUE,kTRUE) ;
	}
      }
    }
    delete uIter ;
  }

  if (pruneSet.getSize()!=0) {
    
    // Deactivate tree branches here
    cxcoutI(Optimization) << "RooTreeData::optimizeReadingForTestStatistic(" << GetName() << "): Observables " << pruneSet
			    << " in dataset are either not used at all, orserving exclusively p.d.f nodes that are now cached, disabling reading of these observables for TTree" << endl ;
    setArgStatus(pruneSet,kFALSE) ;
  }

  delete usedObs ;
  
}


//_____________________________________________________________________________
Bool_t RooTreeData::allClientsCached(RooAbsArg* var, const RooArgSet& cacheList)
{
  // Utility function that determines if all clients of object 'var'
  // appear in given list of cached nodes.

  Bool_t ret(kTRUE), anyClient(kFALSE) ;

  TIterator* cIter = var->valueClientIterator() ;    
  RooAbsArg* client ;
  while ((client=(RooAbsArg*) cIter->Next())) {
    anyClient = kTRUE ;
    if (!cacheList.find(client->GetName())) {
      // If client is not cached recurse
      ret &= allClientsCached(client,cacheList) ;
    }
  }
  delete cIter ;

  return anyClient?ret:kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooTreeData::valid() const 
{
  // Return true if currently loaded coordinate is considered valid within
  // the current range definitions of all observables
  return kTRUE ;
}


