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
// RooTreeDataStore is the abstract base class for data collection that
// use a TTree as internal storage mechanism
// END_HTML
//

#include "RooFit.h"
#include "RooMsgService.h"
#include "RooTreeDataStore.h"

#include "Riostream.h"
#include "TTree.h"
#include "TChain.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "RooFormulaVar.h"
#include "RooRealVar.h"
#include "RooHistError.h"

#include <iomanip>
using namespace std ;

ClassImp(RooTreeDataStore)
;


Int_t RooTreeDataStore::_defTreeBufSize = 4096 ;



//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore() :
  _tree(0),
  _cacheTree(0),
  _cacheOwner(0),
  _defCtor(kTRUE),
  _wgtVar(0),
  _extWgtArray(0),
  _extWgtErrLoArray(0),
  _extWgtErrHiArray(0),
  _extSumW2Array(0),
  _curWgt(1),
  _curWgtErrLo(0),
  _curWgtErrHi(0),
  _curWgtErr(0)
{
}



//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore(TTree* t, const RooArgSet& vars, const char* wgtVarName) :
  RooAbsDataStore("blah","blah",varsNoWeight(vars,wgtVarName)),
  _tree(t),
  _cacheTree(0),
  _cacheOwner(0),
  _defCtor(kTRUE),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName)),
  _extWgtArray(0),
  _extWgtErrLoArray(0),
  _extWgtErrHiArray(0),
  _extSumW2Array(0),
  _curWgt(1)
{
  // Constructor to facilitate reading of legacy RooDataSets
}




//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _tree(0),
  _cacheTree(0), 
  _cacheOwner(0),
  _defCtor(kFALSE),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName)),
  _extWgtArray(0),
  _extWgtErrLoArray(0),
  _extWgtErrHiArray(0),
  _extSumW2Array(0),
  _curWgt(1),
  _curWgtErrLo(0),
  _curWgtErrHi(0),
  _curWgtErr(0)
{
  initialize() ;  
}




//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars, TTree& t, const RooFormulaVar& select, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _tree(0),
  _cacheTree(0),
  _cacheOwner(0),
  _defCtor(kFALSE),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName)),
  _extWgtArray(0),
  _extWgtErrLoArray(0),
  _extWgtErrHiArray(0),
  _extSumW2Array(0),
  _curWgt(1),
  _curWgtErrLo(0),
  _curWgtErrHi(0),
  _curWgtErr(0)
{
  initialize() ;  
  loadValues(&t,&select) ;
}



//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars, TTree& t, const char* selExpr, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _tree(0),
  _cacheTree(0),
  _cacheOwner(0),
  _defCtor(kFALSE),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName)),
  _extWgtArray(0),
  _extWgtErrLoArray(0),
  _extWgtErrHiArray(0),
  _extSumW2Array(0),
  _curWgt(1),
  _curWgtErrLo(0),
  _curWgtErrHi(0),
  _curWgtErr(0)
{
  initialize() ;  

  if (selExpr && *selExpr) {
    // Create a RooFormulaVar cut from given cut expression
    RooFormulaVar select(selExpr,selExpr,_vars) ;
    loadValues(&t,&select);
  } else {
    loadValues(&t);
  }
}



//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars, const RooAbsDataStore& tds, const RooFormulaVar& select, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _tree(0),
  _cacheTree(0),
  _cacheOwner(0),
  _defCtor(kFALSE),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName)),
  _extWgtArray(0),
  _extWgtErrLoArray(0),
  _extWgtErrHiArray(0),
  _extSumW2Array(0),
  _curWgt(1),
  _curWgtErrLo(0),
  _curWgtErrHi(0),
  _curWgtErr(0)
{
  initialize() ;  
  loadValues(&tds,&select) ;
}



//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore(const char* name, const char* title, const RooArgSet& vars, const RooAbsDataStore& ads, const char* selExpr, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _tree(0),
  _cacheTree(0),
  _cacheOwner(0),
  _defCtor(kFALSE),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName)),
  _extWgtArray(0),
  _extWgtErrLoArray(0),
  _extWgtErrHiArray(0),
  _extSumW2Array(0),
  _curWgt(1),
  _curWgtErrLo(0),
  _curWgtErrHi(0),
  _curWgtErr(0)
{
  initialize() ;  

  if (selExpr && *selExpr) {
    // Create a RooFormulaVar cut from given cut expression
    RooFormulaVar select(selExpr,selExpr,_vars) ;
    loadValues(&ads,&select);
  } else {
    loadValues(&ads);
  }
}




//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore(const char *name, const char *title, RooAbsDataStore& tds, 
			 const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
			 Int_t nStart, Int_t nStop, Bool_t /*copyCache*/, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)), _defCtor(kFALSE),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName)),
  _extWgtArray(0),
  _extWgtErrLoArray(0),
  _extWgtErrHiArray(0),
  _extSumW2Array(0),
  _curWgt(1),
  _curWgtErrLo(0),
  _curWgtErrHi(0),
  _curWgtErr(0)
{

  // WVE NEED TO ADJUST THIS FOR WEIGHTS

  // Protected constructor for internal use only
  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(name,title) ;

  // Deep clone cutVar and attach clone to this dataset
  RooFormulaVar* cloneVar = 0;
  if (cutVar) {    
    cloneVar = (RooFormulaVar*) cutVar->cloneTree() ;
    cloneVar->attachDataStore(tds) ;
  }

  // Constructor from existing data set with list of variables that preserves the cache
  initialize();

  attachCache(0,((RooTreeDataStore&)tds)._cachedVars) ;

  // WVE copy values of cached variables here!!!
  _cacheTree->CopyEntries(((RooTreeDataStore&)tds)._cacheTree) ;
  _cacheOwner = 0 ;
  
  loadValues(&tds,cloneVar,cutRange,nStart,nStop);

  if (cloneVar) delete cloneVar ;
}



//_____________________________________________________________________________
RooArgSet RooTreeDataStore::varsNoWeight(const RooArgSet& allVars, const char* wgtName) 
{
  // Utility function for constructors
  // Return RooArgSet that is copy of allVars minus variable matching wgtName if specified

  RooArgSet ret(allVars) ;
  if(wgtName) {
    RooAbsArg* wgt = allVars.find(wgtName) ;
    if (wgt) {
      ret.remove(*wgt,kTRUE,kTRUE) ;
    }
  }
  return ret ;
}



//_____________________________________________________________________________
RooRealVar* RooTreeDataStore::weightVar(const RooArgSet& allVars, const char* wgtName) 
{
  // Utility function for constructors
  // Return pointer to weight variable if it is defined

  if(wgtName) {
    RooRealVar* wgt = dynamic_cast<RooRealVar*>(allVars.find(wgtName)) ;
    return wgt ;
  } 
  return 0 ;
}




//_____________________________________________________________________________
void RooTreeDataStore::attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVarsIn) 
{
  // Initialize cache of dataset: attach variables of cache ArgSet
  // to the corresponding TTree branches

  // iterate over the cache variables for this dataset
  _cachedVars.removeAll() ;
  TIterator* iter = cachedVarsIn.createIterator() ;
  RooAbsArg *var;
  while((0 != (var= (RooAbsArg*)iter->Next()))) {    
    var->attachToTree(*_cacheTree,_defTreeBufSize) ;
    _cachedVars.add(*var) ;
  }
  delete iter ;
  _cacheOwner = newOwner ;

}






//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore(const RooTreeDataStore& other, const char* newname) :
  RooAbsDataStore(other,newname),
  _tree(0),
  _cacheTree(0),
  _defCtor(kFALSE),
  _varsww(other._varsww),
  _wgtVar(other._wgtVar),
  _extWgtArray(other._extWgtArray),
  _extWgtErrLoArray(other._extWgtErrLoArray),
  _extWgtErrHiArray(other._extWgtErrHiArray),
  _extSumW2Array(other._extSumW2Array),
  _curWgt(other._curWgt),
  _curWgtErrLo(other._curWgtErrLo),
  _curWgtErrHi(other._curWgtErrHi),
  _curWgtErr(other._curWgtErr)
{
  initialize() ;  
  loadValues(&other) ;
}


//_____________________________________________________________________________
RooTreeDataStore::RooTreeDataStore(const RooTreeDataStore& other, const RooArgSet& vars, const char* newname) :
  RooAbsDataStore(other,varsNoWeight(vars,other._wgtVar?other._wgtVar->GetName():0),newname),
  _tree(0),
  _cacheTree(0),
  _defCtor(kFALSE),
  _varsww(vars),
  _wgtVar(other._wgtVar?weightVar(vars,other._wgtVar->GetName()):0),
  _extWgtArray(other._extWgtArray),
  _extWgtErrLoArray(other._extWgtErrLoArray),
  _extWgtErrHiArray(other._extWgtErrHiArray),
  _extSumW2Array(other._extSumW2Array),
  _curWgt(other._curWgt),
  _curWgtErrLo(other._curWgtErrLo),
  _curWgtErrHi(other._curWgtErrHi),
  _curWgtErr(other._curWgtErr)
{
  initialize() ;  
  loadValues(&other) ;
}




//_____________________________________________________________________________
RooTreeDataStore::~RooTreeDataStore()
{
  // Destructor
  if (_tree) {
    delete _tree ;
  }
  if (_cacheTree) {
    delete _cacheTree ;
  }
}



//_____________________________________________________________________________
void RooTreeDataStore::initialize() 
{
  // One-time initialization common to all constructor forms.  Attach
  // variables of internal ArgSet to the corresponding TTree branches

  // Recreate (empty) cache tree
  createTree(GetName(),GetTitle()) ;

  // Attach each variable to the dataset
  TIterator* iter = _varsww.createIterator() ;
  RooAbsArg *var;
  while((0 != (var= (RooAbsArg*)iter->Next()))) {
    var->attachToTree(*_tree,_defTreeBufSize) ;
  }
  delete iter ;
}





//_____________________________________________________________________________
void RooTreeDataStore::createTree(const char* name, const char* title)
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
    gDirectory->RecursiveRemove(_tree) ;
  }
  if (!_cacheTree) {
    _cacheTree = new TTree(name,title) ;
    _cacheTree->SetDirectory(0) ;
    gDirectory->RecursiveRemove(_cacheTree) ;
  }

  if (notInMemNow) {
    gDirectory->cd(pwd) ;
  }
  
}




//_____________________________________________________________________________
void RooTreeDataStore::loadValues(const TTree *t, const RooFormulaVar* select, const char* /*rangeName*/, Int_t /*nStart*/, Int_t /*nStop*/) 
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
  RooArgSet *sourceArgSet = (RooArgSet*) _varsww.snapshot(kFALSE) ;
  
  // Attach args in cloned list to cloned source tree
  TIterator* sourceIter =  sourceArgSet->createIterator() ;
  RooAbsArg* sourceArg = 0;
  while ((sourceArg=(RooAbsArg*)sourceIter->Next())) {
    sourceArg->attachToTree(*tClone,_defTreeBufSize) ;
  }

  // Redirect formula servers to sourceArgSet
  RooFormulaVar* selectClone(0) ;
  if (select) {
    selectClone = (RooFormulaVar*) select->cloneTree() ;
    selectClone->recursiveRedirectServers(*sourceArgSet) ;
    selectClone->setOperMode(RooAbsArg::ADirty,kTRUE) ;
  }

  // Loop over events in source tree   
  RooAbsArg* destArg = 0;
  TIterator* destIter = _varsww.createIterator() ;
  Int_t numInvalid(0) ;
  Int_t nevent= (Int_t)tClone->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber=tClone->GetEntryNumber(i);
    if (entryNumber<0) break;
    tClone->GetEntry(entryNumber,1);
 
    // Copy from source to destination
     destIter->Reset() ;
     sourceIter->Reset() ;
     Bool_t allOK(kTRUE) ;
     while ((destArg = (RooAbsArg*)destIter->Next())) {              
       sourceArg = (RooAbsArg*) sourceIter->Next() ;
       destArg->copyCache(sourceArg) ;
       sourceArg->copyCache(destArg) ;
       if (!destArg->isValid()) {
	 numInvalid++ ;
	 allOK=kFALSE ;
	 break ;
       }       
     }   

     // Does this event pass the cuts?
     if (!allOK || (selectClone && selectClone->getVal()==0)) {
       continue ; 
     }

     fill() ;
  }
  delete destIter ;

  if (numInvalid>0) {
    coutI(Eval) << "RooTreeDataStore::loadValues(" << GetName() << ") Ignored " << numInvalid << " out of range events" << endl ;
  }
  
  SetTitle(t->GetTitle());

  delete sourceIter ;
  delete sourceArgSet ;
  delete selectClone ;
  delete tClone ;
}






//_____________________________________________________________________________
void RooTreeDataStore::loadValues(const RooAbsDataStore *ads, const RooFormulaVar* select, 
				  const char* rangeName, Int_t nStart, Int_t nStop)  
{
  // Load values from dataset 't' into this data collection, optionally
  // selecting events using 'select' RooFormulaVar
  //

  // Redirect formula servers to source data row
  RooFormulaVar* selectClone(0) ;
  if (select) {
    selectClone = (RooFormulaVar*) select->cloneTree() ;
    selectClone->recursiveRedirectServers(*ads->get()) ;
    selectClone->setOperMode(RooAbsArg::ADirty,kTRUE) ;
  }

  // Force RDS internal initialization
  ads->get(0) ;

  // Loop over events in source tree   
  RooAbsArg* arg = 0;
  TIterator* destIter = _varsww.createIterator() ;
  Int_t nevent = nStop < ads->numEntries() ? nStop : ads->numEntries() ;
  Bool_t allValid ;

  Bool_t isTDS = dynamic_cast<const RooTreeDataStore*>(ads) ;
  if (isTDS) {
    ((RooTreeDataStore*)(ads))->resetBuffers() ;
  }
  for(Int_t i=nStart; i < nevent ; ++i) {
    ads->get(i) ;

    // Does this event pass the cuts?
    if (selectClone && selectClone->getVal()==0) {
      continue ; 
    }


    if (isTDS) {
      _varsww.assignValueOnly(((RooTreeDataStore*)ads)->_varsww) ;
    } else {
      _varsww.assignValueOnly(*ads->get()) ;
    }

    destIter->Reset() ;
    // Check that all copied values are valid
    allValid=kTRUE ;
    while((arg=(RooAbsArg*)destIter->Next())) {
      if (!arg->isValid() || (rangeName && !arg->inRange(rangeName))) {
	//cout << "arg " << arg->GetName() << " is not valid" << endl ;
	//arg->Print("v") ;
	allValid=kFALSE ;
	break ;
      }
    }
    //cout << "RooTreeData::loadValues(" << GetName() << ") allValid = " << (allValid?"T":"F") << endl ;
    if (!allValid) {
      continue ;
    }
    
    _cachedVars = ((RooTreeDataStore*)ads)->_cachedVars ;
    fill() ;
   }
  delete destIter ;
  if (isTDS) {
    ((RooTreeDataStore*)(ads))->restoreAlternateBuffers() ;
  }
  
  delete selectClone ;
  SetTitle(ads->GetTitle());
}



//_____________________________________________________________________________
Bool_t RooTreeDataStore::valid() const 
{
  // Return true if currently loaded coordinate is considered valid within
  // the current range definitions of all observables
  return kTRUE ;
}




//_____________________________________________________________________________
Int_t RooTreeDataStore::fill()
{
   // Interface function to TTree::Fill
   return _tree->Fill() ;
}
 


//_____________________________________________________________________________
const RooArgSet* RooTreeDataStore::get(Int_t index) const 
{
  // Load the n-th data point (n='index') in memory
  // and return a pointer to the internal RooArgSet
  // holding its coordinates.

  checkInit() ;

  Int_t ret = ((RooTreeDataStore*)this)->GetEntry(index, 1) ;

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
  
  // Update current weight cache
  if (_extWgtArray) {

    // If external array is specified use that  
    _curWgt = _extWgtArray[index] ;
    _curWgtErrLo = _extWgtErrLoArray[index] ;
    _curWgtErrHi = _extWgtErrHiArray[index] ;
    _curWgtErr   = sqrt(_extSumW2Array[index]) ;

  } else if (_wgtVar) {

    // Otherwise look for weight variable
    _curWgt = _wgtVar->getVal() ;
    _curWgtErrLo = _wgtVar->getAsymErrorLo() ;
    _curWgtErrHi = _wgtVar->getAsymErrorHi() ;
    _curWgtErr   = _wgtVar->hasAsymError() ? ((_wgtVar->getAsymErrorHi() - _wgtVar->getAsymErrorLo())/2)  : _wgtVar->getError() ;

  } else {

    // Otherwise return 1 
    _curWgt=1.0 ;
    _curWgtErrLo = 0 ;
    _curWgtErrHi = 0 ;
    _curWgtErr = 0 ;
    
  }

  return &_vars;
}



//_____________________________________________________________________________
Double_t RooTreeDataStore::weight(Int_t index) const 
{
  // Return the weight of the n-th data point (n='index') in memory
  get(index) ;
  return weight() ;
}



//_____________________________________________________________________________
Double_t RooTreeDataStore::weight() const 
{
  // Return the weight of the n-th data point (n='index') in memory
  return _curWgt ;
}


//_____________________________________________________________________________
Double_t RooTreeDataStore::weightError(RooAbsData::ErrorType etype) const 
{
  if (_extWgtArray) {

    // We have a weight array, use that info

    // Return symmetric error on current bin calculated either from Poisson statistics or from SumOfWeights
    Double_t lo,hi ;
    weightError(lo,hi,etype) ;
    return (lo+hi)/2 ;

   } else if (_wgtVar) {

    // We have a a weight variable, use that info
    if (_wgtVar->hasAsymError()) {
      return ( _wgtVar->getAsymErrorHi() - _wgtVar->getAsymErrorLo() ) / 2 ;
    } else {
      return _wgtVar->getError() ;    
    }

  } else {

    // We have no weights
    return 0 ;

  }
}



//_____________________________________________________________________________
void RooTreeDataStore::weightError(Double_t& lo, Double_t& hi, RooAbsData::ErrorType etype) const
{
  if (_extWgtArray) {
    
    // We have a weight array, use that info
    switch (etype) {
      
    case RooAbsData::Auto:
      throw string(Form("RooDataHist::weightError(%s) error type Auto not allowed here",GetName())) ;
      break ;
      
    case RooAbsData::Expected:
      throw string(Form("RooDataHist::weightError(%s) error type Expected not allowed here",GetName())) ;
      break ;
      
    case RooAbsData::Poisson:
      // Weight may be preset or precalculated    
      if (_curWgtErrLo>=0) {
	lo = _curWgtErrLo ;
	hi = _curWgtErrHi ;
	return ;
      }
      
      // Otherwise Calculate poisson errors
      Double_t ym,yp ;  
      RooHistError::instance().getPoissonInterval(Int_t(weight()+0.5),ym,yp,1) ;
      lo = weight()-ym ;
      hi = yp-weight() ;
      return ;
      
    case RooAbsData::SumW2:
      lo = _curWgtErr ;
      hi = _curWgtErr ;
      return ;
      
    case RooAbsData::None:
      lo = 0 ;
      hi = 0 ;
      return ;
    }    
    
  } else if (_wgtVar) {

    // We have a a weight variable, use that info
    if (_wgtVar->hasAsymError()) {
      hi = _wgtVar->getAsymErrorHi() ;
      lo = _wgtVar->getAsymErrorLo() ;
    } else {
      hi = _wgtVar->getError() ;
      lo = _wgtVar->getError() ;
    }  

  } else {

    // We are unweighted
    lo=0 ;
    hi=0 ;

  }
}


//_____________________________________________________________________________
Bool_t RooTreeDataStore::changeObservableName(const char* from, const char* to) 
{
  // Change name of internal observable named 'from' into 'to'

  // Find observable to be changed
  RooAbsArg* var = _vars.find(from) ;

  // Check that we found it
  if (!var) {
    coutE(InputArguments) << "RooTreeDataStore::changeObservableName(" << GetName() << " no observable " << from << " in this dataset" << endl ;
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

  return kFALSE ;
}

  

//_____________________________________________________________________________
RooAbsArg* RooTreeDataStore::addColumn(RooAbsArg& newVar, Bool_t adjustRange)
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

  // WVE need to reset TTRee buffers to original datamembers here
  resetBuffers() ;

  // Clone variable and attach to cloned tree 
  RooAbsArg* newVarClone = newVar.cloneTree() ;
  newVarClone->recursiveRedirectServers(_vars,kFALSE) ;

  // Attach value place holder to this tree
  ((RooAbsArg*)valHolder)->attachToTree(*_tree,_defTreeBufSize) ;
  _vars.add(*valHolder) ;
  _varsww.add(*valHolder) ;


  // Fill values of of placeholder
  for (int i=0 ; i<GetEntries() ; i++) {
    get(i) ;

    newVarClone->syncCache(&_vars) ;
    valHolder->copyCache(newVarClone) ;
    valHolder->fillTreeBranch(*_tree) ;
  }

  // WVE need to restore TTRee buffers to previous values here
  restoreAlternateBuffers() ;

  if (adjustRange) {
//     // Set range of valHolder to (just) bracket all values stored in the dataset
//     Double_t vlo,vhi ;
//     RooRealVar* rrvVal = dynamic_cast<RooRealVar*>(valHolder) ;
//     if (rrvVal) {
//       getRange(*rrvVal,vlo,vhi,0.05) ;
//       rrvVal->setRange(vlo,vhi) ;  
//     }
  }



  delete newVarClone ;  
  return valHolder ;
}



//_____________________________________________________________________________
RooArgSet* RooTreeDataStore::addColumns(const RooArgList& varList)
{
  // Utility function to add multiple columns in one call
  // See addColumn() for details

  TIterator* vIter = varList.createIterator() ;
  RooAbsArg* var ;

  checkInit() ;

  TList cloneSetList ;
  RooArgSet cloneSet ;
  RooArgSet* holderSet = new RooArgSet ;

  // WVE need to reset TTRee buffers to original datamembers here
  resetBuffers() ;


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
      coutE(InputArguments) << "RooTreeDataStore::RooTreeData(" << GetName() 
			    << ") Couldn't deep-clone variable " << var->GetName() << ", abort." << endl ;
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
  RooAbsArg *cloneArg, *holder ;
  // Fill values of of placeholder
  for (int i=0 ; i<GetEntries() ; i++) {
    get(i) ;

    cIter->Reset() ;
    hIter->Reset() ;
    while((cloneArg=(RooAbsArg*)cIter->Next())) {
      holder = (RooAbsArg*)hIter->Next() ;

      cloneArg->syncCache(&_vars) ;
      holder->copyCache(cloneArg) ;
      holder->fillTreeBranch(*_tree) ;
    }
  }

  // WVE need to restore TTRee buffers to previous values here
  restoreAlternateBuffers() ;
  
  delete cIter ;
  delete hIter ;

  cloneSetList.Delete() ;
  return holderSet ;
}




//_____________________________________________________________________________
RooAbsDataStore* RooTreeDataStore::merge(const RooArgSet& allVars, list<RooAbsDataStore*> dstoreList)
{
  // Merge columns of supplied data set(s) with this data set.  All
  // data sets must have equal number of entries.  In case of
  // duplicate columns the column of the last dataset in the list
  // prevails
    
  RooTreeDataStore* mergedStore = new RooTreeDataStore("merged","merged",allVars) ;

  Int_t nevt = dstoreList.front()->numEntries() ;
  for (int i=0 ; i<nevt ; i++) {

    // Cope data from self
    mergedStore->_vars = *get(i) ;
      
    // Copy variables from merge sets
    for (list<RooAbsDataStore*>::iterator iter = dstoreList.begin() ; iter!=dstoreList.end() ; iter++) {
      const RooArgSet* partSet = (*iter)->get(i) ;
      mergedStore->_vars = *partSet ;
    }

    mergedStore->fill() ;
  }
  return mergedStore ;
}





//_____________________________________________________________________________
void RooTreeDataStore::append(RooAbsDataStore& other) 
{
  Int_t nevt = other.numEntries() ;
  for (int i=0 ; i<nevt ; i++) {  
    _vars = *other.get(i) ;
    if (_wgtVar) {
      _wgtVar->setVal(other.weight()) ;
    }
    
    fill() ;
  }
}


//_____________________________________________________________________________
Double_t RooTreeDataStore::sumEntries() const 
{
  if (_wgtVar) {

    Double_t sum(0), carry(0);
    Int_t nevt = numEntries() ;
    for (int i=0 ; i<nevt ; i++) {  
      get(i) ;
      // Kahan's algorithm for summing to avoid loss of precision
      Double_t y = _wgtVar->getVal() - carry;
      Double_t t = sum + y;
      carry = (t - sum) - y;
      sum = t;
    }    
    return sum ;

  } else if (_extWgtArray) {
    
    Double_t sum(0) , carry(0);
    Int_t nevt = numEntries() ;
    for (int i=0 ; i<nevt ; i++) {  
      // Kahan's algorithm for summing to avoid loss of precision
      Double_t y = _extWgtArray[i] - carry;
      Double_t t = sum + y;
      carry = (t - sum) - y;
      sum = t;
    }    
    return sum ;
    
  } else {

    return numEntries() ;

  }
}




//_____________________________________________________________________________
Int_t RooTreeDataStore::numEntries() const 
{
  return _tree->GetEntries() ;
}



//_____________________________________________________________________________
void RooTreeDataStore::reset() 
{
  Reset() ;
}



//_____________________________________________________________________________
void RooTreeDataStore::cacheArgs(const RooAbsArg* owner, RooArgSet& newVarSet, const RooArgSet* nset, Bool_t /*skipZeroWeights*/) 
{
  // Cache given RooAbsArgs with this tree: The tree is
  // given direct write access of the args internal cache
  // the args values is pre-calculated for all data points
  // in this data collection. Upon a get() call, the
  // internal cache of 'newVar' will be loaded with the
  // precalculated value and it's dirty flag will be cleared.

  checkInit() ;

  _cacheOwner = owner ;

  RooArgSet* constExprVarSet = (RooArgSet*) newVarSet.selectByAttrib("ConstantExpression",kTRUE) ;
  TIterator *iter = constExprVarSet->createIterator() ;
  RooAbsArg *arg ;

  Bool_t doTreeFill = (_cachedVars.getSize()==0) ;

  while ((arg=(RooAbsArg*)iter->Next())) {
    // Attach original newVar to this tree
    arg->attachToTree(*_cacheTree,_defTreeBufSize) ;
    //arg->recursiveRedirectServers(_vars) ;
    _cachedVars.add(*arg) ;
  }

  // WVE need to reset TTRee buffers to original datamembers here
  //resetBuffers() ;

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

  // WVE need to restore TTRee buffers to previous values here
  //restoreAlternateBuffers() ;

  delete iter ;
  delete constExprVarSet ;
}




//_____________________________________________________________________________
void RooTreeDataStore::setArgStatus(const RooArgSet& set, Bool_t active) 
{
  // Activate or deactivate the branch status of the TTree branch associated
  // with the given set of dataset observables

  TIterator* iter = set.createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    RooAbsArg* depArg = _vars.find(arg->GetName()) ;
    if (!depArg) {
      coutE(InputArguments) << "RooTreeDataStore::setArgStatus(" << GetName() 
			    << ") dataset doesn't contain variable " << arg->GetName() << endl ;
      continue ;
    }
    depArg->setTreeBranchStatus(*_tree,active) ;
  }
  delete iter ;
}



//_____________________________________________________________________________
void RooTreeDataStore::resetCache() 
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
void RooTreeDataStore::attachBuffers(const RooArgSet& extObs) 
{
  _attachedBuffers.removeAll() ;
  RooFIter iter = _varsww.fwdIterator() ;
  RooAbsArg* arg ;
  while((arg=iter.next())) {
    RooAbsArg* extArg = extObs.find(arg->GetName()) ;
    if (extArg) {
      if (arg->getAttribute("StoreError")) {
	extArg->setAttribute("StoreError") ;
      }
      if (arg->getAttribute("StoreAsymError")) {
	extArg->setAttribute("StoreAsymError") ;
      }
      extArg->attachToTree(*_tree) ;
      _attachedBuffers.add(*extArg) ;
    }
  }
}



//_____________________________________________________________________________
void RooTreeDataStore::resetBuffers() 
{ 
  RooFIter iter = _varsww.fwdIterator() ;
  RooAbsArg* arg ;
  while((arg=iter.next())) {
    arg->attachToTree(*_tree) ;
  }
}  



//_____________________________________________________________________________
void RooTreeDataStore::restoreAlternateBuffers() 
{ 
  RooFIter iter = _attachedBuffers.fwdIterator() ;
  RooAbsArg* arg ;
  while((arg=iter.next())) {
    arg->attachToTree(*_tree) ;
  }
}  



//_____________________________________________________________________________
void RooTreeDataStore::checkInit() const
{
  if (_defCtor) {
    const_cast<RooTreeDataStore*>(this)->initialize() ;
    _defCtor = kFALSE ;    
  }
}





//_____________________________________________________________________________
Stat_t RooTreeDataStore::GetEntries() const
{
   // Interface function to TTree::GetEntries
   return _tree->GetEntries() ;
}
 

//_____________________________________________________________________________
void RooTreeDataStore::Reset(Option_t* option)
{
   // Interface function to TTree::Reset
   _tree->Reset(option) ;
}
 

//_____________________________________________________________________________
Int_t RooTreeDataStore::Fill()
{
   // Interface function to TTree::Fill
   return _tree->Fill() ;
}
 

//_____________________________________________________________________________
Int_t RooTreeDataStore::GetEntry(Int_t entry, Int_t getall)
{
   // Interface function to TTree::GetEntry
   Int_t ret1 = _tree->GetEntry(entry,getall) ; 
   if (!ret1) return 0 ;
   _cacheTree->GetEntry(entry,getall) ; 
   return ret1 ;
}


//_____________________________________________________________________________
void RooTreeDataStore::Draw(Option_t* option) 
{ 
  _tree->Draw(option) ; 
}

//______________________________________________________________________________
void RooTreeDataStore::Streamer(TBuffer &R__b)
{
   // Stream an object of class RooTreeDataStore.

   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RooTreeDataStore::Class(),this);
      initialize() ;
   } else {
      R__b.WriteClassBuffer(RooTreeDataStore::Class(),this);
   }
}

