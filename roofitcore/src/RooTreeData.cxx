/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooTreeData.cc,v 1.40 2002/04/04 00:17:09 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu 
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   09-Mar-2001 WV Migrate from RooFitTools and adapt to RooFitCore
 *
 * Copyright (C) 1999 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION [DATA] --
// RooTreeData is the abstract base class for data collection that
// use a TTree as internal storage mechanism

#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
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
#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include "TChain.h"
#include "TROOT.h"

#include "RooFitCore/RooTreeData.hh"
#include "RooFitCore/RooAbsArg.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooAbsCategory.hh"
#include "RooFitCore/Roo1DTable.hh"
#include "RooFitCore/RooFormula.hh"
#include "RooFitCore/RooCategory.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooStringVar.hh"
#include "RooFitCore/RooHist.hh"
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooTrace.hh"
#include "RooFitCore/RooAbsBinning.hh" 

ClassImp(RooTreeData)
;

Int_t RooTreeData::_defTreeBufSize = 4096 ;


RooTreeData::RooTreeData() 
{
  // Default constructor
  RooTrace::create(this) ; 
  _defCtor = kTRUE ;
}


RooTreeData::RooTreeData(const char *name, const char *title, const RooArgSet& vars) :
  RooAbsData(name,title,vars), _truth("Truth"), _defCtor(kFALSE)
{
  // Constructor of empty collection with specified dimensions
  RooTrace::create(this) ;
  
  createTree(name,title) ;

  // Constructor with list of variables
  initialize();
}


RooTreeData::RooTreeData(const char *name, const char *title, RooTreeData *t, 
                       const RooArgSet& vars, const char *cuts) :
 RooAbsData(name,title,vars), _truth("Truth"), 
  _blindString(t->_blindString), _defCtor(kFALSE)
{
  // Constructor from existing data collection with specified dimensions and
  // optional string expression cut

  RooTrace::create(this) ;
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
  RooAbsData(name,title,vars),_truth("Truth"), 
  _blindString(t->_blindString), _defCtor(kFALSE)
{
  // Constructor from existing data collection with specified dimensions and
  // RooFormulaVar cut

  RooTrace::create(this) ;
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
  RooAbsData(name,title,vars), _truth("Truth"), _defCtor(kFALSE)
{
  // Constructor from external TTree with specified dimensions and
  // RooFormulaVar cut

  RooTrace::create(this) ;
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
                       const RooArgSet& vars, const RooFormulaVar* cutVar, Bool_t copyCache) :
  RooAbsData(name,title,vars), _truth("Truth"), 
  _blindString(t->_blindString), _defCtor(kFALSE)
{
  // Protected constructor for internal use only

  RooTrace::create(this) ;
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
  
  loadValues(t,cloneVar);

  // WVE copy values of cached variables here!!!

  if (cloneVarSet) delete cloneVarSet ;
}



RooTreeData::RooTreeData(const char *name, const char *title, TTree *t, 
                       const RooArgSet& vars, const char *cuts) :
  RooAbsData(name,title,vars), _truth("Truth"), _defCtor(kFALSE)
{
  // Constructor from external TTree with specified dimensions and
  // optional string expression cut

  RooTrace::create(this) ;
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
  RooAbsData(name,name,vars), _truth("Truth"), _defCtor(kFALSE)
{
  // Constructor from external TTree with given name in given file
  // with specified dimensions and optional string expression cut

  RooTrace::create(this) ;
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
  RooAbsData(other,newName), _truth("Truth"), _defCtor(other._defCtor)
{
  // Copy constructor
  RooTrace::create(this) ;
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
  _tree = new TTree(name, title) ;
  gDirectory->RecursiveRemove(_tree) ;
  gDirectory->cd(pwd) ;
  
}


RooTreeData::~RooTreeData()
{
  // Destructor
  RooTrace::destroy(this) ;

  delete _tree ;
}


void RooTreeData::initialize() {
  // Attach variables of internal ArgSet 
  // to the corresponding TTree branches

  // Attach each variable to the dataset
  _iterator->Reset() ;
  RooAbsArg *var;
  while(0 != (var= (RooAbsArg*)_iterator->Next())) {
    var->attachToTree(*_tree,_defTreeBufSize) ;
  }
}


void RooTreeData::initCache(const RooArgSet& cachedVars) 
{
  // Initialize cache of dataset: attach variables of cache ArgSet
  // to the corresponding TTree branches

  // iterate over the cache variables for this dataset
  TIterator* iter = cachedVars.createIterator() ;
  RooAbsArg *var;
  while(0 != (var= (RooAbsArg*)iter->Next())) {
    var->attachToTree(*_tree,_defTreeBufSize) ;
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


void RooTreeData::loadValues(const RooTreeData *t, RooFormulaVar* select) 
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
    while(arg=(RooAbsArg*)iter->Next()) {
      arg->setOperMode(RooAbsArg::ADirty) ;
    }
    delete iter ;
  }

  // Force RDS internal initialization
  t->get(0) ;

  // Loop over events in source tree   
  RooAbsArg* arg = 0;
  Int_t nevent= t->numEntries() ;
  Bool_t allValid ;
  for(Int_t i=0; i < nevent; ++i) {
    t->_tree->GetEntry(i,1) ;

    // Does this event pass the cuts?
    if (select && select->getVal()==0) {
      continue ; 
    }
    
    _vars = t->_vars ;
    _iterator->Reset() ;

    // Check that all copied values are valid
    allValid=kTRUE ;
    while(arg=(RooAbsArg*)_iterator->Next()) {
      if (!arg->isValid()) {
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


void RooTreeData::loadValues(const TTree *t, RooFormulaVar* select) 
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
  while (sourceArg=(RooAbsArg*)sourceIter->Next()) {
    sourceArg->attachToTree(*tClone,_defTreeBufSize) ;
  }

  // Redirect formula servers to sourceArgSet
  if (select) {
    select->recursiveRedirectServers(*sourceArgSet) ;

    RooArgSet branchList ;
    select->branchNodeServerList(&branchList) ;
    TIterator* iter = branchList.createIterator() ;
    RooAbsArg* arg ;
    while(arg=(RooAbsArg*)iter->Next()) {
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
     while (destArg = (RooAbsArg*)_iterator->Next()) {              
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
  while (arg = (RooAbsArg*)_iterator->Next()) {       
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
    while (arg = (RooAbsArg*)_iterator->Next()) {
     arg->writeToStream(cout,kTRUE) ; 
     cout << " " ;
     }  
    cout << endl ;
  }
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
    
  while (arg=(RooAbsArg*)iter->Next()) {
    // Attach original newVar to this tree
    arg->attachToTree(*_tree,_defTreeBufSize) ;
    _cachedVars.add(*arg) ;
  }


  // Refill regular and cached variables of current tree from clone
  for (int i=0 ; i<GetEntries() ; i++) {
    get(i) ;

    // Evaluate the cached variables and store the results
    iter->Reset() ;
    while (arg=(RooAbsArg*)iter->Next()) {
      arg->setValueDirty() ;
      arg->syncCache(nset) ;
      arg->fillTreeBranch(*_tree) ;
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
    while (var=(RooAbsArg*)_iterator->Next()) {
      var->setValueDirty() ; // This triggers recalculation of all clients
    } 
    
    _cacheIter->Reset() ;
    while (var=(RooAbsArg*)_cacheIter->Next()) {
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

  while(var=(RooAbsArg*)vIter->Next()) {
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
    while(clone=(RooAbsArg*)cIter->Next()) {
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
  
  // Loop over dataset and copy event to matching subset
  Int_t i ;
  for (i=0 ; i<numEntries() ; i++) {
    const RooArgSet* row =  get(i) ;
    RooAbsData* subset = (RooAbsData*) dsetList->FindObject(cloneCat->getLabel()) ;
    if (!subset) {
      subset = emptyClone(cloneCat->getLabel(),cloneCat->getLabel()) ;
      dsetList->Add((RooAbsArg*)subset) ;
    }
    subset->add(*row) ;
  }

  delete cloneSet ;
  return dsetList ;
}




RooPlot *RooTreeData::plotOn(RooPlot *frame, const RooFormulaVar* cutVar, Option_t* drawOptions, const RooAbsBinning* bins) const 
{
  // Implementation pending...
  return 0 ;
}



RooPlot *RooTreeData::plotOn(RooPlot *frame, const char* cuts, Option_t* drawOptions, const RooAbsBinning* bins) const 
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
  if (bins) {
    hist= var->createHistogram(histName.Data(), "Events", *bins) ;
  } else {
    hist= var->createHistogram(histName.Data(), "Events", 
			       frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(), frame->GetNbinsX());
  }
  if(0 == fillHistogram(hist,RooArgList(*var),cuts)) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: createHistogram() failed" << endl;
    return 0;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist);
  if(0 == graph) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: unable to create a RooHist object" << endl;
    delete hist;
    return 0;
  }

  // Store the number of entries before the cut, if any was made
  if (cuts) graph->setRawEntries(numEntries(kTRUE)) ;

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);

  // add the RooHist to the specified plot
  frame->addPlotable(graph,drawOptions);

  // cleanup
  delete hist;

  return frame;  
}




RooPlot* RooTreeData::plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat,
				 const char* cuts, Option_t* drawOptions, const RooAbsBinning* bins) const 
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

  if (bins) {
    hist1= var->createHistogram(hist1Name.Data(), "Events", *bins) ;
    hist2= var->createHistogram(hist2Name.Data(), "Events", *bins) ;
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
  if (cuts && strlen(cuts)) {
    cuts1 = Form("(%s)&&(%s>0)",cuts,asymCat.GetName());
    cuts2 = Form("(%s)&&(%s<0)",cuts,asymCat.GetName());
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
  RooHist *graph= new RooHist(*hist1,*hist2);
  if(0 == graph) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: unable to create a RooHist object" << endl;
    delete hist1;
    delete hist2;
    return 0;
  }

  // add the RooHist to the specified plot
  frame->addPlotable(graph,drawOptions);

  // cleanup
  delete hist1;
  delete hist2;

  return frame;  
}

TH1 *RooTreeData::fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts) const
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

  // Loop over events and fill the histogram
  Int_t nevent= (Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {
    Int_t entryNumber= _tree->GetEntryNumber(i);
    if (entryNumber<0) break;
    get(entryNumber);

    if (select && select->eval()==0) {
      continue ;
    }

    Int_t bin(0);
    switch(hdim) {
    case 1:
      bin= hist->FindBin(xvar->getVal());
      break;
    case 2:
      bin= hist->FindBin(xvar->getVal(),yvar->getVal());
      break;
    case 3:
      bin= hist->FindBin(xvar->getVal(),yvar->getVal(),zvar->getVal());
      break;
    default:
      assert(hdim < 3);
      break;
    }
    hist->AddBinContent(bin,weight());
  }

  if(0 != select) delete select;

  return hist;
}



Roo1DTable* RooTreeData::table(const RooAbsCategory& cat, const char* cuts, const char* opts) const
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



Double_t RooTreeData::moment(RooRealVar &var, Double_t order, Double_t offset) const
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
  if(numEntries(kTRUE) <= 0) {
    cout << "RooDataSet::moment(" << GetName() << ") WARNING: empty dataset" << endl ;
    return 0;
  }

  // Calculate requested moment
  Double_t sum(0);
  for(Int_t index= 0; index < numEntries(); index++) {
    get(index) ;
    sum+= weight() * pow(varPtr->getVal() - offset,order);
  }
  return sum/numEntries(kTRUE);
}



RooRealVar* RooTreeData::meanVar(RooRealVar &var) const
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
  Double_t meanVal=moment(var,1) ;
  Double_t N(numEntries(kTRUE)) ;

  Double_t rmsVal= sqrt(moment(var,2,meanVal)*N/(N-1));
  mean->setVal(meanVal) ;
  mean->setError(N > 0 ? rmsVal/sqrt(N) : 0);

  return mean;
}



RooRealVar* RooTreeData::rmsVar(RooRealVar &var) const
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
  Double_t N(numEntries(kTRUE));
  Double_t rmsVal= sqrt(moment(var,2,meanVal)*N/(N-1));
  rms->setVal(rmsVal) ;
  rms->setError(rmsVal/sqrt(2*N));

  return rms;
}



RooPlot* RooTreeData::statOn(RooPlot* frame, const char* what, const char *label, Int_t sigDigits,
			     Option_t *options, Double_t xmin, Double_t xmax, Double_t ymax) 
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
  RooRealVar N("N","Number of Events",numEntries(kTRUE));
  RooRealVar *mean= meanVar(*(RooRealVar*)frame->getPlotVar());
  RooRealVar *rms= rmsVar(*(RooRealVar*)frame->getPlotVar());
  TString *rmsText= rms->format(sigDigits,options);
  TString *meanText= mean->format(sigDigits,options);
  TString *NText= N.format(sigDigits,options);
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



void RooTreeData::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this dataset to the specified output stream.
  //
  //   Standard: number of entries
  //      Shape: list of variables we define & were generated with

  oneLinePrint(os,*this);
  if(opt >= Standard) {
    os << indent << "  Contains " << numEntries(kTRUE) << " entries" << endl;
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






