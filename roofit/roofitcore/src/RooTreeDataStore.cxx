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

/**
\file RooTreeDataStore.cxx
\class RooTreeDataStore
\ingroup Roofitcore

RooTreeDataStore is a TTree-backed data storage. When a file is opened before
creating the data storage, the storage will be file-backed. This reduces memory
pressure because it allows storing the data in the file and reading it on demand.
For a completely memory-backed storage, which is faster than the file-backed storage,
RooVectorDataStore can be used.

With tree-backed storage, the tree can be found in the file with the name
`RooTreeDataStore_name_title` for a dataset created as
`RooDataSet("name", "title", ...)`.

\note A file needs to be opened **before** creating the data storage to enable file-backed
storage.
```
TFile outputFile("filename.root", "RECREATE");
RooAbsData::setDefaultStorageType(RooAbsData::Tree);
RooDataSet mydata(...);
```

One can also change between TTree- and std::vector-backed storage using
RooAbsData::convertToTreeStore() and
RooAbsData::convertToVectorStore().
**/

#include "RooTreeDataStore.h"

#include "RooMsgService.h"
#include "RooFormulaVar.h"
#include "RooRealVar.h"
#include "RooHistError.h"

#include "ROOT/StringUtils.hxx"

#include "TTree.h"
#include "TFile.h"
#include "TChain.h"
#include "TDirectory.h"
#include "TBuffer.h"
#include "TBranch.h"
#include "TROOT.h"

#include <iomanip>
using namespace std ;

ClassImp(RooTreeDataStore);


Int_t RooTreeDataStore::_defTreeBufSize = 10*1024*1024;



////////////////////////////////////////////////////////////////////////////////

RooTreeDataStore::RooTreeDataStore() : _defCtor(true) {}



////////////////////////////////////////////////////////////////////////////////
/// Constructor to facilitate reading of legacy RooDataSets

RooTreeDataStore::RooTreeDataStore(TTree* t, const RooArgSet& vars, const char* wgtVarName) :
  RooAbsDataStore("blah","blah",varsNoWeight(vars,wgtVarName)),
  _tree(t),
  _defCtor(true),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName))
{
}




////////////////////////////////////////////////////////////////////////////////

RooTreeDataStore::RooTreeDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName))
{
  initialize() ;
}


////////////////////////////////////////////////////////////////////////////////

RooTreeDataStore::RooTreeDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, TTree& t, const char* selExpr, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName))
{
  initialize() ;

  if (selExpr && *selExpr) {
    // Create a RooFormulaVar cut from given cut expression
    RooFormulaVar select(selExpr, selExpr, _vars, /*checkVariables=*/false);
    loadValues(&t,&select);
  } else {
    loadValues(&t);
  }
}


////////////////////////////////////////////////////////////////////////////////

RooTreeDataStore::RooTreeDataStore(RooStringView name, RooStringView title, const RooArgSet& vars, const RooAbsDataStore& ads, const char* selExpr, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName))
{
  initialize() ;

  if (selExpr && *selExpr) {
    // Create a RooFormulaVar cut from given cut expression
    RooFormulaVar select(selExpr, selExpr, _vars, /*checkVariables=*/false);
    loadValues(&ads,&select);
  } else {
    loadValues(&ads);
  }
}




////////////////////////////////////////////////////////////////////////////////

RooTreeDataStore::RooTreeDataStore(RooStringView name, RooStringView title, RooAbsDataStore& tds,
          const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
          Int_t nStart, Int_t nStop, const char* wgtVarName) :
  RooAbsDataStore(name,title,varsNoWeight(vars,wgtVarName)), _defCtor(false),
  _varsww(vars),
  _wgtVar(weightVar(vars,wgtVarName))
{
  // WVE NEED TO ADJUST THIS FOR WEIGHTS

  // Protected constructor for internal use only
  _tree = 0 ;
  _cacheTree = 0 ;
  createTree(makeTreeName(), title);

  // Deep clone cutVar and attach clone to this dataset
  std::unique_ptr<RooFormulaVar> cloneVar;
  if (cutVar) {
    cloneVar.reset(static_cast<RooFormulaVar*>(cutVar->cloneTree()));
    cloneVar->attachDataStore(tds) ;
  }

  // Constructor from existing data set with list of variables that preserves the cache
  initialize();

  attachCache(0,((RooTreeDataStore&)tds)._cachedVars) ;

  // WVE copy values of cached variables here!!!
  _cacheTree->CopyEntries(((RooTreeDataStore&)tds)._cacheTree) ;
  _cacheOwner = 0 ;

  loadValues(&tds,cloneVar.get(),cutRange,nStart,nStop);
}


std::unique_ptr<RooAbsDataStore> RooTreeDataStore::reduce(RooStringView name, RooStringView title,
                        const RooArgSet& vars, const RooFormulaVar* cutVar, const char* cutRange,
                        std::size_t nStart, std::size_t nStop) {
  RooArgSet tmp(vars) ;
  if(_wgtVar && !tmp.contains(*_wgtVar)) {
    tmp.add(*_wgtVar) ;
  }
  const char* wgtVarName = _wgtVar ? _wgtVar->GetName() : nullptr;
  return std::make_unique<RooTreeDataStore>(name, title, *this, tmp, cutVar, cutRange, nStart, nStop, wgtVarName);
}


////////////////////////////////////////////////////////////////////////////////
/// Utility function for constructors
/// Return RooArgSet that is copy of allVars minus variable matching wgtName if specified

RooArgSet RooTreeDataStore::varsNoWeight(const RooArgSet& allVars, const char* wgtName)
{
  RooArgSet ret(allVars) ;
  if(wgtName) {
    RooAbsArg* wgt = allVars.find(wgtName) ;
    if (wgt) {
      ret.remove(*wgt,true,true) ;
    }
  }
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for constructors
/// Return pointer to weight variable if it is defined

RooRealVar* RooTreeDataStore::weightVar(const RooArgSet& allVars, const char* wgtName)
{
  if(wgtName) {
    RooRealVar* wgt = dynamic_cast<RooRealVar*>(allVars.find(wgtName)) ;
    return wgt ;
  }
  return 0 ;
}




////////////////////////////////////////////////////////////////////////////////
/// Initialize cache of dataset: attach variables of cache ArgSet
/// to the corresponding TTree branches

void RooTreeDataStore::attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVarsIn)
{
  // iterate over the cache variables for this dataset
  _cachedVars.removeAll() ;
  for (RooAbsArg * var : cachedVarsIn) {
    var->attachToTree(*_cacheTree,_defTreeBufSize) ;
    _cachedVars.add(*var) ;
  }
  _cacheOwner = newOwner ;

}






////////////////////////////////////////////////////////////////////////////////

RooTreeDataStore::RooTreeDataStore(const RooTreeDataStore& other, const char* newname) :
  RooAbsDataStore(other,newname),
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


////////////////////////////////////////////////////////////////////////////////

RooTreeDataStore::RooTreeDataStore(const RooTreeDataStore& other, const RooArgSet& vars, const char* newname) :
  RooAbsDataStore(other,varsNoWeight(vars,other._wgtVar?other._wgtVar->GetName():0),newname),
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




////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooTreeDataStore::~RooTreeDataStore()
{
  if (_tree) {
    delete _tree ;
  }
  if (_cacheTree) {
    delete _cacheTree ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// One-time initialization common to all constructor forms.  Attach
/// variables of internal ArgSet to the corresponding TTree branches

void RooTreeDataStore::initialize()
{
  // Recreate (empty) cache tree
  createTree(makeTreeName(), GetTitle());

  // Attach each variable to the dataset
  for (auto var : _varsww) {
    var->attachToTree(*_tree,_defTreeBufSize) ;
  }
}





////////////////////////////////////////////////////////////////////////////////
/// Create TTree object that lives in memory, independent of current
/// location of gDirectory

void RooTreeDataStore::createTree(RooStringView name, RooStringView title)
{
  if (!_tree) {
    _tree = new TTree(name,title);
    _tree->ResetBit(kCanDelete);
    _tree->ResetBit(kMustCleanup);
    _tree->SetDirectory(nullptr);
  }

  TString pwd(gDirectory->GetPath()) ;
  TString memDir(gROOT->GetName()) ;
  memDir.Append(":/") ;
  bool notInMemNow= (pwd!=memDir) ;

  // cout << "RooTreeData::createTree pwd=" << pwd << " memDir=" << memDir << " notInMemNow = " << (notInMemNow?"T":"F") << endl ;

  if (notInMemNow) {
    gDirectory->cd(memDir) ;
  }

  if (!_cacheTree) {
    _cacheTree = new TTree(TString{static_cast<const char*>(name)} + "_cacheTree", TString{static_cast<const char*>(title)});
    _cacheTree->SetDirectory(0) ;
    gDirectory->RecursiveRemove(_cacheTree) ;
  }

  if (notInMemNow) {
    gDirectory->cd(pwd) ;
  }

}




////////////////////////////////////////////////////////////////////////////////
/// Load values from tree 't' into this data collection, optionally
/// selecting events using the RooFormulaVar 'select'.
///
/// The source tree 't' is cloned to not disturb its branch
/// structure when retrieving information from it.
void RooTreeDataStore::loadValues(const TTree *t, const RooFormulaVar* select, const char* /*rangeName*/, Int_t /*nStart*/, Int_t /*nStop*/)
{
  // Make our local copy of the tree, so we can safely loop through it.
  // We need a custom deleter, because if we don't deregister the Tree from the directory
  // of the original, it tears it down at destruction time!
  auto deleter = [](TTree* tree){tree->SetDirectory(nullptr); delete tree;};
  std::unique_ptr<TTree, decltype(deleter)> tClone(static_cast<TTree*>(t->Clone()), deleter);
  tClone->SetDirectory(t->GetDirectory());

  // Clone list of variables
  std::unique_ptr<RooArgSet> sourceArgSet( _varsww.snapshot(false) );

  // Check that we have the branches:
  bool missingBranches = false;
  for (const auto var : *sourceArgSet) {
     if (!tClone->GetBranch(var->GetName())) {
        missingBranches = true;
        coutE(InputArguments) << "Didn't find a branch in Tree '" << tClone->GetName() << "' to read variable '"
                              << var->GetName() << "' from."
                              << "\n\tNote: Name the RooFit variable the same as the branch." << std::endl;
     }
  }
  if (missingBranches) {
     coutE(InputArguments) << "Cannot import data from TTree '" << tClone->GetName()
                           << "' because some branches are missing !" << std::endl;
     return;
  }

  // Attach args in cloned list to cloned source tree
  for (const auto sourceArg : *sourceArgSet) {
    sourceArg->attachToTree(*tClone,_defTreeBufSize) ;
  }

  // Redirect formula servers to sourceArgSet
  std::unique_ptr<RooFormulaVar> selectClone;
  if (select) {
    selectClone.reset( static_cast<RooFormulaVar*>(select->cloneTree()) );
    selectClone->recursiveRedirectServers(*sourceArgSet) ;
    selectClone->setOperMode(RooAbsArg::ADirty,true) ;
  }

  // Loop over events in source tree
  Int_t numInvalid(0) ;
  const Long64_t nevent = tClone->GetEntries();
  for(Long64_t i=0; i < nevent; ++i) {
    const auto entryNumber = tClone->GetEntryNumber(i);
    if (entryNumber<0) break;
    tClone->GetEntry(entryNumber,1);

    // Copy from source to destination
    bool allOK(true) ;
    for (unsigned int j=0; j < sourceArgSet->size(); ++j) {
      auto destArg = _varsww[j];
      const auto sourceArg = (*sourceArgSet)[j];

      destArg->copyCache(sourceArg) ;
      sourceArg->copyCache(destArg) ;
      if (!destArg->isValid()) {
        numInvalid++ ;
        allOK=false ;
        if (numInvalid < 5) {
          auto& log = coutI(DataHandling);
          log << "RooTreeDataStore::loadValues(" << GetName() << ") Skipping event #" << i << " because " << destArg->GetName()
              << " cannot accommodate the value ";
          if(sourceArg->isCategory()) {
            log << static_cast<RooAbsCategory*>(sourceArg)->getCurrentIndex();
          } else {
            log << static_cast<RooAbsReal*>(sourceArg)->getVal();
          }
          log << std::endl;
        } else if (numInvalid == 5) {
          coutI(DataHandling) << "RooTreeDataStore::loadValues(" << GetName() << ") Skipping ..." << std::endl;
        }
        break ;
      }
    }

    // Does this event pass the cuts?
    if (!allOK || (selectClone && selectClone->getVal()==0)) {
      continue ;
    }

    fill() ;
  }

  if (numInvalid>0) {
    coutW(DataHandling) << "RooTreeDataStore::loadValues(" << GetName() << ") Ignored " << numInvalid << " out-of-range events" << endl ;
  }

  SetTitle(t->GetTitle());
}






////////////////////////////////////////////////////////////////////////////////
/// Load values from dataset 't' into this data collection, optionally
/// selecting events using 'select' RooFormulaVar
///

void RooTreeDataStore::loadValues(const RooAbsDataStore *ads, const RooFormulaVar* select,
              const char* rangeName, std::size_t nStart, std::size_t nStop)
{
  // Redirect formula servers to source data row
  std::unique_ptr<RooFormulaVar> selectClone;
  if (select) {
    selectClone.reset( static_cast<RooFormulaVar*>(select->cloneTree()) );
    selectClone->recursiveRedirectServers(*ads->get()) ;
    selectClone->setOperMode(RooAbsArg::ADirty,true) ;
  }

  // Force RDS internal initialization
  ads->get(0) ;

  // Loop over events in source tree
  const auto numEntr = static_cast<std::size_t>(ads->numEntries());
  std::size_t nevent = nStop < numEntr ? nStop : numEntr;

  auto TDS = dynamic_cast<const RooTreeDataStore*>(ads) ;
  if (TDS) {
    const_cast<RooTreeDataStore*>(TDS)->resetBuffers();
  }

  std::vector<std::string> ranges;
  if (rangeName) {
   ranges = ROOT::Split(rangeName, ",");
  }

  for (auto i=nStart; i < nevent ; ++i) {
    ads->get(i) ;

    // Does this event pass the cuts?
    if (selectClone && selectClone->getVal()==0) {
      continue ;
    }


    if (TDS) {
      _varsww.assignValueOnly(TDS->_varsww) ;
    } else {
      _varsww.assignValueOnly(*ads->get()) ;
    }

    // Check that all copied values are valid
    bool allValid = true;
    for (const auto arg : _varsww) {
      allValid = arg->isValid() && (ranges.empty() || std::any_of(ranges.begin(), ranges.end(),
          [arg](const std::string& range){return arg->inRange(range.c_str());}) );
      if (!allValid)
        break ;
    }

    if (!allValid) {
      continue ;
    }

    _cachedVars.assign(((RooTreeDataStore*)ads)->_cachedVars) ;
    fill() ;
  }

  if (TDS) {
    const_cast<RooTreeDataStore*>(TDS)->restoreAlternateBuffers();
  }

  SetTitle(ads->GetTitle());
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function to TTree::Fill

Int_t RooTreeDataStore::fill()
{
   return _tree->Fill() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Load the n-th data point (n='index') in memory
/// and return a pointer to the internal RooArgSet
/// holding its coordinates.

const RooArgSet* RooTreeDataStore::get(Int_t index) const
{
  checkInit() ;

  Int_t ret = ((RooTreeDataStore*)this)->GetEntry(index, 1) ;

  if(!ret) return 0;

  if (_doDirtyProp) {
    // Raise all dirty flags
    for (auto var : _vars) {
      var->setValueDirty(); // This triggers recalculation of all clients
    }

    for (auto var : _cachedVars) {
      var->setValueDirty(); // This triggers recalculation of all clients, but doesn't recalculate self
      var->clearValueDirty();
    }
  }

  // Update current weight cache
  if (_extWgtArray) {

    // If external array is specified use that
    _curWgt = _extWgtArray[index] ;
    _curWgtErrLo = _extWgtErrLoArray ? _extWgtErrLoArray[index] : -1.;
    _curWgtErrHi = _extWgtErrHiArray ? _extWgtErrHiArray[index] : -1.;
    _curWgtErr   = sqrt( _extSumW2Array ? _extSumW2Array[index] : _extWgtArray[index] );

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


////////////////////////////////////////////////////////////////////////////////
/// Return the weight of the n-th data point (n='index') in memory

double RooTreeDataStore::weight() const
{
  return _curWgt ;
}


////////////////////////////////////////////////////////////////////////////////

double RooTreeDataStore::weightError(RooAbsData::ErrorType etype) const
{
  if (_extWgtArray) {

    // We have a weight array, use that info

    // Return symmetric error on current bin calculated either from Poisson statistics or from SumOfWeights
    double lo = 0, hi =0;
    weightError(lo,hi,etype) ;
    return (lo+hi)/2 ;

   } else if (_wgtVar) {

    // We have a weight variable, use that info
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



////////////////////////////////////////////////////////////////////////////////

void RooTreeDataStore::weightError(double& lo, double& hi, RooAbsData::ErrorType etype) const
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
      double ym,yp ;
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

    // We have a weight variable, use that info
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


////////////////////////////////////////////////////////////////////////////////
/// Change name of internal observable named 'from' into 'to'

bool RooTreeDataStore::changeObservableName(const char* from, const char* to)
{
  // Find observable to be changed
  RooAbsArg* var = _vars.find(from) ;

  // Check that we found it
  if (!var) {
    coutE(InputArguments) << "RooTreeDataStore::changeObservableName(" << GetName() << " no observable " << from << " in this dataset" << endl ;
    return true ;
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

  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Add a new column to the data set which holds the pre-calculated values
/// of 'newVar'. This operation is only meaningful if 'newVar' is a derived
/// value.
///
/// The return value points to the added element holding 'newVar's value
/// in the data collection. The element is always the corresponding fundamental
/// type of 'newVar' (e.g. a RooRealVar if 'newVar' is a RooFormulaVar)
///
/// Note: This function is explicitly NOT intended as a speed optimization
///       opportunity for the user. Components of complex PDFs that can be
///       precalculated with the dataset are automatically identified as such
///       and will be precalculated when fitting to a dataset
///
///       By forcibly precalculating functions with non-trivial Jacobians,
///       or functions of multiple variables occurring in the data set,
///       using addColumn(), you may alter the outcome of the fit.
///
///       Only in cases where such a modification of fit behaviour is intentional,
///       this function should be used.

RooAbsArg* RooTreeDataStore::addColumn(RooAbsArg& newVar, bool adjustRange)
{
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
  newVarClone->recursiveRedirectServers(_vars,false) ;

  // Attach value place holder to this tree
  ((RooAbsArg*)valHolder)->attachToTree(*_tree,_defTreeBufSize) ;
  _vars.add(*valHolder) ;
  _varsww.add(*valHolder) ;


  // Fill values of placeholder
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
//     double vlo,vhi ;
//     RooRealVar* rrvVal = dynamic_cast<RooRealVar*>(valHolder) ;
//     if (rrvVal) {
//       getRange(*rrvVal,vlo,vhi,0.05) ;
//       rrvVal->setRange(vlo,vhi) ;
//     }
  }



  delete newVarClone ;
  return valHolder ;
}


////////////////////////////////////////////////////////////////////////////////
/// Merge columns of supplied data set(s) with this data set.  All
/// data sets must have equal number of entries.  In case of
/// duplicate columns the column of the last dataset in the list
/// prevails

RooAbsDataStore* RooTreeDataStore::merge(const RooArgSet& allVars, list<RooAbsDataStore*> dstoreList)
{
  RooTreeDataStore* mergedStore = new RooTreeDataStore("merged","merged",allVars) ;

  Int_t nevt = dstoreList.front()->numEntries() ;
  for (int i=0 ; i<nevt ; i++) {

    // Cope data from self
    mergedStore->_vars.assign(*get(i)) ;

    // Copy variables from merge sets
    for (list<RooAbsDataStore*>::iterator iter = dstoreList.begin() ; iter!=dstoreList.end() ; ++iter) {
      const RooArgSet* partSet = (*iter)->get(i) ;
      mergedStore->_vars.assign(*partSet) ;
    }

    mergedStore->fill() ;
  }
  return mergedStore ;
}





////////////////////////////////////////////////////////////////////////////////

void RooTreeDataStore::append(RooAbsDataStore& other)
{
  Int_t nevt = other.numEntries() ;
  for (int i=0 ; i<nevt ; i++) {
    _vars.assign(*other.get(i)) ;
    if (_wgtVar) {
      _wgtVar->setVal(other.weight()) ;
    }

    fill() ;
  }
}


////////////////////////////////////////////////////////////////////////////////

double RooTreeDataStore::sumEntries() const
{
  if (_wgtVar) {

    double sum(0), carry(0);
    Int_t nevt = numEntries() ;
    for (int i=0 ; i<nevt ; i++) {
      get(i) ;
      // Kahan's algorithm for summing to avoid loss of precision
      double y = _wgtVar->getVal() - carry;
      double t = sum + y;
      carry = (t - sum) - y;
      sum = t;
    }
    return sum ;

  } else if (_extWgtArray) {

    double sum(0) , carry(0);
    Int_t nevt = numEntries() ;
    for (int i=0 ; i<nevt ; i++) {
      // Kahan's algorithm for summing to avoid loss of precision
      double y = _extWgtArray[i] - carry;
      double t = sum + y;
      carry = (t - sum) - y;
      sum = t;
    }
    return sum ;

  } else {

    return numEntries() ;

  }
}




////////////////////////////////////////////////////////////////////////////////

Int_t RooTreeDataStore::numEntries() const
{
  return _tree->GetEntries() ;
}



////////////////////////////////////////////////////////////////////////////////

void RooTreeDataStore::reset()
{
  Reset() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Cache given RooAbsArgs with this tree: The tree is
/// given direct write access of the args internal cache
/// the args values is pre-calculated for all data points
/// in this data collection. Upon a get() call, the
/// internal cache of 'newVar' will be loaded with the
/// precalculated value and it's dirty flag will be cleared.

void RooTreeDataStore::cacheArgs(const RooAbsArg* owner, RooArgSet& newVarSet, const RooArgSet* nset, bool /*skipZeroWeights*/)
{
  checkInit() ;

  _cacheOwner = owner ;

  std::unique_ptr<RooArgSet> constExprVarSet{static_cast<RooArgSet*>(newVarSet.selectByAttrib("ConstantExpression",true))};

  bool doTreeFill = (_cachedVars.empty()) ;

  for (RooAbsArg * arg : *constExprVarSet) {
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
    for (RooAbsArg * arg : *constExprVarSet) {
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
}




////////////////////////////////////////////////////////////////////////////////
/// Activate or deactivate the branch status of the TTree branch associated
/// with the given set of dataset observables

void RooTreeDataStore::setArgStatus(const RooArgSet& set, bool active)
{
  for (RooAbsArg * arg : set) {
    RooAbsArg* depArg = _vars.find(arg->GetName()) ;
    if (!depArg) {
      coutE(InputArguments) << "RooTreeDataStore::setArgStatus(" << GetName()
             << ") dataset doesn't contain variable " << arg->GetName() << endl ;
      continue ;
    }
    depArg->setTreeBranchStatus(*_tree,active) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Remove tree with values of cached observables
/// and clear list of cached observables

void RooTreeDataStore::resetCache()
{
  // Empty list of cached functions
  _cachedVars.removeAll() ;

  // Delete & recreate cache tree
  delete _cacheTree ;
  _cacheTree = 0 ;
  createTree(makeTreeName().c_str(), GetTitle());

  return ;
}




////////////////////////////////////////////////////////////////////////////////

void RooTreeDataStore::attachBuffers(const RooArgSet& extObs)
{
  _attachedBuffers.removeAll() ;
  for (const auto arg : _varsww) {
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



////////////////////////////////////////////////////////////////////////////////

void RooTreeDataStore::resetBuffers()
{
  for(RooAbsArg * arg : _varsww) {
    arg->attachToTree(*_tree) ;
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooTreeDataStore::restoreAlternateBuffers()
{
  for(RooAbsArg * arg : _attachedBuffers) {
    arg->attachToTree(*_tree) ;
  }
}



////////////////////////////////////////////////////////////////////////////////

void RooTreeDataStore::checkInit() const
{
  if (_defCtor) {
    const_cast<RooTreeDataStore*>(this)->initialize() ;
    _defCtor = false ;
  }
}





////////////////////////////////////////////////////////////////////////////////
/// Interface function to TTree::GetEntries

Stat_t RooTreeDataStore::GetEntries() const
{
   return _tree->GetEntries() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function to TTree::Reset

void RooTreeDataStore::Reset(Option_t* option)
{
   _tree->Reset(option) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function to TTree::Fill

Int_t RooTreeDataStore::Fill()
{
   return _tree->Fill() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Interface function to TTree::GetEntry

Int_t RooTreeDataStore::GetEntry(Int_t entry, Int_t getall)
{
   Int_t ret1 = _tree->GetEntry(entry,getall) ;
   if (!ret1) return 0 ;
   _cacheTree->GetEntry(entry,getall) ;
   return ret1 ;
}


////////////////////////////////////////////////////////////////////////////////

void RooTreeDataStore::Draw(Option_t* option)
{
  _tree->Draw(option) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class RooTreeDataStore.

void RooTreeDataStore::Streamer(TBuffer &R__b)
{
  if (R__b.IsReading()) {
    UInt_t R__s, R__c;
    const Version_t R__v = R__b.ReadVersion(&R__s, &R__c);

    R__b.ReadClassBuffer(RooTreeDataStore::Class(), this, R__v, R__s, R__c);

    if (!_tree) {
      // If the tree has not been deserialised automatically, it is time to load
      // it now.
      TFile* parent = dynamic_cast<TFile*>(R__b.GetParent());
      assert(parent);
      parent->GetObject(makeTreeName().c_str(), _tree);
    }

    initialize();

  } else {

    TTree* tmpTree = _tree;
    auto parent = dynamic_cast<TDirectory*>(R__b.GetParent());
    if (_tree && parent) {
      // Large trees cannot be written because of the 1Gb I/O limitation.
      // Here, we take the tree away from our instance, write it, and continue
      // to write the rest of the class normally
      auto tmpDir = _tree->GetDirectory();

      _tree->SetDirectory(parent);
      _tree->FlushBaskets(false);
      parent->WriteObject(_tree, makeTreeName().c_str());
      _tree->SetDirectory(tmpDir);
      _tree = nullptr;
    }

    R__b.WriteClassBuffer(RooTreeDataStore::Class(), this);

    _tree = tmpTree;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Generate a name for the storage tree from the name and title of this instance.
std::string RooTreeDataStore::makeTreeName() const {
  std::string title = GetTitle();
  std::replace(title.begin(), title.end(), ' ', '_');
  std::replace(title.begin(), title.end(), '-', '_');
  return std::string("RooTreeDataStore_") + GetName() + "_" + title;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the weights of the events in the range [first, first+len).
/// This implementation will fill a vector with every event retrieved one by one
/// (even if the weight is constant). Then, it returns a span.
RooSpan<const double> RooTreeDataStore::getWeightBatch(std::size_t first, std::size_t len) const {

  if (_extWgtArray) {
    return {_extWgtArray + first, len};
  }

  if (!_weightBuffer) {
    _weightBuffer = std::make_unique<std::vector<double>>();
    _weightBuffer->reserve(len);

    for (std::size_t i = 0; i < GetEntries(); ++i) {
      _weightBuffer->push_back(weight(i));
    }
  }

  return {_weightBuffer->data() + first, len};
}
