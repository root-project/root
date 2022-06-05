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
\file RooAbsData.cxx
\class RooAbsData
\ingroup Roofitcore

RooAbsData is the common abstract base class for binned and unbinned
datasets. The abstract interface defines plotting and tabulating entry
points for its contents and provides an iterator over its elements
(bins for binned data sets, data points for unbinned datasets).

### Storing global observables in RooFit datasets

RooFit groups model variables into *observables* and *parameters*, depending on
if their values are stored in the dataset. For fits with parameter
constraints, there is a third kind of variables, called *global observables*.
These represent the results of auxiliary measurements that constrain the
nuisance parameters. In the RooFit implementation, a likelihood is generally
the sum of two terms:
- the likelihood of the data given the parameters, where the normalization set
  is the set of observables (implemented by RooNLLVar)
- the constraint term, where the normalization set is the set of *global
observables* (implemented by RooConstraintSum)

Before this release, the global observable values were always taken from the
model/pdf. With this release, a mechanism is added to store a snapshot of
global observables in any RooDataSet or RooDataHist. For toy studies where the
global observables assume a different values for each toy, the bookkeeping of
the set of global observables and in particular their values is much easier
with this change.

Usage example for a model with global observables `g1` and `g2`:
```
auto data = model.generate(x, 1000); // data has only the single observables x
data->setGlobalObservables(g1, g2); // now, data also stores a snapshot of g1 and g2

// If you fit the model to the data, the global observables and their values
// are taken from the dataset:
model.fitTo(*data);

// You can still define the set of global observables yourself, but the values
// will be takes from the dataset if available:
model.fitTo(*data, GlobalObservables(g1, g2));

// To force `fitTo` to take the global observable values from the model even
// though they are in the dataset, you can use the new `GlobalObservablesSource`
// command argument:
model.fitTo(*data, GlobalObservables(g1, g2), GlobalObservablesSource("model"));
// The only other allowed value for `GlobalObservablesSource` is "data", which
// corresponds to the new default behavior explained above.
```

In case you create a RooFit dataset directly by calling its constructor, you
can also pass the global observables in a command argument instead of calling
RooAbsData::setGlobalObservables() later:
```
RooDataSet data{"dataset", "dataset", x, RooFit::GlobalObservables(g1, g2)};
```

To access the set of global observables stored in a RooAbsData, call
RooAbsData::getGlobalObservables(). It returns a `nullptr` if no global
observable snapshots are stored in the dataset.
**/

#include "RooAbsData.h"

#include "TBuffer.h"
#include "TClass.h"
#include "TMath.h"
#include "TTree.h"

#include "RooFormulaVar.h"
#include "RooCmdConfig.h"
#include "RooAbsRealLValue.h"
#include "RooMsgService.h"
#include "RooMultiCategory.h"
#include "Roo1DTable.h"
#include "RooAbsDataStore.h"
#include "RooVectorDataStore.h"
#include "RooTreeDataStore.h"
#include "RooDataHist.h"
#include "RooCompositeDataStore.h"
#include "RooCategory.h"
#include "RooTrace.h"
#include "RooUniformBinning.h"
#include "RooSimultaneous.h"

#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooPlot.h"
#include "RooCurve.h"
#include "RooHist.h"
#include "RooHelpers.h"

#include "ROOT/StringUtils.hxx"
#include "TMatrixDSym.h"
#include "TPaveText.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "Math/Util.h"

#include <iostream>
#include <memory>


using namespace std;

ClassImp(RooAbsData);
;

static std::map<RooAbsData*,int> _dcc ;

RooAbsData::StorageType RooAbsData::defaultStorageType=RooAbsData::Vector ;

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::setDefaultStorageType(RooAbsData::StorageType s)
{
   if (RooAbsData::Composite == s) {
      cout << "Composite storage is not a valid *default* storage type." << endl;
   } else {
      defaultStorageType = s;
   }
}

////////////////////////////////////////////////////////////////////////////////

RooAbsData::StorageType RooAbsData::getDefaultStorageType( )
{
  return defaultStorageType;
}

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::claimVars(RooAbsData* data)
{
  _dcc[data]++ ;
  //cout << "RooAbsData(" << data << ") claim incremented to " << _dcc[data] << endl ;
}

////////////////////////////////////////////////////////////////////////////////
/// If return value is true variables can be deleted

bool RooAbsData::releaseVars(RooAbsData* data)
{
  if (_dcc[data]>0) {
    _dcc[data]-- ;
  }

  //cout << "RooAbsData(" << data << ") claim decremented to " << _dcc[data] << endl ;
  return (_dcc[data]==0) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooAbsData::RooAbsData()
{
  claimVars(this) ;
  storageType = defaultStorageType;

  RooTrace::create(this) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a set of variables. Only fundamental elements of vars
/// (RooRealVar,RooCategory etc) are stored as part of the dataset

RooAbsData::RooAbsData(RooStringView name, RooStringView title, const RooArgSet& vars, RooAbsDataStore* dstore) :
  TNamed(name,title),
  _vars("Dataset Variables"),
  _cachedVars("Cached Variables"),
  _dstore(dstore)
{
   if (dynamic_cast<RooTreeDataStore *>(dstore)) {
      storageType = RooAbsData::Tree;
   } else if (dynamic_cast<RooVectorDataStore *>(dstore)) {
      storageType = RooAbsData::Vector;
   } else {
      storageType = RooAbsData::Composite;
   }
   // cout << "created dataset " << this << endl ;
   claimVars(this);

   // clone the fundamentals of the given data set into internal buffer
   for (const auto var : vars) {
      if (!var->isFundamental()) {
         coutE(InputArguments) << "RooAbsDataStore::initialize(" << GetName()
                               << "): Data set cannot contain non-fundamental types, ignoring " << var->GetName()
                               << endl;
         throw std::invalid_argument(std::string("Only fundamental variables can be placed into datasets. This is violated for ") + var->GetName());
      } else {
         _vars.addClone(*var);
      }
   }

   // reconnect any parameterized ranges to internal dataset observables
   for (auto var : _vars) {
      var->attachArgs(_vars);
   }

   _namePtr = RooNameReg::instance().constPtr(GetName()) ;

   RooTrace::create(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsData::RooAbsData(const RooAbsData& other, const char* newname) :
  TNamed(newname ? newname : other.GetName(),other.GetTitle()),
  RooPrintable(other), _vars(),
  _cachedVars("Cached Variables"),
  _namePtr(newname ? RooNameReg::instance().constPtr(newname) : other._namePtr)
{
  //cout << "created dataset " << this << endl ;
  claimVars(this) ;
  _vars.addClone(other._vars) ;

  // reconnect any parameterized ranges to internal dataset observables
  for (auto var : _vars) {
    var->attachArgs(_vars);
  }


  if (other._ownedComponents.size()>0) {

    // copy owned components here

    map<string,RooAbsDataStore*> smap ;
    for (auto& itero : other._ownedComponents) {
      RooAbsData* dclone = (RooAbsData*) itero.second->Clone();
      _ownedComponents[itero.first] = dclone;
      smap[itero.first] = dclone->store();
    }

    RooCategory* idx = (RooCategory*) _vars.find(*((RooCompositeDataStore*)other.store())->index()) ;
    _dstore = new RooCompositeDataStore(newname?newname:other.GetName(),other.GetTitle(),_vars,*idx,smap) ;
    storageType = RooAbsData::Composite;

  } else {

    // Convert to vector store if default is vector
    _dstore = other._dstore->clone(_vars,newname?newname:other.GetName()) ;
    storageType = other.storageType;
  }

  copyGlobalObservables(other);

  RooTrace::create(this) ;
}

RooAbsData& RooAbsData::operator=(const RooAbsData& other) {
  TNamed::operator=(other);
  RooPrintable::operator=(other);

  claimVars(this);
  _vars.Clear();
  _vars.addClone(other._vars);
  _namePtr = other._namePtr;

  // reconnect any parameterized ranges to internal dataset observables
  for (const auto var : _vars) {
    var->attachDataSet(*this) ;
  }


  if (other._ownedComponents.size()>0) {

    // copy owned components here

    map<string,RooAbsDataStore*> smap ;
    for (auto& itero : other._ownedComponents) {
      RooAbsData* dclone = (RooAbsData*) itero.second->Clone();
      _ownedComponents[itero.first] = dclone;
      smap[itero.first] = dclone->store();
    }

    RooCategory* idx = (RooCategory*) _vars.find(*((RooCompositeDataStore*)other.store())->index()) ;
    _dstore = new RooCompositeDataStore(GetName(), GetTitle(), _vars, *idx, smap);
    storageType = RooAbsData::Composite;

  } else {

    // Convert to vector store if default is vector
    _dstore = other._dstore->clone(_vars);
    storageType = other.storageType;
  }

  copyGlobalObservables(other);

  return *this;
}


void RooAbsData::copyGlobalObservables(const RooAbsData& other) {
  if (other._globalObservables) {
    if(_globalObservables == nullptr) _globalObservables = std::make_unique<RooArgSet>();
    else _globalObservables->clear();
    other._globalObservables->snapshot(*_globalObservables);
  } else {
    _globalObservables.reset(nullptr);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsData::~RooAbsData()
{
  if (releaseVars(this)) {
    // will cause content to be deleted subsequently in dtor
  } else {
    _vars.releaseOwnership() ;
  }

  // delete owned contents.
  delete _dstore ;

  // Delete owned dataset components
  for(map<std::string,RooAbsData*>::iterator iter = _ownedComponents.begin() ; iter!= _ownedComponents.end() ; ++iter) {
    delete iter->second ;
  }

  RooTrace::destroy(this) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert tree-based storage to vector-based storage

void RooAbsData::convertToVectorStore()
{
   if (auto treeStore = dynamic_cast<RooTreeDataStore*>(_dstore)) {
      _dstore = new RooVectorDataStore(*treeStore, _vars, GetName());
      delete treeStore;
      storageType = RooAbsData::Vector;
   }
}

////////////////////////////////////////////////////////////////////////////////

bool RooAbsData::changeObservableName(const char* from, const char* to)
{
  bool ret =  _dstore->changeObservableName(from,to) ;

  RooAbsArg* tmp = _vars.find(from) ;
  if (tmp) {
    tmp->SetName(to) ;
  }
  return ret ;
}

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::fill()
{
  _dstore->fill() ;
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooAbsData::numEntries() const
{
  return nullptr != _dstore ? _dstore->numEntries() : 0;
}

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::reset()
{
  _dstore->reset() ;
}

////////////////////////////////////////////////////////////////////////////////

const RooArgSet* RooAbsData::get(Int_t index) const
{
  checkInit() ;
  return _dstore->get(index) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Internal method -- Cache given set of functions with data

void RooAbsData::cacheArgs(const RooAbsArg* cacheOwner, RooArgSet& varSet, const RooArgSet* nset, bool skipZeroWeights)
{
  _dstore->cacheArgs(cacheOwner,varSet,nset,skipZeroWeights) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Internal method -- Remove cached function values

void RooAbsData::resetCache()
{
  _dstore->resetCache() ;
  _cachedVars.removeAll() ;
}

////////////////////////////////////////////////////////////////////////////////
/// Internal method -- Attach dataset copied with cache contents to copied instances of functions

void RooAbsData::attachCache(const RooAbsArg* newOwner, const RooArgSet& cachedVars)
{
  _dstore->attachCache(newOwner, cachedVars) ;
}

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::setArgStatus(const RooArgSet& set, bool active)
{
  _dstore->setArgStatus(set,active) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Control propagation of dirty flags from observables in dataset

void RooAbsData::setDirtyProp(bool flag)
{
  _dstore->setDirtyProp(flag) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a reduced copy of this dataset. The caller takes ownership of the returned dataset
///
/// The following optional named arguments are accepted
/// <table>
/// <tr><td> `SelectVars(const RooArgSet& vars)`   <td> Only retain the listed observables in the output dataset
/// <tr><td> `Cut(const char* expression)`   <td> Only retain event surviving the given cut expression
/// <tr><td> `Cut(const RooFormulaVar& expr)`   <td> Only retain event surviving the given cut formula
/// <tr><td> `CutRange(const char* name)`   <td> Only retain events inside range with given name. Multiple CutRange
///     arguments may be given to select multiple ranges
/// <tr><td> `EventRange(int lo, int hi)`   <td> Only retain events with given sequential event numbers
/// <tr><td> `Name(const char* name)`   <td> Give specified name to output dataset
/// <tr><td> `Title(const char* name)`   <td> Give specified title to output dataset
/// </table>

RooAbsData* RooAbsData::reduce(const RooCmdArg& arg1,const RooCmdArg& arg2,const RooCmdArg& arg3,const RooCmdArg& arg4,
                const RooCmdArg& arg5,const RooCmdArg& arg6,const RooCmdArg& arg7,const RooCmdArg& arg8)
{
  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsData::reduce(%s)",GetName())) ;
  pc.defineString("name","Name",0,"") ;
  pc.defineString("title","Title",0,"") ;
  pc.defineString("cutRange","CutRange",0,"") ;
  pc.defineString("cutSpec","CutSpec",0,"") ;
  pc.defineObject("cutVar","CutVar",0,0) ;
  pc.defineInt("evtStart","EventRange",0,0) ;
  pc.defineInt("evtStop","EventRange",1,std::numeric_limits<int>::max()) ;
  pc.defineObject("varSel","SelectVars",0,0) ;
  pc.defineMutex("CutVar","CutSpec") ;

  // Process & check varargs
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(true)) {
    return nullptr;
  }

  // Extract values from named arguments
  const char* cutRange = pc.getString("cutRange",0,true) ;
  const char* cutSpec = pc.getString("cutSpec",0,true) ;
  RooFormulaVar* cutVar = static_cast<RooFormulaVar*>(pc.getObject("cutVar",0)) ;
  Int_t nStart = pc.getInt("evtStart",0) ;
  Int_t nStop = pc.getInt("evtStop",std::numeric_limits<int>::max()) ;
  RooArgSet* varSet = static_cast<RooArgSet*>(pc.getObject("varSel")) ;
  const char* name = pc.getString("name",0,true) ;
  const char* title = pc.getString("title",0,true) ;

  // Make sure varSubset doesn't contain any variable not in this dataset
  RooArgSet varSubset ;
  if (varSet) {
    varSubset.add(*varSet) ;
    for (const auto arg : varSubset) {
      if (!_vars.find(arg->GetName())) {
        coutW(InputArguments) << "RooAbsData::reduce(" << GetName() << ") WARNING: variable "
            << arg->GetName() << " not in dataset, ignored" << endl ;
        varSubset.remove(*arg) ;
      }
    }
  } else {
    varSubset.add(*get()) ;
  }

  RooAbsData* ret = nullptr;
  if (cutSpec) {

    RooFormulaVar cutVarTmp(cutSpec,cutSpec,*get()) ;
    ret =  reduceEng(varSubset,&cutVarTmp,cutRange,nStart,nStop) ;

  } else if (cutVar) {

    ret = reduceEng(varSubset,cutVar,cutRange,nStart,nStop) ;

  } else {

    ret = reduceEng(varSubset,0,cutRange,nStart,nStop) ;

  }

  if (!ret) return nullptr;

  if (name) ret->SetName(name) ;
  if (title) ret->SetTitle(title) ;

  ret->copyGlobalObservables(*this);
  return ret ;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a subset of the data set by applying the given cut on the data points.
/// The cut expression can refer to any variable in the data set. For cuts involving
/// other variables, such as intermediate formula objects, use the equivalent
/// reduce method specifying the as a RooFormulVar reference.

RooAbsData* RooAbsData::reduce(const char* cut)
{
  RooFormulaVar cutVar(cut,cut,*get()) ;
  RooAbsData* ret = reduceEng(*get(),&cutVar,0,0,std::numeric_limits<std::size_t>::max()) ;
  ret->copyGlobalObservables(*this);
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a subset of the data set by applying the given cut on the data points.
/// The 'cutVar' formula variable is used to select the subset of data points to be
/// retained in the reduced data collection.

RooAbsData* RooAbsData::reduce(const RooFormulaVar& cutVar)
{
  RooAbsData* ret = reduceEng(*get(),&cutVar,0,0,std::numeric_limits<std::size_t>::max()) ;
  ret->copyGlobalObservables(*this);
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a subset of the data set by applying the given cut on the data points
/// and reducing the dimensions to the specified set.
///
/// The cut expression can refer to any variable in the data set. For cuts involving
/// other variables, such as intermediate formula objects, use the equivalent
/// reduce method specifying the as a RooFormulVar reference.

RooAbsData* RooAbsData::reduce(const RooArgSet& varSubset, const char* cut)
{
  // Make sure varSubset doesn't contain any variable not in this dataset
  RooArgSet varSubset2(varSubset) ;
  for (const auto arg : varSubset) {
    if (!_vars.find(arg->GetName())) {
      coutW(InputArguments) << "RooAbsData::reduce(" << GetName() << ") WARNING: variable "
             << arg->GetName() << " not in dataset, ignored" << endl ;
      varSubset2.remove(*arg) ;
    }
  }

  RooAbsData* ret = nullptr;
  if (cut && strlen(cut)>0) {
    RooFormulaVar cutVar(cut, cut, *get(), false);
    ret = reduceEng(varSubset2,&cutVar,0,0,std::numeric_limits<std::size_t>::max());
  } else {
    ret = reduceEng(varSubset2,0,0,0,std::numeric_limits<std::size_t>::max());
  }
  ret->copyGlobalObservables(*this);
  return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a subset of the data set by applying the given cut on the data points
/// and reducing the dimensions to the specified set.
///
/// The 'cutVar' formula variable is used to select the subset of data points to be
/// retained in the reduced data collection.

RooAbsData* RooAbsData::reduce(const RooArgSet& varSubset, const RooFormulaVar& cutVar)
{
  // Make sure varSubset doesn't contain any variable not in this dataset
  RooArgSet varSubset2(varSubset) ;
  for(RooAbsArg * arg : varSubset) {
    if (!_vars.find(arg->GetName())) {
      coutW(InputArguments) << "RooAbsData::reduce(" << GetName() << ") WARNING: variable "
             << arg->GetName() << " not in dataset, ignored" << endl ;
      varSubset2.remove(*arg) ;
    }
  }

  RooAbsData* ret = reduceEng(varSubset2,&cutVar,0,0,std::numeric_limits<std::size_t>::max()) ;
  ret->copyGlobalObservables(*this);
  return ret;
}


RooPlot* RooAbsData::plotOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2,
             const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5,
             const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const
{
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return plotOn(frame,l) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Create and fill a ROOT histogram TH1,TH2 or TH3 with the values of this dataset for the variables with given names
/// The range of each observable that is histogrammed is always automatically calculated from the distribution in
/// the dataset. The number of bins can be controlled using the [xyz]bins parameters. For a greater degree of control
/// use the createHistogram() method below with named arguments
///
/// The caller takes ownership of the returned histogram
///
/// \deprecated This function is deprecated and will be removed in ROOT v6.30.
/// Use the overload
/// RooAbsData::createHistogram(const char*, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&)
/// that takes RooFit command arguments.

TH1 *RooAbsData::createHistogram(const char* varNameList, Int_t xbins, Int_t ybins, Int_t zbins) const
{
  coutW(DataHandling) << "'RooAbsData::createHistogram' is deprecated and will be removed in ROOT v6.30: "
                      << "Use the overload of 'RooAbsData::createHistogram' that takes RooFit command arguments."
                      << std::endl;

  // Parse list of variable names
  const auto varNames = ROOT::Split(varNameList, ",:");
  RooLinkedList argList;
  RooRealVar* vars[3] = {nullptr, nullptr, nullptr};

  for (unsigned int i = 0; i < varNames.size(); ++i) {
    if (i >= 3) {
      coutW(InputArguments) << "RooAbsData::createHistogram(" << GetName() << "): Can only create 3-dimensional histograms. Variable "
          << i << " " << varNames[i] << " unused." << std::endl;
      continue;
    }

    vars[i] = static_cast<RooRealVar*>( get()->find(varNames[i].data()) );
    if (!vars[i]) {
      coutE(InputArguments) << "RooAbsData::createHistogram(" << GetName() << ") ERROR: dataset does not contain an observable named " << varNames[i] << std::endl;
      return nullptr;
    }
  }

  if (!vars[0]) {
    coutE(InputArguments) << "RooAbsData::createHistogram(" << GetName() << "): No variable to be histogrammed in list '" << varNameList << "'" << std::endl;
    return nullptr;
  }

  if (xbins<=0  || !vars[0]->hasMax() || !vars[0]->hasMin() ) {
    argList.Add(RooFit::AutoBinning(xbins==0?vars[0]->numBins():abs(xbins)).Clone()) ;
  } else {
    argList.Add(RooFit::Binning(xbins).Clone()) ;
  }

  if (vars[1]) {
    if (ybins<=0 || !vars[1]->hasMax() || !vars[1]->hasMin() ) {
      argList.Add(RooFit::YVar(*vars[1],RooFit::AutoBinning(ybins==0?vars[1]->numBins():abs(ybins))).Clone()) ;
    } else {
      argList.Add(RooFit::YVar(*vars[1],RooFit::Binning(ybins)).Clone()) ;
    }
  }
  if (vars[2]) {
    if (zbins<=0 || !vars[2]->hasMax() || !vars[2]->hasMin() ) {
      argList.Add(RooFit::ZVar(*vars[2],RooFit::AutoBinning(zbins==0?vars[2]->numBins():abs(zbins))).Clone()) ;
    } else {
      argList.Add(RooFit::ZVar(*vars[2],RooFit::Binning(zbins)).Clone()) ;
    }
  }


  // Call implementation function
  TH1* result = createHistogram(GetName(), *vars[0], argList);

  // Delete temporary list of RooCmdArgs
  argList.Delete() ;

  return result ;
}


TH1 *RooAbsData::createHistogram(const char *name, const RooAbsRealLValue& xvar,
             const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4,
             const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const
{
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  return createHistogram(name,xvar,l) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Create and fill a ROOT histogram TH1,TH2 or TH3 with the values of this
/// dataset for the variables with given names.
///
/// \param[in] varNameList Comma-separated variable names.
/// \param[in] binArgX Control the binning for the `x` variable.
/// \param[in] binArgY Control the binning for the `y` variable.
/// \param[in] binArgZ Control the binning for the `z` variable.
/// \return Histogram now owned by user.
///
/// The possible binning command arguments for each axis are:
///
/// <table>
/// <tr><td> `AutoBinning(Int_t nbins, Double_y margin)`   <td> Automatically calculate range with given added fractional margin, set binning to nbins
/// <tr><td> `AutoSymBinning(Int_t nbins, Double_y margin)`   <td> Automatically calculate range with given added fractional margin,
///     with additional constraint that mean of data is in center of range, set binning to nbins
/// <tr><td> `Binning(const char* name)`   <td> Apply binning with given name to x axis of histogram
/// <tr><td> `Binning(RooAbsBinning& binning)`   <td> Apply specified binning to x axis of histogram
/// <tr><td> `Binning(int nbins, double lo, double hi)`   <td> Apply specified binning to x axis of histogram
///
/// <tr><td> `YVar(const RooAbsRealLValue& var,...)`   <td> Observable to be mapped on y axis of ROOT histogram
/// <tr><td> `ZVar(const RooAbsRealLValue& var,...)`   <td> Observable to be mapped on z axis of ROOT histogram
/// </table>

TH1 *RooAbsData::createHistogram(const char* varNameList,
                                 const RooCmdArg& binArgX,
                                 const RooCmdArg& binArgY,
                                 const RooCmdArg& binArgZ) const
{
  // Parse list of variable names
  const auto varNames = ROOT::Split(varNameList, ",:");
  RooRealVar* vars[3] = {nullptr, nullptr, nullptr};

  for (unsigned int i = 0; i < varNames.size(); ++i) {
    if (i >= 3) {
      coutW(InputArguments) << "RooAbsData::createHistogram(" << GetName() << "): Can only create 3-dimensional histograms. Variable "
          << i << " " << varNames[i] << " unused." << std::endl;
      continue;
    }

    vars[i] = static_cast<RooRealVar*>( get()->find(varNames[i].data()) );
    if (!vars[i]) {
      coutE(InputArguments) << "RooAbsData::createHistogram(" << GetName() << ") ERROR: dataset does not contain an observable named " << varNames[i] << std::endl;
      return nullptr;
    }
  }

  if (!vars[0]) {
    coutE(InputArguments) << "RooAbsData::createHistogram(" << GetName() << "): No variable to be histogrammed in list '" << varNameList << "'" << std::endl;
    return nullptr;
  }

  // Fill command argument list
  RooLinkedList argList;
  argList.Add(binArgX.Clone());
  if (vars[1]) {
    argList.Add(RooFit::YVar(*vars[1],binArgY).Clone());
  }
  if (vars[2]) {
    argList.Add(RooFit::ZVar(*vars[2],binArgZ).Clone());
  }

  // Call implementation function
  TH1* result = createHistogram(GetName(), *vars[0], argList);

  // Delete temporary list of RooCmdArgs
  argList.Delete() ;

  return result ;
}

////////////////////////////////////////////////////////////////////////////////
///
/// This function accepts the following arguments
///
/// \param[in] name Name of the ROOT histogram
/// \param[in] xvar Observable to be mapped on x axis of ROOT histogram
/// \param[in] argListIn list of input arguments
/// \return Histogram now owned by user.
///
/// <table>
/// <tr><td> `AutoBinning(Int_t nbins, Double_y margin)`   <td> Automatically calculate range with given added fractional margin, set binning to nbins
/// <tr><td> `AutoSymBinning(Int_t nbins, Double_y margin)`   <td> Automatically calculate range with given added fractional margin,
///     with additional constraint that mean of data is in center of range, set binning to nbins
/// <tr><td> `Binning(const char* name)`   <td> Apply binning with given name to x axis of histogram
/// <tr><td> `Binning(RooAbsBinning& binning)`   <td> Apply specified binning to x axis of histogram
/// <tr><td> `Binning(int nbins, double lo, double hi)`   <td> Apply specified binning to x axis of histogram
///
/// <tr><td> `YVar(const RooAbsRealLValue& var,...)`   <td> Observable to be mapped on y axis of ROOT histogram
/// <tr><td> `ZVar(const RooAbsRealLValue& var,...)`   <td> Observable to be mapped on z axis of ROOT histogram
/// </table>
///
/// The YVar() and ZVar() arguments can be supplied with optional Binning() Auto(Sym)Range() arguments to control the binning of the Y and Z axes, e.g.
/// ```
/// createHistogram("histo",x,Binning(-1,1,20), YVar(y,Binning(-1,1,30)), ZVar(z,Binning("zbinning")))
/// ```
///
/// The caller takes ownership of the returned histogram

TH1 *RooAbsData::createHistogram(const char *name, const RooAbsRealLValue& xvar, const RooLinkedList& argListIn) const
{
  RooLinkedList argList(argListIn) ;

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsData::createHistogram(%s)",GetName())) ;
  pc.defineString("cutRange","CutRange",0,"",true) ;
  pc.defineString("cutString","CutSpec",0,"") ;
  pc.defineObject("yvar","YVar",0,0) ;
  pc.defineObject("zvar","ZVar",0,0) ;
  pc.allowUndefined() ;

  // Process & check varargs
  pc.process(argList) ;
  if (!pc.ok(true)) {
    return nullptr;
  }

  const char* cutSpec = pc.getString("cutString",0,true) ;
  const char* cutRange = pc.getString("cutRange",0,true) ;

  RooArgList vars(xvar) ;
  RooAbsArg* yvar = static_cast<RooAbsArg*>(pc.getObject("yvar")) ;
  if (yvar) {
    vars.add(*yvar) ;
  }
  RooAbsArg* zvar = static_cast<RooAbsArg*>(pc.getObject("zvar")) ;
  if (zvar) {
    vars.add(*zvar) ;
  }

  pc.stripCmdList(argList,"CutRange,CutSpec") ;

  // Swap Auto(Sym)RangeData with a Binning command
  RooLinkedList ownedCmds ;
  RooCmdArg* autoRD = (RooCmdArg*) argList.find("AutoRangeData") ;
  if (autoRD) {
    double xmin,xmax ;
    if (!getRange((RooRealVar&)xvar,xmin,xmax,autoRD->getDouble(0),autoRD->getInt(0))) {
       RooCmdArg* bincmd = (RooCmdArg*) RooFit::Binning(autoRD->getInt(1),xmin,xmax).Clone() ;
       ownedCmds.Add(bincmd) ;
       argList.Replace(autoRD,bincmd) ;
    }
  }

  if (yvar) {
    RooCmdArg* autoRDY = (RooCmdArg*) ((RooCmdArg*)argList.find("YVar"))->subArgs().find("AutoRangeData") ;
    if (autoRDY) {
      double ymin,ymax ;
      if (!getRange((RooRealVar&)(*yvar),ymin,ymax,autoRDY->getDouble(0),autoRDY->getInt(0))) {
         RooCmdArg* bincmd = (RooCmdArg*) RooFit::Binning(autoRDY->getInt(1),ymin,ymax).Clone() ;
         //ownedCmds.Add(bincmd) ;
         ((RooCmdArg*)argList.find("YVar"))->subArgs().Replace(autoRDY,bincmd) ;
      }
      delete autoRDY ;
    }
  }

  if (zvar) {
    RooCmdArg* autoRDZ = (RooCmdArg*) ((RooCmdArg*)argList.find("ZVar"))->subArgs().find("AutoRangeData") ;
    if (autoRDZ) {
      double zmin,zmax ;
      if (!getRange((RooRealVar&)(*zvar),zmin,zmax,autoRDZ->getDouble(0),autoRDZ->getInt(0))) {
         RooCmdArg* bincmd = (RooCmdArg*) RooFit::Binning(autoRDZ->getInt(1),zmin,zmax).Clone() ;
         //ownedCmds.Add(bincmd) ;
         ((RooCmdArg*)argList.find("ZVar"))->subArgs().Replace(autoRDZ,bincmd) ;
      }
      delete autoRDZ ;
    }
  }


  TH1* histo = xvar.createHistogram(name,argList) ;
  fillHistogram(histo,vars,cutSpec,cutRange) ;

  ownedCmds.Delete() ;

  return histo ;
}

////////////////////////////////////////////////////////////////////////////////
/// Construct table for product of categories in catSet

Roo1DTable* RooAbsData::table(const RooArgSet& catSet, const char* cuts, const char* opts) const
{
  RooArgSet catSet2 ;

  string prodName("(") ;
  for(auto * arg : catSet) {
    if (dynamic_cast<RooAbsCategory*>(arg)) {
      RooAbsCategory* varsArg = dynamic_cast<RooAbsCategory*>(_vars.find(arg->GetName())) ;
      if (varsArg != 0) catSet2.add(*varsArg) ;
      else catSet2.add(*arg) ;
      if (prodName.length()>1) {
   prodName += " x " ;
      }
      prodName += arg->GetName() ;
    } else {
      coutW(InputArguments) << "RooAbsData::table(" << GetName() << ") non-RooAbsCategory input argument " << arg->GetName() << " ignored" << endl ;
    }
  }
  prodName += ")" ;

  RooMultiCategory tmp(prodName.c_str(),prodName.c_str(),catSet2) ;
  return table(tmp,cuts,opts) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Print name of dataset

void RooAbsData::printName(ostream& os) const
{
  os << GetName() ;
}

////////////////////////////////////////////////////////////////////////////////
/// Print title of dataset

void RooAbsData::printTitle(ostream& os) const
{
  os << GetTitle() ;
}

////////////////////////////////////////////////////////////////////////////////
/// Print class name of dataset

void RooAbsData::printClassName(ostream& os) const
{
  os << ClassName() ;
}

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::printMultiline(ostream& os, Int_t contents, bool verbose, TString indent) const
{
  _dstore->printMultiline(os,contents,verbose,indent) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Define default print options, for a given print style

Int_t RooAbsData::defaultPrintContents(Option_t* /*opt*/) const
{
  return kName|kClassName|kArgs|kValue ;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate standardized moment.
///
/// \param[in] var Variable to be used for calculating the moment.
/// \param[in] order Order of the moment.
/// \param[in] cutSpec  If specified, the moment is calculated on the subset of the data which pass the C++ cut specification expression 'cutSpec'
/// \param[in] cutRange If specified, calculate inside the range named 'cutRange' (also applies cut spec)
/// \return \f$ \frac{\left< \left( X - \left< X \right> \right)^n \right>}{\sigma^n} \f$,  where n = order.

double RooAbsData::standMoment(const RooRealVar &var, double order, const char* cutSpec, const char* cutRange) const
{
  // Hardwire invariant answer for first and second moment
  if (order==1) return 0 ;
  if (order==2) return 1 ;

  return moment(var,order,cutSpec,cutRange) / TMath::Power(sigma(var,cutSpec,cutRange),order) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Calculate moment of requested order.
///
/// \param[in] var Variable to be used for calculating the moment.
/// \param[in] order Order of the moment.
/// \param[in] cutSpec  If specified, the moment is calculated on the subset of the data which pass the C++ cut specification expression 'cutSpec'
/// \param[in] cutRange If specified, calculate inside the range named 'cutRange' (also applies cut spec)
/// \return \f$ \left< \left( X - \left< X \right> \right)^n \right> \f$ of order \f$n\f$.
///

double RooAbsData::moment(const RooRealVar& var, double order, const char* cutSpec, const char* cutRange) const
{
  double offset = order>1 ? moment(var,1,cutSpec,cutRange) : 0 ;
  return moment(var,order,offset,cutSpec,cutRange) ;

}

////////////////////////////////////////////////////////////////////////////////
/// Return the 'order'-ed moment of observable 'var' in this dataset. If offset is non-zero it is subtracted
/// from the values of 'var' prior to the moment calculation. If cutSpec and/or cutRange are specified
/// the moment is calculated on the subset of the data which pass the C++ cut specification expression 'cutSpec'
/// and/or are inside the range named 'cutRange'

double RooAbsData::moment(const RooRealVar& var, double order, double offset, const char* cutSpec, const char* cutRange) const
{
  // Lookup variable in dataset
  auto arg = _vars.find(var.GetName());
  if (!arg) {
    coutE(InputArguments) << "RooDataSet::moment(" << GetName() << ") ERROR: unknown variable: " << var.GetName() << std::endl;
    return 0;
  }

  auto varPtr = dynamic_cast<const RooRealVar*>(arg);
  // Check if found variable is of type RooRealVar
  if (!varPtr) {
    coutE(InputArguments) << "RooDataSet::moment(" << GetName() << ") ERROR: variable " << var.GetName() << " is not of type RooRealVar" << endl ;
    return 0;
  }

  // Check if dataset is not empty
  if(sumEntries(cutSpec, cutRange) == 0.) {
    coutE(InputArguments) << "RooDataSet::moment(" << GetName() << ") WARNING: empty dataset" << endl ;
    return 0;
  }

  // Setup RooFormulaVar for cutSpec if it is present
  std::unique_ptr<RooFormula> select;
  if (cutSpec) {
    select.reset(new RooFormula("select",cutSpec,*get()));
  }


  // Calculate requested moment
  ROOT::Math::KahanSum<double> sum;
  for(Int_t index= 0; index < numEntries(); index++) {
    const RooArgSet* vars = get(index) ;
    if (select && select->eval()==0) continue ;
    if (cutRange && vars->allInRange(cutRange)) continue ;

    sum += weight() * TMath::Power(varPtr->getVal() - offset,order);
  }

  return sum/sumEntries(cutSpec, cutRange);
}

////////////////////////////////////////////////////////////////////////////////
/// Internal method to check if given RooRealVar maps to a RooRealVar in this dataset

RooRealVar* RooAbsData::dataRealVar(const char* methodname, const RooRealVar& extVar) const
{
  // Lookup variable in dataset
  RooRealVar *xdata = (RooRealVar*) _vars.find(extVar.GetName());
  if(!xdata) {
    coutE(InputArguments) << "RooDataSet::" << methodname << "(" << GetName() << ") ERROR: variable : " << extVar.GetName() << " is not in data" << endl ;
    return nullptr;
  }
  // Check if found variable is of type RooRealVar
  if (!dynamic_cast<RooRealVar*>(xdata)) {
    coutE(InputArguments) << "RooDataSet::" << methodname << "(" << GetName() << ") ERROR: variable : " << extVar.GetName() << " is not of type RooRealVar in data" << endl ;
    return nullptr;
  }
  return xdata;
}

////////////////////////////////////////////////////////////////////////////////
/// Internal method to calculate single correlation and covariance elements

double RooAbsData::corrcov(const RooRealVar &x, const RooRealVar &y, const char* cutSpec, const char* cutRange, bool corr) const
{
  // Lookup variable in dataset
  RooRealVar *xdata = dataRealVar(corr?"correlation":"covariance",x) ;
  RooRealVar *ydata = dataRealVar(corr?"correlation":"covariance",y) ;
  if (!xdata||!ydata) return 0 ;

  // Check if dataset is not empty
  if(sumEntries(cutSpec, cutRange) == 0.) {
    coutW(InputArguments) << "RooDataSet::" << (corr?"correlation":"covariance") << "(" << GetName() << ") WARNING: empty dataset, returning zero" << endl ;
    return 0;
  }

  // Setup RooFormulaVar for cutSpec if it is present
  RooFormula* select = cutSpec ? new RooFormula("select",cutSpec,*get()) : 0 ;

  // Calculate requested moment
  double xysum(0),xsum(0),ysum(0),x2sum(0),y2sum(0);
  const RooArgSet* vars ;
  for(Int_t index= 0; index < numEntries(); index++) {
    vars = get(index) ;
    if (select && select->eval()==0) continue ;
    if (cutRange && vars->allInRange(cutRange)) continue ;

    xysum += weight()*xdata->getVal()*ydata->getVal() ;
    xsum += weight()*xdata->getVal() ;
    ysum += weight()*ydata->getVal() ;
    if (corr) {
      x2sum += weight()*xdata->getVal()*xdata->getVal() ;
      y2sum += weight()*ydata->getVal()*ydata->getVal() ;
    }
  }

  // Normalize entries
  xysum/=sumEntries(cutSpec, cutRange) ;
  xsum/=sumEntries(cutSpec, cutRange) ;
  ysum/=sumEntries(cutSpec, cutRange) ;
  if (corr) {
    x2sum/=sumEntries(cutSpec, cutRange) ;
    y2sum/=sumEntries(cutSpec, cutRange) ;
  }

  // Cleanup
  if (select) delete select ;

  // Return covariance or correlation as requested
  if (corr) {
    return (xysum-xsum*ysum)/(sqrt(x2sum-(xsum*xsum))*sqrt(y2sum-(ysum*ysum))) ;
  } else {
    return (xysum-xsum*ysum);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Return covariance matrix from data for given list of observables

TMatrixDSym* RooAbsData::corrcovMatrix(const RooArgList& vars, const char* cutSpec, const char* cutRange, bool corr) const
{
  RooArgList varList ;
  for(auto * var : static_range_cast<RooRealVar*>(vars)) {
    RooRealVar* datavar = dataRealVar("covarianceMatrix",*var) ;
    if (!datavar) {
      return nullptr;
    }
    varList.add(*datavar) ;
  }


  // Check if dataset is not empty
  if(sumEntries(cutSpec, cutRange) == 0.) {
    coutW(InputArguments) << "RooDataSet::covariance(" << GetName() << ") WARNING: empty dataset, returning zero" << endl ;
    return nullptr;
  }

  // Setup RooFormulaVar for cutSpec if it is present
  std::unique_ptr<RooFormula> select = cutSpec ? std::make_unique<RooFormula>("select",cutSpec,*get()) : nullptr;

  TMatrixDSym xysum(varList.size()) ;
  std::vector<double> xsum(varList.size()) ;
  std::vector<double> x2sum(varList.size()) ;

  // Calculate <x_i> and <x_i y_j>
  for(Int_t index= 0; index < numEntries(); index++) {
    const RooArgSet* dvars = get(index) ;
    if (select && select->eval()==0) continue ;
    if (cutRange && dvars->allInRange(cutRange)) continue ;

    for(std::size_t ix = 0; ix < varList.size(); ++ix) {
      auto varx = static_cast<RooRealVar const&>(varList[ix]);
      xsum[ix] += weight() * varx.getVal() ;
      if (corr) {
        x2sum[ix] += weight() * varx.getVal() * varx.getVal();
      }

      for(std::size_t iy = ix; iy < varList.size(); ++iy) {
        auto vary = static_cast<RooRealVar const&>(varList[iy]);
        xysum(ix,iy) += weight() * varx.getVal() * vary.getVal();
        xysum(iy,ix) = xysum(ix,iy) ;
      }
    }

  }

  // Normalize sums
  for (std::size_t ix=0 ; ix<varList.size() ; ix++) {
    xsum[ix] /= sumEntries(cutSpec, cutRange) ;
    if (corr) {
      x2sum[ix] /= sumEntries(cutSpec, cutRange) ;
    }
    for (std::size_t iy=0 ; iy<varList.size() ; iy++) {
      xysum(ix,iy) /= sumEntries(cutSpec, cutRange) ;
    }
  }

  // Calculate covariance matrix
  TMatrixDSym* C = new TMatrixDSym(varList.size()) ;
  for (std::size_t ix=0 ; ix<varList.size() ; ix++) {
    for (std::size_t iy=0 ; iy<varList.size() ; iy++) {
      (*C)(ix,iy) = xysum(ix,iy)-xsum[ix]*xsum[iy] ;
      if (corr) {
   (*C)(ix,iy) /= std::sqrt((x2sum[ix]-(xsum[ix]*xsum[ix]))*(x2sum[iy]-(xsum[iy]*xsum[iy]))) ;
      }
    }
  }

  return C ;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a RooRealVar containing the mean of observable 'var' in
/// this dataset.  If cutSpec and/or cutRange are specified the
/// moment is calculated on the subset of the data which pass the C++
/// cut specification expression 'cutSpec' and/or are inside the
/// range named 'cutRange'

RooRealVar* RooAbsData::meanVar(const RooRealVar &var, const char* cutSpec, const char* cutRange) const
{
  // Create a new variable with appropriate strings. The error is calculated as
  // RMS/Sqrt(N) which is generally valid.

  // Create holder variable for mean
  std::string name = std::string{var.GetName()} + "Mean";
  std::string title = std::string{"Mean of "} + var.GetTitle();
  auto *meanv= new RooRealVar(name.c_str(), title.c_str(), 0) ;
  meanv->setConstant(false) ;

  // Adjust plot label
  std::string label = "<" + std::string{var.getPlotLabel()} + ">";
  meanv->setPlotLabel(label.c_str());

  // fill in this variable's value and error
  double meanVal=moment(var,1,0,cutSpec,cutRange) ;
  double N(sumEntries(cutSpec,cutRange)) ;

  double rmsVal= sqrt(moment(var,2,meanVal,cutSpec,cutRange)*N/(N-1));
  meanv->setVal(meanVal) ;
  meanv->setError(N > 0 ? rmsVal/sqrt(N) : 0);

  return meanv;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a RooRealVar containing the RMS of observable 'var' in
/// this dataset.  If cutSpec and/or cutRange are specified the
/// moment is calculated on the subset of the data which pass the C++
/// cut specification expression 'cutSpec' and/or are inside the
/// range named 'cutRange'

RooRealVar* RooAbsData::rmsVar(const RooRealVar &var, const char* cutSpec, const char* cutRange) const
{
  // Create a new variable with appropriate strings. The error is calculated as
  // RMS/(2*Sqrt(N)) which is only valid if the variable has a Gaussian distribution.

  // Create RMS value holder
  std::string name(var.GetName()),title("RMS of ") ;
  name += "RMS";
  title += var.GetTitle();
  auto *rms= new RooRealVar(name.c_str(), title.c_str(), 0) ;
  rms->setConstant(false) ;

  // Adjust plot label
  std::string label(var.getPlotLabel());
  label += "_{RMS}";
  rms->setPlotLabel(label.c_str());

  // Fill in this variable's value and error
  double meanVal(moment(var,1,0,cutSpec,cutRange)) ;
  double N(sumEntries(cutSpec, cutRange));
  double rmsVal= sqrt(moment(var,2,meanVal,cutSpec,cutRange)*N/(N-1));
  rms->setVal(rmsVal) ;
  rms->setError(rmsVal/sqrt(2*N));

  return rms;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a box with statistics information to the specified frame. By default a box with the
/// event count, mean and rms of the plotted variable is added.
///
/// The following optional named arguments are accepted
/// <table>
/// <tr><td> `What(const char* whatstr)`   <td> Controls what is printed: "N" = count, "M" is mean, "R" is RMS.
/// <tr><td> `Format(const char* optStr)`   <td> \deprecated Classing parameter formatting options, provided for backward compatibility
///
/// <tr><td> `Format(const char* what,...)`   <td> Parameter formatting options.
///   <table>
///   <tr><td> const char* what <td> Controls what is shown:
///     - "N" adds name
///     - "E" adds error
///     - "A" shows asymmetric error
///     - "U" shows unit
///     - "H" hides the value
///   <tr><td> `FixedPrecision(int n)`   <td> Controls precision, set fixed number of digits
///   <tr><td> `AutoPrecision(int n)`   <td> Controls precision. Number of shown digits is calculated from error + n specified additional digits (1 is sensible default)
///   <tr><td> `VerbatimName(bool flag)`   <td> Put variable name in a \\verb+   + clause.
///   </table>
/// <tr><td> `Label(const chat* label)`   <td> Add header label to parameter box
/// <tr><td> `Layout(double xmin, double xmax, double ymax)`   <td> Specify relative position of left,right side of box and top of box. Position of
///     bottom of box is calculated automatically from number lines in box
/// <tr><td> `Cut(const char* expression)`   <td> Apply given cut expression to data when calculating statistics
/// <tr><td> `CutRange(const char* rangeName)`   <td> Only consider events within given range when calculating statistics. Multiple
///     CutRange() argument may be specified to combine ranges.
///
/// </table>

RooPlot* RooAbsData::statOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2,
             const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5,
             const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{
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
  pc.defineString("cutRange","CutRange",0,"",true) ;
  pc.defineString("cutString","CutSpec",0,"") ;
  pc.defineMutex("Format","FormatArgs") ;

  // Process and check varargs
  pc.process(cmdList) ;
  if (!pc.ok(true)) {
    return frame ;
  }

  const char* label = pc.getString("label") ;
  double xmin = pc.getDouble("xmin") ;
  double xmax = pc.getDouble("xmax") ;
  double ymax = pc.getInt("ymaxi") / 10000. ;
  const char* formatStr = pc.getString("formatStr") ;
  Int_t sigDigit = pc.getInt("sigDigit") ;
  const char* what = pc.getString("what") ;

  const char* cutSpec = pc.getString("cutString",0,true) ;
  const char* cutRange = pc.getString("cutRange",0,true) ;

  if (pc.hasProcessed("FormatArgs")) {
    RooCmdArg* formatCmd = static_cast<RooCmdArg*>(cmdList.FindObject("FormatArgs")) ;
    return statOn(frame,what,label,0,0,xmin,xmax,ymax,cutSpec,cutRange,formatCmd) ;
  } else {
    return statOn(frame,what,label,sigDigit,formatStr,xmin,xmax,ymax,cutSpec,cutRange) ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Implementation back-end of statOn() method with named arguments

RooPlot* RooAbsData::statOn(RooPlot* frame, const char* what, const char *label, Int_t sigDigits,
              Option_t *options, double xmin, double xmax, double ymax,
              const char* cutSpec, const char* cutRange, const RooCmdArg* formatCmd)
{
  bool showLabel= (label != nullptr && strlen(label) > 0);

  std::string whatStr{what};
  std::transform(whatStr.begin(), whatStr.end(), whatStr.begin(), [](unsigned char c){ return std::toupper(c); });
  bool showN = whatStr.find('N') != std::string::npos;
  bool showR = whatStr.find('R') != std::string::npos;
  bool showM = whatStr.find('M') != std::string::npos;
  Int_t nPar= 0;
  if (showN) nPar++ ;
  if (showR) nPar++ ;
  if (showM) nPar++ ;

  // calculate the box's size
  double dy(0.06), ymin(ymax-nPar*dy);
  if(showLabel) ymin-= dy;

  // create the box and set its options
  TPaveText *box= new TPaveText(xmin,ymax,xmax,ymin,"BRNDC");
  if(!box) return nullptr;
  box->SetName(Form("%s_statBox",GetName())) ;
  box->SetFillColor(0);
  box->SetBorderSize(1);
  box->SetTextAlign(12);
  box->SetTextSize(0.04F);
  box->SetFillStyle(1001);

  // add formatted text for each statistic
  RooRealVar N("N","Number of Events",sumEntries(cutSpec,cutRange));
  N.setPlotLabel("Entries") ;
  std::unique_ptr<RooRealVar> meanv{meanVar(*(RooRealVar*)frame->getPlotVar(),cutSpec,cutRange)};
  meanv->setPlotLabel("Mean") ;
  std::unique_ptr<RooRealVar> rms{rmsVar(*(RooRealVar*)frame->getPlotVar(),cutSpec,cutRange)};
  rms->setPlotLabel("RMS") ;
  std::unique_ptr<TString> rmsText, meanText, NText;
  if (options) {
    rmsText.reset(rms->format(sigDigits,options));
    meanText.reset(meanv->format(sigDigits,options));
    NText.reset(N.format(sigDigits,options));
  } else {
    rmsText.reset(rms->format(*formatCmd));
    meanText.reset(meanv->format(*formatCmd));
    NText.reset(N.format(*formatCmd));
  }
  if (showR) box->AddText(rmsText->Data());
  if (showM) box->AddText(meanText->Data());
  if (showN) box->AddText(NText->Data());

  // add the optional label if specified
  if(showLabel) box->AddText(label);

  frame->addObject(box) ;
  return frame ;
}

////////////////////////////////////////////////////////////////////////////////
/// Loop over columns of our tree data and fill the input histogram. Returns a pointer to the
/// input histogram, or zero in case of an error. The input histogram can be any TH1 subclass, and
/// therefore of arbitrary dimension. Variables are matched with the (x,y,...) dimensions of the input
/// histogram according to the order in which they appear in the input plotVars list.

TH1 *RooAbsData::fillHistogram(TH1 *hist, const RooArgList &plotVars, const char *cuts, const char* cutRange) const
{
  // Do we have a valid histogram to use?
  if(nullptr == hist) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: no valid histogram to fill" << endl;
    return nullptr;
  }

  // Check that the number of plotVars matches the input histogram's dimension
  std::size_t hdim= hist->GetDimension();
  if(hdim != plotVars.size()) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: plotVars has the wrong dimension" << endl;
    return nullptr;
  }

  // Check that the plot variables are all actually RooAbsReal's and print a warning if we do not
  // explicitly depend on one of them. Clone any variables that we do not contain directly and
  // redirect them to use our event data.
  RooArgSet plotClones,localVars;
  for(std::size_t index= 0; index < plotVars.size(); index++) {
    const RooAbsArg *var= plotVars.at(index);
    const RooAbsReal *realVar= dynamic_cast<const RooAbsReal*>(var);
    if(0 == realVar) {
      coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: cannot plot variable \"" << var->GetName()
      << "\" of type " << var->ClassName() << endl;
      return nullptr;
    }
    RooAbsArg *found= _vars.find(realVar->GetName());
    if(!found) {
      RooAbsArg *clone= plotClones.addClone(*realVar,true); // do not complain about duplicates
      assert(0 != clone);
      if(!clone->dependsOn(_vars)) {
        coutE(InputArguments) << ClassName() << "::" << GetName()
            << ":fillHistogram: Data does not contain the variable '" << realVar->GetName() << "'." << endl;
        return nullptr;
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
  std::unique_ptr<RooFormula> select;
  if (cuts != nullptr && strlen(cuts) > 0) {
    select = std::make_unique<RooFormula>(cuts, cuts, _vars, false);
    if (!select || !select->ok()) {
      coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: invalid cuts \"" << cuts << "\"" << endl;
      return nullptr;
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
  const auto cutVec = ROOT::Split(cutRange ? cutRange : "", ",");

  // Loop over events and fill the histogram
  if (hist->GetSumw2()->fN==0) {
    hist->Sumw2() ;
  }
  Int_t nevent= numEntries() ; //(Int_t)_tree->GetEntries();
  for(Int_t i=0; i < nevent; ++i) {

    //Int_t entryNumber= _tree->GetEntryNumber(i);
    //if (entryNumber<0) break;
    get(i);

    // Apply expression based selection criteria
    if (select && select->eval()==0) {
      continue ;
    }


    // Apply range based selection criteria
    bool selectByRange = true ;
    if (cutRange) {
      for (const auto arg : _vars) {
        bool selectThisArg = false ;
        for (auto const& cut : cutVec) {
          if (!cut.empty() && arg->inRange(cut.c_str())) {
            selectThisArg = true ;
            break ;
          }
        }
        if (!selectThisArg) {
          selectByRange = false ;
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


    double error2 = TMath::Power(hist->GetBinError(bin),2)-TMath::Power(weight(),2)  ;
    double we = weightError(RooAbsData::SumW2) ;
    if (we==0) we = weight() ;
    error2 += TMath::Power(we,2) ;


//     double we = weightError(RooAbsData::SumW2) ;
//     double error2(0) ;
//     if (we==0) {
//       we = weight() ; //sqrt(weight()) ;
//       error2 = TMath::Power(hist->GetBinError(bin),2)-TMath::Power(weight(),2) + TMath::Power(we,2) ;
//     } else {
//       error2 = TMath::Power(hist->GetBinError(bin),2)-TMath::Power(weight(),2) + TMath::Power(we,2) ;
//     }
    //hist->AddBinContent(bin,weight());
    hist->SetBinError(bin,sqrt(error2)) ;

    //cout << "RooTreeData::fillHistogram() bin = " << bin << " weight() = " << weight() << " we = " << we << endl ;

  }

  return hist;
}

////////////////////////////////////////////////////////////////////////////////
/// Split dataset into subsets based on states of given splitCat in this dataset.
/// A TList of RooDataSets is returned in which each RooDataSet is named
/// after the state name of splitCat of which it contains the dataset subset.
/// The observables splitCat itself is no longer present in the sub datasets.
/// If createEmptyDataSets is false (default) this method only creates datasets for states
/// which have at least one entry The caller takes ownership of the returned list and its contents

TList* RooAbsData::split(const RooAbsCategory& splitCat, bool createEmptyDataSets) const
{
  // Sanity check
  if (!splitCat.dependsOn(*get())) {
    coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName()
    << " doesn't depend on any variable in this dataset" << endl ;
    return nullptr;
  }

  // Clone splitting category and attach to self
  RooAbsCategory* cloneCat =0;
  std::unique_ptr<RooArgSet> cloneSet;
  if (splitCat.isDerived()) {
    cloneSet.reset(static_cast<RooArgSet*>(RooArgSet(splitCat).snapshot(true)));
    if (!cloneSet) {
      coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") Couldn't deep-clone splitting category, abort." << endl ;
      return nullptr;
    }
    cloneCat = (RooAbsCategory*) cloneSet->find(splitCat.GetName()) ;
    cloneCat->attachDataSet(*this) ;
  } else {
    cloneCat = dynamic_cast<RooAbsCategory*>(get()->find(splitCat.GetName())) ;
    if (!cloneCat) {
      coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName()
      << " is fundamental and does not appear in this dataset" << endl ;
      return nullptr;
    }
  }

  // Split a dataset in a series of subsets, each corresponding
  // to a state of splitCat
  TList* dsetList = new TList ;

  // Construct set of variables to be included in split sets = full set - split category
  RooArgSet subsetVars(*get()) ;
  if (splitCat.isDerived()) {
    std::unique_ptr<RooArgSet> vars{splitCat.getVariables()};
    subsetVars.remove(*vars,true,true) ;
  } else {
    subsetVars.remove(splitCat,true,true) ;
  }

  // Add weight variable explicitly if dataset has weights, but no top-level weight
  // variable exists (can happen with composite datastores)
  bool addWV(false) ;
  RooRealVar newweight("weight","weight",-1e9,1e9) ;
  if (isWeighted() && !IsA()->InheritsFrom(RooDataHist::Class())) {
    subsetVars.add(newweight) ;
    addWV = true ;
  }

  // If createEmptyDataSets is true, prepopulate with empty sets corresponding to all states
  if (createEmptyDataSets) {
    for (const auto& nameIdx : *cloneCat) {
      RooAbsData* subset = emptyClone(nameIdx.first.c_str(), nameIdx.first.c_str(), &subsetVars,(addWV?"weight":0)) ;
      dsetList->Add((RooAbsArg*)subset) ;
    }
  }


  // Loop over dataset and copy event to matching subset
  const bool propWeightSquared = isWeighted();
  for (Int_t i = 0; i < numEntries(); ++i) {
    const RooArgSet* row =  get(i);
    RooAbsData* subset = (RooAbsData*) dsetList->FindObject(cloneCat->getCurrentLabel());
    if (!subset) {
      subset = emptyClone(cloneCat->getCurrentLabel(),cloneCat->getCurrentLabel(),&subsetVars,(addWV?"weight":0));
      dsetList->Add((RooAbsArg*)subset);
    }
    if (!propWeightSquared) {
   subset->add(*row, weight());
    } else {
   subset->add(*row, weight(), weightSquared());
    }
  }

  return dsetList;
}

////////////////////////////////////////////////////////////////////////////////
/// Split dataset into subsets based on the categorisation of the RooSimultaneous
/// A TList of RooDataSets is returned in which each RooDataSet is named
/// after the state name of splitCat of which it contains the dataset subset.
/// The observables splitCat itself is no longer present in the sub datasets, as well as the
/// observables of the other categories.
/// If createEmptyDataSets is false (default) this method only creates datasets for states
/// which have at least one entry The caller takes ownership of the returned list and its contents

TList* RooAbsData::split(const RooSimultaneous& simpdf, bool createEmptyDataSets) const
{
  auto& splitCat = const_cast<RooAbsCategoryLValue&>(simpdf.indexCat());

  // Sanity check
  if (!splitCat.dependsOn(*get())) {
    coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName()
    << " doesn't depend on any variable in this dataset" << endl ;
    return nullptr;
  }

  // Clone splitting category and attach to self
  RooAbsCategory* cloneCat =0;
  std::unique_ptr<RooArgSet> cloneSet;
  if (splitCat.isDerived()) {
    cloneSet.reset(static_cast<RooArgSet*>(RooArgSet(splitCat).snapshot(true)));
    if (!cloneSet) {
      coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") Couldn't deep-clone splitting category, abort." << endl ;
      return nullptr;
    }
    cloneCat = (RooAbsCategory*) cloneSet->find(splitCat.GetName()) ;
    cloneCat->attachDataSet(*this) ;
  } else {
    cloneCat = dynamic_cast<RooAbsCategory*>(get()->find(splitCat.GetName())) ;
    if (!cloneCat) {
      coutE(InputArguments) << "RooTreeData::split(" << GetName() << ") ERROR category " << splitCat.GetName()
      << " is fundamental and does not appear in this dataset" << endl ;
      return nullptr;
    }
  }

  // Split a dataset in a series of subsets, each corresponding
  // to a state of splitCat
  TList* dsetList = new TList ;

  // Construct set of variables to be included in split sets = full set - split category
  RooArgSet subsetVars(*get()) ;
  if (splitCat.isDerived()) {
    std::unique_ptr<RooArgSet> vars{splitCat.getVariables()};
    subsetVars.remove(*vars,true,true) ;
  } else {
    subsetVars.remove(splitCat,true,true) ;
  }

  // Add weight variable explicitly if dataset has weights, but no top-level weight
  // variable exists (can happen with composite datastores)
  bool addWV(false) ;
  RooRealVar newweight("weight","weight",-1e9,1e9) ;
  if (isWeighted() && !IsA()->InheritsFrom(RooDataHist::Class())) {
    subsetVars.add(newweight) ;
    addWV = true ;
  }

  // Get the observables for a given pdf in the RooSimultaneous, or an empty
  // RooArgSet if no pdf is set
  auto getPdfObservables = [this, &simpdf](const char * label) {
    RooArgSet obsSet;
    if(RooAbsPdf* catPdf = simpdf.getPdf(label)) {
      catPdf->getObservables(this->get(), obsSet);
    }
    return obsSet;
  };

  // By default, remove all category observables from the subdatasets
  RooArgSet allObservables;
  for( const auto& catPair : splitCat) {
    allObservables.add(getPdfObservables(catPair.first.c_str()));
  }
  subsetVars.remove(allObservables, true, true);


  // If createEmptyDataSets is true, prepopulate with empty sets corresponding to all states
  if (createEmptyDataSets) {
    for (const auto& nameIdx : *cloneCat) {
      // Add in the subset only the observables corresponding to this category
      RooArgSet subsetVarsCat(subsetVars);
      subsetVarsCat.add(getPdfObservables(nameIdx.first.c_str()));
      RooAbsData* subset = emptyClone(nameIdx.first.c_str(), nameIdx.first.c_str(), &subsetVarsCat,(addWV?"weight":0)) ;
      dsetList->Add(subset) ;
    }
  }


  // Loop over dataset and copy event to matching subset
  const bool propWeightSquared = isWeighted();
  for (Int_t i = 0; i < numEntries(); ++i) {
    const RooArgSet* row =  get(i);
    RooAbsData* subset = (RooAbsData*) dsetList->FindObject(cloneCat->getCurrentLabel());
    if (!subset) {
      // Add in the subset only the observables corresponding to this category
      RooArgSet subsetVarsCat(subsetVars);
      subsetVarsCat.add(getPdfObservables(cloneCat->getCurrentLabel()));
      subset = emptyClone(cloneCat->getCurrentLabel(),cloneCat->getCurrentLabel(),&subsetVarsCat,(addWV?"weight":0));
      dsetList->Add(subset);
    }
    if (!propWeightSquared) {
      subset->add(*row, weight());
    } else {
      subset->add(*row, weight(), weightSquared());
    }
  }

  return dsetList;
}

////////////////////////////////////////////////////////////////////////////////
/// Plot dataset on specified frame.
///
/// By default:
/// - An unbinned dataset will use the default binning of the target frame.
/// - A binned dataset will retain its intrinsic binning.
///
/// The following optional named arguments can be used to modify the behaviour:
/// \note Please follow the function links in the left column to learn about PyROOT specifics for a given option.
///
/// <table>
///
/// <tr><th> <th> Data representation options
/// <tr><td> RooFit::Asymmetry(const RooCategory& c)
///     <td> Show the asymmetry of the data in given two-state category [F(+)-F(-)] / [F(+)+F(-)].
///     Category must have two states with indices -1 and +1 or three states with indices -1,0 and +1.
/// <tr><td> RooFit::Efficiency(const RooCategory& c)
///     <td> Show the efficiency F(acc)/[F(acc)+F(rej)]. Category must have two states with indices 0 and 1
/// <tr><td> RooFit::DataError(Int_t)
///     <td> Select the type of error drawn:
///    - `Auto(default)` results in Poisson for unweighted data and SumW2 for weighted data
///    - `Poisson` draws asymmetric Poisson confidence intervals.
///    - `SumW2` draws symmetric sum-of-weights error ( \f$ \left( \sum w \right)^2 / \sum\left(w^2\right) \f$ )
///    - `None` draws no error bars
/// <tr><td> RooFit::Binning(int nbins, double xlo, double xhi)
///     <td> Use specified binning to draw dataset
/// <tr><td> RooFit::Binning(const RooAbsBinning&)
///     <td>  Use specified binning to draw dataset
/// <tr><td> RooFit::Binning(const char* name)
///     <td>  Use binning with specified name to draw dataset
/// <tr><td> RooFit::RefreshNorm()
///     <td> Force refreshing for PDF normalization information in frame.
///     If set, any subsequent PDF will normalize to this dataset, even if it is
///     not the first one added to the frame. By default only the 1st dataset
///     added to a frame will update the normalization information
/// <tr><td> RooFit::Rescale(double f)
///     <td> Rescale drawn histogram by given factor.
/// <tr><td> RooFit::Cut(const char*)
///     <td> Only plot entries that pass the given cut.
///          Apart from cutting in continuous variables `Cut("x>5")`, this can also be used to plot a specific
///          category state. Use something like `Cut("myCategory == myCategory::stateA")`, where
///          `myCategory` resolves to the state number for a given entry and
///          `myCategory::stateA` resolves to the state number of the state named "stateA".
///
/// <tr><td> RooFit::CutRange(const char*)
///     <td> Only plot data from given range. Separate multiple ranges with ",".
///          \note This often requires passing the normalisation when plotting the PDF because RooFit does not save
///          how many events were being plotted (it will only work for cutting slices out of uniformly distributed
///          variables).
/// ```
/// data->plotOn(frame01, CutRange("SB1"));
/// const double nData = data->sumEntries("", "SB1");
/// // Make clear that the target normalisation is nData. The enumerator NumEvent
/// // is needed to switch between relative and absolute scaling.
/// model.plotOn(frame01, Normalization(nData, RooAbsReal::NumEvent),
///   ProjectionRange("SB1"));
/// ```
///
/// <tr><th> <th> Histogram drawing options
/// <tr><td> RooFit::DrawOption(const char* opt)
///     <td> Select ROOT draw option for resulting TGraph object
/// <tr><td> RooFit::LineStyle(Style_t style)
///     <td> Select line style by ROOT line style code, default is solid
/// <tr><td> RooFit::LineColor(Color_t color)
///     <td> Select line color by ROOT color code, default is black
/// <tr><td> RooFit::LineWidth(Width_t width)
///     <td> Select line with in pixels, default is 3
/// <tr><td> RooFit::MarkerStyle(Style_t style)
///     <td> Select the ROOT marker style, default is 21
/// <tr><td> RooFit::MarkerColor(Color_t color)
///     <td> Select the ROOT marker color, default is black
/// <tr><td> RooFit::MarkerSize(Size_t size)
///     <td> Select the ROOT marker size
/// <tr><td> RooFit::FillStyle(Style_t style)
///     <td> Select fill style, default is filled.
/// <tr><td> RooFit::FillColor(Color_t color)
///     <td> Select fill color by ROOT color code
/// <tr><td> RooFit::XErrorSize(double frac)
///     <td> Select size of X error bar as fraction of the bin width, default is 1
///
/// <tr><th> <th> Misc. other options
/// <tr><td> RooFit::Name(const char* name)
///     <td> Give curve specified name in frame. Useful if curve is to be referenced later
/// <tr><td> RooFit::Invisible()
///     <td> Add curve to frame, but do not display. Useful in combination AddTo()
/// <tr><td> RooFit::AddTo(const char* name, double wgtSel, double wgtOther)
///     <td> Add constructed histogram to already existing histogram with given name and relative weight factors
///
/// </table>

RooPlot* RooAbsData::plotOn(RooPlot* frame, const RooLinkedList& argList) const
{
  // New experimental plotOn() with varargs...

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsData::plotOn(%s)",GetName())) ;
  pc.defineString("drawOption","DrawOption",0,"P") ;
  pc.defineString("cutRange","CutRange",0,"",true) ;
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
  pc.defineInt("errorType","DataError",0,(Int_t)RooAbsData::Auto) ;
  pc.defineInt("histInvisible","Invisible",0,0) ;
  pc.defineInt("refreshFrameNorm","RefreshNorm",0,1) ;
  pc.defineString("addToHistName","AddTo",0,"") ;
  pc.defineDouble("addToWgtSelf","AddTo",0,1.) ;
  pc.defineDouble("addToWgtOther","AddTo",1,1.) ;
  pc.defineDouble("xErrorSize","XErrorSize",0,1.) ;
  pc.defineDouble("scaleFactor","Rescale",0,1.) ;
  pc.defineMutex("DataError","Asymmetry","Efficiency") ;
  pc.defineMutex("Binning","BinningName","BinningSpec") ;

  // Process & check varargs
  pc.process(argList) ;
  if (!pc.ok(true)) {
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
    double xlo = pc.getDouble("xlo") ;
    double xhi = pc.getDouble("xhi") ;
    o.bins = new RooUniformBinning((xlo==xhi)?frame->getPlotVar()->getMin():xlo,
               (xlo==xhi)?frame->getPlotVar()->getMax():xhi,pc.getInt("nbins")) ;
  }
  const RooAbsCategoryLValue* asymCat = (const RooAbsCategoryLValue*) pc.getObject("asymCat") ;
  const RooAbsCategoryLValue* effCat = (const RooAbsCategoryLValue*) pc.getObject("effCat") ;
  o.etype = (RooAbsData::ErrorType) pc.getInt("errorType") ;
  o.histInvisible = pc.getInt("histInvisible") ;
  o.xErrorSize = pc.getDouble("xErrorSize") ;
  o.cutRange = pc.getString("cutRange",0,true) ;
  o.histName = pc.getString("histName",0,true) ;
  o.addToHistName = pc.getString("addToHistName",0,true) ;
  o.addToWgtSelf = pc.getDouble("addToWgtSelf") ;
  o.addToWgtOther = pc.getDouble("addToWgtOther") ;
  o.refreshFrameNorm = pc.getInt("refreshFrameNorm") ;
  o.scaleFactor = pc.getDouble("scaleFactor") ;

  // Map auto error type to actual type
  if (o.etype == Auto) {
    o.etype = isNonPoissonWeighted() ? SumW2 : Poisson ;
    if (o.etype == SumW2) {
      coutI(InputArguments) << "RooAbsData::plotOn(" << GetName()
             << ") INFO: dataset has non-integer weights, auto-selecting SumW2 errors instead of Poisson errors" << endl ;
    }
  }

  if (o.addToHistName && !frame->findObject(o.addToHistName,RooHist::Class())) {
    coutE(InputArguments) << "RooAbsData::plotOn(" << GetName() << ") cannot find existing histogram " << o.addToHistName
           << " to add to in RooPlot" << endl ;
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

////////////////////////////////////////////////////////////////////////////////
/// Create and fill a histogram of the frame's variable and append it to the frame.
/// The frame variable must be one of the data sets dimensions.
///
/// The plot range and the number of plot bins is determined by the parameters
/// of the plot variable of the frame (RooAbsReal::setPlotRange(), RooAbsReal::setPlotBins()).
///
/// The optional cut string expression can be used to select the events to be plotted.
/// The cut specification may refer to any variable contained in the data set.
///
/// The drawOptions are passed to the TH1::Draw() method.
/// \see RooAbsData::plotOn(RooPlot*,const RooLinkedList&) const
RooPlot *RooAbsData::plotOn(RooPlot *frame, PlotOpt o) const
{
  if(0 == frame) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return nullptr;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
    << ":plotOn: frame does not specify a plot variable" << endl;
    return nullptr;
  }

  // create and fill a temporary histogram of this variable
  const std::string histName = std::string{GetName()} + "_plot";
  std::unique_ptr<TH1> hist;
  if (o.bins) {
    hist.reset( var->createHistogram(histName.c_str(), RooFit::AxisLabel("Events"), RooFit::Binning(*o.bins)) );
  } else if (!frame->getPlotVar()->getBinning().isUniform()) {
    hist.reset( var->createHistogram(histName.c_str(), RooFit::AxisLabel("Events"),
        RooFit::Binning(frame->getPlotVar()->getBinning())) );
  } else {
    hist.reset( var->createHistogram(histName.c_str(), "Events",
        frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(), frame->GetNbinsX()) );
  }

  // Keep track of sum-of-weights error
  hist->Sumw2() ;

  if(0 == fillHistogram(hist.get(), RooArgList(*var),o.cuts,o.cutRange)) {
    coutE(Plotting) << ClassName() << "::" << GetName()
    << ":plotOn: fillHistogram() failed" << endl;
    return nullptr;
  }

  // If frame has no predefined bin width (event density) it will be adjusted to
  // our histograms bin width so we should force that bin width here
  double nomBinWidth ;
  if (frame->getFitRangeNEvt()==0 && o.bins) {
    nomBinWidth = o.bins->averageBinWidth() ;
  } else {
    nomBinWidth = o.bins ? frame->getFitRangeBinW() : 0 ;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist,nomBinWidth,1,o.etype,o.xErrorSize,o.correctForBinWidth,o.scaleFactor);
  if(0 == graph) {
    coutE(Plotting) << ClassName() << "::" << GetName()
    << ":plotOn: unable to create a RooHist object" << endl;
    return nullptr;
  }

  // If the dataset variable has a wide range than the plot variable,
  // calculate the number of entries in the dataset in the plot variable fit range
  RooAbsRealLValue* dataVar = (RooAbsRealLValue*) _vars.find(var->GetName()) ;
  double nEnt(sumEntries()) ;
  if (dataVar->getMin()<var->getMin() || dataVar->getMax()>var->getMax()) {
    std::unique_ptr<RooAbsData> tmp{const_cast<RooAbsData*>(this)->reduce(*var)};
    nEnt = tmp->sumEntries() ;
  }

  // Store the number of entries before the cut, if any was made
  if ((o.cuts && strlen(o.cuts)) || o.cutRange) {
    coutI(Plotting) << "RooTreeData::plotOn: plotting " << hist->GetSumOfWeights() << " events out of " << nEnt << " total events" << endl ;
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
    std::string hname = std::string{"h_"} + GetName();
    if (o.cutRange && strlen(o.cutRange)>0) {
      hname += std::string{"_CutRange["} + o.cutRange + "]";
    }
    if (o.cuts && strlen(o.cuts)>0) {
      hname += std::string{"_Cut["} + o.cuts + "]";
    }
    graph->SetName(hname.c_str()) ;
  }

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);


  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);

  return frame;
}

////////////////////////////////////////////////////////////////////////////////
/// Create and fill a histogram with the asymmetry N[+] - N[-] / ( N[+] + N[-] ),
/// where N(+/-) is the number of data points with asymCat=+1 and asymCat=-1
/// as function of the frames variable. The asymmetry category 'asymCat' must
/// have exactly 2 (or 3) states defined with index values +1,-1 (and 0)
///
/// The plot range and the number of plot bins is determined by the parameters
/// of the plot variable of the frame (RooAbsReal::setPlotRange(), RooAbsReal::setPlotBins())
///
/// The optional cut string expression can be used to select the events to be plotted.
/// The cut specification may refer to any variable contained in the data set
///
/// The drawOptions are passed to the TH1::Draw() method

RooPlot* RooAbsData::plotAsymOn(RooPlot* frame, const RooAbsCategoryLValue& asymCat, PlotOpt o) const
{
  if(0 == frame) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotAsymOn: frame is null" << endl;
    return nullptr;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
    << ":plotAsymOn: frame does not specify a plot variable" << endl;
    return nullptr;
  }

  // create and fill temporary histograms of this variable for each state
  std::string hist1Name(GetName()),hist2Name(GetName());
  hist1Name += "_plot1";
  std::unique_ptr<TH1> hist1, hist2;
  hist2Name += "_plot2";

  if (o.bins) {
    hist1.reset( var->createHistogram(hist1Name.c_str(), "Events", *o.bins) );
    hist2.reset( var->createHistogram(hist2Name.c_str(), "Events", *o.bins) );
  } else {
    hist1.reset( var->createHistogram(hist1Name.c_str(), "Events",
            frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
            frame->GetNbinsX()) );
    hist2.reset( var->createHistogram(hist2Name.c_str(), "Events",
            frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
            frame->GetNbinsX()) );
  }

  assert(hist1 && hist2);

  std::string cuts1,cuts2 ;
  if (o.cuts && strlen(o.cuts)) {
    cuts1 = Form("(%s)&&(%s>0)",o.cuts,asymCat.GetName());
    cuts2 = Form("(%s)&&(%s<0)",o.cuts,asymCat.GetName());
  } else {
    cuts1 = Form("(%s>0)",asymCat.GetName());
    cuts2 = Form("(%s<0)",asymCat.GetName());
  }

  if(! fillHistogram(hist1.get(), RooArgList(*var),cuts1.c_str(),o.cutRange) ||
     ! fillHistogram(hist2.get(), RooArgList(*var),cuts2.c_str(),o.cutRange)) {
    coutE(Plotting) << ClassName() << "::" << GetName()
    << ":plotAsymOn: createHistogram() failed" << endl;
    return nullptr;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist1,*hist2,0,1,o.etype,o.xErrorSize,false,o.scaleFactor);
  graph->setYAxisLabel(Form("Asymmetry in %s",asymCat.GetName())) ;

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);

  // Rename graph if requested
  if (o.histName) {
    graph->SetName(o.histName) ;
  } else {
    std::string hname{Form("h_%s_Asym[%s]",GetName(),asymCat.GetName())};
    if (o.cutRange && strlen(o.cutRange)>0) {
      hname += Form("_CutRange[%s]",o.cutRange);
    }
    if (o.cuts && strlen(o.cuts)>0) {
      hname += Form("_Cut[%s]",o.cuts);
    }
    graph->SetName(hname.c_str()) ;
  }

  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);

  return frame;
}

////////////////////////////////////////////////////////////////////////////////
/// Create and fill a histogram with the efficiency N[1] / ( N[1] + N[0] ),
/// where N(1/0) is the number of data points with effCat=1 and effCat=0
/// as function of the frames variable. The efficiency category 'effCat' must
/// have exactly 2 +1 and 0.
///
/// The plot range and the number of plot bins is determined by the parameters
/// of the plot variable of the frame (RooAbsReal::setPlotRange(), RooAbsReal::setPlotBins())
///
/// The optional cut string expression can be used to select the events to be plotted.
/// The cut specification may refer to any variable contained in the data set
///
/// The drawOptions are passed to the TH1::Draw() method

RooPlot* RooAbsData::plotEffOn(RooPlot* frame, const RooAbsCategoryLValue& effCat, PlotOpt o) const
{
  if(0 == frame) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotEffOn: frame is null" << endl;
    return nullptr;
  }
  RooAbsRealLValue *var= (RooAbsRealLValue*) frame->getPlotVar();
  if(0 == var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
    << ":plotEffOn: frame does not specify a plot variable" << endl;
    return nullptr;
  }

  // create and fill temporary histograms of this variable for each state
  std::string hist1Name(GetName()),hist2Name(GetName());
  hist1Name += "_plot1";
  std::unique_ptr<TH1> hist1, hist2;
  hist2Name += "_plot2";

  if (o.bins) {
    hist1.reset( var->createHistogram(hist1Name.c_str(), "Events", *o.bins) );
    hist2.reset( var->createHistogram(hist2Name.c_str(), "Events", *o.bins) );
  } else {
    hist1.reset( var->createHistogram(hist1Name.c_str(), "Events",
            frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
            frame->GetNbinsX()) );
    hist2.reset( var->createHistogram(hist2Name.c_str(), "Events",
            frame->GetXaxis()->GetXmin(), frame->GetXaxis()->GetXmax(),
            frame->GetNbinsX()) );
  }

  assert(hist1 && hist2);

  std::string cuts1,cuts2 ;
  if (o.cuts && strlen(o.cuts)) {
    cuts1 = Form("(%s)&&(%s==1)",o.cuts,effCat.GetName());
    cuts2 = Form("(%s)&&(%s==0)",o.cuts,effCat.GetName());
  } else {
    cuts1 = Form("(%s==1)",effCat.GetName());
    cuts2 = Form("(%s==0)",effCat.GetName());
  }

  if(! fillHistogram(hist1.get(), RooArgList(*var),cuts1.c_str(),o.cutRange) ||
     ! fillHistogram(hist2.get(), RooArgList(*var),cuts2.c_str(),o.cutRange)) {
    coutE(Plotting) << ClassName() << "::" << GetName()
    << ":plotEffOn: createHistogram() failed" << endl;
    return nullptr;
  }

  // convert this histogram to a RooHist object on the heap
  RooHist *graph= new RooHist(*hist1,*hist2,0,1,o.etype,o.xErrorSize,true);
  graph->setYAxisLabel(Form("Efficiency of %s=%s", effCat.GetName(), effCat.lookupName(1).c_str()));

  // initialize the frame's normalization setup, if necessary
  frame->updateNormVars(_vars);

  // Rename graph if requested
  if (o.histName) {
    graph->SetName(o.histName) ;
  } else {
      std::string hname(Form("h_%s_Eff[%s]",GetName(),effCat.GetName())) ;
    if (o.cutRange && strlen(o.cutRange)>0) {
      hname += Form("_CutRange[%s]",o.cutRange);
    }
    if (o.cuts && strlen(o.cuts)>0) {
      hname += Form("_Cut[%s]",o.cuts);
    }
    graph->SetName(hname.c_str()) ;
  }

  // add the RooHist to the specified plot
  frame->addPlotable(graph,o.drawOptions,o.histInvisible,o.refreshFrameNorm);

  return frame;
}

////////////////////////////////////////////////////////////////////////////////
/// Create and fill a 1-dimensional table for given category column
/// This functions is the equivalent of plotOn() for category dimensions.
///
/// The optional cut string expression can be used to select the events to be tabulated
/// The cut specification may refer to any variable contained in the data set
///
/// The option string is currently not used

Roo1DTable* RooAbsData::table(const RooAbsCategory& cat, const char* cuts, const char* /*opts*/) const
{
  // First see if var is in data set
  RooAbsCategory* tableVar = (RooAbsCategory*) _vars.find(cat.GetName()) ;
  std::unique_ptr<RooArgSet> tableSet;
  if (!tableVar) {
    if (!cat.dependsOn(_vars)) {
      coutE(Plotting) << "RooTreeData::Table(" << GetName() << "): Argument " << cat.GetName()
      << " is not in dataset and is also not dependent on data set" << endl ;
      return nullptr;
    }

    // Clone derived variable
    tableSet.reset(static_cast<RooArgSet*>(RooArgSet(cat).snapshot(true)));
    if (!tableSet) {
      coutE(Plotting) << "RooTreeData::table(" << GetName() << ") Couldn't deep-clone table category, abort." << std::endl;
      return nullptr;
    }
    tableVar = (RooAbsCategory*) tableSet->find(cat.GetName()) ;

    //Redirect servers of derived clone to internal ArgSet representing the data in this set
    tableVar->recursiveRedirectServers(_vars) ;
  }

  std::unique_ptr<RooFormulaVar> cutVar;
  std::string tableName{GetName()};
  if (cuts && strlen(cuts)) {
    tableName += "(";
    tableName += cuts;
    tableName += ")";
    // Make cut selector if cut is specified
    cutVar = std::make_unique<RooFormulaVar>("cutVar",cuts,_vars) ;
  }
  Roo1DTable* table2 = tableVar->createTable(tableName.c_str());

  // Dump contents
  Int_t nevent= numEntries() ;
  for(Int_t i=0; i < nevent; ++i) {
    get(i);

    if (cutVar && cutVar->getVal()==0) continue ;

    table2->fill(*tableVar,weight()) ;
  }

  return table2 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill Doubles 'lowest' and 'highest' with the lowest and highest value of
/// observable 'var' in this dataset. If the return value is true and error
/// occurred

bool RooAbsData::getRange(const RooAbsRealLValue& var, double& lowest, double& highest, double marginFrac, bool symMode) const
{
  // Lookup variable in dataset
  const auto arg = _vars.find(var.GetName());
  if (!arg) {
    coutE(InputArguments) << "RooDataSet::getRange(" << GetName() << ") ERROR: unknown variable: " << var.GetName() << endl ;
    return true;
  }

  auto varPtr = dynamic_cast<const RooRealVar*>(arg);
  // Check if found variable is of type RooRealVar
  if (!varPtr) {
    coutE(InputArguments) << "RooDataSet::getRange(" << GetName() << ") ERROR: variable " << var.GetName() << " is not of type RooRealVar" << endl ;
    return true;
  }

  // Check if dataset is not empty
  if(sumEntries() == 0.) {
    coutE(InputArguments) << "RooDataSet::getRange(" << GetName() << ") WARNING: empty dataset" << endl ;
    return true;
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
    if (symMode==false) {

      double margin = marginFrac*(highest-lowest) ;
      lowest -= margin ;
      highest += margin ;
      if (lowest<var.getMin()) lowest = var.getMin() ;
      if (highest>var.getMax()) highest = var.getMax() ;

    } else {

      double mom1 = moment(*varPtr,1) ;
      double delta = ((highest-mom1)>(mom1-lowest)?(highest-mom1):(mom1-lowest))*(1+marginFrac) ;
      lowest = mom1-delta ;
      highest = mom1+delta ;
      if (lowest<var.getMin()) lowest = var.getMin() ;
      if (highest>var.getMax()) highest = var.getMax() ;

    }
  }

  return false ;
}

////////////////////////////////////////////////////////////////////////////////
/// Prepare dataset for use with cached constant terms listed in
/// 'cacheList' of expression 'arg'. Deactivate tree branches
/// for any dataset observable that is either not used at all,
/// or is used exclusively by cached branch nodes.

void RooAbsData::optimizeReadingWithCaching(RooAbsArg& arg, const RooArgSet& cacheList, const RooArgSet& keepObsList)
{
  RooArgSet pruneSet ;

  // Add unused observables in this dataset to pruneSet
  pruneSet.add(*get()) ;
  std::unique_ptr<RooArgSet> usedObs{arg.getObservables(*this)};
  pruneSet.remove(*usedObs,true,true) ;

  // Add observables exclusively used to calculate cached observables to pruneSet
  for(auto * var : *get()) {
    if (allClientsCached(var,cacheList)) {
      pruneSet.add(*var) ;
    }
  }


  if (!pruneSet.empty()) {

    // Go over all used observables and check if any of them have parameterized
    // ranges in terms of pruned observables. If so, remove those observable
    // from the pruning list
    for(auto const* rrv : dynamic_range_cast<RooRealVar*>(*usedObs)) {
      if (rrv && !rrv->getBinning().isShareable()) {
        RooArgSet depObs ;
        RooAbsReal* loFunc = rrv->getBinning().lowBoundFunc() ;
        RooAbsReal* hiFunc = rrv->getBinning().highBoundFunc() ;
        if (loFunc) {
          loFunc->leafNodeServerList(&depObs,0,true) ;
        }
        if (hiFunc) {
          hiFunc->leafNodeServerList(&depObs,0,true) ;
        }
        if (!depObs.empty()) {
          pruneSet.remove(depObs,true,true) ;
        }
      }
    }
  }


  // Remove all observables in keep list from prune list
  pruneSet.remove(keepObsList,true,true) ;

  if (!pruneSet.empty()) {

    // Deactivate tree branches here
    cxcoutI(Optimization) << "RooTreeData::optimizeReadingForTestStatistic(" << GetName() << "): Observables " << pruneSet
             << " in dataset are either not used at all, orserving exclusively p.d.f nodes that are now cached, disabling reading of these observables for TTree" << endl ;
    setArgStatus(pruneSet,false) ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Utility function that determines if all clients of object 'var'
/// appear in given list of cached nodes.

bool RooAbsData::allClientsCached(RooAbsArg* var, const RooArgSet& cacheList)
{
  bool ret(true), anyClient(false) ;

  for (const auto client : var->valueClients()) {
    anyClient = true ;
    if (!cacheList.find(client->GetName())) {
      // If client is not cached recurse
      ret &= allClientsCached(client,cacheList) ;
    }
  }

  return anyClient?ret:false ;
}

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::attachBuffers(const RooArgSet& extObs)
{
  _dstore->attachBuffers(extObs) ;
}

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::resetBuffers()
{
  _dstore->resetBuffers() ;
}

////////////////////////////////////////////////////////////////////////////////

bool RooAbsData::canSplitFast() const
{
  return !_ownedComponents.empty();
}

////////////////////////////////////////////////////////////////////////////////

RooAbsData* RooAbsData::getSimData(const char* name)
{
  auto i = _ownedComponents.find(name);
  return i==_ownedComponents.end() ? nullptr : i->second;
}

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::addOwnedComponent(const char* idxlabel, RooAbsData& data)
{
  _ownedComponents[idxlabel]= &data ;
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class RooAbsData.

void RooAbsData::Streamer(TBuffer &R__b)
{
   if (R__b.IsReading()) {
      R__b.ReadClassBuffer(RooAbsData::Class(),this);
      _namePtr = RooNameReg::instance().constPtr(GetName()) ;

      // Convert on the fly to vector storage if that the current working default
      if (defaultStorageType==RooAbsData::Vector) {
   convertToVectorStore() ;
      }

   } else {
      R__b.WriteClassBuffer(RooAbsData::Class(),this);
   }
}

////////////////////////////////////////////////////////////////////////////////

void RooAbsData::checkInit() const
{
  _dstore->checkInit() ;
}

////////////////////////////////////////////////////////////////////////////////
/// Forward draw command to data store

void RooAbsData::Draw(Option_t* option)
{
  if (_dstore) _dstore->Draw(option) ;
}

////////////////////////////////////////////////////////////////////////////////

bool RooAbsData::hasFilledCache() const
{
  return _dstore->hasFilledCache() ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a pointer to the TTree which stores the data. Returns a nullpointer
/// if vector-based storage is used. The RooAbsData remains owner of the tree.
/// GetClonedTree() can be used to get a tree even if the internal storage does not use one.

const TTree *RooAbsData::tree() const
{
   if (storageType == RooAbsData::Tree) {
      return _dstore->tree();
   } else {
      coutW(InputArguments) << "RooAbsData::tree(" << GetName() << ") WARNING: is not of StorageType::Tree. "
                            << "Use GetClonedTree() instead or convert to tree storage." << endl;
      return nullptr;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return a clone of the TTree which stores the data or create such a tree
/// if vector storage is used. The user is responsible for deleting the tree

TTree *RooAbsData::GetClonedTree() const
{
   if (storageType == RooAbsData::Tree) {
      auto tmp = const_cast<TTree *>(_dstore->tree());
      return tmp->CloneTree();
   } else {
      RooTreeDataStore buffer(GetName(), GetTitle(), *get(), *_dstore);
      return buffer.tree().CloneTree();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Convert vector-based storage to tree-based storage

void RooAbsData::convertToTreeStore()
{
   if (storageType != RooAbsData::Tree) {
      auto *newStore = new RooTreeDataStore(GetName(), GetTitle(), _vars, *_dstore);
      delete _dstore;
      _dstore = newStore;
      storageType = RooAbsData::Tree;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// If one of the TObject we have a referenced to is deleted, remove the
/// reference.

void RooAbsData::RecursiveRemove(TObject *obj)
{
  for(auto &iter : _ownedComponents) {
    if (iter.second == obj) {
      iter.second = nullptr;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Sets the global observables stored in this data. A snapshot of the
/// observables will be saved.
/// \param[in] globalObservables The set of global observables to take a snapshot of.

void RooAbsData::setGlobalObservables(RooArgSet const& globalObservables) {
  if(_globalObservables == nullptr) _globalObservables = std::make_unique<RooArgSet>();
  else _globalObservables->clear();
  globalObservables.snapshot(*_globalObservables);
  for(auto * arg : *_globalObservables) {
    arg->setAttribute("global",true);
    // Global observables are also always constant in fits
    if(auto lval = dynamic_cast<RooAbsRealLValue*>(arg)) lval->setConstant(true);
    if(auto lval = dynamic_cast<RooAbsCategoryLValue*>(arg)) lval->setConstant(true);
  }
}


////////////////////////////////////////////////////////////////////////////////

void RooAbsData::SetName(const char* name)
{
  TNamed::SetName(name) ;
  auto newPtr = RooNameReg::instance().constPtr(GetName()) ;
  if (newPtr != _namePtr) {
    //cout << "Rename '" << _namePtr->GetName() << "' to '" << name << "' (set flag in new name)" << endl;
    _namePtr = newPtr;
    const_cast<TNamed*>(_namePtr)->SetBit(RooNameReg::kRenamedArg);
    RooNameReg::incrementRenameCounter();
  }
}




////////////////////////////////////////////////////////////////////////////////

void RooAbsData::SetNameTitle(const char *name, const char *title)
{
  TNamed::SetTitle(title) ;
  SetName(name);
}



////////////////////////////////////////////////////////////////////////////////
/// Return sum of squared weights of this data.

double RooAbsData::sumEntriesW2() const {
  const RooSpan<const double> eventWeights = getWeightBatch(0, numEntries(), /*sumW2=*/true);
  if (eventWeights.empty()) {
    return numEntries() * weightSquared();
  }

  ROOT::Math::KahanSum<double, 4u> kahanWeight;
  for (std::size_t i = 0; i < eventWeights.size(); ++i) {
    kahanWeight.AddIndexed(eventWeights[i], i);
  }
  return kahanWeight.Sum();
}


////////////////////////////////////////////////////////////////////////////////
/// Write information to retrieve data columns into `evalData.spans`.
/// All spans belonging to variables of this dataset are overwritten. Spans to other
/// variables remain intact.
/// \param[out] evalData Store references to all data batches in this struct's `spans`.
/// The key to retrieve an item is the pointer of the variable that owns the data.
/// \param begin Index of first event that ends up in the batch.
/// \param len   Number of events in each batch.
RooAbsData::RealSpans RooAbsData::getBatches(std::size_t begin, std::size_t len) const {
  return store()->getBatches(begin, len);
}


RooAbsData::CategorySpans RooAbsData::getCategoryBatches(std::size_t first, std::size_t len) const {
  return store()->getCategoryBatches(first, len);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a TH2F histogram of the distribution of the specified variable
/// using this dataset. Apply any cuts to select which events are used.
/// The variable being plotted can either be contained directly in this
/// dataset, or else be a function of the variables in this dataset.
/// The histogram will be created using RooAbsReal::createHistogram() with
/// the name provided (with our dataset name prepended).

TH2F *RooAbsData::createHistogram(const RooAbsRealLValue &var1, const RooAbsRealLValue &var2, const char *cuts,
                                  const char *name) const
{
   checkInit();
   return createHistogram(var1, var2, var1.getBins(), var2.getBins(), cuts, name);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a TH2F histogram of the distribution of the specified variable
/// using this dataset. Apply any cuts to select which events are used.
/// The variable being plotted can either be contained directly in this
/// dataset, or else be a function of the variables in this dataset.
/// The histogram will be created using RooAbsReal::createHistogram() with
/// the name provided (with our dataset name prepended).

TH2F *RooAbsData::createHistogram(const RooAbsRealLValue &var1, const RooAbsRealLValue &var2, int nx, int ny,
                                  const char *cuts, const char *name) const
{
   checkInit();
   static int counter(0);

   std::unique_ptr<RooAbsReal> ownedPlotVarX;
   // Is this variable in our dataset?
   auto *plotVarX = static_cast<RooAbsReal *>(_vars.find(var1.GetName()));
   if (plotVarX == nullptr) {
      // Is this variable a client of our dataset?
      if (!var1.dependsOn(_vars)) {
         coutE(InputArguments) << GetName() << "::createHistogram: Argument " << var1.GetName()
                               << " is not in dataset and is also not dependent on data set" << std::endl;
         return nullptr;
      }

      // Clone derived variable
      ownedPlotVarX.reset(static_cast<RooAbsReal *>(var1.Clone()));
      plotVarX = ownedPlotVarX.get();

      // Redirect servers of derived clone to internal ArgSet representing the data in this set
      plotVarX->redirectServers(const_cast<RooArgSet &>(_vars));
   }

   std::unique_ptr<RooAbsReal>  ownedPlotVarY;
   // Is this variable in our dataset?
   RooAbsReal *plotVarY = (RooAbsReal *)_vars.find(var2.GetName());
   if (plotVarY == nullptr) {
      // Is this variable a client of our dataset?
      if (!var2.dependsOn(_vars)) {
         coutE(InputArguments) << GetName() << "::createHistogram: Argument " << var2.GetName()
                               << " is not in dataset and is also not dependent on data set" << std::endl;
         return nullptr;
      }

      // Clone derived variable
      ownedPlotVarY.reset(static_cast<RooAbsReal *>(var2.Clone()));
      plotVarY = ownedPlotVarY.get();

      // Redirect servers of derived clone to internal ArgSet representing the data in this set
      plotVarY->redirectServers(const_cast<RooArgSet &>(_vars));
   }

   // Create selection formula if selection cuts are specified
   std::unique_ptr<RooFormula> select;
   if (0 != cuts && strlen(cuts)) {
      select = std::make_unique<RooFormula>(cuts, cuts, _vars);
      if (!select->ok()) {
         return nullptr;
      }
   }

   const std::string histName = std::string{GetName()} + "_" + name  + "_" + Form("%08x", counter++);

   // create the histogram
   auto *histogram =
      new TH2F(histName.c_str(), "Events", nx, var1.getMin(), var1.getMax(), ny, var2.getMin(), var2.getMax());
   if (!histogram) {
      coutE(DataHandling) << GetName() << "::createHistogram: unable to create a new histogram" << endl;
      return nullptr;
   }

   // Dump contents
   Int_t nevent = numEntries();
   for (Int_t i = 0; i < nevent; ++i) {
      get(i);

      if (select && select->eval() == 0)
         continue;
      histogram->Fill(plotVarX->getVal(), plotVarY->getVal(), weight());
   }

   return histogram;
}
