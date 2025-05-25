/// \cond ROOFIT_INTERNAL

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
\file RooAbsTestStatistic.cxx
\class RooAbsTestStatistic
\ingroup Roofitcore

Abstract base class for all test
statistics. Test statistics that evaluate the PDF at each data
point should inherit from the RooAbsOptTestStatistic class which
implements several generic optimizations that can be done for such
quantities.

This test statistic base class organizes calculation of test
statistic values for RooSimultaneous PDF as a combination of test
statistic values for the PDF components of the simultaneous PDF and
organizes multi-processor parallel calculation of test statistic
values. For the latter, the test statistic value is calculated in
partitions in parallel executing processes and a posteriori
combined in the main thread.
**/

#include "RooAbsTestStatistic.h"

#include "RooAbsPdf.h"
#include "RooSimultaneous.h"
#include "RooAbsData.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooRealMPFE.h"
#include "RooErrorHandler.h"
#include "RooMsgService.h"
#include "RooAbsCategoryLValue.h"
#include "RooFitImplHelpers.h"
#include "RooAbsOptTestStatistic.h"
#include "RooCategory.h"

#include "TTimeStamp.h"
#include "TClass.h"
#include <string>
#include <stdexcept>

using std::endl, std::ostream;

////////////////////////////////////////////////////////////////////////////////
/// Create a test statistic from the given function and the data.
/// \param[in] name Name of the test statistic
/// \param[in] title Title (for plotting)
/// \param[in] real Function to be used for tests
/// \param[in] data Data to fit function to
/// \param[in] projDeps A set of projected observables
/// \param[in] cfg statistic configuration object
///
/// cfg contains:
/// - rangeName Fit data only in range with given name
/// - addCoefRangeName If not null, all RooAddPdf components of `real` will be instructed to fix their fraction definitions to the given named range.
/// - nCPU If larger than one, the test statistic calculation will be parallelized over multiple processes.
///   By default the data is split with 'bulk' partitioning (each process calculates a contiguous block of fraction 1/nCPU
///   of the data). For binned data this approach may be suboptimal as the number of bins with >0 entries
///   in each processing block many vary greatly thereby distributing the workload rather unevenly.
/// - interleave is set to true, the interleave partitioning strategy is used where each partition
///   i takes all bins for which (ibin % ncpu == i) which is more likely to result in an even workload.
/// - verbose Be more verbose.
/// - splitCutRange If true, a different rangeName constructed as rangeName_{catName} will be used
///    as range definition for each index state of a RooSimultaneous. This means that a different range can be defined
///    for each category such as
///    ```
///    myVariable.setRange("range_pi0", 135, 210);
///    myVariable.setRange("range_gamma", 50, 210);
///    ```
///    if the categories are called "pi0" and "gamma".

RooAbsTestStatistic::RooAbsTestStatistic(const char *name, const char *title, RooAbsReal& real, RooAbsData& data,
                                         const RooArgSet& projDeps, RooAbsTestStatistic::Configuration const& cfg) :
  RooAbsReal(name,title),
  _paramSet("paramSet","Set of parameters",this),
  _func(&real),
  _data(&data),
  _projDeps(static_cast<RooArgSet*>(projDeps.Clone())),
  _rangeName(cfg.rangeName),
  _addCoefRangeName(cfg.addCoefRangeName),
  _splitRange(cfg.splitCutRange),
  _verbose(cfg.verbose),
  // Determine if RooAbsReal is a RooSimultaneous
  _gofOpMode{(cfg.nCPU>1 || cfg.nCPU==-1) ? MPMaster : (dynamic_cast<RooSimultaneous*>(_func) ? SimMaster : Slave)},
  _nEvents{data.numEntries()},
  _nCPU(cfg.nCPU != -1 ? cfg.nCPU : 1),
  _mpinterl(cfg.interleave),
  _takeGlobalObservablesFromData{cfg.takeGlobalObservablesFromData}
{
  // Register all parameters as servers
  _paramSet.add(*std::unique_ptr<RooArgSet>{real.getParameters(&data)});
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsTestStatistic::RooAbsTestStatistic(const RooAbsTestStatistic& other, const char* name) :
  RooAbsReal(other,name),
  _paramSet("paramSet","Set of parameters",this),
  _func(other._func),
  _data(other._data),
  _projDeps(static_cast<RooArgSet*>(other._projDeps->Clone())),
  _rangeName(other._rangeName),
  _addCoefRangeName(other._addCoefRangeName),
  _splitRange(other._splitRange),
  _verbose(other._verbose),
  // Determine if RooAbsReal is a RooSimultaneous
  _gofOpMode{(other._nCPU>1 || other._nCPU==-1) ? MPMaster : (dynamic_cast<RooSimultaneous*>(_func) ? SimMaster : Slave)},
  _nEvents{_data->numEntries()},
  _nCPU(other._nCPU != -1 ? other._nCPU : 1),
  _mpinterl(other._mpinterl),
  _doOffset(other._doOffset),
  _takeGlobalObservablesFromData{other._takeGlobalObservablesFromData},
  _offset(other._offset),
  _evalCarry(other._evalCarry)
{
  // Our parameters are those of original
  _paramSet.add(other._paramSet) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsTestStatistic::~RooAbsTestStatistic()
{
  if (MPMaster == _gofOpMode && _init) {
    for (Int_t i = 0; i < _nCPU; ++i) delete _mpfeArray[i];
    delete[] _mpfeArray ;
  }

  delete _projDeps ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate and return value of test statistic. If the test statistic
/// is calculated from a RooSimultaneous, the test statistic calculation
/// is performed separately on each simultaneous p.d.f component and associated
/// data, and then combined. If the test statistic calculation is parallelized,
/// partitions are calculated in nCPU processes and combined a posteriori.

double RooAbsTestStatistic::evaluate() const
{
  // One-time Initialization
  if (!_init) {
    const_cast<RooAbsTestStatistic*>(this)->initialize() ;
  }

  if (SimMaster == _gofOpMode) {
    // Evaluate array of owned GOF objects
    double ret = 0.;

    if (_mpinterl == RooFit::BulkPartition || _mpinterl == RooFit::Interleave ) {
      ret = combinedValue(reinterpret_cast<RooAbsReal**>(const_cast<std::unique_ptr<RooAbsTestStatistic>*>(_gofArray.data())),_gofArray.size());
    } else {
      double sum = 0.;
      double carry = 0.;
      int i = 0;
      for (auto& gof : _gofArray) {
        if (i % _numSets == _setNum || (_mpinterl==RooFit::Hybrid && gof->_mpinterl != RooFit::SimComponents )) {
          double y = gof->getValV();
          carry += gof->getCarry();
          y -= carry;
          const double t = sum + y;
          carry = (t - sum) - y;
          sum = t;
        }
        ++i;
      }
      ret = sum ;
      _evalCarry = carry;
    }

    // Only apply global normalization if SimMaster doesn't have MP master
    if (numSets()==1) {
      const double norm = globalNormalization();
      ret /= norm;
      _evalCarry /= norm;
    }

    return ret ;

  } else if (MPMaster == _gofOpMode) {

    // Start calculations in parallel
    for (Int_t i = 0; i < _nCPU; ++i) _mpfeArray[i]->calculate();

    double sum(0);
    double carry = 0.;
    for (Int_t i = 0; i < _nCPU; ++i) {
      double y = _mpfeArray[i]->getValV();
      carry += _mpfeArray[i]->getCarry();
      y -= carry;
      const double t = sum + y;
      carry = (t - sum) - y;
      sum = t;
    }

    double ret = sum ;
    _evalCarry = carry;

    const double norm = globalNormalization();
    ret /= norm;
    _evalCarry /= norm;

    return ret ;

  } else {

    // Evaluate as straight FUNC
    Int_t nFirst(0);
    Int_t nLast(_nEvents);
    Int_t nStep(1);

    switch (_mpinterl) {
    case RooFit::BulkPartition:
      nFirst = _nEvents * _setNum / _numSets ;
      nLast  = _nEvents * (_setNum+1) / _numSets ;
      nStep  = 1 ;
      break;

    case RooFit::Interleave:
      nFirst = _setNum ;
      nLast  = _nEvents ;
      nStep  = _numSets ;
      break ;

    case RooFit::SimComponents:
      nFirst = 0 ;
      nLast  = _nEvents ;
      nStep  = 1 ;
      break ;

    case RooFit::Hybrid:
      throw std::logic_error("this should never happen");
      break ;
    }

    runRecalculateCache(nFirst, nLast, nStep);
    double ret = evaluatePartition(nFirst,nLast,nStep);

    if (numSets()==1) {
      const double norm = globalNormalization();
      ret /= norm;
      _evalCarry /= norm;
    }

    return ret ;

  }
}



////////////////////////////////////////////////////////////////////////////////
/// One-time initialization of the test statistic. Setup
/// infrastructure for simultaneous p.d.f processing and/or
/// parallelized processing if requested

bool RooAbsTestStatistic::initialize()
{
  if (_init) return false;

  if (MPMaster == _gofOpMode) {
    initMPMode(_func,_data,_projDeps,_rangeName,_addCoefRangeName) ;
  } else if (SimMaster == _gofOpMode) {
    initSimMode(static_cast<RooSimultaneous*>(_func),_data,_projDeps,_rangeName,_addCoefRangeName) ;
  }
  _init = true;
  return false;
}



////////////////////////////////////////////////////////////////////////////////
/// Forward server redirect calls to component test statistics

bool RooAbsTestStatistic::redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive)
{
  if (SimMaster == _gofOpMode) {
    // Forward to slaves
    for(auto& gof : _gofArray) {
      gof->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange);
    }
  } else if (MPMaster == _gofOpMode&& _mpfeArray) {
    // Forward to slaves
    for (Int_t i = 0; i < _nCPU; ++i) {
      if (_mpfeArray[i]) {
   _mpfeArray[i]->recursiveRedirectServers(newServerList,mustReplaceAll,nameChange);
//    std::cout << "redirecting servers on " << _mpfeArray[i]->GetName() << std::endl;
      }
    }
  }
  return RooAbsReal::redirectServersHook(newServerList, mustReplaceAll, nameChange, isRecursive);
}



////////////////////////////////////////////////////////////////////////////////
/// Add extra information on component test statistics when printing
/// itself as part of a tree structure

void RooAbsTestStatistic::printCompactTreeHook(ostream& os, const char* indent)
{
  if (SimMaster == _gofOpMode) {
    // Forward to slaves
    os << indent << "RooAbsTestStatistic begin GOF contents" << std::endl ;
    for (std::size_t i = 0; i < _gofArray.size(); ++i) {
      TString indent2(indent);
      indent2 += "[" + std::to_string(i) + "] ";
      _gofArray[i]->printCompactTreeHook(os,indent2);
    }
    os << indent << "RooAbsTestStatistic end GOF contents" << std::endl;
  } else if (MPMaster == _gofOpMode) {
    // WVE implement this
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Forward constant term optimization management calls to component
/// test statistics

void RooAbsTestStatistic::constOptimizeTestStatistic(ConstOpCode opcode, bool doAlsoTrackingOpt)
{
  initialize();
  if (SimMaster == _gofOpMode) {
    // Forward to slaves
    int i = 0;
    for (auto& gof : _gofArray) {
      // In SimComponents Splitting strategy only constOptimize the terms that are actually used
      RooFit::MPSplit effSplit = (_mpinterl!=RooFit::Hybrid) ? _mpinterl : gof->_mpinterl;
      if ( (i % _numSets == _setNum) || (effSplit != RooFit::SimComponents) ) {
        gof->constOptimizeTestStatistic(opcode,doAlsoTrackingOpt);
      }
      ++i;
    }
  } else if (MPMaster == _gofOpMode) {
    for (Int_t i = 0; i < _nCPU; ++i) {
      _mpfeArray[i]->constOptimizeTestStatistic(opcode,doAlsoTrackingOpt);
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Set MultiProcessor set number identification of this instance

void RooAbsTestStatistic::setMPSet(Int_t inSetNum, Int_t inNumSets)
{
  _setNum = inSetNum; _numSets = inNumSets;
  _extSet = _mpinterl==RooFit::SimComponents ? _setNum : (_numSets - 1);

  if (SimMaster == _gofOpMode) {
    // Forward to slaves
    initialize();
    for(auto& gof : _gofArray) {
      gof->setMPSet(inSetNum,inNumSets);
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize multi-processor calculation mode. Create component test statistics in separate
/// processed that are connected to this process through a RooAbsRealMPFE front-end class.

void RooAbsTestStatistic::initMPMode(RooAbsReal* real, RooAbsData* data, const RooArgSet* projDeps, std::string const& rangeName, std::string const& addCoefRangeName)
{
  _mpfeArray = new pRooRealMPFE[_nCPU];

  // Create proto-goodness-of-fit
  Configuration cfg;
  cfg.rangeName = rangeName;
  cfg.addCoefRangeName = addCoefRangeName;
  cfg.nCPU = 1;
  cfg.interleave = _mpinterl;
  cfg.verbose = _verbose;
  cfg.splitCutRange = _splitRange;
  cfg.takeGlobalObservablesFromData = _takeGlobalObservablesFromData;
  // This configuration parameter is stored in the RooAbsOptTestStatistic.
  // It would have been cleaner to move the member variable into RooAbsTestStatistic,
  // but to avoid incrementing the class version we do the dynamic_cast trick.
  if(auto thisAsRooAbsOptTestStatistic = dynamic_cast<RooAbsOptTestStatistic const*>(this)) {
    cfg.integrateOverBinsPrecision = thisAsRooAbsOptTestStatistic->_integrateBinsPrecision;
  }
  RooAbsTestStatistic* gof = create(GetName(),GetTitle(),*real,*data,*projDeps,cfg);
  gof->recursiveRedirectServers(_paramSet);

  for (Int_t i = 0; i < _nCPU; ++i) {
    gof->setMPSet(i,_nCPU);
    gof->SetName(Form("%s_GOF%d",GetName(),i));
    gof->SetTitle(Form("%s_GOF%d",GetTitle(),i));

    ccoutD(Eval) << "RooAbsTestStatistic::initMPMode: starting remote server process #" << i << std::endl;
    _mpfeArray[i] = new RooRealMPFE(Form("%s_%zx_MPFE%d",GetName(),reinterpret_cast<size_t>(this),i),Form("%s_%zx_MPFE%d",GetTitle(),reinterpret_cast<size_t>(this),i),*gof,false);
    //_mpfeArray[i]->setVerbose(true,true);
    _mpfeArray[i]->initialize();
    if (i > 0) {
      _mpfeArray[i]->followAsSlave(*_mpfeArray[0]);
    }
  }
  _mpfeArray[_nCPU - 1]->addOwnedComponents(*gof);
  coutI(Eval) << "RooAbsTestStatistic::initMPMode: started " << _nCPU << " remote server process." << std::endl;
  //cout << "initMPMode --- done" << std::endl ;
  return ;
}



////////////////////////////////////////////////////////////////////////////////
/// Initialize simultaneous p.d.f processing mode. Strip simultaneous
/// p.d.f into individual components, split dataset in subset
/// matching each component and create component test statistics for
/// each of them.

void RooAbsTestStatistic::initSimMode(RooSimultaneous* simpdf, RooAbsData* data,
                                      const RooArgSet* projDeps,
                                      std::string const& rangeName, std::string const& addCoefRangeName)
{

  RooAbsCategoryLValue& simCat = const_cast<RooAbsCategoryLValue&>(simpdf->indexCat());

  std::unique_ptr<TList> dsetList{const_cast<RooAbsData*>(data)->split(*simpdf,processEmptyDataSets())};

  // Create array of regular fit contexts, containing subset of data and single fitCat PDF
  for (const auto& catState : simCat) {
    const std::string& catName = catState.first;
    RooAbsCategory::value_type catIndex = catState.second;

    // If the channel is not in the selected range of the category variable, we
    // won't create a slave calculator for this channel.
    if(!rangeName.empty()) {
      // Only the RooCategory supports ranges, not the other
      // RooAbsCategoryLValue-derived classes.
      auto simCatAsRooCategory = dynamic_cast<RooCategory*>(&simCat);
      if(simCatAsRooCategory && !simCatAsRooCategory->isStateInRange(rangeName.c_str(), catIndex)) {
        continue;
      }
    }

    // Retrieve the PDF for this simCat state
    RooAbsPdf* pdf = simpdf->getPdf(catName.c_str());
    RooAbsData* dset = static_cast<RooAbsData*>(dsetList->FindObject(catName.c_str()));

    if (pdf && dset && (0. != dset->sumEntries() || processEmptyDataSets())) {
      ccoutI(Fitting) << "RooAbsTestStatistic::initSimMode: creating slave calculator #" << _gofArray.size() << " for state " << catName
          << " (" << dset->numEntries() << " dataset entries)" << std::endl;


      // *** START HERE
      // WVE HACK determine if we have a RooRealSumPdf and then treat it like a binned likelihood
      auto binnedInfo = RooHelpers::getBinnedL(*pdf);
      RooAbsReal &actualPdf = binnedInfo.binnedPdf ? *binnedInfo.binnedPdf : *pdf;
      // WVE END HACK
      // Below here directly pass binnedPdf instead of PROD(binnedPdf,constraints) as constraints are evaluated elsewhere anyway
      // and omitting them reduces model complexity and associated handling/cloning times
      Configuration cfg;
      cfg.addCoefRangeName = addCoefRangeName;
      cfg.interleave = _mpinterl;
      cfg.verbose = _verbose;
      cfg.splitCutRange = _splitRange;
      cfg.binnedL = binnedInfo.isBinnedL;
      cfg.takeGlobalObservablesFromData = _takeGlobalObservablesFromData;
      // This configuration parameter is stored in the RooAbsOptTestStatistic.
      // It would have been cleaner to move the member variable into RooAbsTestStatistic,
      // but to avoid incrementing the class version we do the dynamic_cast trick.
      if(auto thisAsRooAbsOptTestStatistic = dynamic_cast<RooAbsOptTestStatistic const*>(this)) {
        cfg.integrateOverBinsPrecision = thisAsRooAbsOptTestStatistic->_integrateBinsPrecision;
      }
      cfg.rangeName = RooHelpers::getRangeNameForSimComponent(rangeName, _splitRange, catName);
      cfg.nCPU = _nCPU;
      _gofArray.emplace_back(create(catName.c_str(),catName.c_str(),actualPdf,*dset,*projDeps,cfg));
      // *** END HERE

      // Fill per-component split mode with Bulk Partition for now so that Auto will map to bulk-splitting of all components
      if (_mpinterl==RooFit::Hybrid) {
        _gofArray.back()->_mpinterl = dset->numEntries()<10 ? RooFit::SimComponents : RooFit::BulkPartition;
      }

      // Servers may have been redirected between instantiation and (deferred) initialization

      RooArgSet actualParams;
      actualPdf.getParameters(dset->get(), actualParams);
      RooArgSet selTargetParams;
      _paramSet.selectCommon(actualParams, selTargetParams);

      _gofArray.back()->recursiveRedirectServers(selTargetParams);
    }
  }
  for(auto& gof : _gofArray) {
    gof->setSimCount(_gofArray.size());
  }
  coutI(Fitting) << "RooAbsTestStatistic::initSimMode: created " << _gofArray.size() << " slave calculators." << std::endl;

  dsetList->Delete(); // delete the content.
}


////////////////////////////////////////////////////////////////////////////////
/// Change dataset that is used to given one. If cloneData is true, a clone of
/// in the input dataset is made.  If the test statistic was constructed with
/// a range specification on the data, the cloneData argument is ignored and
/// the data is always cloned.
bool RooAbsTestStatistic::setData(RooAbsData& indata, bool cloneData)
{
  // Trigger refresh of likelihood offsets
  if (isOffsetting()) {
    enableOffsetting(false);
    enableOffsetting(true);
  }

  switch(operMode()) {
  case Slave:
    // Delegate to implementation
    return setDataSlave(indata, cloneData);
  case SimMaster:
    // Forward to slaves
    if (indata.canSplitFast()) {
      for(auto& gof : _gofArray) {
        RooAbsData* compData = indata.getSimData(gof->GetName());
        gof->setDataSlave(*compData, cloneData);
      }
    } else if (0 == indata.numEntries()) {
      // For an unsplit empty dataset, simply assign empty dataset to each component
      for(auto& gof : _gofArray) {
        gof->setDataSlave(indata, cloneData);
      }
    } else {
      std::unique_ptr<TList> dlist{indata.split(*static_cast<RooSimultaneous*>(_func), processEmptyDataSets())};
      if (!dlist) {
        coutE(Fitting) << "RooAbsTestStatistic::initSimMode(" << GetName() << ") ERROR: index category of simultaneous pdf is missing in dataset, aborting" << std::endl;
        throw std::runtime_error("RooAbsTestStatistic::initSimMode() ERROR, index category of simultaneous pdf is missing in dataset, aborting");
      }

      for(auto& gof : _gofArray) {
        if (auto compData = static_cast<RooAbsData*>(dlist->FindObject(gof->GetName()))) {
          gof->setDataSlave(*compData,false,true);
        } else {
          coutE(DataHandling) << "RooAbsTestStatistic::setData(" << GetName() << ") ERROR: Cannot find component data for state " << gof->GetName() << std::endl;
        }
      }
    }
    break;
  case MPMaster:
    // Not supported
    coutF(DataHandling) << "RooAbsTestStatistic::setData(" << GetName() << ") FATAL: setData() is not supported in multi-processor mode" << std::endl;
    throw std::runtime_error("RooAbsTestStatistic::setData is not supported in MPMaster mode");
    break;
  }

  return true;
}



void RooAbsTestStatistic::enableOffsetting(bool flag)
{
  // Apply internal value offsetting to control numeric precision
  if (!_init) {
    const_cast<RooAbsTestStatistic*>(this)->initialize() ;
  }

  switch(operMode()) {
  case Slave:
    _doOffset = flag ;
    // Clear offset if feature is disabled to that it is recalculated next time it is enabled
    if (!_doOffset) {
      _offset = ROOT::Math::KahanSum<double>{0.} ;
    }
    setValueDirty() ;
    break ;
  case SimMaster:
    _doOffset = flag;
    for(auto& gof : _gofArray) {
      gof->enableOffsetting(flag);
    }
    break ;
  case MPMaster:
    _doOffset = flag;
    for (Int_t i = 0; i < _nCPU; ++i) {
      _mpfeArray[i]->enableOffsetting(flag);
    }
    break;
  }
}


double RooAbsTestStatistic::getCarry() const
{ return _evalCarry; }

/// \endcond
