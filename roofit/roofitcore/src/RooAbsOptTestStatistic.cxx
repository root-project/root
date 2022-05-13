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
\file RooAbsOptTestStatistic.cxx
\class RooAbsOptTestStatistic
\ingroup Roofitcore

RooAbsOptTestStatistic is the abstract base class for test
statistics objects that evaluate a function or PDF at each point of a given
dataset.  This class provides generic optimizations, such as
caching and precalculation of constant terms that can be made for
all such quantities.

Implementations should define evaluatePartition(), which calculates the
value of a (sub)range of the dataset and optionally combinedValue(),
which combines the values calculated for each partition. If combinedValue()
is not overloaded, the default implementation will add the partition results
to obtain the combined result.

Support for calculation in partitions is needed to allow multi-core
parallelized calculation of test statistics.
**/

#include "Riostream.h"
#include "TClass.h"
#include <string.h>


#include "RooAbsOptTestStatistic.h"
#include "RooMsgService.h"
#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooDataHist.h"
#include "RooArgSet.h"
#include "RooRealVar.h"
#include "RooErrorHandler.h"
#include "RooGlobalFunc.h"
#include "RooBinning.h"
#include "RooAbsDataStore.h"
#include "RooCategory.h"
#include "RooDataSet.h"
#include "RooProdPdf.h"
#include "RooAddPdf.h"
#include "RooProduct.h"
#include "RooRealSumPdf.h"
#include "RooTrace.h"
#include "RooVectorDataStore.h"
#include "RooBinSamplingPdf.h"

using namespace std;

ClassImp(RooAbsOptTestStatistic);
;


////////////////////////////////////////////////////////////////////////////////
/// Default Constructor

RooAbsOptTestStatistic:: RooAbsOptTestStatistic()
{
  // Initialize all non-persisted data members

  _funcObsSet = 0 ;
  _funcCloneSet = 0 ;
  _funcClone = 0 ;

  _normSet = 0 ;
  _projDeps = 0 ;

  _origFunc = 0 ;
  _origData = 0 ;

  _ownData = true ;
  _sealed = false ;
  _optimized = false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create a test statistic, and optimise its calculation.
/// \param[in] name Name of the instance.
/// \param[in] title Title (for e.g. plotting).
/// \param[in] real Function to evaluate.
/// \param[in] indata Dataset for which to compute test statistic.
/// \param[in] projDeps A set of projected observables.
/// \param[in] cfg the statistic configuration
///
/// cfg contains:
/// - rangeName If not null, only events in the dataset inside the range will be used in the test
///   statistic calculation.
/// - addCoefRangeName If not null, all RooAddPdf components of `real` will be
///   instructed to fix their fraction definitions to the given named range.
/// - nCPU If > 1, the test statistic calculation will be parallelised over multiple processes. By default, the data
///   is split with 'bulk' partitioning (each process calculates a contiguous block of fraction 1/nCPU
///   of the data). For binned data, this approach may be suboptimal as the number of bins with >0 entries
///   in each processing block may vary greatly; thereby distributing the workload rather unevenly.
/// - interleave Strategy how to distribute events among workers. If an interleave partitioning strategy is used where each partition
///   i takes all bins for which (ibin % ncpu == i), an even distribution of work is more likely.
/// - splitCutRange If true, a different rangeName constructed as `rangeName_{catName}` will be used
///   as range definition for each index state of a RooSimultaneous.
/// - cloneInputData Not used. Data is always cloned.
/// - integrateOverBinsPrecision If > 0, PDF in binned fits are integrated over the bins. This sets the precision. If = 0,
///   only unbinned PDFs fit to RooDataHist are integrated. If < 0, PDFs are never integrated.
RooAbsOptTestStatistic::RooAbsOptTestStatistic(const char *name, const char *title, RooAbsReal& real,
                                               RooAbsData& indata, const RooArgSet& projDeps,
                                               RooAbsTestStatistic::Configuration const& cfg) :
  RooAbsTestStatistic(name,title,real,indata,projDeps,cfg),
  _projDeps(0),
  _sealed(false),
  _optimized(false),
  _integrateBinsPrecision(cfg.integrateOverBinsPrecision)
{
  // Don't do a thing in master mode

  if (operMode()!=Slave) {
    _funcObsSet = 0 ;
    _funcCloneSet = 0 ;
    _funcClone = 0 ;
    _normSet = 0 ;
    _projDeps = 0 ;
    _origFunc = 0 ;
    _origData = 0 ;
    _ownData = false ;
    _sealed = false ;
    return ;
  }

  _origFunc = 0 ; //other._origFunc ;
  _origData = 0 ; // other._origData ;

  initSlave(real, indata, projDeps, _rangeName.c_str(), _addCoefRangeName.c_str()) ;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooAbsOptTestStatistic::RooAbsOptTestStatistic(const RooAbsOptTestStatistic& other, const char* name) :
  RooAbsTestStatistic(other,name), _sealed(other._sealed), _sealNotice(other._sealNotice), _optimized(false),
  _integrateBinsPrecision(other._integrateBinsPrecision)
{
  // Don't do a thing in master mode
  if (operMode()!=Slave) {

    _funcObsSet = 0 ;
    _funcCloneSet = 0 ;
    _funcClone = 0 ;
    _normSet = other._normSet ? ((RooArgSet*) other._normSet->snapshot()) : 0 ;
    _projDeps = 0 ;
    _origFunc = 0 ;
    _origData = 0 ;
    _ownData = false ;
    return ;
  }

  _origFunc = 0 ; //other._origFunc ;
  _origData = 0 ; // other._origData ;
  _projDeps = 0 ;

  initSlave(*other._funcClone,*other._dataClone,other._projDeps?*other._projDeps:RooArgSet(),other._rangeName.c_str(),other._addCoefRangeName.c_str()) ;
}



////////////////////////////////////////////////////////////////////////////////

void RooAbsOptTestStatistic::initSlave(RooAbsReal& real, RooAbsData& indata, const RooArgSet& projDeps, const char* rangeName,
                   const char* addCoefRangeName) {
  // ******************************************************************
  // *** PART 1 *** Clone incoming pdf, attach to each other *
  // ******************************************************************

  // Clone FUNC
  _funcClone = static_cast<RooAbsReal*>(real.cloneTree());
  _funcCloneSet = 0 ;

  // Attach FUNC to data set
  _funcObsSet = _funcClone->getObservables(indata) ;

  if (_funcClone->getAttribute("BinnedLikelihood")) {
    _funcClone->setAttribute("BinnedLikelihoodActive") ;
  }

  // Reattach FUNC to original parameters
  RooArgSet* origParams = (RooArgSet*) real.getParameters(indata) ;
  _funcClone->recursiveRedirectServers(*origParams) ;

  // Mark all projected dependents as such
  if (projDeps.getSize()>0) {
    RooArgSet *projDataDeps = (RooArgSet*) _funcObsSet->selectCommon(projDeps) ;
    projDataDeps->setAttribAll("projectedDependent") ;
    delete projDataDeps ;
  }

  // If PDF is a RooProdPdf (with possible constraint terms)
  // analyze pdf for actual parameters (i.e those in unconnected constraint terms should be
  // ignored as here so that the test statistic will not be recalculated if those
  // are changed
  RooProdPdf* pdfWithCons = dynamic_cast<RooProdPdf*>(_funcClone) ;
  if (pdfWithCons) {

    RooArgSet* connPars = pdfWithCons->getConnectedParameters(*indata.get()) ;
    // Add connected parameters as servers
    _paramSet.removeAll() ;
    _paramSet.add(*connPars) ;
    delete connPars ;

  } else {
    // Add parameters as servers
    _paramSet.add(*origParams) ;
  }


  delete origParams ;

  // Store normalization set
  _normSet = (RooArgSet*) indata.get()->snapshot(false) ;

  // Expand list of observables with any observables used in parameterized ranges.
  // This NEEDS to be a counting loop since we are inserting during the loop.
  for (std::size_t i = 0; i < _funcObsSet->size(); ++i) {
    auto realDepRLV = dynamic_cast<const RooAbsRealLValue*>((*_funcObsSet)[i]);
    if (realDepRLV && realDepRLV->isDerived()) {
      RooArgSet tmp2;
      realDepRLV->leafNodeServerList(&tmp2, 0, true);
      _funcObsSet->add(tmp2,true);
    }
  }



  // ******************************************************************
  // *** PART 2 *** Clone and adjust incoming data, attach to PDF     *
  // ******************************************************************

  // Check if the fit ranges of the dependents in the data and in the FUNC are consistent
  const RooArgSet* dataDepSet = indata.get() ;
  for (const auto arg : *_funcObsSet) {

    // Check that both dataset and function argument are of type RooRealVar
    RooRealVar* realReal = dynamic_cast<RooRealVar*>(arg) ;
    if (!realReal) continue ;
    RooRealVar* datReal = dynamic_cast<RooRealVar*>(dataDepSet->find(realReal->GetName())) ;
    if (!datReal) continue ;

    // Check that range of observables in pdf is equal or contained in range of observables in data

    if (!realReal->getBinning().lowBoundFunc() && realReal->getMin()<(datReal->getMin()-1e-6)) {
      coutE(InputArguments) << "RooAbsOptTestStatistic: ERROR minimum of FUNC observable " << arg->GetName()
                 << "(" << realReal->getMin() << ") is smaller than that of "
                 << arg->GetName() << " in the dataset (" << datReal->getMin() << ")" << endl ;
      RooErrorHandler::softAbort() ;
      return ;
    }

    if (!realReal->getBinning().highBoundFunc() && realReal->getMax()>(datReal->getMax()+1e-6)) {
      coutE(InputArguments) << "RooAbsOptTestStatistic: ERROR maximum of FUNC observable " << arg->GetName()
                 << " is larger than that of " << arg->GetName() << " in the dataset" << endl ;
      RooErrorHandler::softAbort() ;
      return ;
    }
  }

  // Copy data and strip entries lost by adjusted fit range, _dataClone ranges will be copied from realDepSet ranges
  if (rangeName && strlen(rangeName)) {
    _dataClone = indata.reduce(RooFit::SelectVars(*_funcObsSet),RooFit::CutRange(rangeName)) ;
    //     cout << "RooAbsOptTestStatistic: reducing dataset to fit in range named " << rangeName << " resulting dataset has " << _dataClone->sumEntries() << " events" << endl ;
  } else {
    _dataClone = (RooAbsData*) indata.Clone() ;
  }
  _ownData = true ;


  // ******************************************************************
  // *** PART 3 *** Make adjustments for fit ranges, if specified     *
  // ******************************************************************

  std::unique_ptr<RooArgSet> origObsSet( real.getObservables(indata) );
  RooArgSet* dataObsSet = (RooArgSet*) _dataClone->get() ;
  if (rangeName && strlen(rangeName)) {
    cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") constructing test statistic for sub-range named " << rangeName << endl ;

    bool observablesKnowRange = false;
    // Adjust FUNC normalization ranges to requested fitRange, store original ranges for RooAddPdf coefficient interpretation
    for (const auto arg : *_funcObsSet) {

      RooRealVar* realObs = dynamic_cast<RooRealVar*>(arg) ;
      if (realObs) {

        auto transferRangeAndBinning = [&](RooRealVar & toVar, const char* toName, const char* fromName) {
          toVar.setRange(toName, realObs->getMin(fromName),realObs->getMax(fromName));
          // If the realObs also has a binning with a name matching the
          // rangeName, it will be set as the default binning. If `fromName` is
          // a nullptr to signify taking the default binning from `realObs`,
          // don't check if it exists as there is always a default binning.
          if(!fromName || realObs->hasBinning(fromName)) {
            toVar.setBinning(realObs->getBinning(fromName), toName);
          }
        };

        observablesKnowRange |= realObs->hasRange(rangeName);

        // If no explicit range is given for RooAddPdf coefficients, create explicit named range equivalent to original observables range
        if (!(addCoefRangeName && strlen(addCoefRangeName))) {
          transferRangeAndBinning(*realObs, Form("NormalizationRangeFor%s",rangeName), nullptr);
        }

        // Adjust range of function observable to those of given named range
        transferRangeAndBinning(*realObs, nullptr, rangeName);

        // Adjust range of data observable to those of given named range
        RooRealVar* dataObs = (RooRealVar*) dataObsSet->find(realObs->GetName()) ;
        transferRangeAndBinning(*dataObs, nullptr, rangeName);

        // Keep track of list of fit ranges in string attribute fit range of original p.d.f.
        if (!_splitRange) {
          const std::string fitRangeName = std::string("fit_") + GetName();
          const char* origAttrib = real.getStringAttribute("fitrange") ;
          std::string newAttr = origAttrib ? origAttrib : "";

          if (newAttr.find(fitRangeName) == std::string::npos) {
            newAttr += (newAttr.empty() ? "" : ",") + fitRangeName;
          }
          real.setStringAttribute("fitrange", newAttr.c_str());
          RooRealVar* origObs = (RooRealVar*) origObsSet->find(arg->GetName()) ;
          if (origObs) {
            transferRangeAndBinning(*origObs, fitRangeName.c_str(), rangeName);
          }
        }
      }
    }

    if (!observablesKnowRange)
      coutW(Fitting) << "None of the fit observables seem to know the range '" << rangeName << "'. This means that the full range will be used." << std::endl;
  }


  // ******************************************************************
  // *** PART 3.2 *** Binned fits                                     *
  // ******************************************************************

  // If dataset is binned, activate caching of bins that are invalid because the're outside the
  // updated range definition (WVE need to add virtual interface here)
  RooDataHist* tmph = dynamic_cast<RooDataHist*>(_dataClone) ;
  if (tmph) {
    tmph->cacheValidEntries() ;
  }

  setUpBinSampling();


  // Fix RooAddPdf coefficients to original normalization range
  if (rangeName && strlen(rangeName)) {

    // WVE Remove projected dependents from normalization
    _funcClone->fixAddCoefNormalization(*_dataClone->get(),false) ;

    if (addCoefRangeName && strlen(addCoefRangeName)) {
      cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName()
                 << ") fixing interpretation of coefficients of any RooAddPdf component to range " << addCoefRangeName << endl ;
      _funcClone->fixAddCoefRange(addCoefRangeName,false) ;
    } else {
      cxcoutI(Fitting) << "RooAbsOptTestStatistic::ctor(" << GetName()
              << ") fixing interpretation of coefficients of any RooAddPdf to full domain of observables " << endl ;
      _funcClone->fixAddCoefRange(Form("NormalizationRangeFor%s",rangeName),false) ;
    }
  }


  // This is deferred from part 2 - but must happen after part 3 - otherwise invalid bins cannot be properly marked in cacheValidEntries
  _dataClone->attachBuffers(*_funcObsSet) ;
  setEventCount(_dataClone->numEntries()) ;




  // *********************************************************************
  // *** PART 4 *** Adjust normalization range for projected observables *
  // *********************************************************************

  // Remove projected dependents from normalization set
  if (projDeps.getSize()>0) {

    _projDeps = (RooArgSet*) projDeps.snapshot(false) ;

    //RooArgSet* tobedel = (RooArgSet*) _normSet->selectCommon(*_projDeps) ;
    _normSet->remove(*_projDeps,true,true) ;

    // Mark all projected dependents as such
    RooArgSet *projDataDeps = (RooArgSet*) _funcObsSet->selectCommon(*_projDeps) ;
    projDataDeps->setAttribAll("projectedDependent") ;
    delete projDataDeps ;
  }


  coutI(Optimization) << "RooAbsOptTestStatistic::ctor(" << GetName() << ") optimizing internal clone of p.d.f for likelihood evaluation."
      << "Lazy evaluation and associated change tracking will disabled for all nodes that depend on observables" << endl ;


  // *********************************************************************
  // *** PART 4 *** Finalization and activation of optimization          *
  // *********************************************************************

  // Redirect pointers of base class to clone
  _func = _funcClone ;
  _data = _dataClone ;

  _funcClone->getVal(_normSet) ;

  optimizeCaching() ;

  // It would be unusual if the global observables are used in the likelihood
  // outside of the constraint terms, but if they are we have to be consistent
  // and also redirect them to the snapshots in the dataset if appropriate.
  if(_takeGlobalObservablesFromData && _data->getGlobalObservables()) {
    recursiveRedirectServers(*_data->getGlobalObservables()) ;
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsOptTestStatistic::~RooAbsOptTestStatistic()
{
  if (operMode()==Slave) {
    delete _funcClone ;
    delete _funcObsSet ;
    if (_projDeps) {
      delete _projDeps ;
    }
    if (_ownData) {
      delete _dataClone ;
    }
  }
  delete _normSet ;
}



////////////////////////////////////////////////////////////////////////////////
/// Method to combined test statistic results calculated into partitions into
/// the global result. This default implementation adds the partition return
/// values

double RooAbsOptTestStatistic::combinedValue(RooAbsReal** array, Int_t n) const
{
  // Default implementation returns sum of components
  double sum(0), carry(0);
  for (Int_t i = 0; i < n; ++i) {
    double y = array[i]->getValV();
    carry += reinterpret_cast<RooAbsOptTestStatistic*>(array[i])->getCarry();
    y -= carry;
    const double t = sum + y;
    carry = (t - sum) - y;
    sum = t;
  }
  _evalCarry = carry;
  return sum ;
}



////////////////////////////////////////////////////////////////////////////////
/// Catch server redirect calls and forward to internal clone of function

bool RooAbsOptTestStatistic::redirectServersHook(const RooAbsCollection& newServerList, bool mustReplaceAll, bool nameChange, bool isRecursive)
{
  RooAbsTestStatistic::redirectServersHook(newServerList,mustReplaceAll,nameChange,isRecursive) ;
  if (operMode()!=Slave) return false ;
  bool ret = _funcClone->recursiveRedirectServers(newServerList,false,nameChange) ;
  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Catch print hook function and forward to function clone

void RooAbsOptTestStatistic::printCompactTreeHook(ostream& os, const char* indent)
{
  RooAbsTestStatistic::printCompactTreeHook(os,indent) ;
  if (operMode()!=Slave) return ;
  TString indent2(indent) ;
  indent2 += "opt >>" ;
  _funcClone->printCompactTree(os,indent2.Data()) ;
  os << indent2 << " dataset clone = " << _dataClone << " first obs = " << _dataClone->get()->first() << endl ;
}



////////////////////////////////////////////////////////////////////////////////
/// Driver function to propagate constant term optimizations in test statistic.
/// If code Activate is sent, constant term optimization will be executed.
/// If code Deactivate is sent, any existing constant term optimizations will
/// be abandoned. If codes ConfigChange or ValueChange are sent, any existing
/// constant term optimizations will be redone.

void RooAbsOptTestStatistic::constOptimizeTestStatistic(ConstOpCode opcode, bool doAlsoTrackingOpt)
{
  //   cout << "ROATS::constOpt(" << GetName() << ") funcClone structure dump BEFORE const-opt" << endl ;
  //   _funcClone->Print("t") ;

  RooAbsTestStatistic::constOptimizeTestStatistic(opcode,doAlsoTrackingOpt);
  if (operMode()!=Slave) return ;

  if (_dataClone->hasFilledCache() && _dataClone->store()->cacheOwner()!=this) {
    if (opcode==Activate) {
      cxcoutW(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName()
             << ") dataset cache is owned by another object, no constant term optimization can be applied" << endl ;
    }
    return ;
  }

  if (!allowFunctionCache()) {
    if (opcode==Activate) {
      cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName()
             << ") function caching prohibited by test statistic, no constant term optimization is applied" << endl ;
    }
    return ;
  }

  if (_dataClone->hasFilledCache() && opcode==Activate) {
    opcode=ValueChange ;
  }

  switch(opcode) {
  case Activate:
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName()
           << ") optimizing evaluation of test statistic by finding all nodes in p.d.f that depend exclusively"
           << " on observables and constant parameters and precalculating their values" << endl ;
    optimizeConstantTerms(true,doAlsoTrackingOpt) ;
    break ;

  case DeActivate:
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName()
           << ") deactivating optimization of constant terms in test statistic" << endl ;
    optimizeConstantTerms(false) ;
    break ;

  case ConfigChange:
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName()
           << ") one ore more parameter were changed from constant to floating or vice versa, "
           << "re-evaluating constant term optimization" << endl ;
    optimizeConstantTerms(false) ;
    optimizeConstantTerms(true,doAlsoTrackingOpt) ;
    break ;

  case ValueChange:
    cxcoutI(Optimization) << "RooAbsOptTestStatistic::constOptimize(" << GetName()
           << ") the value of one ore more constant parameter were changed re-evaluating constant term optimization" << endl ;
    // Request a forcible cache update of all cached nodes
    _dataClone->store()->forceCacheUpdate() ;

    break ;
  }

//   cout << "ROATS::constOpt(" << GetName() << ") funcClone structure dump AFTER const-opt" << endl ;
//   _funcClone->Print("t") ;
}



////////////////////////////////////////////////////////////////////////////////
/// This method changes the value caching logic for all nodes that depends on any of the observables
/// as defined by the given dataset. When evaluating a test statistic constructed from the RooAbsReal
/// with a dataset the observables are guaranteed to change with every call, thus there is no point
/// in tracking these changes which result in a net overhead. Thus for observable-dependent nodes,
/// the evaluation mechanism is changed from being dependent on a 'valueDirty' flag to guaranteed evaluation.
/// On the dataset side, the observables objects are modified to no longer send valueDirty messages
/// to their client

void RooAbsOptTestStatistic::optimizeCaching()
{
//   cout << "RooAbsOptTestStatistic::optimizeCaching(" << GetName() << "," << this << ")" << endl ;

  // Trigger create of all object caches now in nodes that have deferred object creation
  // so that cache contents can be processed immediately
  _funcClone->getVal(_normSet) ;

  // Set value caching mode for all nodes that depend on any of the observables to ADirty
  _funcClone->optimizeCacheMode(*_funcObsSet) ;

  // Disable propagation of dirty state flags for observables
  _dataClone->setDirtyProp(false) ;

  // Disable reading of observables that are not used
  _dataClone->optimizeReadingWithCaching(*_funcClone, RooArgSet(),requiredExtraObservables()) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Driver function to activate global constant term optimization.
/// If activated, constant terms are found and cached with the dataset.
/// The operation mode of cached nodes is set to AClean meaning that
/// their getVal() call will never result in an evaluate call.
/// Finally the branches in the dataset that correspond to observables
/// that are exclusively used in constant terms are disabled as
/// they serve no more purpose

void RooAbsOptTestStatistic::optimizeConstantTerms(bool activate, bool applyTrackingOpt)
{
  if(activate) {

    if (_optimized) {
      return ;
    }

    // Trigger create of all object caches now in nodes that have deferred object creation
    // so that cache contents can be processed immediately
    _funcClone->getVal(_normSet) ;


    //  WVE - Patch to allow customization of optimization level per component pdf
    if (_funcClone->getAttribute("NoOptimizeLevel1")) {
      coutI(Minimization) << " Optimization customization: Level-1 constant-term optimization prohibited by attribute NoOptimizeLevel1 set on top-level pdf  "
                          << _funcClone->IsA()->GetName() << "::" << _funcClone->GetName() << endl ;
      return ;
    }
    if (_funcClone->getAttribute("NoOptimizeLevel2")) {
      coutI(Minimization) << " Optimization customization: Level-2 constant-term optimization prohibited by attribute NoOptimizeLevel2 set on top-level pdf  "
                          << _funcClone->IsA()->GetName() << "::" << _funcClone->GetName() << endl ;
      applyTrackingOpt=false ;
    }

    // Apply tracking optimization here. Default strategy is to track components
    // of RooAddPdfs and RooRealSumPdfs. If these components are a RooProdPdf
    // or a RooProduct respectively, track the components of these products instead
    // of the product term
    RooArgSet trackNodes ;


    // Add safety check here - applyTrackingOpt will only be applied if present
    // dataset is constructed in terms of a RooVectorDataStore
    if (applyTrackingOpt) {
      if (!dynamic_cast<RooVectorDataStore*>(_dataClone->store())) {
        coutW(Optimization) << "RooAbsOptTestStatistic::optimizeConstantTerms(" << GetName()
                     << ") WARNING Cache-and-track optimization (Optimize level 2) is only available for datasets"
                     << " implement in terms of RooVectorDataStore - ignoring this option for current dataset" << endl ;
        applyTrackingOpt = false ;
      }
    }

    if (applyTrackingOpt) {
      RooArgSet branches ;
      _funcClone->branchNodeServerList(&branches) ;
      for (auto arg : branches) {
        arg->setCacheAndTrackHints(trackNodes);
      }
      // Do not set CacheAndTrack on constant expressions
      RooArgSet* constNodes = (RooArgSet*) trackNodes.selectByAttrib("Constant",true) ;
      trackNodes.remove(*constNodes) ;
      delete constNodes ;

      // Set CacheAndTrack flag on all remaining nodes
      trackNodes.setAttribAll("CacheAndTrack",true) ;
    }

    // Find all nodes that depend exclusively on constant parameters
    _cachedNodes.removeAll() ;

    _funcClone->findConstantNodes(*_dataClone->get(),_cachedNodes) ;

    // Cache constant nodes with dataset - also cache entries corresponding to zero-weights in data when using BinnedLikelihood
    _dataClone->cacheArgs(this,_cachedNodes,_normSet,!_funcClone->getAttribute("BinnedLikelihood")) ;

    // Put all cached nodes in AClean value caching mode so that their evaluate() is never called
    for (auto cacheArg : _cachedNodes) {
      cacheArg->setOperMode(RooAbsArg::AClean) ;
    }

    RooArgSet* constNodes = (RooArgSet*) _cachedNodes.selectByAttrib("ConstantExpressionCached",true) ;
    RooArgSet actualTrackNodes(_cachedNodes) ;
    actualTrackNodes.remove(*constNodes) ;
    if (constNodes->getSize()>0) {
      if (constNodes->getSize()<20) {
        coutI(Minimization) << " The following expressions have been identified as constant and will be precalculated and cached: " << *constNodes << endl ;
      } else {
        coutI(Minimization) << " A total of " << constNodes->getSize() << " expressions have been identified as constant and will be precalculated and cached." << endl ;
      }
    }
    if (actualTrackNodes.getSize()>0) {
      if (actualTrackNodes.getSize()<20) {
        coutI(Minimization) << " The following expressions will be evaluated in cache-and-track mode: " << actualTrackNodes << endl ;
      } else {
        coutI(Minimization) << " A total of " << constNodes->getSize() << " expressions will be evaluated in cache-and-track-mode." << endl ;
      }
    }
    delete constNodes ;

    // Disable reading of observables that are no longer used
    _dataClone->optimizeReadingWithCaching(*_funcClone, _cachedNodes,requiredExtraObservables()) ;

    _optimized = true ;

  } else {

    // Delete the cache
    _dataClone->resetCache() ;

    // Reactivate all tree branches
    _dataClone->setArgStatus(*_dataClone->get(),true) ;

    // Reset all nodes to ADirty
    optimizeCaching() ;

    // Disable propagation of dirty state flags for observables
    _dataClone->setDirtyProp(false) ;

    _cachedNodes.removeAll() ;


    _optimized = false ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Change dataset that is used to given one. If cloneData is true, a clone of
/// in the input dataset is made.  If the test statistic was constructed with
/// a range specification on the data, the cloneData argument is ignored and
/// the data is always cloned.
bool RooAbsOptTestStatistic::setDataSlave(RooAbsData& indata, bool cloneData, bool ownNewData)
{

  if (operMode()==SimMaster) {
    //cout << "ROATS::setDataSlave() ERROR this is SimMaster _funcClone = " << _funcClone << endl ;
    return false ;
  }

  //cout << "ROATS::setDataSlave() new dataset size = " << indata.numEntries() << endl ;
  //indata.Print("v") ;


  // If the current dataset is owned, transfer the ownership to unique pointer
  // that will get out of scope at the end of this function. We can't delete it
  // right now, because there might be global observables in the model that
  // first need to be redirected to the new dataset with a later call to
  // RooAbsArg::recursiveRedirectServers.
  std::unique_ptr<RooAbsData> oldOwnedData;
  if (_ownData) {
    oldOwnedData.reset(_dataClone);
    _dataClone = nullptr ;
  }

  if (!cloneData && _rangeName.size()>0) {
    coutW(InputArguments) << "RooAbsOptTestStatistic::setData(" << GetName() << ") WARNING: test statistic was constructed with range selection on data, "
          << "ignoring request to _not_ clone the input dataset" << endl ;
    cloneData = true ;
  }

  if (cloneData) {
    // Cloning input dataset
    if (_rangeName.size()==0) {
      _dataClone = (RooAbsData*) indata.reduce(*indata.get()) ;
    } else {
      _dataClone = ((RooAbsData&)indata).reduce(RooFit::SelectVars(*indata.get()),RooFit::CutRange(_rangeName.c_str())) ;
    }
    _ownData = true ;

  } else {

    // Taking input dataset
    _dataClone = &indata ;
    _ownData = ownNewData ;

  }

  // Attach function clone to dataset
  _dataClone->attachBuffers(*_funcObsSet) ;
  _dataClone->setDirtyProp(false) ;
  _data = _dataClone ;

  // ReCache constant nodes with dataset
  if (_cachedNodes.getSize()>0) {
    _dataClone->cacheArgs(this,_cachedNodes,_normSet) ;
  }

  // Adjust internal event count
  setEventCount(indata.numEntries()) ;

  setValueDirty() ;

  // It would be unusual if the global observables are used in the likelihood
  // outside of the constraint terms, but if they are we have to be consistent
  // and also redirect them to the snapshots in the dataset if appropriate.
  if(_takeGlobalObservablesFromData && _data->getGlobalObservables()) {
    recursiveRedirectServers(*_data->getGlobalObservables()) ;
  }

  return true ;
}




////////////////////////////////////////////////////////////////////////////////

RooAbsData& RooAbsOptTestStatistic::data()
{
  if (_sealed) {
    bool notice = (sealNotice() && strlen(sealNotice())) ;
    coutW(ObjectHandling) << "RooAbsOptTestStatistic::data(" << GetName()
           << ") WARNING: object sealed by creator - access to data is not permitted: "
           << (notice?sealNotice():"<no user notice>") << endl ;
    static RooDataSet dummy ("dummy","dummy",RooArgSet()) ;
    return dummy ;
  }
  return *_dataClone ;
}


////////////////////////////////////////////////////////////////////////////////

const RooAbsData& RooAbsOptTestStatistic::data() const
{
  if (_sealed) {
    bool notice = (sealNotice() && strlen(sealNotice())) ;
    coutW(ObjectHandling) << "RooAbsOptTestStatistic::data(" << GetName()
           << ") WARNING: object sealed by creator - access to data is not permitted: "
           << (notice?sealNotice():"<no user notice>") << endl ;
    static RooDataSet dummy ("dummy","dummy",RooArgSet()) ;
    return dummy ;
  }
  return *_dataClone ;
}


////////////////////////////////////////////////////////////////////////////////
/// Inspect PDF to find out if we are doing a binned fit to a 1-dimensional unbinned PDF.
/// If this is the case, enable finer sampling of bins by wrapping PDF into a RooBinSamplingPdf.
/// The member _integrateBinsPrecision decides how we act:
/// - < 0: Don't do anything.
/// - = 0: Only enable feature if fitting unbinned PDF to RooDataHist.
/// - > 0: Enable as requested.
void RooAbsOptTestStatistic::setUpBinSampling() {

  auto& pdf = static_cast<RooAbsPdf&>(*_funcClone);
  if (auto newPdf = RooBinSamplingPdf::create(pdf, *_dataClone, _integrateBinsPrecision)) {
    newPdf->addOwnedComponents(*_funcClone);
    _funcClone = newPdf.release();
  }

}


/// Returns a suffix string that is unique for RooAbsOptTestStatistic
/// instances that don't share the same cloned input data object.
const char* RooAbsOptTestStatistic::cacheUniqueSuffix() const {
   return Form("_%lx", _dataClone->uniqueId().value()) ;
}
