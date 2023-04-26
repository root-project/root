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
 * std::listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////

/** \class RooAbsReal

   RooAbsReal is the common abstract base class for objects that represent a
   real value and implements functionality common to all real-valued objects
   such as the ability to plot them, to construct integrals of them, the
   ability to advertise (partial) analytical integrals etc.

   Implementation of RooAbsReal may be derived, thus no interface
   is provided to modify the contents.

   \ingroup Roofitcore
*/


#include "RooFitDriver.h"
#include "RooMsgService.h"

#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooBinning.h"
#include "RooPlot.h"
#include "RooCurve.h"
#include "RooHist.h"
#include "RooRealVar.h"
#include "RooArgProxy.h"
#include "RooFormulaVar.h"
#include "RooRealBinding.h"
#include "RooRealIntegral.h"
#include "RooAbsCategoryLValue.h"
#include "RooCustomizer.h"
#include "RooAbsData.h"
#include "RooScaledFunc.h"
#include "RooAddPdf.h"
#include "RooCmdConfig.h"
#include "RooCategory.h"
#include "RooNumIntConfig.h"
#include "RooAddition.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooDataWeightedAverage.h"
#include "RooNumRunningInt.h"
#include "RooGlobalFunc.h"
#include "RooParamBinning.h"
#include "RooProfileLL.h"
#include "RooFunctor.h"
#include "RooDerivative.h"
#include "RooGenFunction.h"
#include "RooMultiGenFunction.h"
#include "RooXYChi2Var.h"
#include "RooMinimizer.h"
#include "RooChi2Var.h"
#include "RooFitResult.h"
#include "RooMoment.h"
#include "RooFirstMoment.h"
#include "RooSecondMoment.h"
#include "RooBrentRootFinder.h"
#include "RooVectorDataStore.h"
#include "RooCachedReal.h"
#include "RooHelpers.h"
#include "RunContext.h"
#include "TreeReadBuffer.h"
#include "ValueChecking.h"

#include "ROOT/StringUtils.hxx"
#include "Compression.h"
#include "Math/IFunction.h"
#include "TMath.h"
#include "TObjString.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TAttLine.h"
#include "TF1.h"
#include "TF2.h"
#include "TF3.h"
#include "TMatrixD.h"
#include "TVector.h"
#include "strlcpy.h"
#ifndef NDEBUG
#include <TSystem.h> // To print stack traces when caching errors are detected
#endif

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <sys/types.h>

ClassImp(RooAbsReal)

bool RooAbsReal::_globalSelectComp = false;
bool RooAbsReal::_hideOffset = true ;

void RooAbsReal::setHideOffset(bool flag) { _hideOffset = flag ; }
bool RooAbsReal::hideOffset() { return _hideOffset ; }

RooAbsReal::ErrorLoggingMode RooAbsReal::_evalErrorMode = RooAbsReal::PrintErrors ;
Int_t RooAbsReal::_evalErrorCount = 0 ;
std::map<const RooAbsArg*,std::pair<std::string,std::list<RooAbsReal::EvalError> > > RooAbsReal::_evalErrorList ;


////////////////////////////////////////////////////////////////////////////////
/// coverity[UNINIT_CTOR]
/// Default constructor

RooAbsReal::RooAbsReal() : _specIntegratorConfig(0), _selectComp(true), _lastNSet(0)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with unit label

RooAbsReal::RooAbsReal(const char *name, const char *title, const char *unit) :
  RooAbsArg(name,title), _plotMin(0), _plotMax(0), _plotBins(100),
  _value(0),  _unit(unit), _forceNumInt(false), _specIntegratorConfig(0), _selectComp(true), _lastNSet(0)
{
  setValueDirty() ;
  setShapeDirty() ;

}



////////////////////////////////////////////////////////////////////////////////
/// Constructor with plot range and unit label

RooAbsReal::RooAbsReal(const char *name, const char *title, double inMinVal,
             double inMaxVal, const char *unit) :
  RooAbsArg(name,title), _plotMin(inMinVal), _plotMax(inMaxVal), _plotBins(100),
  _value(0), _unit(unit), _forceNumInt(false), _specIntegratorConfig(0), _selectComp(true), _lastNSet(0)
{
  setValueDirty() ;
  setShapeDirty() ;

}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor
RooAbsReal::RooAbsReal(const RooAbsReal& other, const char* name) :
  RooAbsArg(other,name), _plotMin(other._plotMin), _plotMax(other._plotMax),
  _plotBins(other._plotBins), _value(other._value), _unit(other._unit), _label(other._label),
  _forceNumInt(other._forceNumInt), _selectComp(other._selectComp), _lastNSet(0)
{
  if (other._specIntegratorConfig) {
    _specIntegratorConfig = new RooNumIntConfig(*other._specIntegratorConfig) ;
  } else {
    _specIntegratorConfig = 0 ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsReal::~RooAbsReal()
{
  if (_specIntegratorConfig) delete _specIntegratorConfig ;
}



////////////////////////////////////////////////////////////////////////////////
/// Equality operator comparing to a double

bool RooAbsReal::operator==(double value) const
{
  return (getVal()==value) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Equality operator when comparing to another RooAbsArg.
/// Only functional when the other arg is a RooAbsReal

bool RooAbsReal::operator==(const RooAbsArg& other) const
{
  const RooAbsReal* otherReal = dynamic_cast<const RooAbsReal*>(&other) ;
  return otherReal ? operator==(otherReal->getVal()) : false ;
}


////////////////////////////////////////////////////////////////////////////////

bool RooAbsReal::isIdentical(const RooAbsArg& other, bool assumeSameType) const
{
  if (!assumeSameType) {
    const RooAbsReal* otherReal = dynamic_cast<const RooAbsReal*>(&other) ;
    return otherReal ? operator==(otherReal->getVal()) : false ;
  } else {
    return getVal() == static_cast<const RooAbsReal&>(other).getVal();
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Return this variable's title std::string. If appendUnit is true and
/// this variable has units, also append a std::string " (<unit>)".

TString RooAbsReal::getTitle(bool appendUnit) const
{
  TString title(GetTitle());
  if(appendUnit && 0 != strlen(getUnit())) {
    title.Append(" (");
    title.Append(getUnit());
    title.Append(")");
  }
  return title;
}



////////////////////////////////////////////////////////////////////////////////
/// Return value of object. If the cache is clean, return the
/// cached value, otherwise recalculate on the fly and refill
/// the cache

double RooAbsReal::getValV(const RooArgSet* nset) const
{
  if (nset && nset!=_lastNSet) {
    ((RooAbsReal*) this)->setProxyNormSet(nset) ;
    _lastNSet = (RooArgSet*) nset ;
  }

  if (isValueDirtyAndClear()) {
    _value = traceEval(nullptr) ;
    //     clearValueDirty() ;
  }
  //   cout << "RooAbsReal::getValV(" << GetName() << ") writing _value = " << _value << std::endl ;

  double ret(_value) ;
  if (hideOffset()) ret += offset() ;

  return ret ;
}


////////////////////////////////////////////////////////////////////////////////
/// Compute batch of values for input data stored in `evalData`.
///
/// This is a faster, multi-value version of getVal(). It calls evaluateSpan() to trigger computations, and
/// finalises those (e.g. error checking or automatic normalisation) before returning a span with the results.
/// This span will also be stored in `evalData`, so subsquent calls of getValues() will return immediately.
///
/// If `evalData` is empty, a single value will be returned, which is the result of evaluating the current value
/// of each object that's serving values to us. If `evalData` contains a batch of values for one or more of the
/// objects serving values to us, a batch of values for each entry stored in `evalData` is returned. To fill a
/// RunContext with values from a dataset, use RooAbsData::getBatches().
///
/// \param[in] evalData  Object holding spans of input data. The results are also stored here.
/// \param[in] normSet   Use these variables for normalisation (relevant for PDFs), and pass this normalisation
/// on to object serving values to us.
/// \return RooSpan pointing to the computation results. The memory this span points to is owned by `evalData`.
RooSpan<const double> RooAbsReal::getValues(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet) const {
  auto item = evalData.spans.find(this);
  if (item != evalData.spans.end()) {
    return item->second;
  }

  normSet = normSet ? normSet : _lastNSet;

  std::map<RooFit::Detail::DataKey, RooSpan<const double>> dataSpans;
  for (auto const &evalDataItem : evalData.spans) {
    dataSpans[evalDataItem.first] = evalDataItem.second;
  }

  std::unique_ptr<RooAbsReal> clone = RooFit::Detail::compileForNormSet<RooAbsReal>(*this, normSet ? *normSet : RooArgSet{});
  ROOT::Experimental::RooFitDriver driver(*clone);
  driver.setData(dataSpans);
  auto& results = evalData.ownedMemory[this];
  results = driver.getValues(); // the compiler should use the move assignment here
  evalData.spans[this] = results;
  return results;
}

////////////////////////////////////////////////////////////////////////////////

std::vector<double> RooAbsReal::getValues(RooAbsData const& data) const {
  std::unique_ptr<RooAbsReal> clone = RooFit::Detail::compileForNormSet<RooAbsReal>(*this, *data.get());
  ROOT::Experimental::RooFitDriver driver(*clone, RooFit::BatchModeOption::Cpu);
  driver.setData(data, "");
  return driver.getValues();
}

////////////////////////////////////////////////////////////////////////////////

Int_t RooAbsReal::numEvalErrorItems()
{
  return _evalErrorList.size() ;
}


////////////////////////////////////////////////////////////////////////////////

RooAbsReal::EvalErrorIter RooAbsReal::evalErrorIter()
{
  return _evalErrorList.begin() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate current value of object, with error tracing wrapper

double RooAbsReal::traceEval(const RooArgSet* /*nset*/) const
{
  double value = evaluate() ;

  if (TMath::IsNaN(value)) {
    logEvalError("function value is NAN") ;
  }

  //cxcoutD(Tracing) << "RooAbsReal::getValF(" << GetName() << ") operMode = " << _operMode << " recalculated, new value = " << value << std::endl ;

  //Standard tracing code goes here
  if (!isValidReal(value)) {
    coutW(Tracing) << "RooAbsReal::traceEval(" << GetName()
         << "): validation failed: " << value << std::endl ;
  }

  //Call optional subclass tracing code
  //   traceEvalHook(value) ;

  return value ;
}



////////////////////////////////////////////////////////////////////////////////
/// Variant of getAnalyticalIntegral that is also passed the normalization set
/// that should be applied to the integrand of which the integral is requested.
/// For certain operator p.d.f it is useful to overload this function rather
/// than analyticalIntegralWN() as the additional normalization information
/// may be useful in determining a more efficient decomposition of the
/// requested integral.

Int_t RooAbsReal::getAnalyticalIntegralWN(RooArgSet& allDeps, RooArgSet& analDeps,
                 const RooArgSet* /*normSet*/, const char* rangeName) const
{
  return _forceNumInt ? 0 : getAnalyticalIntegral(allDeps,analDeps,rangeName) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface function getAnalyticalIntergral advertises the
/// analytical integrals that are supported. 'integSet'
/// is the set of dependents for which integration is requested. The
/// function should copy the subset of dependents it can analytically
/// integrate to anaIntSet and return a unique identification code for
/// this integration configuration.  If no integration can be
/// performed, zero should be returned.

Int_t RooAbsReal::getAnalyticalIntegral(RooArgSet& /*integSet*/, RooArgSet& /*anaIntSet*/, const char* /*rangeName*/) const
{
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implements the actual analytical integral(s) advertised by
/// getAnalyticalIntegral.  This functions will only be called with
/// codes returned by getAnalyticalIntegral, except code zero.

double RooAbsReal::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
//   cout << "RooAbsReal::analyticalIntegralWN(" << GetName() << ") code = " << code << " normSet = " << (normSet?*normSet:RooArgSet()) << std::endl ;
  if (code==0) return getVal(normSet) ;
  return analyticalIntegral(code,rangeName) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implements the actual analytical integral(s) advertised by
/// getAnalyticalIntegral.  This functions will only be called with
/// codes returned by getAnalyticalIntegral, except code zero.

double RooAbsReal::analyticalIntegral(Int_t code, const char* /*rangeName*/) const
{
  // By default no analytical integrals are implemented
  coutF(Eval)  << "RooAbsReal::analyticalIntegral(" << GetName() << ") code " << code << " not implemented" << std::endl ;
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Get the label associated with the variable

const char *RooAbsReal::getPlotLabel() const
{
  return _label.IsNull() ? fName.Data() : _label.Data();
}



////////////////////////////////////////////////////////////////////////////////
/// Set the label associated with this variable

void RooAbsReal::setPlotLabel(const char *label)
{
  _label= label;
}



////////////////////////////////////////////////////////////////////////////////
///Read object contents from stream (dummy for now)

bool RooAbsReal::readFromStream(std::istream& /*is*/, bool /*compact*/, bool /*verbose*/)
{
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
///Write object contents to stream (dummy for now)

void RooAbsReal::writeToStream(std::ostream& /*os*/, bool /*compact*/) const
{
}



////////////////////////////////////////////////////////////////////////////////
/// Print object value

void RooAbsReal::printValue(std::ostream& os) const
{
  os << getVal() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Structure printing

void RooAbsReal::printMultiline(std::ostream& os, Int_t contents, bool verbose, TString indent) const
{
  RooAbsArg::printMultiline(os,contents,verbose,indent) ;
  os << indent << "--- RooAbsReal ---" << std::endl;
  TString unit(_unit);
  if(!unit.IsNull()) unit.Prepend(' ');
  //os << indent << "  Value = " << getVal() << unit << std::endl;
  os << std::endl << indent << "  Plot label is \"" << getPlotLabel() << "\"" << "\n";
}


////////////////////////////////////////////////////////////////////////////////
/// Create a RooProfileLL object that eliminates all nuisance parameters in the
/// present function. The nuisance parameters are defined as all parameters
/// of the function except the stated paramsOfInterest

RooAbsReal* RooAbsReal::createProfile(const RooArgSet& paramsOfInterest)
{
  // Construct name of profile object
  auto name = std::string(GetName()) + "_Profile[";
  bool first = true;
  for (auto const& arg : paramsOfInterest) {
    if (first) {
      first = false ;
    } else {
      name.append(",") ;
    }
    name.append(arg->GetName()) ;
  }
  name.append("]") ;

  // Create and return profile object
  return new RooProfileLL(name.c_str(),(std::string("Profile of ") + GetTitle()).c_str(),*this,paramsOfInterest) ;
}






////////////////////////////////////////////////////////////////////////////////
/// Create an object that represents the integral of the function over one or more observables std::listed in `iset`.
/// The actual integration calculation is only performed when the returned object is evaluated. The name
/// of the integral object is automatically constructed from the name of the input function, the variables
/// it integrates and the range integrates over.
///
/// \note The integral over a PDF is usually not normalised (*i.e.*, it is usually not
/// 1 when integrating the PDF over the full range). In fact, this integral is used *to compute*
/// the normalisation of each PDF. See the rf110 tutorial at https://root.cern.ch/doc/master/group__tutorial__roofit.html
/// for details on PDF normalisation.
///
/// The following named arguments are accepted
/// |  | Effect on integral creation
/// |--|-------------------------------
/// | `NormSet(const RooArgSet&)`            | Specify normalization set, mostly useful when working with PDFs
/// | `NumIntConfig(const RooNumIntConfig&)` | Use given configuration for any numeric integration, if necessary
/// | `Range(const char* name)`              | Integrate only over given range. Multiple ranges may be specified by passing multiple Range() arguments

RooAbsReal* RooAbsReal::createIntegral(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2,
                   const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5,
                   const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const
{


  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::createIntegral(%s)",GetName())) ;
  pc.defineString("rangeName","RangeWithName",0,"",true) ;
  pc.defineSet("normSet","NormSet",0,0) ;
  pc.defineObject("numIntConfig","NumIntConfig",0,0) ;

  // Process & check varargs
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(true)) {
    return 0 ;
  }

  // Extract values from named arguments
  const char* rangeName = pc.getString("rangeName",0,true) ;
  const RooArgSet* nset = pc.getSet("normSet",0);
  const RooNumIntConfig* cfg = static_cast<const RooNumIntConfig*>(pc.getObject("numIntConfig",0)) ;

  return createIntegral(iset,nset,cfg,rangeName) ;
}





////////////////////////////////////////////////////////////////////////////////
/// Create an object that represents the integral of the function over one or more observables std::listed in iset.
/// The actual integration calculation is only performed when the return object is evaluated. The name
/// of the integral object is automatically constructed from the name of the input function, the variables
/// it integrates and the range integrates over. If nset is specified the integrand is request
/// to be normalized over nset (only meaningful when the integrand is a pdf). If rangename is specified
/// the integral is performed over the named range, otherwise it is performed over the domain of each
/// integrated observable. If cfg is specified it will be used to configure any numeric integration
/// aspect of the integral. It will not force the integral to be performed numerically, which is
/// decided automatically by RooRealIntegral.

RooAbsReal* RooAbsReal::createIntegral(const RooArgSet& iset, const RooArgSet* nset,
                   const RooNumIntConfig* cfg, const char* rangeName) const
{
  if (!rangeName || strchr(rangeName,',')==0) {
    // Simple case: integral over full range or single limited range
    return createIntObj(iset,nset,cfg,rangeName) ;
  }

  // Integral over multiple ranges
  RooArgSet components ;

  std::vector<std::string> tokens = ROOT::Split(rangeName, ",");

  if(RooHelpers::checkIfRangesOverlap(iset, tokens)) {
    std::stringstream errMsg;
    errMsg << GetName() << " : integrating with respect to the variables " << iset << " on the ranges  \"" << rangeName
           << "\" is not possible because the ranges are overlapping";
    const std::string errMsgString = errMsg.str();
    coutE(Integration) << errMsgString << std::endl;
    throw std::invalid_argument(errMsgString);
  }

  for (const std::string& token : tokens) {
    RooAbsReal* compIntegral = createIntObj(iset,nset,cfg, token.c_str());
    components.add(*compIntegral);
  }

  TString title(GetTitle()) ;
  title.Prepend("Integral of ") ;
  TString fullName(GetName()) ;
  fullName.Append(integralNameSuffix(iset,nset,rangeName)) ;

  return new RooAddition(fullName.Data(),title.Data(),components,true) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Internal utility function for createIntegral() that creates the actual integral object.
RooAbsReal* RooAbsReal::createIntObj(const RooArgSet& iset2, const RooArgSet* nset2,
                 const RooNumIntConfig* cfg, const char* rangeName) const
{
  // Make internal use copies of iset and nset
  RooArgSet iset(iset2) ;
  const RooArgSet* nset = nset2 ;


  // Initialize local variables perparing for recursive loop
  bool error = false ;
  const RooAbsReal* integrand = this ;
  RooAbsReal* integral = 0 ;

  // Handle trivial case of no integration here explicitly
  if (iset.empty()) {

    TString title(GetTitle()) ;
    title.Prepend("Integral of ") ;

    TString name(GetName()) ;
    name.Append(integralNameSuffix(iset,nset,rangeName)) ;

    return new RooRealIntegral(name,title,*this,iset,nset,cfg,rangeName) ;
  }

  // Process integration over remaining integration variables
  while(iset.getSize()>0) {


    // Find largest set of observables that can be integrated in one go
    RooArgSet innerSet ;
    findInnerMostIntegration(iset,innerSet,rangeName) ;

    // If largest set of observables that can be integrated is empty set, problem was ill defined
    // Postpone error messaging and handling to end of function, exit loop here
    if (innerSet.empty()) {
      error = true ;
      break ;
    }

    // Prepare name and title of integral to be created
    TString title(integrand->GetTitle()) ;
    title.Prepend("Integral of ") ;

    TString name(integrand->GetName()) ;
    name.Append(integrand->integralNameSuffix(innerSet,nset,rangeName)) ;

    // Construct innermost integral
    integral = new RooRealIntegral(name,title,*integrand,innerSet,nset,cfg,rangeName) ;

    // Integral of integral takes ownership of innermost integral
    if (integrand != this) {
      integral->addOwnedComponents(*integrand) ;
    }

    // Remove already integrated observables from to-do std::list
    iset.remove(innerSet) ;

    // Send info message on recursion if needed
    if (integrand == this && iset.getSize()>0) {
      coutI(Integration) << GetName() << " : multidimensional integration over observables with parameterized ranges in terms of other integrated observables detected, using recursive integration strategy to construct final integral" << std::endl ;
    }

    // Prepare for recursion, next integral should integrate last integrand
    integrand = integral ;


    // Only need normalization set in innermost integration
    nset = 0 ;
  }

  if (error) {
    coutE(Integration) << GetName() << " : ERROR while defining recursive integral over observables with parameterized integration ranges, please check that integration rangs specify uniquely defined integral " << std::endl;
    delete integral ;
    integral = 0 ;
    return integral ;
  }


  // After-burner: apply interpolating cache on (numeric) integral if requested by user
  const char* cacheParamsStr = getStringAttribute("CACHEPARAMINT") ;
  if (cacheParamsStr && strlen(cacheParamsStr)) {

    std::unique_ptr<RooArgSet> intParams{integral->getVariables()};

    RooArgSet cacheParams = RooHelpers::selectFromArgSet(*intParams, cacheParamsStr);

    if (cacheParams.getSize()>0) {
      cxcoutD(Caching) << "RooAbsReal::createIntObj(" << GetName() << ") INFO: constructing " << cacheParams.getSize()
           << "-dim value cache for integral over " << iset2 << " as a function of " << cacheParams << " in range " << (rangeName?rangeName:"<none>") <<  std::endl ;
      std::string name = Form("%s_CACHE_[%s]",integral->GetName(),cacheParams.contentsString().c_str()) ;
      RooCachedReal* cachedIntegral = new RooCachedReal(name.c_str(),name.c_str(),*integral,cacheParams) ;
      cachedIntegral->setInterpolationOrder(2) ;
      cachedIntegral->addOwnedComponents(*integral) ;
      cachedIntegral->setCacheSource(true) ;
      if (integral->operMode()==ADirty) {
   cachedIntegral->setOperMode(ADirty) ;
      }
      //cachedIntegral->disableCache(true) ;
      integral = cachedIntegral ;
    }
  }

  return integral ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for createIntObj() that aids in the construct of recursive integrals
/// over functions with multiple observables with parameterized ranges. This function
/// finds in a given set allObs over which integration is requested the largeset subset
/// of observables that can be integrated simultaneously. This subset consists of
/// observables with fixed ranges and observables with parameterized ranges whose
/// parameterization does not depend on any observable that is also integrated.

void RooAbsReal::findInnerMostIntegration(const RooArgSet& allObs, RooArgSet& innerObs, const char* rangeName) const
{
  // Make std::lists of
  // a) integrated observables with fixed ranges,
  // b) integrated observables with parameterized ranges depending on other integrated observables
  // c) integrated observables used in definition of any parameterized ranges of integrated observables
  RooArgSet obsWithFixedRange(allObs) ;
  RooArgSet obsWithParamRange ;
  RooArgSet obsServingAsRangeParams ;

  // Loop over all integrated observables
  for (const auto aarg : allObs) {
    // Check if observable is real-valued lvalue
    RooAbsRealLValue* arglv = dynamic_cast<RooAbsRealLValue*>(aarg) ;
    if (arglv) {

      // Check if range is parameterized
      RooAbsBinning& binning = arglv->getBinning(rangeName,false,true) ;
      if (binning.isParameterized()) {
        RooArgSet loBoundObs;
        RooArgSet hiBoundObs;
        binning.lowBoundFunc()->getObservables(&allObs, loBoundObs) ;
        binning.highBoundFunc()->getObservables(&allObs, hiBoundObs) ;

        // Check if range parameterization depends on other integrated observables
        if (loBoundObs.overlaps(allObs) || hiBoundObs.overlaps(allObs)) {
          obsWithParamRange.add(*aarg) ;
          obsWithFixedRange.remove(*aarg) ;
          obsServingAsRangeParams.add(loBoundObs,false) ;
          obsServingAsRangeParams.add(hiBoundObs,false) ;
        }
      }
    }
  }

  // Make std::list of fixed-range observables that are _not_ involved in the parameterization of ranges of other observables
  RooArgSet obsWithFixedRangeNP(obsWithFixedRange) ;
  obsWithFixedRangeNP.remove(obsServingAsRangeParams) ;

  // Make std::list of param-range observables that are _not_ involved in the parameterization of ranges of other observables
  RooArgSet obsWithParamRangeNP(obsWithParamRange) ;
  obsWithParamRangeNP.remove(obsServingAsRangeParams) ;

  // Construct inner-most integration: over observables (with fixed or param range) not used in any other param range definitions
  innerObs.removeAll() ;
  innerObs.add(obsWithFixedRangeNP) ;
  innerObs.add(obsWithParamRangeNP) ;

}


////////////////////////////////////////////////////////////////////////////////
/// Construct std::string with unique suffix name to give to integral object that encodes
/// integrated observables, normalization observables and the integration range name

TString RooAbsReal::integralNameSuffix(const RooArgSet& iset, const RooArgSet* nset, const char* rangeName, bool omitEmpty) const
{
  TString name ;
  if (iset.getSize()>0) {

    RooArgSet isetTmp(iset) ;
    isetTmp.sort() ;

    name.Append("_Int[") ;
    bool first(true) ;
    for(RooAbsArg * arg : isetTmp) {
      if (first) {
   first=false ;
      } else {
   name.Append(",") ;
      }
      name.Append(arg->GetName()) ;
    }
    if (rangeName) {
      name.Append("|") ;
      name.Append(rangeName) ;
    }
    name.Append("]");
  } else if (!omitEmpty) {
    name.Append("_Int[]") ;
  }

  if (nset && nset->getSize()>0 ) {

    RooArgSet nsetTmp(*nset) ;
    nsetTmp.sort() ;

    name.Append("_Norm[") ;
    bool first(true);
    for(RooAbsArg * arg : nsetTmp) {
      if (first) {
   first=false ;
      } else {
   name.Append(",") ;
      }
      name.Append(arg->GetName()) ;
    }
    const RooAbsPdf* thisPdf = dynamic_cast<const RooAbsPdf*>(this) ;
    if (thisPdf && thisPdf->normRange()) {
      name.Append("|") ;
      name.Append(thisPdf->normRange()) ;
    }
    name.Append("]") ;
  }

  return name ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for plotOn() that creates a projection of a function or p.d.f
/// to be plotted on a RooPlot.
/// \ref createPlotProjAnchor "createPlotProjection()"

const RooAbsReal* RooAbsReal::createPlotProjection(const RooArgSet& depVars, const RooArgSet& projVars,
                                               RooArgSet*& cloneSet) const
{
  return createPlotProjection(depVars,&projVars,cloneSet) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Utility function for plotOn() that creates a projection of a function or p.d.f
/// to be plotted on a RooPlot.
/// \anchor createPlotProjAnchor
///
/// Create a new object \f$ G \f$ that represents the normalized projection:
/// \f[
///  G[x,p] = \frac{\int F[x,y,p] \; \mathrm{d}\{y\}}
///                {\int F[x,y,p] \; \mathrm{d}\{x\} \, \mathrm{d}\{y\}}
/// \f]
/// where \f$ F[x,y,p] \f$ is the function we represent, and
/// \f$ \{ p \} \f$ are the remaining variables ("parameters").
///
/// \param[in] dependentVars Dependent variables over which to normalise, \f$ \{x\} \f$.
/// \param[in] projectedVars Variables to project out, \f$ \{ y \} \f$.
/// \param[out] cloneSet Will be set to a RooArgSet*, which will contain a clone of *this plus its projection integral object.
/// The latter will also be returned. The caller takes ownership of this set.
/// \param[in] rangeName Optional range for projection integrals
/// \param[in] condObs Conditional observables, which are not integrated for normalisation, even if they
/// are in `dependentVars` or `projectedVars`.
/// \return A pointer to the newly created object, or zero in case of an
/// error. The caller is responsible for deleting the `cloneSet` (which includes the returned projection object).
const RooAbsReal *RooAbsReal::createPlotProjection(const RooArgSet &dependentVars, const RooArgSet *projectedVars,
                     RooArgSet *&cloneSet, const char* rangeName, const RooArgSet* condObs) const
{
  // Get the set of our leaf nodes
  RooArgSet leafNodes;
  RooArgSet treeNodes;
  leafNodeServerList(&leafNodes,this);
  treeNodeServerList(&treeNodes,this) ;


  // Check that the dependents are all fundamental. Filter out any that we
  // do not depend on, and make substitutions by name in our leaf std::list.
  // Check for overlaps with the projection variables.
  for (const auto arg : dependentVars) {
    if(!arg->isFundamental() && !dynamic_cast<const RooAbsLValue*>(arg)) {
      coutE(Plotting) << ClassName() << "::" << GetName() << ":createPlotProjection: variable \"" << arg->GetName()
          << "\" of wrong type: " << arg->ClassName() << std::endl;
      return 0;
    }

    RooAbsArg *found= treeNodes.find(arg->GetName());
    if(!found) {
      coutE(Plotting) << ClassName() << "::" << GetName() << ":createPlotProjection: \"" << arg->GetName()
                << "\" is not a dependent and will be ignored." << std::endl;
      continue;
    }
    if(found != arg) {
      if (leafNodes.find(found->GetName())) {
        leafNodes.replace(*found,*arg);
      } else {
        leafNodes.add(*arg) ;

        // Remove any dependents of found, replace by dependents of LV node
        RooArgSet lvDep;
        arg->getObservables(&leafNodes, lvDep);
        for (const auto lvs : lvDep) {
          RooAbsArg* tmp = leafNodes.find(lvs->GetName()) ;
          if (tmp) {
            leafNodes.remove(*tmp) ;
            leafNodes.add(*lvs) ;
          }
        }
      }
    }

    // check if this arg is also in the projection set
    if(0 != projectedVars && projectedVars->find(arg->GetName())) {
      coutE(Plotting) << ClassName() << "::" << GetName() << ":createPlotProjection: \"" << arg->GetName()
                << "\" cannot be both a dependent and a projected variable." << std::endl;
      return 0;
    }
  }

  // Remove the projected variables from the std::list of leaf nodes, if necessary.
  if(0 != projectedVars) leafNodes.remove(*projectedVars,true);

  // Make a deep-clone of ourself so later operations do not disturb our original state
  cloneSet= (RooArgSet*)RooArgSet(*this).snapshot(true);
  if (!cloneSet) {
    coutE(Plotting) << "RooAbsPdf::createPlotProjection(" << GetName() << ") Couldn't deep-clone PDF, abort," << std::endl ;
    return 0 ;
  }
  RooAbsReal *theClone= (RooAbsReal*)cloneSet->find(GetName());

  // The remaining entries in our std::list of leaf nodes are the external
  // dependents (x) and parameters (p) of the projection. Patch them back
  // into the theClone. This orphans the nodes they replace, but the orphans
  // are still in the cloneList and so will be cleaned up eventually.
  //cout << "redirection leafNodes : " ; leafNodes.Print("1") ;

  std::unique_ptr<RooArgSet> plotLeafNodes{static_cast<RooArgSet*>(leafNodes.selectCommon(dependentVars))};
  theClone->recursiveRedirectServers(*plotLeafNodes,false,false,false);

  // Create the set of normalization variables to use in the projection integrand
  RooArgSet normSet(dependentVars);
  if(0 != projectedVars) normSet.add(*projectedVars);
  if(0 != condObs) {
    normSet.remove(*condObs,true,true) ;
  }

  // Try to create a valid projection integral. If no variables are to be projected,
  // create a null projection anyway to bind our normalization over the dependents
  // consistently with the way they would be bound with a non-trivial projection.
  RooArgSet empty;
  if(0 == projectedVars) projectedVars= &empty;

  TString name = GetName() ;
  name += integralNameSuffix(*projectedVars,&normSet,rangeName,true) ;

  TString title(GetTitle());
  title.Prepend("Projection of ");


  RooAbsReal* projected= theClone->createIntegral(*projectedVars,normSet,rangeName) ;

  if(0 == projected || !projected->isValid()) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":createPlotProjection: cannot integrate out ";
    projectedVars->printStream(std::cout,kName|kArgs,kSingleLine);
    // cleanup and exit
    if(0 != projected) delete projected;
    return 0;
  }

  if(projected->InheritsFrom(RooRealIntegral::Class())){
    static_cast<RooRealIntegral*>(projected)->setAllowComponentSelection(true);
  }

  projected->SetName(name.Data()) ;
  projected->SetTitle(title.Data()) ;

  // Add the projection integral to the cloneSet so that it eventually gets cleaned up by the caller.
  cloneSet->addOwned(*projected);

  // return a const pointer to remind the caller that they do not delete the returned object
  // directly (it is contained in the cloneSet instead).
  return projected;
}




////////////////////////////////////////////////////////////////////////////////
/// Fill the ROOT histogram 'hist' with values sampled from this
/// function at the bin centers.  Our value is calculated by first
/// integrating out any variables in projectedVars and then scaling
/// the result by scaleFactor. Returns a pointer to the input
/// histogram, or zero in case of an error. The input histogram can
/// be any TH1 subclass, and therefore of arbitrary
/// dimension. Variables are matched with the (x,y,...) dimensions of
/// the input histogram according to the order in which they appear
/// in the input plotVars std::list. If scaleForDensity is true the
/// histogram is filled with a the functions density rather than
/// the functions value (i.e. the value at the bin center is multiplied
/// with bin volume)

TH1 *RooAbsReal::fillHistogram(TH1 *hist, const RooArgList &plotVars,
                double scaleFactor, const RooArgSet *projectedVars, bool scaleForDensity,
                const RooArgSet* condObs, bool setError) const
{
  // Do we have a valid histogram to use?
  if(0 == hist) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: no valid histogram to fill" << std::endl;
    return 0;
  }

  // Check that the number of plotVars matches the input histogram's dimension
  Int_t hdim= hist->GetDimension();
  if(hdim != plotVars.getSize()) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: plotVars has the wrong dimension" << std::endl;
    return 0;
  }


  // Check that the plot variables are all actually RooRealVars and print a warning if we do not
  // explicitly depend on one of them. Fill a set (not std::list!) of cloned plot variables.
  RooArgSet plotClones;
  for(Int_t index= 0; index < plotVars.getSize(); index++) {
    const RooAbsArg *var= plotVars.at(index);
    const RooRealVar *realVar= dynamic_cast<const RooRealVar*>(var);
    if(0 == realVar) {
      coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: cannot plot variable \"" << var->GetName()
      << "\" of type " << var->ClassName() << std::endl;
      return 0;
    }
    if(!this->dependsOn(*realVar)) {
      coutE(InputArguments) << ClassName() << "::" << GetName()
      << ":fillHistogram: WARNING: variable is not an explicit dependent: " << realVar->GetName() << std::endl;
    }
    plotClones.addClone(*realVar,true); // do not complain about duplicates
  }

  // Reconnect all plotClones to each other, imported when plotting N-dim integrals with entangled parameterized ranges
  for(RooAbsArg * pc : plotClones) {
    pc->recursiveRedirectServers(plotClones,false,false,true) ;
  }

  // Call checkObservables
  RooArgSet allDeps(plotClones) ;
  if (projectedVars) {
    allDeps.add(*projectedVars) ;
  }
  if (checkObservables(&allDeps)) {
    coutE(InputArguments) << "RooAbsReal::fillHistogram(" << GetName() << ") error in checkObservables, abort" << std::endl ;
    return hist ;
  }

  // Create a standalone projection object to use for calculating bin contents
  RooArgSet *cloneSet = 0;
  const RooAbsReal *projected= createPlotProjection(plotClones,projectedVars,cloneSet,0,condObs);

  cxcoutD(Plotting) << "RooAbsReal::fillHistogram(" << GetName() << ") plot projection object is " << projected->GetName() << std::endl ;

  // Prepare to loop over the histogram bins
  Int_t xbins(0),ybins(1),zbins(1);
  RooRealVar *xvar = 0;
  RooRealVar *yvar = 0;
  RooRealVar *zvar = 0;
  TAxis *xaxis = 0;
  TAxis *yaxis = 0;
  TAxis *zaxis = 0;
  switch(hdim) {
  case 3:
    zbins= hist->GetNbinsZ();
    zvar= dynamic_cast<RooRealVar*>(plotClones.find(plotVars.at(2)->GetName()));
    zaxis= hist->GetZaxis();
    assert(0 != zvar && 0 != zaxis);
    if (scaleForDensity) {
      scaleFactor*= (zaxis->GetXmax() - zaxis->GetXmin())/zbins;
    }
    // fall through to next case...
  case 2:
    ybins= hist->GetNbinsY();
    yvar= dynamic_cast<RooRealVar*>(plotClones.find(plotVars.at(1)->GetName()));
    yaxis= hist->GetYaxis();
    assert(0 != yvar && 0 != yaxis);
    if (scaleForDensity) {
      scaleFactor*= (yaxis->GetXmax() - yaxis->GetXmin())/ybins;
    }
    // fall through to next case...
  case 1:
    xbins= hist->GetNbinsX();
    xvar= dynamic_cast<RooRealVar*>(plotClones.find(plotVars.at(0)->GetName()));
    xaxis= hist->GetXaxis();
    assert(0 != xvar && 0 != xaxis);
    if (scaleForDensity) {
      scaleFactor*= (xaxis->GetXmax() - xaxis->GetXmin())/xbins;
    }
    break;
  default:
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: cannot fill histogram with "
           << hdim << " dimensions" << std::endl;
    break;
  }

  // Loop over the input histogram's bins and fill each one with our projection's
  // value, calculated at the center.
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
  Int_t xbin(0),ybin(0),zbin(0);
  Int_t bins= xbins*ybins*zbins;
  for(Int_t bin= 0; bin < bins; bin++) {
    switch(hdim) {
    case 3:
      if(bin % (xbins*ybins) == 0) {
   zbin++;
   zvar->setVal(zaxis->GetBinCenter(zbin));
      }
      // fall through to next case...
    case 2:
      if(bin % xbins == 0) {
   ybin= (ybin%ybins) + 1;
   yvar->setVal(yaxis->GetBinCenter(ybin));
      }
      // fall through to next case...
    case 1:
      xbin= (xbin%xbins) + 1;
      xvar->setVal(xaxis->GetBinCenter(xbin));
      break;
    default:
      coutE(InputArguments) << "RooAbsReal::fillHistogram: Internal Error!" << std::endl;
      break;
    }

    double result= scaleFactor*projected->getVal();
    if (RooAbsReal::numEvalErrors()>0) {
      coutW(Plotting) << "WARNING: Function evaluation error(s) at coordinates [x]=" << xvar->getVal() ;
      if (hdim==2) ccoutW(Plotting) << " [y]=" << yvar->getVal() ;
      if (hdim==3) ccoutW(Plotting) << " [z]=" << zvar->getVal() ;
      ccoutW(Plotting) << std::endl ;
      // RooAbsReal::printEvalErrors(ccoutW(Plotting),10) ;
      result = 0 ;
    }
    RooAbsReal::clearEvalErrorLog() ;

    hist->SetBinContent(hist->GetBin(xbin,ybin,zbin),result);
    if (setError) {
      hist->SetBinError(hist->GetBin(xbin,ybin,zbin),sqrt(result)) ;
    }

    //cout << "bin " << bin << " -> (" << xbin << "," << ybin << "," << zbin << ") = " << result << std::endl;
  }
  RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;

  // cleanup
  delete cloneSet;

  return hist;
}



////////////////////////////////////////////////////////////////////////////////
/// Fill a RooDataHist with values sampled from this function at the
/// bin centers.  If extendedMode is true, the p.d.f. values is multiplied
/// by the number of expected events in each bin
///
/// An optional scaling by a given scaleFactor can be performed.
/// Returns a pointer to the input RooDataHist, or zero
/// in case of an error.
///
/// If correctForBinSize is true the RooDataHist
/// is filled with the functions density (function value times the
/// bin volume) rather than function value.
///
/// If showProgress is true
/// a process indicator is printed on stdout in steps of one percent,
/// which is mostly useful for the sampling of expensive functions
/// such as likelihoods

RooDataHist* RooAbsReal::fillDataHist(RooDataHist *hist, const RooArgSet* normSet, double scaleFactor,
                  bool correctForBinSize, bool showProgress) const
{
  // Do we have a valid histogram to use?
  if(0 == hist) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillDataHist: no valid RooDataHist to fill" << std::endl;
    return 0;
  }

  // Call checkObservables
  RooArgSet allDeps(*hist->get()) ;
  if (checkObservables(&allDeps)) {
    coutE(InputArguments) << "RooAbsReal::fillDataHist(" << GetName() << ") error in checkObservables, abort" << std::endl ;
    return hist ;
  }

  // Make deep clone of self and attach to dataset observables
  //RooArgSet* origObs = getObservables(hist) ;
  std::unique_ptr<RooArgSet> cloneSet{static_cast<RooArgSet*>(RooArgSet(*this).snapshot(true))};
  RooAbsReal* theClone = (RooAbsReal*) cloneSet->find(GetName()) ;
  theClone->recursiveRedirectServers(*hist->get()) ;
  //const_cast<RooAbsReal*>(this)->recursiveRedirectServers(*hist->get()) ;

  // Iterator over all bins of RooDataHist and fill weights
  Int_t onePct = hist->numEntries()/100 ;
  if (onePct==0) {
    onePct++ ;
  }
  for (Int_t i=0 ; i<hist->numEntries() ; i++) {
    if (showProgress && (i%onePct==0)) {
      ccoutP(Eval) << "." << std::flush ;
    }
    const RooArgSet* obs = hist->get(i) ;
    double binVal = theClone->getVal(normSet?normSet:obs)*scaleFactor ;
    if (correctForBinSize) {
      binVal*= hist->binVolume() ;
    }
    hist->set(i, binVal, 0.);
  }

  return hist;
}




////////////////////////////////////////////////////////////////////////////////
/// Create and fill a ROOT histogram TH1, TH2 or TH3 with the values of this function for the variables with given names.
/// \param[in] varNameList List of variables to use for x, y, z axis, separated by ':'
/// \param[in] xbins Number of bins for first variable
/// \param[in] ybins Number of bins for second variable
/// \param[in] zbins Number of bins for third variable
/// \return TH1*, which is one of TH[1-3]. The histogram is owned by the caller.
///
/// For a greater degree of control use
/// RooAbsReal::createHistogram(const char *, const RooAbsRealLValue&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&) const
///

TH1* RooAbsReal::createHistogram(RooStringView varNameList, Int_t xbins, Int_t ybins, Int_t zbins) const
{
  std::unique_ptr<RooArgSet> vars{getVariables()};

  auto varNames = ROOT::Split(varNameList, ",:");
  std::vector<RooRealVar*> histVars(3, nullptr);

  for(std::size_t iVar = 0; iVar < varNames.size(); ++iVar) {
    if(varNames[iVar].empty()) continue;
    if(iVar >= 3) {
      std::stringstream errMsg;
      errMsg << "RooAbsPdf::createHistogram(" << GetName() << ") ERROR more than three variable names passed, but maxumum number of supported variables is three";
      coutE(Plotting) << errMsg.str() << std::endl;
      throw std::invalid_argument(errMsg.str());
    }
    auto var = static_cast<RooRealVar*>(vars->find(varNames[iVar].c_str()));
    if(!var) {
      std::stringstream errMsg;
      errMsg << "RooAbsPdf::createHistogram(" << GetName() << ") ERROR variable " << varNames[iVar] << " does not exist in argset: " << *vars;
      coutE(Plotting) << errMsg.str() << std::endl;
      throw std::runtime_error(errMsg.str());
    }
    histVars[iVar] = var;
  }

  // Construct std::list of named arguments to pass to the implementation version of createHistogram()

  RooLinkedList argList ;
  if (xbins>0) {
    argList.Add(RooFit::Binning(xbins).Clone()) ;
  }

  if (histVars[1]) {
    argList.Add(RooFit::YVar(*histVars[1], ybins > 0 ? RooFit::Binning(ybins) : RooCmdArg::none()).Clone()) ;
  }

  if (histVars[2]) {
    argList.Add(RooFit::ZVar(*histVars[2], zbins > 0 ? RooFit::Binning(zbins) : RooCmdArg::none()).Clone()) ;
  }

  // Call implementation function
  TH1* result = createHistogram(GetName(), *histVars[0], argList) ;

  // Delete temporary std::list of RooCmdArgs
  argList.Delete() ;

  return result ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create and fill a ROOT histogram TH1, TH2 or TH3 with the values of this function.
///
/// \param[in] name  Name of the ROOT histogram
/// \param[in] xvar  Observable to be std::mapped on x axis of ROOT histogram
/// \param[in] arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8  Arguments according to std::list below
/// \return TH1 *, one of TH{1,2,3}. The caller takes ownership.
///
/// <table>
/// <tr><th><th> Effect on histogram creation
/// <tr><td> `IntrinsicBinning()`                           <td> Apply binning defined by function or pdf (as advertised via binBoundaries() method)
/// <tr><td> `Binning(const char* name)`                    <td> Apply binning with given name to x axis of histogram
/// <tr><td> `Binning(RooAbsBinning& binning)`              <td> Apply specified binning to x axis of histogram
/// <tr><td> `Binning(int nbins, [double lo, double hi])`   <td> Apply specified binning to x axis of histogram
/// <tr><td> `ConditionalObservables(Args_t &&... argsOrArgSet)` <td> Do not normalise PDF over following observables when projecting PDF into histogram.
//                                                               Arguments can either be multiple RooRealVar or a single RooArgSet containing them.
/// <tr><td> `Scaling(bool)`                              <td> Apply density-correction scaling (multiply by bin volume), default is true
/// <tr><td> `Extended(bool)`                             <td> Plot event yield instead of probability density (for extended pdfs only)
///
/// <tr><td> `YVar(const RooAbsRealLValue& var,...)`    <td> Observable to be std::mapped on y axis of ROOT histogram.
/// The YVar() and ZVar() arguments can be supplied with optional Binning() arguments to control the binning of the Y and Z axes, e.g.
/// ```
/// createHistogram("histo",x,Binning(-1,1,20), YVar(y,Binning(-1,1,30)), ZVar(z,Binning("zbinning")))
/// ```
/// <tr><td> `ZVar(const RooAbsRealLValue& var,...)`    <td> Observable to be std::mapped on z axis of ROOT histogram
/// </table>
///
///

TH1 *RooAbsReal::createHistogram(const char *name, const RooAbsRealLValue& xvar,
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
/// Internal method implementing createHistogram

TH1* RooAbsReal::createHistogram(const char *name, const RooAbsRealLValue& xvar, RooLinkedList& argList) const
{

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::createHistogram(%s)",GetName())) ;
  pc.defineInt("scaling","Scaling",0,1) ;
  pc.defineInt("intBinning","IntrinsicBinning",0,2) ;
  pc.defineInt("extended","Extended",0,2) ;

  pc.defineSet("compSet","SelectCompSet",0);
  pc.defineString("compSpec","SelectCompSpec",0) ;
  pc.defineSet("projObs","ProjectedObservables",0,0) ;
  pc.defineObject("yvar","YVar",0,0) ;
  pc.defineObject("zvar","ZVar",0,0) ;
  pc.defineMutex("SelectCompSet","SelectCompSpec") ;
  pc.defineMutex("IntrinsicBinning","Binning") ;
  pc.defineMutex("IntrinsicBinning","BinningName") ;
  pc.defineMutex("IntrinsicBinning","BinningSpec") ;
  pc.allowUndefined() ;

  // Process & check varargs
  pc.process(argList) ;
  if (!pc.ok(true)) {
    return 0 ;
  }

  RooArgList vars(xvar) ;
  RooAbsArg* yvar = static_cast<RooAbsArg*>(pc.getObject("yvar")) ;
  if (yvar) {
    vars.add(*yvar) ;
  }
  RooAbsArg* zvar = static_cast<RooAbsArg*>(pc.getObject("zvar")) ;
  if (zvar) {
    vars.add(*zvar) ;
  }

  auto projObs = pc.getSet("projObs");
  RooArgSet* intObs = 0 ;

  bool doScaling = pc.getInt("scaling") ;
  Int_t doIntBinning = pc.getInt("intBinning") ;
  Int_t doExtended = pc.getInt("extended") ;

  // If doExtended is two, selection is automatic, set to 1 of pdf is extended, to zero otherwise
  const RooAbsPdf* pdfSelf = dynamic_cast<const RooAbsPdf*>(this) ;
  if (!pdfSelf && doExtended == 1) {
    coutW(InputArguments) << "RooAbsReal::createHistogram(" << GetName() << ") WARNING extended mode requested for a non-pdf object, ignored" << std::endl ;
    doExtended=0 ;
  }
  if (pdfSelf && doExtended==1 && pdfSelf->extendMode()==RooAbsPdf::CanNotBeExtended) {
    coutW(InputArguments) << "RooAbsReal::createHistogram(" << GetName() << ") WARNING extended mode requested for a non-extendable pdf, ignored" << std::endl ;
    doExtended=0 ;
  }
  if (pdfSelf && doExtended==2) {
    doExtended = pdfSelf->extendMode()==RooAbsPdf::CanNotBeExtended ? 0 : 1 ;
  } else if(!pdfSelf) {
    doExtended = 0;
  }

  const char* compSpec = pc.getString("compSpec") ;
  const RooArgSet* compSet = pc.getSet("compSet");
  bool haveCompSel = ( (compSpec && strlen(compSpec)>0) || compSet) ;

  std::unique_ptr<RooBinning> intBinning;
  if (doIntBinning>0) {
    // Given RooAbsPdf* pdf and RooRealVar* obs
    std::unique_ptr<std::list<double>> bl{binBoundaries(const_cast<RooAbsRealLValue&>(xvar),xvar.getMin(),xvar.getMax())};
    if (!bl) {
      // Only emit warning when intrinsic binning is explicitly requested
      if (doIntBinning==1) {
   coutW(InputArguments) << "RooAbsReal::createHistogram(" << GetName()
               << ") WARNING, intrinsic model binning requested for histogram, but model does not define bin boundaries, reverting to default binning"<< std::endl ;
      }
    } else {
      if (doIntBinning==2) {
   coutI(InputArguments) << "RooAbsReal::createHistogram(" << GetName()
               << ") INFO: Model has intrinsic binning definition, selecting that binning for the histogram"<< std::endl ;
      }
      std::vector<double> ba(bl->size());
      int i=0 ;
      for (auto const& elem : *bl) { ba[i++] = elem ; }
      intBinning = std::make_unique<RooBinning>(bl->size()-1,ba.data()) ;
    }
  }

  RooLinkedList argListCreate(argList) ;
  RooCmdConfig::stripCmdList(argListCreate,"Scaling,ProjectedObservables,IntrinsicBinning,SelectCompSet,SelectCompSpec,Extended") ;

  TH1* histo(0) ;
  if (intBinning) {
    RooCmdArg tmp = RooFit::Binning(*intBinning) ;
    argListCreate.Add(&tmp) ;
    histo = xvar.createHistogram(name,argListCreate) ;
  } else {
    histo = xvar.createHistogram(name,argListCreate) ;
  }

  // Do component selection here
  if (haveCompSel) {

    // Get complete set of tree branch nodes
    RooArgSet branchNodeSet ;
    branchNodeServerList(&branchNodeSet) ;

    // Discard any non-RooAbsReal nodes
    for(RooAbsArg * arg : branchNodeSet) {
      if (!dynamic_cast<RooAbsReal*>(arg)) {
   branchNodeSet.remove(*arg) ;
      }
    }

    std::unique_ptr<RooArgSet> dirSelNodes;
    if (compSet) {
      dirSelNodes.reset(static_cast<RooArgSet*>(branchNodeSet.selectCommon(*compSet)));
    } else {
      dirSelNodes.reset(static_cast<RooArgSet*>(branchNodeSet.selectByName(compSpec)));
    }
    if (!dirSelNodes->empty()) {
      coutI(Plotting) << "RooAbsPdf::createHistogram(" << GetName() << ") directly selected PDF components: " << *dirSelNodes << std::endl ;

      // Do indirect selection and activate both
      plotOnCompSelect(dirSelNodes.get()) ;
    } else {
      if (compSet) {
   coutE(Plotting) << "RooAbsPdf::createHistogram(" << GetName() << ") ERROR: component selection set " << *compSet << " does not match any components of p.d.f." << std::endl ;
      } else {
   coutE(Plotting) << "RooAbsPdf::createHistogram(" << GetName() << ") ERROR: component selection expression '" << compSpec << "' does not select any components of p.d.f." << std::endl ;
      }
      return 0 ;
    }
  }

  double scaleFactor(1.0) ;
  if (doExtended) {
    scaleFactor = pdfSelf->expectedEvents(vars) ;
    doScaling=false ;
  }

  fillHistogram(histo,vars,scaleFactor,intObs,doScaling,projObs,false) ;

  // Deactivate component selection
  if (haveCompSel) {
      plotOnCompSelect(0) ;
  }


  return histo ;
}


////////////////////////////////////////////////////////////////////////////////
/// Helper function for plotting of composite p.d.fs. Given
/// a set of selected components that should be plotted,
/// find all nodes that (in)directly depend on these selected
/// nodes. Mark all directly and indirecty selected nodes
/// as 'selected' using the selectComp() method

void RooAbsReal::plotOnCompSelect(RooArgSet* selNodes) const
{
  // Get complete set of tree branch nodes
  RooArgSet branchNodeSet;
  branchNodeServerList(&branchNodeSet);

  // Discard any non-PDF nodes
  // Iterate by number because collection is being modified! Iterators may invalidate ...
  for (unsigned int i = 0; i < branchNodeSet.size(); ++i) {
    const auto arg = branchNodeSet[i];
    if (!dynamic_cast<RooAbsReal*>(arg)) {
      branchNodeSet.remove(*arg) ;
    }
  }

  // If no set is specified, restored all selection bits to true
  if (!selNodes) {
    // Reset PDF selection bits to true
    for (const auto arg : branchNodeSet) {
      static_cast<RooAbsReal*>(arg)->selectComp(true);
    }
    return ;
  }


  // Add all nodes below selected nodes
  RooArgSet tmp;
  for (const auto arg : branchNodeSet) {
    for (const auto selNode : *selNodes) {
      if (selNode->dependsOn(*arg)) {
        tmp.add(*arg,true);
      }
    }
  }

  // Add all nodes that depend on selected nodes
  for (const auto arg : branchNodeSet) {
    if (arg->dependsOn(*selNodes)) {
      tmp.add(*arg,true);
    }
  }

  tmp.remove(*selNodes, true);
  tmp.remove(*this);
  selNodes->add(tmp);
  coutI(Plotting) << "RooAbsPdf::plotOn(" << GetName() << ") indirectly selected PDF components: " << tmp << std::endl ;

  // Set PDF selection bits according to selNodes
  for (const auto arg : branchNodeSet) {
    bool select = selNodes->find(arg->GetName()) != nullptr;
    static_cast<RooAbsReal*>(arg)->selectComp(select);
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Plot (project) PDF on specified frame. If a PDF is plotted in an empty frame, it
/// will show a unit normalized curve in the frame variable, taken at the present value
/// of other observables defined for this PDF.
///
/// \param[in] frame pointer to RooPlot
/// \param[in] arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10 Ordered arguments
///
/// If a PDF is plotted in a frame in which a dataset has already been plotted, it will
/// show a projected curve integrated over all variables that were present in the shown
/// dataset except for the one on the x-axis. The normalization of the curve will also
/// be adjusted to the event count of the plotted dataset. An informational message
/// will be printed for each projection step that is performed.
///
/// This function takes the following named arguments
/// <table>
/// <tr><th><th> Projection control
/// <tr><td> `Slice(const RooArgSet& set)`     <td> Override default projection behaviour by omitting observables std::listed
///                                    in set from the projection, i.e. by not integrating over these.
///                                    Slicing is usually only sensible in discrete observables, by e.g. creating a slice
///                                    of the PDF at the current value of the category observable.
///
/// <tr><td> `Slice(RooCategory& cat, const char* label)`        <td> Override default projection behaviour by omitting the specified category
///                                    observable from the projection, i.e., by not integrating over all states of this category.
///                                    The slice is positioned at the given label value. To pass multiple Slice() commands, please use the
///                                    Slice(std::map<RooCategory*, std::string> const&) argument explained below.
///
/// <tr><td> `Slice(std::map<RooCategory*, std::string> const&)`        <td> Omits multiple categories from the projection, as explianed above.
///                                    Can be used with initializer std::lists for convenience, e.g.
/// ```{.cpp}
///   pdf.plotOn(frame, Slice({{&tagCategory, "2tag"}, {&jetCategory, "3jet"}});
/// ```
///
/// <tr><td> `Project(const RooArgSet& set)`   <td> Override default projection behaviour by projecting over observables
///                                    given in the set, ignoring the default projection behavior. Advanced use only.
///
/// <tr><td> `ProjWData(const RooAbsData& d)`  <td> Override default projection _technique_ (integration). For observables present in given dataset
///                                    projection of PDF is achieved by constructing an average over all observable values in given set.
///                                    Consult RooFit plotting tutorial for further explanation of meaning & use of this technique
///
/// <tr><td> `ProjWData(const RooArgSet& s, const RooAbsData& d)`   <td> As above but only consider subset 's' of observables in dataset 'd' for projection through data averaging
///
/// <tr><td> `ProjectionRange(const char* rn)` <td> Override default range of projection integrals to a different range speficied by given range name.
///                                    This technique allows you to project a finite width slice in a real-valued observable
///
/// <tr><td> `NumCPU(Int_t ncpu)`              <td> Number of CPUs to use simultaneously to calculate data-weighted projections (only in combination with ProjWData)
///
///
/// <tr><th><th> Misc content control
/// <tr><td> `PrintEvalErrors(Int_t numErr)`   <td> Control number of p.d.f evaluation errors printed per curve. A negative
///                                    value suppress output completely, a zero value will only print the error count per p.d.f component,
///                                    a positive value is will print details of each error up to numErr messages per p.d.f component.
///
/// <tr><td> `EvalErrorValue(double value)`  <td> Set curve points at which (pdf) evaluation errors occur to specified value. By default the
///                                    function value is plotted.
///
/// <tr><td> `Normalization(double scale, ScaleType code)`   <td> Adjust normalization by given scale factor. Interpretation of number depends on code:
///                    - Relative: relative adjustment factor for a normalized function,
///                    - NumEvent: scale to match given number of events.
///                    - Raw: relative adjustment factor for an un-normalized function.
///
/// <tr><td> `Name(const chat* name)`          <td> Give curve specified name in frame. Useful if curve is to be referenced later
///
/// <tr><td> `Asymmetry(const RooCategory& c)` <td> Show the asymmetry of the PDF in given two-state category [F(+)-F(-)] / [F(+)+F(-)] rather than
///                                    the PDF projection. Category must have two states with indices -1 and +1 or three states with
///                                    indeces -1,0 and +1.
///
/// <tr><td> `ShiftToZero(bool flag)`        <td> Shift entire curve such that lowest visible point is at exactly zero. Mostly useful when plotting \f$ -\log(L) \f$ or \f$ \chi^2 \f$ distributions
///
/// <tr><td> `AddTo(const char* name, double_t wgtSelf, double_t wgtOther)`   <td> Add constructed projection to already existing curve with given name and relative weight factors
/// <tr><td> `Components(const char* names)`  <td>  When plotting sums of PDFs, plot only the named components (*e.g.* only
///                                                 the signal of a signal+background model).
/// <tr><td> `Components(const RooArgSet& compSet)` <td> As above, but pass a RooArgSet of the components themselves.
///
/// <tr><th><th> Plotting control
/// <tr><td> `DrawOption(const char* opt)`     <td> Select ROOT draw option for resulting TGraph object. Currently supported options are "F" (fill), "L" (line), and "P" (points).
///           \note Option "P" will cause RooFit to plot (and treat) this pdf as if it were data! This is intended for plotting "corrected data"-type pdfs such as "data-minus-background" or unfolded datasets.
///
/// <tr><td> `LineStyle(Int_t style)`          <td> Select line style by ROOT line style code, default is solid
///
/// <tr><td> `LineColor(Int_t color)`          <td> Select line color by ROOT color code, default is blue
///
/// <tr><td> `LineWidth(Int_t width)`          <td> Select line with in pixels, default is 3
///
/// <tr><td> `MarkerStyle(Int_t style)`   <td> Select the ROOT marker style, default is 21
///
/// <tr><td> `MarkerColor(Int_t color)`   <td> Select the ROOT marker color, default is black
///
/// <tr><td> `MarkerSize(double size)`   <td> Select the ROOT marker size
///
/// <tr><td> `FillStyle(Int_t style)`          <td> Select fill style, default is not filled. If a filled style is selected, also use VLines()
///                                    to add vertical downward lines at end of curve to ensure proper closure. Add `DrawOption("F")` for filled drawing.
/// <tr><td> `FillColor(Int_t color)`          <td> Select fill color by ROOT color code
///
/// <tr><td> `Range(const char* name)`         <td> Only draw curve in range defined by given name
///
/// <tr><td> `Range(double lo, double hi)`     <td> Only draw curve in specified range
///
/// <tr><td> `VLines()`                        <td> Add vertical lines to y=0 at end points of curve
///
/// <tr><td> `Precision(double eps)`         <td> Control precision of drawn curve w.r.t to scale of plot, default is 1e-3. Higher precision
///                                    will result in more and more densely spaced curve points
///
/// <tr><td> `Invisible(bool flag)`           <td> Add curve to frame, but do not display. Useful in combination AddTo()
///
/// <tr><td> `VisualizeError(const RooFitResult& fitres, double Z=1, bool linearMethod=true)`
///                                  <td> Visualize the uncertainty on the parameters, as given in fitres, at 'Z' sigma'. The linear method is fast but may not be accurate in the presence of strong correlations (~>0.9) and at Z>2 due to linear and Gaussian approximations made. Intervals from the sampling method can be asymmetric, and may perform better in the presence of strong correlations, but may take (much) longer to calculate
///
/// <tr><td> `VisualizeError(const RooFitResult& fitres, const RooArgSet& param, double Z=1, bool linearMethod=true)`
///                                  <td> Visualize the uncertainty on the subset of parameters 'param', as given in fitres, at 'Z' sigma'
/// </table>
///
/// Details on error band visualization
/// -----------------------------------
/// *VisualizeError() uses plotOnWithErrorBand(). Documentation of the latter:*
/// \see plotOnWithErrorBand()

RooPlot* RooAbsReal::plotOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2,
             const RooCmdArg& arg3, const RooCmdArg& arg4,
             const RooCmdArg& arg5, const RooCmdArg& arg6,
             const RooCmdArg& arg7, const RooCmdArg& arg8,
             const RooCmdArg& arg9, const RooCmdArg& arg10) const
{
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  l.Add((TObject*)&arg9) ;  l.Add((TObject*)&arg10) ;
  return plotOn(frame,l) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Internal back-end function of plotOn() with named arguments

RooPlot* RooAbsReal::plotOn(RooPlot* frame, RooLinkedList& argList) const
{
  // Special handling here if argList contains RangeWithName argument with multiple
  // range names -- Need to translate this call into multiple calls

  RooCmdArg* rcmd = (RooCmdArg*) argList.FindObject("RangeWithName") ;
  if (rcmd && TString(rcmd->getString(0)).Contains(",")) {

    // List joint ranges as choice of normalization for all later processing
    RooCmdArg rnorm = RooFit::NormRange(rcmd->getString(0)) ;
    argList.Add(&rnorm) ;

    std::vector<std::string> rlist;

    // Separate named ranges using strtok
    for (const std::string& rangeNameToken : ROOT::Split(rcmd->getString(0), ",")) {
      rlist.emplace_back(rangeNameToken);
    }

    for (const auto& rangeString : rlist) {
      // Process each range with a separate command with a single range to be plotted
      rcmd->setString(0, rangeString.c_str());
      RooAbsReal::plotOn(frame,argList);
    }
    return frame ;

  }

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::plotOn(%s)",GetName())) ;
  pc.defineString("drawOption","DrawOption",0,"L") ;
  pc.defineString("projectionRangeName","ProjectionRange",0,"",true) ;
  pc.defineString("curveNameSuffix","CurveNameSuffix",0,"") ;
  pc.defineString("sliceCatState","SliceCat",0,"",true) ;
  pc.defineDouble("scaleFactor","Normalization",0,1.0) ;
  pc.defineInt("scaleType","Normalization",0,Relative) ;
  pc.defineSet("sliceSet","SliceVars",0) ;
  pc.defineObject("sliceCatList","SliceCat",0,0,true) ;
  // This dummy is needed for plotOn to recognize the "SliceCatMany" command.
  // It is not used directly, but the "SliceCat" commands are nested in it.
  // Removing this dummy definition results in "ERROR: unrecognized command: SliceCatMany".
  pc.defineObject("dummy1","SliceCatMany",0) ;
  pc.defineSet("projSet","Project",0) ;
  pc.defineObject("asymCat","Asymmetry",0) ;
  pc.defineDouble("precision","Precision",0,1e-3) ;
  pc.defineDouble("evalErrorVal","EvalErrorValue",0,0) ;
  pc.defineInt("doEvalError","EvalErrorValue",0,0) ;
  pc.defineInt("shiftToZero","ShiftToZero",0,0) ;
  pc.defineSet("projDataSet","ProjData",0) ;
  pc.defineObject("projData","ProjData",1) ;
  pc.defineObject("errorFR","VisualizeError",0) ;
  pc.defineDouble("errorZ","VisualizeError",0,1.) ;
  pc.defineSet("errorPars","VisualizeError",0) ;
  pc.defineInt("linearMethod","VisualizeError",0,0) ;
  pc.defineInt("binProjData","ProjData",0,0) ;
  pc.defineDouble("rangeLo","Range",0,-999.) ;
  pc.defineDouble("rangeHi","Range",1,-999.) ;
  pc.defineInt("numee","PrintEvalErrors",0,10) ;
  pc.defineInt("rangeAdjustNorm","Range",0,0) ;
  pc.defineInt("rangeWNAdjustNorm","RangeWithName",0,0) ;
  pc.defineInt("VLines","VLines",0,2) ; // 2==ExtendedWings
  pc.defineString("rangeName","RangeWithName",0,"") ;
  pc.defineString("normRangeName","NormRange",0,"") ;
  pc.defineInt("markerColor","MarkerColor",0,-999) ;
  pc.defineInt("markerStyle","MarkerStyle",0,-999) ;
  pc.defineDouble("markerSize","MarkerSize",0,-999) ;
  pc.defineInt("lineColor","LineColor",0,-999) ;
  pc.defineInt("lineStyle","LineStyle",0,-999) ;
  pc.defineInt("lineWidth","LineWidth",0,-999) ;
  pc.defineInt("fillColor","FillColor",0,-999) ;
  pc.defineInt("fillStyle","FillStyle",0,-999) ;
  pc.defineString("curveName","Name",0,"") ;
  pc.defineInt("curveInvisible","Invisible",0,0) ;
  pc.defineInt("showProg","ShowProgress",0,0) ;
  pc.defineInt("numCPU","NumCPU",0,1) ;
  pc.defineInt("interleave","NumCPU",1,0) ;
  pc.defineString("addToCurveName","AddTo",0,"") ;
  pc.defineDouble("addToWgtSelf","AddTo",0,1.) ;
  pc.defineDouble("addToWgtOther","AddTo",1,1.) ;
  pc.defineInt("moveToBack","MoveToBack",0,0) ;
  pc.defineMutex("SliceVars","Project") ;
  pc.defineMutex("AddTo","Asymmetry") ;
  pc.defineMutex("Range","RangeWithName") ;
  pc.defineMutex("VisualizeError","VisualizeErrorData") ;

  // Process & check varargs
  pc.process(argList) ;
  if (!pc.ok(true)) {
    return frame ;
  }

  PlotOpt o ;
  TString drawOpt(pc.getString("drawOption"));

  RooFitResult* errFR = (RooFitResult*) pc.getObject("errorFR") ;
  double errZ = pc.getDouble("errorZ") ;
  RooArgSet* errPars = pc.getSet("errorPars") ;
  bool linMethod = pc.getInt("linearMethod") ;
  if (!drawOpt.Contains("P") && errFR) {
    return plotOnWithErrorBand(frame,*errFR,errZ,errPars,argList,linMethod) ;
  } else {
    o.errorFR = errFR;
  }

  // Extract values from named arguments
  o.numee       = pc.getInt("numee") ;
  o.drawOptions = drawOpt.Data();
  o.curveNameSuffix = pc.getString("curveNameSuffix") ;
  o.scaleFactor = pc.getDouble("scaleFactor") ;
  o.stype = (ScaleType) pc.getInt("scaleType")  ;
  o.projData = (const RooAbsData*) pc.getObject("projData") ;
  o.binProjData = pc.getInt("binProjData") ;
  o.projDataSet = pc.getSet("projDataSet");
  o.numCPU = pc.getInt("numCPU") ;
  o.interleave = (RooFit::MPSplit) pc.getInt("interleave") ;
  o.eeval      = pc.getDouble("evalErrorVal") ;
  o.doeeval   = pc.getInt("doEvalError") ;

  const RooArgSet* sliceSetTmp = pc.getSet("sliceSet");
  std::unique_ptr<RooArgSet> sliceSet{sliceSetTmp ? static_cast<RooArgSet*>(sliceSetTmp->Clone()) : nullptr};
  const RooArgSet* projSet = pc.getSet("projSet") ;
  const RooAbsCategoryLValue* asymCat = (const RooAbsCategoryLValue*) pc.getObject("asymCat") ;


  // Look for category slice arguments and add them to the master slice std::list if found
  const char* sliceCatState = pc.getString("sliceCatState",0,true) ;
  const RooLinkedList& sliceCatList = pc.getObjectList("sliceCatList") ;
  if (sliceCatState) {

    // Make the master slice set if it doesnt exist
    if (!sliceSet) {
      sliceSet = std::make_unique<RooArgSet>();
    }

    // Loop over all categories provided by (multiple) Slice() arguments
    auto catTokens = ROOT::Split(sliceCatState, ",");
    auto iter = sliceCatList.begin();
    for (unsigned int i=0; i < catTokens.size(); ++i) {
      if (auto scat = static_cast<RooCategory*>(*iter)) {
        // Set the slice position to the value indicate by slabel
        scat->setLabel(catTokens[i]) ;
        // Add the slice category to the master slice set
        sliceSet->add(*scat,false) ;
      }
      ++iter;
    }
  }

  o.precision = pc.getDouble("precision") ;
  o.shiftToZero = (pc.getInt("shiftToZero")!=0) ;
  Int_t vlines = pc.getInt("VLines");
  if (pc.hasProcessed("Range")) {
    o.rangeLo = pc.getDouble("rangeLo") ;
    o.rangeHi = pc.getDouble("rangeHi") ;
    o.postRangeFracScale = pc.getInt("rangeAdjustNorm") ;
    if (vlines==2) vlines=0 ; // Default is NoWings if range was specified
  } else if (pc.hasProcessed("RangeWithName")) {
    o.normRangeName = pc.getString("rangeName",0,true) ;
    o.rangeLo = frame->getPlotVar()->getMin(pc.getString("rangeName",0,true)) ;
    o.rangeHi = frame->getPlotVar()->getMax(pc.getString("rangeName",0,true)) ;
    o.postRangeFracScale = pc.getInt("rangeWNAdjustNorm") ;
    if (vlines==2) vlines=0 ; // Default is NoWings if range was specified
  }


  // If separate normalization range was specified this overrides previous settings
  if (pc.hasProcessed("NormRange")) {
    o.normRangeName = pc.getString("normRangeName") ;
    o.postRangeFracScale = true ;
  }

  o.wmode = (vlines==2)?RooCurve::Extended:(vlines==1?RooCurve::Straight:RooCurve::NoWings) ;
  o.projectionRangeName = pc.getString("projectionRangeName",0,true) ;
  o.curveName = pc.getString("curveName",0,true) ;
  o.curveInvisible = pc.getInt("curveInvisible") ;
  o.progress = pc.getInt("showProg") ;
  o.addToCurveName = pc.getString("addToCurveName",0,true) ;
  o.addToWgtSelf = pc.getDouble("addToWgtSelf") ;
  o.addToWgtOther = pc.getDouble("addToWgtOther") ;

  if (o.addToCurveName && !frame->findObject(o.addToCurveName,RooCurve::Class())) {
    coutE(InputArguments) << "RooAbsReal::plotOn(" << GetName() << ") cannot find existing curve " << o.addToCurveName << " to add to in RooPlot" << std::endl ;
    return frame ;
  }

  RooArgSet projectedVars ;
  if (sliceSet) {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") Preprocessing: have slice " << *sliceSet << std::endl ;

    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,true) ;

    // Take out the sliced variables
    for (const auto sliceArg : *sliceSet) {
      RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
      if (arg) {
        projectedVars.remove(*arg) ;
      } else {
        coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") slice variable "
            << sliceArg->GetName() << " was not projected anyway" << std::endl ;
      }
    }
  } else if (projSet) {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") Preprocessing: have projSet " << *projSet << std::endl ;
    makeProjectionSet(frame->getPlotVar(),projSet,projectedVars,false) ;
  } else {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") Preprocessing: have neither sliceSet nor projSet " << std::endl ;
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,true) ;
  }
  o.projSet = &projectedVars ;

  cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") Preprocessing: projectedVars = " << projectedVars << std::endl ;


  RooPlot* ret ;
  if (!asymCat) {
    // Forward to actual calculation
    ret = RooAbsReal::plotOn(frame,o) ;
  } else {
    // Forward to actual calculation
    ret = RooAbsReal::plotAsymOn(frame,*asymCat,o) ;
  }

  // Optionally adjust line/fill attributes
  Int_t lineColor = pc.getInt("lineColor") ;
  Int_t lineStyle = pc.getInt("lineStyle") ;
  Int_t lineWidth = pc.getInt("lineWidth") ;
  Int_t markerColor = pc.getInt("markerColor") ;
  Int_t markerStyle = pc.getInt("markerStyle") ;
  Size_t markerSize  = pc.getDouble("markerSize") ;
  Int_t fillColor = pc.getInt("fillColor") ;
  Int_t fillStyle = pc.getInt("fillStyle") ;
  if (lineColor!=-999) ret->getAttLine()->SetLineColor(lineColor) ;
  if (lineStyle!=-999) ret->getAttLine()->SetLineStyle(lineStyle) ;
  if (lineWidth!=-999) ret->getAttLine()->SetLineWidth(lineWidth) ;
  if (fillColor!=-999) ret->getAttFill()->SetFillColor(fillColor) ;
  if (fillStyle!=-999) ret->getAttFill()->SetFillStyle(fillStyle) ;
  if (markerColor!=-999) ret->getAttMarker()->SetMarkerColor(markerColor) ;
  if (markerStyle!=-999) ret->getAttMarker()->SetMarkerStyle(markerStyle) ;
  if (markerSize!=-999) ret->getAttMarker()->SetMarkerSize(markerSize) ;

  if ((fillColor != -999 || fillStyle != -999) && !drawOpt.Contains("F")) {
    coutW(Plotting) << "Fill color or style was set for plotting \"" << GetName()
        << "\", but these only have an effect when 'DrawOption(\"F\")' for fill is used at the same time." << std::endl;
  }

  // Move last inserted object to back to drawing stack if requested
  if (pc.getInt("moveToBack") && frame->numItems()>1) {
    frame->drawBefore(frame->getObject(0)->GetName(), frame->getCurve()->GetName());
  }

  return ret ;
}



/// Plotting engine function for internal use
///
/// Plot ourselves on given frame. If frame contains a histogram, all dimensions of the plotted
/// function that occur in the previously plotted dataset are projected via partial integration,
/// otherwise no projections are performed. Optionally, certain projections can be performed
/// by summing over the values present in a provided dataset ('projData'), to correctly
/// project out data dependents that are not properly described by the PDF (e.g. per-event errors).
///
/// The functions value can be multiplied with an optional scale factor. The interpretation
/// of the scale factor is unique for generic real functions, for PDFs there are various interpretations
/// possible, which can be selection with 'stype' (see RooAbsPdf::plotOn() for details).
///
/// The default projection behaviour can be overriden by supplying an optional set of dependents
/// to project. For most cases, plotSliceOn() and plotProjOn() provide a more intuitive interface
/// to modify the default projection behaviour.
//_____________________________________________________________________________
// coverity[PASS_BY_VALUE]
RooPlot* RooAbsReal::plotOn(RooPlot *frame, PlotOpt o) const
{


  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // ProjDataVars is either all projData observables, or the user indicated subset of it
  RooArgSet projDataVars ;
  if (o.projData) {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") have ProjData with observables = " << *o.projData->get() << std::endl ;
    if (o.projDataSet) {
      std::unique_ptr<RooArgSet> tmp{static_cast<RooArgSet*>(o.projData->get()->selectCommon(*o.projDataSet))};
      projDataVars.add(*tmp) ;
      cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") have ProjDataSet = " << *o.projDataSet << " will only use this subset of projData" << std::endl ;
    } else {
      cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") using full ProjData" << std::endl ;
      projDataVars.add(*o.projData->get()) ;
    }
  }

  cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") ProjDataVars = " << projDataVars << std::endl ;

  // Make std::list of variables to be projected
  RooArgSet projectedVars ;
  RooArgSet sliceSet ;
  if (o.projSet) {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") have input projSet = " << *o.projSet << std::endl ;
    makeProjectionSet(frame->getPlotVar(),o.projSet,projectedVars,false) ;
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") calculated projectedVars = " << *o.projSet << std::endl ;

    // Print std::list of non-projected variables
    if (frame->getNormVars()) {
      RooArgSet sliceSetTmp;
      getObservables(frame->getNormVars(), sliceSetTmp) ;

      cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") frame->getNormVars() that are also observables = " << sliceSetTmp << std::endl ;

      sliceSetTmp.remove(projectedVars,true,true) ;
      sliceSetTmp.remove(*frame->getPlotVar(),true,true) ;

      if (o.projData) {
        std::unique_ptr<RooArgSet> tmp{static_cast<RooArgSet*>(projDataVars.selectCommon(*o.projSet))};
        sliceSetTmp.remove(*tmp,true,true) ;
      }

      if (!sliceSetTmp.empty()) {
   coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot on "
         << frame->getPlotVar()->GetName() << " represents a slice in " << sliceSetTmp << std::endl ;
      }
      sliceSet.add(sliceSetTmp) ;
    }
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,true) ;
  }

  cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") projectedVars = " << projectedVars << " sliceSet = " << sliceSet << std::endl ;


  RooArgSet* projDataNeededVars = 0 ;
  // Take out data-projected dependents from projectedVars
  if (o.projData) {
    projDataNeededVars = (RooArgSet*) projectedVars.selectCommon(projDataVars) ;
    projectedVars.remove(projDataVars,true,true) ;
  }

  // Get the plot variable and remember its original value
  auto* plotVar = static_cast<RooRealVar*>(frame->getPlotVar());
  double oldPlotVarVal = plotVar->getVal();

  // Inform user about projections
  if (projectedVars.getSize()) {
    coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName()
          << " integrates over variables " << projectedVars
          << (o.projectionRangeName?Form(" in range %s",o.projectionRangeName):"") << std::endl;
  }
  if (projDataNeededVars && projDataNeededVars->getSize()>0) {
    coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName()
          << " averages using data variables " << *projDataNeededVars << std::endl ;
  }

  // Create projection integral
  RooArgSet* projectionCompList = 0 ;

  RooArgSet deps;
  getObservables(frame->getNormVars(), deps) ;
  deps.remove(projectedVars,true,true) ;
  if (projDataNeededVars) {
    deps.remove(*projDataNeededVars,true,true) ;
  }
  deps.remove(*plotVar,true,true) ;
  deps.add(*plotVar) ;

  // Now that we have the final set of dependents, call checkObservables()

  // WVE take out conditional observables
  if (checkObservables(&deps)) {
    coutE(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") error in checkObservables, abort" << std::endl ;
    if (projDataNeededVars) delete projDataNeededVars ;
    return frame ;
  }

  RooAbsReal *projection = (RooAbsReal*) createPlotProjection(deps, &projectedVars, projectionCompList, o.projectionRangeName) ;
  cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot projection object is " << projection->GetName() << std::endl ;
  if (dologD(Plotting)) {
    projection->printStream(ccoutD(Plotting),0,kVerbose) ;
  }

  // Always fix RooAddPdf normalizations
  RooArgSet fullNormSet(deps) ;
  fullNormSet.add(projectedVars) ;
  if (projDataNeededVars && projDataNeededVars->getSize()>0) {
    fullNormSet.add(*projDataNeededVars) ;
  }

  std::unique_ptr<RooArgSet> projectionComponents(projection->getComponents());
  for(auto * pdf : dynamic_range_cast<RooAbsPdf*>(*projectionComponents)) {
    if (pdf) {
      pdf->selectNormalization(&fullNormSet) ;
    }
  }

  // Apply data projection, if requested
  if (o.projData && projDataNeededVars && projDataNeededVars->getSize()>0) {

    // If data set contains more rows than needed, make reduced copy first
    RooAbsData* projDataSel = (RooAbsData*)o.projData;

    if (projDataNeededVars->getSize()<o.projData->get()->getSize()) {

      // Determine if there are any slice variables in the projection set
        std::unique_ptr<RooArgSet> sliceDataSet{static_cast<RooArgSet*>(sliceSet.selectCommon(*o.projData->get()))};
      TString cutString ;
      if (!sliceDataSet->empty()) {
   bool first(true) ;
   for(RooAbsArg * sliceVar : *sliceDataSet) {
     if (!first) {
       cutString.Append("&&") ;
     } else {
       first=false ;
     }

     RooAbsRealLValue* real ;
     RooAbsCategoryLValue* cat ;
     if ((real = dynamic_cast<RooAbsRealLValue*>(sliceVar))) {
       cutString.Append(Form("%s==%f",real->GetName(),real->getVal())) ;
     } else if ((cat = dynamic_cast<RooAbsCategoryLValue*>(sliceVar))) {
       cutString.Append(Form("%s==%d",cat->GetName(),cat->getCurrentIndex())) ;
     }
   }
      }

      if (!cutString.IsNull()) {
   projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars,cutString) ;
   coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") reducing given projection dataset to entries with " << cutString << std::endl ;
      } else {
   projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars) ;
      }
      coutI(Plotting) << "RooAbsReal::plotOn(" << GetName()
            << ") only the following components of the projection data will be used: " << *projDataNeededVars << std::endl ;
    }

    // Request binning of unbinned projection dataset that consists exclusively of category observables
    if (!o.binProjData && dynamic_cast<RooDataSet*>(projDataSel)!=0) {

      // Determine if dataset contains only categories
      bool allCat(true) ;
      for(RooAbsArg * arg2 : *projDataSel->get()) {
   if (!dynamic_cast<RooCategory*>(arg2)) allCat = false ;
      }
      if (allCat) {
   o.binProjData = true ;
   coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") unbinned projection dataset consist only of discrete variables,"
         << " performing projection with binned copy for optimization." << std::endl ;

      }
    }

    // Bin projection dataset if requested
    if (o.binProjData) {
      RooAbsData* tmp = new RooDataHist(Form("%s_binned",projDataSel->GetName()),"Binned projection data",*projDataSel->get(),*projDataSel) ;
      if (projDataSel!=o.projData) delete projDataSel ;
      projDataSel = tmp ;
    }



    // Attach dataset
    projection->getVal(projDataSel->get()) ;
    projection->attachDataSet(*projDataSel) ;

    // Construct optimized data weighted average
    RooAbsTestStatistic::Configuration cfg;
    cfg.nCPU = o.numCPU;
    cfg.interleave = o.interleave;
    RooDataWeightedAverage dwa(Form("%sDataWgtAvg",GetName()),"Data Weighted average",*projection,*projDataSel,RooArgSet()/**projDataSel->get()*/,
            std::move(cfg), true) ;
    //RooDataWeightedAverage dwa(Form("%sDataWgtAvg",GetName()),"Data Weighted average",*projection,*projDataSel,*projDataSel->get(),o.numCPU,o.interleave,true) ;

    // Do _not_ activate cache-and-track as necessary information to define normalization observables are not present in the underlying dataset
    dwa.constOptimizeTestStatistic(Activate,false) ;

    RooRealBinding projBind(dwa,*plotVar) ;
    RooScaledFunc scaleBind(projBind,o.scaleFactor);

    // Set default range, if not specified
    if (o.rangeLo==0 && o.rangeHi==0) {
      o.rangeLo = frame->GetXaxis()->GetXmin() ;
      o.rangeHi = frame->GetXaxis()->GetXmax() ;
    }

    // Construct name of curve for data weighed average
    TString curveName(projection->GetName()) ;
    curveName.Append(Form("_DataAvg[%s]",projDataSel->get()->contentsString().c_str())) ;
    // Append slice set specification if any
    if (sliceSet.getSize()>0) {
      curveName.Append(Form("_Slice[%s]",sliceSet.contentsString().c_str())) ;
    }
    // Append any suffixes imported from RooAbsPdf::plotOn
    if (o.curveNameSuffix) {
      curveName.Append(o.curveNameSuffix) ;
    }

    // Curve constructor for data weighted average
    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
    RooCurve *curve = new RooCurve(projection->GetName(),projection->GetTitle(),scaleBind,
               o.rangeLo,o.rangeHi,frame->GetNbinsX(),o.precision,o.precision,o.shiftToZero,o.wmode,o.numee,o.doeeval,o.eeval) ;
    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;

    curve->SetName(curveName.Data()) ;

    // Add self to other curve if requested
    if (o.addToCurveName) {
      RooCurve* otherCurve = static_cast<RooCurve*>(frame->findObject(o.addToCurveName,RooCurve::Class())) ;

      // Curve constructor for sum of curves
      RooCurve* sumCurve = new RooCurve(projection->GetName(),projection->GetTitle(),*curve,*otherCurve,o.addToWgtSelf,o.addToWgtOther) ;
      sumCurve->SetName(Form("%s_PLUS_%s",curve->GetName(),otherCurve->GetName())) ;
      delete curve ;
      curve = sumCurve ;

    }

    if (o.curveName) {
      curve->SetName(o.curveName) ;
    }

    // add this new curve to the specified plot frame
    frame->addPlotable(curve, o.drawOptions, o.curveInvisible);

    if (projDataSel!=o.projData) delete projDataSel ;

  } else {

    // Set default range, if not specified
    if (o.rangeLo==0 && o.rangeHi==0) {
      o.rangeLo = frame->GetXaxis()->GetXmin() ;
      o.rangeHi = frame->GetXaxis()->GetXmax() ;
    }

    // Calculate a posteriori range fraction scaling if requested (2nd part of normalization correction for
    // result fit on subrange of data)
    if (o.postRangeFracScale) {
      if (!o.normRangeName) {
   o.normRangeName = "plotRange" ;
   plotVar->setRange("plotRange",o.rangeLo,o.rangeHi) ;
      }

      // Evaluate fractional correction integral always on full p.d.f, not component.
      GlobalSelectComponentRAII selectCompRAII(true);
      std::unique_ptr<RooAbsReal> intFrac{projection->createIntegral(*plotVar,*plotVar,o.normRangeName)};
      _globalSelectComp = true; //It's unclear why this is done a second time. Maybe unnecessary.
      if(o.stype != RooAbsReal::Raw || this->InheritsFrom(RooAbsPdf::Class())){
        // this scaling should only be !=1  when plotting partial ranges
        // still, raw means raw
        o.scaleFactor /= intFrac->getVal() ;
      }
    }

    // create a new curve of our function using the clone to do the evaluations
    // Curve constructor for regular projections

    // Set default name of curve
    TString curveName(projection->GetName()) ;
    if (sliceSet.getSize()>0) {
      curveName.Append(Form("_Slice[%s]",sliceSet.contentsString().c_str())) ;
    }
    if (o.curveNameSuffix) {
      // Append any suffixes imported from RooAbsPdf::plotOn
      curveName.Append(o.curveNameSuffix) ;
    }

    TString opt(o.drawOptions);
    if(opt.Contains("P")){
      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
      RooHist *graph= new RooHist(*projection,*plotVar,1.,o.scaleFactor,frame->getNormVars(),o.errorFR);
      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;

      // Override name of curve by user name, if specified
      if (o.curveName) {
        graph->SetName(o.curveName) ;
      }

      // add this new curve to the specified plot frame
      frame->addPlotable(graph, o.drawOptions, o.curveInvisible);
    } else {
      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
      RooCurve *curve = new RooCurve(*projection,*plotVar,o.rangeLo,o.rangeHi,frame->GetNbinsX(),
                                     o.scaleFactor,0,o.precision,o.precision,o.shiftToZero,o.wmode,o.numee,o.doeeval,o.eeval,o.progress);
      RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;
      curve->SetName(curveName.Data()) ;

      // Add self to other curve if requested
      if (o.addToCurveName) {
        RooCurve* otherCurve = static_cast<RooCurve*>(frame->findObject(o.addToCurveName,RooCurve::Class())) ;
        RooCurve* sumCurve = new RooCurve(projection->GetName(),projection->GetTitle(),*curve,*otherCurve,o.addToWgtSelf,o.addToWgtOther) ;
        sumCurve->SetName(Form("%s_PLUS_%s",curve->GetName(),otherCurve->GetName())) ;
        delete curve ;
        curve = sumCurve ;
      }

      // Override name of curve by user name, if specified
      if (o.curveName) {
        curve->SetName(o.curveName) ;
      }

      // add this new curve to the specified plot frame
      frame->addPlotable(curve, o.drawOptions, o.curveInvisible);
    }
  }

  if (projDataNeededVars) delete projDataNeededVars ;
  delete projectionCompList ;
  plotVar->setVal(oldPlotVarVal); // reset the plot variable value to not disturb the original state
  return frame;
}




////////////////////////////////////////////////////////////////////////////////
/// \deprecated OBSOLETE -- RETAINED FOR BACKWARD COMPATIBILITY. Use plotOn() with Slice() instead

RooPlot* RooAbsReal::plotSliceOn(RooPlot *frame, const RooArgSet& sliceSet, Option_t* drawOptions,
             double scaleFactor, ScaleType stype, const RooAbsData* projData) const
{
  RooArgSet projectedVars ;
  makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,true) ;

  // Take out the sliced variables
  for(RooAbsArg * sliceArg : sliceSet) {
    RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
    if (arg) {
      projectedVars.remove(*arg) ;
    } else {
      coutI(Plotting) << "RooAbsReal::plotSliceOn(" << GetName() << ") slice variable "
            << sliceArg->GetName() << " was not projected anyway" << std::endl ;
    }
  }

  PlotOpt o ;
  o.drawOptions = drawOptions ;
  o.scaleFactor = scaleFactor ;
  o.stype = stype ;
  o.projData = projData ;
  o.projSet = &projectedVars ;
  return plotOn(frame,o) ;
}




//_____________________________________________________________________________
// coverity[PASS_BY_VALUE]
RooPlot* RooAbsReal::plotAsymOn(RooPlot *frame, const RooAbsCategoryLValue& asymCat, PlotOpt o) const

{
  // Plotting engine for asymmetries. Implements the functionality if plotOn(frame,Asymmetry(...)))
  //
  // Plot asymmetry of ourselves, defined as
  //
  //   asym = f(asymCat=-1) - f(asymCat=+1) / ( f(asymCat=-1) + f(asymCat=+1) )
  //
  // on frame. If frame contains a histogram, all dimensions of the plotted
  // asymmetry function that occur in the previously plotted dataset are projected via partial integration.
  // Otherwise no projections are performed,
  //
  // The asymmetry function can be multiplied with an optional scale factor. The default projection
  // behaviour can be overriden by supplying an optional set of dependents to project.

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // ProjDataVars is either all projData observables, or the user indicated subset of it
  RooArgSet projDataVars ;
  if (o.projData) {
    if (o.projDataSet) {
      std::unique_ptr<RooArgSet> tmp{static_cast<RooArgSet*>(o.projData->get()->selectCommon(*o.projDataSet))};
      projDataVars.add(*tmp) ;
    } else {
      projDataVars.add(*o.projData->get()) ;
    }
  }

  // Must depend on asymCat
  if (!dependsOn(asymCat)) {
    coutE(Plotting) << "RooAbsReal::plotAsymOn(" << GetName()
          << ") function doesn't depend on asymmetry category " << asymCat.GetName() << std::endl ;
    return frame ;
  }

  // asymCat must be a signCat
  if (!asymCat.isSignType()) {
    coutE(Plotting) << "RooAbsReal::plotAsymOn(" << GetName()
          << ") asymmetry category must have 2 or 3 states with index values -1,0,1" << std::endl ;
    return frame ;
  }

  // Make std::list of variables to be projected
  RooArgSet projectedVars ;
  RooArgSet sliceSet ;
  if (o.projSet) {
    makeProjectionSet(frame->getPlotVar(),o.projSet,projectedVars,false) ;

    // Print std::list of non-projected variables
    if (frame->getNormVars()) {
      RooArgSet sliceSetTmp;
      getObservables(frame->getNormVars(), sliceSetTmp) ;
      sliceSetTmp.remove(projectedVars,true,true) ;
      sliceSetTmp.remove(*frame->getPlotVar(),true,true) ;

      if (o.projData) {
        std::unique_ptr<RooArgSet> tmp{static_cast<RooArgSet*>(projDataVars.selectCommon(*o.projSet))};
        sliceSetTmp.remove(*tmp,true,true) ;
      }

      if (!sliceSetTmp.empty()) {
   coutI(Plotting) << "RooAbsReal::plotAsymOn(" << GetName() << ") plot on "
         << frame->getPlotVar()->GetName() << " represents a slice in " << sliceSetTmp << std::endl ;
      }
      sliceSet.add(sliceSetTmp) ;
    }
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,true) ;
  }


  // Take out data-projected dependens from projectedVars
  RooArgSet* projDataNeededVars = 0 ;
  if (o.projData) {
    projDataNeededVars = (RooArgSet*) projectedVars.selectCommon(projDataVars) ;
    projectedVars.remove(projDataVars,true,true) ;
  }

  // Take out plotted asymmetry from projection
  if (projectedVars.find(asymCat.GetName())) {
    projectedVars.remove(*projectedVars.find(asymCat.GetName())) ;
  }

  // Clone the plot variable
  RooAbsReal* realVar = (RooRealVar*) frame->getPlotVar() ;
  RooRealVar* plotVar = (RooRealVar*) realVar->Clone() ;

  // Inform user about projections
  if (projectedVars.getSize()) {
    coutI(Plotting) << "RooAbsReal::plotAsymOn(" << GetName() << ") plot on " << plotVar->GetName()
          << " projects variables " << projectedVars << std::endl ;
  }
  if (projDataNeededVars && projDataNeededVars->getSize()>0) {
    coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName()
          << " averages using data variables "<<  *projDataNeededVars << std::endl ;
  }


  // Customize two copies of projection with fixed negative and positive asymmetry
  std::unique_ptr<RooAbsCategoryLValue> asymPos{static_cast<RooAbsCategoryLValue*>(asymCat.Clone("asym_pos"))};
  std::unique_ptr<RooAbsCategoryLValue> asymNeg{static_cast<RooAbsCategoryLValue*>(asymCat.Clone("asym_neg"))};
  asymPos->setIndex(1) ;
  asymNeg->setIndex(-1) ;
  RooCustomizer custPos{*this,"pos"};
  RooCustomizer custNeg{*this,"neg"};
  //custPos->setOwning(true) ;
  //custNeg->setOwning(true) ;
  custPos.replaceArg(asymCat,*asymPos) ;
  custNeg.replaceArg(asymCat,*asymNeg) ;
  std::unique_ptr<RooAbsReal> funcPos{static_cast<RooAbsReal*>(custPos.build())};
  std::unique_ptr<RooAbsReal> funcNeg{static_cast<RooAbsReal*>(custNeg.build())};

  // Create projection integral
  RooArgSet *posProjCompList, *negProjCompList ;

  // Add projDataVars to normalized dependents of projection
  // This is needed only for asymmetries (why?)
  RooArgSet depPos(*plotVar,*asymPos) ;
  RooArgSet depNeg(*plotVar,*asymNeg) ;
  depPos.add(projDataVars) ;
  depNeg.add(projDataVars) ;

  const RooAbsReal *posProj = funcPos->createPlotProjection(depPos, &projectedVars, posProjCompList, o.projectionRangeName) ;
  const RooAbsReal *negProj = funcNeg->createPlotProjection(depNeg, &projectedVars, negProjCompList, o.projectionRangeName) ;
  if (!posProj || !negProj) {
    coutE(Plotting) << "RooAbsReal::plotAsymOn(" << GetName() << ") Unable to create projections, abort" << std::endl ;
    return frame ;
  }

  // Create a RooFormulaVar representing the asymmetry
  TString asymName(GetName()) ;
  asymName.Append("_Asym[") ;
  asymName.Append(asymCat.GetName()) ;
  asymName.Append("]") ;
  TString asymTitle(asymCat.GetName()) ;
  asymTitle.Append(" Asymmetry of ") ;
  asymTitle.Append(GetTitle()) ;
  RooFormulaVar funcAsym{asymName,asymTitle,"(@0-@1)/(@0+@1)",RooArgSet(*posProj,*negProj)};

  if (o.projData) {

    // If data set contains more rows than needed, make reduced copy first
    RooAbsData* projDataSel = (RooAbsData*)o.projData;
    if (projDataNeededVars && projDataNeededVars->getSize()<o.projData->get()->getSize()) {

      // Determine if there are any slice variables in the projection set
      RooArgSet* sliceDataSet = (RooArgSet*) sliceSet.selectCommon(*o.projData->get()) ;
      TString cutString ;
      if (!sliceDataSet->empty()) {
   bool first(true) ;
   for(RooAbsArg * sliceVar : *sliceDataSet) {
     if (!first) {
       cutString.Append("&&") ;
     } else {
       first=false ;
     }

     RooAbsRealLValue* real ;
     RooAbsCategoryLValue* cat ;
     if ((real = dynamic_cast<RooAbsRealLValue*>(sliceVar))) {
       cutString.Append(Form("%s==%f",real->GetName(),real->getVal())) ;
     } else if ((cat = dynamic_cast<RooAbsCategoryLValue*>(sliceVar))) {
       cutString.Append(Form("%s==%d",cat->GetName(),cat->getCurrentIndex())) ;
     }
   }
      }
      delete sliceDataSet ;

      if (!cutString.IsNull()) {
   projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars,cutString) ;
   coutI(Plotting) << "RooAbsReal::plotAsymOn(" << GetName()
         << ") reducing given projection dataset to entries with " << cutString << std::endl ;
      } else {
   projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars) ;
      }
      coutI(Plotting) << "RooAbsReal::plotAsymOn(" << GetName()
            << ") only the following components of the projection data will be used: " << *projDataNeededVars << std::endl ;
    }


    RooAbsTestStatistic::Configuration cfg;
    cfg.nCPU = o.numCPU;
    cfg.interleave = o.interleave;
    RooDataWeightedAverage dwa(Form("%sDataWgtAvg",GetName()),"Data Weighted average",funcAsym,*projDataSel,RooArgSet()/**projDataSel->get()*/,
            std::move(cfg),true) ;
    //RooDataWeightedAverage dwa(Form("%sDataWgtAvg",GetName()),"Data Weighted average",*funcAsym,*projDataSel,*projDataSel->get(),o.numCPU,o.interleave,true) ;
    dwa.constOptimizeTestStatistic(Activate) ;

    RooRealBinding projBind(dwa,*plotVar) ;

    ((RooAbsReal*)posProj)->attachDataSet(*projDataSel) ;
    ((RooAbsReal*)negProj)->attachDataSet(*projDataSel) ;

    RooScaledFunc scaleBind(projBind,o.scaleFactor);

    // Set default range, if not specified
    if (o.rangeLo==0 && o.rangeHi==0) {
      o.rangeLo = frame->GetXaxis()->GetXmin() ;
      o.rangeHi = frame->GetXaxis()->GetXmax() ;
    }

    // Construct name of curve for data weighed average
    TString curveName(funcAsym.GetName()) ;
    curveName.Append(Form("_DataAvg[%s]",projDataSel->get()->contentsString().c_str())) ;
    // Append slice set specification if any
    if (sliceSet.getSize()>0) {
      curveName.Append(Form("_Slice[%s]",sliceSet.contentsString().c_str())) ;
    }
    // Append any suffixes imported from RooAbsPdf::plotOn
    if (o.curveNameSuffix) {
      curveName.Append(o.curveNameSuffix) ;
    }


    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
    RooCurve *curve = new RooCurve(funcAsym.GetName(),funcAsym.GetTitle(),scaleBind,
               o.rangeLo,o.rangeHi,frame->GetNbinsX(),o.precision,o.precision,false,o.wmode,o.numee,o.doeeval,o.eeval) ;
    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;

    dynamic_cast<TAttLine*>(curve)->SetLineColor(2) ;
    // add this new curve to the specified plot frame
    frame->addPlotable(curve, o.drawOptions);

    ccoutW(Eval) << std::endl ;

    if (projDataSel!=o.projData) delete projDataSel ;

  } else {

    // Set default range, if not specified
    if (o.rangeLo==0 && o.rangeHi==0) {
      o.rangeLo = frame->GetXaxis()->GetXmin() ;
      o.rangeHi = frame->GetXaxis()->GetXmax() ;
    }

    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors) ;
    RooCurve* curve= new RooCurve(funcAsym,*plotVar,o.rangeLo,o.rangeHi,frame->GetNbinsX(),
              o.scaleFactor,0,o.precision,o.precision,false,o.wmode,o.numee,o.doeeval,o.eeval);
    RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::PrintErrors) ;

    dynamic_cast<TAttLine*>(curve)->SetLineColor(2) ;


    // Set default name of curve
    TString curveName(funcAsym.GetName()) ;
    if (sliceSet.getSize()>0) {
      curveName.Append(Form("_Slice[%s]",sliceSet.contentsString().c_str())) ;
    }
    if (o.curveNameSuffix) {
      // Append any suffixes imported from RooAbsPdf::plotOn
      curveName.Append(o.curveNameSuffix) ;
    }
    curve->SetName(curveName.Data()) ;

    // add this new curve to the specified plot frame
    frame->addPlotable(curve, o.drawOptions);

  }

  // Cleanup
  delete posProjCompList ;
  delete negProjCompList ;

  delete plotVar ;

  return frame;
}



////////////////////////////////////////////////////////////////////////////////
/// Calculate error on self by *linearly* propagating errors on parameters using the covariance matrix
/// from a fit result.
/// The error is calculated as follows
/// \f[
///     \mathrm{error}^2(x) = F_\mathbf{a}(x) \cdot \mathrm{Cov}(\mathbf{a},\mathbf{a}') \cdot F_{\mathbf{a}'}^{\mathrm{T}}(x)
/// \f]
/// where \f$ F_mathbf{a}(x) = \frac{ f(x, mathbf{a} + \mathrm{d}mathbf{a}) - f(x, mathbf{a} - \mathrm{d}mathbf{a}) }{2} \f$,
/// with \f$ f(x) = \f$ `this` and \f$ \mathrm{d}mathbf{a} \f$ the vector of one-sigma uncertainties of all
/// fit parameters taken from the fit result and
/// \f$ \mathrm{Cov}(mathbf{a},mathbf{a}') \f$ = the covariance matrix from the fit result.
///

double RooAbsReal::getPropagatedError(const RooFitResult &fr, const RooArgSet &nset) const
{
  // Calling getParameters() might be costly, but necessary to get the right
  // parameters in the RooAbsReal. The RooFitResult only stores snapshots.
  RooArgSet allParamsInAbsReal;
  getParameters(&nset, allParamsInAbsReal);

  RooArgList paramList;
  for(auto * rrvFitRes : static_range_cast<RooRealVar*>(fr.floatParsFinal())) {

     auto rrvInAbsReal = static_cast<RooRealVar const*>(allParamsInAbsReal.find(*rrvFitRes));

     // If this RooAbsReal is a RooRealVar in the fit result, we don't need to
     // propagate anything and can just return the error in the fit result
     if(rrvFitRes->namePtr() == namePtr()) return rrvFitRes->getError();

     // Strip out parameters with zero error
     if (rrvFitRes->getError() <= rrvFitRes->getVal() * std::numeric_limits<double>::epsilon()) continue;

     // Ignore parameters in the fit result that this RooAbsReal doesn't depend on
     if(!rrvInAbsReal) continue;

     // Checking for float equality is a bad. We check if the values are
     // negligibly far away from each other, relative to the uncertainty.
     if(std::abs(rrvInAbsReal->getVal() - rrvFitRes->getVal()) > 0.01 * rrvFitRes->getError()) {
        std::stringstream errMsg;
        errMsg << "RooAbsReal::getPropagatedError(): the parameters of the RooAbsReal don't have"
               << " the same values as in the fit result! The logic of getPropagatedError is broken in this case.";

        throw std::runtime_error(errMsg.str());
     }

     paramList.add(*rrvInAbsReal);
  }

  std::vector<double> plusVar;
  std::vector<double> minusVar;
  plusVar.reserve(paramList.size());
  minusVar.reserve(paramList.size());

  // Create std::vector of plus,minus variations for each parameter
  TMatrixDSym V(paramList.size() == fr.floatParsFinal().size() ?
      fr.covarianceMatrix() :
      fr.reducedCovarianceMatrix(paramList)) ;

  for (Int_t ivar=0 ; ivar<paramList.getSize() ; ivar++) {

    auto& rrv = static_cast<RooRealVar&>(paramList[ivar]);

    double cenVal = rrv.getVal() ;
    double errVal = sqrt(V(ivar,ivar)) ;

    // Make Plus variation
    rrv.setVal(cenVal+errVal) ;
    plusVar.push_back(getVal(nset)) ;

    // Make Minus variation
    rrv.setVal(cenVal-errVal) ;
    minusVar.push_back(getVal(nset)) ;

    rrv.setVal(cenVal) ;
  }

  // Re-evaluate this RooAbsReal with the central parameters just to be
  // extra-safe that a call to `getPropagatedError()` doesn't change any state.
  // It should not be necessarry because thanks to the dirty flag propagation
  // the RooAbsReal is re-evaluated anyway the next time getVal() is called.
  // Still there are imaginable corner cases where it would not be triggered,
  // for example if the user changes the RooFit operation more after the error
  // propagation.
  getVal(nset);

  TMatrixDSym C(paramList.getSize()) ;
  std::vector<double> errVec(paramList.getSize()) ;
  for (int i=0 ; i<paramList.getSize() ; i++) {
    errVec[i] = std::sqrt(V(i,i)) ;
    for (int j=i ; j<paramList.getSize() ; j++) {
      C(i,j) = V(i,j) / std::sqrt(V(i,i)*V(j,j));
      C(j,i) = C(i,j) ;
    }
  }

  // Make std::vector of variations
  TVectorD F(plusVar.size()) ;
  for (unsigned int j=0 ; j<plusVar.size() ; j++) {
    F[j] = (plusVar[j]-minusVar[j])/2 ;
  }

  // Calculate error in linear approximation from variations and correlation coefficient
  double sum = F*(C*F) ;

  return sqrt(sum) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Plot function or PDF on frame with support for visualization of the uncertainty encoded in the given fit result fr.
/// \param[in] frame RooPlot to plot on
/// \param[in] fr The RooFitResult, where errors can be extracted
/// \param[in] Z  The desired significance (width) of the error band
/// \param[in] params If non-zero, consider only the subset of the parameters in fr for the error evaluation
/// \param[in] argList Optional `RooCmdArg` that can be applied to a regular plotOn() operation
/// \param[in] linMethod By default (linMethod=true), a linearized error is shown.
/// \return The RooPlot the band was plotted on (for chaining of plotting commands).
///
/// The linearized error is calculated as follows:
/// \f[
///   \mathrm{error}(x) = Z * F_a(x) * \mathrm{Corr}(a,a') * F_{a'}^\mathrm{T}(x),
/// \f]
///
/// where
/// \f[
///     F_a(x) = \frac{ f(x,a+\mathrm{d}a) - f(x,a-\mathrm{d}a) }{2},
/// \f]
/// with \f$ f(x) \f$ the plotted curve and \f$ \mathrm{d}a \f$ taken from the fit result, and
/// \f$ \mathrm{Corr}(a,a') \f$ = the correlation matrix from the fit result, and \f$ Z \f$ = requested signifance (\f$ Z \sigma \f$ band)
///
/// The linear method is fast (required 2*N evaluations of the curve, where N is the number of parameters), but may
/// not be accurate in the presence of strong correlations (~>0.9) and at Z>2 due to linear and Gaussian approximations made
///
/// Alternatively, a more robust error is calculated using a sampling method. In this method a number of curves
/// is calculated with variations of the parameter values, as drawn from a multi-variate Gaussian p.d.f. that is constructed
/// from the fit results covariance matrix. The error(x) is determined by calculating a central interval that capture N% of the variations
/// for each value of x, where N% is controlled by Z (i.e. Z=1 gives N=68%). The number of sampling curves is chosen to be such
/// that at least 30 curves are expected to be outside the N% interval, and is minimally 100 (e.g. Z=1->Ncurve=100, Z=2->Ncurve=659, Z=3->Ncurve=11111)
/// Intervals from the sampling method can be asymmetric, and may perform better in the presence of strong correlations, but may take (much)
/// longer to calculate.

RooPlot* RooAbsReal::plotOnWithErrorBand(RooPlot* frame,const RooFitResult& fr, double Z,const RooArgSet* params, const RooLinkedList& argList, bool linMethod) const
{
  RooLinkedList plotArgListTmp(argList) ;
  RooCmdConfig::stripCmdList(plotArgListTmp,"VisualizeError,MoveToBack") ;

  // Strip any 'internal normalization' arguments from std::list
  RooLinkedList plotArgList ;
  for (auto * cmd : static_range_cast<RooCmdArg*>(plotArgListTmp)) {
    if (std::string("Normalization")==cmd->GetName()) {
      if (((RooCmdArg*)cmd)->getInt(1)!=0) {
      } else {
   plotArgList.Add(cmd) ;
      }
    } else {
      plotArgList.Add(cmd) ;
    }
  }

  // Function to plot a single curve, creating a copy of the plotArgList to
  // pass as plot command arguments. The "FillColor" command is removed because
  // it has no effect on plotting single curves and would cause a warning.
  auto plotFunc = [&](RooAbsReal const& absReal) {
    RooLinkedList tmp(plotArgList) ;
    RooCmdConfig::stripCmdList(tmp, "FillColor");
    absReal.plotOn(frame, tmp);
  };

  // Generate central value curve
  plotFunc(*this);
  RooCurve* cenCurve = frame->getCurve() ;
  if(!cenCurve){
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOnWithErrorBand: no curve for central value available" << std::endl;
    return frame;
  }
  frame->remove(0,false) ;

  RooCurve* band(0) ;
  if (!linMethod) {

    // *** Interval method ***
    //
    // Make N variations of parameters samples from V and visualize N% central interval where N% is defined from Z

    // Clone self for internal use
    RooAbsReal* cloneFunc = (RooAbsReal*) cloneTree() ;
    RooArgSet cloneParams;
    cloneFunc->getObservables(&fr.floatParsFinal(), cloneParams) ;
    RooArgSet errorParams{cloneParams};
    if(params) {
      // clear and fill errorParams only with parameters that both in params and cloneParams
      cloneParams.selectCommon(*params, errorParams);
    }

    // Generate 100 random parameter points distributed according to fit result covariance matrix
    RooAbsPdf* paramPdf = fr.createHessePdf(errorParams) ;
    Int_t n = Int_t(100./TMath::Erfc(Z/sqrt(2.))) ;
    if (n<100) n=100 ;

    coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") INFO: visualizing " << Z << "-sigma uncertainties in parameters "
        << errorParams << " from fit result " << fr.GetName() << " using " << n << " samplings." << std::endl ;

    // Generate variation curves with above set of parameter values
    double ymin = frame->GetMinimum() ;
    double ymax = frame->GetMaximum() ;
    RooDataSet* d = paramPdf->generate(errorParams,n) ;
    std::vector<RooCurve*> cvec ;
    for (int i=0 ; i<d->numEntries() ; i++) {
      cloneParams.assign(*d->get(i)) ;
      plotFunc(*cloneFunc);
      cvec.push_back(frame->getCurve()) ;
      frame->remove(0,false) ;
    }
    frame->SetMinimum(ymin) ;
    frame->SetMaximum(ymax) ;


    // Generate upper and lower curve points from 68% interval around each point of central curve
    band = cenCurve->makeErrorBand(cvec,Z) ;

    // Cleanup
    delete paramPdf ;
    delete cloneFunc ;
    for (std::vector<RooCurve*>::iterator i=cvec.begin() ; i!=cvec.end() ; ++i) {
      delete (*i) ;
    }

  } else {

    // *** Linear Method ***
    //
    // Make a one-sigma up- and down fluctation for each parameter and visualize
    // a from a linearized calculation as follows
    //
    //   error(x) = F(a) C_aa' F(a')
    //
    //   Where F(a) = (f(x,a+da) - f(x,a-da))/2
    //   and C_aa' is the correlation matrix

    // Strip out parameters with zero error
    RooArgList fpf_stripped;
    for (auto const* frv : static_range_cast<RooRealVar*>(fr.floatParsFinal())) {
       if (frv->getError() > frv->getVal() * std::numeric_limits<double>::epsilon()) {
          fpf_stripped.add(*frv);
       }
    }

    // Clone self for internal use
    RooAbsReal* cloneFunc = (RooAbsReal*) cloneTree() ;
    RooArgSet cloneParams;
    cloneFunc->getObservables(&fpf_stripped, cloneParams) ;
    RooArgSet errorParams{cloneParams};
    if(params) {
      // clear and fill errorParams only with parameters that both in params and cloneParams
      cloneParams.selectCommon(*params, errorParams);
    }


    // Make std::list of parameter instances of cloneFunc in order of error matrix
    RooArgList paramList ;
    const RooArgList& fpf = fr.floatParsFinal() ;
    std::vector<int> fpf_idx ;
    for (Int_t i=0 ; i<fpf.getSize() ; i++) {
      RooAbsArg* par = errorParams.find(fpf[i].GetName()) ;
      if (par) {
   paramList.add(*par) ;
   fpf_idx.push_back(i) ;
      }
    }

    std::vector<RooCurve*> plusVar, minusVar ;

    // Create std::vector of plus,minus variations for each parameter

    TMatrixDSym V(paramList.getSize()==fr.floatParsFinal().getSize()?
        fr.covarianceMatrix():
        fr.reducedCovarianceMatrix(paramList)) ;


    for (Int_t ivar=0 ; ivar<paramList.getSize() ; ivar++) {

      RooRealVar& rrv = (RooRealVar&)fpf[fpf_idx[ivar]] ;

      double cenVal = rrv.getVal() ;
      double errVal = sqrt(V(ivar,ivar)) ;

      // Make Plus variation
      ((RooRealVar*)paramList.at(ivar))->setVal(cenVal+Z*errVal) ;


      plotFunc(*cloneFunc);
      plusVar.push_back(frame->getCurve()) ;
      frame->remove(0,false) ;


      // Make Minus variation
      ((RooRealVar*)paramList.at(ivar))->setVal(cenVal-Z*errVal) ;
      plotFunc(*cloneFunc);
      minusVar.push_back(frame->getCurve()) ;
      frame->remove(0,false) ;

      ((RooRealVar*)paramList.at(ivar))->setVal(cenVal) ;
    }

    TMatrixDSym C(paramList.getSize()) ;
    std::vector<double> errVec(paramList.getSize()) ;
    for (int i=0 ; i<paramList.getSize() ; i++) {
      errVec[i] = sqrt(V(i,i)) ;
      for (int j=i ; j<paramList.getSize() ; j++) {
   C(i,j) = V(i,j)/sqrt(V(i,i)*V(j,j)) ;
   C(j,i) = C(i,j) ;
      }
    }

    band = cenCurve->makeErrorBand(plusVar,minusVar,C,Z) ;


    // Cleanup
    delete cloneFunc ;
    for (std::vector<RooCurve*>::iterator i=plusVar.begin() ; i!=plusVar.end() ; ++i) {
      delete (*i) ;
    }
    for (std::vector<RooCurve*>::iterator i=minusVar.begin() ; i!=minusVar.end() ; ++i) {
      delete (*i) ;
    }

  }

  delete cenCurve ;
  if (!band) return frame ;

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsPdf::plotOn(%s)",GetName())) ;
  pc.defineString("drawOption","DrawOption",0,"F") ;
  pc.defineString("curveNameSuffix","CurveNameSuffix",0,"") ;
  pc.defineInt("lineColor","LineColor",0,-999) ;
  pc.defineInt("lineStyle","LineStyle",0,-999) ;
  pc.defineInt("lineWidth","LineWidth",0,-999) ;
  pc.defineInt("markerColor","MarkerColor",0,-999) ;
  pc.defineInt("markerStyle","MarkerStyle",0,-999) ;
  pc.defineDouble("markerSize","MarkerSize",0,-999) ;
  pc.defineInt("fillColor","FillColor",0,-999) ;
  pc.defineInt("fillStyle","FillStyle",0,-999) ;
  pc.defineString("curveName","Name",0,"") ;
  pc.defineInt("curveInvisible","Invisible",0,0) ;
  pc.defineInt("moveToBack","MoveToBack",0,0) ;
  pc.allowUndefined() ;

  // Process & check varargs
  pc.process(argList) ;
  if (!pc.ok(true)) {
    return frame ;
  }

  // Insert error band in plot frame
  frame->addPlotable(band,pc.getString("drawOption"),pc.getInt("curveInvisible")) ;

  // Optionally adjust line/fill attributes
  Int_t lineColor = pc.getInt("lineColor") ;
  Int_t lineStyle = pc.getInt("lineStyle") ;
  Int_t lineWidth = pc.getInt("lineWidth") ;
  Int_t markerColor = pc.getInt("markerColor") ;
  Int_t markerStyle = pc.getInt("markerStyle") ;
  Size_t markerSize  = pc.getDouble("markerSize") ;
  Int_t fillColor = pc.getInt("fillColor") ;
  Int_t fillStyle = pc.getInt("fillStyle") ;
  if (lineColor!=-999) frame->getAttLine()->SetLineColor(lineColor) ;
  if (lineStyle!=-999) frame->getAttLine()->SetLineStyle(lineStyle) ;
  if (lineWidth!=-999) frame->getAttLine()->SetLineWidth(lineWidth) ;
  if (fillColor!=-999) frame->getAttFill()->SetFillColor(fillColor) ;
  if (fillStyle!=-999) frame->getAttFill()->SetFillStyle(fillStyle) ;
  if (markerColor!=-999) frame->getAttMarker()->SetMarkerColor(markerColor) ;
  if (markerStyle!=-999) frame->getAttMarker()->SetMarkerStyle(markerStyle) ;
  if (markerSize!=-999) frame->getAttMarker()->SetMarkerSize(markerSize) ;

  // Adjust name if requested
  if (pc.getString("curveName",0,true)) {
    band->SetName(pc.getString("curveName",0,true)) ;
  } else if (pc.getString("curveNameSuffix",0,true)) {
    TString name(band->GetName()) ;
    name.Append(pc.getString("curveNameSuffix",0,true)) ;
    band->SetName(name.Data()) ;
  }

  // Move last inserted object to back to drawing stack if requested
  if (pc.getInt("moveToBack") && frame->numItems()>1) {
    frame->drawBefore(frame->getObject(0)->GetName(), frame->getCurve()->GetName());
  }


  return frame ;
}




////////////////////////////////////////////////////////////////////////////////
/// Utility function for plotOn(), perform general sanity check on frame to ensure safe plotting operations

bool RooAbsReal::plotSanityChecks(RooPlot* frame) const
{
  // check that we are passed a valid plot frame to use
  if(0 == frame) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOn: frame is null" << std::endl;
    return true;
  }

  // check that this frame knows what variable to plot
  RooAbsReal* var = frame->getPlotVar() ;
  if(!var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
    << ":plotOn: frame does not specify a plot variable" << std::endl;
    return true;
  }

  // check that the plot variable is not derived
  if(!dynamic_cast<RooAbsRealLValue*>(var)) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOn: cannot plot variable \""
          << var->GetName() << "\" of type " << var->ClassName() << std::endl;
    return true;
  }

  // check if we actually depend on the plot variable
  if(!this->dependsOn(*var)) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOn: WARNING: variable is not an explicit dependent: "
          << var->GetName() << std::endl;
  }

  return false ;
}




////////////////////////////////////////////////////////////////////////////////
/// Utility function for plotOn() that constructs the set of
/// observables to project when plotting ourselves as function of
/// 'plotVar'. 'allVars' is the std::list of variables that must be
/// projected, but may contain variables that we do not depend on. If
/// 'silent' is cleared, warnings about inconsistent input parameters
/// will be printed.

void RooAbsReal::makeProjectionSet(const RooAbsArg* plotVar, const RooArgSet* allVars,
               RooArgSet& projectedVars, bool silent) const
{
  cxcoutD(Plotting) << "RooAbsReal::makeProjectionSet(" << GetName() << ") plotVar = " << plotVar->GetName()
          << " allVars = " << (allVars?(*allVars):RooArgSet()) << std::endl ;

  projectedVars.removeAll() ;
  if (!allVars) return ;

  // Start out with suggested std::list of variables
  projectedVars.add(*allVars) ;

  // Take out plot variable
  RooAbsArg *found= projectedVars.find(plotVar->GetName());
  if(found) {
    projectedVars.remove(*found);

    // Take out eventual servers of plotVar
    std::unique_ptr<RooArgSet> plotServers{plotVar->getObservables(&projectedVars)};
    for(RooAbsArg * ps : *plotServers) {
      RooAbsArg* tmp = projectedVars.find(ps->GetName()) ;
      if (tmp) {
   cxcoutD(Plotting) << "RooAbsReal::makeProjectionSet(" << GetName() << ") removing " << tmp->GetName()
           << " from projection set because it a server of " << plotVar->GetName() << std::endl ;
   projectedVars.remove(*tmp) ;
      }
    }

    if (!silent) {
      coutW(Plotting) << "RooAbsReal::plotOn(" << GetName()
            << ") WARNING: cannot project out frame variable ("
            << found->GetName() << "), ignoring" << std::endl ;
    }
  }

  // Take out all non-dependents of function
  for(RooAbsArg * arg : *allVars) {
    if (!dependsOnValue(*arg)) {
      projectedVars.remove(*arg,true) ;

      cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName()
         << ") function doesn't depend on projection variable "
         << arg->GetName() << ", ignoring" << std::endl ;
    }
  }
}




////////////////////////////////////////////////////////////////////////////////
/// If true, the current pdf is a selected component (for use in plotting)

bool RooAbsReal::isSelectedComp() const
{
  return _selectComp || _globalSelectComp ;
}



////////////////////////////////////////////////////////////////////////////////
/// Global switch controlling the activation of the selectComp() functionality

void RooAbsReal::globalSelectComp(bool flag)
{
  _globalSelectComp = flag ;
}




////////////////////////////////////////////////////////////////////////////////
/// Create an interface adaptor f(vars) that binds us to the specified variables
/// (in arbitrary order). For example, calling bindVars({x1,x3}) on an object
/// F(x1,x2,x3,x4) returns an object f(x1,x3) that is evaluated using the
/// current values of x2 and x4. The caller takes ownership of the returned adaptor.

RooAbsFunc *RooAbsReal::bindVars(const RooArgSet &vars, const RooArgSet* nset, bool clipInvalid) const
{
  RooAbsFunc *binding= new RooRealBinding(*this,vars,nset,clipInvalid);
  if(binding && !binding->isValid()) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":bindVars: cannot bind to " << vars << std::endl ;
    delete binding;
    binding= 0;
  }
  return binding;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy the cached value of another RooAbsArg to our cache.
/// Warning: This function just copies the cached values of source,
/// it is the callers responsibility to make sure the cache is clean.

void RooAbsReal::copyCache(const RooAbsArg* source, bool /*valueOnly*/, bool setValDirty)
{
  auto other = static_cast<const RooAbsReal*>(source);
  assert(dynamic_cast<const RooAbsReal*>(source));

  _value = other->_treeReadBuffer ? other->_treeReadBuffer->operator double() : other->_value;

  if (setValDirty) {
    setValueDirty() ;
  }
}


////////////////////////////////////////////////////////////////////////////////

void RooAbsReal::attachToVStore(RooVectorDataStore& vstore)
{
  RooVectorDataStore::RealVector* rv = vstore.addReal(this) ;
  rv->setBuffer(this,&_value) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Attach object to a branch of given TTree. By default it will
/// register the internal value cache RooAbsReal::_value as branch
/// buffer for a double tree branch with the same name as this
/// object. If no double branch is found with the name of this
/// object, this method looks for a Float_t Int_t, UChar_t and UInt_t, etc
/// branch. If any of these are found, a TreeReadBuffer
/// that branch is created, and saved in _treeReadBuffer.
/// TreeReadBuffer::operator double() can be used to convert the values.
/// This is used by copyCache().
void RooAbsReal::attachToTree(TTree& t, Int_t bufSize)
{
  // First determine if branch is taken
  TString cleanName(cleanBranchName()) ;
  TBranch* branch = t.GetBranch(cleanName) ;
  if (branch) {

    // Determine if existing branch is Float_t or double
    TLeaf* leaf = (TLeaf*)branch->GetListOfLeaves()->At(0) ;

    // Check that leaf is _not_ an array
    Int_t dummy ;
    TLeaf* counterLeaf = leaf->GetLeafCounter(dummy) ;
    if (counterLeaf) {
      coutE(Eval) << "RooAbsReal::attachToTree(" << GetName() << ") ERROR: TTree branch " << GetName()
        << " is an array and cannot be attached to a RooAbsReal" << std::endl ;
      return ;
    }

    TString typeName(leaf->GetTypeName()) ;


    // For different type names, store three items:
    // first: A tag attached to this instance. Not used inside RooFit, any more, but users might rely on it.
    // second: A function to attach
    std::map<std::string, std::pair<std::string, std::function<std::unique_ptr<TreeReadBuffer>()>>> typeMap {
      {"Float_t",   {"FLOAT_TREE_BRANCH",            [&](){ return createTreeReadBuffer<Float_t  >(cleanName, t); }}},
      {"Int_t",     {"INTEGER_TREE_BRANCH",          [&](){ return createTreeReadBuffer<Int_t    >(cleanName, t); }}},
      {"UChar_t",   {"BYTE_TREE_BRANCH",             [&](){ return createTreeReadBuffer<UChar_t  >(cleanName, t); }}},
      {"Bool_t",    {"BOOL_TREE_BRANCH",             [&](){ return createTreeReadBuffer<Bool_t   >(cleanName, t); }}},
      {"Char_t",    {"SIGNEDBYTE_TREE_BRANCH",       [&](){ return createTreeReadBuffer<Char_t   >(cleanName, t); }}},
      {"UInt_t",    {"UNSIGNED_INTEGER_TREE_BRANCH", [&](){ return createTreeReadBuffer<UInt_t   >(cleanName, t); }}},
      {"Long64_t",  {"LONG_TREE_BRANCH",             [&](){ return createTreeReadBuffer<Long64_t >(cleanName, t); }}},
      {"ULong64_t", {"UNSIGNED_LONG_TREE_BRANCH",    [&](){ return createTreeReadBuffer<ULong64_t>(cleanName, t); }}},
      {"Short_t",   {"SHORT_TREE_BRANCH",            [&](){ return createTreeReadBuffer<Short_t  >(cleanName, t); }}},
      {"UShort_t",  {"UNSIGNED_SHORT_TREE_BRANCH",   [&](){ return createTreeReadBuffer<UShort_t >(cleanName, t); }}},
    };

    auto typeDetails = typeMap.find(typeName.Data());
    if (typeDetails != typeMap.end()) {
      coutI(DataHandling) << "RooAbsReal::attachToTree(" << GetName() << ") TTree " << typeDetails->first << " branch " << GetName()
                  << " will be converted to double precision." << std::endl ;
      setAttribute(typeDetails->second.first.c_str(), true);
      _treeReadBuffer = typeDetails->second.second();
    } else {
      _treeReadBuffer = nullptr;

      if (!typeName.CompareTo("Double_t")) {
        t.SetBranchAddress(cleanName, &_value);
      }
      else {
        coutE(InputArguments) << "RooAbsReal::attachToTree(" << GetName() << ") data type " << typeName << " is not supported." << std::endl ;
      }
    }
  } else {

    TString format(cleanName);
    format.Append("/D");
    branch = t.Branch(cleanName, &_value, (const Text_t*)format, bufSize);
  }

}



////////////////////////////////////////////////////////////////////////////////
/// Fill the tree branch that associated with this object with its current value

void RooAbsReal::fillTreeBranch(TTree& t)
{
  // First determine if branch is taken
  TBranch* branch = t.GetBranch(cleanBranchName()) ;
  if (!branch) {
    coutE(Eval) << "RooAbsReal::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree: " << cleanBranchName() << std::endl ;
    assert(0) ;
  }
  branch->Fill() ;

}



////////////////////////////////////////////////////////////////////////////////
/// (De)Activate associated tree branch

void RooAbsReal::setTreeBranchStatus(TTree& t, bool active)
{
  TBranch* branch = t.GetBranch(cleanBranchName()) ;
  if (branch) {
    t.SetBranchStatus(cleanBranchName(),active?1:0) ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Create a RooRealVar fundamental object with our properties. The new
/// object will be created without any fit limits.

RooAbsArg *RooAbsReal::createFundamental(const char* newname) const
{
  RooRealVar *fund= new RooRealVar(newname?newname:GetName(),GetTitle(),_value,getUnit());
  fund->removeRange();
  fund->setPlotLabel(getPlotLabel());
  fund->setAttribute("fundamentalCopy");
  return fund;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for use in getAnalyticalIntegral(). If the
/// content of proxy 'a' occurs in set 'allDeps' then the argument
/// held in 'a' is copied from allDeps to analDeps

bool RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps,
               const RooArgProxy& a) const
{
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  bool result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for use in getAnalyticalIntegral(). If the
/// contents of proxies a,b occur in set 'allDeps' then the arguments
/// held in a,b are copied from allDeps to analDeps

bool RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps,
               const RooArgProxy& a, const RooArgProxy& b) const
{
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;
  bool result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for use in getAnalyticalIntegral(). If the
/// contents of proxies a,b,c occur in set 'allDeps' then the arguments
/// held in a,b,c are copied from allDeps to analDeps

bool RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps,
               const RooArgProxy& a, const RooArgProxy& b,
               const RooArgProxy& c) const
{
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;
  nameList.Add(new TObjString(c.absArg()->GetName())) ;
  bool result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for use in getAnalyticalIntegral(). If the
/// contents of proxies a,b,c,d occur in set 'allDeps' then the arguments
/// held in a,b,c,d are copied from allDeps to analDeps

bool RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps,
               const RooArgProxy& a, const RooArgProxy& b,
               const RooArgProxy& c, const RooArgProxy& d) const
{
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;
  nameList.Add(new TObjString(c.absArg()->GetName())) ;
  nameList.Add(new TObjString(d.absArg()->GetName())) ;
  bool result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}


////////////////////////////////////////////////////////////////////////////////
/// Utility function for use in getAnalyticalIntegral(). If the
/// contents of 'refset' occur in set 'allDeps' then the arguments
/// held in 'refset' are copied from allDeps to analDeps.

bool RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps,
              const RooArgSet& refset) const
{
  TList nameList ;
  for(RooAbsArg * arg : refset) {
    nameList.Add(new TObjString(arg->GetName())) ;
  }

  bool result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



////////////////////////////////////////////////////////////////////////////////
/// Check if allArgs contains matching elements for each name in nameList. If it does,
/// add the corresponding args from allArgs to matchedArgs and return true. Otherwise
/// return false and do not change matchedArgs.

bool RooAbsReal::matchArgsByName(const RooArgSet &allArgs, RooArgSet &matchedArgs,
              const TList &nameList) const
{
  RooArgSet matched("matched");
  bool isMatched(true);
  for(auto * name : static_range_cast<TObjString*>(nameList)) {
    RooAbsArg *found= allArgs.find(name->String().Data());
    if(found) {
      matched.add(*found);
    }
    else {
      isMatched= false;
      break;
    }
  }

  // nameList may not contain multiple entries with the same name
  // that are both matched
  if (isMatched && (matched.getSize()!=nameList.GetSize())) {
    isMatched = false ;
  }

  if(isMatched) matchedArgs.add(matched);
  return isMatched;
}



////////////////////////////////////////////////////////////////////////////////
/// Returns the default numeric integration configuration for all RooAbsReals

RooNumIntConfig* RooAbsReal::defaultIntegratorConfig()
{
  return &RooNumIntConfig::defaultConfig() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the specialized integrator configuration for _this_ RooAbsReal.
/// If this object has no specialized configuration, a null pointer is returned.

RooNumIntConfig* RooAbsReal::specialIntegratorConfig() const
{
  return _specIntegratorConfig ;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the specialized integrator configuration for _this_ RooAbsReal.
/// If this object has no specialized configuration, a null pointer is returned,
/// unless createOnTheFly is true in which case a clone of the default integrator
/// configuration is created, installed as specialized configuration, and returned

RooNumIntConfig* RooAbsReal::specialIntegratorConfig(bool createOnTheFly)
{
  if (!_specIntegratorConfig && createOnTheFly) {
    _specIntegratorConfig = new RooNumIntConfig(*defaultIntegratorConfig()) ;
  }
  return _specIntegratorConfig ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the numeric integration configuration used for this object. If
/// a specialized configuration was associated with this object, that configuration
/// is returned, otherwise the default configuration for all RooAbsReals is returned

const RooNumIntConfig* RooAbsReal::getIntegratorConfig() const
{
  const RooNumIntConfig* config = specialIntegratorConfig() ;
  if (config) return config ;
  return defaultIntegratorConfig() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return the numeric integration configuration used for this object. If
/// a specialized configuration was associated with this object, that configuration
/// is returned, otherwise the default configuration for all RooAbsReals is returned

RooNumIntConfig* RooAbsReal::getIntegratorConfig()
{
  RooNumIntConfig* config = specialIntegratorConfig() ;
  if (config) return config ;
  return defaultIntegratorConfig() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set the given integrator configuration as default numeric integration
/// configuration for this object

void RooAbsReal::setIntegratorConfig(const RooNumIntConfig& config)
{
  if (_specIntegratorConfig) {
    delete _specIntegratorConfig ;
  }
  _specIntegratorConfig = new RooNumIntConfig(config) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Remove the specialized numeric integration configuration associated
/// with this object

void RooAbsReal::setIntegratorConfig()
{
  if (_specIntegratorConfig) {
    delete _specIntegratorConfig ;
  }
  _specIntegratorConfig = 0 ;
}




////////////////////////////////////////////////////////////////////////////////
/// Interface function to force use of a given set of observables
/// to interpret function value. Needed for functions or p.d.f.s
/// whose shape depends on the choice of normalization such as
/// RooAddPdf

void RooAbsReal::selectNormalization(const RooArgSet*, bool)
{
}




////////////////////////////////////////////////////////////////////////////////
/// Interface function to force use of a given normalization range
/// to interpret function value. Needed for functions or p.d.f.s
/// whose shape depends on the choice of normalization such as
/// RooAddPdf

void RooAbsReal::selectNormalizationRange(const char*, bool)
{
}



////////////////////////////////////////////////////////////////////////////////
/// Advertise capability to determine maximum value of function for given set of
/// observables. If no direct generator method is provided, this information
/// will assist the accept/reject generator to operate more efficiently as
/// it can skip the initial trial sampling phase to empirically find the function
/// maximum

Int_t RooAbsReal::getMaxVal(const RooArgSet& /*vars*/) const
{
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return maximum value for set of observables identified by code assigned
/// in getMaxVal

double RooAbsReal::maxVal(Int_t /*code*/) const
{
  assert(1) ;
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Interface to insert remote error logging messages received by RooRealMPFE into current error loggin stream

void RooAbsReal::logEvalError(const RooAbsReal* originator, const char* origName, const char* message, const char* serverValueString)
{
  if (_evalErrorMode==Ignore) {
    return ;
  }

  if (_evalErrorMode==CountErrors) {
    _evalErrorCount++ ;
    return ;
  }

  static bool inLogEvalError = false ;

  if (inLogEvalError) {
    return ;
  }
  inLogEvalError = true ;

  EvalError ee ;
  ee.setMessage(message) ;

  if (serverValueString) {
    ee.setServerValues(serverValueString) ;
  }

  if (_evalErrorMode==PrintErrors) {
   oocoutE(nullptr,Eval) << "RooAbsReal::logEvalError(" << "<STATIC>" << ") evaluation error, " << std::endl
         << " origin       : " << origName << std::endl
         << " message      : " << ee._msg << std::endl
         << " server values: " << ee._srvval << std::endl ;
  } else if (_evalErrorMode==CollectErrors) {
    _evalErrorList[originator].first = origName ;
    _evalErrorList[originator].second.push_back(ee) ;
  }


  inLogEvalError = false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Log evaluation error message. Evaluation errors may be routed through a different
/// protocol than generic RooFit warning message (which go straight through RooMsgService)
/// because evaluation errors can occur in very large numbers in the use of likelihood
/// evaluations. In logEvalError mode, controlled by global method enableEvalErrorLogging()
/// messages reported through this function are not printed but all stored in a std::list,
/// along with server values at the time of reporting. Error messages logged in this
/// way can be printed in a structured way, eliminating duplicates and with the ability
/// to truncate the std::list by printEvalErrors. This is the standard mode of error logging
/// during MINUIT operations. If enableEvalErrorLogging() is false, all errors
/// reported through this method are passed for immediate printing through RooMsgService.
/// A std::string with server names and values is constructed automatically for error logging
/// purposes, unless a custom std::string with similar information is passed as argument.

void RooAbsReal::logEvalError(const char* message, const char* serverValueString) const
{
  if (_evalErrorMode==Ignore) {
    return ;
  }

  if (_evalErrorMode==CountErrors) {
    _evalErrorCount++ ;
    return ;
  }

  static bool inLogEvalError = false ;

  if (inLogEvalError) {
    return ;
  }
  inLogEvalError = true ;

  EvalError ee ;
  ee.setMessage(message) ;

  if (serverValueString) {
    ee.setServerValues(serverValueString) ;
  } else {
    std::string srvval ;
    std::ostringstream oss ;
    bool first(true) ;
    for (Int_t i=0 ; i<numProxies() ; i++) {
      RooAbsProxy* p = getProxy(i) ;
      if (!p) continue ;
      //if (p->name()[0]=='!') continue ;
      if (first) {
   first=false ;
      } else {
   oss << ", " ;
      }
      p->print(oss,true) ;
    }
    ee.setServerValues(oss.str().c_str()) ;
  }

  std::ostringstream oss2 ;
  printStream(oss2,kName|kClassName|kArgs,kInline)  ;

  if (_evalErrorMode==PrintErrors) {
   coutE(Eval) << "RooAbsReal::logEvalError(" << GetName() << ") evaluation error, " << std::endl
          << " origin       : " << oss2.str() << std::endl
          << " message      : " << ee._msg << std::endl
          << " server values: " << ee._srvval << std::endl ;
  } else if (_evalErrorMode==CollectErrors) {
    if (_evalErrorList[this].second.size() >= 2048) {
       // avoid overflowing the error std::list, so if there are very many, print
       // the oldest one first, and pop it off the std::list
       const EvalError& oee = _evalErrorList[this].second.front();
       // print to debug stream, since these would normally be suppressed, and
       // we do not want to increase the error count in the message service...
       ccoutD(Eval) << "RooAbsReal::logEvalError(" << GetName()
              << ") delayed evaluation error, " << std::endl
                   << " origin       : " << oss2.str() << std::endl
                   << " message      : " << oee._msg << std::endl
                   << " server values: " << oee._srvval << std::endl ;
       _evalErrorList[this].second.pop_front();
    }
    _evalErrorList[this].first = oss2.str().c_str() ;
    _evalErrorList[this].second.push_back(ee) ;
  }

  inLogEvalError = false ;
  //coutE(Tracing) << "RooAbsReal::logEvalError(" << GetName() << ") message = " << message << std::endl ;
}




////////////////////////////////////////////////////////////////////////////////
/// Clear the stack of evaluation error messages

void RooAbsReal::clearEvalErrorLog()
{
  if (_evalErrorMode==PrintErrors) {
    return ;
  } else if (_evalErrorMode==CollectErrors) {
    _evalErrorList.clear() ;
  } else {
    _evalErrorCount = 0 ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Retrieve bin boundaries if this distribution is binned in `obs`.
/// \param[in] obs Observable to retrieve boundaries for.
/// \param[in] xlo Beginning of range.
/// \param[in] xhi End of range.
/// \return The caller owns the returned std::list.
std::list<double>* RooAbsReal::binBoundaries(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const {
  return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Interface for returning an optional hint for initial sampling points when constructing a curve projected on observable `obs`.
/// \param[in] obs Observable to retrieve sampling hint for.
/// \param[in] xlo Beginning of range.
/// \param[in] xhi End of range.
/// \return The caller owns the returned std::list.
std::list<double>* RooAbsReal::plotSamplingHint(RooAbsRealLValue& /*obs*/, double /*xlo*/, double /*xhi*/) const {
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Print all outstanding logged evaluation error on the given ostream. If maxPerNode
/// is zero, only the number of errors for each source (object with unique name) is std::listed.
/// If maxPerNode is greater than zero, up to maxPerNode detailed error messages are shown
/// per source of errors. A truncation message is shown if there were more errors logged
/// than shown.

void RooAbsReal::printEvalErrors(std::ostream& os, Int_t maxPerNode)
{
  if (_evalErrorMode == CountErrors) {
    os << _evalErrorCount << " errors counted" << std::endl ;
  }

  if (maxPerNode<0) return ;

  for(auto iter = _evalErrorList.begin();iter!=_evalErrorList.end() ; ++iter) {
    if (maxPerNode==0) {

      // Only print node name with total number of errors
      os << iter->second.first ;
      //iter->first->printStream(os,kName|kClassName|kArgs,kInline)  ;
      os << " has " << iter->second.second.size() << " errors" << std::endl ;

    } else {

      // Print node name and details of 'maxPerNode' errors
      os << iter->second.first << std::endl ;
      //iter->first->printStream(os,kName|kClassName|kArgs,kSingleLine) ;

      Int_t i(0) ;
      for(auto iter2 = iter->second.second.begin();iter2!=iter->second.second.end() ; ++iter2, i++) {
   os << "     " << iter2->_msg << " @ " << iter2->_srvval << std::endl ;
   if (i>maxPerNode) {
     os << "    ... (remaining " << iter->second.second.size() - maxPerNode << " messages suppressed)" << std::endl ;
     break ;
   }
      }
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Return the number of logged evaluation errors since the last clearing.

Int_t RooAbsReal::numEvalErrors()
{
  if (_evalErrorMode==CountErrors) {
    return _evalErrorCount ;
  }

  Int_t ntot(0) ;
  for(auto const& elem : _evalErrorList) {
    ntot += elem.second.second.size() ;
  }
  return ntot ;
}



////////////////////////////////////////////////////////////////////////////////
/// Fix the interpretation of the coefficient of any RooAddPdf component in
/// the expression tree headed by this object to the given set of observables.
///
/// If the force flag is false, the normalization choice is only fixed for those
/// RooAddPdf components that have the default 'automatic' interpretation of
/// coefficients (i.e. the interpretation is defined by the observables passed
/// to getVal()). If force is true, also RooAddPdf that already have a fixed
/// interpretation are changed to a new fixed interpretation.

void RooAbsReal::fixAddCoefNormalization(const RooArgSet& addNormSet, bool force)
{
  std::unique_ptr<RooArgSet> compSet{getComponents()};
  for(auto * pdf : dynamic_range_cast<RooAbsPdf*>(*compSet)) {
    if (pdf) {
      if (addNormSet.getSize()>0) {
   pdf->selectNormalization(&addNormSet,force) ;
      } else {
   pdf->selectNormalization(0,force) ;
      }
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Fix the interpretation of the coefficient of any RooAddPdf component in
/// the expression tree headed by this object to the given set of observables.
///
/// If the force flag is false, the normalization range choice is only fixed for those
/// RooAddPdf components that currently use the default full domain to interpret their
/// coefficients. If force is true, also RooAddPdf that already have a fixed
/// interpretation range are changed to a new fixed interpretation range.

void RooAbsReal::fixAddCoefRange(const char* rangeName, bool force)
{
  std::unique_ptr<RooArgSet> compSet{getComponents()};
  for(auto * pdf : dynamic_range_cast<RooAbsPdf*>(*compSet)) {
    if (pdf) {
      pdf->selectNormalizationRange(rangeName,force) ;
    }
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Interface method for function objects to indicate their preferred order of observables
/// for scanning their values into a (multi-dimensional) histogram or RooDataSet. The observables
/// to be ordered are offered in argument 'obs' and should be copied in their preferred
/// order into argument 'orderdObs', This default implementation indicates no preference
/// and copies the original order of 'obs' into 'orderedObs'

void RooAbsReal::preferredObservableScanOrder(const RooArgSet& obs, RooArgSet& orderedObs) const
{
  // Dummy implementation, do nothing
  orderedObs.removeAll() ;
  orderedObs.add(obs) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Calls createRunningIntegral(const RooArgSet&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&, const RooCmdArg&)

RooFit::OwningPtr<RooAbsReal> RooAbsReal::createRunningIntegral(const RooArgSet& iset, const RooArgSet& nset)
{
  return createRunningIntegral(iset,RooFit::SupNormSet(nset)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create an object that represents the running integral of the function over one or more observables std::listed in iset, i.e.
/// \f[
///   \int_{x_\mathrm{lo}}^x f(x') \, \mathrm{d}x'
/// \f]
///
/// The actual integration calculation is only performed when the return object is evaluated. The name
/// of the integral object is automatically constructed from the name of the input function, the variables
/// it integrates and the range integrates over. The default strategy to calculate the running integrals is
///
///   - If the integrand (this object) supports analytical integration, construct an integral object
///     that calculate the running integrals value by calculating the analytical integral each
///     time the running integral object is evaluated
///
///   - If the integrand (this object) requires numeric integration to construct the running integral
///     create an object of class RooNumRunningInt which first samples the entire function and integrates
///     the sampled function numerically. This method has superior performance as there is no need to
///     perform a full (numeric) integration for each evaluation of the running integral object, but
///     only when one of its parameters has changed.
///
/// The choice of strategy can be changed with the ScanAll() argument, which forces the use of the
/// scanning technique implemented in RooNumRunningInt for all use cases, and with the ScanNone()
/// argument which forces the 'integrate each evaluation' technique for all use cases. The sampling
/// granularity for the scanning technique can be controlled with the ScanParameters technique
/// which allows to specify the number of samples to be taken, and to which order the resulting
/// running integral should be interpolated. The default values are 1000 samples and 2nd order
/// interpolation.
///
/// The following named arguments are accepted
/// | | Effect on integral creation
/// |-|-------------------------------
/// | `SupNormSet(const RooArgSet&)`         | Observables over which should be normalized _in addition_ to the integration observables
/// | `ScanParameters(Int_t nbins, Int_t intOrder)`    | Parameters for scanning technique of making CDF: number of sampled bins and order of interpolation applied on numeric cdf
/// | `ScanNum()`                            | Apply scanning technique if cdf integral involves numeric integration
/// | `ScanAll()`                            | Always apply scanning technique
/// | `ScanNone()`                           | Never apply scanning technique

RooFit::OwningPtr<RooAbsReal> RooAbsReal::createRunningIntegral(const RooArgSet& iset, const RooCmdArg& arg1, const RooCmdArg& arg2,
             const RooCmdArg& arg3, const RooCmdArg& arg4, const RooCmdArg& arg5,
             const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::createRunningIntegral(%s)",GetName())) ;
  pc.defineSet("supNormSet","SupNormSet",0,0) ;
  pc.defineInt("numScanBins","ScanParameters",0,1000) ;
  pc.defineInt("intOrder","ScanParameters",1,2) ;
  pc.defineInt("doScanNum","ScanNum",0,1) ;
  pc.defineInt("doScanAll","ScanAll",0,0) ;
  pc.defineInt("doScanNon","ScanNone",0,0) ;
  pc.defineMutex("ScanNum","ScanAll","ScanNone") ;

  // Process & check varargs
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(true)) {
    return 0 ;
  }

  // Extract values from named arguments
  RooArgSet nset ;
  if (const RooArgSet* snset = pc.getSet("supNormSet",0)) {
    nset.add(*snset) ;
  }
  Int_t numScanBins = pc.getInt("numScanBins") ;
  Int_t intOrder = pc.getInt("intOrder") ;
  Int_t doScanNum = pc.getInt("doScanNum") ;
  Int_t doScanAll = pc.getInt("doScanAll") ;
  Int_t doScanNon = pc.getInt("doScanNon") ;

  // If scanning technique is not requested make integral-based cdf and return
  if (doScanNon) {
    return createIntRI(iset,nset) ;
  }
  if (doScanAll) {
    return createScanRI(iset,nset,numScanBins,intOrder) ;
  }
  if (doScanNum) {
    std::unique_ptr<RooRealIntegral> tmp{static_cast<RooRealIntegral*>(createIntegral(iset))};
    Int_t isNum= (tmp->numIntRealVars().size()==1) ;

    if (isNum) {
      coutI(NumIntegration) << "RooAbsPdf::createRunningIntegral(" << GetName() << ") integration over observable(s) " << iset << " involves numeric integration," << std::endl
             << "      constructing cdf though numeric integration of sampled pdf in " << numScanBins << " bins and applying order "
             << intOrder << " interpolation on integrated histogram." << std::endl
             << "      To override this choice of technique use argument ScanNone(), to change scan parameters use ScanParameters(nbins,order) argument" << std::endl ;
    }

    return isNum ? createScanRI(iset,nset,numScanBins,intOrder) : createIntRI(iset,nset) ;
  }
  return 0 ;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for createRunningIntegral that construct an object
/// implementing the numeric scanning technique for calculating the running integral

RooFit::OwningPtr<RooAbsReal> RooAbsReal::createScanRI(const RooArgSet& iset, const RooArgSet& nset, Int_t numScanBins, Int_t intOrder)
{
  std::string name = std::string(GetName()) + "_NUMRUNINT_" + integralNameSuffix(iset,&nset).Data() ;
  RooRealVar* ivar = (RooRealVar*) iset.first() ;
  ivar->setBins(numScanBins,"numcdf") ;
  auto ret = std::make_unique<RooNumRunningInt>(name.c_str(),name.c_str(),*this,*ivar,"numrunint") ;
  ret->setInterpolationOrder(intOrder) ;
  return RooFit::OwningPtr<RooAbsReal>{ret.release()};
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function for createRunningIntegral. It creates an
/// object implementing the standard (analytical) integration
/// technique for calculating the running integral.

RooFit::OwningPtr<RooAbsReal> RooAbsReal::createIntRI(const RooArgSet& iset, const RooArgSet& nset)
{
  // Make std::list of input arguments keeping only RooRealVars
  RooArgList ilist ;
  for(RooAbsArg * arg : iset) {
    if (dynamic_cast<RooRealVar*>(arg)) {
      ilist.add(*arg) ;
    } else {
      coutW(InputArguments) << "RooAbsPdf::createRunningIntegral(" << GetName() << ") WARNING ignoring non-RooRealVar input argument " << arg->GetName() << std::endl ;
    }
  }

  RooArgList cloneList ;
  RooArgList loList ;
  RooArgSet clonedBranchNodes ;

  // Setup customizer that stores all cloned branches in our non-owning std::list
  RooCustomizer cust(*this,"cdf") ;
  cust.setCloneBranchSet(clonedBranchNodes) ;
  cust.setOwning(false) ;

  // Make integration observable x_prime for each observable x as well as an x_lowbound
  for(auto * rrv : static_range_cast<RooRealVar*>(ilist)) {

    // Make clone x_prime of each c.d.f observable x represening running integral
    RooRealVar* cloneArg = (RooRealVar*) rrv->clone(Form("%s_prime",rrv->GetName())) ;
    cloneList.add(*cloneArg) ;
    cust.replaceArg(*rrv,*cloneArg) ;

    // Make clone x_lowbound of each c.d.f observable representing low bound of x
    RooRealVar* cloneLo = (RooRealVar*) rrv->clone(Form("%s_lowbound",rrv->GetName())) ;
    cloneLo->setVal(rrv->getMin()) ;
    loList.add(*cloneLo) ;

    // Make parameterized binning from [x_lowbound,x] for each x_prime
    RooParamBinning pb(*cloneLo,*rrv,100) ;
    cloneArg->setBinning(pb,"CDF") ;

  }

  RooAbsReal* tmp = (RooAbsReal*) cust.build() ;

  // Construct final normalization set for c.d.f = integrated observables + any extra specified by user
  RooArgSet finalNset(nset) ;
  finalNset.add(cloneList,true) ;
  std::unique_ptr<RooAbsReal> cdf{tmp->createIntegral(cloneList,finalNset,"CDF")};

  // Transfer ownership of cloned items to top-level c.d.f object
  cdf->addOwnedComponents(*tmp) ;
  cdf->addOwnedComponents(cloneList) ;
  cdf->addOwnedComponents(loList) ;

  return RooFit::OwningPtr<RooAbsReal>{cdf.release()};
}


////////////////////////////////////////////////////////////////////////////////
/// Return a RooFunctor object bound to this RooAbsReal with given definition of observables
/// and parameters

RooFunctor* RooAbsReal::functor(const RooArgList& obs, const RooArgList& pars, const RooArgSet& nset) const
{
  RooArgSet realObs;
  getObservables(&obs, realObs);
  if (realObs.size() != obs.size()) {
    coutE(InputArguments) << "RooAbsReal::functor(" << GetName() << ") ERROR: one or more specified observables are not variables of this p.d.f" << std::endl ;
    return nullptr;
  }
  RooArgSet realPars;
  getObservables(&pars, realPars);
  if (realPars.size() != pars.size()) {
    coutE(InputArguments) << "RooAbsReal::functor(" << GetName() << ") ERROR: one or more specified parameters are not variables of this p.d.f" << std::endl ;
    return nullptr;
  }

  return new RooFunctor(*this,obs,pars,nset) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return a ROOT TF1,2,3 object bound to this RooAbsReal with given definition of observables
/// and parameters

TF1* RooAbsReal::asTF(const RooArgList& obs, const RooArgList& pars, const RooArgSet& nset) const
{
  // Check that specified input are indeed variables of this function
  RooArgSet realObs;
  getObservables(&obs, realObs) ;
  if (realObs.size() != obs.size()) {
    coutE(InputArguments) << "RooAbsReal::functor(" << GetName() << ") ERROR: one or more specified observables are not variables of this p.d.f" << std::endl ;
    return 0 ;
  }
  RooArgSet realPars;
  getObservables(&pars, realPars) ;
  if (realPars.size() != pars.size()) {
    coutE(InputArguments) << "RooAbsReal::functor(" << GetName() << ") ERROR: one or more specified parameters are not variables of this p.d.f" << std::endl ;
    return 0 ;
  }

  // Check that all obs and par are of type RooRealVar
  for (int i=0 ; i<obs.getSize() ; i++) {
    if (dynamic_cast<RooRealVar*>(obs.at(i))==0) {
      coutE(ObjectHandling) << "RooAbsReal::asTF(" << GetName() << ") ERROR: proposed observable " << obs.at(0)->GetName() << " is not of type RooRealVar" << std::endl ;
      return 0 ;
    }
  }
  for (int i=0 ; i<pars.getSize() ; i++) {
    if (dynamic_cast<RooRealVar*>(pars.at(i))==0) {
      coutE(ObjectHandling) << "RooAbsReal::asTF(" << GetName() << ") ERROR: proposed parameter " << pars.at(0)->GetName() << " is not of type RooRealVar" << std::endl ;
      return 0 ;
    }
  }

  // Create functor and TFx of matching dimension
  TF1* tf=0 ;
  RooFunctor* f ;
  switch(obs.getSize()) {
  case 1: {
    RooRealVar* x = (RooRealVar*)obs.at(0) ;
    f = functor(obs,pars,nset) ;
    tf = new TF1(GetName(),f,x->getMin(),x->getMax(),pars.getSize()) ;
    break ;
  }
  case 2: {
    RooRealVar* x = (RooRealVar*)obs.at(0) ;
    RooRealVar* y = (RooRealVar*)obs.at(1) ;
    f = functor(obs,pars,nset) ;
    tf = new TF2(GetName(),f,x->getMin(),x->getMax(),y->getMin(),y->getMax(),pars.getSize()) ;
    break ;
  }
  case 3: {
    RooRealVar* x = (RooRealVar*)obs.at(0) ;
    RooRealVar* y = (RooRealVar*)obs.at(1) ;
    RooRealVar* z = (RooRealVar*)obs.at(2) ;
    f = functor(obs,pars,nset) ;
    tf = new TF3(GetName(),f,x->getMin(),x->getMax(),y->getMin(),y->getMax(),z->getMin(),z->getMax(),pars.getSize()) ;
    break ;
  }
  default:
    coutE(InputArguments) << "RooAbsReal::asTF(" << GetName() << ") ERROR: " << obs.getSize()
           << " observables specified, but a ROOT TFx can only have  1,2 or 3 observables" << std::endl ;
    return 0 ;
  }

  // Set initial parameter values of TFx to those of RooRealVars
  for (int i=0 ; i<pars.getSize() ; i++) {
    RooRealVar* p = (RooRealVar*) pars.at(i) ;
    tf->SetParameter(i,p->getVal()) ;
    tf->SetParName(i,p->GetName()) ;
    //tf->SetParLimits(i,p->getMin(),p->getMax()) ;
  }

  return tf ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return function representing first, second or third order derivative of this function

RooDerivative* RooAbsReal::derivative(RooRealVar& obs, Int_t order, double eps)
{
  std::string name=Form("%s_DERIV_%s",GetName(),obs.GetName()) ;
  std::string title=Form("Derivative of %s w.r.t %s ",GetName(),obs.GetName()) ;
  return new RooDerivative(name.c_str(),title.c_str(),*this,obs,order,eps) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return function representing first, second or third order derivative of this function

RooDerivative* RooAbsReal::derivative(RooRealVar& obs, const RooArgSet& normSet, Int_t order, double eps)
{
  std::string name=Form("%s_DERIV_%s",GetName(),obs.GetName()) ;
  std::string title=Form("Derivative of %s w.r.t %s ",GetName(),obs.GetName()) ;
  return new RooDerivative(name.c_str(),title.c_str(),*this,obs,normSet,order,eps) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return function representing moment of function of given order.
/// \param[in] obs Observable to calculate the moments for
/// \param[in] order Order of the moment
/// \param[in] central If true, the central moment is given by \f$ \langle (x- \langle x \rangle )^2 \rangle \f$
/// \param[in] takeRoot Calculate the square root

RooAbsMoment* RooAbsReal::moment(RooRealVar& obs, Int_t order, bool central, bool takeRoot)
{
  std::string name=Form("%s_MOMENT_%d%s_%s",GetName(),order,(central?"C":""),obs.GetName()) ;
  std::string title=Form("%sMoment of order %d of %s w.r.t %s ",(central?"Central ":""),order,GetName(),obs.GetName()) ;
  if (order==1) return new RooFirstMoment(name.c_str(),title.c_str(),*this,obs) ;
  if (order==2) return new RooSecondMoment(name.c_str(),title.c_str(),*this,obs,central,takeRoot) ;
  return new RooMoment(name.c_str(),title.c_str(),*this,obs,order,central,takeRoot) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Return function representing moment of p.d.f (normalized w.r.t given observables) of given order.
/// \param[in] obs Observable to calculate the moments for
/// \param[in] normObs Normalise w.r.t. these observables
/// \param[in] order Order of the moment
/// \param[in] central If true, the central moment is given by \f$ \langle (x- \langle x \rangle )^2 \rangle \f$
/// \param[in] takeRoot Calculate the square root
/// \param[in] intNormObs If true, the moment of the function integrated over all normalization observables is returned.

RooAbsMoment* RooAbsReal::moment(RooRealVar& obs, const RooArgSet& normObs, Int_t order, bool central, bool takeRoot, bool intNormObs)
{
  std::string name=Form("%s_MOMENT_%d%s_%s",GetName(),order,(central?"C":""),obs.GetName()) ;
  std::string title=Form("%sMoment of order %d of %s w.r.t %s ",(central?"Central ":""),order,GetName(),obs.GetName()) ;

  if (order==1) return new RooFirstMoment(name.c_str(),title.c_str(),*this,obs,normObs,intNormObs) ;
  if (order==2) return new RooSecondMoment(name.c_str(),title.c_str(),*this,obs,normObs,central,takeRoot,intNormObs) ;
  return new RooMoment(name.c_str(),title.c_str(),*this,obs,normObs,order,central,takeRoot,intNormObs) ;
}



////////////////////////////////////////////////////////////////////////////////
///
/// Return value of x (in range xmin,xmax) at which function equals yval.
/// (Calculation is performed with Brent root finding algorithm)

double RooAbsReal::findRoot(RooRealVar& x, double xmin, double xmax, double yval)
{
  double result(0) ;
  RooBrentRootFinder(RooRealBinding(*this,x)).findRoot(result,xmin,xmax,yval) ;
  return result ;
}




////////////////////////////////////////////////////////////////////////////////

RooGenFunction* RooAbsReal::iGenFunction(RooRealVar& x, const RooArgSet& nset)
{
  return new RooGenFunction(*this,x,RooArgList(),nset.getSize()>0?nset:RooArgSet(x)) ;
}



////////////////////////////////////////////////////////////////////////////////

RooMultiGenFunction* RooAbsReal::iGenFunction(const RooArgSet& observables, const RooArgSet& nset)
{
  return new RooMultiGenFunction(*this,observables,RooArgList(),nset.getSize()>0?nset:observables) ;
}




////////////////////////////////////////////////////////////////////////////////
/// Perform a \f$ \chi^2 \f$ fit to given histogram. By default the fit is executed through the MINUIT
/// commands MIGRAD, HESSE in succession
///
/// The following named arguments are supported
///
/// <table>
/// <tr><th> <th> Options to control construction of chi2
/// <tr><td> `Range(const char* name)`         <td> Fit only data inside range with given name
/// <tr><td> `Range(double lo, double hi)` <td> Fit only data inside given range. A range named "fit" is created on the fly on all observables.
///                                               Multiple comma separated range names can be specified.
/// <tr><td> `NumCPU(int num)`                 <td> Parallelize NLL calculation on num CPUs
/// <tr><td> `Optimize(bool flag)`           <td> Activate constant term optimization (on by default)
/// <tr><td> `IntegrateBins()`                 <td> Integrate PDF within each bin. This sets the desired precision.
///
/// <tr><th> <th> Options to control flow of fit procedure
/// <tr><td> `InitialHesse(bool flag)`      <td> Flag controls if HESSE before MIGRAD as well, off by default
/// <tr><td> `Hesse(bool flag)`             <td> Flag controls if HESSE is run after MIGRAD, on by default
/// <tr><td> `Minos(bool flag)`             <td> Flag controls if MINOS is run after HESSE, on by default
/// <tr><td> `Minos(const RooArgSet& set)`    <td> Only run MINOS on given subset of arguments
/// <tr><td> `Save(bool flag)`              <td> Flac controls if RooFitResult object is produced and returned, off by default
/// <tr><td> `Strategy(Int_t flag)`           <td> Set Minuit strategy (0 through 2, default is 1)
///
/// <tr><th> <th> Options to control informational output
/// <tr><td> `Verbose(bool flag)`           <td> Flag controls if verbose output is printed (NLL, parameter changes during fit
/// <tr><td> `Timer(bool flag)`             <td> Time CPU and wall clock consumption of fit steps, off by default
/// <tr><td> `PrintLevel(Int_t level)`        <td> Set Minuit print level (-1 through 3, default is 1). At -1 all RooFit informational
///                                              messages are suppressed as well
/// <tr><td> `Warnings(bool flag)`          <td> Enable or disable MINUIT warnings (enabled by default)
/// <tr><td> `PrintEvalErrors(Int_t numErr)`  <td> Control number of p.d.f evaluation errors printed per likelihood evaluation. A negative
///                                              value suppress output completely, a zero value will only print the error count per p.d.f component,
///                                              a positive value is will print details of each error up to numErr messages per p.d.f component.
/// </table>
///

RooFitResult* RooAbsReal::chi2FitTo(RooDataHist& data, const RooCmdArg& arg1,  const RooCmdArg& arg2,
                const RooCmdArg& arg3,  const RooCmdArg& arg4, const RooCmdArg& arg5,
                const RooCmdArg& arg6,  const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return chi2FitTo(data,l) ;

}



////////////////////////////////////////////////////////////////////////////////
/// \copydoc RooAbsReal::chi2FitTo(RooDataHist&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&)

RooFitResult* RooAbsReal::chi2FitTo(RooDataHist& data, const RooLinkedList& cmdList)
{
  // Select the pdf-specific commands
  RooCmdConfig pc(Form("RooAbsPdf::chi2FitTo(%s)",GetName())) ;

  // Pull arguments to be passed to chi2 construction from std::list
  RooLinkedList fitCmdList(cmdList) ;
  RooLinkedList chi2CmdList = pc.filterCmdList(fitCmdList,"Range,RangeWithName,NumCPU,Optimize,IntegrateBins") ;

  std::unique_ptr<RooAbsReal> chi2{createChi2(data,chi2CmdList)};
  RooFitResult* ret = chi2FitDriver(*chi2,fitCmdList) ;

  return ret ;
}




////////////////////////////////////////////////////////////////////////////////
/// Create a \f$ \chi^2 \f$ variable from a histogram and this function.
///
/// \param arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8 ordered arguments
///
/// The std::list of supported command arguments is given in the documentation for
///     RooChi2Var::RooChi2Var(const char *name, const char* title, RooAbsReal& func, RooDataHist& hdata, const RooCmdArg&,const RooCmdArg&,const RooCmdArg&, const RooCmdArg&,const RooCmdArg&,const RooCmdArg&, const RooCmdArg&,const RooCmdArg&,const RooCmdArg&).
///
/// \param data Histogram with data
/// \return \f$ \chi^2 \f$ variable

RooAbsReal* RooAbsReal::createChi2(RooDataHist& data, const RooCmdArg& arg1,  const RooCmdArg& arg2,
               const RooCmdArg& arg3,  const RooCmdArg& arg4, const RooCmdArg& arg5,
               const RooCmdArg& arg6,  const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  std::string name = Form("chi2_%s_%s",GetName(),data.GetName()) ;

  return new RooChi2Var(name.c_str(),name.c_str(),*this,data,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
}




////////////////////////////////////////////////////////////////////////////////
/// \see RooAbsReal::createChi2(RooDataHist&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&)
/// \param data hist data
/// \param cmdList List with RooCmdArg() from the table

RooAbsReal* RooAbsReal::createChi2(RooDataHist& data, const RooLinkedList& cmdList)
{
  // Fill array of commands
  const RooCmdArg* cmds[8] ;
  Int_t i(0) ;
  for(auto * arg : static_range_cast<RooCmdArg*>(cmdList)) {
    cmds[i++] = arg ;
  }
  for (;i<8 ; i++) {
    cmds[i] = &RooCmdArg::none() ;
  }

  return createChi2(data,*cmds[0],*cmds[1],*cmds[2],*cmds[3],*cmds[4],*cmds[5],*cmds[6],*cmds[7]) ;

}





////////////////////////////////////////////////////////////////////////////////
/// Perform a 2-D \f$ \chi^2 \f$ fit using a series of x and y values stored in the dataset `xydata`.
/// The y values can either be the event weights, or can be another column designated
/// by the YVar() argument. The y value must have errors defined for the \f$ \chi^2 \f$ to
/// be well defined.
///
/// <table>
/// <tr><th><th> Options to control construction of the chi-square
/// <tr><td> `YVar(RooRealVar& yvar)`          <td>  Designate given column in dataset as Y value
/// <tr><td> `Integrate(bool flag)`          <td>  Integrate function over range specified by X errors
///                                    rather than take value at bin center.
///
/// <tr><th><th> Options to control flow of fit procedure
/// <tr><td> `InitialHesse(bool flag)`      <td>  Flag controls if HESSE before MIGRAD as well, off by default
/// <tr><td> `Hesse(bool flag)`             <td>  Flag controls if HESSE is run after MIGRAD, on by default
/// <tr><td> `Minos(bool flag)`             <td>  Flag controls if MINOS is run after HESSE, on by default
/// <tr><td> `Minos(const RooArgSet& set)`    <td>  Only run MINOS on given subset of arguments
/// <tr><td> `Save(bool flag)`              <td>  Flac controls if RooFitResult object is produced and returned, off by default
/// <tr><td> `Strategy(Int_t flag)`           <td>  Set Minuit strategy (0 through 2, default is 1)
///
/// <tr><th><th> Options to control informational output
/// <tr><td> `Verbose(bool flag)`           <td>  Flag controls if verbose output is printed (NLL, parameter changes during fit
/// <tr><td> `Timer(bool flag)`             <td>  Time CPU and wall clock consumption of fit steps, off by default
/// <tr><td> `PrintLevel(Int_t level)`        <td>  Set Minuit print level (-1 through 3, default is 1). At -1 all RooFit informational
///                                   messages are suppressed as well
/// <tr><td> `Warnings(bool flag)`          <td>  Enable or disable MINUIT warnings (enabled by default)
/// <tr><td> `PrintEvalErrors(Int_t numErr)`  <td>  Control number of p.d.f evaluation errors printed per likelihood evaluation. A negative
///                                   value suppress output completely, a zero value will only print the error count per p.d.f component,
///                                   a positive value is will print details of each error up to numErr messages per p.d.f component.
/// </table>

RooFitResult* RooAbsReal::chi2FitTo(RooDataSet& xydata, const RooCmdArg& arg1,  const RooCmdArg& arg2,
                  const RooCmdArg& arg3,  const RooCmdArg& arg4, const RooCmdArg& arg5,
                  const RooCmdArg& arg6,  const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return chi2FitTo(xydata,l) ;
}




////////////////////////////////////////////////////////////////////////////////
/// \copydoc RooAbsReal::chi2FitTo(RooDataSet&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&)

RooFitResult* RooAbsReal::chi2FitTo(RooDataSet& xydata, const RooLinkedList& cmdList)
{
  // Select the pdf-specific commands
  RooCmdConfig pc(Form("RooAbsPdf::chi2FitTo(%s)",GetName())) ;

  // Pull arguments to be passed to chi2 construction from std::list
  RooLinkedList fitCmdList(cmdList) ;
  RooLinkedList chi2CmdList = pc.filterCmdList(fitCmdList,"YVar,Integrate") ;

  std::unique_ptr<RooAbsReal> xychi2{createChi2(xydata,chi2CmdList)};
  RooFitResult* ret = chi2FitDriver(*xychi2,fitCmdList) ;

  return ret ;
}




////////////////////////////////////////////////////////////////////////////////
/// Create a \f$ \chi^2 \f$ from a series of x and y values stored in a dataset.
/// The y values can either be the event weights (default), or can be another column designated
/// by the YVar() argument. The y value must have errors defined for the \f$ \chi^2 \f$ to
/// be well defined.
///
/// The following named arguments are supported
///
/// | | Options to control construction of the \f$ \chi^2 \f$
/// |-|-----------------------------------------
/// | `YVar(RooRealVar& yvar)`  | Designate given column in dataset as Y value
/// | `Integrate(bool flag)`  | Integrate function over range specified by X errors rather than take value at bin center.
///

RooAbsReal* RooAbsReal::createChi2(RooDataSet& data, const RooCmdArg& arg1,  const RooCmdArg& arg2,
                 const RooCmdArg& arg3,  const RooCmdArg& arg4, const RooCmdArg& arg5,
                 const RooCmdArg& arg6,  const RooCmdArg& arg7, const RooCmdArg& arg8)
{
  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  return createChi2(data,l) ;
}



////////////////////////////////////////////////////////////////////////////////
/// See RooAbsReal::createChi2(RooDataSet&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&,const RooCmdArg&)

RooAbsReal* RooAbsReal::createChi2(RooDataSet& data, const RooLinkedList& cmdList)
{
  // Select the pdf-specific commands
  RooCmdConfig pc(Form("RooAbsPdf::fitTo(%s)",GetName())) ;

  pc.defineInt("integrate","Integrate",0,0) ;
  pc.defineObject("yvar","YVar",0,0) ;

  // Process and check varargs
  pc.process(cmdList) ;
  if (!pc.ok(true)) {
    return 0 ;
  }

  // Decode command line arguments
  bool integrate = pc.getInt("integrate") ;
  RooRealVar* yvar = (RooRealVar*) pc.getObject("yvar") ;

  std::string name = Form("chi2_%s_%s",GetName(),data.GetName()) ;

  if (yvar) {
    return new RooXYChi2Var(name.c_str(),name.c_str(),*this,data,*yvar,integrate) ;
  } else {
    return new RooXYChi2Var(name.c_str(),name.c_str(),*this,data,integrate) ;
  }
}






////////////////////////////////////////////////////////////////////////////////
/// Internal driver function for chi2 fits

RooFitResult* RooAbsReal::chi2FitDriver(RooAbsReal& fcn, RooLinkedList& cmdList)
{
  // Select the pdf-specific commands
  RooCmdConfig pc(Form("RooAbsPdf::chi2FitDriver(%s)",GetName())) ;

  pc.defineInt("optConst","Optimize",0,1) ;
  pc.defineInt("verbose","Verbose",0,0) ;
  pc.defineInt("doSave","Save",0,0) ;
  pc.defineInt("doTimer","Timer",0,0) ;
  pc.defineInt("plevel","PrintLevel",0,1) ;
  pc.defineInt("strat","Strategy",0,1) ;
  pc.defineInt("initHesse","InitialHesse",0,0) ;
  pc.defineInt("hesse","Hesse",0,1) ;
  pc.defineInt("minos","Minos",0,0) ;
  pc.defineInt("ext","Extended",0,2) ;
  pc.defineInt("numee","PrintEvalErrors",0,10) ;
  pc.defineInt("doWarn","Warnings",0,1) ;
  pc.defineString("mintype","Minimizer",0,"") ;
  pc.defineString("minalg","Minimizer",1,"minuit") ;
  pc.defineSet("minosSet","Minos",0,nullptr) ;

  // Process and check varargs
  pc.process(cmdList) ;
  if (!pc.ok(true)) {
    return 0 ;
  }

  // Decode command line arguments
  const char* minType = pc.getString("mintype","") ;
  const char* minAlg = pc.getString("minalg","minuit") ;
  Int_t optConst = pc.getInt("optConst") ;
  Int_t verbose  = pc.getInt("verbose") ;
  Int_t doSave   = pc.getInt("doSave") ;
  Int_t doTimer  = pc.getInt("doTimer") ;
  Int_t plevel    = pc.getInt("plevel") ;
  Int_t strat    = pc.getInt("strat") ;
  Int_t initHesse= pc.getInt("initHesse") ;
  Int_t hesse    = pc.getInt("hesse") ;
  Int_t minos    = pc.getInt("minos") ;
  Int_t numee    = pc.getInt("numee") ;
  Int_t doWarn   = pc.getInt("doWarn") ;
  const RooArgSet* minosSet = pc.getSet("minosSet");

  RooFitResult *ret = 0 ;

  // Instantiate MINUIT
  RooMinimizer m(fcn) ;
  m.setMinimizerType(minType);

  if (doWarn==0) {
    // m.setNoWarn() ; WVE FIX THIS
  }

  m.setPrintEvalErrors(numee) ;
  if (plevel!=1) {
    m.setPrintLevel(plevel) ;
  }

  if (optConst) {
    // Activate constant term optimization
    m.optimizeConst(optConst);
  }

  if (verbose) {
    // Activate verbose options
    m.setVerbose(1) ;
  }
  if (doTimer) {
    // Activate timer options
    m.setProfile(1) ;
  }

  if (strat!=1) {
    // Modify fit strategy
    m.setStrategy(strat) ;
  }

  if (initHesse) {
    // Initialize errors with hesse
    m.hesse() ;
  }

  // Minimize using migrad
  m.minimize(minType, minAlg) ;

  if (hesse) {
    // Evaluate errors with Hesse
    m.hesse() ;
  }

  if (minos) {
    // Evaluate errs with Minos
    if (minosSet) {
      m.minos(*minosSet) ;
    } else {
      m.minos() ;
    }
  }

  // Optionally return fit result
  if (doSave) {
    std::string name = Form("fitresult_%s",fcn.GetName()) ;
    std::string title = Form("Result of fit of %s ",GetName()) ;
    ret = m.save(name.c_str(),title.c_str()) ;
  }

  // Cleanup
  return ret ;

}


////////////////////////////////////////////////////////////////////////////////
/// Return current evaluation error logging mode.

RooAbsReal::ErrorLoggingMode RooAbsReal::evalErrorLoggingMode()
{
  return _evalErrorMode ;
}

////////////////////////////////////////////////////////////////////////////////
/// Set evaluation error logging mode. Options are
///
/// PrintErrors - Print each error through RooMsgService() as it occurs
/// CollectErrors - Accumulate errors, but do not print them. A subsequent call
///                 to printEvalErrors() will print a summary
/// CountErrors - Accumulate error count, but do not print them.
///

void RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::ErrorLoggingMode m)
{
  _evalErrorMode =  m;
}


////////////////////////////////////////////////////////////////////////////////

void RooAbsReal::setParameterizeIntegral(const RooArgSet& paramVars)
{
  std::string plist ;
  for (auto const* arg : paramVars) {
    if (!dependsOnValue(*arg)) {
      coutW(InputArguments) << "RooAbsReal::setParameterizeIntegral(" << GetName()
             << ") function does not depend on std::listed parameter " << arg->GetName() << ", ignoring" << std::endl ;
      continue ;
    }
    if (plist.size()>0) plist += ":" ;
    plist += arg->GetName() ;
  }
  setStringAttribute("CACHEPARAMINT",plist.c_str()) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Evaluate this object for a batch/span of data points.
/// This is the backend used by getValues() to perform computations. A span pointing to the computation results
/// will be stored in `evalData`, and also be returned to getValues(), which then finalises the computation.
///
/// \note Derived classes should override this function to reach maximal performance. If this function is not overridden, the slower,
/// single-valued evaluate() will be called in a loop.
///
/// A computation proceeds as follows:
/// - Request input data from all our servers using `getValues(evalData, normSet)`. Those will return
///   - batches of size 1 if their value is constant over the entire dataset.
///   - batches of the size of the dataset stored in `evalData` otherwise.
///   If `evalData` already contains values for those objects, these will return data
///   without recomputing those.
/// - Create a new batch in `evalData` of the same size as those returned from the servers.
/// - Use the input data to perform the computations, and store those in the batch.
///
/// \note Error checking and normalisation (of PDFs) will be performed in getValues().
///
/// \param[in,out] evalData Object holding data that should be used in computations.
/// Computation results have to be stored here.
/// \param[in]  normSet  Optional normalisation set passed down to the servers of this object.
/// \return     Span pointing to the results. The memory is owned by `evalData`.
RooSpan<double> RooAbsReal::evaluateSpan(RooBatchCompute::RunContext& evalData, const RooArgSet* normSet) const {

  // Find all servers that are serving real numbers to us, retrieve their batch data,
  // and switch them into "always clean" operating mode, so they return always the last-set value.
  struct ServerData {
    RooAbsReal* server;
    RooSpan<const double> batch;
    double oldValue;
    RooAbsArg::OperMode oldOperMode;
  };
  std::vector<ServerData> ourServers;
  std::size_t dataSize = 1;

  for (auto absArgServer : servers()) {
    if (absArgServer->IsA()->InheritsFrom(RooAbsReal::Class()) && absArgServer->isValueServer(*this)) {
      auto server = static_cast<RooAbsReal*>(absArgServer);
      ourServers.push_back({server,
          server->getValues(evalData, normSet),
          server->getVal(normSet),
          server->operMode()});
      // Prevent the server from evaluating; just return cached result, which we will side load:
      server->setOperMode(RooAbsArg::AClean);
      dataSize = std::max(dataSize, ourServers.back().batch.size());
    }
  }


  // Make sure that we restore all state when we finish:
  struct RestoreStateRAII {
    RestoreStateRAII(std::vector<ServerData>& servers) :
      _servers{servers} { }

    ~RestoreStateRAII() {
      for (auto& serverData : _servers) {
        serverData.server->setCachedValue(serverData.oldValue, true);
        serverData.server->setOperMode(serverData.oldOperMode);
      }
    }

    std::vector<ServerData>& _servers;
  } restoreState{ourServers};


  // Advising to implement the batch interface makes only sense if the batch was not a scalar.
  // Otherwise, there would be no speedup benefit.
  if(dataSize > 1 && RooMsgService::instance().isActive(this, RooFit::FastEvaluations, RooFit::INFO)) {
    coutI(FastEvaluations) << "The class " << ClassName() << " does not implement the faster batch evaluation interface."
        << " Consider requesting or implementing it to benefit from a speed up." << std::endl;
  }


  // For each event, write temporary values into our servers' caches, and run a single-value computation.
  auto outputData = evalData.makeBatch(this, dataSize);

  for (std::size_t i=0; i < outputData.size(); ++i) {
    for (auto& serv : ourServers) {
      serv.server->setCachedValue(serv.batch[std::min(i, serv.batch.size()-1)], /*notifyClients=*/ false);
    }

    outputData[i] = evaluate();
  }

  return outputData;
}


/** Base function for computing multiple values of a RooAbsReal
\param output The array where the results are stored
\param nEvents The number of events to be processed
\param dataMap A std::map containing the input data for the computations
**/
void RooAbsReal::computeBatch(cudaStream_t*, double* output, size_t nEvents, RooFit::Detail::DataMap const& dataMap) const {
  // Find all servers that are serving real numbers to us, retrieve their batch data,
  // and switch them into "always clean" operating mode, so they return always the last-set value.
  struct ServerData {
    RooAbsArg* server;
    RooSpan<const double> batch;
    double oldValue;
    RooAbsArg::OperMode oldOperMode;
    bool oldValueDirty;
    bool oldShapeDirty;
  };
  std::vector<ServerData> ourServers;
  ourServers.reserve(servers().size());

  for (auto server : servers()) {
    auto serverValues = dataMap.at(server);
    if(serverValues.empty()) continue;

    // maybe we are still missing inhibit dirty here
    auto oldOperMode = server->operMode();
    // See note at the bottom of this function to learn why we can only set
    // the operation mode to "always clean" if there are no other value
    // clients.
    server->setOperMode(RooAbsArg::AClean);
    ourServers.push_back({server,
        serverValues,
        server->isCategory() ? static_cast<RooAbsCategory const*>(server)->getCurrentIndex() : static_cast<RooAbsReal const*>(server)->_value,
        oldOperMode,
        server->_valueDirty,
        server->_shapeDirty});
    // Prevent the server from evaluating; just return cached result, which we will side load:
  }


  // Make sure that we restore all state when we finish:
  struct RestoreStateRAII {
    RestoreStateRAII(std::vector<ServerData>& servers) :
      _servers{servers} { }

    ~RestoreStateRAII() {
      for (auto& serverData : _servers) {
        serverData.server->setCachedValue(serverData.oldValue, true);
        serverData.server->setOperMode(serverData.oldOperMode);
        serverData.server->_valueDirty = serverData.oldValueDirty;
        serverData.server->_shapeDirty = serverData.oldShapeDirty;
      }
    }

    std::vector<ServerData>& _servers;
  } restoreState{ourServers};


  // Advising to implement the batch interface makes only sense if the batch was not a scalar.
  // Otherwise, there would be no speedup benefit.
  if(nEvents > 1 && RooMsgService::instance().isActive(this, RooFit::FastEvaluations, RooFit::INFO)) {
    coutI(FastEvaluations) << "The class " << ClassName() << " does not implement the faster batch evaluation interface."
        << " Consider requesting or implementing it to benefit from a speed up." << std::endl;
  }


  // For each event, write temporary values into our servers' caches, and run a single-value computation.

  for (std::size_t i=0; i < nEvents; ++i) {
    for (auto& serv : ourServers) {
      serv.server->setCachedValue(serv.batch[std::min(i, serv.batch.size()-1)], false);
    }

    output[i] = evaluate();
  }
}




double RooAbsReal::_DEBUG_getVal(const RooArgSet* normalisationSet) const {

  const bool tmpFast = _fast;
  const double tmp = _value;

  double fullEval = 0.;
  try {
    fullEval = getValV(normalisationSet);
  }
  catch (CachingError& error) {
    throw CachingError(std::move(error),
        FormatPdfTree() << *this);
  }

  const double ret = (_fast && !_inhibitDirty) ? _value : fullEval;

  if (std::isfinite(ret) && ( ret != 0. ? (ret - fullEval)/ret : ret - fullEval) > 1.E-9) {
#ifndef NDEBUG
    gSystem->StackTrace();
#endif
    FormatPdfTree formatter;
    formatter << "--> (Scalar computation wrong here:)\n"
            << GetName() << " " << this << " _fast=" << tmpFast
            << "\n\tcached _value=" << std::setprecision(16) << tmp
            << "\n\treturning    =" << ret
            << "\n\trecomputed   =" << fullEval
            << "\n\tnew _value   =" << _value << "] ";
    formatter << "\nServers:";
    for (const auto server : _serverList) {
      formatter << "\n  ";
      server->printStream(formatter.stream(), kName | kClassName | kArgs | kExtras | kAddress | kValue, kInline);
    }

    throw CachingError(formatter);
  }

  return ret;
}


////////////////////////////////////////////////////////////////////////////////
/// Walk through expression tree headed by the `this` object, and check a batch computation.
///
/// Check if the results in `evalData` for event `evtNo` are identical to the current value of the nodes.
/// If a difference is found, an exception is thrown, and propagates up the expression tree. The tree is formatted
/// to see where the computation error happened.
/// @param evalData Data with results of batch computation. This is checked against the current value of the expression tree.
/// @param evtNo    Event from `evalData` to check for.
/// @param normSet  Optional normalisation set that was used in computation.
/// @param relAccuracy Accuracy required for passing the check.
void RooAbsReal::checkBatchComputation(const RooBatchCompute::RunContext& evalData, std::size_t evtNo, const RooArgSet* normSet, double relAccuracy) const {
  for (const auto server : _serverList) {
    try {
      auto realServer = dynamic_cast<RooAbsReal*>(server);
      if (realServer)
        realServer->checkBatchComputation(evalData, evtNo, normSet, relAccuracy);
    } catch (CachingError& error) {
      throw CachingError(std::move(error),
          FormatPdfTree() << *this);
    }
  }

  const auto item = evalData.spans.find(this);
  if (item == evalData.spans.end())
    return;

  auto batch = item->second;
  const double value = getVal(normSet);
  const double batchVal = batch.size() == 1 ? batch[0] : batch[evtNo];
  const double relDiff = value != 0. ? (value - batchVal)/value : value - batchVal;

  if (std::abs(relDiff) > relAccuracy && std::abs(value) > 1.E-300) {
    FormatPdfTree formatter;
    formatter << "--> (Batch computation wrong:)\n";
    printStream(formatter.stream(), kName | kClassName | kArgs | kExtras | kAddress, kInline);
    formatter << "\n batch=" << batch.data() << " size=" << batch.size() << std::setprecision(17)
    << "\n batch[" << std::setw(7) << evtNo-1 << "]=     " << (evtNo > 0 && evtNo - 1 < batch.size() ? std::to_string(batch[evtNo-1]) : "---")
    << "\n batch[" << std::setw(7) << evtNo   << "]=     " << batchVal << " !!!"
    << "\n expected ('value'): " << value
    << "\n eval(unnorm.)     : " << evaluate()
    << "\n delta         " <<                     " =     " << value - batchVal
    << "\n rel delta     " <<                     " =     " << relDiff
    << "\n _batch[" << std::setw(7) << evtNo+1 << "]=     " << (batch.size() > evtNo+1 ? std::to_string(batch[evtNo+1]) : "---");



    formatter << "\nServers: ";
    for (const auto server : _serverList) {
      formatter << "\n - ";
      server->printStream(formatter.stream(), kName | kClassName | kArgs | kExtras | kAddress | kValue, kInline);
      formatter << std::setprecision(17);

      auto serverAsReal = dynamic_cast<RooAbsReal*>(server);
      if (serverAsReal) {
        auto serverBatch = evalData.spans.count(serverAsReal) != 0 ? evalData.spans.find(serverAsReal)->second : RooSpan<const double>();
        if (serverBatch.size() > evtNo) {
          formatter << "\n   _batch[" << evtNo-1 << "]=" << (serverBatch.size() > evtNo-1 ? std::to_string(serverBatch[evtNo-1]) : "---")
                    << "\n   _batch[" << evtNo << "]=" << serverBatch[evtNo]
                    << "\n   _batch[" << evtNo+1 << "]=" << (serverBatch.size() > evtNo+1 ? std::to_string(serverBatch[evtNo+1]) : "---");
        }
        else {
          formatter << std::setprecision(17)
          << "\n   getVal()=" << serverAsReal->getVal(normSet);
        }
      }
    }

    throw CachingError(formatter);
  }
}


bool RooAbsReal::redirectServersHook(const RooAbsCollection & newServerList, bool mustReplaceAll,
                                    bool nameChange, bool isRecursiveStep)
{
  _lastNSet = nullptr ;
  return RooAbsArg::redirectServersHook(newServerList, mustReplaceAll, nameChange, isRecursiveStep);
}


////////////////////////////////////////////////////////////////////////////////

void RooAbsReal::enableOffsetting(bool flag)
{
  for (RooAbsArg* arg : servers()) {
    if(auto realArg = dynamic_cast<RooAbsReal*>(arg)) {
      realArg->enableOffsetting(flag) ;
    }
  }
}
