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
// RooAbsReal is the common abstract base class for objects that represent a
// real value and implements functionality common to all real-valued objects
// such as the ability to plot them, to construct integrals of them, the
// ability to advertise (partial) analytical integrals etc..

// Implementation of RooAbsReal may be derived, thus no interface
// is provided to modify the contents.
// 
// 
// END_HTML
//

#include <sys/types.h>


#include "RooFit.h"
#include "RooMsgService.h"

#include "RooAbsReal.h"
#include "RooAbsReal.h"
#include "RooArgSet.h"
#include "RooArgList.h"
#include "RooPlot.h"
#include "RooCurve.h"
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

#include "Riostream.h"

#include "TObjString.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TAttLine.h"

#include <sstream>

using namespace std ;
 
ClassImp(RooAbsReal)
;

Bool_t RooAbsReal::_cacheCheck(kFALSE) ;
Bool_t RooAbsReal::_globalSelectComp = kFALSE ;

Bool_t RooAbsReal::_doLogEvalError ;
map<const RooAbsArg*,pair<string,list<RooAbsReal::EvalError> > > RooAbsReal::_evalErrorList ;


//_____________________________________________________________________________
RooAbsReal::RooAbsReal() : _specIntegratorConfig(0), _treeVar(kFALSE), _selectComp(kTRUE)
{
  // Default constructor
}



//_____________________________________________________________________________
RooAbsReal::RooAbsReal(const char *name, const char *title, const char *unit) : 
  RooAbsArg(name,title), _plotMin(0), _plotMax(0), _plotBins(100), 
  _value(0),  _unit(unit), _forceNumInt(kFALSE), _specIntegratorConfig(0), _treeVar(kFALSE), _selectComp(kTRUE)
{
  // Constructor with unit label
  setValueDirty() ;
  setShapeDirty() ;

}



//_____________________________________________________________________________
RooAbsReal::RooAbsReal(const char *name, const char *title, Double_t inMinVal,
		       Double_t inMaxVal, const char *unit) :
  RooAbsArg(name,title), _plotMin(inMinVal), _plotMax(inMaxVal), _plotBins(100),
  _value(0), _unit(unit), _forceNumInt(kFALSE), _specIntegratorConfig(0), _treeVar(kFALSE), _selectComp(kTRUE)
{
  // Constructor with plot range and unit label
  setValueDirty() ;
  setShapeDirty() ;

}



//_____________________________________________________________________________
RooAbsReal::RooAbsReal(const RooAbsReal& other, const char* name) : 
  RooAbsArg(other,name), _plotMin(other._plotMin), _plotMax(other._plotMax), 
  _plotBins(other._plotBins), _value(other._value), _unit(other._unit), _forceNumInt(other._forceNumInt), 
  _treeVar(other._treeVar), _selectComp(other._selectComp)
{
  // Copy constructor

  if (other._specIntegratorConfig) {
    _specIntegratorConfig = new RooNumIntConfig(*other._specIntegratorConfig) ;
  } else {
    _specIntegratorConfig = 0 ;
  }
}



//_____________________________________________________________________________
RooAbsReal::~RooAbsReal()
{
  // Destructor

  if (_specIntegratorConfig) delete _specIntegratorConfig ;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::operator==(Double_t value) const
{
  // Equality operator comparing to a Double_t
  return (getVal()==value) ;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::operator==(const RooAbsArg& other) 
{
  // Equality operator when comparing to another RooAbsArg.
  // Only functional when the other arg is a RooAbsReal

  const RooAbsReal* otherReal = dynamic_cast<const RooAbsReal*>(&other) ;
  return otherReal ? operator==(otherReal->getVal()) : kFALSE ;
}



//_____________________________________________________________________________
TString RooAbsReal::getTitle(Bool_t appendUnit) const 
{
  // Return this variable's title string. If appendUnit is true and
  // this variable has units, also append a string " (<unit>)".
  
  TString title(GetTitle());
  if(appendUnit && 0 != strlen(getUnit())) {
    title.Append(" (");
    title.Append(getUnit());
    title.Append(")");
  }
  return title;
}



//_____________________________________________________________________________
Double_t RooAbsReal::getVal(const RooArgSet* set) const
{
  // Return value of object. If the cache is clean, return the
  // cached value, otherwise recalculate on the fly and refill
  // the cache

  if (isValueDirty() || isShapeDirty()) {

    _value = traceEval(set) ;

    clearValueDirty() ; 
    clearShapeDirty() ; 

  } else if (_cacheCheck) {
    
    // Check if cache contains value that evaluate() gives now
    Double_t checkValue = traceEval(set);

    if (checkValue != _value) {
      // If not, print warning
      coutW(Eval) << "RooAbsReal::getVal(" << GetName() << ") WARNING: cache contains " << _value 
		  << " but evaluate() returns " << checkValue << endl ;

      // And update cache (so that we see the difference)
      _value = checkValue ;
    }                                                                                                
    
  }

  return _value ;
}



//_____________________________________________________________________________
Double_t RooAbsReal::traceEval(const RooArgSet* /*nset*/) const
{
  // Calculate current value of object, with error tracing wrapper

  Double_t value = evaluate() ;
  cxcoutD(Tracing) << "RooAbsReal::getVal(" << GetName() << ") operMode = " << _operMode << " recalculated, new value = " << value << endl ;
  
  //Standard tracing code goes here
  if (!isValidReal(value)) {
    coutW(Tracing) << "RooAbsReal::traceEval(" << GetName() 
		   << "): validation failed: " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}



//_____________________________________________________________________________
Int_t RooAbsReal::getAnalyticalIntegralWN(RooArgSet& allDeps, RooArgSet& analDeps, 
					  const RooArgSet* /*normSet*/, const char* rangeName) const
{
  // Variant of getAnalyticalIntegral that is also passed the normalization set
  // that should be applied to the integrand of which the integral is request.
  // For certain operator p.d.f it is useful to overload this function rather
  // than analyticalIntegralWN() as the additional normalization information
  // may be useful in determining a more efficient decomposition of the
  // requested integral

  return _forceNumInt ? 0 : getAnalyticalIntegral(allDeps,analDeps,rangeName) ;
}



//_____________________________________________________________________________
Int_t RooAbsReal::getAnalyticalIntegral(RooArgSet& /*integSet*/, RooArgSet& /*anaIntSet*/, const char* /*rangeName*/) const
{
  // Interface function getAnalyticalIntergral advertises the
  // analytical integrals that are supported. 'integSet'
  // is the set of dependents for which integration is requested. The
  // function should copy the subset of dependents it can analytically
  // integrate to anaIntSet and return a unique identification code for
  // this integration configuration.  If no integration can be
  // performed, zero should be returned.
  
  return 0 ;
}



//_____________________________________________________________________________
Double_t RooAbsReal::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  // Implements the actual analytical integral(s) advertised by
  // getAnalyticalIntegral.  This functions will only be called with
  // codes returned by getAnalyticalIntegral, except code zero.

  if (code==0) return getVal(normSet) ;
  return analyticalIntegral(code,rangeName) ;
}



//_____________________________________________________________________________
Double_t RooAbsReal::analyticalIntegral(Int_t code, const char* /*rangeName*/) const
{
  // Implements the actual analytical integral(s) advertised by
  // getAnalyticalIntegral.  This functions will only be called with
  // codes returned by getAnalyticalIntegral, except code zero.

  // By default no analytical integrals are implemented
  coutF(Eval)  << "RooAbsReal::analyticalIntegral(" << GetName() << ") code " << code << " not implemented" << endl ;
  return 0 ;
}



//_____________________________________________________________________________
const char *RooAbsReal::getPlotLabel() const 
{
  // Get the label associated with the variable

  return _label.IsNull() ? fName.Data() : _label.Data();
}



//_____________________________________________________________________________
void RooAbsReal::setPlotLabel(const char *label) 
{
  // Set the label associated with this variable

  _label= label;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::readFromStream(istream& /*is*/, Bool_t /*compact*/, Bool_t /*verbose*/) 
{
  //Read object contents from stream (dummy for now)

  return kFALSE ;
} 



//_____________________________________________________________________________
void RooAbsReal::writeToStream(ostream& /*os*/, Bool_t /*compact*/) const
{
  //Write object contents to stream (dummy for now)
}



//_____________________________________________________________________________
void RooAbsReal::printValue(ostream& os) const
{
  // Print object value 
  os << getVal() ;
}



//_____________________________________________________________________________
void RooAbsReal::printMultiline(ostream& os, Int_t contents, Bool_t verbose, TString indent) const
{
  // Structure printing

  RooAbsArg::printMultiline(os,contents,verbose,indent) ;
  os << indent << "--- RooAbsReal ---" << endl;
  TString unit(_unit);
  if(!unit.IsNull()) unit.Prepend(' ');
  os << indent << "  Value = " << getVal() << unit << endl;
  os << endl << indent << "  Plot label is \"" << getPlotLabel() << "\"" << endl;

}


//_____________________________________________________________________________
Bool_t RooAbsReal::isValid() const 
{
  // Check if current value is valid

  return isValidReal(_value) ;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::isValidReal(Double_t /*value*/, Bool_t /*printError*/) const 
{
  // Interface function to check if given value is a valid value for this object.
  // This default implementation considers all values valid

  return kTRUE ;
}




//_____________________________________________________________________________
RooAbsReal* RooAbsReal::createProfile(const RooArgSet& paramsOfInterest) 
{
  // Create a RooProfileLL object that eliminates all nuisance parameters in the
  // present function. The nuisance parameters are defined as all parameters
  // of the function except the stated paramsOfInterest

  // Construct name of profile object
  TString name(Form("%s_Profile[",GetName())) ;
  TIterator* iter = paramsOfInterest.createIterator() ;
  RooAbsArg* arg ;
  Bool_t first(kTRUE) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (first) {
      first=kFALSE ;
    } else {
      name.Append(",") ;
    }
    name.Append(arg->GetName()) ;
  }
  delete iter ;
  name.Append("]") ;
  
  // Create and return profile object
  return new RooProfileLL(name.Data(),Form("Profile of %s",GetTitle()),*this,paramsOfInterest) ;
}
       





//_____________________________________________________________________________
RooAbsReal* RooAbsReal::createIntegral(const RooArgSet& iset, const RooCmdArg arg1, const RooCmdArg arg2,
				       const RooCmdArg arg3, const RooCmdArg arg4, const RooCmdArg arg5, 
				       const RooCmdArg arg6, const RooCmdArg arg7, const RooCmdArg arg8) const 
{
  // Create an object that represents the integral of the function over one or more observables listed in iset
  // The actual integration calculation is only performed when the return object is evaluated. The name
  // of the integral object is automatically constructed from the name of the input function, the variables
  // it integrates and the range integrates over
  //
  // The following named arguments are accepted
  //
  // NormSet(const RooArgSet&)            -- Specify normalization set, mostly useful when working with PDFS
  // NumIntConfig(const RooNumIntConfig&) -- Use given configuration for any numeric integration, if necessary
  // Range(const char* name)              -- Integrate only over given range. Multiple ranges may be specified
  //                                         by passing multiple Range() arguments  



  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::createIntegral(%s)",GetName())) ;
  pc.defineString("rangeName","RangeWithName",0,"",kTRUE) ;
  pc.defineObject("normSet","NormSet",0,0) ;
  pc.defineObject("numIntConfig","NumIntConfig",0,0) ;

  // Process & check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Extract values from named arguments
  const char* rangeName = pc.getString("rangeName",0,kTRUE) ;
  const RooArgSet* nset = static_cast<const RooArgSet*>(pc.getObject("normSet",0)) ;
  const RooNumIntConfig* cfg = static_cast<const RooNumIntConfig*>(pc.getObject("numIntConfig",0)) ;

  return createIntegral(iset,nset,cfg,rangeName) ;
}





RooAbsReal* RooAbsReal::createIntegral(const RooArgSet& iset, const RooArgSet* nset, 
				       const RooNumIntConfig* cfg, const char* rangeName) const 
{
  // Create an object that represents the integral of the function over one or more observables listed in iset
  // The actual integration calculation is only performed when the return object is evaluated. The name
  // of the integral object is automatically constructed from the name of the input function, the variables
  // it integrates and the range integrates over. If nset is specified the integrand is request
  // to be normalized over nset (only meaningful when the integrand is a pdf). If rangename is specified
  // the integral is performed over the named range, otherwise it is performed over the domain of each
  // integrated observable. If cfg is specified it will be used to configure any numeric integration
  // aspect of the integral. It will not force the integral to be performed numerically, which is
  // decided automatically by RooRealIntegral

  if (!rangeName || strchr(rangeName,',')==0) {
    // Simple case: integral over full range or single limited range
    return createIntObj(iset,nset,cfg,rangeName) ;
  } 

  // Integral over multiple ranges
  char* buf = new char[strlen(rangeName)+1] ;
  strcpy(buf,rangeName) ;
  char* range = strtok(buf,",") ;
  RooArgSet components ;
  while (range) {
    RooAbsReal* compIntegral = createIntObj(iset,nset,cfg,range) ;
    components.add(*compIntegral) ;
    range = strtok(0,",") ;
  }
  delete[] buf ;

  TString title(GetTitle()) ;
  title.Prepend("Integral of ") ;
  TString fullName(GetName()) ;
  fullName.Append(integralNameSuffix(iset,nset,rangeName)) ;

  return new RooAddition(fullName.Data(),title.Data(),components,kTRUE) ;
}



//_____________________________________________________________________________
RooAbsReal* RooAbsReal::createIntObj(const RooArgSet& iset2, const RooArgSet* nset2, 
				     const RooNumIntConfig* cfg, const char* rangeName) const 
{
  // Utility function for createIntegral that creates the actual integreal object

  // Make internal use copies of iset and nset
  RooArgSet iset(iset2) ;
  const RooArgSet* nset = nset2 ;

  // Initialize local variables perparing for recursive loop
  Bool_t error = kFALSE ;
  const RooAbsReal* integrand = this ;
  RooAbsReal* integral = 0 ;

  // Handle trivial case of no integration here explicitly
  if (iset.getSize()==0) {

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
    if (innerSet.getSize()==0) {
      error = kTRUE ; 
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

    // Remove already integrated observables from to-do list
    iset.remove(innerSet) ;

    // Send info message on recursion if needed
    if (integrand == this && iset.getSize()>0) {
      coutI(Integration) << GetName() << " : multidimensional integration over observables with parameterized ranges in terms of other integrated observables detected, using recursive integration strategy to construct final integral" << endl ;
    }

    // Prepare for recursion, next integral should integrate last integrand
    integrand = integral ;


    // Only need normalization set in innermost integration
    nset = 0 ; 
  }

  if (error) {
    coutE(Integration) << GetName() << " : ERROR while defining recursive integral over observables with parameterized integration ranges, please check that integration rangs specify uniquely defined integral " << endl; 
    delete integral ;
    integral = 0 ;
  }

  return integral ;
}



//_____________________________________________________________________________
void RooAbsReal::findInnerMostIntegration(const RooArgSet& allObs, RooArgSet& innerObs, const char* rangeName) const
{
  // Utility function for createIntObj() that aids in the construct of recursive integrals
  // over functions with multiple observables with parameterized ranges. This function
  // finds in a given set allObs over which integration is requested the largeset subset
  // of observables that can be integrated simultaneously. This subset consists of
  // observables with fixed ranges and observables with parameterized ranges whose
  // parameterization does not depend on any observable that is also integrated.

  // Make lists of 
  // a) integrated observables with fixed ranges, 
  // b) integrated observables with parameterized ranges depending on other integrated observables
  // c) integrated observables used in definition of any parameterized ranges of integrated observables
  RooArgSet obsWithFixedRange(allObs) ;
  RooArgSet obsWithParamRange ;
  RooArgSet obsServingAsRangeParams ;

  // Loop over all integrated observables
  TIterator* oiter = allObs.createIterator() ;
  RooAbsArg* aarg ;
  while((aarg=(RooAbsArg*)oiter->Next())) {
    // Check if observable is real-valued lvalue
    RooAbsRealLValue* arglv = dynamic_cast<RooAbsRealLValue*>(aarg) ;
    if (arglv) {

      // Check if range is parameterized
      RooAbsBinning& binning = arglv->getBinning(rangeName,kTRUE,kTRUE) ;
      if (binning.isParameterized()) { 
	RooArgSet* loBoundObs = binning.lowBoundFunc()->getObservables(allObs) ;
	RooArgSet* hiBoundObs = binning.highBoundFunc()->getObservables(allObs) ;

	// Check if range parameterization depends on other integrated observables
	if (loBoundObs->overlaps(allObs) || hiBoundObs->overlaps(allObs)) {
	  obsWithParamRange.add(*aarg) ;
	  obsWithFixedRange.remove(*aarg) ;
	  obsServingAsRangeParams.add(*loBoundObs,kFALSE) ;
	  obsServingAsRangeParams.add(*hiBoundObs,kFALSE) ;
	}
	delete loBoundObs ;
	delete hiBoundObs ;
      }
    }
  }
  delete oiter ;

  // Make list of fixed-range observables that are _not_ involved in the parameterization of ranges of other observables
  RooArgSet obsWithFixedRangeNP(obsWithFixedRange) ;
  obsWithFixedRangeNP.remove(obsServingAsRangeParams) ; 
  
  // Make list of param-range observables that are _not_ involved in the parameterization of ranges of other observables
  RooArgSet obsWithParamRangeNP(obsWithParamRange) ;
  obsWithParamRangeNP.remove(obsServingAsRangeParams) ; 
  
  // Construct inner-most integration: over observables (with fixed or param range) not used in any other param range definitions
  innerObs.removeAll() ;
  innerObs.add(obsWithFixedRangeNP) ;
  innerObs.add(obsWithParamRangeNP) ;

}


//_____________________________________________________________________________
TString RooAbsReal::integralNameSuffix(const RooArgSet& iset, const RooArgSet* nset, const char* rangeName, Bool_t omitEmpty) const 
{
  // Construct string with unique suffix name to give to integral object that encodes
  // integrated observables, normalization observables and the integration range name

  TString name ;
  if (iset.getSize()>0) {
    name.Append("_Int[") ;
    TIterator* iter = iset.createIterator() ;
    RooAbsArg* arg ;
    Bool_t first(kTRUE) ;
    while((arg=(RooAbsArg*)iter->Next())) {
      if (first) {
	first=kFALSE ;
      } else {
	name.Append(",") ;
      }
      name.Append(arg->GetName()) ;
    }
    delete iter ;
    if (rangeName) {
      name.Append("|") ;
      name.Append(rangeName) ;
    }
    name.Append("]");
  } else if (!omitEmpty) {
    name.Append("_Int[]") ;
  }

  if (nset && nset->getSize()>0 ) {
    name.Append("_Norm[") ;
    Bool_t first(kTRUE); 
    TIterator* iter  = nset->createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      if (first) {
	first=kFALSE ;
      } else {
	name.Append(",") ;
      }
      name.Append(arg->GetName()) ;
    }
    delete iter ;
    name.Append("]") ;
  }

  return name ;
}



//_____________________________________________________________________________
const RooAbsReal* RooAbsReal::createPlotProjection(const RooArgSet& depVars, const RooArgSet& projVars, 
                                               RooArgSet*& cloneSet) const 
{
  // Utility function for plotOn() that creates a projection of a function or p.d.f 
  // to be plotted on a RooPlot. 
  return createPlotProjection(depVars,&projVars,cloneSet) ; 
}



//_____________________________________________________________________________
const RooAbsReal* RooAbsReal::createPlotProjection(const RooArgSet& depVars, const RooArgSet& projVars) const 
{
  // Utility function for plotOn() that creates a projection of a function or p.d.f 
  // to be plotted on a RooPlot. 
  RooArgSet* cloneSet = new RooArgSet() ;
  return createPlotProjection(depVars,&projVars,cloneSet) ; 
}



//_____________________________________________________________________________
const RooAbsReal *RooAbsReal::createPlotProjection(const RooArgSet &dependentVars, const RooArgSet *projectedVars,
					       RooArgSet *&cloneSet, const char* rangeName) const 
{
  // Utility function for plotOn() that creates a projection of a function or p.d.f 
  // to be plotted on a RooPlot. 
  //
  // Create a new object G that represents the normalized projection:
  //
  //             Integral [ F[x,y,p] , { y } ]
  //  G[x,p] = ---------------------------------
  //            Integral [ F[x,y,p] , { x,y } ]
  //
  // where F[x,y,p] is the function we represent, "x" are the
  // specified dependentVars, "y" are the specified projectedVars, and
  // "p" are our remaining variables ("parameters"). Return a
  // pointer to the newly created object, or else zero in case of an
  // error.  The caller is responsible for deleting the contents of
  // cloneSet (which includes the returned projection object) 

  // Get the set of our leaf nodes
  RooArgSet leafNodes;
  RooArgSet treeNodes;
  leafNodeServerList(&leafNodes,this);
  treeNodeServerList(&treeNodes,this) ;


  // Check that the dependents are all fundamental. Filter out any that we
  // do not depend on, and make substitutions by name in our leaf list.
  // Check for overlaps with the projection variables.

  TIterator *dependentIterator= dependentVars.createIterator();
  assert(0 != dependentIterator);
  const RooAbsArg *arg = 0;
  while((arg= (const RooAbsArg*)dependentIterator->Next())) {
    if(!arg->isFundamental() && !dynamic_cast<const RooAbsLValue*>(arg)) {
      coutE(Plotting) << ClassName() << "::" << GetName() << ":createPlotProjection: variable \"" << arg->GetName()
	   << "\" of wrong type: " << arg->ClassName() << endl;
      delete dependentIterator;
      return 0;
    }

    RooAbsArg *found= treeNodes.find(arg->GetName()); 
    if(!found) {
      coutE(Plotting) << ClassName() << "::" << GetName() << ":createPlotProjection: \"" << arg->GetName()
		      << "\" is not a dependent and will be ignored." << endl;
      continue;
    }
    if(found != arg) {
      if (leafNodes.find(found->GetName())) {
	leafNodes.replace(*found,*arg);
      } else {
	leafNodes.add(*arg) ;

	// Remove any dependents of found, replace by dependents of LV node
	RooArgSet* lvDep = arg->getObservables(&leafNodes) ;
	RooAbsArg* lvs ;
	TIterator* iter = lvDep->createIterator() ;
	while((lvs=(RooAbsArg*)iter->Next())) {
	  RooAbsArg* tmp = leafNodes.find(lvs->GetName()) ;
	  if (tmp) {
	    leafNodes.remove(*tmp) ;
	    leafNodes.add(*lvs) ;
	  }
	}
	delete iter ;
	
      }
    }

    // check if this arg is also in the projection set
    if(0 != projectedVars && projectedVars->find(arg->GetName())) {
      coutE(Plotting) << ClassName() << "::" << GetName() << ":createPlotProjection: \"" << arg->GetName()
		      << "\" cannot be both a dependent and a projected variable." << endl;
      delete dependentIterator;
      return 0;
    }
  }

  // Remove the projected variables from the list of leaf nodes, if necessary.
  if(0 != projectedVars) leafNodes.remove(*projectedVars,kTRUE);

  // Make a deep-clone of ourself so later operations do not disturb our original state
  cloneSet= (RooArgSet*)RooArgSet(*this).snapshot(kTRUE);
  if (!cloneSet) {
    coutE(Plotting) << "RooAbsPdf::createPlotProjection(" << GetName() << ") Couldn't deep-clone PDF, abort," << endl ;
    return 0 ;
  }
  RooAbsReal *theClone= (RooAbsReal*)cloneSet->find(GetName());

  // The remaining entries in our list of leaf nodes are the the external
  // dependents (x) and parameters (p) of the projection. Patch them back
  // into the theClone. This orphans the nodes they replace, but the orphans
  // are still in the cloneList and so will be cleaned up eventually.
  //cout << "redirection leafNodes : " ; leafNodes.Print("1") ;

  RooArgSet* plotLeafNodes = (RooArgSet*) leafNodes.selectCommon(dependentVars) ;
  theClone->recursiveRedirectServers(*plotLeafNodes,kFALSE,kFALSE,kFALSE);
  delete plotLeafNodes ;

  // Create the set of normalization variables to use in the projection integrand
  RooArgSet normSet(dependentVars);
  if(0 != projectedVars) normSet.add(*projectedVars);

  // Try to create a valid projection integral. If no variables are to be projected,
  // create a null projection anyway to bind our normalization over the dependents
  // consistently with the way they would be bound with a non-trivial projection.
  RooArgSet empty;
  if(0 == projectedVars) projectedVars= &empty;

  TString name = GetName() ;
  name += integralNameSuffix(*projectedVars,&normSet,rangeName,kTRUE) ;

  TString title(GetTitle());  
  title.Prepend("Projection of ");

  RooRealIntegral *projected= new RooRealIntegral(name.Data(),title.Data(),*theClone,*projectedVars,&normSet,0,rangeName);
  if(0 == projected || !projected->isValid()) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":createPlotProjection: cannot integrate out ";
    projectedVars->printStream(cout,kName|kArgs,kSingleLine);
    // cleanup and exit
    if(0 != projected) delete projected;
    delete dependentIterator;
    return 0;
  }
  // Add the projection integral to the cloneSet so that it eventually gets cleaned up by the caller.
  cloneSet->addOwned(*projected);

  // cleanup
  delete dependentIterator;

  // return a const pointer to remind the caller that they do not delete the returned object
  // directly (it is contained in the cloneSet instead).
  return projected;
}




//_____________________________________________________________________________
TH1 *RooAbsReal::fillHistogram(TH1 *hist, const RooArgList &plotVars,
			       Double_t scaleFactor, const RooArgSet *projectedVars, Bool_t scaleForDensity) const 
{
  // Fill the ROOT histogram 'hist' with values sampled from this
  // function at the bin centers.  Our value is calculated by first
  // integrating out any variables in projectedVars and then scaling
  // the result by scaleFactor. Returns a pointer to the input
  // histogram, or zero in case of an error. The input histogram can
  // be any TH1 subclass, and therefore of arbitrary
  // dimension. Variables are matched with the (x,y,...) dimensions of
  // the input histogram according to the order in which they appear
  // in the input plotVars list. If scaleForDensity is true the
  // histogram is filled with a the functions density rather than
  // the functions value (i.e. the value at the bin center is multiplied
  // with bin volume)

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


  // Check that the plot variables are all actually RooRealVars and print a warning if we do not
  // explicitly depend on one of them. Fill a set (not list!) of cloned plot variables.
  RooArgSet plotClones;
  for(Int_t index= 0; index < plotVars.getSize(); index++) {
    const RooAbsArg *var= plotVars.at(index);
    const RooRealVar *realVar= dynamic_cast<const RooRealVar*>(var);
    if(0 == realVar) {
      coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillHistogram: cannot plot variable \"" << var->GetName()
	   << "\" of type " << var->ClassName() << endl;
      return 0;
    }
    if(!this->dependsOn(*realVar)) {
      coutE(InputArguments) << ClassName() << "::" << GetName()
	   << ":fillHistogram: WARNING: variable is not an explicit dependent: " << realVar->GetName() << endl;
    }
    plotClones.addClone(*realVar,kTRUE); // do not complain about duplicates
  }

  // Reconnect all plotClones to each other, imported when plotting N-dim integrals with entangled parameterized ranges
  TIterator* pciter= plotClones.createIterator() ;
  RooAbsArg* pc ;
  while((pc=(RooAbsArg*)pciter->Next())) {
    pc->recursiveRedirectServers(plotClones,kFALSE,kFALSE,kTRUE) ;
  }

  delete pciter ;
  


  // Call checkObservables
  RooArgSet allDeps(plotClones) ;
  if (projectedVars) allDeps.add(*projectedVars) ;
  if (checkObservables(&allDeps)) {
    coutE(InputArguments) << "RooAbsReal::fillHistogram(" << GetName() << ") error in checkObservables, abort" << endl ;
    return hist ;
  }

  // Create a standalone projection object to use for calculating bin contents
  RooArgSet *cloneSet = 0;
  const RooAbsReal *projected= createPlotProjection(plotClones,projectedVars,cloneSet);
  cxcoutD(Plotting) << "RooAbsReal::fillHistogram(" << GetName() << ") plot projection object is " << projected->GetName() << endl ;


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
			  << hdim << " dimensions" << endl;
    break;
  }

  // Loop over the input histogram's bins and fill each one with our projection's
  // value, calculated at the center.
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
      coutE(InputArguments) << "RooAbsReal::fillHistogram: Internal Error!" << endl;
      break;
    }
    Double_t result= scaleFactor*projected->getVal();
    hist->SetBinContent(hist->GetBin(xbin,ybin,zbin),result);
    //cout << "bin " << bin << " -> (" << xbin << "," << ybin << "," << zbin << ") = " << result << endl;
  }

  // cleanup
  delete cloneSet;

  return hist;
}



//_____________________________________________________________________________
RooDataHist* RooAbsReal::fillDataHist(RooDataHist *hist, const RooArgSet* normSet, 
				      Double_t scaleFactor, Bool_t correctForBinSize, Bool_t showProgress) const 
{
  // Fill a RooDataHist with values sampled from this function at the
  // bin centers.  Our value is calculated by first integrating out
  // any variables in projectedVars and then scaling the result by
  // scaleFactor. Returns a pointer to the input RooDataHist, or zero
  // in case of an error. If correctForBinSize is true the RooDataHist
  // is filled with the functions density (function value times the
  // bin volume) rather than function value.  If showProgress is true
  // a process indicator is printed on stdout in steps of one percent,
  // which is mostly useful for the sampling of expensive functions
  // such as likelihoods

  // Do we have a valid histogram to use?
  if(0 == hist) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":fillDataHist: no valid RooDataHist to fill" << endl;
    return 0;
  }

  // Call checkObservables
  RooArgSet allDeps(*hist->get()) ;
  if (checkObservables(&allDeps)) {
    coutE(InputArguments) << "RooAbsReal::fillDataHist(" << GetName() << ") error in checkObservables, abort" << endl ;
    return hist ;
  }
  
  // Make deep clone of self and attach to dataset observables
  RooArgSet* cloneSet = (RooArgSet*) RooArgSet(*this).snapshot(kTRUE) ;
  RooAbsReal* theClone = (RooAbsReal*) cloneSet->find(GetName()) ;
  theClone->attachDataSet(*hist) ;

  // Iterator over all bins of RooDataHist and fill weights
  Int_t onePct = hist->numEntries()/100 ;
  if (onePct==0) {
    onePct++ ;
  }
  for (Int_t i=0 ; i<hist->numEntries() ; i++) {    
    if (showProgress && (i%onePct==0)) {
      ccoutP(Eval) << "." << flush ;
    }
    const RooArgSet* obs = hist->get(i) ;
    Double_t binVal = theClone->getVal(normSet?normSet:obs)*scaleFactor ;
    if (correctForBinSize) binVal*= hist->binVolume() ;
    hist->set(binVal) ;
  }

  delete cloneSet ;

  return hist;
}




//_____________________________________________________________________________
TH1* RooAbsReal::createHistogram(const char* varNameList, Int_t xbins, Int_t ybins, Int_t zbins) const 
{
  // Create and fill a ROOT histogram TH1,TH2 or TH3 with the values of this function for the variables with given names
  // The number of bins can be controlled using the [xyz]bins parameters. For a greater degree of control
  // use the createHistogram() method below with named arguments
  //
  // The caller takes ownership of the returned histogram

  // Parse list of variable names
  char buf[1024] ;
  strcpy(buf,varNameList) ;
  char* varName = strtok(buf,",:") ;

  RooArgSet* vars = getVariables() ;
  
  RooRealVar* xvar = (RooRealVar*) vars->find(varName) ;
  varName = strtok(0,",") ;
  RooRealVar* yvar = varName ? (RooRealVar*) vars->find(varName) : 0 ;
  varName = strtok(0,",") ;
  RooRealVar* zvar = varName ? (RooRealVar*) vars->find(varName) : 0 ;

  delete vars ;

  // Construct list of named arguments to pass to the implementation version of createHistogram()

  RooLinkedList argList ; 
  if (xbins>0) {
    argList.Add(RooFit::Binning(xbins).Clone()) ;
  }
 
  if (yvar) {        
    if (ybins>0) {
      argList.Add(RooFit::YVar(*yvar,RooFit::Binning(ybins)).Clone()) ;
    } else {
      argList.Add(RooFit::YVar(*yvar).Clone()) ;
    }
  }


  if (zvar) {        
    if (zbins>0) {
      argList.Add(RooFit::ZVar(*zvar,RooFit::Binning(zbins)).Clone()) ;
    } else {
      argList.Add(RooFit::ZVar(*zvar).Clone()) ;
    }
  }


  // Call implementation function
  TH1* result = createHistogram(GetName(),*xvar,argList) ;

  // Delete temporary list of RooCmdArgs 
  argList.Delete() ;

  return result ;
}



//_____________________________________________________________________________
TH1 *RooAbsReal::createHistogram(const char *name, const RooAbsRealLValue& xvar,
				 const RooCmdArg& arg1, const RooCmdArg& arg2, const RooCmdArg& arg3, const RooCmdArg& arg4, 
				 const RooCmdArg& arg5, const RooCmdArg& arg6, const RooCmdArg& arg7, const RooCmdArg& arg8) const 
{
  // Create and fill a ROOT histogram TH1,TH2 or TH3 with the values of this function. 
  //
  // This function accepts the following arguments
  //
  // name -- Name of the ROOT histogram
  // xvar -- Observable to be mapped on x axis of ROOT histogram
  //
  // Binning(const char* name)                    -- Apply binning with given name to x axis of histogram
  // Binning(RooAbsBinning& binning)              -- Apply specified binning to x axis of histogram
  // Binning(int nbins, [double lo, double hi])   -- Apply specified binning to x axis of histogram
  // ConditionalObservables(const RooArgSet& set) -- Do not normalized PDF over following observables when projecting PDF into histogram
  // Scaling(Bool_t)                              -- Apply density-correction scaling (multiply by bin volume), default is kTRUE
  //
  // YVar(const RooAbsRealLValue& var,...)    -- Observable to be mapped on y axis of ROOT histogram
  // ZVar(const RooAbsRealLValue& var,...)    -- Observable to be mapped on z axis of ROOT histogram
  //
  // The YVar() and ZVar() arguments can be supplied with optional Binning() arguments to control the binning of the Y and Z axes, e.g.
  // createHistogram("histo",x,Binning(-1,1,20), YVar(y,Binning(-1,1,30)), ZVar(z,Binning("zbinning")))
  //
  // The caller takes ownership of the returned histogram


  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;

  return createHistogram(name,xvar,l) ;
}


//_____________________________________________________________________________
TH1* RooAbsReal::createHistogram(const char *name, const RooAbsRealLValue& xvar, RooLinkedList& argList) const 
{
  // Internal method implementing createHistogram


  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::createHistogram(%s)",GetName())) ;
  pc.defineInt("scaling","Scaling",0,1) ;
  pc.defineObject("projObs","ProjectedObservables",0,0) ;
  pc.defineObject("yvar","YVar",0,0) ;
  pc.defineObject("zvar","ZVar",0,0) ;  
  pc.allowUndefined() ;

  // Process & check varargs 
  pc.process(argList) ;
  if (!pc.ok(kTRUE)) {
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

  RooArgSet* projObs = static_cast<RooArgSet*>(pc.getObject("projObs")) ;

  Bool_t doScaling = pc.getInt("scaling") ;

  // Strip any 'Scaling' commands from list forwarded to createHistogram
  RooLinkedList l2 ;
  for (Int_t i=0 ; i<argList.GetSize() ; i++) {
    if (TString(argList.At(i)->GetName()).CompareTo("Scaling")) {
      l2.Add(argList.At(i)) ;
    }
  }

  TH1* histo = xvar.createHistogram(name,l2) ;
  fillHistogram(histo,vars,1.0,projObs,doScaling) ;

  return histo ;
}



//_____________________________________________________________________________
RooPlot* RooAbsReal::plotOn(RooPlot* frame, const RooCmdArg& arg1, const RooCmdArg& arg2,
			    const RooCmdArg& arg3, const RooCmdArg& arg4,
			    const RooCmdArg& arg5, const RooCmdArg& arg6,
			    const RooCmdArg& arg7, const RooCmdArg& arg8,
			    const RooCmdArg& arg9, const RooCmdArg& arg10) const
{
  // Plot (project) PDF on specified frame. If a PDF is plotted in an empty frame, it
  // will show a unit normalized curve in the frame variable, taken at the present value 
  // of other observables defined for this PDF
  //
  // If a PDF is plotted in a frame in which a dataset has already been plotted, it will
  // show a projected curve integrated over all variables that were present in the shown
  // dataset except for the one on the x-axis. The normalization of the curve will also
  // be adjusted to the event count of the plotted dataset. An informational message
  // will be printed for each projection step that is performed
  //
  // This function takes the following named arguments
  //
  // Projection control
  // ------------------
  // Slice(const RooArgSet& set)     -- Override default projection behaviour by omittting observables listed 
  //                                    in set from the projection, resulting a 'slice' plot. Slicing is usually
  //                                    only sensible in discrete observables. The slice is position at the 'current'
  //                                    value of the observable objects
  //
  // Slice(RooCategory& cat,         -- Override default projection behaviour by omittting specified category 
  //       const char* label)           observable from the projection, resulting in a 'slice' plot. The slice is positioned
  //                                    at the given label value. Multiple Slice() commands can be given to specify slices
  //                                    in multiple observables
  //
  // Project(const RooArgSet& set)   -- Override default projection behaviour by projecting over observables
  //                                    given in set and complete ignoring the default projection behavior. Advanced use only.
  //
  // ProjWData(const RooAbsData& d)  -- Override default projection _technique_ (integration). For observables present in given dataset
  //                                    projection of PDF is achieved by constructing an average over all observable values in given set.
  //                                    Consult RooFit plotting tutorial for further explanation of meaning & use of this technique
  //
  // ProjWData(const RooArgSet& s,   -- As above but only consider subset 's' of observables in dataset 'd' for projection through data averaging
  //           const RooAbsData& d)
  //
  // ProjectionRange(const char* rn) -- Override default range of projection integrals to a different range speficied by given range name.
  //                                    This technique allows you to project a finite width slice in a real-valued observable
  //
  // NumCPU(Int_t ncpu)              -- Number of CPUs to use simultaneously to calculate data-weighted projections (only in combination with ProjWData)
  //
  //
  // Misc content control
  // --------------------
  // PrintEvalErrors(Int_t numErr)   -- Control number of p.d.f evaluation errors printed per curve. A negative
  //                                    value suppress output completely, a zero value will only print the error count per p.d.f component,
  //                                    a positive value is will print details of each error up to numErr messages per p.d.f component.
  // 
  // EvalErrorValue(Double_t value)  -- Set curve points at which (pdf) evaluation error occur to specified value. By default the 
  //                                    function value is plotted.
  //
  // Normalization(Double_t scale,   -- Adjust normalization by given scale factor. Interpretation of number depends on code: Relative:
  //                ScaleType code)     relative adjustment factor, NumEvent: scale to match given number of events.
  //
  // Name(const chat* name)          -- Give curve specified name in frame. Useful if curve is to be referenced later
  //
  // Asymmetry(const RooCategory& c) -- Show the asymmetry of the PDF in given two-state category [F(+)-F(-)] / [F(+)+F(-)] rather than
  //                                    the PDF projection. Category must have two states with indices -1 and +1 or three states with
  //                                    indeces -1,0 and +1.
  //
  // ShiftToZero(Bool_t flag)        -- Shift entire curve such that lowest visible point is at exactly zero. Mostly useful when
  //                                    plotting -log(L) or chi^2 distributions
  //
  // AddTo(const char* name,         -- Add constructed projection to already existing curve with given name and relative weight factors
  //                                    double_t wgtSelf, double_t wgtOther)
  //
  // Plotting control 
  // ----------------
  // DrawOption(const char* opt)     -- Select ROOT draw option for resulting TGraph object
  //
  // LineStyle(Int_t style)          -- Select line style by ROOT line style code, default is solid
  //
  // LineColor(Int_t color)          -- Select line color by ROOT color code, default is blue
  //
  // LineWidth(Int_t width)          -- Select line with in pixels, default is 3
  //
  // FillStyle(Int_t style)          -- Select fill style, default is not filled. If a filled style is selected, also use VLines()
  //                                    to add vertical downward lines at end of curve to ensure proper closure
  // FillColor(Int_t color)          -- Select fill color by ROOT color code
  //
  // Range(const char* name)         -- Only draw curve in range defined by given name
  //
  // Range(double lo, double hi)     -- Only draw curve in specified range
  //
  // VLines()                        -- Add vertical lines to y=0 at end points of curve
  //
  // Precision(Double_t eps)         -- Control precision of drawn curve w.r.t to scale of plot, default is 1e-3. Higher precision
  //                                    will result in more and more densely spaced curve points
  //
  // Invisible(Bool_t flag)           -- Add curve to frame, but do not display. Useful in combination AddTo()


  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  l.Add((TObject*)&arg9) ;  l.Add((TObject*)&arg10) ;
  return plotOn(frame,l) ;
}



//_____________________________________________________________________________
RooPlot* RooAbsReal::plotOn(RooPlot* frame, RooLinkedList& argList) const
{
  // Internal back-end function of plotOn() with named arguments

  // Special handling here if argList contains RangeWithName argument with multiple
  // range names -- Need to translate this call into multiple calls

  RooCmdArg* rcmd = (RooCmdArg*) argList.FindObject("RangeWithName") ;
  if (rcmd && TString(rcmd->getString(0)).Contains(",")) {

    // List joint ranges as choice of normalization for all later processing
    RooCmdArg rnorm = RooFit::NormRange(rcmd->getString(0)) ;
    argList.Add(&rnorm) ;

    list<string> rlist ;

    // Separate named ranges using strtok
    char buf[1024] ;
    strcpy(buf,rcmd->getString(0)) ;
    char* oneRange = strtok(buf,",") ;
    while(oneRange) {
      rlist.push_back(oneRange) ;
      oneRange = strtok(0,",") ;      
    }

    for (list<string>::iterator riter=rlist.begin() ; riter!=rlist.end() ; ++riter) {
      // Process each range with a separate command with a single range to be plotted
      rcmd->setString(0,riter->c_str()) ;
      RooAbsReal::plotOn(frame,argList) ;
    }
    return frame ;
    
  }

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::plotOn(%s)",GetName())) ;
  pc.defineString("drawOption","DrawOption",0,"L") ;
  pc.defineString("projectionRangeName","ProjectionRange",0,"",kTRUE) ;
  pc.defineString("curveNameSuffix","CurveNameSuffix",0,"") ;
  pc.defineString("sliceCatState","SliceCat",0,"",kTRUE) ;
  pc.defineDouble("scaleFactor","Normalization",0,1.0) ;
  pc.defineObject("sliceSet","SliceVars",0) ;
  pc.defineObject("sliceCatList","SliceCat",0,0,kTRUE) ;
  pc.defineObject("projSet","Project",0) ;
  pc.defineObject("asymCat","Asymmetry",0) ;
  pc.defineDouble("precision","Precision",0,1e-3) ;
  pc.defineDouble("evalErrorVal","EvalErrorValue",0,0) ;
  pc.defineInt("doEvalError","EvalErrorValue",0,0) ;
  pc.defineInt("shiftToZero","ShiftToZero",0,0) ;  
  pc.defineObject("projDataSet","ProjData",0) ;
  pc.defineObject("projData","ProjData",1) ;
  pc.defineInt("binProjData","ProjData",0,0) ;
  pc.defineDouble("rangeLo","Range",0,-999.) ;
  pc.defineDouble("rangeHi","Range",1,-999.) ;
  pc.defineInt("numee","PrintEvalErrors",0,10) ;
  pc.defineInt("rangeAdjustNorm","Range",0,0) ;
  pc.defineInt("rangeWNAdjustNorm","RangeWithName",0,0) ;
  pc.defineInt("VLines","VLines",0,2) ; // 2==ExtendedWings
  pc.defineString("rangeName","RangeWithName",0,"") ;
  pc.defineString("normRangeName","NormRange",0,"") ;
  pc.defineInt("lineColor","LineColor",0,-999) ;
  pc.defineInt("lineStyle","LineStyle",0,-999) ;
  pc.defineInt("lineWidth","LineWidth",0,-999) ;
  pc.defineInt("fillColor","FillColor",0,-999) ;
  pc.defineInt("fillStyle","FillStyle",0,-999) ;
  pc.defineString("curveName","Name",0,"") ;
  pc.defineInt("curveInvisible","Invisible",0,0) ;
  pc.defineInt("numCPU","NumCPU",0,1) ;
  pc.defineInt("interleave","NumCPU",1,0) ; 
  pc.defineString("addToCurveName","AddTo",0,"") ;
  pc.defineDouble("addToWgtSelf","AddTo",0,1.) ;
  pc.defineDouble("addToWgtOther","AddTo",1,1.) ;
  pc.defineInt("moveToBack","MoveToBack",0,0) ;
  pc.defineMutex("SliceVars","Project") ;
  pc.defineMutex("AddTo","Asymmetry") ;
  pc.defineMutex("Range","RangeWithName") ;

  // Process & check varargs 
  pc.process(argList) ;
  if (!pc.ok(kTRUE)) {
    return frame ;
  }

  PlotOpt o ;


  // Extract values from named arguments
  o.numee       = pc.getInt("numee") ;
  o.drawOptions = pc.getString("drawOption") ;
  o.curveNameSuffix = pc.getString("curveNameSuffix") ;
  o.scaleFactor = pc.getDouble("scaleFactor") ;
  o.projData = (const RooAbsData*) pc.getObject("projData") ;
  o.binProjData = pc.getInt("binProjData") ;
  o.projDataSet = (const RooArgSet*) pc.getObject("projDataSet") ;
  o.numCPU = pc.getInt("numCPU") ;
  o.interleave = pc.getInt("interleave") ;
  o.eeval      = pc.getDouble("evalErrorVal") ;
  o.doeeval   = pc.getInt("doEvalError") ;

  const RooArgSet* sliceSetTmp = (const RooArgSet*) pc.getObject("sliceSet") ;
  RooArgSet* sliceSet = sliceSetTmp ? ((RooArgSet*) sliceSetTmp->Clone()) : 0 ;
  const RooArgSet* projSet = (const RooArgSet*) pc.getObject("projSet") ;
  const RooAbsCategoryLValue* asymCat = (const RooAbsCategoryLValue*) pc.getObject("asymCat") ;


  // Look for category slice arguments and add them to the master slice list if found
  const char* sliceCatState = pc.getString("sliceCatState",0,kTRUE) ;
  const RooLinkedList& sliceCatList = pc.getObjectList("sliceCatList") ;
  if (sliceCatState) {

    // Make the master slice set if it doesnt exist
    if (!sliceSet) {
      sliceSet = new RooArgSet ;
    }

    // Prepare comma separated label list for parsing
    char buf[1024] ;
    strcpy(buf,sliceCatState) ;
    const char* slabel = strtok(buf,",") ;

    // Loop over all categories provided by (multiple) Slice() arguments
    TIterator* iter = sliceCatList.MakeIterator() ;
    RooCategory* scat ;
    while((scat=(RooCategory*)iter->Next())) {
      if (slabel) {
	// Set the slice position to the value indicate by slabel
	scat->setLabel(slabel) ;
	// Add the slice category to the master slice set
	sliceSet->add(*scat,kFALSE) ;
      }
      slabel = strtok(0,",") ;
    }
    delete iter ;
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
    o.normRangeName = pc.getString("rangeName",0,kTRUE) ;
    o.rangeLo = frame->getPlotVar()->getMin(pc.getString("rangeName",0,kTRUE)) ;
    o.rangeHi = frame->getPlotVar()->getMax(pc.getString("rangeName",0,kTRUE)) ;
    o.postRangeFracScale = pc.getInt("rangeWNAdjustNorm") ;
    if (vlines==2) vlines=0 ; // Default is NoWings if range was specified
  } 


  // If separate normalization range was specified this overrides previous settings
  if (pc.hasProcessed("NormRange")) {
    o.normRangeName = pc.getString("normRangeName") ;
    o.postRangeFracScale = kTRUE ;    
  }

  o.wmode = (vlines==2)?RooCurve::Extended:(vlines==1?RooCurve::Straight:RooCurve::NoWings) ;
  o.projectionRangeName = pc.getString("projectionRangeName",0,kTRUE) ;
  o.curveName = pc.getString("curveName",0,kTRUE) ;
  o.curveInvisible = pc.getInt("curveInvisible") ;
  o.addToCurveName = pc.getString("addToCurveName",0,kTRUE) ;
  o.addToWgtSelf = pc.getDouble("addToWgtSelf") ;
  o.addToWgtOther = pc.getDouble("addToWgtOther") ;  

  if (o.addToCurveName && !frame->findObject(o.addToCurveName,RooCurve::Class())) {
    coutE(InputArguments) << "RooAbsReal::plotOn(" << GetName() << ") cannot find existing curve " << o.addToCurveName << " to add to in RooPlot" << endl ;
    return frame ;
  }
  
  RooArgSet projectedVars ;
  if (sliceSet) {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") Preprocessing: have slice " << *sliceSet << endl ;

    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
    
    // Take out the sliced variables
    TIterator* iter = sliceSet->createIterator() ;
    RooAbsArg* sliceArg ;
    while((sliceArg=(RooAbsArg*)iter->Next())) {
      RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
      if (arg) {
	projectedVars.remove(*arg) ;
      } else {
	coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") slice variable " 
			<< sliceArg->GetName() << " was not projected anyway" << endl ;
      }
    }
    delete iter ;
  } else if (projSet) {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") Preprocessing: have projSet " << *projSet << endl ;
    makeProjectionSet(frame->getPlotVar(),projSet,projectedVars,kFALSE) ;
  } else {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") Preprocessing: have neither sliceSet nor projSet " << endl ;
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  }
  o.projSet = &projectedVars ;

  cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") Preprocessing: projectedVars = " << projectedVars << endl ;


  RooPlot* ret ;
  if (!asymCat) {
    // Forward to actual calculation
    ret = RooAbsReal::plotOn(frame,o) ;
  } else {        
    // Forward to actual calculation
    ret = RooAbsReal::plotAsymOn(frame,*asymCat,o) ;
  }

  delete sliceSet ;

  // Optionally adjust line/fill attributes
  Int_t lineColor = pc.getInt("lineColor") ;
  Int_t lineStyle = pc.getInt("lineStyle") ;
  Int_t lineWidth = pc.getInt("lineWidth") ;
  Int_t fillColor = pc.getInt("fillColor") ;
  Int_t fillStyle = pc.getInt("fillStyle") ;
  if (lineColor!=-999) ret->getAttLine()->SetLineColor(lineColor) ;
  if (lineStyle!=-999) ret->getAttLine()->SetLineStyle(lineStyle) ;
  if (lineWidth!=-999) ret->getAttLine()->SetLineWidth(lineWidth) ;
  if (fillColor!=-999) ret->getAttFill()->SetFillColor(fillColor) ;
  if (fillStyle!=-999) ret->getAttFill()->SetFillStyle(fillStyle) ;

  // Move last inserted object to back to drawing stack if requested
  if (pc.getInt("moveToBack") && frame->numItems()>1) {   
    frame->drawBefore(frame->getObject(0)->GetName(), frame->getCurve()->GetName());    
  }
  
  return ret ;
}




//_____________________________________________________________________________
RooPlot* RooAbsReal::plotOn(RooPlot *frame, PlotOpt o) const
{
  // Plotting engine function for internal use
  // 
  // Plot ourselves on given frame. If frame contains a histogram, all dimensions of the plotted
  // function that occur in the previously plotted dataset are projected via partial integration,
  // otherwise no projections are performed. Optionally, certain projections can be performed
  // by summing over the values present in a provided dataset ('projData'), to correctly
  // project out data dependents that are not properly described by the PDF (e.g. per-event errors).
  //
  // The functions value can be multiplied with an optional scale factor. The interpretation
  // of the scale factor is unique for generic real functions, for PDFs there are various interpretations
  // possible, which can be selection with 'stype' (see RooAbsPdf::plotOn() for details).
  //
  // The default projection behaviour can be overriden by supplying an optional set of dependents
  // to project. For most cases, plotSliceOn() and plotProjOn() provide a more intuitive interface
  // to modify the default projection behavour.

  // Sanity checks
  if (plotSanityChecks(frame)) return frame ;

  // ProjDataVars is either all projData observables, or the user indicated subset of it
  RooArgSet projDataVars ;
  if (o.projData) {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") have ProjData with observables = " << *o.projData->get() << endl ;
    if (o.projDataSet) {
      RooArgSet* tmp = (RooArgSet*) o.projData->get()->selectCommon(*o.projDataSet) ;
      projDataVars.add(*tmp) ;
      cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") have ProjDataSet = " << *o.projDataSet << " will only use this subset of projData" << endl ;
      delete tmp ;
    } else {
      cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") using full ProjData" << endl ;
      projDataVars.add(*o.projData->get()) ;
    }
  }

  cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") ProjDataVars = " << projDataVars << endl ;

  // Make list of variables to be projected
  RooArgSet projectedVars ;
  RooArgSet sliceSet ;
  if (o.projSet) {
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") have input projSet = " << *o.projSet << endl ;
    makeProjectionSet(frame->getPlotVar(),o.projSet,projectedVars,kFALSE) ;
    cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") calculated projectedVars = " << *o.projSet << endl ;

    // Print list of non-projected variables
    if (frame->getNormVars()) {
      RooArgSet *sliceSetTmp = getObservables(*frame->getNormVars()) ;

      cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") frame->getNormVars() that are also observables = " << *sliceSetTmp << endl ;

      sliceSetTmp->remove(projectedVars,kTRUE,kTRUE) ;
      sliceSetTmp->remove(*frame->getPlotVar(),kTRUE,kTRUE) ;

      if (o.projData) {
	RooArgSet* tmp = (RooArgSet*) projDataVars.selectCommon(*o.projSet) ;
	sliceSetTmp->remove(*tmp,kTRUE,kTRUE) ;
	delete tmp ;
      }

      if (sliceSetTmp->getSize()) {
	coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot on " 
			<< frame->getPlotVar()->GetName() << " represents a slice in " << *sliceSetTmp << endl ;
      }
      sliceSet.add(*sliceSetTmp) ;
      delete sliceSetTmp ;
    }
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  }

  cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") projectedVars = " << projectedVars << " sliceSet = " << sliceSet << endl ;


  RooArgSet* projDataNeededVars = 0 ;
  // Take out data-projected dependents from projectedVars
  if (o.projData) {
    projDataNeededVars = (RooArgSet*) projectedVars.selectCommon(projDataVars) ;
    projectedVars.remove(projDataVars,kTRUE,kTRUE) ;
  }

  // Clone the plot variable
  RooAbsReal* realVar = (RooRealVar*) frame->getPlotVar() ;
  RooArgSet* plotCloneSet = (RooArgSet*) RooArgSet(*realVar).snapshot(kTRUE) ;
  if (!plotCloneSet) {
    coutE(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") Couldn't deep-clone self, abort," << endl ;
    return frame ;
  }
  RooRealVar* plotVar = (RooRealVar*) plotCloneSet->find(realVar->GetName());

  // Inform user about projections
  if (projectedVars.getSize()) {
    coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName() 
		    << " integrates over variables " << projectedVars 
		    << (o.projectionRangeName?Form(" in range %s",o.projectionRangeName):"") << endl;
  }  
  if (projDataNeededVars && projDataNeededVars->getSize()>0) {
    coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName() 
		    << " averages using data variables " << *projDataNeededVars << endl ;
  }

  // Create projection integral
  RooArgSet* projectionCompList ;

  RooArgSet* deps = getObservables(frame->getNormVars()) ;
  deps->remove(projectedVars,kTRUE,kTRUE) ;
  if (projDataNeededVars) {
    deps->remove(*projDataNeededVars,kTRUE,kTRUE) ;
  }
  deps->remove(*plotVar,kTRUE,kTRUE) ;
  deps->add(*plotVar) ;

  // Now that we have the final set of dependents, call checkObservables()
  if (checkObservables(deps)) {
    coutE(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") error in checkObservables, abort" << endl ;
    delete deps ;
    delete plotCloneSet ;
    if (projDataNeededVars) delete projDataNeededVars ;
    return frame ;
  }

  RooArgSet normSet(*deps) ;
  //normSet.add(projDataVars) ;

  RooAbsReal *projection = (RooAbsReal*) createPlotProjection(normSet, &projectedVars, projectionCompList, o.projectionRangeName) ;
  cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot projection object is " << projection->GetName() << endl ;
  if (dologD(Plotting)) {
    projection->printStream(ccoutD(Plotting),0,kVerbose) ;
  }

  // Always fix RooAddPdf normalizations
  RooArgSet fullNormSet(*deps) ;
  fullNormSet.add(projectedVars) ;
  if (projDataNeededVars && projDataNeededVars->getSize()>0) {
    fullNormSet.add(*projDataNeededVars) ;
  }
  RooArgSet* compSet = projection->getComponents() ;
  TIterator* iter = compSet->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(arg) ;
    if (pdf) {
      pdf->selectNormalization(&fullNormSet) ;
    } 
  }
  delete iter ;
  delete compSet ;
  
  
  // Apply data projection, if requested
  if (o.projData && projDataNeededVars && projDataNeededVars->getSize()>0) {

    // If data set contains more rows than needed, make reduced copy first
    RooAbsData* projDataSel = (RooAbsData*)o.projData;

    if (projDataNeededVars->getSize()<o.projData->get()->getSize()) {
      
      // Determine if there are any slice variables in the projection set
      RooArgSet* sliceDataSet = (RooArgSet*) sliceSet.selectCommon(*o.projData->get()) ;
      TString cutString ;
      if (sliceDataSet->getSize()>0) {
	TIterator* iter2 = sliceDataSet->createIterator() ;
	RooAbsArg* sliceVar ; 
	Bool_t first(kTRUE) ;
	while((sliceVar=(RooAbsArg*)iter2->Next())) {
	  if (!first) {
	    cutString.Append("&&") ;
	  } else {
	    first=kFALSE ;	    
	  }

	  RooAbsRealLValue* real ;
	  RooAbsCategoryLValue* cat ;	  
	  if ((real = dynamic_cast<RooAbsRealLValue*>(sliceVar))) {
	    cutString.Append(Form("%s==%f",real->GetName(),real->getVal())) ;
	  } else if ((cat = dynamic_cast<RooAbsCategoryLValue*>(sliceVar))) {
	    cutString.Append(Form("%s==%d",cat->GetName(),cat->getIndex())) ;	    
	  }
	}
	delete iter2 ;
      }
      delete sliceDataSet ;

      if (!cutString.IsNull()) {
	projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars,cutString) ;
	coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") reducing given projection dataset to entries with " << cutString << endl ;
      } else {
	projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars) ;
      }
      coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() 
		      << ") only the following components of the projection data will be used: " << *projDataNeededVars << endl ;
    }

    // Request binning of unbinned projection dataset that consists exclusively of category observables
    if (!o.binProjData && dynamic_cast<RooDataSet*>(projDataSel)!=0) {
      
      // Determine if dataset contains only categories
      TIterator* iter2 = projDataSel->get()->createIterator() ;
      Bool_t allCat(kTRUE) ;
      RooAbsArg* arg2 ;
      while((arg2=(RooAbsArg*)iter2->Next())) {
	if (!dynamic_cast<RooCategory*>(arg2)) allCat = kFALSE ;
      }
      delete iter2 ;
      if (allCat) {
	o.binProjData = kTRUE ;
	coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") unbinned projection dataset consist only of discrete variables,"
			<< " performing projection with binned copy for optimization." << endl ;
	
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
    RooDataWeightedAverage dwa(Form("%sDataWgtAvg",GetName()),"Data Weighted average",*projection,*projDataSel,o.numCPU,o.interleave,kTRUE) ;
    dwa.constOptimizeTestStatistic(Activate) ;

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
    RooAbsReal::enableEvalErrorLogging(kTRUE) ;
    RooCurve *curve = new RooCurve(projection->GetName(),projection->GetTitle(),scaleBind,
				   o.rangeLo,o.rangeHi,frame->GetNbinsX(),o.precision,o.precision,o.shiftToZero,o.wmode,o.numee,o.doeeval,o.eeval) ;
    RooAbsReal::enableEvalErrorLogging(kFALSE) ;

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
    frame->addPlotable(curve, o.drawOptions);

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
      Bool_t tmp = _globalSelectComp ;
      globalSelectComp(kTRUE) ;
      RooAbsReal* intFrac = projection->createIntegral(*plotVar,*plotVar,o.normRangeName) ;
      globalSelectComp(kTRUE) ;
      o.scaleFactor /= intFrac->getVal() ;
      globalSelectComp(tmp) ;
      delete intFrac ;

    }

    // create a new curve of our function using the clone to do the evaluations
    // Curve constructor for regular projections

    RooAbsReal::enableEvalErrorLogging(kTRUE) ;
    RooCurve *curve = new RooCurve(*projection,*plotVar,o.rangeLo,o.rangeHi,frame->GetNbinsX(),
				   o.scaleFactor,0,o.precision,o.precision,o.shiftToZero,o.wmode,o.numee,o.doeeval,o.eeval);
    RooAbsReal::enableEvalErrorLogging(kFALSE) ;



    // Set default name of curve
    TString curveName(projection->GetName()) ;
    if (sliceSet.getSize()>0) {
      curveName.Append(Form("_Slice[%s]",sliceSet.contentsString().c_str())) ;      
    }
    if (o.curveNameSuffix) {
      // Append any suffixes imported from RooAbsPdf::plotOn
      curveName.Append(o.curveNameSuffix) ;
    }
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

  if (projDataNeededVars) delete projDataNeededVars ;
  delete deps ;
  delete projectionCompList ;
  delete plotCloneSet ;
  return frame;
}




//_____________________________________________________________________________
RooPlot* RooAbsReal::plotSliceOn(RooPlot *frame, const RooArgSet& sliceSet, Option_t* drawOptions, 
				 Double_t scaleFactor, ScaleType stype, const RooAbsData* projData) const
{
  // OBSOLETE -- RETAINED FOR BACKWARD COMPATIBILITY. Use the plotOn(frame,Slice(...)) instead

  RooArgSet projectedVars ;
  makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  
  // Take out the sliced variables
  TIterator* iter = sliceSet.createIterator() ;
  RooAbsArg* sliceArg ;
  while((sliceArg=(RooAbsArg*)iter->Next())) {
    RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
    if (arg) {
      projectedVars.remove(*arg) ;
    } else {
      coutI(Plotting) << "RooAbsReal::plotSliceOn(" << GetName() << ") slice variable " 
		      << sliceArg->GetName() << " was not projected anyway" << endl ;
    }
  }
  delete iter ;

  PlotOpt o ;
  o.drawOptions = drawOptions ;
  o.scaleFactor = scaleFactor ;
  o.stype = stype ;
  o.projData = projData ;
  o.projSet = &projectedVars ;
  return plotOn(frame,o) ;
}




//_____________________________________________________________________________
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
      RooArgSet* tmp = (RooArgSet*) o.projData->get()->selectCommon(*o.projDataSet) ;
      projDataVars.add(*tmp) ;
      delete tmp ;
    } else {
      projDataVars.add(*o.projData->get()) ;
    }
  }

  // Must depend on asymCat
  if (!dependsOn(asymCat)) {
    coutE(Plotting) << "RooAbsReal::plotAsymOn(" << GetName() 
		    << ") function doesn't depend on asymmetry category " << asymCat.GetName() << endl ;
    return frame ;
  }

  // asymCat must be a signCat
  if (!asymCat.isSignType()) {
    coutE(Plotting) << "RooAbsReal::plotAsymOn(" << GetName()
		    << ") asymmetry category must have 2 or 3 states with index values -1,0,1" << endl ;
    return frame ;
  }
  
  // Make list of variables to be projected
  RooArgSet projectedVars ;
  RooArgSet sliceSet ;
  if (o.projSet) {
    makeProjectionSet(frame->getPlotVar(),o.projSet,projectedVars,kFALSE) ;

    // Print list of non-projected variables
    if (frame->getNormVars()) {
      RooArgSet *sliceSetTmp = getObservables(*frame->getNormVars()) ;
      sliceSetTmp->remove(projectedVars,kTRUE,kTRUE) ;
      sliceSetTmp->remove(*frame->getPlotVar(),kTRUE,kTRUE) ;

      if (o.projData) {
	RooArgSet* tmp = (RooArgSet*) projDataVars.selectCommon(*o.projSet) ;
	sliceSetTmp->remove(*tmp,kTRUE,kTRUE) ;
	delete tmp ;
      }

      if (sliceSetTmp->getSize()) {
	coutI(Plotting) << "RooAbsReal::plotAsymOn(" << GetName() << ") plot on " 
			<< frame->getPlotVar()->GetName() << " represents a slice in " << *sliceSetTmp << endl ;
      }
      sliceSet.add(*sliceSetTmp) ;
      delete sliceSetTmp ;
    }
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  }


  // Take out data-projected dependens from projectedVars
  RooArgSet* projDataNeededVars = 0 ;
  if (o.projData) {
    projDataNeededVars = (RooArgSet*) projectedVars.selectCommon(projDataVars) ;
    projectedVars.remove(projDataVars,kTRUE,kTRUE) ;
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
		    << " projects variables " << projectedVars << endl ;
  }  
  if (projDataNeededVars && projDataNeededVars->getSize()>0) {
    coutI(Plotting) << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName() 
		    << " averages using data variables "<<  *projDataNeededVars << endl ;
  }


  // Customize two copies of projection with fixed negative and positive asymmetry
  RooAbsCategoryLValue* asymPos = (RooAbsCategoryLValue*) asymCat.Clone("asym_pos") ;
  RooAbsCategoryLValue* asymNeg = (RooAbsCategoryLValue*) asymCat.Clone("asym_neg") ;
  asymPos->setIndex(1) ;
  asymNeg->setIndex(-1) ;
  RooCustomizer* custPos = new RooCustomizer(*this,"pos") ;
  RooCustomizer* custNeg = new RooCustomizer(*this,"neg") ;
  custPos->setOwning(kTRUE) ;
  custNeg->setOwning(kTRUE) ;
  custPos->replaceArg(asymCat,*asymPos) ;
  custNeg->replaceArg(asymCat,*asymNeg) ;
  RooAbsReal* funcPos = (RooAbsReal*) custPos->build() ;
  RooAbsReal* funcNeg = (RooAbsReal*) custNeg->build() ;

  // Create projection integral 
  RooArgSet *posProjCompList, *negProjCompList ;

  // Add projDataVars to normalized dependents of projection
  // This is needed only for asymmetries (why?)
  RooArgSet depPos(*plotVar,*asymPos) ;
  RooArgSet depNeg(*plotVar,*asymNeg) ;
  depPos.add(projDataVars) ;
  depNeg.add(projDataVars) ;

  const RooAbsReal *posProj = funcPos->createPlotProjection(depPos, &projectedVars, posProjCompList) ;
  const RooAbsReal *negProj = funcNeg->createPlotProjection(depNeg, &projectedVars, negProjCompList) ;
  if (!posProj || !negProj) {
    coutE(Plotting) << "RooAbsReal::plotAsymOn(" << GetName() << ") Unable to create projections, abort" << endl ;
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
  RooFormulaVar* funcAsym = new RooFormulaVar(asymName,asymTitle,"(@0-@1)/(@0+@1)",RooArgSet(*posProj,*negProj)) ;

  if (o.projData) {
    
    // If data set contains more rows than needed, make reduced copy first
    RooAbsData* projDataSel = (RooAbsData*)o.projData;
    if (projDataNeededVars->getSize()<o.projData->get()->getSize()) {
      
      // Determine if there are any slice variables in the projection set
      RooArgSet* sliceDataSet = (RooArgSet*) sliceSet.selectCommon(*o.projData->get()) ;
      TString cutString ;
      if (sliceDataSet->getSize()>0) {
	TIterator* iter = sliceDataSet->createIterator() ;
	RooAbsArg* sliceVar ; 
	Bool_t first(kTRUE) ;
 	while((sliceVar=(RooAbsArg*)iter->Next())) {
	  if (!first) {
	    cutString.Append("&&") ;
 	  } else {
	    first=kFALSE ;	    
 	  }

 	  RooAbsRealLValue* real ;
	  RooAbsCategoryLValue* cat ;	  
 	  if ((real = dynamic_cast<RooAbsRealLValue*>(sliceVar))) {
	    cutString.Append(Form("%s==%f",real->GetName(),real->getVal())) ;
	  } else if ((cat = dynamic_cast<RooAbsCategoryLValue*>(sliceVar))) {
	    cutString.Append(Form("%s==%d",cat->GetName(),cat->getIndex())) ;	    
	  }
 	}
	delete iter ;
      }
      delete sliceDataSet ;

      if (!cutString.IsNull()) {
	projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars,cutString) ;
 	coutI(Plotting) << "RooAbsReal::plotAsymOn(" << GetName() 
			<< ") reducing given projection dataset to entries with " << cutString << endl ;
      } else {
	projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars) ;
      }
      coutI(Plotting) << "RooAbsReal::plotAsymOn(" << GetName() 
		      << ") only the following components of the projection data will be used: " << *projDataNeededVars << endl ;
    }    
    

    RooDataWeightedAverage dwa(Form("%sDataWgtAvg",GetName()),"Data Weighted average",*funcAsym,*projDataSel,o.numCPU,o.interleave,kTRUE) ;
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
    TString curveName(funcAsym->GetName()) ;
    curveName.Append(Form("_DataAvg[%s]",projDataSel->get()->contentsString().c_str())) ;
    // Append slice set specification if any
    if (sliceSet.getSize()>0) {
      curveName.Append(Form("_Slice[%s]",sliceSet.contentsString().c_str())) ;      
    }
    // Append any suffixes imported from RooAbsPdf::plotOn
    if (o.curveNameSuffix) {
      curveName.Append(o.curveNameSuffix) ;
    }


    RooAbsReal::enableEvalErrorLogging(kTRUE) ;
    RooCurve *curve = new RooCurve(funcAsym->GetName(),funcAsym->GetTitle(),scaleBind,
				   o.rangeLo,o.rangeHi,frame->GetNbinsX(),o.precision,o.precision,kFALSE,o.wmode,o.numee,o.doeeval,o.eeval) ;
    RooAbsReal::enableEvalErrorLogging(kFALSE) ;

    dynamic_cast<TAttLine*>(curve)->SetLineColor(2) ;
    // add this new curve to the specified plot frame
    frame->addPlotable(curve, o.drawOptions);

    ccoutW(Eval) << endl ;

    if (projDataSel!=o.projData) delete projDataSel ;
       
  } else {

    // Set default range, if not specified
    if (o.rangeLo==0 && o.rangeHi==0) {
      o.rangeLo = frame->GetXaxis()->GetXmin() ;
      o.rangeHi = frame->GetXaxis()->GetXmax() ;
    }

    RooAbsReal::enableEvalErrorLogging(kTRUE) ;
    RooCurve* curve= new RooCurve(*funcAsym,*plotVar,o.rangeLo,o.rangeHi,frame->GetNbinsX(),
				  o.scaleFactor,0,o.precision,o.precision,kFALSE,o.wmode,o.numee,o.doeeval,o.eeval);
    RooAbsReal::enableEvalErrorLogging(kFALSE) ;

    dynamic_cast<TAttLine*>(curve)->SetLineColor(2) ;


    // Set default name of curve
    TString curveName(funcAsym->GetName()) ;
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
  delete custPos ;
  delete custNeg ;
  delete posProjCompList ;
  delete negProjCompList ;
  delete asymPos ;
  delete asymNeg ;
  delete funcAsym ;

  delete plotVar ;

  return frame;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::plotSanityChecks(RooPlot* frame) const
{
  // Utility function for plotOn(), perform general sanity check on frame to ensure safe plotting operations

  // check that we are passed a valid plot frame to use
  if(0 == frame) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return kTRUE;
  }

  // check that this frame knows what variable to plot
  RooAbsReal* var = frame->getPlotVar() ;
  if(!var) {
    coutE(Plotting) << ClassName() << "::" << GetName()
	 << ":plotOn: frame does not specify a plot variable" << endl;
    return kTRUE;
  }

  // check that the plot variable is not derived
  if(!dynamic_cast<RooAbsRealLValue*>(var)) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOn: cannot plot variable \""
		    << var->GetName() << "\" of type " << var->ClassName() << endl;
    return kTRUE;
  }

  // check if we actually depend on the plot variable
  if(!this->dependsOn(*var)) {
    coutE(Plotting) << ClassName() << "::" << GetName() << ":plotOn: WARNING: variable is not an explicit dependent: "
		    << var->GetName() << endl;
  }
  
  return kFALSE ;
}




//_____________________________________________________________________________
void RooAbsReal::makeProjectionSet(const RooAbsArg* plotVar, const RooArgSet* allVars, 
				   RooArgSet& projectedVars, Bool_t silent) const
{
  // Utility function for plotOn() that constructs the set of
  // observables to project when plotting ourselves as function of
  // 'plotVar'. 'allVars' is the list of variables that must be
  // projected, but may contain variables that we do not depend on. If
  // 'silent' is cleared, warnings about inconsistent input parameters
  // will be printed.

  cxcoutD(Plotting) << "RooAbsReal::makeProjectionSet(" << GetName() << ") plotVar = " << plotVar->GetName() 
		    << " allVars = " << (allVars?(*allVars):RooArgSet()) << endl ;

  projectedVars.removeAll() ;
  if (!allVars) return ;

  // Start out with suggested list of variables  
  projectedVars.add(*allVars) ;

  // Take out plot variable
  RooAbsArg *found= projectedVars.find(plotVar->GetName());
  if(found) {
    projectedVars.remove(*found);

    // Take out eventual servers of plotVar
    RooArgSet* plotServers = plotVar->getObservables(&projectedVars) ;
    TIterator* psIter = plotServers->createIterator() ;
    RooAbsArg* ps ;
    while((ps=(RooAbsArg*)psIter->Next())) {
      RooAbsArg* tmp = projectedVars.find(ps->GetName()) ;
      if (tmp) {
	cxcoutD(Plotting) << "RooAbsReal::makeProjectionSet(" << GetName() << ") removing " << tmp->GetName() 
			  << " from projection set because it a server of " << plotVar->GetName() << endl ;
	projectedVars.remove(*tmp) ;
      }
    }
    delete psIter ;
    delete plotServers ;

    if (!silent) {
      coutW(Plotting) << "RooAbsReal::plotOn(" << GetName() 
		      << ") WARNING: cannot project out frame variable (" 
		      << found->GetName() << "), ignoring" << endl ; 
    }
  }

  // Take out all non-dependents of function
  TIterator* iter = allVars->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dependsOnValue(*arg)) {
      projectedVars.remove(*arg,kTRUE) ;

      cxcoutD(Plotting) << "RooAbsReal::plotOn(" << GetName() 
			<< ") function doesn't depend on projection variable " 
			<< arg->GetName() << ", ignoring" << endl ;
    }
  }
  delete iter ;
}




//_____________________________________________________________________________
Bool_t RooAbsReal::isSelectedComp() const 
{ 
  // If true, the current pdf is a selected component (for use in plotting)
  return _selectComp || _globalSelectComp ; 
}



//_____________________________________________________________________________
void RooAbsReal::globalSelectComp(Bool_t flag) 
{ 
  // Global switch controlling the activation of the selectComp() functionality
  _globalSelectComp = flag ; 
}




//_____________________________________________________________________________
RooAbsFunc *RooAbsReal::bindVars(const RooArgSet &vars, const RooArgSet* nset, Bool_t clipInvalid) const 
{
  // Create an interface adaptor f(vars) that binds us to the specified variables
  // (in arbitrary order). For example, calling bindVars({x1,x3}) on an object
  // F(x1,x2,x3,x4) returns an object f(x1,x3) that is evaluated using the
  // current values of x2 and x4. The caller takes ownership of the returned adaptor.

  RooAbsFunc *binding= new RooRealBinding(*this,vars,nset,clipInvalid);
  if(binding && !binding->isValid()) {
    coutE(InputArguments) << ClassName() << "::" << GetName() << ":bindVars: cannot bind to " << vars << endl ;
    delete binding;
    binding= 0;
  }
  return binding;
}

void RooAbsReal::copyCache(const RooAbsArg* source) 
{
  // Copy the cached value of another RooAbsArg to our cache.
  // Warning: This function copies the cached values of source,
  // it is the callers responsibility to make sure the cache is clean

  RooAbsReal* other = static_cast<RooAbsReal*>(const_cast<RooAbsArg*>(source)) ;

  if (!other->_treeVar) {
    _value = other->_value ;
  } else {
    if (source->getAttribute("FLOAT_TREE_BRANCH")) {
      _value = other->_floatValue ;
    } else if (source->getAttribute("INTEGER_TREE_BRANCH")) {
      _value = other->_intValue ;
    } else if (source->getAttribute("BYTE_TREE_BRANCH")) {
      _value = other->_byteValue ;
    } else if (source->getAttribute("UNSIGNED_INTEGER_TREE_BRANCH")) {
      _value = other->_uintValue ;
    } 
  }
  setValueDirty() ;
}



void RooAbsReal::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree. By default it will
  // register the internal value cache RooAbsReal::_value as branch
  // buffer for a Double_t tree branch with the same name as this
  // object. If no Double_t branch is found with the name of this
  // object, this method looks for a Float_t Int_t, UChar_t and UInt_t
  // branch in that order. If any of these are found the buffer for
  // that branch is set to a correctly typed conversion buffer in this
  // RooRealVar.  A flag is set that will cause copyCache to copy the
  // object value from the appropriate conversion buffer instead of
  // the _value buffer.

  // First determine if branch is taken
  TString cleanName(cleanBranchName()) ;
  TBranch* branch = t.GetBranch(cleanName) ;
  if (branch) { 
    
    // Determine if existing branch is Float_t or Double_t
    TLeaf* leaf = (TLeaf*)branch->GetListOfLeaves()->At(0) ;

    // Check that leaf is _not_ an array
    Int_t dummy ;
    TLeaf* counterLeaf = leaf->GetLeafCounter(dummy) ;
    if (counterLeaf) {
      coutE(Eval) << "RooAbsReal::attachToTree(" << GetName() << ") ERROR: TTree branch " << GetName() 
		  << " is an array and cannot be attached to a RooAbsReal" << endl ;      
      return ;
    }
    
    TString typeName(leaf->GetTypeName()) ;

    if (!typeName.CompareTo("Float_t")) {
      coutI(Eval) << "RooAbsReal::attachToTree(" << GetName() << ") TTree Float_t branch " << GetName() 
		  << " will be converted to double precision" << endl ;
      setAttribute("FLOAT_TREE_BRANCH",kTRUE) ;
      _treeVar = kTRUE ;
      t.SetBranchAddress(cleanName,&_floatValue) ;
    } else if (!typeName.CompareTo("Int_t")) {
      coutI(Eval) << "RooAbsReal::attachToTree(" << GetName() << ") TTree Int_t branch " << GetName() 
		  << " will be converted to double precision" << endl ;
      setAttribute("INTEGER_TREE_BRANCH",kTRUE) ;
      _treeVar = kTRUE ;
      t.SetBranchAddress(cleanName,&_intValue) ;
    } else if (!typeName.CompareTo("UChar_t")) {
      coutI(Eval) << "RooAbsReal::attachToTree(" << GetName() << ") TTree UChar_t branch " << GetName() 
		  << " will be converted to double precision" << endl ;
      setAttribute("BYTE_TREE_BRANCH",kTRUE) ;
      _treeVar = kTRUE ;
      t.SetBranchAddress(cleanName,&_byteValue) ;
    }  else if (!typeName.CompareTo("UInt_t")) { 
      coutI(Eval) << "RooAbsReal::attachToTree(" << GetName() << ") TTree UInt_t branch " << GetName() 
		  << " will be converted to double precision" << endl ;
      setAttribute("UNSIGNED_INTEGER_TREE_BRANCH",kTRUE) ;
      _treeVar = kTRUE ;
      t.SetBranchAddress(cleanName,&_uintValue) ;
    } else {
      t.SetBranchAddress(cleanName,&_value) ;
    }   
    
    if (branch->GetCompressionLevel()<0) {
      // cout << "RooAbsReal::attachToTree(" << GetName() << ") Fixing compression level of branch " << cleanName << endl ;
      branch->SetCompressionLevel(1) ;
    }

//      cout << "RooAbsReal::attachToTree(" << cleanName << "): branch already exists in tree " << (void*)&t << ", changing address" << endl ;

  } else {

    TString format(cleanName);
    format.Append("/D");
    branch = t.Branch(cleanName, &_value, (const char*)format, bufSize);
    branch->SetCompressionLevel(1) ;
    //      cout << "RooAbsReal::attachToTree(" << cleanName << "): creating new branch in tree " << (void*)&t << endl ;
  }

}



//_____________________________________________________________________________
void RooAbsReal::fillTreeBranch(TTree& t) 
{
  // Fill the tree branch that associated with this object with its current value

  // First determine if branch is taken
  TBranch* branch = t.GetBranch(cleanBranchName()) ;
  if (!branch) { 
    coutE(Eval) << "RooAbsReal::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree: " << cleanBranchName() << endl ;
    assert(0) ;
  }
  branch->Fill() ;
  
}



//_____________________________________________________________________________
void RooAbsReal::setTreeBranchStatus(TTree& t, Bool_t active) 
{
  // (De)Activate associated tree branch

  TBranch* branch = t.GetBranch(cleanBranchName()) ;
  if (branch) { 
    t.SetBranchStatus(cleanBranchName(),active?1:0) ;
  }
}



//_____________________________________________________________________________
RooAbsArg *RooAbsReal::createFundamental(const char* newname) const 
{
  // Create a RooRealVar fundamental object with our properties. The new
  // object will be created without any fit limits.

  RooRealVar *fund= new RooRealVar(newname?newname:GetName(),GetTitle(),_value,getUnit());
  fund->removeRange();
  fund->setPlotLabel(getPlotLabel());
  fund->setAttribute("fundamentalCopy");
  return fund;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a) const
{
  // Utility function for use in getAnalyticalIntegral(). If the
  // content of proxy 'a' occurs in set 'allDeps' then the argument
  // held in 'a' is copied from allDeps to analDeps

  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a, const RooArgProxy& b) const
{
  // Utility function for use in getAnalyticalIntegral(). If the
  // contents of proxies a,b occur in set 'allDeps' then the arguments
  // held in a,b are copied from allDeps to analDeps

  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;  
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a, const RooArgProxy& b,
			      const RooArgProxy& c) const
{
  // Utility function for use in getAnalyticalIntegral(). If the
  // contents of proxies a,b,c occur in set 'allDeps' then the arguments
  // held in a,b,c are copied from allDeps to analDeps

  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;
  nameList.Add(new TObjString(c.absArg()->GetName())) ;
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a, const RooArgProxy& b,
			      const RooArgProxy& c, const RooArgProxy& d) const
{
  // Utility function for use in getAnalyticalIntegral(). If the
  // contents of proxies a,b,c,d occur in set 'allDeps' then the arguments
  // held in a,b,c,d are copied from allDeps to analDeps

  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;
  nameList.Add(new TObjString(c.absArg()->GetName())) ;
  nameList.Add(new TObjString(d.absArg()->GetName())) ;
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}


//_____________________________________________________________________________
Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			     const RooArgSet& refset) const 
{
  // Utility function for use in getAnalyticalIntegral(). If the
  // contents of 'refset' occur in set 'allDeps' then the arguments
  // held in 'refset' are copied from allDeps to analDeps.

  TList nameList ;
  TIterator* iter = refset.createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    nameList.Add(new TObjString(arg->GetName())) ;    
  }
  delete iter ;

  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



//_____________________________________________________________________________
Bool_t RooAbsReal::matchArgsByName(const RooArgSet &allArgs, RooArgSet &matchedArgs,
				  const TList &nameList) const 
{
  // Check if allArgs contains matching elements for each name in nameList. If it does,
  // add the corresponding args from allArgs to matchedArgs and return kTRUE. Otherwise
  // return kFALSE and do not change matchedArgs.
  
  RooArgSet matched("matched");
  TIterator *iterator= nameList.MakeIterator();
  TObjString *name = 0;
  Bool_t isMatched(kTRUE);
  while((isMatched && (name= (TObjString*)iterator->Next()))) {
    RooAbsArg *found= allArgs.find(name->String().Data());
    if(found) {
      matched.add(*found);
    }
    else {
      isMatched= kFALSE;
    }
  }

  // nameList may not contain multiple entries with the same name
  // that are both matched
  if (isMatched && (matched.getSize()!=nameList.GetSize())) {
    isMatched = kFALSE ;
  }

  delete iterator;
  if(isMatched) matchedArgs.add(matched);
  return isMatched;
}



//_____________________________________________________________________________
RooNumIntConfig* RooAbsReal::defaultIntegratorConfig() 
{
  // Returns the default numeric integration configuration for all RooAbsReals
  return &RooNumIntConfig::defaultConfig() ;
}


//_____________________________________________________________________________
RooNumIntConfig* RooAbsReal::specialIntegratorConfig() const 
{
  // Returns the specialized integrator configuration for _this_ RooAbsReal.
  // If this object has no specialized configuration, a null pointer is returned
  return _specIntegratorConfig ;
}



//_____________________________________________________________________________
const RooNumIntConfig* RooAbsReal::getIntegratorConfig() const 
{
  // Return the numeric integration configuration used for this object. If
  // a specialized configuration was associated with this object, that configuration
  // is returned, otherwise the default configuration for all RooAbsReals is returned

  const RooNumIntConfig* config = specialIntegratorConfig() ;
  if (config) return config ;
  return defaultIntegratorConfig() ;
}



//_____________________________________________________________________________
void RooAbsReal::setIntegratorConfig(const RooNumIntConfig& config) 
{
  // Set the given integrator configuration as default numeric integration
  // configuration for this object
  if (_specIntegratorConfig) {
    delete _specIntegratorConfig ;
  }
  _specIntegratorConfig = new RooNumIntConfig(config) ;  
}



//_____________________________________________________________________________
void RooAbsReal::setIntegratorConfig() 
{
  // Remove the specialized numeric integration configuration associated
  // with this object
  if (_specIntegratorConfig) {
    delete _specIntegratorConfig ;
  }
  _specIntegratorConfig = 0 ;
}




//_____________________________________________________________________________
void RooAbsReal::selectNormalization(const RooArgSet*, Bool_t) 
{
  // Interface function to force use of a given set of observables
  // to interpret function value. Needed for functions or p.d.f.s
  // whose shape depends on the choice of normalization such as
  // RooAddPdf
}
 



//_____________________________________________________________________________
void RooAbsReal::selectNormalizationRange(const char*, Bool_t) 
{
  // Interface function to force use of a given normalization range
  // to interpret function value. Needed for functions or p.d.f.s
  // whose shape depends on the choice of normalization such as
  // RooAddPdf
}



//_____________________________________________________________________________
void RooAbsReal::setCacheCheck(Bool_t flag) 
{ 
  // Activate cache validation mode
  _cacheCheck = flag ; 
}



//_____________________________________________________________________________
Int_t RooAbsReal::getMaxVal(const RooArgSet& /*vars*/) const 
{
  // Advertise capability to determine maximum value of function for given set of 
  // observables. If no direct generator method is provided, this information
  // will assist the accept/reject generator to operate more efficiently as
  // it can skip the initial trial sampling phase to empirically find the function
  // maximum

  return 0 ;
}



//_____________________________________________________________________________
Double_t RooAbsReal::maxVal(Int_t /*code*/) 
{
  // Return maximum value for set of observables identified by code assigned
  // in getMaxVal

  assert(1) ;
  return 0 ;
}



//_____________________________________________________________________________
void RooAbsReal::logEvalError(const char* message, const char* serverValueString) const
{
  // Log evaluation error message. Evaluation errors may be routed through a different
  // protocol than generic RooFit warning message (which go straight through RooMsgService)
  // because evaluation errors can occur in very large numbers in the use of likelihood
  // evaluations. In logEvalError mode, controlled by global method enableEvalErrorLogging()
  // messages reported through this function are not printed but all stored in a list,
  // along with server values at the time of reporting. Error messages logged in this
  // way can be printed in a structured way, eliminating duplicates and with the ability
  // to truncate the list by printEvalErrors. This is the standard mode of error logging
  // during MINUIT operations. If enableEvalErrorLogging() is false, all errors
  // reported through this method are passed for immediate printing through RooMsgService.
  // A string with server names and values is constructed automatically for error logging
  // purposes, unless a custom string with similar information is passed as argument.

  EvalError ee ;
  ee.setMessage(message) ;

  if (serverValueString) {
    ee.setServerValues(serverValueString) ;
  } else {
    string srvval ;
    ostringstream oss ;
    Bool_t first(kTRUE) ;
    for (Int_t i=0 ; i<numProxies() ; i++) {
      RooAbsProxy* p = getProxy(i) ;
      if (p->name()[0]=='!') continue ;
      if (first) {
	first=kFALSE ;
      } else {
	oss << ", " ;
      }
      getProxy(i)->print(oss,kTRUE) ;
    }
    ee.setServerValues(oss.str().c_str()) ;
  }

  ostringstream oss2 ;
  printStream(oss2,kName|kClassName|kArgs,kInline)  ;

  if (!_doLogEvalError) {
   coutE(Eval) << "RooAbsReal::logEvalError(" << GetName() << ") evaluation error, " << endl 
	       << " origin       : " << oss2.str() << endl 
	       << " message      : " << ee._msg << endl
	       << " server values: " << ee._srvval << endl ;
  } else {
    _evalErrorList[this].first = oss2.str().c_str() ;
    _evalErrorList[this].second.push_back(ee) ;
  }
    
  //coutE(Tracing) << "RooAbsReal::logEvalError(" << GetName() << ") message = " << message << endl ;
}



//_____________________________________________________________________________
void RooAbsReal::clearEvalErrorLog() 
{
  // Clear the stack of evaluation error messages
  if (!_doLogEvalError) return ;
  _evalErrorList.clear() ;
}



//_____________________________________________________________________________
void RooAbsReal::printEvalErrors(ostream& os, Int_t maxPerNode) 
{
  // Print all outstanding logged evaluation error on the given ostream. If maxPerNode
  // is zero, only the number of errors for each source (object with unique name) is listed.
  // If maxPerNode is greater than zero, up to maxPerNode detailed error messages are shown
  // per source of errors. A truncation message is shown if there were more errors logged
  // than shown.

  if (maxPerNode<0) return ;

  map<const RooAbsArg*,pair<string,list<EvalError> > >::iterator iter = _evalErrorList.begin() ;

  for(;iter!=_evalErrorList.end() ; ++iter) {
    if (maxPerNode==0) {

      // Only print node name with total number of errors
      os << iter->second.first ;
      //iter->first->printStream(os,kName|kClassName|kArgs,kInline)  ;
      os << " has " << iter->second.second.size() << " errors" << endl ;      

    } else {

      // Print node name and details of 'maxPerNode' errors
      os << iter->second.first << endl ;
      //iter->first->printStream(os,kName|kClassName|kArgs,kSingleLine) ;

      Int_t i(0) ;
      std::list<EvalError>::iterator iter2 = iter->second.second.begin() ;
      for(;iter2!=iter->second.second.end() ; ++iter2, i++) {
	os << "     " << iter2->_msg << " @ " << iter2->_srvval << endl ;
	if (i>maxPerNode) {
	  os << "    ... (remaining " << iter->second.second.size() - maxPerNode << " messages suppressed)" << endl ;
	  break ;
	}
      } 
    }
  }
}



//_____________________________________________________________________________
Int_t RooAbsReal::numEvalErrors()
{
  // Return the number of logged evaluation errors since the last clearing.

  Int_t ntot(0) ;
  map<const RooAbsArg*,pair<string,list<EvalError> > >::iterator iter = _evalErrorList.begin() ;
  for(;iter!=_evalErrorList.end() ; ++iter) {
    ntot += iter->second.second.size() ;
  }
  return ntot ;
}



//_____________________________________________________________________________
void RooAbsReal::fixAddCoefNormalization(const RooArgSet& addNormSet, Bool_t force) 
{
  // Fix the interpretation of the coefficient of any RooAddPdf component in
  // the expression tree headed by this object to the given set of observables.
  //
  // If the force flag is false, the normalization choice is only fixed for those
  // RooAddPdf components that have the default 'automatic' interpretation of
  // coefficients (i.e. the interpretation is defined by the observables passed
  // to getVal()). If force is true, also RooAddPdf that already have a fixed
  // interpretation are changed to a new fixed interpretation.

  RooArgSet* compSet = getComponents() ;
  TIterator* iter = compSet->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(arg) ;
    if (pdf) {
      if (addNormSet.getSize()>0) {
	pdf->selectNormalization(&addNormSet,force) ;
      } else {
	pdf->selectNormalization(0,force) ;
      }
    } 
  }
  delete iter ;
  delete compSet ;  
}



//_____________________________________________________________________________
void RooAbsReal::fixAddCoefRange(const char* rangeName, Bool_t force) 
{
  // Fix the interpretation of the coefficient of any RooAddPdf component in
  // the expression tree headed by this object to the given set of observables.
  //
  // If the force flag is false, the normalization range choice is only fixed for those
  // RooAddPdf components that currently use the default full domain to interpret their
  // coefficients. If force is true, also RooAddPdf that already have a fixed
  // interpretation range are changed to a new fixed interpretation range.

  RooArgSet* compSet = getComponents() ;
  TIterator* iter = compSet->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(arg) ;
    if (pdf) {
      pdf->selectNormalizationRange(rangeName,force) ;
    }
  }
  delete iter ;
  delete compSet ;    
}



//_____________________________________________________________________________
void RooAbsReal::preferredObservableScanOrder(const RooArgSet& obs, RooArgSet& orderedObs) const
{
  // Interface method for function objects to indicate their prefferred order of observables
  // for scanning their values into a (multi-dimensional) histogram or RooDataSet. The observables
  // to be ordered are offered in argument 'obs' and should be copied in their preferred
  // order into argument 'orderdObs', This default implementation indicates no preference
  // and copies the original order of 'obs' into 'orderedObs'

  // Dummy implementation, do nothing 
  orderedObs.removeAll() ;
  orderedObs.add(obs) ;
}



//_____________________________________________________________________________
RooAbsReal* RooAbsReal::createRunningIntegral(const RooArgSet& iset, const RooArgSet& nset) 
{
  // Create a running integral over this function, i.e. given a f(x), create an object
  // representing 'int[x_lo,x] f(x_prime) dx_prime'

  return createRunningIntegral(iset,RooFit::SupNormSet(nset)) ;
}



//_____________________________________________________________________________
RooAbsReal* RooAbsReal::createRunningIntegral(const RooArgSet& iset, const RooCmdArg arg1, const RooCmdArg arg2,
				 const RooCmdArg arg3, const RooCmdArg arg4, const RooCmdArg arg5, 
				 const RooCmdArg arg6, const RooCmdArg arg7, const RooCmdArg arg8) 
{
  // Create an object that represents the running integral of the function over one or more observables listed in iset, i.e.
  // 
  //   int[x_lo,x] f(x_prime) dx_prime
  // 
  // The actual integration calculation is only performed when the return object is evaluated. The name
  // of the integral object is automatically constructed from the name of the input function, the variables
  // it integrates and the range integrates over. The default strategy to calculate the running integrals is
  //
  //   - If the integrand (this object) supports analytical integration, construct an integral object
  //     that calculate the running integrals value by calculating the analytical integral each
  //     time the running integral object is evaluated
  //
  //   - If the integrand (this object) requires numeric integration to construct the running integral
  //     create an object of class RooNumRunningInt which first samples the entire function and integrates
  //     the sampled function numerically. This method has superior performance as there is no need to
  //     perform a full (numeric) integration for each evaluation of the running integral object, but
  //     only when one of its parameters has changed.
  //
  // The choice of strategy can be changed with the ScanAll() argument, which forces the use of the
  // scanning technique implemented in RooNumRunningInt for all use cases, and with the ScanNone()
  // argument which forces the 'integrate each evaluation' technique for all use cases. The sampling
  // granularity for the scanning technique can be controlled with the ScanParameters technique
  // which allows to specify the number of samples to be taken, and to which order the resulting
  // running integral should be interpolated. The default values are 1000 samples and 2nd order
  // interpolation.
  //
  // The following named arguments are accepted
  //
  // SupNormSet(const RooArgSet&)         -- Observables over which should be normalized _in_addition_ to the
  //                                         integration observables
  // ScanParameters(Int_t nbins,          -- Parameters for scanning technique of making CDF: number
  //                Int_t intOrder)          of sampled bins and order of interpolation applied on numeric cdf
  // ScanNum()                            -- Apply scanning technique if cdf integral involves numeric integration
  // ScanAll()                            -- Always apply scanning technique 
  // ScanNone()                           -- Never apply scanning technique                  

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::createRunningIntegral(%s)",GetName())) ;
  pc.defineObject("supNormSet","SupNormSet",0,0) ;
  pc.defineInt("numScanBins","ScanParameters",0,1000) ;
  pc.defineInt("intOrder","ScanParameters",1,2) ;
  pc.defineInt("doScanNum","ScanNum",0,1) ;
  pc.defineInt("doScanAll","ScanAll",0,0) ;
  pc.defineInt("doScanNon","ScanNone",0,0) ;
  pc.defineMutex("ScanNum","ScanAll","ScanNone") ;

  // Process & check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Extract values from named arguments
  const RooArgSet* snset = static_cast<const RooArgSet*>(pc.getObject("supNormSet",0)) ;
  RooArgSet nset ;
  if (snset) {
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
    RooRealIntegral* tmp = (RooRealIntegral*) createIntegral(iset) ;
    Int_t isNum= (tmp->numIntRealVars().getSize()==1) ;
    delete tmp ;

    if (isNum) {
      coutI(NumIntegration) << "RooAbsPdf::createRunningIntegral(" << GetName() << ") integration over observable(s) " << iset << " involves numeric integration," << endl 
			    << "      constructing cdf though numeric integration of sampled pdf in " << numScanBins << " bins and applying order " 
			    << intOrder << " interpolation on integrated histogram." << endl 
			    << "      To override this choice of technique use argument ScanNone(), to change scan parameters use ScanParameters(nbins,order) argument" << endl ;
    }
    
    return isNum ? createScanRI(iset,nset,numScanBins,intOrder) : createIntRI(iset,nset) ;
  }
  return 0 ;
}



//_____________________________________________________________________________
RooAbsReal* RooAbsReal::createScanRI(const RooArgSet& iset, const RooArgSet& nset, Int_t numScanBins, Int_t intOrder) 
{
  // Utility function for createRunningIntegral that construct an object
  // implementing the numeric scanning technique for calculating the running integral
  
  string name = string(GetName()) + "_NUMRUNINT_" + integralNameSuffix(iset,&nset).Data() ;  
  RooRealVar* ivar = (RooRealVar*) iset.first() ;
  ivar->setBins(numScanBins,"numcdf") ;
  RooNumRunningInt* ret = new RooNumRunningInt(name.c_str(),name.c_str(),*this,*ivar,"numrunint") ;
  ret->setInterpolationOrder(intOrder) ;
  return ret ;
}



//_____________________________________________________________________________
RooAbsReal* RooAbsReal::createIntRI(const RooArgSet& iset, const RooArgSet& nset) 
{
  // Utility function for createRunningIntegral that construct an
  // object implementing the standard (analytical) integration
  // technique for calculating the running integral

  // Make list of input arguments keeping only RooRealVars
  RooArgList ilist ;
  TIterator* iter2 = iset.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter2->Next())) {
    if (dynamic_cast<RooRealVar*>(arg)) {
      ilist.add(*arg) ;
    } else {
      coutW(InputArguments) << "RooAbsPdf::createRunningIntegral(" << GetName() << ") WARNING ignoring non-RooRealVar input argument " << arg->GetName() << endl ;      
    }
  }
  delete iter2 ;

  RooArgList cloneList ;
  RooArgList loList ;
  RooArgSet clonedBranchNodes ;

  // Setup customizer that stores all cloned branches in our non-owning list
  RooCustomizer cust(*this,"cdf") ;
  cust.setCloneBranchSet(clonedBranchNodes) ;
  cust.setOwning(kFALSE) ;

  // Make integration observable x_prime for each observable x as well as an x_lowbound 
  TIterator* iter = ilist.createIterator() ;
  RooRealVar* rrv ;
  while((rrv=(RooRealVar*)iter->Next())) {

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
  delete iter ;

  RooAbsReal* tmp = (RooAbsReal*) cust.build() ;

  // Construct final normalization set for c.d.f = integrated observables + any extra specified by user
  RooArgSet finalNset(nset) ;
  finalNset.add(cloneList,kTRUE) ;
  RooAbsReal* cdf = tmp->createIntegral(cloneList,finalNset,"CDF") ;

  // Transfer ownership of cloned items to top-level c.d.f object
  cdf->addOwnedComponents(*tmp) ;
  cdf->addOwnedComponents(cloneList) ;
  cdf->addOwnedComponents(loList) ;

  return cdf ;
}

