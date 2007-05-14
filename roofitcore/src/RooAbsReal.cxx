/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooAbsReal.cxx,v 1.116 2007/05/11 09:11:58 verkerke Exp $
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

// -- CLASS DESCRIPTION [REAL] --
// RooAbsReal is the common abstract base class for objects that represent a
// real value. Implementation of RooAbsReal may be derived, there no interface
// is provided to modify the contents.
// 
// This class holds in addition a unit and label string, as well
// as a plot range and number of plot bins and plot creation methods.

#include "RooFit.h"

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
#include "RooDataProjBinding.h"
#include "RooAddPdf.h"
#include "RooCmdConfig.h"
#include "RooCategory.h"
#include "RooNumIntConfig.h"
#include "RooAddition.h"

#include "Riostream.h"

#include "TObjString.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TAttLine.h"
 
ClassImp(RooAbsReal)
;

Bool_t RooAbsReal::_cacheCheck(kFALSE) ;

RooAbsReal::RooAbsReal() : _specIntegratorConfig(0)
{
}

RooAbsReal::RooAbsReal(const char *name, const char *title, const char *unit) : 
  RooAbsArg(name,title), _plotMin(0), _plotMax(0), _plotBins(100), 
  _value(0),  _unit(unit), _forceNumInt(kFALSE), _specIntegratorConfig(0)
{
  // Constructor with unit label
  setValueDirty() ;
  setShapeDirty() ;
}

RooAbsReal::RooAbsReal(const char *name, const char *title, Double_t minVal,
		       Double_t maxVal, const char *unit) :
  RooAbsArg(name,title), _plotMin(minVal), _plotMax(maxVal), _plotBins(100),
  _value(0), _unit(unit), _forceNumInt(kFALSE), _specIntegratorConfig(0)
{
  // Constructor with plot range and unit label
  setValueDirty() ;
  setShapeDirty() ;
}


RooAbsReal::RooAbsReal(const RooAbsReal& other, const char* name) : 
  RooAbsArg(other,name), _plotMin(other._plotMin), _plotMax(other._plotMax), 
  _plotBins(other._plotBins), _value(other._value), _unit(other._unit), _forceNumInt(other._forceNumInt)
{

  // Copy constructor
  if (other._specIntegratorConfig) {
    _specIntegratorConfig = new RooNumIntConfig(*other._specIntegratorConfig) ;
  } else {
    _specIntegratorConfig = 0 ;
  }
}


RooAbsReal::~RooAbsReal()
{
  if (_specIntegratorConfig) delete _specIntegratorConfig ;
  // Destructor
}



Bool_t RooAbsReal::operator==(Double_t value) const
{
  // Equality operator comparing to a Double_t
  return (getVal()==value) ;
}


Bool_t RooAbsReal::operator==(const RooAbsArg& other) 
{
  const RooAbsReal* otherReal = dynamic_cast<const RooAbsReal*>(&other) ;
  return otherReal ? operator==(otherReal->getVal()) : kFALSE ;
}



TString RooAbsReal::getTitle(Bool_t appendUnit) const {
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

Double_t RooAbsReal::getVal(const RooArgSet* set) const
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval(set) ;
    clearValueDirty() ; 
    clearShapeDirty() ; 
  } else if (_cacheCheck) {
    
    // Check if cache contains value that evaluate() gives now
    Double_t checkValue = traceEval(set);

    if (checkValue != _value) {
      // If not, print warning
      cout << "RooAbsReal::getVal(" << GetName() << ") WARNING: cache contains " << _value 
	   << " but evaluate() returns " << checkValue << endl ;

      // And update cache (so that we see the difference)
      _value = checkValue ;
    }                                                                                                
    
  }
  
  return _value ;
}


Double_t RooAbsReal::traceEval(const RooArgSet* /*nset*/) const
{
  // Calculate current value of object, with error tracing wrapper
  Double_t value = evaluate() ;
  
  //Standard tracing code goes here
  if (!isValidReal(value)) {
    cout << "RooAbsReal::traceEval(" << GetName() 
	 << "): validation failed: " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}


Int_t RooAbsReal::getAnalyticalIntegralWN(RooArgSet& allDeps, RooArgSet& analDeps, 
					  const RooArgSet* /*normSet*/, const char* rangeName) const
{
  // Default implementation of getAnalyticalIntegralWN for real valued objects defers to
  // normalization invariant getAnalyticalIntegral()
  return _forceNumInt ? 0 : getAnalyticalIntegral(allDeps,analDeps,rangeName) ;
}


Int_t RooAbsReal::getAnalyticalIntegral(RooArgSet& /*allDeps*/, RooArgSet& /*analDeps*/, const char* /*rangeName*/) const
{
  // By default we do not supply any analytical integrals
  return 0 ;
}


Double_t RooAbsReal::analyticalIntegralWN(Int_t code, const RooArgSet* normSet, const char* rangeName) const
{
  // Default implementation of analyticalIntegralWN handles only the pass-through
  // scenario (code =0). All other codes are deferred to to the normalization
  // invariant analyticalIntegral() 

  if (code==0) return getVal(normSet) ;
  return analyticalIntegral(code,rangeName) ;
}


Double_t RooAbsReal::analyticalIntegral(Int_t code, const char* /*rangeName*/) const
{
  // By default no analytical integrals are implemented
  cout << "RooAbsReal::analyticalIntegral(" << GetName() << ") code " << code << " not implemented" << endl ;
  assert(0) ;
  return 0 ;
}


const char *RooAbsReal::getPlotLabel() const {
  // Get the label associated with the variable
  return _label.IsNull() ? fName.Data() : _label.Data();
}

void RooAbsReal::setPlotLabel(const char *label) {
  // Set the label associated with this variable
  _label= label;
}



Bool_t RooAbsReal::readFromStream(istream& /*is*/, Bool_t /*compact*/, Bool_t /*verbose*/) 
{
  //Read object contents from stream (dummy for now)
  return kFALSE ;
} 

void RooAbsReal::writeToStream(ostream& /*os*/, Bool_t /*compact*/) const
{
  //Write object contents to stream (dummy for now)
}

void RooAbsReal::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsArg::printToStream() we add:
  //
  //     Shape : value, units, plot range
  //   Verbose : default binning and print label

  RooAbsArg::printToStream(os,opt,indent);
  if(opt >= Shape) {
    os << indent << "--- RooAbsReal ---" << endl;
    TString unit(_unit);
    if(!unit.IsNull()) unit.Prepend(' ');
    os << indent << "  Value = " << getVal() << unit << endl;
//     os << indent << "  Plot range is [ " << getPlotMin() << unit << " , "
//        << getPlotMax() << unit << " ]" << endl;
    if(opt >= Verbose) {
//       os << indent << "  Plot bins = " << getPlotBins();
//       Double_t range= getPlotMax()-getPlotMin();
//       if(range > 0) os << " (" << range/getPlotBins() << unit << "/bin)";
      os << endl << indent << "  Plot label is \"" << getPlotLabel() << "\"" << endl;
    }
  }
}

void RooAbsReal::setPlotMin(Double_t value) {
  // Set minimum value of output associated with this object

  // Check if new limit is consistent
  if (_plotMin>_plotMax) {
    cout << "RooAbsReal::setPlotMin(" << GetName() 
	 << "): Proposed new integration min. larger than max., setting min. to max." << endl ;
    _plotMin = _plotMax ;
  } else {
    _plotMin = value ;
  }

}

void RooAbsReal::setPlotMax(Double_t value) {
  // Set maximum value of output associated with this object

  // Check if new limit is consistent
  if (_plotMax<_plotMin) {
    cout << "RooAbsReal::setPlotMax(" << GetName() 
	 << "): Proposed new integration max. smaller than min., setting max. to min." << endl ;
    _plotMax = _plotMin ;
  } else {
    _plotMax = value ;
  }

}


void RooAbsReal::setPlotRange(Double_t, Double_t) {
  // Set a new plot range
  cout << "RooAbsReal::setPlotBins(" << GetName() 
       << ") WARNING: setPlotRange deprecated. Specify plot range in RooAbsRealLValue::frame() when different from fitRange" << endl ;

//   // Check if new limit is consistent
//   if (min>max) {
//     cout << "RooAbsReal::setPlotMinMax(" << GetName() 
// 	 << "): Proposed new integration max. smaller than min., setting max. to min." << endl ;
//     _plotMin = min ;
//     _plotMax = min ;
//   } else {
//     _plotMin = min ;
//     _plotMax = max ;
//   }
}


void RooAbsReal::setPlotBins(Int_t /*value*/) {
  // Set number of histogram bins 
  cout << "RooAbsReal::setPlotBins(" << GetName() 
       << ") WARNING: setPlotBins deprecated. Specify plot bins in RooAbsRealLValue::frame() when different from fitBins" << endl ;
//   _plotBins = value ;  
}


Bool_t RooAbsReal::inPlotRange(Double_t value) const {
  // Check if given value is in the min-max range for this object
  return (value >= _plotMin && value <= _plotMax) ? kTRUE : kFALSE;
}



Bool_t RooAbsReal::isValid() const {
  // Check if current value is valid
  return isValidReal(_value) ;
}


Bool_t RooAbsReal::isValidReal(Double_t /*value*/, Bool_t /*printError*/) const 
{
  // Check if given value is valid
  return kTRUE ;
}


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


RooAbsReal* RooAbsReal::createIntegral(const RooArgSet& iset, const RooArgSet* nset, const RooNumIntConfig* cfg, const char* rangeName) const 
{
  TString title(GetTitle()) ;
  title.Prepend("Integral of ") ;

  if (!rangeName || strchr(rangeName,',')==0) {
    // Simple case: integral over full range or single limited range
    TString name(GetName()) ;
    name.Append(integralNameSuffix(iset,nset,rangeName)) ;
    return new RooRealIntegral(name,title,*this,iset,nset,cfg,rangeName) ;
  } 

  // Integral over multiple ranges
  char* buf = new char[strlen(rangeName)+1] ;
  strcpy(buf,rangeName) ;
  char* range = strtok(buf,",") ;
  RooArgSet components ;
  while (range) {
    TString compName(GetName()) ;
    compName.Append(integralNameSuffix(iset,nset,range)) ;
    RooAbsReal* compIntegral = new RooRealIntegral(compName,title,*this,iset,nset,cfg,range) ;
    components.add(*compIntegral) ;
    range = strtok(0,",") ;
  }
  delete[] buf ;
  TString fullName(GetName()) ;
  fullName.Append(integralNameSuffix(iset,nset,rangeName)) ;
  return new RooAddition(fullName,title,components,kTRUE) ;
}

TString RooAbsReal::integralNameSuffix(const RooArgSet& iset, const RooArgSet* nset, const char* rangeName) const 
{

  TString name ;
  name.Append("_Int[") ;
  if (iset.getSize()>0) {
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
  }

  if (rangeName) {
    name.Append("|") ;
    name.Append(rangeName) ;
  }
  
  name.Append("]_Norm[") ;
  if (nset && nset->getSize()>0) {
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
  }
  name.Append("]") ;

  return name ;
}



const RooAbsReal* RooAbsReal::createProjection(const RooArgSet& depVars, const RooArgSet& projVars, 
                                               RooArgSet*& cloneSet) const 
{
  return createProjection(depVars,&projVars,cloneSet) ; 
}



const RooAbsReal* RooAbsReal::createProjection(const RooArgSet& depVars, const RooArgSet& projVars) const 
{
  RooArgSet* cloneSet = new RooArgSet() ;
  return createProjection(depVars,&projVars,cloneSet) ; 
}



const RooAbsReal *RooAbsReal::createProjection(const RooArgSet &dependentVars, const RooArgSet *projectedVars,
					       RooArgSet *&cloneSet, const char* rangeName) const {
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
  // cloneSet (which includes the returned projection object) whatever
  // the return value. Note that you should normally call getVal()
  // on the returned object, without providing any set of normalization
  // variables. Otherwise you are requesting an additional normalization
  // beyond what is already specified in the equation above.

  // Get the set of our leaf nodes
  RooArgSet leafNodes;
  RooArgSet treeNodes;
  leafNodeServerList(&leafNodes,this);
  treeNodeServerList(&treeNodes,this) ;

//   cout << "RooAbsReal::createProjection(" << GetName() << ")" << endl 
//        << "   depVars " ; dependentVars.Print("1") ;
//   cout << "   projVars " ; if (projectedVars) projectedVars->Print("1") ; else cout << "<none>" << endl ;

//   cout << "initial leafNodes: " ; leafNodes.Print("1") ;

  // Check that the dependents are all fundamental. Filter out any that we
  // do not depend on, and make substitutions by name in our leaf list.
  // Check for overlaps with the projection variables.
  TIterator *dependentIterator= dependentVars.createIterator();
  assert(0 != dependentIterator);
  const RooAbsArg *arg = 0;
  while((arg= (const RooAbsArg*)dependentIterator->Next())) {
    if(!arg->isFundamental() && !dynamic_cast<const RooAbsLValue*>(arg)) {
      cout << ClassName() << "::" << GetName() << ":createProjection: variable \"" << arg->GetName()
	   << "\" of wrong type: " << arg->ClassName() << endl;
      delete dependentIterator;
      return 0;
    }
    //RooAbsArg *found= leafNodes.find(arg->GetName()); 
    RooAbsArg *found= treeNodes.find(arg->GetName()); 
    if(!found) {
      cout << ClassName() << "::" << GetName() << ":createProjection: \"" << arg->GetName()
	   << "\" is not a dependent and will be ignored." << endl;
      continue;
    }
    if(found != arg) {
      if (leafNodes.find(found->GetName())) {
	//cout << " replacing " << found << " with " << arg << " in leafNodes" << endl ; 
	leafNodes.replace(*found,*arg);
      } else {
	//cout << " adding " << found << " to leafNodes " << endl ;
	leafNodes.add(*arg) ;

	// Remove any dependents of found, replace by dependents of LV node
	RooArgSet* lvDep = arg->getObservables(&leafNodes) ;
	RooAbsArg* lvs ;
	TIterator* iter = lvDep->createIterator() ;
	while((lvs=(RooAbsArg*)iter->Next())) {
	  RooAbsArg* tmp = leafNodes.find(lvs->GetName()) ;
	  if (tmp) {
	    //cout << " replacing " << tmp << " with " << lvs << " in leaf nodes (2)" << endl ;
	    leafNodes.remove(*tmp) ;
	    leafNodes.add(*lvs) ;
	  }
	}
	delete iter ;
	
      }
    }

    // check if this arg is also in the projection set
    if(0 != projectedVars && projectedVars->find(arg->GetName())) {
      cout << ClassName() << "::" << GetName() << ":createProjection: \"" << arg->GetName()
	   << "\" cannot be both a dependent and a projected variable." << endl;
      delete dependentIterator;
      return 0;
    }
  }

  // Remove the projected variables from the list of leaf nodes, if necessary.
  if(0 != projectedVars) leafNodes.remove(*projectedVars,kTRUE);

//   cout << "createProjection: final leafNodes: " ; leafNodes.Print("1") ;
//   cout << "@@@@@@@@@ Before any manipulation" << endl ;
//   printCompactTree("","cp_orig_before.tree") ;

  // Make a deep-clone of ourself so later operations do not disturb our original state
  //cout << "RooAbsReal::createProjection(" << GetName() << ") making snapshot" << endl ;
  cloneSet= (RooArgSet*)RooArgSet(*this).snapshot(kTRUE);
  //cout << "RooAbsReal::createProjection(" << GetName() << ") finished snapshot" << endl ;

  if (!cloneSet) {
    cout << "RooAbsPdf::createProjection(" << GetName() << ") Couldn't deep-clone PDF, abort," << endl ;
    return 0 ;
  }
  RooAbsReal *clone= (RooAbsReal*)cloneSet->find(GetName());

  // The remaining entries in our list of leaf nodes are the the external
  // dependents (x) and parameters (p) of the projection. Patch them back
  // into the clone. This orphans the nodes they replace, but the orphans
  // are still in the cloneList and so will be cleaned up eventually.
  //cout << "redirection leafNodes : " ; leafNodes.Print("1") ;

//   cout << "@@@@@@@@@@ Original After deep-copy" << endl ;
//    printCompactTree("","cp_orig_afterDC.tree") ;
//   cout << "@@@@@@@@@@ Clone After deep-copy" << endl ;
//    clone->printCompactTree("","cp_clone.tree") ;


  //cout << "RooAbsReal::createProjection(" << GetName() << ") making rRS on leaf nodes" << endl ;
  clone->recursiveRedirectServers(leafNodes);
  //cout << "RooAbsReal::createProjection(" << GetName() << ") finished rRS" << endl ;

//   cout << "@@@@@@@@@@ Original after leaf node redirect" << endl ;
//   printCompactTree("","cp_orig_afterLNR.tree") ;
//   cout << "@@@@@@@@@@ Clone after leaf node redirect" << endl ;
//    clone->printCompactTree("","cp_clone_afterLNR.tree") ;


  // Create the set of normalization variables to use in the projection integrand

//   // WVE dependentVars might be LValue, expand into final servers
//   RooArgSet* xxx = dependentVars.getObservables(...) ;
//   RooArgSet normSet(*xxx) ;
//   delete xxx ;
    
  RooArgSet normSet(dependentVars);
  if(0 != projectedVars) normSet.add(*projectedVars);

  // Try to create a valid projection integral. If no variables are to be projected,
  // create a null projection anyway to bind our normalization over the dependents
  // consistently with the way they would be bound with a non-trivial projection.
  RooArgSet empty;
  if(0 == projectedVars) projectedVars= &empty;

  TString name(GetName()),title(GetTitle());
  name.Append("Projected");
  title.Prepend("Projection of ");

  RooRealIntegral *projected= new RooRealIntegral(name.Data(),title.Data(),*clone,*projectedVars,&normSet,0,rangeName);
  if(0 == projected || !projected->isValid()) {
    cout << ClassName() << "::" << GetName() << ":createProjection: cannot integrate out ";
    projectedVars->printToStream(cout,OneLine);
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




TH1 *RooAbsReal::fillHistogram(TH1 *hist, const RooArgList &plotVars,
			       Double_t scaleFactor, const RooArgSet *projectedVars) const {
  // Loop over the bins of the input histogram and add an amount equal to our value evaluated
  // at the bin center to each one. Our value is calculated by first integrating out any variables
  // in projectedVars and then scaling the result by scaleFactor. Returns a pointer to the
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


  // Check that the plot variables are all actually RooRealVars and print a warning if we do not
  // explicitly depend on one of them. Fill a set (not list!) of cloned plot variables.
  RooArgSet plotClones;
  for(Int_t index= 0; index < plotVars.getSize(); index++) {
    const RooAbsArg *var= plotVars.at(index);
    const RooRealVar *realVar= dynamic_cast<const RooRealVar*>(var);
    if(0 == realVar) {
      cout << ClassName() << "::" << GetName() << ":fillHistogram: cannot plot variable \"" << var->GetName()
	   << "\" of type " << var->ClassName() << endl;
      return 0;
    }
    if(!this->dependsOn(*realVar)) {
      cout << ClassName() << "::" << GetName()
	   << ":fillHistogram: WARNING: variable is not an explicit dependent: " << realVar->GetName() << endl;
    }
    plotClones.addClone(*realVar,kTRUE); // do not complain about duplicates
  }

  // Call checkObservables
  RooArgSet allDeps(plotClones) ;
  if (projectedVars) allDeps.add(*projectedVars) ;
  if (checkObservables(&allDeps)) {
    cout << "RooAbsReal::fillHistogram(" << GetName() << ") error in checkObservables, abort" << endl ;
    return hist ;
  }

  // Create a standalone projection object to use for calculating bin contents
  RooArgSet *cloneSet = 0;
  const RooAbsReal *projected= createProjection(plotClones,projectedVars,cloneSet);

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
    scaleFactor*= (zaxis->GetXmax() - zaxis->GetXmin())/zbins;
    // fall through to next case...
  case 2:
    ybins= hist->GetNbinsY(); 
    yvar= dynamic_cast<RooRealVar*>(plotClones.find(plotVars.at(1)->GetName()));
    yaxis= hist->GetYaxis();
    assert(0 != yvar && 0 != yaxis);
    scaleFactor*= (yaxis->GetXmax() - yaxis->GetXmin())/ybins;
    // fall through to next case...
  case 1:
    xbins= hist->GetNbinsX();
    xvar= dynamic_cast<RooRealVar*>(plotClones.find(plotVars.at(0)->GetName()));
    xaxis= hist->GetXaxis();
    assert(0 != xvar && 0 != xaxis);
    scaleFactor*= (xaxis->GetXmax() - xaxis->GetXmin())/xbins;
    break;
  default:
    cout << ClassName() << "::" << GetName() << ":fillHistogram: cannot fill histogram with "
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
      cout << "RooAbsReal::fillHistogram: Internal Error!" << endl;
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
  // Binning(double lo, double hi, int nbins)     -- Apply specified binning to x axis of histogram
  // ConditionalObservables(const RooArgSet& set) -- Do not normalized PDF over following observables when projecting PDF into histogram
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

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::createHistogram(%s)",GetName())) ;
  pc.defineObject("projObs","ProjectedObservables",0,0) ;
  pc.defineObject("yvar","YVar",0,0) ;
  pc.defineObject("zvar","ZVar",0,0) ;
  pc.allowUndefined() ;
  
  // Process & check varargs 
  pc.process(l) ;
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

  TH1* histo = xvar.createHistogram(name,l) ;
  fillHistogram(histo,vars,1.0,projObs) ;

  return histo ;
}




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
  //                                    only sensible in discrete observables
  // Project(const RooArgSet& set)   -- Override default projection behaviour by projecting over observables
  //                                    given in set and complete ignoring the default projection behavior. Advanced use only.
  // ProjWData(const RooAbsData& d)  -- Override default projection _technique_ (integration). For observables present in given dataset
  //                                    projection of PDF is achieved by constructing an average over all observable values in given set.
  //                                    Consult RooFit plotting tutorial for further explanation of meaning & use of this technique
  // ProjWData(const RooArgSet& s,   -- As above but only consider subset 's' of observables in dataset 'd' for projection through data averaging
  //           const RooAbsData& d)
  // ProjectionRange(const char* rn) -- Override default range of projection integrals to a different range speficied by given range name.
  //                                    This technique allows you to project a finite width slice in a real-valued observable
  // 
  // Misc content control
  // --------------------
  // Normalization(Double_t scale,   -- Adjust normalization by given scale factor. Interpretation of number depends on code: Relative:
  //                ScaleType code)     relative adjustment factor, NumEvent: scale to match given number of events.
  // Name(const chat* name)          -- Give curve specified name in frame. Useful if curve is to be referenced later
  // Asymmetry(const RooCategory& c) -- Show the asymmetry of the PDF in given two-state category [F(+)-F(-)] / [F(+)+F(-)] rather than
  //                                    the PDF projection. Category must have two states with indices -1 and +1 or three states with
  //                                    indeces -1,0 and +1.
  // ShiftToZero(Bool_t flag)        -- Shift entire curve such that lowest visible point is at exactly zero. Mostly useful when
  //                                    plotting -log(L) or chi^2 distributions
  // AddTo(const char* name,         -- Add constructed projection to already existing curve with given name and relative weight factors
  //       double_t wgtSelf, double_t wgtOther)
  //
  // Plotting control 
  // ----------------
  // DrawOption(const char* opt)     -- Select ROOT draw option for resulting TGraph object
  // LineStyle(Int_t style)          -- Select line style by ROOT line style code, default is solid
  // LineColor(Int_t color)          -- Select line color by ROOT color code, default is blue
  // LineWidth(Int_t width)          -- Select line with in pixels, default is 3
  // FillStyle(Int_t style)          -- Select fill style, default is not filled. If a filled style is selected, also use VLines()
  //                                    to add vertical downward lines at end of curve to ensure proper closure
  // FillColor(Int_t color)          -- Select fill color by ROOT color code
  // Range(const char* name)         -- Only draw curve in range defined by given name
  // Range(double lo, double hi)     -- Only draw curve in specified range
  // VLines()                        -- Add vertical lines to y=0 at end points of curve
  // Precision(Double_t eps)         -- Control precision of drawn curve w.r.t to scale of plot, default is 1e-3. Higher precision
  //                                    will result in more and more densely spaced curve points
  // Invisble(Bool_t flag)           -- Add curve to frame, but do not display. Useful in combination AddTo()


  RooLinkedList l ;
  l.Add((TObject*)&arg1) ;  l.Add((TObject*)&arg2) ;  
  l.Add((TObject*)&arg3) ;  l.Add((TObject*)&arg4) ;
  l.Add((TObject*)&arg5) ;  l.Add((TObject*)&arg6) ;  
  l.Add((TObject*)&arg7) ;  l.Add((TObject*)&arg8) ;
  l.Add((TObject*)&arg9) ;  l.Add((TObject*)&arg10) ;
  return plotOn(frame,l) ;
}

RooPlot* RooAbsReal::plotOn(RooPlot* frame, RooLinkedList& argList) const
{

  // New experimental plotOn() with varargs...

  // Define configuration for this method
  RooCmdConfig pc(Form("RooAbsReal::plotOn(%s)",GetName())) ;
  pc.defineString("drawOption","DrawOption",0,"L") ;
  pc.defineString("projectionRangeName","ProjectionRange",0,"",kTRUE) ;
  pc.defineDouble("scaleFactor","Normalization",0,1.0) ;
  pc.defineObject("sliceSet","SliceVars",0) ;
  pc.defineObject("projSet","Project",0) ;
  pc.defineObject("asymCat","Asymmetry",0) ;
  pc.defineDouble("precision","Precision",0,1e-3) ;
  pc.defineInt("shiftToZero","ShiftToZero",0,0) ;  
  pc.defineObject("projDataSet","ProjData",0) ;
  pc.defineObject("projData","ProjData",1) ;
  pc.defineDouble("rangeLo","Range",0,-999.) ;
  pc.defineDouble("rangeHi","Range",1,-999.) ;
  pc.defineInt("rangeAdjustNorm","Range",0,0) ;
  pc.defineInt("rangeWNAdjustNorm","RangeWithName",0,0) ;
  pc.defineInt("VLines","VLines",0,2) ; // 2==ExtendedWings
  pc.defineString("rangeName","RangeWithName",0,"") ;
  pc.defineInt("lineColor","LineColor",0,-999) ;
  pc.defineInt("lineStyle","LineStyle",0,-999) ;
  pc.defineInt("lineWidth","LineWidth",0,-999) ;
  pc.defineInt("fillColor","FillColor",0,-999) ;
  pc.defineInt("fillStyle","FillStyle",0,-999) ;
  pc.defineString("curveName","Name",0,"") ;
  pc.defineInt("curveInvisible","Invisible",0,0) ;
  pc.defineString("addToCurveName","AddTo",0,"") ;
  pc.defineDouble("addToWgtSelf","AddTo",0,1.) ;
  pc.defineDouble("addToWgtOther","AddTo",1,1.) ;
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
  o.drawOptions = pc.getString("drawOption") ;
  o.scaleFactor = pc.getDouble("scaleFactor") ;
  o.projData = (const RooAbsData*) pc.getObject("projData") ;
  o.projDataSet = (const RooArgSet*) pc.getObject("projDataSet") ;

  const RooArgSet* sliceSet = (const RooArgSet*) pc.getObject("sliceSet") ;
  const RooArgSet* projSet = (const RooArgSet*) pc.getObject("projSet") ;
  const RooAbsCategoryLValue* asymCat = (const RooAbsCategoryLValue*) pc.getObject("asymCat") ;

  o.precision = pc.getDouble("precision") ;
  o.shiftToZero = (pc.getInt("shiftToZero")!=0) ;
  Int_t vlines = pc.getInt("VLines");
  if (pc.hasProcessed("Range")) {
    o.rangeLo = pc.getDouble("rangeLo") ;
    o.rangeHi = pc.getDouble("rangeHi") ;
    o.postRangeFracScale = pc.getInt("rangeAdjustNorm") ;
    if (vlines==2) vlines=0 ; // Default is NoWings if range was specified
  } else if (pc.hasProcessed("RangeWithName")) {    
    o.rangeLo = frame->getPlotVar()->getMin(pc.getString("rangeName",0,kTRUE)) ;
    o.rangeHi = frame->getPlotVar()->getMax(pc.getString("rangeName",0,kTRUE)) ;
    o.postRangeFracScale = pc.getInt("rangeAdjustNorm") ;
    if (vlines==2) vlines=0 ; // Default is NoWings if range was specified
  } else {

    // Use range of last fit, if it was non-default and no other range was specified
    RooArgSet* plotDep = getObservables(*frame->getPlotVar()) ;
    RooRealVar* plotDepVar = (RooRealVar*) plotDep->find(frame->getPlotVar()->GetName()) ;
    if (plotDepVar->hasBinning("fit")) {
      o.rangeLo = plotDepVar->getMin("fit") ;
      o.rangeHi = plotDepVar->getMax("fit") ;
      o.postRangeFracScale = kTRUE ;
      if (vlines==2) vlines=0 ; // Default is NoWings if range was specified
    }
    delete plotDep ;
  }
  o.wmode = (vlines==2)?RooCurve::Extended:(vlines==1?RooCurve::Straight:RooCurve::NoWings) ;
  o.projectionRangeName = pc.getString("projectionRangeName",0,kTRUE) ;
  o.curveName = pc.getString("curveName",0,kTRUE) ;
  o.curveInvisible = pc.getInt("curveInvisible") ;
  o.addToCurveName = pc.getString("addToCurveName",0,kTRUE) ;
  o.addToWgtSelf = pc.getDouble("addToWgtSelf") ;
  o.addToWgtOther = pc.getDouble("addToWgtOther") ;  

  if (o.addToCurveName && !frame->findObject(o.addToCurveName,RooCurve::Class())) {
    cout << "RooAbsReal::plotOn(" << GetName() << ") cannot find existing curve " << o.addToCurveName << " to add to in RooPlot" << endl ;
    return frame ;
  }
  
  RooArgSet projectedVars ;
  if (sliceSet) {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
    
    // Take out the sliced variables
    TIterator* iter = sliceSet->createIterator() ;
    RooAbsArg* sliceArg ;
    while((sliceArg=(RooAbsArg*)iter->Next())) {
      RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
      if (arg) {
	projectedVars.remove(*arg) ;
      } else {
	cout << "RooAbsReal::plotOn(" << GetName() << ") slice variable " 
	     << sliceArg->GetName() << " was not projected anyway" << endl ;
      }
    }
    delete iter ;
  } else if (projSet) {
    makeProjectionSet(frame->getPlotVar(),projSet,projectedVars,kFALSE) ;
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  }
  o.projSet = &projectedVars ;

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
  Int_t fillColor = pc.getInt("fillColor") ;
  Int_t fillStyle = pc.getInt("fillStyle") ;
  if (lineColor!=-999) ret->getAttLine()->SetLineColor(lineColor) ;
  if (lineStyle!=-999) ret->getAttLine()->SetLineStyle(lineStyle) ;
  if (lineWidth!=-999) ret->getAttLine()->SetLineWidth(lineWidth) ;
  if (fillColor!=-999) ret->getAttFill()->SetFillColor(fillColor) ;
  if (fillStyle!=-999) ret->getAttFill()->SetFillStyle(fillStyle) ;
  
  return ret ;
}






RooPlot* RooAbsReal::plotOn(RooPlot *frame, PlotOpt o) const
{
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
    if (o.projDataSet) {
      RooArgSet* tmp = (RooArgSet*) o.projData->get()->selectCommon(*o.projDataSet) ;
      projDataVars.add(*tmp) ;
      delete tmp ;
    } else {
      projDataVars.add(*o.projData->get()) ;
    }
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
	cout << "RooAbsReal::plotOn(" << GetName() << ") plot on " 
	     << frame->getPlotVar()->GetName() << " represents a slice in " ;
	sliceSetTmp->Print("1") ;
      }
      sliceSet.add(*sliceSetTmp) ;
      delete sliceSetTmp ;
    }
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  }

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
    cout << "RooAbsReal::plotOn(" << GetName() << ") Couldn't deep-clone self, abort," << endl ;
    return frame ;
  }
  RooRealVar* plotVar = (RooRealVar*) plotCloneSet->find(realVar->GetName());

  // Inform user about projections
  if (projectedVars.getSize()) {
    cout << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName() 
	 << " integrates over variables " << projectedVars 
	 << (o.projectionRangeName?Form(" in range %s",o.projectionRangeName):"") << endl;
  }  
  if (projDataNeededVars && projDataNeededVars->getSize()>0) {
    cout << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName() 
	 << " averages using data variables " ; projDataNeededVars->Print("1") ;
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
    cout << "RooAbsReal::plotOn(" << GetName() << ") error in checkObservables, abort" << endl ;
    delete deps ;
    delete plotCloneSet ;
    if (projDataNeededVars) delete projDataNeededVars ;
    return frame ;
  }

  //cout << "RooAbsReal::plotOn(" << GetName() << ") begin createProjection" << endl ;
  RooAbsReal *projection = (RooAbsReal*) createProjection(*deps, &projectedVars, projectionCompList, o.projectionRangeName) ;
  //cout << "RooAbsReal::plotOn(" << GetName() << ") end createProjection" << endl ;

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

    // Disable dirty state propagation in projection    
    RooArgSet branchList("branchList") ;
    ((RooAbsReal*)projection)->setOperMode(RooAbsArg::ADirty) ;
    projection->branchNodeServerList(&branchList) ;
    TIterator* bIter = branchList.createIterator() ;
    RooAbsArg* branch ;
    while((branch=(RooAbsArg*)bIter->Next())) {
      branch->setOperMode(RooAbsArg::ADirty) ;
    }
    delete bIter ;

    // If data set contains more rows than needed, make reduced copy first
    RooAbsData* projDataSel = (RooAbsData*)o.projData;

//     cout << "projDataNeededVars = " ; projDataNeededVars->Print("1") ;
//     cout << "projData vars      = " ; projData->get()->Print("1") ;
//     cout << "sliceSet           = " ; sliceSet.Print("1") ;

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
	cout << "RooAbsReal::plotOn(" << GetName() << ") reducing given projection dataset to entries with " << cutString << endl ;
      } else {
	projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars) ;
      }
      cout << "RooAbsReal::plotOn(" << GetName() 
	   << ") only the following components of the projection data will be used: " ; 
      projDataNeededVars->Print("1") ;
    }



    // Attach dataset
    projection->getVal(projDataSel->get()) ;
    
    //cout << "RooAbsReal::plotOn(" << GetName() << ") start attachDataSet" << endl ;
    projection->attachDataSet(*projDataSel) ;
    //cout << "RooAbsReal::plotOn(" << GetName() << ") end attachDataSet" << endl ;

    RooDataProjBinding projBind(*projection,*projDataSel,*plotVar) ;

//     // WVE -- Experimental, deactictivate for now
//     projection->optimizeDirty(*projDataSel,deps) ;
//     projection->doConstOpt(*projDataSel,deps);

    RooScaledFunc scaleBind(projBind,o.scaleFactor);

    // Set default range, if not specified
    if (o.rangeLo==0 && o.rangeHi==0) {
      o.rangeLo = frame->GetXaxis()->GetXmin() ;
      o.rangeHi = frame->GetXaxis()->GetXmax() ;
    }

    RooCurve *curve = new RooCurve(projection->GetName(),projection->GetTitle(),scaleBind,
				   o.rangeLo,o.rangeHi,frame->GetNbinsX(),o.precision,o.precision,o.shiftToZero,o.wmode) ;
    cout << endl ;

    // Add self to other curve if requested
    if (o.addToCurveName) {
      RooCurve* otherCurve = static_cast<RooCurve*>(frame->findObject(o.addToCurveName,RooCurve::Class())) ;
      RooCurve* sumCurve = new RooCurve(projection->GetName(),projection->GetTitle(),*curve,*otherCurve,o.addToWgtSelf,o.addToWgtOther) ;
      delete curve ;
      curve = sumCurve ;
    }

    if (o.curveName) {
      curve->SetName(o.curveName) ;
    }

    // add this new curve to the specified plot frame
    frame->addPlotable(curve, o.drawOptions);

    if (projDataSel!=o.projData) delete projDataSel ;
    delete projDataNeededVars ;
       
  } else {
    
    // Set default range, if not specified
    if (o.rangeLo==0 && o.rangeHi==0) {
      o.rangeLo = frame->GetXaxis()->GetXmin() ;
      o.rangeHi = frame->GetXaxis()->GetXmax() ;
    }

    // Calculate a posteriori range fraction scaling if requested (2nd part of normalization correction for
    // result fit on subrange of data)
    if (o.postRangeFracScale) {
      plotVar->setRange("plotRange",o.rangeLo,o.rangeHi) ;
      RooAbsReal* intFrac = projection->createIntegral(*plotVar,*plotVar,"plotRange") ;
      o.scaleFactor /= intFrac->getVal() ;
      delete intFrac ;
    }

    // create a new curve of our function using the clone to do the evaluations
    RooCurve *curve = new RooCurve(*projection,*plotVar,o.rangeLo,o.rangeHi,frame->GetNbinsX(),
				   o.scaleFactor,0,o.precision,o.precision,o.shiftToZero,o.wmode);

    // Add self to other curve if requested
    if (o.addToCurveName) {
      RooCurve* otherCurve = static_cast<RooCurve*>(frame->findObject(o.addToCurveName,RooCurve::Class())) ;
      RooCurve* sumCurve = new RooCurve(projection->GetName(),projection->GetTitle(),*curve,*otherCurve,o.addToWgtSelf,o.addToWgtOther) ;
      delete curve ;
      curve = sumCurve ;
    }

    if (o.curveName) {
      curve->SetName(o.curveName) ;
    }

    // add this new curve to the specified plot frame
    frame->addPlotable(curve, o.drawOptions, o.curveInvisible);
  }

  delete deps ;
  delete projectionCompList ;
  delete plotCloneSet ;
  return frame;
}




RooPlot* RooAbsReal::plotSliceOn(RooPlot *frame, const RooArgSet& sliceSet, Option_t* drawOptions, 
				 Double_t scaleFactor, ScaleType stype, const RooAbsData* projData) const
{
  // Plot ourselves on given frame, as done in plotOn(), except that the variables 
  // listed in 'sliceSet' are taken out from the default list of projected dimensions created
  // by plotOn().

  // Plot
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
      cout << "RooAbsReal::plotSliceOn(" << GetName() << ") slice variable " 
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





RooPlot* RooAbsReal::plotAsymOn(RooPlot *frame, const RooAbsCategoryLValue& asymCat, PlotOpt o) const

{
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
    cout << "RooAbsReal::plotAsymOn(" << GetName() 
	 << ") function doesn't depend on asymmetry category " << asymCat.GetName() << endl ;
    return frame ;
  }

  // asymCat must be a signCat
  if (!asymCat.isSignType()) {
    cout << "RooAbsReal::plotAsymOn(" << GetName()
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
	cout << "RooAbsReal::plotAsymOn(" << GetName() << ") plot on " 
	     << frame->getPlotVar()->GetName() << " represents a slice in " ;
	sliceSetTmp->Print("1") ;
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
    cout << "RooAbsReal::plotAsymOn(" << GetName() << ") plot on " << plotVar->GetName() 
	 << " projects variables " ; projectedVars.Print("1") ;
  }  
  if (projDataNeededVars && projDataNeededVars->getSize()>0) {
    cout << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName() 
	 << " averages using data variables " ; projDataNeededVars->Print("1") ;
  }


  // Customize two copies of projection with fixed negative and positive asymmetry
  RooAbsCategoryLValue* asymPos = (RooAbsCategoryLValue*) asymCat.Clone("asym_pos") ;
  RooAbsCategoryLValue* asymNeg = (RooAbsCategoryLValue*) asymCat.Clone("asym_neg") ;
  asymPos->setIndex(1) ;
  asymNeg->setIndex(-1) ;
  RooCustomizer custPos(*this,"pos") ;
  RooCustomizer custNeg(*this,"neg") ;
  custPos.replaceArg(asymCat,*asymPos) ;
  custNeg.replaceArg(asymCat,*asymNeg) ;
  RooAbsReal* funcPos = (RooAbsReal*) custPos.build() ;
  RooAbsReal* funcNeg = (RooAbsReal*) custNeg.build() ;

  // Create projection integral 
  RooArgSet *posProjCompList, *negProjCompList ;

  // Add projDataVars to normalized dependents of projection
  // This is needed only for asymmetries (why?)
  RooArgSet depPos(*plotVar,*asymPos) ;
  RooArgSet depNeg(*plotVar,*asymNeg) ;
  depPos.add(projDataVars) ;
  depNeg.add(projDataVars) ;

  const RooAbsReal *posProj = funcPos->createProjection(depPos, &projectedVars, posProjCompList) ;
  const RooAbsReal *negProj = funcNeg->createProjection(depNeg, &projectedVars, negProjCompList) ;
  if (!posProj || !negProj) {
    cout << "RooAbsReal::plotAsymOn(" << GetName() << ") Unable to create projections, abort" << endl ;
    return frame ; 
  }

  // Create a RooFormulaVar representing the asymmetry
  TString asymName(GetName()) ;
  asymName.Append("Asymmetry[") ;
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
 	cout << "RooAbsReal::plotAsymOn(" << GetName() 
	     << ") reducing given projection dataset to entries with " << cutString << endl ;
      } else {
	projDataSel = ((RooAbsData*)o.projData)->reduce(*projDataNeededVars) ;
      }
      cout << "RooAbsReal::plotAsymOn(" << GetName() 
	   << ") only the following components of the projection data will be used: " ; 
      projDataNeededVars->Print("1") ;
    }    

    RooDataProjBinding projBind(*funcAsym,*projDataSel,*plotVar) ;
    ((RooAbsReal*)posProj)->attachDataSet(*projDataSel) ;
    ((RooAbsReal*)negProj)->attachDataSet(*projDataSel) ;

    RooScaledFunc scaleBind(projBind,o.scaleFactor);

    // Set default range, if not specified
    if (o.rangeLo==0 && o.rangeHi==0) {
      o.rangeLo = frame->GetXaxis()->GetXmin() ;
      o.rangeHi = frame->GetXaxis()->GetXmax() ;
    }

    RooCurve *curve = new RooCurve(funcAsym->GetName(),funcAsym->GetTitle(),scaleBind,
				   o.rangeLo,o.rangeHi,frame->GetNbinsX(),o.precision,o.precision,kFALSE,o.wmode) ;
    dynamic_cast<TAttLine*>(curve)->SetLineColor(2) ;
    // add this new curve to the specified plot frame
    frame->addPlotable(curve, o.drawOptions);

    cout << endl ;

    if (projDataSel!=o.projData) delete projDataSel ;
       
  } else {

    // Set default range, if not specified
    if (o.rangeLo==0 && o.rangeHi==0) {
      o.rangeLo = frame->GetXaxis()->GetXmin() ;
      o.rangeHi = frame->GetXaxis()->GetXmax() ;
    }

    RooCurve* curve= new RooCurve(*funcAsym,*plotVar,o.rangeLo,o.rangeHi,frame->GetNbinsX(),
				  o.scaleFactor,0,o.precision,o.precision,kFALSE,o.wmode);
    dynamic_cast<TAttLine*>(curve)->SetLineColor(2) ;
    
    // add this new curve to the specified plot frame
    frame->addPlotable(curve, o.drawOptions);

  }

  // Cleanup
  delete posProjCompList ;
  delete negProjCompList ;
  delete asymPos ;
  delete asymNeg ;
  delete funcAsym ;

  return frame;
}



Bool_t RooAbsReal::plotSanityChecks(RooPlot* frame) const
{
  // Perform general sanity check on frame to ensure safe plotting operations

  // check that we are passed a valid plot frame to use
  if(0 == frame) {
    cout << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return kTRUE;
  }

  // check that this frame knows what variable to plot
  RooAbsReal* var = frame->getPlotVar() ;
  if(!var) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: frame does not specify a plot variable" << endl;
    return kTRUE;
  }

  // check that the plot variable is not derived
  if(!dynamic_cast<RooAbsRealLValue*>(var)) {
     cout << ClassName() << "::" << GetName() << ":plotOn: cannot plot variable \""
	  << var->GetName() << "\" of type " << var->ClassName() << endl;
    return kTRUE;
  }

  // check if we actually depend on the plot variable
  if(!this->dependsOn(*var)) {
    cout << ClassName() << "::" << GetName() << ":plotOn: WARNING: variable is not an explicit dependent: "
	 << var->GetName() << endl;
  }
  
  return kFALSE ;
}




void RooAbsReal::makeProjectionSet(const RooAbsArg* plotVar, const RooArgSet* allVars, 
				   RooArgSet& projectedVars, Bool_t silent) const
{
  // Construct the set of dependents to project when plotting ourselves as function
  // of 'plotVar'. 'allVars' is the list of variables that must be projected, but
  // may contain variables that we do not depend on. If 'silent' is cleared,
  // warnings about inconsistent input parameters will be printed.

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
      if (tmp) projectedVars.remove(*tmp) ;
    }
    delete psIter ;
    delete plotServers ;

    if (!silent) {
      cout << "RooAbsReal::plotOn(" << GetName() 
	   << ") WARNING: cannot project out frame variable (" 
	   << found->GetName() << "), ignoring" << endl ; 
    }
  }

  // Take out all non-dependents of function
  TIterator* iter = allVars->createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!dependsOn(*arg)) {
      projectedVars.remove(*arg,kTRUE) ;
      if (!silent) {
	cout << "RooAbsReal::plotOn(" << GetName() 
	     << ") WARNING: function doesn't depend on projection variable " 
	     << arg->GetName() << ", ignoring" << endl ;
      }
    }
  }
  delete iter ;
}





RooAbsFunc *RooAbsReal::bindVars(const RooArgSet &vars, const RooArgSet* nset, Bool_t clipInvalid) const {
  // Create an interface adaptor f(vars) that binds us to the specified variables
  // (in arbitrary order). For example, calling bindVars({x1,x3}) on an object
  // F(x1,x2,x3,x4) returns an object f(x1,x3) that is evaluated using the
  // current values of x2 and x4. The caller takes ownership of the returned adaptor.

  RooAbsFunc *binding= new RooRealBinding(*this,vars,nset,clipInvalid);
  if(binding && !binding->isValid()) {
    cout << ClassName() << "::" << GetName() << ":bindVars: cannot bind to ";
    vars.Print();
    delete binding;
    binding= 0;
  }
  return binding;
}

void RooAbsReal::copyCache(const RooAbsArg* source) 
{
  // Copy the cached value of another RooAbsArg to our cache

  // Warning: This function copies the cached values of source,
  //          it is the callers responsibility to make sure the cache is clean
  RooAbsReal* other = dynamic_cast<RooAbsReal*>(const_cast<RooAbsArg*>(source)) ;
  assert(other!=0) ;

  if (source->getAttribute("FLOAT_TREE_BRANCH")) {
    _value = *reinterpret_cast<Float_t*>(&other->_value) ;
  } else if (source->getAttribute("INTEGER_TREE_BRANCH")) {
    _value = *reinterpret_cast<Int_t*>(&other->_value) ;
  } else if (source->getAttribute("BYTE_TREE_BRANCH")) {
    _value = *reinterpret_cast<UChar_t*>(&other->_value) ;
  } else if (source->getAttribute("UNSIGNED_INTEGER_TREE_BRANCH")) {
    _value = *reinterpret_cast<UInt_t*>(&other->_value) ;
  } else {
    _value = other->_value ;
  }
  setValueDirty() ;
}


void RooAbsReal::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  TString cleanName(cleanBranchName()) ;
  TBranch* branch = t.GetBranch(cleanName) ;
  if (branch) { 
    
    // Determine if existing branch is Float_t or Double_t
    TString typeName(((TLeaf*)branch->GetListOfLeaves()->At(0))->GetTypeName()) ;
    if (!typeName.CompareTo("Float_t")) {
      cout << "RooAbsReal::attachToTree(" << GetName() << ") TTree Float_t branch " << GetName() 
	   << " will be converted to double precision" << endl ;
      setAttribute("FLOAT_TREE_BRANCH",kTRUE) ;
    } else if (!typeName.CompareTo("Int_t")) {
      cout << "RooAbsReal::attachToTree(" << GetName() << ") TTree Int_t branch " << GetName() 
	   << " will be converted to double precision" << endl ;
      setAttribute("INTEGER_TREE_BRANCH",kTRUE) ;
    } else if (!typeName.CompareTo("UChar_t")) {
      cout << "RooAbsReal::attachToTree(" << GetName() << ") TTree UChar_t branch " << GetName() 
	   << " will be converted to double precision" << endl ;
      setAttribute("BYTE_TREE_BRANCH",kTRUE) ;
    }  else if (!typeName.CompareTo("UInt_t")) { 
      cout << "RooAbsReal::attachToTree(" << GetName() << ") TTree UInt_t branch " << GetName() 
	   << " will be converted to double precision" << endl ;
      setAttribute("UNSIGNED_INTEGER_TREE_BRANCH",kTRUE) ;
    }    
    
    t.SetBranchAddress(cleanName,&_value) ;
    if (branch->GetCompressionLevel()<0) {
      cout << "RooAbsReal::attachToTree(" << GetName() << ") Fixing compression level of branch " << cleanName << endl ;
      branch->SetCompressionLevel(1) ;
    }

//     cout << "RooAbsReal::attachToTree(" << cleanName << "): branch already exists in tree " 
// 	 << (void*)&t << ", changing address" << endl ;

  } else {
    TString format(cleanName);
    format.Append("/D");
    branch = t.Branch(cleanName, &_value, (const Text_t*)format, bufSize);
    branch->SetCompressionLevel(1) ;
//     cout << "RooAbsReal::attachToTree(" << cleanName << "): creating new branch in tree" 
// 	 << (void*)&t << endl ;
  }
}


void RooAbsReal::fillTreeBranch(TTree& t) 
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  TBranch* branch = t.GetBranch(cleanBranchName()) ;
  if (!branch) { 
    cout << "RooAbsReal::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree: " << cleanBranchName() << endl ;
    assert(0) ;
  }
  branch->Fill() ;
  
}



void RooAbsReal::setTreeBranchStatus(TTree& t, Bool_t active) 
{
  // (De)Activate associate tree branch
  TBranch* branch = t.GetBranch(cleanBranchName()) ;
  if (branch) { 
    t.SetBranchStatus(cleanBranchName(),active?1:0) ;
  }
}



RooAbsArg *RooAbsReal::createFundamental(const char* newname) const {
  // Create a RooRealVar fundamental object with our properties. The new
  // object will be created without any fit limits.

  RooRealVar *fund= new RooRealVar(newname?newname:GetName(),GetTitle(),_value,getUnit());
  fund->removeRange();
//   fund->setPlotRange(getPlotMin(),getPlotMax());
//   fund->setPlotBins(getPlotBins());
  fund->setPlotLabel(getPlotLabel());
  fund->setAttribute("fundamentalCopy");
  return fund;
}



Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a) const
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a, const RooArgProxy& b) const
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;  
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a, const RooArgProxy& b,
			      const RooArgProxy& c) const
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;
  nameList.Add(new TObjString(c.absArg()->GetName())) ;
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			      const RooArgProxy& a, const RooArgProxy& b,
			      const RooArgProxy& c, const RooArgProxy& d) const
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  nameList.Add(new TObjString(a.absArg()->GetName())) ;
  nameList.Add(new TObjString(b.absArg()->GetName())) ;
  nameList.Add(new TObjString(c.absArg()->GetName())) ;
  nameList.Add(new TObjString(d.absArg()->GetName())) ;
  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}


Bool_t RooAbsReal::matchArgs(const RooArgSet& allDeps, RooArgSet& analDeps, 
			    const RooArgSet& set) const 
{
  // Wrapper function for matchArgsByName()
  TList nameList ;
  TIterator* iter = set.createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    nameList.Add(new TObjString(arg->GetName())) ;    
  }
  delete iter ;

  Bool_t result = matchArgsByName(allDeps,analDeps,nameList) ;
  nameList.Delete() ;
  return result ;
}



Bool_t RooAbsReal::matchArgsByName(const RooArgSet &allArgs, RooArgSet &matchedArgs,
				  const TList &nameList) const {
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



RooNumIntConfig* RooAbsReal::defaultIntegratorConfig() 
{
  return &RooNumIntConfig::defaultConfig() ;
}


RooNumIntConfig* RooAbsReal::specialIntegratorConfig() const 
{
  return _specIntegratorConfig ;
}


const RooNumIntConfig* RooAbsReal::getIntegratorConfig() const 
{
  const RooNumIntConfig* config = specialIntegratorConfig() ;
  if (config) return config ;
  return defaultIntegratorConfig() ;
}


void RooAbsReal::setIntegratorConfig(const RooNumIntConfig& config) 
{
  if (_specIntegratorConfig) {
    delete _specIntegratorConfig ;
  }
  _specIntegratorConfig = new RooNumIntConfig(config) ;  
}


void RooAbsReal::setIntegratorConfig() 
{
  if (_specIntegratorConfig) {
    delete _specIntegratorConfig ;
  }
  _specIntegratorConfig = 0 ;
}



void RooAbsReal::optimizeDirty(RooAbsData& dataset, const RooArgSet* normSet, Bool_t /*verbose*/) 
{
  getVal(normSet) ;

  RooArgSet branchList("branchList") ;
  setOperMode(RooAbsArg::ADirty) ;
  branchNodeServerList(&branchList) ;

  TIterator* bIter = branchList.createIterator() ;
  RooAbsArg* branch ;
  while((branch=(RooAbsArg*)bIter->Next())) {
    if (branch->dependsOn(*dataset.get())) {

      RooArgSet* bdep = branch->getObservables(dataset.get()) ;
      if (bdep->getSize()>0) {
	branch->setOperMode(RooAbsArg::ADirty) ;
      } else {
	//cout << "using lazy evaluation for node " << branch->GetName() << endl ;
      }
      delete bdep ;
    }
  }
  delete bIter ;
  
  dataset.setDirtyProp(kFALSE) ;
}


void RooAbsReal::doConstOpt(RooAbsData& dataset, const RooArgSet* normSet, Bool_t verbose) 
{
  // optimizeDirty must have been run first!

  // Find cachable branches and cache them with the data set
  RooArgSet cacheList ;
  findCacheableBranches(this,&dataset,cacheList,normSet,verbose) ;
  dataset.cacheArgs(cacheList,normSet) ;  

  // Find unused data variables after caching and disable them
  RooArgSet pruneList("pruneList") ;
  findUnusedDataVariables(&dataset,pruneList,verbose) ;
  findRedundantCacheServers(&dataset,cacheList,pruneList,verbose) ;

  if (pruneList.getSize()!=0) {
    // Deactivate tree branches here
    if (verbose) {
      cout << "RooAbsReal::optimizeConst: The following unused tree branches are deactivated: " ; 
      pruneList.Print("1") ;
    }
    dataset.setArgStatus(pruneList,kFALSE) ;
  }

  TIterator* cIter = cacheList.createIterator() ;
  RooAbsArg *cacheArg ;
  while((cacheArg=(RooAbsArg*)cIter->Next())){
    cacheArg->setOperMode(RooAbsArg::AClean) ;
  }
  delete cIter ;
}


void RooAbsReal::undoConstOpt(RooAbsData& dataset, const RooArgSet* normSet, Bool_t verbose) 
{
  // Delete the cache
  dataset.resetCache() ;

  // Reactivate all tree branches
  dataset.setArgStatus(*dataset.get(),kTRUE) ;
  
  // Reset all nodes to ADirty 
  optimizeDirty(dataset,normSet,verbose) ;
}


Bool_t RooAbsReal::findCacheableBranches(RooAbsArg* arg, RooAbsData* dset, 
					    RooArgSet& cacheList, const RooArgSet* normSet, Bool_t verbose) 
{
  // Find branch PDFs with all-constant parameters, and add them
  // to the dataset cache list

  // Evaluate function with current normalization in case servers
  // are created on the fly
  RooAbsReal* realArg = dynamic_cast<RooAbsReal*>(arg) ;
  if (realArg) {
    realArg->getVal(normSet) ;
  }

  TIterator* sIter = arg->serverIterator() ;
  RooAbsArg* server ;

  while((server=(RooAbsArg*)sIter->Next())) {
    if (server->isDerived()) {
      // Check if this branch node is eligible for precalculation
      Bool_t canOpt(kTRUE) ;

      RooArgSet* branchParamList = server->getParameters(dset) ;
      TIterator* pIter = branchParamList->createIterator() ;
      RooAbsArg* param ;
      while((param = (RooAbsArg*)pIter->Next())) {
	if (!param->isConstant()) canOpt=kFALSE ;
      }
      delete pIter ;
      delete branchParamList ;

      if (canOpt) {
	if (!cacheList.find(server->GetName())) {

	  if (verbose) {
	    cout << "RooAbsReal::optimize: component " 
		 << server->GetName() << " will be cached" << endl ;
	  }

	  // Add to cache list
	  cacheList.add(*server) ;
	} 
      } else {
	// Recurse if we cannot optimize at this level
	findCacheableBranches(server,dset,cacheList,normSet,verbose) ;
      }
    }
  }
  delete sIter ;
  return kFALSE ;
}



void RooAbsReal::findUnusedDataVariables(RooAbsData* dset,RooArgSet& pruneList, Bool_t /*verbose*/) 
{
  RooArgSet* observables = getObservables(dset) ;
  TIterator* vIter = observables->createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*) vIter->Next())) {    
    if (!dependsOn(*arg)) {
      pruneList.add(*arg) ;
    }
  }
  delete vIter ;
  delete observables ;
}


void RooAbsReal::findRedundantCacheServers(RooAbsData* dset,RooArgSet& cacheList, RooArgSet& pruneList, Bool_t /*verbose*/) 
{
  TIterator* vIter = dset->get()->createIterator() ;
  RooAbsArg *var ;
  while ((var=(RooAbsArg*) vIter->Next())) {
    if (allClientsCached(var,cacheList)) {
      pruneList.add(*var) ;
    }
  }
  delete vIter ;
}



Bool_t RooAbsReal::allClientsCached(RooAbsArg* var, RooArgSet& cacheList)
{
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


void RooAbsReal::selectNormalization(const RooArgSet*, Bool_t) 
{
} 


void RooAbsReal::selectNormalizationRange(const char*, Bool_t) 
{
}



Int_t RooAbsReal::getMaxVal(const RooArgSet& /*vars*/) const 
  // Advertise capability to determine maximum value of function for given set of 
  // observables. If no direct generator method is provided, this information
  // will assist the accept/reject generator to operate more efficiently as
  // it can skip the initial trial sampling phase to empirically find the function
  // maximum
{
  return 0 ;
}


Double_t RooAbsReal::maxVal(Int_t /*code*/) 
{
  // Return maximum value for set of observables identified by code assigned
  // in getMaxVal

  assert(1) ;
  return 0 ;
}

