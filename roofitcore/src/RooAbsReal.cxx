/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsReal.cc,v 1.69 2001/12/04 22:40:56 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 *   AB, Adrian Bevan, Liverpool University, bevan@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *   24-Aug-2001 AB Added TH2F * createHistogram methods
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooAbsReal is the common abstract base class for objects that represent a
// real value. Implementation of RooAbsReal may be derived, there no interface
// is provided to modify the contents.
// 
// This class holds in addition a unit and label string, as well
// as a plot range and number of plot bins and plot creation methods.

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooArgList.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooCurve.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooArgProxy.hh"
#include "RooFitCore/RooFormulaVar.hh"
#include "RooFitCore/RooRealFixedBinIter.hh"
#include "RooFitCore/RooRealBinding.hh"
#include "RooFitCore/RooRealIntegral.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooCustomizer.hh"
#include "RooFitCore/RooAbsData.hh"
#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooScaledFunc.hh"
#include "RooFitCore/RooDataProjBinding.hh"

#include <iostream.h>

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


RooAbsReal::RooAbsReal(const char *name, const char *title, const char *unit) : 
  RooAbsArg(name,title), _unit(unit), _plotBins(100), _value(0), 
  _plotMin(0), _plotMax(0), _forceNumInt(kFALSE)
{
  // Constructor with unit label
  setValueDirty() ;
  setShapeDirty() ;
}

RooAbsReal::RooAbsReal(const char *name, const char *title, Double_t minVal,
		       Double_t maxVal, const char *unit) :
  RooAbsArg(name,title), _unit(unit), _plotBins(100), _value(0), 
  _plotMin(minVal), _plotMax(maxVal), _forceNumInt(kFALSE)
{
  // Constructor with plot range and unit label
  setValueDirty() ;
  setShapeDirty() ;
}


RooAbsReal::RooAbsReal(const RooAbsReal& other, const char* name) : 
  RooAbsArg(other,name), _unit(other._unit), _plotBins(other._plotBins), 
  _plotMin(other._plotMin), _plotMax(other._plotMax), _value(other._value),
  _forceNumInt(other._forceNumInt)
{

  // Copy constructor
}


RooAbsReal::~RooAbsReal()
{
  // Destructor
}



Bool_t RooAbsReal::operator==(Double_t value) const
{
  // Equality operator comparing to a Double_t
  return (getVal()==value) ;
}


Double_t RooAbsReal::getVal(const RooArgSet* set) const
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval(set) ;
    clearValueDirty() ; 
    clearShapeDirty() ; 
  }
  
  return _value ;
}


Double_t RooAbsReal::traceEval(const RooArgSet* nset) const
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


Int_t RooAbsReal::getAnalyticalIntegralWN(RooArgSet& allDeps, RooArgSet& analDeps, const RooArgSet* normSet) const
{
  // Default implementation of getAnalyticalIntegralWN for real valued objects defers to
  // normalization invariant getAnalyticalIntegral()
  return _forceNumInt ? 0 : getAnalyticalIntegral(allDeps,analDeps) ;
}


Int_t RooAbsReal::getAnalyticalIntegral(RooArgSet& allDeps, RooArgSet& analDeps) const
{
  // By default we do not supply any analytical integrals
  return 0 ;
}


Double_t RooAbsReal::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const
{
  // Default implementation of analyticalIntegralWN handles only the pass-through
  // scenario (code =0). All other codes are deferred to to the normalization
  // invariant analyticalIntegral() 

  if (code==0) return getVal(normSet) ;
  return analyticalIntegral(code) ;
}


Double_t RooAbsReal::analyticalIntegral(Int_t code) const
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



Bool_t RooAbsReal::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  //Read object contents from stream (dummy for now)
  return kFALSE ;
} 

void RooAbsReal::writeToStream(ostream& os, Bool_t compact) const
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


void RooAbsReal::setPlotRange(Double_t min, Double_t max) {
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


void RooAbsReal::setPlotBins(Int_t value) {
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


Bool_t RooAbsReal::isValidReal(Double_t value, Bool_t printError) const 
{
  // Check if given value is valid
  return kTRUE ;
}



const RooAbsReal *RooAbsReal::createProjection(const RooArgSet &dependentVars, const RooArgSet *projectedVars,
					       RooArgSet *&cloneSet) const {
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

  // Check that the dependents are all fundamental. Filter out any that we
  // do not depend on, and make substitutions by name in our leaf list.
  // Check for overlaps with the projection variables.
  TIterator *dependentIterator= dependentVars.createIterator();
  assert(0 != dependentIterator);
  const RooAbsArg *arg(0);
  while(arg= (const RooAbsArg*)dependentIterator->Next()) {
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
	RooArgSet* lvDep = arg->getDependents(&leafNodes) ;
	RooAbsArg* lvs ;
	TIterator* iter = lvDep->createIterator() ;
	while(lvs=(RooAbsArg*)iter->Next()) {
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

  // Make a deep-clone of ourself so later operations do not disturb our original state
  cloneSet= (RooArgSet*)RooArgSet(*this).snapshot();
  RooAbsReal *clone= (RooAbsReal*)cloneSet->find(GetName());

  // The remaining entries in our list of leaf nodes are the the external
  // dependents (x) and parameters (p) of the projection. Patch them back
  // into the clone. This orphans the nodes they replace, but the orphans
  // are still in the cloneList and so will be cleaned up eventually.
  //cout << "redirection leafNodes : " ; leafNodes.Print("1") ;

  TIterator* it = leafNodes.createIterator() ;
  RooAbsArg* a ;
  while(a = (RooAbsArg*) it->Next()) {
    //cout << "redir ln = " << a << " " << a->GetName() << endl ;
  }
  clone->recursiveRedirectServers(leafNodes);

  // Create the set of normalization variables to use in the projection integrand

//   // WVE dependentVars might be LValue, expand into final servers
//   RooArgSet* xxx = dependentVars.getDependents(...) ;
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

  RooRealIntegral *projected= new RooRealIntegral(name.Data(),title.Data(),*clone,*projectedVars,&normSet);
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

  // Create a standalone projection object to use for calculating bin contents
  RooArgSet *cloneSet(0);
  const RooAbsReal *projected= createProjection(plotClones,projectedVars,cloneSet);

  // Prepare to loop over the histogram bins
  Int_t xbins(0),ybins(1),zbins(1);
  RooRealVar *xvar(0),*yvar(0),*zvar(0);
  TAxis *xaxis(0),*yaxis(0),*zaxis(0);
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





RooPlot* RooAbsReal::plotOn(RooPlot *frame, Option_t* drawOptions, 
			    Double_t scaleFactor, ScaleType stype, 
			    const RooAbsData* projData, const RooArgSet* projSet) const
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
  
  // Make list of variables to be projected
  RooArgSet projectedVars ;
  if (projSet) {
    makeProjectionSet(frame->getPlotVar(),projSet,projectedVars,kFALSE) ;

    // Print list of non-projected variables
    if (frame->getNormVars()) {
      RooArgSet sliceSet(*frame->getNormVars()) ;
      sliceSet.remove(projectedVars,kTRUE) ;
      sliceSet.remove(*sliceSet.find(frame->getPlotVar()->GetName())) ;
      if (sliceSet.getSize()) {
	cout << "RooAbsReal::plotOn(" << GetName() << ") plot on " 
	     << frame->getPlotVar()->GetName() << " represents a slice in " ;
	sliceSet.Print("1") ;
      }
    }
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  }

  RooArgSet* projDataNeededVars(0) ;
  // Take out data-projected dependents from projectedVars
  if (projData) {
    projDataNeededVars = (RooArgSet*) projectedVars.selectCommon(*projData->get()) ;
    projectedVars.remove(*projData->get(),kTRUE,kTRUE) ;
  }

  // Clone the plot variable
  RooAbsReal* realVar = (RooRealVar*) frame->getPlotVar() ;
  RooArgSet* plotCloneSet = (RooArgSet*) RooArgSet(*realVar).snapshot(kTRUE) ;
  RooRealVar* plotVar = (RooRealVar*) plotCloneSet->find(realVar->GetName());

  // Inform user about projections
  if (projectedVars.getSize()) {
    cout << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName() 
	 << " projects variables " ; projectedVars.Print("1") ;
  }  
  if (projDataNeededVars) {
    cout << "RooAbsReal::plotOn(" << GetName() << ") plot on " << plotVar->GetName() 
	 << " projects with data variables " ; projDataNeededVars->Print("1") ;
  }

  // Create projection integral
  RooArgSet* projectionCompList ;

  RooArgSet* deps = getDependents(frame->getNormVars()) ;
  deps->remove(projectedVars,kTRUE,kTRUE) ;
  if (projDataNeededVars) {
    deps->remove(*projDataNeededVars,kTRUE,kTRUE) ;
  }
  deps->remove(*plotVar,kTRUE,kTRUE) ;
  deps->add(*plotVar) ;

  const RooAbsReal *projection = createProjection(*deps, &projectedVars, projectionCompList) ;
  delete deps ;

  // Reset name of projection to name of PDF to get appropriate curve name
  //((RooAbsArg*)projection)->SetName(GetName()) ;


  // Apply data projection, if requested
  if (projData) {

    // Disable dirty state propagation in projection
    RooArgSet branchList("branchList") ;
    ((RooAbsReal*)projection)->setOperMode(RooAbsArg::ADirty) ;
    projection->branchNodeServerList(&branchList) ;
    TIterator* bIter = branchList.createIterator() ;
    RooAbsArg* branch ;
    while(branch=(RooAbsArg*)bIter->Next()) {
      branch->setOperMode(RooAbsArg::ADirty) ;
    }
    delete bIter ;

    // If data set contains more columns than needed, make reduced copy first
    RooDataSet* projDataSel((RooDataSet*)projData) ;
    if (projDataNeededVars->getSize()<projData->get()->getSize()) {
      projDataSel = (RooDataSet*) ((RooDataSet*)projData)->reduce(*projDataNeededVars) ;
      cout << "RooAbsReal::plotOn(" << GetName() 
	   << ") only the following components of the projection data will be used: " ; 
      projDataNeededVars->Print("1") ;
    }

    projectionCompList->find(GetName())->attachDataSet(*projDataSel) ;
    RooDataProjBinding projBind(*projection,*projDataSel,*plotVar) ;
    RooScaledFunc scaleBind(projBind,scaleFactor);
    RooCurve *curve = new RooCurve(projection->GetName(),projection->GetTitle(),scaleBind,
				   frame->GetXaxis()->GetXmin(),frame->GetXaxis()->GetXmax(),
				   frame->GetNbinsX()) ;
    cout << endl ;

    // add this new curve to the specified plot frame
    frame->addPlotable(curve, drawOptions);

    if (projDataSel!=projData) delete projDataSel ;
    delete projDataNeededVars ;
       
  } else {
    
    // create a new curve of our function using the clone to do the evaluations
    RooCurve *curve = new RooCurve(*projection,*plotVar,frame->GetXaxis()->GetXmin(),
				   frame->GetXaxis()->GetXmax(),frame->GetNbinsX(), scaleFactor);

    // add this new curve to the specified plot frame
    frame->addPlotable(curve, drawOptions);
  }

  delete projectionCompList ;
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
  while(sliceArg=(RooAbsArg*)iter->Next()) {
    RooAbsArg* arg = projectedVars.find(sliceArg->GetName()) ;
    if (arg) {
      projectedVars.remove(*arg) ;
    } else {
      cout << "RooAbsReal::plotSliceOn(" << GetName() << ") slice variable " 
	   << sliceArg->GetName() << " was not projected anyway" << endl ;
    }
  }
  delete iter ;

  return plotOn(frame,drawOptions,scaleFactor,stype,projData,&projectedVars) ;
}





RooPlot* RooAbsReal::plotAsymOn(RooPlot *frame, const RooAbsCategoryLValue& asymCat, Option_t* drawOptions, 
				Double_t scaleFactor, const RooAbsData* projData, const RooArgSet* projSet) const
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
  if (projSet) {
    makeProjectionSet(frame->getPlotVar(),projSet,projectedVars,kFALSE) ;

    // Print list of non-projected variables
    if (frame->getNormVars()) {
      RooArgSet sliceSet(*frame->getNormVars()) ;
      sliceSet.remove(projectedVars,kTRUE) ;
      sliceSet.remove(*sliceSet.find(frame->getPlotVar()->GetName())) ;
      if (sliceSet.getSize()) {
	cout << "RooAbsReal::plotAsymOn(" << GetName() << ") plot on " 
	     << frame->getPlotVar()->GetName() << " represents a slice in " ;
	sliceSet.Print("1") ;
      }
    }
  } else {
    makeProjectionSet(frame->getPlotVar(),frame->getNormVars(),projectedVars,kTRUE) ;
  }

  // Take out data-projected dependens from projectedVars
  if (projData) {
    projectedVars.remove(*projData->get(),kTRUE,kTRUE) ;
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
  const RooAbsReal *posProj = funcPos->createProjection(RooArgSet(*plotVar,*asymPos), &projectedVars, posProjCompList) ;
  const RooAbsReal *negProj = funcNeg->createProjection(RooArgSet(*plotVar,*asymNeg), &projectedVars, negProjCompList) ;

  // Create a RooFormulaVar representing the asymmetry
  TString asymName(GetName()) ;
  asymName.Append("Asymmetry[") ;
  asymName.Append(asymCat.GetName()) ;
  asymName.Append("]") ;
  TString asymTitle(asymCat.GetName()) ;
  asymTitle.Append(" Asymmetry of ") ;
  asymTitle.Append(GetTitle()) ;
  RooFormulaVar* funcAsym = new RooFormulaVar(asymName,asymTitle,"(@0-@1)/(@0+@1)",RooArgSet(*posProj,*negProj)) ;

  if (projData) {
    
    RooDataProjBinding projBind(*funcAsym,*projData,*plotVar) ;
    ((RooAbsReal*)posProj)->attachDataSet(*projData) ;
    ((RooAbsReal*)negProj)->attachDataSet(*projData) ;
    RooScaledFunc scaleBind(projBind,scaleFactor);
    RooCurve *curve = new RooCurve(funcAsym->GetName(),funcAsym->GetTitle(),scaleBind,
				   frame->GetXaxis()->GetXmin(),frame->GetXaxis()->GetXmax(),frame->GetNbinsX()) ;
    dynamic_cast<TAttLine*>(curve)->SetLineColor(2) ;
    // add this new curve to the specified plot frame
    frame->addPlotable(curve, drawOptions);
       
  } else {

    RooCurve* curve= new RooCurve(*funcAsym,*plotVar,frame->GetXaxis()->GetXmin(),
				  frame->GetXaxis()->GetXmax(),frame->GetNbinsX(),scaleFactor);
    dynamic_cast<TAttLine*>(curve)->SetLineColor(2) ;
    
    // add this new curve to the specified plot frame
    frame->addPlotable(curve, drawOptions);
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
    RooArgSet* plotServers = plotVar->getDependents(&projectedVars) ;
    TIterator* psIter = plotServers->createIterator() ;
    RooAbsArg* ps ;
    while(ps=(RooAbsArg*)psIter->Next()) {
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
  while(arg=(RooAbsArg*)iter->Next()) {
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
    Float_t& tmp = (Float_t&) other->_value ;
    _value = tmp ;
  } else if (source->getAttribute("INTEGER_TREE_BRANCH")) {
    Int_t& tmp = (Int_t&) other->_value ;
    _value = tmp ;
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
    } 

    t.SetBranchAddress(cleanName,&_value) ;
//     cout << "RooAbsReal::attachToTree(" << cleanName << "): branch already exists in tree " 
// 	 << (void*)&t << ", changing address" << endl ;
  } else {
    TString format(cleanName);
    format.Append("/D");
    t.Branch(cleanName, &_value, (const Text_t*)format, bufSize);

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


TString RooAbsReal::cleanBranchName() const
{
  // Construct a mangled name from the actual name that
  // is free of any math symbols that might be interpreted by TTree

  TString cleanName(GetName()) ;
  cleanName.ReplaceAll("/","D") ;
  cleanName.ReplaceAll("-","M") ;
  cleanName.ReplaceAll("+","P") ;
  cleanName.ReplaceAll("*","X") ;
  cleanName.ReplaceAll("[","L") ;
  cleanName.ReplaceAll("]","R") ;
  cleanName.ReplaceAll("(","L") ;
  cleanName.ReplaceAll(")","R") ;
  cleanName.ReplaceAll("{","L") ;
  cleanName.ReplaceAll("}","R") ;

  if (cleanName.Length()<=60) return cleanName ;

  // Name is too long, truncate and include CRC32 checksum of full name in clean name
  static char buf[1024] ;
  strcpy(buf,cleanName.Data()) ;
  sprintf(buf+46,"_CRC%08x",crc32(cleanName.Data())) ;

  return TString(buf) ;
}





UInt_t RooAbsReal::crc32(const char* data) const
{
  // Calculate and extract length of string
  Int_t len = strlen(data) ;
  if (len<4) {
    cout << "RooAbsReal::crc32 cannot calculate checksum of less than 4 bytes of data" << endl ;
    return 0 ;
  }

  // Initialize CRC table on first use
  static Bool_t init(kFALSE) ;
  static unsigned int crctab[256];
  if (!init) {
    int i, j;
    unsigned int crc;
    for (i = 0; i < 256; i++){
      crc = i << 24;
      for (j = 0; j < 8; j++) {
	if (crc & 0x80000000) {
	  crc = (crc << 1) ^ 0x04c11db7 ;
	} else {
	  crc = crc << 1;
	}
      }
      crctab[i] = crc;
    }
    init = kTRUE ;
  }
  
  unsigned int        result(0);
  int                 i(0);
  unsigned char       octet(0);
  
  result = *data++ << 24;
  result |= *data++ << 16;
  result |= *data++ << 8;
  result |= *data++;
  result = ~ result;
  len -=4;
  
  for (i=0; i<len; i++) {
    result = (result << 8 | *data++) ^ crctab[result >> 24];
  }

  return ~result;
}



RooAbsArg *RooAbsReal::createFundamental(const char* newname) const {
  // Create a RooRealVar fundamental object with our properties. The new
  // object will be created without any fit limits.

  RooRealVar *fund= new RooRealVar(newname?newname:GetName(),GetTitle(),_value,getUnit());
  fund->removeFitRange();
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
  while (arg=(RooAbsArg*)iter->Next()) {
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
  TObjString *name(0);
  Bool_t isMatched(kTRUE);
  while(isMatched && (name= (TObjString*)iterator->Next())) {
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




