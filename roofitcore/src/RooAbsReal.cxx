/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsReal.cc,v 1.43 2001/09/24 23:05:57 verkerke Exp $
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

// -- CLASS DESCRIPTION --
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

#include <iostream.h>

#include "TObjString.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2.h"
#include "TBranch.h"
#include "TLeaf.h"

ClassImp(RooAbsReal)
;


RooAbsReal::RooAbsReal(const char *name, const char *title, const char *unit) : 
  RooAbsArg(name,title), _unit(unit), _plotBins(100), _value(0), 
  _plotMin(0), _plotMax(0), _plotBinW(0)
{
  // Constructor
  setValueDirty() ;
  setShapeDirty() ;
}

RooAbsReal::RooAbsReal(const char *name, const char *title, Double_t minVal,
		       Double_t maxVal, const char *unit) :
  RooAbsArg(name,title), _unit(unit), _plotBins(100), _value(0), 
  _plotMin(minVal), _plotMax(maxVal)
{
  // Constructor with plot range
  calcBinWidth() ;
  setValueDirty() ;
  setShapeDirty() ;
}


RooAbsReal::RooAbsReal(const RooAbsReal& other, const char* name) : 
  RooAbsArg(other,name), _unit(other._unit), _plotBins(other._plotBins), 
  _plotMin(other._plotMin), _plotMax(other._plotMax), _value(other._value),
  _plotBinW(other._plotBinW)
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
  // By default defer to normSet-invariant version
  return getAnalyticalIntegral(allDeps,analDeps) ;
}


Int_t RooAbsReal::getAnalyticalIntegral(RooArgSet& allDeps, RooArgSet& analDeps) const
{
  // By default we do not supply any analytical integrals
  return 0 ;
}


Double_t RooAbsReal::analyticalIntegralWN(Int_t code, const RooArgSet* normSet) const
{
  // Implement pass-through scenario, defer other codes to subclass implementations
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
    os << indent << "  Plot range is [ " << getPlotMin() << unit << " , "
       << getPlotMax() << unit << " ]" << endl;
    if(opt >= Verbose) {
      os << indent << "  Plot bins = " << getPlotBins();
      Double_t range= getPlotMax()-getPlotMin();
      if(range > 0) os << " (" << range/getPlotBins() << unit << "/bin)";
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

  calcBinWidth() ;
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

  calcBinWidth() ;
}


void RooAbsReal::setPlotRange(Double_t min, Double_t max) {
  // Check if new limit is consistent
  if (min>max) {
    cout << "RooAbsReal::setPlotMinMax(" << GetName() 
	 << "): Proposed new integration max. smaller than min., setting max. to min." << endl ;
    _plotMin = min ;
    _plotMax = min ;
  } else {
    _plotMin = min ;
    _plotMax = max ;
  }
  calcBinWidth() ;
}


void RooAbsReal::setPlotBins(Int_t value) {
  // Set number of histogram bins 
  _plotBins = value ;  
  calcBinWidth() ;
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



TH1F *RooAbsReal::createHistogram(const char *name, const char *yAxisLabel, Int_t bins) const {
  // Create an empty 1D-histogram with appropriate scale and labels for this variable.
  // This method uses the default plot range which can be changed using the
  // setPlotMin(),setPlotMax() methods. Uses the default binning (setPlotBins())
  // unless you specify your own binning.
  // The caller takes ownership of the returned object and is responsible for deleting it.

  return createHistogram(name, yAxisLabel, _plotMin, _plotMax, bins > 0 ? bins : getPlotBins());
}

TH1F *RooAbsReal::createHistogram(const char *name, const char *yAxisLabel,
				  Double_t lo, Double_t hi, Int_t bins) const {
  // Create an empty 1D-histogram with appropriate scale and labels for this variable.
  // Binning must be specified with this method since the default binning is associated
  // with the default plot ranges, but you have asked for a non-default range.
  // The caller takes ownership of the returned object and is responsible for deleting it.

  // Use a histogram name of the form <name>_<var>
  TString histName(name);
  if(!histName.IsNull()) histName.Append("_");
  histName.Append(GetName());

  // create the histogram
  TH1F* histogram= new TH1F(histName.Data(), fTitle, bins, lo, hi);
  if(!histogram) {
    cout << fName << "::createHistogram: unable to create a new 1D histogram" << endl;
    return 0;
  }

  // Set the x-axis title from our own title, adding units if we have them.
  TString xTitle(fTitle);
  if(strlen(getUnit())) {
    xTitle.Append(" (");
    xTitle.Append(getUnit());
    xTitle.Append(")");
  }
  histogram->SetXTitle(xTitle.Data());

  // Set the y-axis title if given one
  if(0 != yAxisLabel && strlen(yAxisLabel)) {
    TString yTitle(yAxisLabel);
    Double_t delta= (hi - lo)/bins;
    yTitle.Append(Form(" / %g",delta));
    if(strlen(getUnit())) {
      yTitle.Append(" ");
      yTitle.Append(getUnit());
    }
    histogram->SetYTitle(yTitle.Data());
  }
  return histogram;
}

TH2F *RooAbsReal::createHistogram(const char *name, const RooAbsReal &yvar, const char *zAxisLabel,
				  Int_t xbins, Int_t ybins) const 
{
  // Create an empty 2D-histogram with appropriate scale and labels for this variable (x)
  // and the specified y variable.
  // This method uses the default plot ranges for x and y which can be changed using the
  // setPlotMin(),setPlotMax() methods. Uses the default binning (setPlotBins())
  // unless you specify your own binning.
  // The caller takes ownership of the returned object and is responsible for deleting it.

  return createHistogram(name, yvar, zAxisLabel,
			 this->getPlotMin(), this->getPlotMax(), xbins > 0 ? xbins : this->getPlotBins(),
			 yvar.getPlotMin(), yvar.getPlotMax(), ybins > 0 ? ybins : yvar.getPlotBins());
}

TH2F *RooAbsReal::createHistogram(const char *name, const RooAbsReal &yvar, const char *zAxisLabel,
                                  Double_t xlo, Double_t xhi, Int_t xbins,
                                  Double_t ylo, Double_t yhi, Int_t ybins) const 
{
  // Create a 2D-histogram with appropriate scale and labels for this variable.
  // Binning must be specified with this method since the default binning is associated
  // with the default plot ranges, but you have asked for a non-default range.
  // The caller takes ownership of the returned object and is responsible for deleting it.

  // Use a histogram name of the form <name>_<xvar>_<yvar>
  TString histName(name);
  if(!histName.IsNull()) histName.Append("_");
  histName.Append(GetName());
  histName.Append("_");
  histName.Append(yvar.GetName());

  // create the histogram
  TH2F* histogram= new TH2F(histName.Data(), fTitle, xbins, xlo, xhi, ybins, ylo, yhi);
  if(!histogram) {
    cout << fName << "::createHistogram: unable to create a new 2D histogram" << endl;
    return 0;
  }

  // Set the x-axis title from our own title, adding units if we have them.
  TString xTitle(fTitle);
  if(strlen(getUnit())) {
    xTitle.Append(" (");
    xTitle.Append(getUnit());
    xTitle.Append(")");
  }
  histogram->SetXTitle(xTitle.Data());

  // Set the y-axis title from the y variable, adding units if it has them.
  TString yTitle(yvar.GetTitle());
  if(strlen(yvar.getUnit())) {
    yTitle.Append(" (");
    yTitle.Append(yvar.getUnit());
    yTitle.Append(")");
  }
  histogram->SetYTitle(yTitle.Data());

  // Set the z-axis title if given one
  if(0 != zAxisLabel && strlen(zAxisLabel)) {
    TString zTitle(zAxisLabel);
    Double_t delta= (xhi - xlo)/xbins;
    zTitle.Append(Form(" / ( %g",delta));
    if(strlen(getUnit())) {
      zTitle.Append(" ");
      zTitle.Append(getUnit());
    }
    delta= (yhi - ylo)/ybins;
    zTitle.Append(Form(" x %g",delta));
    if(strlen(yvar.getUnit())) {
      zTitle.Append(" ");
      zTitle.Append(yvar.getUnit());
    }
    zTitle.Append(" )");
    histogram->SetZTitle(zTitle.Data());
  }

  return histogram;
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
  leafNodeServerList(&leafNodes,this);

  // Check that the dependents are all fundamental. Filter out any that we
  // do not depend on, and make substitutions by name in our leaf list.
  // Check for overlaps with the projection variables.
  TIterator *dependentIterator= dependentVars.createIterator();
  assert(0 != dependentIterator);
  const RooAbsArg *arg(0);
  while(arg= (const RooAbsArg*)dependentIterator->Next()) {
    if(!arg->isFundamental()) {
      cout << ClassName() << "::" << GetName() << ":createProjection: variable \"" << arg->GetName()
	   << "\" of wrong type: " << arg->ClassName() << endl;
      delete dependentIterator;
      return 0;
    }
    RooAbsArg *found= leafNodes.find(arg->GetName());
    if(!found) {
      cout << ClassName() << "::" << GetName() << ":createProjection: \"" << arg->GetName()
	   << "\" is not a dependent and will be ignored." << endl;
      continue;
    }
    if(found != arg) leafNodes.replace(*found,*arg);
    // check if this arg is also in the projection set
    if(0 != projectedVars && projectedVars->find(arg->GetName())) {
      cout << ClassName() << "::" << GetName() << ":createProjection: \"" << arg->GetName()
	   << "\" cannot be both a dependent and a projected variable." << endl;
      delete dependentIterator;
      return 0;
    }
  }

  // Remove the projected variables from the list of leaf nodes, if necessary.
  if(0 != projectedVars) leafNodes.remove(*projectedVars);

  // Make a deep-clone of ourself so later operations do not disturb our original state
  cloneSet= (RooArgSet*)RooArgSet(*this).snapshot();
  RooAbsReal *clone= (RooAbsReal*)cloneSet->find(GetName());

  // The remaining entries in our list of leaf nodes are the the external
  // dependents (x) and parameters (p) of the projection. Patch them back
  // into the clone. This orphans the nodes they replace, but the orphans
  // are still in the cloneList and so will be cleaned up eventually.
  clone->recursiveRedirectServers(leafNodes);

  // Create the set of normalization variables to use in the projection integrand
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
    else {
      plotClones.addClone(*realVar,kTRUE); // do not complain about duplicates
    }
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
	zvar->setVal(zaxis->GetBinCenter(zbin+1));
      }
      // fall through to next case...
    case 2:
      if(bin % xbins == 0) {
	ybin= (ybin + 1)%ybins;
	yvar->setVal(yaxis->GetBinCenter(ybin+1));
      }
      // fall through to next case...
    case 1:
      xbin= (xbin + 1)%xbins;
      xvar->setVal(xaxis->GetBinCenter(xbin+1));
      break;
    default:
      cout << "RooAbsReal::fillHistogram: Internal Error!" << endl;
      break;
    }
    Double_t result= scaleFactor*projected->getVal();
    hist->SetBinContent(hist->GetBin(xbin+1,ybin+1,zbin+1),result);
  }

  // cleanup
  delete cloneSet;

  return hist;
}

RooPlot *RooAbsReal::plotOn(RooPlot* frame, Option_t* drawOptions, Double_t scaleFactor) const {
  // Plot a smooth curve of this object's value on the specified frame.

  // check that we are passed a valid plot frame to use
  if(0 == frame) {
    cout << ClassName() << "::" << GetName() << ":plotOn: frame is null" << endl;
    return 0;
  }

  // check that this frame knows what variable to plot
  RooAbsReal *var= frame->getPlotVar();
  if(0 == var) {
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: frame does not specify a plot variable" << endl;
    return 0;
  }

  // check that the plot variable is not derived
  RooRealVar* realVar= dynamic_cast<RooRealVar*>(var);
  if(0 == realVar) {
     cout << ClassName() << "::" << GetName() << ":plotOn: cannot plot variable \""
	  << var->GetName() << "\" of type " << var->ClassName() << endl;
    return 0;
  }

  // check if we actually depend on the plot variable
  if(!this->dependsOn(*realVar)) {
    cout << ClassName() << "::" << GetName() << ":plotOn: WARNING: variable is not an explicit dependent: "
	 << realVar->GetName() << endl;
  }

  // deep-clone ourselves so that the plotting process will not disturb
  // our original expression tree
  RooArgSet *cloneList = (RooArgSet*) RooArgSet(*this).snapshot() ;
  RooAbsReal *clone= (RooAbsReal*) cloneList->find(GetName()) ;

  // redirect our clone to use the plot variable
  RooArgSet plotSet(*realVar);
  clone->recursiveRedirectServers(plotSet);

  // normalize ourself to any previous contents in the frame
  if(frame->getFitRangeNorm() > 0) {
    scaleFactor*= frame->getFitRangeNorm();
  }
  frame->updateNormVars(plotSet);

  // create a new curve of our function using the clone to do the evaluations
  RooCurve* curve= new RooCurve(*clone,*realVar,scaleFactor,frame->getNormVars());

  // add this new curve to the specified plot frame
  frame->addPlotable(curve, drawOptions);

  // cleanup 
  delete cloneList;

  return frame;
}



RooAbsFunc *RooAbsReal::bindVars(const RooArgSet &vars, const RooArgSet* nset) const {
  // Create an interface adaptor f(vars) that binds us to the specified variables
  // (in arbitrary order). For example, calling bindVars({x1,x3}) on an object
  // F(x1,x2,x3,x4) returns an object f(x1,x3) that is evaluated using the
  // current values of x2 and x4. The caller takes ownership of the returned adaptor.

  RooAbsFunc *binding= new RooRealBinding(*this,vars,nset);
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
  } else {
    _value = other->_value ;
  }
  setValueDirty() ;
}


void RooAbsReal::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree
  TString cleanName(GetName()) ;
  cleanName.ReplaceAll("/","D") ;
  cleanName.ReplaceAll("-","M") ;
  cleanName.ReplaceAll("+","P") ;
  cleanName.ReplaceAll("*","X") ;
  cleanName.ReplaceAll("[","L") ;
  cleanName.ReplaceAll("]","R") ;
  cleanName.ReplaceAll("(","L") ;
  cleanName.ReplaceAll(")","R") ;

  // First determine if branch is taken
  TBranch* branch = t.GetBranch(cleanName) ;
  if (branch) { 
    
    // Determine if existing branch is Float_t or Double_t
    TString typeName(((TLeaf*)branch->GetListOfLeaves()->At(0))->GetTypeName()) ;
    if (!typeName.CompareTo("Float_t")) {
      cout << "RooAbsReal::attachToTree(" << GetName() << ") TTree branch " << GetName() 
	   << " will be converted to double precision" << endl ;
      setAttribute("FLOAT_TREE_BRANCH",kTRUE) ;
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

RooAbsArg *RooAbsReal::createFundamental() const {
  // Create a RooRealVar fundamental object with our properties. The new
  // object will be created without any fit limits.

  RooRealVar *fund= new RooRealVar(GetName(),GetTitle(),_value,getUnit());
  fund->removeFitRange();
  fund->setPlotMin(getPlotMin());
  fund->setPlotMax(getPlotMax());
  fund->setPlotLabel(getPlotLabel());
  fund->setPlotBins(getPlotBins());
  fund->setAttribute("fundamentalCopy");
  return fund;
}

RooPlot *RooAbsReal::frame() const {
  // Create a new RooPlot on the heap with a drawing frame initialized for this
  // object, but no plot contents. Use x.frame() as the first argument to a
  // y.plotOn(...) method, for example. The caller is responsible for deleting
  // the returned object.

  return new RooPlot(*this);
}


Int_t RooAbsReal::getPlotBin() const 
{
  return Int_t((getVal() - getPlotMin())/ _plotBinW) ;
}


RooAbsBinIter* RooAbsReal::createPlotBinIterator() const 
{
  return new RooRealFixedBinIter(*this) ;
}


void RooAbsReal::calcBinWidth() 
{
  _plotBinW = (getPlotMax() - getPlotMin()) / numPlotBins() ;
}


Double_t RooAbsReal::plotBinCenter(Int_t i) const 
{
  if (i<0 || i>=numPlotBins()) {
    cout << "RooAbsReal::plotBinCenter(" << GetName() << ") ERROR: bin index " << i 
	 << " is out of range (0," << getPlotBins()-1 << ")" << endl ;
    return 0 ;
  }

  return getPlotMin() + (i + 0.5)*_plotBinW ;
}


Double_t RooAbsReal::plotBinLow(Int_t i) const 
{
  if (i<0 || i>=numPlotBins()) {
    cout << "RooAbsReal::plotBinLow(" << GetName() << ") ERROR: bin index " << i 
	 << " is out of range (0," << getPlotBins()-1 << ")" << endl ;
    return 0 ;
  }

  return getPlotMin() + i*_plotBinW ;
}


Double_t RooAbsReal::plotBinHigh(Int_t i) const 
{
  if (i<0 || i>=numPlotBins()) {
    cout << "RooAbsReal::plotBinHigh(" << GetName() << ") ERROR: bin index " << i 
	 << " is out of range (0," << getPlotBins()-1 << ")" << endl ;
    return 0 ;
  }

  return getPlotMin() + (i + 1)*_plotBinW ;
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
  delete iterator;
  if(isMatched) matchedArgs.add(matched);
  return isMatched;
}


