/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsReal.cc,v 1.12 2001/05/07 06:26:13 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include "RooFitCore/RooAbsReal.hh"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooCurve.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRealFunc1D.hh"

#include <iostream.h>

#include "TObjString.h"
#include "TH1.h"

ClassImp(RooAbsReal) 
;


RooAbsReal::RooAbsReal(const char *name, const char *title, const char *unit) : 
  RooAbsArg(name,title), _unit(unit), _plotBins(100), _value(0), _plotMin(0), _plotMax(0)
{
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}

RooAbsReal::RooAbsReal(const char *name, const char *title, Double_t minVal,
		       Double_t maxVal, const char *unit) :
  RooAbsArg(name,title), _unit(unit), _plotBins(100), _value(0), _plotMin(minVal), _plotMax(maxVal)
{
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}


RooAbsReal::RooAbsReal(const RooAbsReal& other, const char* name) : 
  RooAbsArg(other,name), _unit(other._unit), _plotBins(other._plotBins), 
  _plotMin(other._plotMin), _plotMax(other._plotMax), _value(other._value)
{
}


RooAbsReal::~RooAbsReal()
{
}


RooAbsReal& RooAbsReal::operator=(const RooAbsReal& other)
{
  RooAbsArg::operator=(other) ;

  _value    = other._value ;
  setValueDirty(kTRUE) ;

  return *this ;
}


RooAbsArg& RooAbsReal::operator=(const RooAbsArg& aother)
{
  return operator=((const RooAbsReal&)aother) ;
}



Bool_t RooAbsReal::operator==(Double_t value) const
{
  return (getVal()==value) ;
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
}


void RooAbsReal::setPlotBins(Int_t value) {
  // Set number of histogram bins 
  _plotBins = value ;  
}


Bool_t RooAbsReal::inPlotRange(Double_t value) const {
  // Check if given value is in the min-max range for this object
  return (value >= _plotMin && value <= _plotMax) ? kTRUE : kFALSE;
}



Bool_t RooAbsReal::isValid() const {
  return isValid(getVal()) ;
}


Bool_t RooAbsReal::isValid(Double_t value) const {
  return kTRUE ;
}



TH1F *RooAbsReal::createHistogram(const char *label, const char *axis, Int_t bins) {
  // Create a 1D-histogram with appropriate scale and labels for this variable.

  return createHistogram(label, axis, _plotMin, _plotMax, bins);
}

TH1F *RooAbsReal::createHistogram(const char *label, const char *axis,
				  Double_t lo, Double_t hi, Int_t bins) {
  // Create a 1D-histogram with appropriate scale and labels for this variable.

  TString histName(label);
  if(!histName.IsNull()) histName.Append("_");
  histName.Append(GetName());

  // use the default binning, if no override is specified
  if(bins <= 0) bins= getPlotBins();
  TH1F* histogram= new TH1F(histName.Data(), fTitle, bins, lo, hi);
  if(!histogram) {
    cout << fName << "::createHistogram: unable to create a new histogram" << endl;
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
  if(strlen(axis)) {
    TString yTitle(axis);
    Double_t delta= (_plotMax-_plotMin)/bins;
    yTitle.Append(Form(" %g",delta));
    if(strlen(getUnit())) {
      yTitle.Append(" ");
      yTitle.Append(getUnit());
    }
    histogram->SetYTitle(yTitle.Data());
  }
  return histogram;
}

RooPlot *RooAbsReal::plot(const RooRealVar& var, Option_t* drawOptions) const {
  // Create an empty frame for the specified variable and add to it a curve
  // calculated for the specified variable.

  return plot(new RooPlot(var), drawOptions);
}

RooPlot *RooAbsReal::plot(RooPlot* frame, Option_t* drawOptions) const {
  // check that we are passed a valid plot frame to use
  if(0 == frame) {
    cout << ClassName() << "::" << GetName() << ":plot: frame is null" << endl;
    return 0;
  }
  // check that this frame knows what variable to plot
  RooAbsReal *var= frame->getPlotVar();
  if(0 == var) {
    cout << ClassName() << "::" << GetName()
	 << ":plot: frame does not specify a plot variable" << endl;
    return 0;
  }
  // check that the plot variable is not derived
  RooRealVar* realVar= dynamic_cast<RooRealVar*>(var);
  if(0 == realVar) {
    cout << ClassName() << "::" << GetName()
	 << ":plot: cannot plot derived variable \"" << var->GetName() << "\"" << endl;
    return 0;
  }
  // check that we actually depend on the plot variable
  if(!this->dependsOn(*realVar)) {
    cout << GetName() << "::plot: variable is not a dependent: " << realVar->GetName() << endl;
    return 0;
  }

  // deep-clone ourselves so that the plotting process will not disturb
  // our original expression tree
  RooArgSet *cloneList = RooArgSet("",*this).snapshot() ;
  RooAbsReal *clone= (RooAbsReal*) cloneList->find(GetName()) ;

  // redirect our clone to use the plot variable !!! WVE Check!
  RooArgSet args("args",*realVar);
  clone->recursiveRedirectServers(args);

  // create a new curve of our function using the clone to do the evaluations
  RooCurve *curve= new RooCurve(*clone,*realVar);

  // add a copy of the temporary curve to the specified plot frame
  frame->addPlotable(curve, drawOptions);

  // cleanup
  delete cloneList;

  return frame;
}

RooRealFunc1D RooAbsReal::operator()(RooRealVar &var) const {
  return RooRealFunc1D(*this,var);
}
