/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsReal.cc,v 1.20 2001/05/16 07:41:07 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
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
#include "RooFitCore/RooPlot.hh"
#include "RooFitCore/RooCurve.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRealFunc1D.hh"
#include "RooFitCore/RooArgProxy.hh"

#include <iostream.h>

#include "TObjString.h"
#include "TTree.h"
#include "TH1.h"

ClassImp(RooAbsReal) 
;


RooAbsReal::RooAbsReal(const char *name, const char *title, const char *unit) : 
  RooAbsArg(name,title), _unit(unit), _plotBins(100), _value(0), _plotMin(0), _plotMax(0)
{
  // Constructor
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}

RooAbsReal::RooAbsReal(const char *name, const char *title, Double_t minVal,
		       Double_t maxVal, const char *unit) :
  RooAbsArg(name,title), _unit(unit), _plotBins(100), _value(0), _plotMin(minVal), _plotMax(maxVal)
{
  // Constructor with plot range
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}


RooAbsReal::RooAbsReal(const RooAbsReal& other, const char* name) : 
  RooAbsArg(other,name), _unit(other._unit), _plotBins(other._plotBins), 
  _plotMin(other._plotMin), _plotMax(other._plotMax), _value(other._value)
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


Double_t RooAbsReal::getVal(const RooDataSet* dset) const
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isValueDirty() || isShapeDirty()) {
    _value = traceEval() ;
    setValueDirty(kFALSE) ;
    setShapeDirty(kFALSE) ;
  } 
  
  return _value ;
}


Double_t RooAbsReal::traceEval() const
{
  // Calculate current value of object, with error tracing wrapper
  Double_t value = evaluate() ;
  
  //Standard tracing code goes here
  if (!isValid(value)) {
    cout << "RooAbsReal::traceEval(" << GetName() 
	 << "): validation failed: " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
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
  // Check if current value is valid
  return isValid(getVal()) ;
}


Bool_t RooAbsReal::isValid(Double_t value, Bool_t printError) const 
{
  // Check if given value is valid
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

RooPlot *RooAbsReal::plotOn(RooPlot* frame, Option_t* drawOptions) const {
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
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: cannot plot derived variable \"" << var->GetName() << "\"" << endl;
    return 0;
  }

  // check that we actually depend on the plot variable
  if(!this->dependsOn(*realVar)) {
    cout << GetName() << "::plotOn: variable is not a dependent: " << realVar->GetName() << endl;
    return 0;
  }

  // deep-clone ourselves so that the plotting process will not disturb
  // our original expression tree
  RooArgSet *cloneList = RooArgSet(*this).snapshot() ;
  RooAbsReal *clone= (RooAbsReal*) cloneList->find(GetName()) ;

  // redirect our clone to use the plot variable !!! WVE Check!
  RooArgSet args(*realVar);
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
  // Create a 1-dimensional function object from the RooAbsReal
  return RooRealFunc1D(*this,var);
}


void RooAbsReal::copyCache(const RooAbsArg* source) 
{
  // Copy the cached value of another RooAbsArg to our cache

  // Warning: This function copies the cached values of source,
  //          it is the callers responsibility to make sure the cache is clean
  RooAbsReal* other = dynamic_cast<RooAbsReal*>(const_cast<RooAbsArg*>(source)) ;
  assert(other) ;

  _value = other->_value ;
  setValueDirty(kTRUE) ;
}


void RooAbsReal::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  if (t.GetBranch(GetName())) {
    t.SetBranchAddress(GetName(),&_value) ;
//     cout << "RooAbsReal::attachToTree(" << GetName() << "): branch already exists in tree " 
// 	 << (void*)&t << ", changing address" << endl ;
  } else {
    TString format(GetName());
    format.Append("/D");
    t.Branch(GetName(), &_value, (const Text_t*)format, bufSize);
//     cout << "RooAbsReal::attachToTree(" << GetName() << "): creating new branch in tree" 
// 	 << (void*)&t << endl ;
  }
}

void RooAbsReal::postTreeLoadHook() 
{
  // Hook function that allows post-processing after a cache write from a TTRee (dummy)

//   if (isDerived())
//     cout << "RooAbsReal::postTreeLoadHook(" << GetName() 
// 	 << "): loaded cache, value " << _value << endl ;
}
 
RooPlot *RooAbsReal::frame() const {
  // Create a new RooPlot on the heap with a drawing frame initialized for this
  // object, but no plot contents. Use x.frame() as the first argument to a
  // y.plotOn(...) method, for example. The caller is responsible for deleting
  // the returned object.

  return new RooPlot(*this);
}
