/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsReal.cc,v 1.30 2001/08/03 18:11:33 verkerke Exp $
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
#include "RooFitCore/RooArgProxy.hh"
#include "RooFitCore/RooFormulaVar.hh"

#include "RooFitCore/RooRealBinding.hh"

#include <iostream.h>

#include "TObjString.h"
#include "TTree.h"
#include "TH1.h"

ClassImp(RooAbsReal) 
;


RooAbsReal::RooAbsReal(const char *name, const char *title, const char *unit) : 
  RooAbsArg(name,title), _unit(unit), _plotBins(100), _value(0), 
  _plotMin(0), _plotMax(0)
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
  setValueDirty() ;
  setShapeDirty() ;
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
  Double_t value = evaluate(nset) ;
  
  //Standard tracing code goes here
  if (!isValidReal(value)) {
    cout << "RooAbsReal::traceEval(" << GetName() 
	 << "): validation failed: " << value << endl ;
  }

  //Call optional subclass tracing code
  traceEvalHook(value) ;

  return value ;
}


Int_t RooAbsReal::getAnalyticalIntegral(RooArgSet& allDeps, RooArgSet& analDeps) const
{
  // By default we do supply any analytical integrals
  return 0 ;
}


Double_t RooAbsReal::analyticalIntegral(Int_t code) const
{
  // By default no analytical integrals are implemented
  return getVal() ;
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
  return isValidReal(getVal()) ;
}


Bool_t RooAbsReal::isValidReal(Double_t value, Bool_t printError) const 
{
  // Check if given value is valid
  return kTRUE ;
}



TH1F *RooAbsReal::createHistogram(const char *label, const char *axis, Int_t bins) const {
  // Create a 1D-histogram with appropriate scale and labels for this variable.
  // This method uses the default plot range which can be changed using the
  // setPlotMin(),setPlotMax() methods. Uses the default binning (setPlotBins())
  // unless you specify your own binning.

  return createHistogram(label, axis, _plotMin, _plotMax, bins > 0 ? bins : getPlotBins());
}

TH1F *RooAbsReal::createHistogram(const char *label, const char *axis,
				  Double_t lo, Double_t hi, Int_t bins) const {
  // Create a 1D-histogram with appropriate scale and labels for this variable.
  // Binning must be specified with this method since the default binning is associated
  // with the default plot ranges, but you have asked for a non-default range.

  TString histName(label);
  if(!histName.IsNull()) histName.Append("_");
  histName.Append(GetName());

  // create the histogram
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
    yTitle.Append(Form(" / %g",delta));
    if(strlen(getUnit())) {
      yTitle.Append(" ");
      yTitle.Append(getUnit());
    }
    histogram->SetYTitle(yTitle.Data());
  }
  return histogram;
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
    cout << ClassName() << "::" << GetName()
	 << ":plotOn: cannot plot derived variable \"" << var->GetName() << "\"" << endl;
    return 0;
  }

  // check if we actually depend on the plot variable
  if(!this->dependsOn(*realVar)) {
    cout << GetName() << "::plotOn:WARNING: variable is not an explicit dependent: "
	 << realVar->GetName() << endl;
  }

  // deep-clone ourselves so that the plotting process will not disturb
  // our original expression tree
  RooArgSet *cloneList = RooArgSet(*this).snapshot() ;
  RooAbsReal *clone= (RooAbsReal*) cloneList->find(GetName()) ;

  // redirect our clone to use the plot variable
  RooArgSet args(*realVar);
  clone->recursiveRedirectServers(args);

  // normalize ourself to any previous contents in the frame
  if(frame->getFitRangeNorm() > 0) scaleFactor*= frame->getFitRangeNorm();
  frame->updateNormVars(RooArgSet(*realVar));

  // create a new curve of our function using the clone to do the evaluations
  RooCurve* curve= new RooCurve(*clone,*realVar,scaleFactor,frame->getNormVars());

  // add this new curve to the specified plot frame
  frame->addPlotable(curve, drawOptions);

  // cleanup 
  delete cloneList;

  return frame;
}

RooAbsFunc *RooAbsReal::bindVars(const RooArgSet &vars) const {
  // Create an interface adaptor f(vars) that binds us to the specified variables
  // (in arbitrary order). For example, calling bindVars({x1,x3}) on an object
  // F(x1,x2,x3,x4) returns an object f(x1,x3) that is evaluated using the
  // current values of x2 and x4. The caller takes ownership of the returned adaptor.

  RooAbsFunc *binding= new RooRealBinding(*this,vars);
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

  _value = other->_value ;
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
  if (t.GetBranch(cleanName)) {
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
