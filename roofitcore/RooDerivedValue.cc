/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id$
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

#include <iostream.h>
#include "TObjString.h"
#include "TH1.h"
#include "RooFitCore/RooDerivedValue.hh"
#include "RooFitCore/RooArgSet.hh"

ClassImp(RooDerivedValue) 
;


RooDerivedValue::RooDerivedValue(const char *name, const char *title, const char *unit= "") : 
  RooAbsArg(name,title), _unit(unit), _defaultHistBins(100), _value(0), _minValue(0), _maxValue(0)
{
  setDirty(kTRUE) ;
}

RooDerivedValue::RooDerivedValue(const char *name, const char *title, Double_t minVal, Double_t maxVal, const char *unit= "") :
  RooAbsArg(name,title), _unit(unit), _defaultHistBins(100), _value(0), _minValue(minVal), _maxValue(maxVal)
{
  setDirty(kTRUE) ;
}


RooDerivedValue::RooDerivedValue(const RooDerivedValue& other) : 
  RooAbsArg(other), _unit(other._unit), _defaultHistBins(other._defaultHistBins), 
  _minValue(other._minValue), _maxValue(other._maxValue), _value(other._value)
{
  setDirty(kTRUE) ;
}


RooDerivedValue::~RooDerivedValue()
{
}


RooAbsArg& RooDerivedValue::operator=(RooAbsArg& aother)
{
  RooAbsArg::operator=(aother) ;

  RooDerivedValue& other=(RooDerivedValue&)aother ;
  _value = other._value ;
  _unit = other._unit ;
  _minValue = other._minValue ;
  _maxValue = other._maxValue ;
  _defaultHistBins = other._defaultHistBins ;

  setDirty(kTRUE) ;
}


Double_t RooDerivedValue::GetVar() 
{
  // Return value of object. Calculated if dirty, otherwise cached value is returned.
  if (isDirty()) {
    setDirty(false) ;
    _value = Evaluate() ;
  } 
  
  return _value ;
}


const char *RooDerivedValue::GetLabel() const {
  // Get the label associated with the variable
  return _label.IsNull() ? fName.Data() : _label.Data();
}

void RooDerivedValue::SetLabel(const char *label) {
  // Set the label associated with this variable
  _label= label;
}



Bool_t RooDerivedValue::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  //Read object contents from stream (dummy for now)
} 

void RooDerivedValue::writeToStream(ostream& os, Bool_t compact)
{
  //Write object contents to stream (dummy for now)
}

void RooDerivedValue::PrintToStream(ostream& os, Option_t* opt= 0) 
{
  //Print object contents
  os << "RooDerivedValue: " << GetName() << " = " << GetVar();
  if(!_unit.IsNull()) os << ' ' << _unit;
  os << " : \"" << fTitle << "\"" ;

  printAttribList(os) ;
  os << endl ;
}


void RooDerivedValue::SetMin(Double_t value) {
  // Set minimum value of output associated with this object
  _minValue= value;
  updateLimits();
}

void RooDerivedValue::SetMax(Double_t value) {
  // Set maximum value of output associated with this object
  _maxValue= value;
  updateLimits();
}


void RooDerivedValue::updateLimits() {
  // Check consistency of limits and current value (needs work)
  if(_minValue > _maxValue) {
    cout << "RooValue: " << GetName() << " has min limit > max limit"
	 << endl;
  }
  if(_minValue > _value) {
    cout << "RooValue: " << GetName() << " increasing value to min limit "
	 << _minValue << endl;
    _value= _minValue;
  }
  if(_maxValue < _value) {
    cout << "RooValue: " << GetName() << " decreasing value to max limit "
	 << _maxValue << endl;
    _value= _maxValue; // WVE needs fixing
  }
}

Bool_t RooDerivedValue::inRange(Double_t value) const {
  // Check if given value is in the min-max range for this object
  return (value >= _minValue && value <= _maxValue) ? kTRUE : kFALSE;
}



TH1F *RooDerivedValue::createHistogram(const char *label, const char *axis,
				  Int_t bins) {
  // Create a 1D-histogram with appropriate scale and labels for this variable
  return createHistogram(label, axis, _minValue, _maxValue, bins);
}

TH1F *RooDerivedValue::createHistogram(const char *label, const char *axis,
				  Double_t lo, Double_t hi, Int_t bins) {
  // Create a 1D-histogram with appropriate scale and labels for this variable
  char buffer[256];
  if(label) {
    sprintf(buffer, "%s:%s", label, fName.Data());
  }
  else {
    sprintf(buffer, "%s", fName.Data());
  }
  // use the default binning, if no override is specified
  if(bins <= 0) bins= defaultHistBins();
  TH1F* histogram= new TH1F(buffer, fTitle, bins, lo, hi);
  if(!histogram) {
    cout << fName << ": unable to create new histogram" << endl;
    return 0;
  }
  const char *unit= GetUnit();
  if(*unit) {
    sprintf(buffer, "%s (%s)", fTitle.Data(), unit);
    histogram->SetXTitle(buffer);
  }
  else {
    histogram->SetXTitle((Text_t*)fTitle.Data());
  }
  if(axis) {
    Double_t delta= (_maxValue-_minValue)/bins;
    if(unit) {
      sprintf(buffer, "%s / %g %s", axis, delta, unit);
    }
    else {
      sprintf(buffer, "%s / %g", axis, delta);
    }
    histogram->SetYTitle(buffer);
  }
  return histogram;
}

