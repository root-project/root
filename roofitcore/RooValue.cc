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


#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "TObjString.h"
#include "TTree.h"

#include "RooFitCore/RooValue.hh"

ClassImp(RooValue)


RooValue::RooValue(const char *name, const char *title,
		       Double_t value, const char *unit, RooBlindBase* blinder) :
  RooDerivedValue(name, title, 0, 0, unit), _error(0), _blinder(blinder)
{
  _value = value ;
  setConstant(kTRUE) ;
  setDirty(kTRUE) ;
}  

RooValue::RooValue(const char *name, const char *title,
		       Double_t minValue, Double_t maxValue,
		       const char *unit, RooBlindBase* blinder) :
  RooDerivedValue(name, title, minValue, maxValue, unit), _blinder(blinder)
{
  _value= 0.5*(minValue + maxValue);
  updateLimits();
  setDirty(kTRUE) ;
}  

RooValue::RooValue(const char *name, const char *title,
		       Double_t value, Double_t minValue, Double_t maxValue,
		       const char *unit, RooBlindBase* blinder) :
  RooDerivedValue(name, title, minValue, maxValue, unit), _error(0), _blinder(blinder)
{
//   if (_blinder) _blinder->redoBlind() ;
  _value = value ;
  updateLimits();
  setDirty(kTRUE) ;
}  

RooValue::RooValue(const RooValue& other) :
  RooDerivedValue(other), 
  _error(other._error),
  _blinder(other._blinder)
{
  setConstant(other.isConstant()) ;
  setLimits(other.useLimits()) ;
  setProjected(other.isProjected()) ;
}

RooValue::~RooValue() 
{
}

RooValue::operator Double_t&() {
  return _value;
}

RooValue::operator Double_t() {
  return this->GetVar();
}

void RooValue::Set(Double_t value, Double_t minValue, Double_t maxValue) {
  // Set current, minimum and maximum value
  _value= value;
  _minValue= minValue;
  _maxValue= maxValue;
  setConstant(kFALSE) ;

  setDirty(kTRUE) ;
  updateLimits();
}

void RooValue::SetVar(Double_t value) {
  // Set current value
  setDirty(kTRUE) ;
  _value = value;
}


Double_t RooValue::operator=(Double_t newValue) {
  // Assignment operator for Double_t
  Double_t range = _maxValue - _minValue ;
  if(isConstant ()|| (newValue >= _minValue && newValue <= _maxValue)) {
    _value= newValue;
    if(isConstant() && (newValue < _minValue || newValue > _maxValue)) {
      // force limits to new value if necessary
      _minValue= _maxValue= _value;
    }
  }
  // Check which limit we exceeded and truncate. Print a warning message
  // unless we are very close to the boundary.
  else if(newValue > _maxValue) {
    if(newValue - _maxValue > 1e-6*range) {
      cout << GetName() << ": value " << newValue
      << " rounded down to max limit " << _maxValue << endl;
    }
    _value= _maxValue;
  }
  else if(newValue < _minValue) {
    if(_minValue - newValue > 1e-6*range) {
      cout << GetName() << ": value " << newValue
      << " rounded up to min limit " << _minValue << endl;
    }
    _value= _minValue;
  }

  setDirty(kTRUE) ;
  return _value;
}



void RooValue::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  if (t.GetBranch(GetName())) {
    //cout << "RooValue::attachToTree(" << GetName() << "): branch in tree " << t.GetName() << " already exists" << endl ;
    t.SetBranchAddress(GetName(),&_value) ;
  } else {    
    TString format(GetName());
    format.Append("/D");
    t.Branch(GetName(), &_value, (const Text_t*)format, bufSize);
  }
}


Bool_t RooValue::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream

  // compact only at the moment
  // Read single token
  TString token ;
  is >> token ;

  // Convert token to double
  char *endptr(0) ;
  Double_t value = strtod(token.Data(),&endptr) ;	  
  int nscan = endptr-((const char *)token.Data()) ;	  
  if (nscan<token.Length() && !token.IsNull()) {
    if (verbose) {
      cout << "RooValue::readFromStream(" << GetName() 
	   << "): cannot convert token \"" << token 
	   << "\" to floating point number" << endl ;
    }
    return kTRUE ;
  }

  if (inRange(value)) {
    SetVar(value) ;
    return kFALSE ;  
  } else {
    if (verbose) {
      cout << "RooValue::readFromStream(" << GetName() 
	   << "): value out of range: " << value << endl ;
    }
    return kTRUE;
  }
}



void RooValue::writeToStream(ostream& os, Bool_t compact)
{
  // Write object contents to given stream

  // compact only at the moment
  os << GetVar() ;
}



RooAbsArg&
RooValue::operator=(RooAbsArg& aorig)
{
  // Assignment operator for RooValue
  RooDerivedValue::operator=(aorig) ;

  RooValue& orig = (RooValue&)aorig ;
  _error = orig._error ;
  _blinder = orig._blinder ;

  return (*this) ;
}

void RooValue::PrintToStream(ostream & stream, Option_t* options) {
  // Print contents of object
  TString opts(options);
  opts.ToLower();
  Bool_t showError= opts.Contains("e");
  if(opts.Contains("e")) {
    stream << fName << " = " << GetVar() << " +/- " << _error;

    if(!_unit.IsNull()) stream << ' ' << _unit;
    printAttribList(stream) ;
    stream << endl;
  }
  else if(opts.Contains("t")) {
    stream << fName << ": " << fTitle;
    if(isConstant()) {
      stream << ", fixed at " << GetVar();
    }
    else {
      stream << ", range is (" << _minValue << "," << _maxValue << ")";
    }
    if(!_unit.IsNull()) stream << ' ' << _unit;
    printAttribList(stream) ;
    stream << endl;
  }
  else {
    stream << *this << endl;
  }
}

// Print contents of object
ostream& operator<<(ostream& os, RooValue &var) {
  os << "RooValue: " << var.GetName() << " = " << var.GetVar();
  if (var._blinder) os << " (blind)" ; 
  if(!var._unit.IsNull()) os << ' ' << var._unit;
  os << " : \"" << var.fTitle << "\"" ;
  if(!var.isConstant())
    os << " (" << var._minValue << ',' << var._maxValue << ')';

  TIterator *attribIter= var._attribList.MakeIterator();
  if (attribIter) {
    TObjString* attrib ;
    while (attrib=(TObjString*)attribIter->Next()) {
      os << " " << attrib->String() ;
    }
  }
    
  return os;
}



TString *RooValue::format(Int_t sigDigits, const char *options) {
  // Format numeric value in a variety of ways

  // parse the options string
  TString opts(options);
  opts.ToLower();
  Bool_t showName= opts.Contains("n");
  Bool_t hideValue= opts.Contains("h");
  Bool_t showError= opts.Contains("e");
  Bool_t showUnit= opts.Contains("u");
  Bool_t tlatexMode= opts.Contains("l");
  Bool_t latexMode= opts.Contains("x");
  Bool_t useErrorForPrecision=
    (showError && !isConstant()) || opts.Contains("p");
  // calculate the precision to use
  if(sigDigits < 1) sigDigits= 1;
  Double_t what= (useErrorForPrecision) ? _error : _value;
  Int_t leadingDigit= (Int_t)floor(log10(fabs(what)));
  Int_t where= leadingDigit - sigDigits + 1;
  char fmt[16];
  sprintf(fmt,"%%.%df", where < 0 ? -where : 0);
  TString *text= new TString();
  if(latexMode) text->Append("$");
  // begin the string with "<name> = " if requested
  if(showName) {
    text->Append(GetLabel());
    text->Append(" = ");
  }
  // append our value if requested
  char buffer[256];
  if(!hideValue) {
    Double_t chopped= chopAt(_value, where);
    sprintf(buffer, fmt, _value);
    text->Append(buffer);
  }
  // append our error if requested and this variable is not constant
  if(!isConstant() && showError) {
    if(tlatexMode) {
      text->Append(" #pm ");
    }
    else if(latexMode) {
      text->Append("\\pm ");
    }
    else {
      text->Append(" +/- ");
    }
    sprintf(buffer, fmt, _error);
    text->Append(buffer);
  }
  // append our units if requested
  if(!_unit.IsNull() && showUnit) {
    text->Append(' ');
    text->Append(_unit);
  }
  if(latexMode) text->Append("$");
  return text;
}

Double_t RooValue::chopAt(Double_t what, Int_t where) {
  // What does this do?
  Double_t scale= pow(10.0,where);
  Int_t trunc= (Int_t)floor(what/scale + 0.5);
  return (Double_t)trunc*scale;
}

