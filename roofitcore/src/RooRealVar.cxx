/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealVar.cc,v 1.2 2001/03/17 03:47:39 verkerke Exp $
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
#include <ctype.h>
#include "TObjString.h"
#include "TTree.h"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooRealVar)


RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t value, const char *unit, RooBlindBase* blinder) :
  RooAbsReal(name, title, 0, 0, unit), _error(0), _blinder(blinder),
  _integMin(-1e10), _integMax(1e10)
{
  _value = value ;
  setConstant(kTRUE) ;
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}  

RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t minValue, Double_t maxValue,
		       const char *unit, RooBlindBase* blinder) :
  RooAbsReal(name, title, minValue, maxValue, unit), _blinder(blinder),
  _integMin(minValue), _integMax(maxValue)

{
  _value= 0.5*(minValue + maxValue);
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}  

RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t value, Double_t minValue, Double_t maxValue,
		       const char *unit, RooBlindBase* blinder) :
  RooAbsReal(name, title, minValue, maxValue, unit), _error(0), _blinder(blinder),
  _integMin(minValue), _integMax(maxValue)
{
  //   if (_blinder) _blinder->redoBlind() ;
  _value = value ;
  setValueDirty(kTRUE) ;
  setShapeDirty(kTRUE) ;
}  

RooRealVar::RooRealVar(const RooRealVar& other) :
  RooAbsReal(other), 
  _error(other._error),
  _blinder(other._blinder),
  _integMin(other._integMin),
  _integMax(other._integMax)
{
  setConstant(other.isConstant()) ;
  setProjected(other.isProjected()) ;
}

RooRealVar::~RooRealVar() 
{
}

RooRealVar::operator Double_t&() {
  return _value;
}

RooRealVar::operator Double_t() {
  return this->getVal();
}


void RooRealVar::setVal(Double_t value) {

  // Set current value
  Double_t clipValue ;
  inIntegRange(value,&clipValue) ;

  setValueDirty(kTRUE) ;
  _value = clipValue;
}


void RooRealVar::setIntegMin(Double_t value) 
{
  // Check if new limit is consistent
  if (_integMin>_integMax) {
    cout << "RooRealVar::setIntegMin(" << GetName() 
	 << "): Proposed new integration min. larger than max., setting min. to max." << endl ;
    _integMin = _integMax ;
  } else {
    _integMin = value ;
  }

  // Clip current value in window if it fell out
  Double_t clipValue ;
  if (!inIntegRange(_value,&clipValue)) {
    setVal(clipValue) ;
  }

  setShapeDirty(kTRUE) ;
}


void RooRealVar::setIntegMax(Double_t value)
{
  // Check if new limit is consistent
  if (_integMax<_integMin) {
    cout << "RooRealVar::setIntegMax(" << GetName() 
	 << "): Proposed new integration max. smaller than min., setting max. to min." << endl ;
    _integMax = _integMax ;
  } else {
    _integMax = value ;
  }

  // Clip current value in window if it fell out
  Double_t clipValue ;
  if (!inIntegRange(_value,&clipValue)) {
    setVal(clipValue) ;
  }

  setShapeDirty(kTRUE) ;
}


void RooRealVar::setIntegRange(Double_t min, Double_t max) {
  // Check if new limit is consistent
  if (min>max) {
    cout << "RooRealVar::setIntegMinMax(" << GetName() 
	 << "): Proposed new integration max. smaller than min., setting max. to min." << endl ;
    _integMin = min ;
    _integMax = min ;
  } else {
    _integMin = min ;
    _integMax = max ;
  }

  setShapeDirty(kTRUE) ;  
}



Double_t RooRealVar::operator=(Double_t newValue) 
{
  // Clip 
  inIntegRange(newValue,&_value) ;

  setValueDirty(kTRUE) ;
  return _value;
}



Bool_t RooRealVar::inIntegRange(Double_t value, Double_t* clippedValPtr) const
{
  // Check which limit we exceeded and truncate. Print a warning message
  // unless we are very close to the boundary.  
  
  Double_t range = _integMax - _integMin ;
  Double_t clippedValue(value);
  Bool_t inRange(kTRUE) ;

  if (hasIntegLimits()) {
    if(value > _integMax) {
      if(value - _integMax > 1e-6*range) {
	if (clippedValPtr)
	  cout << "RooRealVar::inIntegRange(" << GetName() << "): value " << value
	       << " rounded down to max limit " << _integMax << endl;
      }
      clippedValue = _integMax;
      inRange = kFALSE ;
    }
    else if(value < _integMin) {
      if(_integMin - value > 1e-6*range) {
	if (clippedValPtr)
	  cout << "RooRealVar::inIntegRange(" << GetName() << "): value " << value
	       << " rounded up to min limit " << _integMin << endl;
      }
      clippedValue = _integMin;
      inRange = kFALSE ;
    } 
  }

  if (clippedValPtr) *clippedValPtr=clippedValue ;
  return inRange ;
}




Bool_t RooRealVar::isValid() 
{
  return isValid(getVal()) ;
}


Bool_t RooRealVar::isValid(Double_t value) {
  return inIntegRange(value) ;
}



void RooRealVar::attachToTree(TTree& t, Int_t bufSize)
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  if (t.GetBranch(GetName())) {
    //cout << "RooRealVar::attachToTree(" << GetName() << "): branch in tree " << t.GetName() << " already exists" << endl ;
    t.SetBranchAddress(GetName(),&_value) ;
  } else {    
    TString format(GetName());
    format.Append("/D");
    t.Branch(GetName(), &_value, (const Text_t*)format, bufSize);
  }
}


Bool_t RooRealVar::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
  TString token,errorPrefix("RooRealVar::readFromStream(") ;
  errorPrefix.Append(GetName()) ;
  errorPrefix.Append(")") ;
  RooStreamParser parser(is,errorPrefix) ;
  Double_t value(0) ;

  if (compact) {
    // Compact mode: Read single token
    if (parser.readDouble(value,verbose)) return kTRUE ;

    if (inIntegRange(value)) {
      setVal(value) ;
      return kFALSE ;  
    } else {
      if (verbose) {
	cout << "RooRealVar::readFromStream(" << GetName() 
	     << "): value out of range: " << value << endl ;
      }
      return kTRUE;
    }

  } else {
    // Extended mode: Read multiple tokens on a single line   
    Bool_t haveValue(kFALSE) ;
    while(1) {
      token=parser.readToken() ;
      if (token.IsNull()) break ;

      if (!token.CompareTo("+/-")) {

	// Next token is error
	Double_t error ;
	if (parser.readDouble(error)) break ;
	setError(error) ;

      } else if (!token.CompareTo("C")) {

	// Set constant
	setConstant(kTRUE) ;

      } else if (!token.CompareTo("P")) {

	// Next tokens are plot limits
	Double_t plotMin, plotMax ;
        Int_t plotBins ;
	if (parser.expectToken("(",kTRUE) ||
	    parser.readDouble(plotMin,kTRUE) ||
	    parser.expectToken("-",kTRUE) ||
	    parser.readDouble(plotMax,kTRUE) ||
            parser.expectToken(":",kTRUE) ||
            parser.readInteger(plotBins,kTRUE) || 
	    parser.expectToken(")",kTRUE)) break ;
	setPlotRange(plotMin,plotMax) ;

      } else if (!token.CompareTo("I")) {

	// Next tokens are integration limits
	Double_t integMin, integMax ;
	if (parser.expectToken("(",kTRUE) ||
	    parser.readDouble(integMin,kTRUE) ||
	    parser.expectToken("-",kTRUE) ||
	    parser.readDouble(integMax,kTRUE) ||
	    parser.expectToken(")",kTRUE)) break ;
	setIntegRange(integMin,integMax) ;
      } else {
	// Token is value
	if (parser.convertToDouble(token,value)) { parser.zapToEnd() ; break ; }
	haveValue = kTRUE ;
	// Defer value assignment to end
      }
    }    

    if (haveValue) setVal(value) ;
    return kFALSE ;
  }
}


void RooRealVar::writeToStream(ostream& os, Bool_t compact)
{
  // Write object contents to given stream

  if (compact) {
    // Write value only
    os << getVal() ;
  } else {
    // Write value
    os << getVal() << " " ;
  
    // Append error if non-zero 
    Double_t err = getError() ;
    if (err!=0) {
      os << "+/- " << err << " " ;
    }
    // Append limits if not constants
    if (isConstant()) {
      os << "C " ;
    }      
    // Append plot limits
    os << "P(" << getPlotMin() << " - " << getPlotMax() << " : " << getPlotBins() << ") " ;      
    // Append integration limits if not +Inf:-Inf
    if (hasIntegLimits()) {
      os << "I(" << getIntegMin() << " - " << getIntegMax() << ") " ;      
    }
    // Add comment with unit, if unit exists
    if (!_unit.IsNull())
      os << "// [" << getUnit() << "]" ;
  }
}



RooAbsArg&
RooRealVar::operator=(RooAbsArg& aorig)
{
  // Assignment operator for RooRealVar
  RooAbsReal::operator=(aorig) ;

  RooRealVar& orig = (RooRealVar&)aorig ;
  _error = orig._error ;
  _blinder = orig._blinder ;
  _integMin = orig._integMin ;
  _integMax = orig._integMax ;

  return (*this) ;
}

void RooRealVar::printToStream(ostream& os, PrintOption opt) {
  switch(opt) {
  case Verbose:
    os << fName << " = " << getVal() << " +/- " << _error;    
    if(!_unit.IsNull()) os << ' ' << _unit;
    printAttribList(os) ;
    os << endl;
    break ;
    
  case Shape:
    os << fName << ": " << fTitle;
    if(isConstant()) {
      os << ", fixed at " << getVal();
    }
    else {
      os << ", range is (" << _integMin << "," << _integMax << ")";
    }
    if(!_unit.IsNull()) os << ' ' << _unit;
    printAttribList(os) ;
    os << endl;
    break ;
    
  case Standard:
    os << "RooRealVar: " << GetName() << " = " << getVal();
    if (_blinder) os << " (blind)" ; 
    if(!_unit.IsNull()) os << ' ' << _unit;
    os << " : " << GetTitle() ;
    if(!isConstant() && hasIntegLimits())
      os << " (" << _integMin << ',' << _integMax << ')';
    else if (isConstant()) 
      os << " Constant" ;
    os << endl ;	
    break ;
  }
}


TString *RooRealVar::format(Int_t sigDigits, const char *options) {
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
    text->Append(getPlotLabel());
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

Double_t RooRealVar::chopAt(Double_t what, Int_t where) {
  // What does this do?
  Double_t scale= pow(10.0,where);
  Int_t trunc= (Int_t)floor(what/scale + 0.5);
  return (Double_t)trunc*scale;
}

