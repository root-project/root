/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealVar.cc,v 1.26 2001/09/27 18:22:30 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooRealVar represents a fundamental (non-derived) real valued object
// 
// This class also holds an error and a fit range associated with the real value


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
		       Double_t value, const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(0), _fitBins(100)
{
  // Constructor with value and unit
  _value = value ;
  removeFitRange();
  setConstant(kTRUE) ;
}  

RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t minValue, Double_t maxValue,
		       const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(0), _fitBins(100)
{
  // Constructor with range and unit. Value is set to middle of range

  _value= 0.5*(minValue + maxValue);

  setPlotRange(minValue,maxValue) ;
  setFitRange(minValue,maxValue) ;
}  

RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t value, Double_t minValue, Double_t maxValue,
		       const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(0), _fitBins(100)
{
  // Constructor with value, range and unit
  _value = value ;
  setPlotRange(minValue,maxValue) ;
  setFitRange(minValue,maxValue) ;
}  

RooRealVar::RooRealVar(const RooRealVar& other, const char* name) :
  RooAbsRealLValue(other,name), 
  _error(other._error),
  _fitMin(other._fitMin),
  _fitMax(other._fitMax),
  _fitBins(other._fitBins)
{
  // Copy Constructor
}


RooRealVar::~RooRealVar() 
{
  // Destructor
}

void RooRealVar::setVal(Double_t value) {
  // Set current value
  Double_t clipValue ;
  inFitRange(value,&clipValue) ;

  setValueDirty() ;
  _value = clipValue;
}

void RooRealVar::setFitMin(Double_t value) 
{
  // Check if new limit is consistent
  if (value >= _fitMax) {
    cout << "RooRealVar::setFitMin(" << GetName() 
	 << "): Proposed new fit min. larger than max., setting min. to max." << endl ;
    _fitMin = _fitMax ;
  } else {
    _fitMin = value ;
  }

  // Clip current value in window if it fell out
  Double_t clipValue ;
  if (!inFitRange(_value,&clipValue)) {
    setVal(clipValue) ;
  }

  setShapeDirty() ;
}

void RooRealVar::setFitMax(Double_t value)
{
  // Check if new limit is consistent
  if (value < _fitMin) {
    cout << "RooRealVar::setFitMax(" << GetName() 
	 << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    _fitMax = _fitMin ;
  } else {
    _fitMax = value ;
  }

  // Clip current value in window if it fell out
  Double_t clipValue ;
  if (!inFitRange(_value,&clipValue)) {
    setVal(clipValue) ;
  }

  setShapeDirty() ;
}

void RooRealVar::setFitRange(Double_t min, Double_t max) {
  // Check if new limit is consistent
  if (min>max) {
    cout << "RooRealVar::setFitRange(" << GetName() 
	 << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    _fitMin = min ;
    _fitMax = min ;
  } else {
    _fitMin = min ;
    _fitMax = max ;
  }

  setShapeDirty() ;  
}


void RooRealVar::copyCache(const RooAbsArg* source) 
{
  // Overloaded from RooAbsRealLValue to skip back-prop

  // Copy cache of another RooAbsArg to our cache
  RooAbsReal::copyCache(source) ;
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
    if (isValidReal(value,verbose)) {
      setVal(value) ;
      return kFALSE ;
    } else {
      return kTRUE ;
    }

  } else {
    // Extended mode: Read multiple tokens on a single line   
    Bool_t haveValue(kFALSE) ;
    while(1) {      
      if (parser.atEOL()) break ;
      token=parser.readToken() ;

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
	Double_t plotMin(0), plotMax(0) ;
        Int_t plotBins(0) ;
	if (parser.expectToken("(",kTRUE) ||
	    parser.readDouble(plotMin,kTRUE) ||
	    parser.expectToken("-",kTRUE) ||
	    parser.readDouble(plotMax,kTRUE) ||
            parser.expectToken(":",kTRUE) ||
            parser.readInteger(plotBins,kTRUE) || 
	    parser.expectToken(")",kTRUE)) break ;
	setPlotRange(plotMin,plotMax) ;

      } else if (!token.CompareTo("F")) {

	// Next tokens are fit limits
	Double_t fitMin, fitMax ;
	Int_t fitBins ;
	if (parser.expectToken("(",kTRUE) ||
	    parser.readDouble(fitMin,kTRUE) ||
	    parser.expectToken("-",kTRUE) ||
	    parser.readDouble(fitMax,kTRUE) ||
	    parser.expectToken(":",kTRUE) ||
	    parser.readInteger(fitBins,kTRUE) ||
	    parser.expectToken(")",kTRUE)) break ;
	setFitRange(fitMin,fitMax) ;
	setFitBins(fitBins) ;
	setConstant(kFALSE) ;
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

void RooRealVar::writeToStream(ostream& os, Bool_t compact) const
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

    // Append fit limits if not +Inf:-Inf
    os << "F(" ;
    if(hasFitMin()) {
      os << getFitMin();
    }
    else {
      os << "-INF";
    }
    if(hasFitMax()) {
      os << " - " << getFitMax() << ") ";
    }
    else {
      os << " - +INF";
    }
    os << " : " << getFitBins() << ") " ;

    // Add comment with unit, if unit exists
    if (!_unit.IsNull())
      os << "// [" << getUnit() << "]" ;
  }
}


void RooRealVar::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this object to the specified stream. In addition to the info
  // from RooAbsRealLValue::printToStream() we add:
  //
  //   Verbose : fit range and error

  RooAbsRealLValue::printToStream(os,opt,indent);
  if(opt >= Verbose) {
    os << indent << "--- RooRealVar ---" << endl;
    TString unit(_unit);
    if(!unit.IsNull()) unit.Prepend(' ');
    if(opt >= Verbose) {
      os << indent << "  Error = " << getError() << unit << endl;
    }
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

