/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooRealVar.cc,v 1.39 2002/03/12 18:12:02 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [REAL] --
// RooRealVar represents a fundamental (non-derived) real valued object
// 
// This class also holds an error and a fit range associated with the real value


#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <iomanip.h>
#include "TObjString.h"
#include "TTree.h"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooErrorVar.hh"

ClassImp(RooRealVar)
;

Bool_t RooRealVar::_printScientific(kFALSE) ;
Int_t  RooRealVar::_printSigDigits(5) ;


RooRealVar::RooRealVar() 
{
  // Default constructor
}

RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t value, const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(-1), _asymErrLo(1), _asymErrHi(-1) //, _fitBins(100)
{
  // Constructor with value and unit
  _binning = new RooUniformBinning(-1,1,100) ;
  _value = value ;
  removeFitRange();
  setConstant(kTRUE) ;
}  

RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t minValue, Double_t maxValue,
		       const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(-1), _asymErrLo(1), _asymErrHi(-1) // , _fitBins(100)
{
  // Constructor with range and unit. Value is set to middle of range

  _binning = new RooUniformBinning(minValue,maxValue,100) ;

  _value= 0.5*(minValue + maxValue);

//   setPlotRange(minValue,maxValue) ;
  setFitRange(minValue,maxValue) ;
}  

RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t value, Double_t minValue, Double_t maxValue,
		       const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(-1), _asymErrLo(1), _asymErrHi(-1)  //, _fitBins(100)
{
  // Constructor with value, range and unit
  _value = value ;

  _binning = new RooUniformBinning(minValue,maxValue,100) ;
  setFitRange(minValue,maxValue) ;
}  

RooRealVar::RooRealVar(const RooRealVar& other, const char* name) :
  RooAbsRealLValue(other,name), 
  _error(other._error),
  _asymErrLo(other._asymErrLo),
  _asymErrHi(other._asymErrHi)
{
  // Copy Constructor
  _binning = other._binning->clone() ;
}


RooRealVar::~RooRealVar() 
{
  // Destructor
  delete _binning ;
}

void RooRealVar::setVal(Double_t value) {
  // Set current value
  Double_t clipValue ;
  inFitRange(value,&clipValue) ;

  setValueDirty() ;
  _value = clipValue;
}

RooErrorVar* RooRealVar::errorVar() const 
{
  TString name(GetName()), title(GetTitle()) ;
  name.Append("err") ;
  title.Append(" Error") ;

  return new RooErrorVar(name,title,*this) ;
}


void RooRealVar::setBinning(const RooAbsBinning& binning) 
{
  if (_binning) delete _binning ;
  _binning = binning.clone() ;
}


void RooRealVar::setFitMin(Double_t value) 
{
  // Set new minimum of fit range 

  // Check if new limit is consistent
  if (value >= getFitMax()) {
    cout << "RooRealVar::setFitMin(" << GetName() 
	 << "): Proposed new fit min. larger than max., setting min. to max." << endl ;
    _binning->setMin(getFitMax()) ;
  } else {
    _binning->setMin(value) ;
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
  // Set new maximum of fit range 

  // Check if new limit is consistent
  if (value < getFitMin()) {
    cout << "RooRealVar::setFitMax(" << GetName() 
	 << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    _binning->setMax(getFitMin()) ;
  } else {
    _binning->setMax(value) ;
  }

  // Clip current value in window if it fell out
  Double_t clipValue ;
  if (!inFitRange(_value,&clipValue)) {
    setVal(clipValue) ;
  }

  setShapeDirty() ;
}

void RooRealVar::setFitRange(Double_t min, Double_t max) 
{
  // Set new fit range 

  // Check if new limit is consistent
  if (min>max) {
    cout << "RooRealVar::setFitRange(" << GetName() 
	 << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    _binning->setRange(min,min) ;
  } else {
    _binning->setRange(min,max) ;
  }

  setShapeDirty() ;  
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
    Bool_t haveConstant(kFALSE) ;
    removeError() ;
    removeAsymError() ;

    while(1) {      
      if (parser.atEOL() || parser.atEOF()) break ;
      token=parser.readToken() ;

      if (!token.CompareTo("+")) {
	
	// Expect +/- as 3-token sequence
	if (parser.expectToken("/",kTRUE) ||
	    parser.expectToken("-",kTRUE)) {
	  break ;
	}

	// Next token is error
	Double_t error ;
	if (parser.readDouble(error)) break ;
	setError(error) ;

	// Look for optional asymmetric error
	TString tmp = parser.readToken() ;
	if (tmp.CompareTo("(")) {
	  // No error, but back token
	  parser.putBackToken(tmp) ;
	} else {
	  // Have error
	  Double_t asymErrLo, asymErrHi ;
	  if (parser.readDouble(asymErrLo,kTRUE) ||
	      parser.expectToken(",",kTRUE) || 
	      parser.readDouble(asymErrHi,kTRUE) ||
	      parser.expectToken(")",kTRUE)) break ;		      
	  setAsymError(asymErrLo,asymErrHi) ;
	}

      } else if (!token.CompareTo("C")) {

	// Set constant
	setConstant(kTRUE) ;
	haveConstant = kTRUE ;

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
//   	setPlotRange(plotMin,plotMax) ;
	cout << "RooRealVar::readFromStream(" << GetName() 
	     << ") WARNING: plot range deprecated, removed P(...) token" << endl ;

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
	//setFitBins(fitBins) ;
	//setFitRange(fitMin,fitMax) ;
	cout << "RooRealVar::readFromStream(" << GetName() 
	     << ") WARNING: F(lo-hi:bins) token deprecated, use L(lo-hi) B(bins)" << endl ;	
	if (!haveConstant) setConstant(kFALSE) ;

      } else if (!token.CompareTo("L")) {

	// Next tokens are fit limits
	Double_t fitMin, fitMax ;
	Int_t fitBins ;
	if (parser.expectToken("(",kTRUE) ||
	    parser.readDouble(fitMin,kTRUE) ||
	    parser.expectToken("-",kTRUE) ||
	    parser.readDouble(fitMax,kTRUE) ||
	    parser.expectToken(")",kTRUE)) break ;
	setFitRange(fitMin,fitMax) ;
	if (!haveConstant) setConstant(kFALSE) ;

      } else if (!token.CompareTo("B")) { 

	// Next tokens are fit limits
	Int_t fitBins ;
	if (parser.expectToken("(",kTRUE) ||
	    parser.readInteger(fitBins,kTRUE) ||
	    parser.expectToken(")",kTRUE)) break ;
	setFitBins(fitBins) ;
	
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

    // Write value with error (if not zero)    
    if (_printScientific) {
      char fmtVal[16], fmtErr[16] ;
      sprintf(fmtVal,"%%.%de",_printSigDigits) ;
      sprintf(fmtErr,"%%.%de",(_printSigDigits+1)/2) ;
      if (_value>=0) os << " " ;
      os << Form(fmtVal,_value) ;

      if (hasError()) {
	os << " +/- " << Form(fmtErr,getError()) ;
      } else {
	os << setw(_printSigDigits+9) << " " ;
      }

      if (hasAsymError()) {
	os << " (" << Form(fmtErr,getAsymErrorLo())
	   << ", " << Form(fmtErr,getAsymErrorHi()) << ")" ;
      }
      os << " " ;
    } else {
      os << format(_printSigDigits,"EFA")->Data() << " " ;
    }

    // Append limits if not constants
    if (isConstant()) {
      os << "C " ;
    }      

    // Append fit limits if not +Inf:-Inf
    if (hasFitMin() || hasFitMax()) {
      os << "L(" ;
      if(hasFitMin()) {
	os << getFitMin();
      }
      else {
	os << "-INF";
      }
      if(hasFitMax()) {
	os << " - " << getFitMax() ;
      }
      else {
	os << " - +INF";
      }
      os << ") " ;
    }

    if (getFitBins()!=100) {
      os << "B(" << getFitBins() << ") " ;
    }

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



TString *RooRealVar::format(Int_t sigDigits, const char *options) const {
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
  Bool_t asymError= opts.Contains("a") ;
  Bool_t useErrorForPrecision= ((showError && !isConstant()) || opts.Contains("p")) && !opts.Contains("f") ;
  // calculate the precision to use
  if(sigDigits < 1) sigDigits= 1;
  Int_t leadingDigitVal= (Int_t)floor(log10(fabs(useErrorForPrecision?_error:_value)));
  Int_t leadingDigitErr= (Int_t)floor(log10(fabs(_error)));
  Int_t whereVal= leadingDigitVal - sigDigits + 1;
  Int_t whereErr= leadingDigitErr - sigDigits + 1;
  char fmtVal[16], fmtErr[16];

  if (_value<0) whereVal -= 1 ;
  sprintf(fmtVal,"%%.%df", whereVal < 0 ? -whereVal : 0);
  sprintf(fmtErr,"%%.%df", whereErr < 0 ? -whereErr : 0);
  TString *text= new TString();
  if(latexMode) text->Append("$");
  // begin the string with "<name> = " if requested
  if(showName) {
    text->Append(getPlotLabel());
    text->Append(" = ");
  }

  // Add leading space if value is positive
  if (_value>=0) text->Append(" ") ;

  // append our value if requested
  char buffer[256];
  if(!hideValue) {
    Double_t chopped= chopAt(_value, whereVal);
    sprintf(buffer, fmtVal, _value);
    text->Append(buffer);
  }
  // append our error if requested and this variable is not constant
  if(hasError() && showError) {
    if(tlatexMode) {
      text->Append(" #pm ");
    }
    else if(latexMode) {
      text->Append("\\pm ");
    }
    else {
      text->Append(" +/- ");
    }
    sprintf(buffer, fmtErr, getError());
    text->Append(buffer);
  }
  
  if (asymError && hasAsymError()) {
    text->Append(" (") ;      
    sprintf(buffer, fmtErr, getAsymErrorLo());
    text->Append(buffer);
    text->Append(", ") ;
    sprintf(buffer, fmtErr, getAsymErrorHi());
    text->Append(buffer);
    text->Append(")") ;
  }

  // append our units if requested
  if(!_unit.IsNull() && showUnit) {
    text->Append(' ');
    text->Append(_unit);
  }
  if(latexMode) text->Append("$");
  return text;
}

Double_t RooRealVar::chopAt(Double_t what, Int_t where) const {
  // What does this do?
  Double_t scale= pow(10.0,where);
  Int_t trunc= (Int_t)floor(what/scale + 0.5);
  return (Double_t)trunc*scale;
}



void RooRealVar::attachToTree(TTree& t, Int_t bufSize)
{
  // Follow usual procedure for value
  RooAbsReal::attachToTree(t,bufSize) ;

  // Attach/create additional branch for error
  if (getAttribute("StoreError")) {
    TString errName(GetName()) ;
    errName.Append("_err") ;
    TBranch* branch = t.GetBranch(errName) ;
    if (branch) {     
      t.SetBranchAddress(errName,&_error) ;
    } else {
      TString format(errName);
      format.Append("/D");
      t.Branch(errName, &_error, (const Text_t*)format, bufSize);
    }
  }

  // Attach/create additional branches for asymmetric error
  if (getAttribute("StoreAsymError")) {
    TString loName(GetName()) ;
    loName.Append("_aerr_lo") ;
    TBranch* lobranch = t.GetBranch(loName) ;
    if (lobranch) {     
      t.SetBranchAddress(loName,&_asymErrLo) ;
    } else {
      TString format(loName);
      format.Append("/D");
      t.Branch(loName, &_asymErrLo, (const Text_t*)format, bufSize);
    }

    TString hiName(GetName()) ;
    hiName.Append("_aerr_hi") ;
    TBranch* hibranch = t.GetBranch(hiName) ;
    if (hibranch) {     
      t.SetBranchAddress(hiName,&_asymErrHi) ;
    } else {
      TString format(hiName);
      format.Append("/D");
      t.Branch(hiName, &_asymErrHi, (const Text_t*)format, bufSize);
    }
  }
}

void RooRealVar::fillTreeBranch(TTree& t) 
{
  // Attach object to a branch of given TTree

  // First determine if branch is taken
  TString cleanName(cleanBranchName()) ;
  TBranch* valBranch = t.GetBranch(cleanName) ;
  if (!valBranch) { 
    cout << "RooAbsReal::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree" << endl ;
    assert(0) ;
  }
  valBranch->Fill() ;

  if (getAttribute("StoreError")) {
    TString errName(GetName()) ;
    errName.Append("_err") ;
    TBranch* errBranch = t.GetBranch(errName) ;
    if (errBranch) errBranch->Fill() ;
  }

  if (getAttribute("StoreAsymError")) {
    TString loName(GetName()) ;
    loName.Append("_aerr_lo") ;
    TBranch* loBranch = t.GetBranch(loName) ;
    if (loBranch) loBranch->Fill() ;

    TString hiName(GetName()) ;
    hiName.Append("_aerr_hi") ;
    TBranch* hiBranch = t.GetBranch(hiName) ;
    if (hiBranch) hiBranch->Fill() ;
  }
}


void RooRealVar::copyCache(const RooAbsArg* source) 
{
  // Copy the cached value of another RooAbsArg to our cache

  // Warning: This function copies the cached values of source,
  //          it is the callers responsibility to make sure the cache is clean

  // Follow usual procedure for valueklog
  RooAbsReal::copyCache(source) ;

  // Copy error too, if source has one
  RooRealVar* other = dynamic_cast<RooRealVar*>(const_cast<RooAbsArg*>(source)) ;
  if (other) {
    // Copy additional error value
    _error = other->_error ;
    _asymErrLo = other->_asymErrLo ;
    _asymErrHi = other->_asymErrHi ;
  }
}


void RooRealVar::Streamer(TBuffer &R__b)
{
   // Stream an object of class RooRealVar.

   UInt_t R__s, R__c;
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
      RooAbsRealLValue::Streamer(R__b);
      if (R__v==1) {
	cout << "RooRealVar::Streamer(" << GetName() << ") converting version 1 data format" << endl ;
	Double_t fitMin, fitMax ;
	Int_t fitBins ; 
	R__b >> fitMin;
	R__b >> fitMax;
	R__b >> fitBins;
	_binning = new RooUniformBinning(fitMin,fitMax,fitBins) ;
      }
      R__b >> _error;
      R__b >> _asymErrLo;
      R__b >> _asymErrHi;
      if (R__v!=1) {
	R__b >> _binning;
      }
      R__b.CheckByteCount(R__s, R__c, RooRealVar::IsA());
   } else {
      R__c = R__b.WriteVersion(RooRealVar::IsA(), kTRUE);
      RooAbsRealLValue::Streamer(R__b);
      Double_t fitMin(getFitMin()),fitMax(getFitMax()),fitBins(getFitBins()) ;
      R__b << _error;
      R__b << _asymErrLo;
      R__b << _asymErrHi;
      R__b << _binning;
      R__b.SetByteCount(R__c, kTRUE);
   }
}
