/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
//
// BEGIN_HTML
// RooRealVar represents a fundamental (non-derived) real valued object
// 
// This class also holds an (asymmetic) error, a default range and
// a optionally series of alternate named ranges.
// END_HTML
//


#include "RooFit.h"
#include "Riostream.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <iomanip>
#include "TObjString.h"
#include "TTree.h"
#include "RooRealVar.h"
#include "RooStreamParser.h"
#include "RooErrorVar.h"
#include "RooRangeBinning.h"
#include "RooCmdConfig.h"
#include "RooMsgService.h"
#include "RooParamBinning.h"
#include "RooVectorDataStore.h"


ClassImp(RooRealVar)
;

Bool_t RooRealVar::_printScientific(kFALSE) ;
Int_t  RooRealVar::_printSigDigits(5) ;
RooSharedPropertiesList RooRealVar::_sharedPropList ;
RooRealVarSharedProperties RooRealVar::_nullProp("00000000-0000-0000-0000-000000000000") ;


//_____________________________________________________________________________
RooRealVar::RooRealVar()  :  _error(0), _asymErrLo(0), _asymErrHi(0), _binning(0), _sharedProp(0)
{  
  // Default constructor
}


//_____________________________________________________________________________
RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t value, const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(-1), _asymErrLo(1), _asymErrHi(-1), _sharedProp(0)
{
  // Constructor with value and unit

  // _instanceList.registerInstance(this) ;
  _binning = new RooUniformBinning(-1,1,100) ;
  _value = value ;
  removeRange();
  setConstant(kTRUE) ;
}  


//_____________________________________________________________________________
RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t minValue, Double_t maxValue,
		       const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(-1), _asymErrLo(1), _asymErrHi(-1), _sharedProp(0)
{
  // Constructor with range and unit. Initial value is center of range

  _binning = new RooUniformBinning(minValue,maxValue,100) ;

  if (RooNumber::isInfinite(minValue)) {
    if (RooNumber::isInfinite(maxValue)) {
      // [-inf,inf]
      _value = 0 ;
    } else {
      // [-inf,X]
      _value= maxValue ;
    }
  } else {
    if (RooNumber::isInfinite(maxValue)) {
      // [X,inf]
      _value = minValue ;
    } else {
      // [X,X]
      _value= 0.5*(minValue + maxValue);
    }
  }

  //   setPlotRange(minValue,maxValue) ;
  setRange(minValue,maxValue) ;
}  


//_____________________________________________________________________________
RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t value, Double_t minValue, Double_t maxValue,
		       const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(-1), _asymErrLo(1), _asymErrHi(-1), _sharedProp(0)
{
  // Constructor with value, range and unit

  _value = value ;

  _binning = new RooUniformBinning(minValue,maxValue,100) ;
  setRange(minValue,maxValue) ;
}  


//_____________________________________________________________________________
RooRealVar::RooRealVar(const RooRealVar& other, const char* name) :
  RooAbsRealLValue(other,name), 
  _error(other._error),
  _asymErrLo(other._asymErrLo),
  _asymErrHi(other._asymErrHi)
{
  // Copy Constructor

  _sharedProp =  (RooRealVarSharedProperties*) _sharedPropList.registerProperties(other.sharedProp()) ;
  _binning = other._binning->clone() ;
  _binning->insertHook(*this) ;

  //cout << "RooRealVar::cctor(this = " << this << " name = " << GetName() << ", other = " << &other << ")" << endl ;
  
  RooAbsBinning* ab ;
  TIterator* iter = other._altNonSharedBinning.MakeIterator() ;
  while((ab=(RooAbsBinning*)iter->Next())) {
    RooAbsBinning* abc = ab->clone() ;
    //cout << "cloning binning " << ab << " into " << abc << endl ;
    _altNonSharedBinning.Add(abc) ;
    abc->insertHook(*this) ;
  }
  delete iter ;
  
  
}



//_____________________________________________________________________________
RooRealVar::~RooRealVar() 
{
  // Destructor
//   cout << "RooRealVar::dtor(" << this << ")" << endl ;

  delete _binning ;
  _altNonSharedBinning.Delete() ;

  if (_sharedProp) {
    _sharedPropList.unregisterProperties(_sharedProp) ;
  }
}


//_____________________________________________________________________________
Double_t RooRealVar::getVal(const RooArgSet*) const 
{ 
  // Return value of variable

  return _value ; 
}



//_____________________________________________________________________________
void RooRealVar::setVal(Double_t value) 
{
  // Set value of variable to 'value'. If 'value' is outside
  // range of object, clip value into range

  Double_t clipValue ;
  inRange(value,0,&clipValue) ;

  if (clipValue != _value) {
    setValueDirty() ;
    _value = clipValue;
  }
}



//_____________________________________________________________________________
void RooRealVar::setVal(Double_t value, const char* rangeName) 
{
  // Set value of variable to 'value'. If 'value' is outside
  // range named 'rangeName' of object, clip value into that range

  Double_t clipValue ;
  inRange(value,rangeName,&clipValue) ;

  if (clipValue != _value) {
    setValueDirty() ;
    _value = clipValue;
  }
}



//_____________________________________________________________________________
RooErrorVar* RooRealVar::errorVar() const 
{
  // Return a RooAbsRealLValue representing the error associated
  // with this variable. The callers takes ownership of the
  // return object

  TString name(GetName()), title(GetTitle()) ;
  name.Append("err") ;
  title.Append(" Error") ;

  return new RooErrorVar(name,title,*this) ;
}



//_____________________________________________________________________________
Bool_t RooRealVar::hasBinning(const char* name) const
{
  // Returns true if variable has a binning with 'name'

  return sharedProp()->_altBinning.FindObject(name) ? kTRUE : kFALSE ;
}



//_____________________________________________________________________________
const RooAbsBinning& RooRealVar::getBinning(const char* name, Bool_t verbose, Bool_t createOnTheFly) const 
{
  // Return binning definition with name. If binning with 'name' is not found it is created
  // on the fly as a clone of the default binning if createOnTheFly is true, otherwise
  // a reference to the default binning is returned. If verbose is true a message
  // is printed if a binning is created on the gly

  return const_cast<RooRealVar*>(this)->getBinning(name, verbose, createOnTheFly) ;
}



//_____________________________________________________________________________
RooAbsBinning& RooRealVar::getBinning(const char* name, Bool_t verbose, Bool_t createOnTheFly) 
{
  // Return binning definition with name. If binning with 'name' is not found it is created
  // on the fly as a clone of the default binning if createOnTheFly is true, otherwise
  // a reference to the default binning is returned. If verbose is true a message
  // is printed if a binning is created on the gly

  // Return default (normalization) binning and range if no name is specified
  if (name==0) {
    return *_binning ;
  }
  
  // Check if non-shared binning with this name has been created already
  RooAbsBinning* binning = (RooAbsBinning*) _altNonSharedBinning.FindObject(name) ;
  if (binning) {
    return *binning ;
  }

  // Check if binning with this name has been created already
  binning = (RooAbsBinning*) (sharedProp()->_altBinning).FindObject(name) ;
  if (binning) {
    return *binning ;
  }


  // Return default binning if requested binning doesn't exist
  if (!createOnTheFly) {
    return *_binning ;
  }  

  // Create a new RooRangeBinning with this name with default range
  binning = new RooRangeBinning(getMin(),getMax(),name) ;
  if (verbose) {
    coutI(Eval) << "RooRealVar::getBinning(" << GetName() << ") new range named '" 
		<< name << "' created with default bounds" << endl ;
  }
  sharedProp()->_altBinning.Add(binning) ;
  
  return *binning ;
}



//_____________________________________________________________________________
void RooRealVar::setBinning(const RooAbsBinning& binning, const char* name) 
{
  // Add given binning under name 'name' with this variable. If name is null
  // the binning is installed as the default binning

  // Process insert hooks required for parameterized binnings
  if (!name) {
    RooAbsBinning* newBinning = binning.clone() ;
    if (_binning) {
      _binning->removeHook(*this) ;
      delete _binning ;
    }
    newBinning->insertHook(*this) ;
    _binning = newBinning ;
  } else {

    RooLinkedList* altBinning = binning.isShareable() ? &(sharedProp()->_altBinning) : &_altNonSharedBinning ;

    RooAbsBinning* newBinning = binning.clone() ;

    // Remove any old binning with this name
    RooAbsBinning* oldBinning = (RooAbsBinning*) altBinning->FindObject(name) ;
    if (oldBinning) {
      altBinning->Remove(oldBinning) ;
      oldBinning->removeHook(*this) ;
      delete oldBinning ;
    }

    // Insert new binning in list of alternative binnings
    newBinning->SetName(name) ;
    newBinning->SetTitle(name) ;
    newBinning->insertHook(*this) ;
    altBinning->Add(newBinning) ;
    
  }
  

}



//_____________________________________________________________________________
void RooRealVar::setMin(const char* name, Double_t value) 
{
  // Set minimum of name range to given value. If name is null
  // minimum of default range is set

  // Set new minimum of fit range 
  RooAbsBinning& binning = getBinning(name,kTRUE,kTRUE) ;

  // Check if new limit is consistent
  if (value >= getMax()) {
    coutW(InputArguments) << "RooRealVar::setMin(" << GetName() 
			  << "): Proposed new fit min. larger than max., setting min. to max." << endl ;
    binning.setMin(getMax()) ;
  } else {
    binning.setMin(value) ;
  }

  // Clip current value in window if it fell out
  if (!name) {
    Double_t clipValue ;
    if (!inRange(_value,0,&clipValue)) {
      setVal(clipValue) ;
    }
  }
    
  setShapeDirty() ;
}


//_____________________________________________________________________________
void RooRealVar::setMax(const char* name, Double_t value)
{
  // Set maximum of name range to given value. If name is null
  // maximum of default range is set

  // Set new maximum of fit range 
  RooAbsBinning& binning = getBinning(name,kTRUE,kTRUE) ;

  // Check if new limit is consistent
  if (value < getMin()) {
    coutW(InputArguments) << "RooRealVar::setMax(" << GetName() 
			  << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    binning.setMax(getMin()) ;
  } else {
    binning.setMax(value) ;
  }

  // Clip current value in window if it fell out
  if (!name) {
    Double_t clipValue ;
    if (!inRange(_value,0,&clipValue)) {
      setVal(clipValue) ;
    }
  }

  setShapeDirty() ;
}


//_____________________________________________________________________________
void RooRealVar::setRange(const char* name, Double_t min, Double_t max) 
{
  // Set range named 'name to [min,max]. If name is null
  // range of default range is adjusted. If no range with
  // 'name' exists it is created on the fly

  Bool_t exists = name ? (sharedProp()->_altBinning.FindObject(name)?kTRUE:kFALSE) : kTRUE ;

  // Set new fit range 
  RooAbsBinning& binning = getBinning(name,kFALSE,kTRUE) ;

  // Check if new limit is consistent
  if (min>max) {
    coutW(InputArguments) << "RooRealVar::setRange(" << GetName() 
			  << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    binning.setRange(min,min) ;
  } else {
    binning.setRange(min,max) ;
  }

  if (!exists) {
    coutI(Eval) << "RooRealVar::setRange(" << GetName() 
		<< ") new range named '" << name << "' created with bounds [" 
		<< min << "," << max << "]" << endl ;
  }

  setShapeDirty() ;  
}



//_____________________________________________________________________________
void RooRealVar::setRange(const char* name, RooAbsReal& min, RooAbsReal& max) 
{
  // Create or modify a parameterized range named 'name' that has external functions
  // min and max parameterizing its boundaries.

  RooParamBinning pb(min,max,100) ;
  setBinning(pb,name) ;
}



//_____________________________________________________________________________
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

    Bool_t reprocessToken = kFALSE ;
    while(1) {      
      if (parser.atEOL() || parser.atEOF()) break ;

      if (!reprocessToken) {
	token=parser.readToken() ;
      }
      reprocessToken = kFALSE ;

      if (!token.CompareTo("+")) {
	
	// Expect +/- as 3-token sequence
	if (parser.expectToken("/",kTRUE) ||
	    parser.expectToken("-",kTRUE)) {
	  break ;
	}

	// Next token is error or asymmetric error, check if first char of token is a '('
	TString tmp = parser.readToken() ;
	if (tmp.CompareTo("(")) {
	  // Symmetric error, convert token do double

	  Double_t error ;
	  parser.convertToDouble(tmp,error) ;
	  setError(error) ;

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
	coutW(Eval) << "RooRealVar::readFromStrem(" << GetName() 
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
	//setBins(fitBins) ;
	//setRange(fitMin,fitMax) ;
	coutW(Eval) << "RooRealVar::readFromStream(" << GetName() 
	     << ") WARNING: F(lo-hi:bins) token deprecated, use L(lo-hi) B(bins)" << endl ;	
	if (!haveConstant) setConstant(kFALSE) ;

      } else if (!token.CompareTo("L")) {

	// Next tokens are fit limits
	Double_t fitMin, fitMax ;
//	Int_t fitBins ;
	if (parser.expectToken("(",kTRUE) ||
	    parser.readDouble(fitMin,kTRUE) ||
	    parser.expectToken("-",kTRUE) ||
	    parser.readDouble(fitMax,kTRUE) ||
	    parser.expectToken(")",kTRUE)) break ;
	setRange(fitMin,fitMax) ;
	if (!haveConstant) setConstant(kFALSE) ;

      } else if (!token.CompareTo("B")) { 

	// Next tokens are fit limits
	Int_t fitBins ;
	if (parser.expectToken("(",kTRUE) ||
	    parser.readInteger(fitBins,kTRUE) ||
	    parser.expectToken(")",kTRUE)) break ;
	setBins(fitBins) ;
	
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


//_____________________________________________________________________________
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
      snprintf(fmtVal,16,"%%.%de",_printSigDigits) ;
      snprintf(fmtErr,16,"%%.%de",(_printSigDigits+1)/2) ;
      if (_value>=0) os << " " ;
      os << Form(fmtVal,_value) ;

      if (hasAsymError()) {
	os << " +/- (" << Form(fmtErr,getAsymErrorLo())
	   << ", " << Form(fmtErr,getAsymErrorHi()) << ")" ;
      } else  if (hasError()) {
	os << " +/- " << Form(fmtErr,getError()) ;
      } 

      os << " " ;
    } else {
      TString* tmp = format(_printSigDigits,"EFA") ;
      os << tmp->Data() << " " ;
      delete tmp ;
    }

    // Append limits if not constants
    if (isConstant()) {
      os << "C " ;
    }      

    // Append fit limits
    os << "L(" ;
    if(hasMin()) {
      os << getMin();
    }
    else {
      os << "-INF";
    }
    if(hasMax()) {
      os << " - " << getMax() ;
    }
    else {
      os << " - +INF";
    }
    os << ") " ;

    if (getBins()!=100) {
      os << "B(" << getBins() << ") " ;
    }

    // Add comment with unit, if unit exists
    if (!_unit.IsNull())
      os << "// [" << getUnit() << "]" ;
  }
}



//_____________________________________________________________________________
void RooRealVar::printValue(ostream& os) const 
{
  // Print value of variable
  os << getVal() ;

  if(hasError() && !hasAsymError()) {
    os << " +/- " << getError() ;
  } else if (hasAsymError()) {
    os << " +/- (" << getAsymErrorLo() << "," << getAsymErrorHi() << ")" ;
  }


}


//_____________________________________________________________________________
void RooRealVar::printExtras(ostream& os) const
{
  // Print extras of variable: (asymmetric) error, constant flag, limits and binning

  // Append limits if not constants
  if (isConstant()) {
    os << "C " ;
  }      

  // Append fit limits
  os << " L(" ;
  if(hasMin()) {
    os << getMin();
  }
  else {
    os << "-INF";
  }
  if(hasMax()) {
    os << " - " << getMax() ;
  }
  else {
    os << " - +INF";
  }
  os << ") " ;
  
  if (getBins()!=100) {
    os << "B(" << getBins() << ") " ;
  }
  
  // Add comment with unit, if unit exists
  if (!_unit.IsNull())
    os << "// [" << getUnit() << "]" ;
  
}


//_____________________________________________________________________________
Int_t RooRealVar::defaultPrintContents(Option_t* opt) const 
{
  // Mapping of Print() option string to RooPrintable contents specifications

  if (opt && TString(opt)=="I") {
    return kName|kClassName|kValue ;
  }
  return kName|kClassName|kValue|kExtras ;
}


//_____________________________________________________________________________
void RooRealVar::printMultiline(ostream& os, Int_t contents, Bool_t verbose, TString indent) const
{
  // Detailed printing interface

  RooAbsRealLValue::printMultiline(os,contents,verbose,indent);
  os << indent << "--- RooRealVar ---" << endl;
  TString unit(_unit);
  if(!unit.IsNull()) unit.Prepend(' ');
  os << indent << "  Error = " << getError() << unit << endl;
}



//_____________________________________________________________________________
TString* RooRealVar::format(const RooCmdArg& formatArg) const 
{
  // Format contents of RooRealVar for pretty printing on RooPlot
  // parameter boxes. This function processes the named arguments
  // taken by paramOn() and translates them to an option string
  // parsed by RooRealVar::format(Int_t sigDigits, const char *options) 

  RooCmdArg tmp(formatArg) ;
  tmp.setProcessRecArgs(kTRUE) ;

  RooCmdConfig pc(Form("RooRealVar::format(%s)",GetName())) ;
  pc.defineString("what","FormatArgs",0,"") ;
  pc.defineInt("autop","FormatArgs::AutoPrecision",0,2) ;
  pc.defineInt("fixedp","FormatArgs::FixedPrecision",0,2) ;
  pc.defineInt("tlatex","FormatArgs::TLatexStyle",0,0) ;
  pc.defineInt("latex","FormatArgs::LatexStyle",0,0) ;
  pc.defineInt("latext","FormatArgs::LatexTableStyle",0,0) ;
  pc.defineInt("verbn","FormatArgs::VerbatimName",0,0) ;
  pc.defineMutex("FormatArgs::TLatexStyle","FormatArgs::LatexStyle","FormatArgs::LatexTableStyle") ;
  pc.defineMutex("FormatArgs::AutoPrecision","FormatArgs::FixedPrecision") ;

  // Process & check varargs 
  pc.process(tmp) ;
  if (!pc.ok(kTRUE)) {
    return 0 ;
  }

  // Extract values from named arguments
  TString options ;
  options = pc.getString("what") ;
  
  if (pc.getInt("tlatex")) {
    options += "L" ;
  } else if (pc.getInt("latex")) {
    options += "X" ;
  } else if (pc.getInt("latext")) {
    options += "Y" ;
  }   

  if (pc.getInt("verbn")) options += "V" ;
  Int_t sigDigits = 2 ;
  if (pc.hasProcessed("FormatArgs::AutoPrecision")) {
    options += "P" ;
    sigDigits = pc.getInt("autop") ;
  } else if (pc.hasProcessed("FormatArgs::FixedPrecision")) {
    options += "F" ;
    sigDigits = pc.getInt("fixedp") ;
  }
  
  return format(sigDigits,options) ;
}




//_____________________________________________________________________________
TString *RooRealVar::format(Int_t sigDigits, const char *options) const 
{
  // Format numeric value of RooRealVar and its error in a variety of ways
  //
  // To control what is shown use the following options
  // N = show name
  // H = hide value
  // E = show error
  // A = show asymmetric error instead of parabolic error (if available)
  // U = show unit
  //
  // To control how it is shown use these options
  // L = TLatex mode
  // X = Latex mode
  // Y = Latex table mode ( '=' replaced by '&' )
  // V = Make name \verbatim in Latex mode
  // P = use error to control shown precision
  // F = force fixed precision
  //
  
  //cout << "format = " << options << endl ;

  // parse the options string
  TString opts(options);
  opts.ToLower();
  Bool_t showName= opts.Contains("n");
  Bool_t hideValue= opts.Contains("h");
  Bool_t showError= opts.Contains("e");
  Bool_t showUnit= opts.Contains("u");
  Bool_t tlatexMode= opts.Contains("l");
  Bool_t latexMode= opts.Contains("x");
  Bool_t latexTableMode = opts.Contains("y") ;
  Bool_t latexVerbatimName = opts.Contains("v") ;

  if (latexTableMode) latexMode = kTRUE ;
  Bool_t asymError= opts.Contains("a") ;
  Bool_t useErrorForPrecision= (((showError && hasError(kFALSE) && !isConstant()) || opts.Contains("p")) && !opts.Contains("f")) ;
  // calculate the precision to use
  if(sigDigits < 1) sigDigits= 1;
  Int_t leadingDigitVal = 0;
  if (useErrorForPrecision) {    
    leadingDigitVal = (Int_t)floor(log10(fabs(_error+1e-10)));
    if (_value==0&&_error==0) leadingDigitVal=0 ;
  } else {
    leadingDigitVal = (Int_t)floor(log10(fabs(_value+1e-10)));
    if (_value==0) leadingDigitVal=0 ;
  }
  Int_t leadingDigitErr= (Int_t)floor(log10(fabs(_error)));
  Int_t whereVal= leadingDigitVal - sigDigits + 1;
  Int_t whereErr= leadingDigitErr - sigDigits + 1;
  char fmtVal[16], fmtErr[16];

  if (_value<0) whereVal -= 1 ;
  snprintf(fmtVal,16,"%%.%df", whereVal < 0 ? -whereVal : 0);
  snprintf(fmtErr,16,"%%.%df", whereErr < 0 ? -whereErr : 0);
  TString *text= new TString();
  if(latexMode) text->Append("$");
  // begin the string with "<name> = " if requested
  if(showName) {
    if (latexTableMode && latexVerbatimName) {
      text->Append("\\verb+") ;
    }
    text->Append(getPlotLabel());
    if (latexVerbatimName) text->Append("+") ;

    if (!latexTableMode) {
      text->Append(" = ");
    } else {
      text->Append(" $ & $ ");
    }
  }

  // Add leading space if value is positive
  if (_value>=0) text->Append(" ") ;

  // append our value if requested
  char buffer[256];
  if(!hideValue) {
    chopAt(_value, whereVal);
    snprintf(buffer, 256,fmtVal, _value);
    text->Append(buffer);
  }

  // append our error if requested and this variable is not constant
  if(hasError(kFALSE) && showError && !(asymError && hasAsymError(kFALSE))) {
    if(tlatexMode) {
      text->Append(" #pm ");
    }
    else if(latexMode) {
      text->Append("\\pm ");
    }
    else {
      text->Append(" +/- ");
    }
    snprintf(buffer, 256,fmtErr, getError());
    text->Append(buffer);
  }
  
  if (asymError && hasAsymError() && showError) {
    if(tlatexMode) {
      text->Append(" #pm ");
      text->Append("_{") ;      
      snprintf(buffer, 256,fmtErr, getAsymErrorLo());
      text->Append(buffer);
      text->Append("}^{+") ;
      snprintf(buffer, 256,fmtErr, getAsymErrorHi());
      text->Append(buffer);
      text->Append("}") ;
    }
    else if(latexMode) {
      text->Append("\\pm ");
      text->Append("_{") ;      
      snprintf(buffer, 256,fmtErr, getAsymErrorLo());
      text->Append(buffer);
      text->Append("}^{+") ;
      snprintf(buffer, 256,fmtErr, getAsymErrorHi());
      text->Append(buffer);
      text->Append("}") ;
    }
    else {
      text->Append(" +/- ");
      text->Append(" (") ;      
      snprintf(buffer, 256, fmtErr, getAsymErrorLo());
      text->Append(buffer);
      text->Append(", ") ;
      snprintf(buffer, 256, fmtErr, getAsymErrorHi());
      text->Append(buffer);
      text->Append(")") ;
    }

  }

  // append our units if requested
  if(!_unit.IsNull() && showUnit) {
    text->Append(' ');
    text->Append(_unit);
  }
  if(latexMode) text->Append("$");
  return text;
}



//_____________________________________________________________________________
Double_t RooRealVar::chopAt(Double_t what, Int_t where) const 
{
  // Utility to calculate number of decimals to show
  // based on magnitude of error

  Double_t scale= pow(10.0,where);
  Int_t trunc= (Int_t)floor(what/scale + 0.5);
  return (Double_t)trunc*scale;
}



//_____________________________________________________________________________
void RooRealVar::attachToVStore(RooVectorDataStore& vstore) 
{
  // Overload RooAbsReal::attachToTree to also attach
  // branches for errors  and/or asymmetric errors
  // attribute StoreError and/or StoreAsymError are set

  // Follow usual procedure for value

  if (getAttribute("StoreError") || getAttribute("StoreAsymError")) {
    
    RooVectorDataStore::RealFullVector* rfv = vstore.addRealFull(this) ;
    rfv->setBuffer(&_value) ;
  
    // Attach/create additional branch for error
    if (getAttribute("StoreError")) {
      rfv->setErrorBuffer(&_error) ;
    }
    
    // Attach/create additional branches for asymmetric error
    if (getAttribute("StoreAsymError")) {
      rfv->setAsymErrorBuffer(&_asymErrLo,&_asymErrHi) ;
    }

  } else {

    RooAbsReal::attachToVStore(vstore) ;

  }
}



//_____________________________________________________________________________
void RooRealVar::attachToTree(TTree& t, Int_t bufSize)
{
  // Overload RooAbsReal::attachToTree to also attach
  // branches for errors  and/or asymmetric errors
  // attribute StoreError and/or StoreAsymError are set

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
      TString format2(errName);
      format2.Append("/D");
      t.Branch(errName, &_error, (const Text_t*)format2, bufSize);
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
      TString format2(loName);
      format2.Append("/D");
      t.Branch(loName, &_asymErrLo, (const Text_t*)format2, bufSize);
    }

    TString hiName(GetName()) ;
    hiName.Append("_aerr_hi") ;
    TBranch* hibranch = t.GetBranch(hiName) ;
    if (hibranch) {     
      t.SetBranchAddress(hiName,&_asymErrHi) ;
    } else {
      TString format2(hiName);
      format2.Append("/D");
      t.Branch(hiName, &_asymErrHi, (const Text_t*)format2, bufSize);
    }
  }
}


//_____________________________________________________________________________
void RooRealVar::fillTreeBranch(TTree& t) 
{
  // Overload RooAbsReal::fillTreeBranch to also
  // fill tree branches with (asymmetric) errors
  // if requested.

  // First determine if branch is taken
  TString cleanName(cleanBranchName()) ;
  TBranch* valBranch = t.GetBranch(cleanName) ;
  if (!valBranch) { 
    coutE(Eval) << "RooAbsReal::fillTreeBranch(" << GetName() << ") ERROR: not attached to tree" << endl ;
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



//_____________________________________________________________________________
void RooRealVar::copyCache(const RooAbsArg* source, Bool_t valueOnly) 
{
  // Copy the cached value of another RooAbsArg to our cache
  // Warning: This function copies the cached values of source,
  //          it is the callers responsibility to make sure the cache is clean

  // Follow usual procedure for valueklog
  RooAbsReal::copyCache(source) ;

  if (valueOnly) return ;

  // Copy error too, if source has one
  RooRealVar* other = dynamic_cast<RooRealVar*>(const_cast<RooAbsArg*>(source)) ;
  if (other) {
    // Copy additional error value
    _error = other->_error ;
    _asymErrLo = other->_asymErrLo ;
    _asymErrHi = other->_asymErrHi ;
  }
}



//_____________________________________________________________________________
void RooRealVar::Streamer(TBuffer &R__b)
{
  // Stream an object of class RooRealVar.

  UInt_t R__s, R__c;
  if (R__b.IsReading()) {
    
    Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
    RooAbsRealLValue::Streamer(R__b);
    if (R__v==1) {
      coutI(Eval) << "RooRealVar::Streamer(" << GetName() << ") converting version 1 data format" << endl ;
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
    if (R__v>=2) {
      R__b >> _binning;
    }
    if (R__v==3) {
      R__b >> _sharedProp ;
      _sharedProp = (RooRealVarSharedProperties*) _sharedPropList.registerProperties(_sharedProp,kFALSE) ;
    }
    if (R__v>=4) {
      RooRealVarSharedProperties* tmpSharedProp = new RooRealVarSharedProperties() ;
      tmpSharedProp->Streamer(R__b) ;
      if (!(_nullProp==*tmpSharedProp)) {
	_sharedProp = (RooRealVarSharedProperties*) _sharedPropList.registerProperties(tmpSharedProp,kFALSE) ;
      } else {
	delete tmpSharedProp ;
	_sharedProp = 0 ;
      }
    }
    
    R__b.CheckByteCount(R__s, R__c, RooRealVar::IsA());
    
  } else {
    
    R__c = R__b.WriteVersion(RooRealVar::IsA(), kTRUE);
    RooAbsRealLValue::Streamer(R__b);
    R__b << _error;
    R__b << _asymErrLo;
    R__b << _asymErrHi;
    R__b << _binning;      
    if (_sharedProp) {
      _sharedProp->Streamer(R__b) ;
    } else {
      _nullProp.Streamer(R__b) ;
    }
    R__b.SetByteCount(R__c, kTRUE);      
    
  }
}



//_____________________________________________________________________________
void RooRealVar::deleteSharedProperties()
{
  // No longer used?

  if (_sharedProp) {
    _sharedPropList.unregisterProperties(_sharedProp) ;
    _sharedProp = 0 ;
  }  
}


//_____________________________________________________________________________
void RooRealVar::printScientific(Bool_t flag) 
{ 
  // If true, contents of RooRealVars will be printed in scientific notation

  _printScientific = flag ; 
}


//_____________________________________________________________________________
void RooRealVar::printSigDigits(Int_t ndig) 
{ 
  // Set number of digits to show when printing RooRealVars

  _printSigDigits = ndig>1?ndig:1 ; 
}
