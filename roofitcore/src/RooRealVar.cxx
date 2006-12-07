/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooRealVar.cc,v 1.59 2006/07/04 15:07:58 wverkerke Exp $
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

// -- CLASS DESCRIPTION [REAL] --
// RooRealVar represents a fundamental (non-derived) real valued object
// 
// This class also holds an error and a fit range associated with the real value


#include "RooFitCore/RooFit.hh"

#include <math.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <iomanip>
#include "TObjString.h"
#include "TTree.h"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooErrorVar.hh"
#include "RooFitCore/RooRangeBinning.hh"
#include "RooFitCore/RooCmdConfig.hh"

ClassImp(RooRealVar)
;

Bool_t RooRealVar::_printScientific(kFALSE) ;
Int_t  RooRealVar::_printSigDigits(5) ;
RooSharedPropertiesList RooRealVar::_sharedPropList ;
RooRealVarSharedProperties RooRealVar::_nullProp("00000000-0000-0000-0000-000000000000") ;

RooRealVar::RooRealVar() 
{
  // Default constructor
}

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

RooRealVar::RooRealVar(const char *name, const char *title,
		       Double_t minValue, Double_t maxValue,
		       const char *unit) :
  RooAbsRealLValue(name, title, unit), _error(-1), _asymErrLo(1), _asymErrHi(-1), _sharedProp(0)
{
  // Constructor with range and unit. Value is set to middle of range
  _binning = new RooUniformBinning(minValue,maxValue,100) ;

  _value= 0.5*(minValue + maxValue);

//   setPlotRange(minValue,maxValue) ;
  setRange(minValue,maxValue) ;
}  

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

RooRealVar::RooRealVar(const RooRealVar& other, const char* name) :
  RooAbsRealLValue(other,name), 
  _error(other._error),
  _asymErrLo(other._asymErrLo),
  _asymErrHi(other._asymErrHi)
{
  // Copy Constructor

  _sharedProp =  (RooRealVarSharedProperties*) _sharedPropList.registerProperties(other.sharedProp()) ;
  _binning = other._binning->clone() ;
}


RooRealVar::~RooRealVar() 
{
  // Destructor
  delete _binning ;
  if (_sharedProp) {
    _sharedPropList.unregisterProperties(_sharedProp) ;
  }
}

Double_t RooRealVar::getVal(const RooArgSet*) const 
{ 
  return _value ; 
}


void RooRealVar::setVal(Double_t value) {
  // Set current value
  Double_t clipValue ;
  inRange(value,&clipValue) ;

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


Bool_t RooRealVar::hasBinning(const char* name) const
{
  return sharedProp()->_altBinning.FindObject(name) ? kTRUE : kFALSE ;
}


const RooAbsBinning& RooRealVar::getBinning(const char* name, Bool_t verbose, Bool_t createOnTheFly) const 
{
  return const_cast<RooRealVar*>(this)->getBinning(name, verbose, createOnTheFly) ;
}


RooAbsBinning& RooRealVar::getBinning(const char* name, Bool_t verbose, Bool_t createOnTheFly) 
{
  // Return default (normalization) binning and range if no name is specified
  if (name==0) {
    return *_binning ;
  }
  
  // Check if binning with this name has been created already
  RooAbsBinning* binning = (RooAbsBinning*) (sharedProp()->_altBinning).FindObject(name) ;
  if (binning) {
    return *binning ;
  }

  // If binning is not found, check for it in (live) ancestors in cloning history
//   RooLinkedList ancestors = getCloningAncestors() ;
//   TIterator* ancIter = ancestors.MakeIterator() ;
//   RooRealVar* anc ;
//   while((anc=(RooRealVar*)ancIter->Next())) {
//     if (instanceList().isLive(anc) && anc->hasBinning(name)) {
//       cout << "RooRealVar::getBinning(" << GetName() << ") INFO: this instance of " <<  GetName() << "(" << this << ") has no range named '" 
// 	   << name << "' but cloning ancestor " << anc << " does, using ancestor range" << endl ;
//       delete ancIter ;
//       return anc->getBinning(name,kFALSE,kFALSE) ;
//     }
//   }
//   delete ancIter ;


  // Return default binning if requested binning doesn't exist
  if (!createOnTheFly) {
    return *_binning ;
  }  

  // Create a new RooRangeBinning with this name with default range
  binning = new RooRangeBinning(getMin(),getMax(),name) ;
  if (verbose) {
    cout << "RooRealVar::getBinning(" << GetName() << ") new range named '" 
	 << name << "' created with default bounds" << endl ;
  }
  sharedProp()->_altBinning.Add(binning) ;
  
  return *binning ;
}



void RooRealVar::setBinning(const RooAbsBinning& binning, const char* name) 
{
  if (!name) {
    if (_binning) delete _binning ;
    _binning = binning.clone() ;
  } else {

    // Remove any old binning with this name
    RooAbsBinning* oldBinning = (RooAbsBinning*) (sharedProp()->_altBinning).FindObject(name) ;
    if (oldBinning) {
      sharedProp()->_altBinning.Remove(oldBinning) ;
      delete oldBinning ;
    }

    // Insert new binning in list of alternative binnings
    RooAbsBinning* newBinning = binning.clone() ;
    newBinning->SetName(name) ;
    newBinning->SetTitle(name) ;
    sharedProp()->_altBinning.Add(newBinning) ;

  }
  

}


void RooRealVar::setMin(const char* name, Double_t value) 
{
  // Set new minimum of fit range 
  RooAbsBinning& binning = getBinning(name,kTRUE,kTRUE) ;

  // Check if new limit is consistent
  if (value >= getMax()) {
    cout << "RooRealVar::setMin(" << GetName() 
	 << "): Proposed new fit min. larger than max., setting min. to max." << endl ;
    binning.setMin(getMax()) ;
  } else {
    binning.setMin(value) ;
  }

  // Clip current value in window if it fell out
  if (!name) {
    Double_t clipValue ;
    if (!inRange(_value,&clipValue)) {
      setVal(clipValue) ;
    }
  }
    
  setShapeDirty() ;
}

void RooRealVar::setMax(const char* name, Double_t value)
{
  // Set new maximum of fit range 
  RooAbsBinning& binning = getBinning(name,kTRUE,kTRUE) ;

  // Check if new limit is consistent
  if (value < getMin()) {
    cout << "RooRealVar::setMax(" << GetName() 
	 << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    binning.setMax(getMin()) ;
  } else {
    binning.setMax(value) ;
  }

  // Clip current value in window if it fell out
  if (!name) {
    Double_t clipValue ;
    if (!inRange(_value,&clipValue)) {
      setVal(clipValue) ;
    }
  }

  setShapeDirty() ;
}

void RooRealVar::setRange(const char* name, Double_t min, Double_t max) 
{
  Bool_t exists = name ? (sharedProp()->_altBinning.FindObject(name)?kTRUE:kFALSE) : kTRUE ;

  // Set new fit range 
  RooAbsBinning& binning = getBinning(name,kFALSE,kTRUE) ;

  // Check if new limit is consistent
  if (min>max) {
    cout << "RooRealVar::setRange(" << GetName() 
	 << "): Proposed new fit max. smaller than min., setting max. to min." << endl ;
    binning.setRange(min,min) ;
  } else {
    binning.setRange(min,max) ;
  }

  if (!exists) {
    cout << "RooRealVar::setRange(" << GetName() 
	 << ") new range named '" << name << "' created with bounds [" 
	 << min << "," << max << "]" << endl ;
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
	cout << "RooRealVar::readFromStrem(" << GetName() 
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
	cout << "RooRealVar::readFromStream(" << GetName() 
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

      if (hasAsymError()) {
	os << " +/- (" << Form(fmtErr,getAsymErrorLo())
	   << ", " << Form(fmtErr,getAsymErrorHi()) << ")" ;
      } else  if (hasError()) {
	os << " +/- " << Form(fmtErr,getError()) ;
      } 

      os << " " ;
    } else {
      os << format(_printSigDigits,"EFA")->Data() << " " ;
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


TString* RooRealVar::format(const RooCmdArg& formatArg) const 
{
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




TString *RooRealVar::format(Int_t sigDigits, const char *options) const {
  // Format numeric value in a variety of ways
  //
  // What is shown?
  // N = show name
  // H = hide value
  // E = show error
  // A = show asymmetric error instead of parabolic error (if available)
  // U = show unit
  //
  // How is it shown?
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
  Bool_t useErrorForPrecision= (((showError && !isConstant()) || opts.Contains("p")) && !opts.Contains("f")) ;
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
  sprintf(fmtVal,"%%.%df", whereVal < 0 ? -whereVal : 0);
  sprintf(fmtErr,"%%.%df", whereErr < 0 ? -whereErr : 0);
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
    sprintf(buffer, fmtVal, _value);
    text->Append(buffer);
  }
  // append our error if requested and this variable is not constant
  if(hasError() && showError && !(asymError && hasAsymError())) {
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
  
  if (asymError && hasAsymError() && showError) {
    if(tlatexMode) {
      text->Append(" #pm ");
      text->Append("_{") ;      
      sprintf(buffer, fmtErr, getAsymErrorLo());
      text->Append(buffer);
      text->Append("}^{+") ;
      sprintf(buffer, fmtErr, getAsymErrorHi());
      text->Append(buffer);
      text->Append("}") ;
    }
    else if(latexMode) {
      text->Append("\\pm ");
      text->Append("_{") ;      
      sprintf(buffer, fmtErr, getAsymErrorLo());
      text->Append(buffer);
      text->Append("}^{+") ;
      sprintf(buffer, fmtErr, getAsymErrorHi());
      text->Append(buffer);
      text->Append("}") ;
    }
    else {
      text->Append(" +/- ");
      text->Append(" (") ;      
      sprintf(buffer, fmtErr, getAsymErrorLo());
      text->Append(buffer);
      text->Append(", ") ;
      sprintf(buffer, fmtErr, getAsymErrorHi());
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
      if (R__v>=2) {
	R__b >> _binning;
      }
      if (R__v==3) {
 	R__b >> _sharedProp ;
	_sharedProp = (RooRealVarSharedProperties*) _sharedPropList.registerProperties(_sharedProp) ;
      }
      if (R__v==4) {
	RooRealVarSharedProperties* tmpSharedProp = new RooRealVarSharedProperties() ;
	tmpSharedProp->Streamer(R__b) ;
	if (!(_nullProp==*tmpSharedProp)) {
	  _sharedProp = (RooRealVarSharedProperties*) _sharedPropList.registerProperties(tmpSharedProp) ;
	} else {
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


void RooRealVar::setFitBins(Int_t nBins) 
{
  cout << "WARNING setFitBins() IS OBSOLETE, PLEASE USE setBins()" << endl ;
  setBins(nBins) ;
}

void RooRealVar::setFitMin(Double_t value) 
{
  cout << "WARNING setFitMin() IS OBSOLETE, PLEASE USE setMin()" << endl ;
  setMin(value) ;
}

void RooRealVar::setFitMax(Double_t value) 
{
  cout << "WARNING setFitMax() IS OBSOLETE, PLEASE USE setMin()" << endl ;
  setMax(value) ;
}

void RooRealVar::setFitRange(Double_t min, Double_t max) 
{
  cout << "WARNING setFitRange() IS OBSOLETE, PLEASE USE setRange()" << endl ;
  setRange(min,max) ;
}

void RooRealVar::removeFitMin() 
{
  cout << "WARNING removeFitMin() IS OBSOLETE, PLEASE USE removeMin()" << endl ;
  removeMin() ;
}

void RooRealVar::removeFitMax() 
{
  cout << "WARNING removeFitMax() IS OBSOLETE, PLEASE USE removeMax()" << endl ;
  removeMax() ;
}

void RooRealVar::removeFitRange() 
{
  cout << "WARNING removeFitRange() IS OBSOLETE, PLEASE USE removeRange()" << endl ;
  removeRange() ;
}

void RooRealVar::deleteSharedProperties()
{
  if (_sharedProp) {
    _sharedPropList.unregisterProperties(_sharedProp) ;
    _sharedProp = 0 ;
  }  
}

