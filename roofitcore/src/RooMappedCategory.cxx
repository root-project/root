/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooMappedCategory.cc,v 1.5 2001/03/29 01:59:09 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/
#include "BaBar/BaBar.hh"

#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooFitCore/RooMappedCategory.hh"
#include "RooFitCore/RooStreamParser.hh"

ClassImp(RooMappedCategory)


RooMappedCategory::RooMappedCategory(const char *name, const char *title, RooAbsCategory& inputCat) :
  RooAbsCategory(name, title)
{
  addServer(inputCat) ;
}


RooMappedCategory::RooMappedCategory(const char *name, const RooMappedCategory& other) :
  RooAbsCategory(name,other), _defout(other._defout) 
{
  initCopy(other) ;
}



RooMappedCategory::RooMappedCategory(const RooMappedCategory& other) :
  RooAbsCategory(other), _defout(other._defout) 
{
  initCopy(other) ;
}



void RooMappedCategory::initCopy(const RooMappedCategory& other) 
{
  int i ;
  for (i=0 ; i<other._inlo.GetEntries() ; i++) {
    _inlo.Add(other._inlo.At(i)) ;
    _inhi.Add(other._inhi.At(i)) ;
    _out.Add(other._out.At(i)) ;
  }
}



RooMappedCategory::~RooMappedCategory() 
{
  _inlo.Delete() ;
  _inhi.Delete() ;
}


Bool_t RooMappedCategory::setDefault(int def) {
  const RooCatType* defType = lookupType(def,kTRUE) ;
  if (!defType) 
    return kTRUE ;

  _defout = *defType ;
  return kFALSE ;
}

Bool_t RooMappedCategory::setDefault(const char* def_key) {
  const RooCatType* defType = lookupType(def_key,kTRUE) ;
  if (!defType)
    return kTRUE ;

  _defout = *defType ;
  return kFALSE ;
}


Bool_t RooMappedCategory::mapValue(const char* in_key, int out_idx) 
{
  const RooCatType* inType  = inputCat()->lookupType(in_key,kTRUE) ;
  const RooCatType* outType = lookupType(out_idx,kTRUE) ;
  return addMap(outType,inType,inType) ;
}


Bool_t RooMappedCategory::mapValue(int in_idx, const char* out_key) {
  const RooCatType* inType  = inputCat()->lookupType(in_idx,kTRUE) ;
  const RooCatType* outType = lookupType(out_key,kTRUE) ;
  return addMap(outType,inType,inType) ;
}


Bool_t RooMappedCategory::mapValue(const char* in_key, const char* out_key) 
{
  const RooCatType* inType  = inputCat()->lookupType(in_key,kTRUE) ;
  const RooCatType* outType = lookupType(out_key,kTRUE) ;
  return addMap(outType,inType,inType) ;
}


Bool_t RooMappedCategory::mapValue(int in_idx, int out_idx) {
  const RooCatType* inType  = inputCat()->lookupType(in_idx) ;
  const RooCatType* outType = lookupType(out_idx) ;
  return addMap(outType,inType,inType) ;
}


Bool_t RooMappedCategory::mapRange(int inlo_idx, int inhi_idx, int out_idx) 
{
  const RooCatType* inTypeLo = inputCat()->lookupType(inlo_idx,kTRUE) ;
  const RooCatType* inTypeHi = inputCat()->lookupType(inhi_idx,kTRUE) ;
  const RooCatType* outType  = lookupType(out_idx,kTRUE) ;
  return addMap(outType,inTypeLo,inTypeHi) ;
}

Bool_t RooMappedCategory::mapRange(int inlo_idx, int inhi_idx, const char* out_key) 
{
  const RooCatType* inTypeLo = inputCat()->lookupType(inlo_idx,kTRUE) ;
  const RooCatType* inTypeHi = inputCat()->lookupType(inhi_idx,kTRUE) ;
  const RooCatType* outType  = lookupType(out_key,kTRUE) ;
  return addMap(outType,inTypeLo,inTypeHi) ;
}

Bool_t RooMappedCategory::mapRange(const char* inlo_key, const char* inhi_key, int out_idx) {
  const RooCatType* inTypeLo = inputCat()->lookupType(inlo_key,kTRUE) ;
  const RooCatType* inTypeHi = inputCat()->lookupType(inhi_key,kTRUE) ;
  const RooCatType* outType  = lookupType(out_idx,kTRUE) ;
  return addMap(outType,inTypeLo,inTypeHi) ;
}

Bool_t RooMappedCategory::mapRange(const char* inlo_key, const char* inhi_key, const char* out_key) {
  const RooCatType* inTypeLo = inputCat()->lookupType(inlo_key,kTRUE) ;
  const RooCatType* inTypeHi = inputCat()->lookupType(inhi_key,kTRUE) ;
  const RooCatType* outType  = lookupType(out_key,kTRUE) ;
  return addMap(outType,inTypeLo,inTypeHi) ;
}


Bool_t RooMappedCategory::addMap( const RooCatType* out, const RooCatType* inlo, const RooCatType* inhi)
{
  if (!inlo || !inhi || !out) return kTRUE ;
  _inlo.Add(new RooCatType(*inlo)) ;
  _inhi.Add(new RooCatType(*inhi)) ;
  _out.Add(new RooCatType(*out)) ;
}


RooCatType
RooMappedCategory::evaluate() const
{
  int i=0 ;
  int input = inputCat()->getIndex() ;
    for (i=0 ; i<_out.GetEntries() ; i++) {
      if (input>= ((RooCatType*)_inlo.At(i))->getVal() && 
	  input<= ((RooCatType*)_inhi.At(i))->getVal()) 
	return *((RooCatType*)_out.At(i)) ;
    }  
  return _defout ;
}


void RooMappedCategory::printToStream(ostream& os, PrintOption opt) const
{
  if (opt==Shape) {
    cout << "RooMappedCategory: input category:" << endl ;
    inputCat()->printToStream(os,Shape) ;

    os << "RooMappedCategory: value mapping:" << endl ;

    int i ; for (i=0 ; i<_out.GetEntries() ; i++) {
      RooCatType* inlo = (RooCatType*) _inlo.At(i) ;
      RooCatType* inhi = (RooCatType*) _inhi.At(i) ;
      RooCatType* out  = (RooCatType*) _out.At(i) ;

      if (*inlo==*inhi) {
	os << "   " << inlo->GetName() << " -> " << out->GetName() 
	   << " (" << out->getVal() << ")" << endl ;
      } else {
	os << "   (" << inlo->GetName() << " - " << inhi->GetName() 
	   << ") -> " << out->GetName() << " (" << out->getVal() << ")" << endl ;
      }
    }
    if (!TString(_defout.GetName()).IsNull()) {
      os << "   Default -> " << _defout.GetName() << " (" << _defout.getVal() << ")" << endl ;
    }
  } else {
    os << "RooMappedCategory: " << GetName() << " = " << inputCat()->GetName() 
       << ":" << inputCat()->getLabel() << "(" << inputCat()->getIndex() 
       << ") -> " << getLabel() << "(" << getIndex() << ")" ;
    os << " : \"" << fTitle << "\"" ;

    printAttribList(os) ;
    os << endl ;
  }
}


Bool_t RooMappedCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  if (compact) {
    cout << "RooMappedCategory::readFromSteam(" << GetName() << "): can't read in compact mode" << endl ;
    return kTRUE ;    
  } else {
    TString token,errorPrefix("RooMappedCategory::readFromStream(") ;
    errorPrefix.Append(GetName()) ;
    errorPrefix.Append(")") ;
    RooStreamParser parser(is,errorPrefix) ;
    
    TString destKey,srcKey1,srcKey2 ;
    Bool_t readToken(kTRUE) ;

    // Loop over definition sequences
    while(1) {      
      if (readToken) token=parser.readToken() ;
      if (token.IsNull()) break ;
      readToken=kTRUE ;

      destKey = token ;
      if (parser.expectToken(":",kTRUE)) return kTRUE ;

      // Loop over list of sources for this destination
      while(1) { 
	srcKey1 = parser.readToken() ;	
	token = parser.readToken() ;
	if (!token.CompareTo("-")) {	  
	  // Map a range
	  srcKey2 = parser.readToken() ;
	  if (mapRange(srcKey1,srcKey2,destKey)) return kTRUE ;
	  token = parser.readToken() ;
	} else {
	  if (!srcKey1.CompareTo("*")) {
	    // Set the default destination
	    if (setDefault(destKey)) return kTRUE ;
	  } else {
	    // Map a value
	    if (mapValue(srcKey1,destKey)) return kTRUE ;
	  }
	}

	// Unless next token is ',' current token 
        // is destination part of next sequence
	if (token.CompareTo(",")) {	  	  
	  readToken = kFALSE ;
	  break ;
	} 	
      }      
    } 
    return kFALSE ;
  }
}



void RooMappedCategory::writeToStream(ostream& os, Bool_t compact) const
{
  if (compact) {
    cout << "RooMappedCategory::writeToStream(" << GetName() << "): can't write in compact mode" << endl ;    
  } else {
    int i ;
    RooCatType lastOut("",999999) ;
    Bool_t first(kTRUE) ;
    for (i=0 ; i<_inlo.GetEntries() ; i++) {
      RooCatType* inlo = (RooCatType*) _inlo.At(i) ;
      RooCatType* inhi = (RooCatType*) _inhi.At(i) ;
      RooCatType* out = (RooCatType*) _out.At(i) ;

      if (*out != lastOut) {
	cout << (first?"":" ") << out->GetName() << ":" ;
	first=kFALSE ;
      } else {
	cout << "," ;
      }
      cout << inlo->GetName() ;
      if (*inhi != *inlo) cout << "-" << inhi->GetName() ;
      lastOut = *out ;
    }
    if (!TString(_defout.GetName()).IsNull()) {
      cout << " " << _defout.GetName() << ":*" ;
    }
  }
}
