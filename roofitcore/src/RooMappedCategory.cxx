/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id$
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


ClassImp(RooMappedCategory)


RooMappedCategory::RooMappedCategory(const char *name, const char *title, RooAbsIndex& inputCat) :
  RooAbsIndex(name, title)
{
  addServer(inputCat) ;
}


RooMappedCategory::~RooMappedCategory() 
{
  _inlo.Delete() ;
  _inhi.Delete() ;
}


Bool_t RooMappedCategory::setDefault(int def) {
  const RooCat* defType = lookupType(def,kTRUE) ;
  if (!defType) 
    return kTRUE ;

  _defout = *defType ;
  return kFALSE ;
}

Bool_t RooMappedCategory::setDefault(char* def_key) {
  const RooCat* defType = lookupType(def_key) ;
  if (!defType)
    return kTRUE ;

  _defout = *defType ;
  return kFALSE ;
}


Bool_t RooMappedCategory::mapValue(char* in_key, int out_idx) 
{
  const RooCat* inType  = inputCat()->lookupType(in_key,kTRUE) ;
  const RooCat* outType = lookupType(out_idx,kTRUE) ;
  return addMap(outType,inType,inType) ;
}


Bool_t RooMappedCategory::mapValue(int in_idx, char* out_key) {
  const RooCat* inType  = inputCat()->lookupType(in_idx,kTRUE) ;
  const RooCat* outType = lookupType(out_key,kTRUE) ;
  return addMap(outType,inType,inType) ;
}


Bool_t RooMappedCategory::mapValue(char* in_key, char* out_key) 
{
  const RooCat* inType  = inputCat()->lookupType(in_key,kTRUE) ;
  const RooCat* outType = lookupType(out_key,kTRUE) ;
  return addMap(outType,inType,inType) ;
}


Bool_t RooMappedCategory::mapValue(int in_idx, int out_idx) {
  const RooCat* inType  = inputCat()->lookupType(in_idx) ;
  const RooCat* outType = lookupType(out_idx) ;
  return addMap(outType,inType,inType) ;
}


Bool_t RooMappedCategory::mapRange(int inlo_idx, int inhi_idx, int out_idx) 
{
  const RooCat* inTypeLo = inputCat()->lookupType(inlo_idx,kTRUE) ;
  const RooCat* inTypeHi = inputCat()->lookupType(inhi_idx,kTRUE) ;
  const RooCat* outType  = lookupType(out_idx,kTRUE) ;
  return addMap(outType,inTypeLo,inTypeHi) ;
}

Bool_t RooMappedCategory::mapRange(int inlo_idx, int inhi_idx, char* out_key) 
{
  const RooCat* inTypeLo = inputCat()->lookupType(inlo_idx,kTRUE) ;
  const RooCat* inTypeHi = inputCat()->lookupType(inhi_idx,kTRUE) ;
  const RooCat* outType  = lookupType(out_key,kTRUE) ;
  return addMap(outType,inTypeLo,inTypeHi) ;
}

Bool_t RooMappedCategory::mapRange(char* inlo_key, char* inhi_key, int out_idx) {
  const RooCat* inTypeLo = inputCat()->lookupType(inlo_key,kTRUE) ;
  const RooCat* inTypeHi = inputCat()->lookupType(inhi_key,kTRUE) ;
  const RooCat* outType  = lookupType(out_idx,kTRUE) ;
  return addMap(outType,inTypeLo,inTypeHi) ;
}

Bool_t RooMappedCategory::mapRange(char* inlo_key, char* inhi_key, char* out_key) {
  const RooCat* inTypeLo = inputCat()->lookupType(inlo_key,kTRUE) ;
  const RooCat* inTypeHi = inputCat()->lookupType(inhi_key,kTRUE) ;
  const RooCat* outType  = lookupType(out_key,kTRUE) ;
  return addMap(outType,inTypeLo,inTypeHi) ;
}


Bool_t RooMappedCategory::addMap( const RooCat* out, const RooCat* inlo, const RooCat* inhi)
{
  if (!inlo || !inhi || !out) return kTRUE ;
  _inlo.Add(new RooCat(*inlo)) ;
  _inhi.Add(new RooCat(*inhi)) ;
  _out.Add(new RooCat(*out)) ;
}


RooCat
RooMappedCategory::evaluate() 
{
  int i=0 ;
  int input = inputCat()->getIndex() ;
    for (i=0 ; i<_out.GetEntries() ; i++) {
      if (input>= ((RooCat*)_inlo.At(i))->getVal() && 
	  input<= ((RooCat*)_inhi.At(i))->getVal()) 
	return *((RooCat*)_out.At(i)) ;
    }  
  return _defout ;
}


void RooMappedCategory::printToStream(ostream& os, PrintOption opt) 
{
  if (opt==Shape) {
    cout << "RooMappedCategory: input category:" << endl ;
    inputCat()->printToStream(os,Shape) ;

    os << "RooMappedCategory: value mapping:" << endl ;

    int i ; for (i=0 ; i<_out.GetEntries() ; i++) {
      RooCat* inlo = (RooCat*) _inlo.At(i) ;
      RooCat* inhi = (RooCat*) _inhi.At(i) ;
      RooCat* out  = (RooCat*) _out.At(i) ;

      if (*inlo==*inhi) {
	os << "   " << inlo->GetName() << " -> " << out->GetName() 
	   << " (" << out->getVal() << ")" << endl ;
      } else {
	os << "   (" << inlo->GetName() << " - " << inhi->GetName() 
	   << ") -> " << out->GetName() << " (" << out->getVal() << ")" << endl ;
      }
    }
    os << "   Default -> " << _defout.GetName() << " (" << _defout.getVal() << ")" << endl ;
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
    cout << "RooMappedCategory::readFromSteam(" << GetName() << "): Compact mode not implemented" << endl ;
    return kTRUE ;    
  }

  // Read a line
  char line[1024] ;
  is.getline(line,sizeof(line)) ;
  if(!is.good() || is.eof()) return false;

  cout << "line = " << line << endl ;

  char *token = strtok(line," ") ;
  if(TString(token).CompareTo("=")) {
    cout << "RooMappedCategory::readFromStream: parse error, expecing '='" << endl;
    return false ;
  }

  // Decode mapping sequences
  while(token = strtok(0," ")) {
    cout << "token = |" << token << "|" << endl ; 
   
    // Find colon
    char *ptr = strchr(token,':') ;
    if (!ptr) {
      cout << "RooMappedCategory::readFromStream: ignoring invalid token '" << token << "'" << endl;
      continue ;
    }

    // Isolate output word & move on to input list
    *ptr=0 ; ptr++ ;
    char *ptr2 ;
    while(true) {
      ptr2 = strchr(ptr,',') ; 
      if (ptr2) *ptr2 = 0 ;

      // Range or single number or default?
      char *psep ;
      if (*ptr=='*') {
	setDefault(token) ;
      } else if (psep=strchr(ptr,'-')) {
	*psep=0 ;
	mapRange(ptr,psep+1,token) ;
      } else {
	mapValue(ptr,token) ;
      }

      if (!ptr2) break ;
      ptr = ptr2+1 ;      
    } 
  }
  
  return true ;
}



void RooMappedCategory::writeToStream(ostream& os, Bool_t compact) 
{
  os << getIndex() << endl ;
}
