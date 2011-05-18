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

// -- CLASS DESCRIPTION [CAT] --
// RooMappedCategory provides a category-to-category mapping defined
// by pattern matching on their state labels
//
// The mapping function consists of a series of wild card regular expressions.
// Each expression is matched to the input categories state labels, and an associated
// output state label.


#include "RooFit.h"

#include "Riostream.h"
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooMappedCategory.h"
#include "RooStreamParser.h"
#include "RooMsgService.h"

using namespace std ;

ClassImp(RooMappedCategory)
ClassImp(RooMappedCategory::Entry)


RooMappedCategory::RooMappedCategory(const char *name, const char *title, RooAbsCategory& inputCat, const char* defOut, Int_t defOutIdx) :
  RooAbsCategory(name, title), _inputCat("input","Input category",this,inputCat)
{
  // Constructor with input category and name of default output state, which is assigned
  // to all input category states that do not follow any mapping rule.
  if (defOutIdx==NoCatIdx) {
    _defCat = (RooCatType*) defineType(defOut) ;
  } else {
    _defCat = (RooCatType*) defineType(defOut,defOutIdx) ;
  }
}


RooMappedCategory::RooMappedCategory(const RooMappedCategory& other, const char *name) :
  RooAbsCategory(other,name), _inputCat("input",this,other._inputCat), _mapArray(other._mapArray)
{
  _defCat = (RooCatType*) lookupType(other._defCat->GetName()) ;
}



RooMappedCategory::~RooMappedCategory() 
{
  // Destructor
}



Bool_t RooMappedCategory::map(const char* inKeyRegExp, const char* outKey, Int_t outIdx)
{
  // Add mapping rule: any input category state label matching the 'inKeyRegExp'
  // wildcard expression will be mapped to an output state with name 'outKey'
  //
  // Rules are evaluated in the order they were added. In case an input state
  // matches more than one rule, the first rules output state will be assigned

  if (!inKeyRegExp || !outKey) return kTRUE ;  

  // Check if pattern is already registered
  if (_mapArray.find(inKeyRegExp)!=_mapArray.end()) {
    coutE(InputArguments) << "RooMappedCategory::map(" << GetName() << "): ERROR expression " 
			  << inKeyRegExp << " already mapped" << endl ;
    return kTRUE ;
  }

  // Check if output type exists, if not register
  const RooCatType* outType = lookupType(outKey) ;
  if (!outType) {
    if (outIdx==NoCatIdx) {
      outType = defineType(outKey) ;
    } else {
      outType = defineType(outKey,outIdx) ;
    }
  }
  if (!outType) {
    coutE(InputArguments) << "RooMappedCategory::map(" << GetName() 
			  << "): ERROR, unable to output type " << outKey << endl ;
    return kTRUE ;    
  }

  // Create new map entry ;
  Entry e(inKeyRegExp,outType) ;
  if (!e.ok()) {
    coutE(InputArguments) << "RooMappedCategory::map(" << GetName() 
			  << "): ERROR, expression " << inKeyRegExp << " didn't compile" << endl ;
    return kTRUE ;    
  }

  _mapArray[inKeyRegExp] = e ;
  return kFALSE ;
}



RooCatType RooMappedCategory::evaluate() const
{
  // Calculate the current value of the object
  const char* inKey = _inputCat.label() ;

  // Scan array of regexps
  for ( std::map<string,Entry>::const_iterator iter = _mapArray.begin() ; iter != _mapArray.end() ; iter++) {
    if (iter->second.match(inKey)) {
      return iter->second.outCat() ;
    }
  }

  // Return default if nothing found
  return *_defCat ;
}


void RooMappedCategory::printMultiline(ostream& os, Int_t content, Bool_t verbose, TString indent) const
{
  // Print info about this mapped category to the specified stream. In addition to the info
  // from RooAbsCategory::printStream() we add:
  //
  //  Standard : input category
  //     Shape : default value
  //   Verbose : list of mapping rules

  RooAbsCategory::printMultiline(os,content,verbose,indent);

  if (verbose) {
    os << indent << "--- RooMappedCategory ---" << endl
       << indent << "  Maps from " ;
    _inputCat.arg().printStream(os,0,kStandard);
    
    os << indent << "  Default value is ";
    _defCat->printStream(os,kName|kValue,kSingleLine);
    
    os << indent << "  Mapping rules:" << endl;
    for (std::map<string,Entry>::const_iterator iter = _mapArray.begin() ; iter!=_mapArray.end() ; iter++) {
      os << indent << "  " << iter->first << " -> " << iter->second.outCat().GetName() << endl ;
    }
  }
}


Bool_t RooMappedCategory::readFromStream(istream& is, Bool_t compact, Bool_t /*verbose*/) 
{
  // Read object contents from given stream
   if (compact) {
     coutE(InputArguments) << "RooMappedCategory::readFromSteam(" << GetName() << "): can't read in compact mode" << endl ;
     return kTRUE ;    
   } else {

     //Clear existing definitions, but preserve default output
     TString defCatName(_defCat->GetName()) ;
     _mapArray.clear() ;
     clearTypes() ;
     _defCat = (RooCatType*) defineType(defCatName) ;

     TString token,errorPrefix("RooMappedCategory::readFromStream(") ;
     errorPrefix.Append(GetName()) ;
     errorPrefix.Append(")") ;
     RooStreamParser parser(is,errorPrefix) ;
     parser.setPunctuation(":,") ;
  
     TString destKey,srcKey ;
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
	 srcKey = parser.readToken() ;	
	 token = parser.readToken() ;

	 // Map a value
	 if (map(srcKey,destKey)) return kTRUE ;
       
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
   //return kFALSE ; // statement unreachable (OSF)
}


//_____________________________________________________________________________
void RooMappedCategory::printMetaArgs(ostream& os) const 
{
  // Customized printing of arguments of a RooMappedCategory to more intuitively reflect the contents of the
  // product operator construction

  // Scan array of regexps
  RooCatType prevOutCat ;
  Bool_t first(kTRUE) ;
  os << "map=(" ;
  for (std::map<string,Entry>::const_iterator iter = _mapArray.begin() ; iter!=_mapArray.end() ; iter++) {
    if (iter->second.outCat().getVal()!=prevOutCat.getVal()) {
      if (!first) { os << " " ; }
      first=kFALSE ;
      
      os << iter->second.outCat().GetName() << ":" << iter->first ;
      prevOutCat=iter->second.outCat() ;
    } else {
      os << "," << iter->first ;
    }
  }
  
  if (!first) { os << " " ; }
  os << _defCat->GetName() << ":*" ;  
  
  os << ") " ;    
}




void RooMappedCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream
  if (compact) {
    // Write value only
    os << getLabel() ;
  } else {
    // Write mapping expression

    // Scan array of regexps
    RooCatType prevOutCat ;
    Bool_t first(kTRUE) ;
    for (std::map<string,Entry>::const_iterator iter = _mapArray.begin() ; iter!=_mapArray.end() ; iter++) {
      if (iter->second.outCat().getVal()!=prevOutCat.getVal()) {
	if (!first) { os << " " ; }
	first=kFALSE ;

	os << iter->second.outCat().GetName() << "<-" << iter->first ;
	prevOutCat=iter->second.outCat() ;
      } else {
	os << "," << iter->first ;
      }
    }
    
    if (!first) { os << " " ; }
    os << _defCat->GetName() << ":*" ;  
  }
}




//_____________________________________________________________________________
RooMappedCategory::Entry& RooMappedCategory::Entry::operator=(const RooMappedCategory::Entry& other)
{
  if (&other==this) return *this ;

  _expr = other._expr ;
  _cat = other._cat ;

  if (_regexp) {
    delete _regexp ;
  }
  _regexp = new TRegexp(_expr.Data(),kTRUE) ;

  return *this;  
}



//_____________________________________________________________________________
TString RooMappedCategory::Entry::mangle(const char* exp) const
{
  // Mangle name : escape regexp character '+'
  TString t ;
  const char *c = exp ;
  while(*c) {
    if (*c=='+') t.Append('\\') ;
    t.Append(*c) ;
    c++ ;
  }
  return t ;
}



//_____________________________________________________________________________
void RooMappedCategory::Entry::Streamer(TBuffer &R__b)
{
  typedef ::RooMappedCategory::Entry ThisClass;

   // Stream an object of class RooWorkspace::CodeRepo.
   if (R__b.IsReading()) {

     UInt_t R__s, R__c;
     R__b.ReadVersion(&R__s, &R__c); 
     
     // Stream contents of ClassFiles map     
     R__b >> _expr ;
     _cat.Streamer(R__b) ;     
     _regexp = new TRegexp(_expr.Data(),kTRUE) ;
     R__b.CheckByteCount(R__s, R__c, ThisClass::IsA());

   } else {
     
     UInt_t R__c;
     R__c = R__b.WriteVersion(ThisClass::IsA(), kTRUE);
     
     // Stream contents of ClassRelInfo map
     R__b << _expr ;
     _cat.Streamer(R__b) ;

     R__b.SetByteCount(R__c, kTRUE);
     
   }
}
