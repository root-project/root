/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitTools
 *    File: $Id: RooMappedCategory.cc,v 1.14 2001/08/02 22:36:30 verkerke Exp $
 * Authors:
 *   WV, Wouter Verkerke, UCSB, verkerke@slac.stanford.edu
 * History:
 *   01-Mar-2001 WV Create initial version
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooMappedCategory provides a category-to-category mapping defined
// by pattern matching.
//
// The mapping function consists of a series of wild card regular expressions,
// which are matched to the input categories state labels, and an associated
// output state label.


#include <iostream.h>
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "RooFitCore/RooMappedCategory.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooMapCatEntry.hh"

ClassImp(RooMappedCategory)


RooMappedCategory::RooMappedCategory(const char *name, const char *title, RooAbsCategory& inputCat, const char* defOut) :
  RooAbsCategory(name, title), _inputCat("inputCat","Input category",this,inputCat)
{
  // Constructor with input category and name of default (unmapped) output state
  _defCat = (RooCatType*) defineType(defOut) ;
}


RooMappedCategory::RooMappedCategory(const RooMappedCategory& other, const char *name) :
  RooAbsCategory(other,name), _inputCat("inputCat",this,other._inputCat)
{
  _defCat = (RooCatType*) lookupType(other._defCat->GetName()) ;

  // Copy constructor
  int i ;
  for (i=0 ; i<other._mapArray.GetEntries() ; i++) {
    _mapArray.Add(new RooMapCatEntry(*(RooMapCatEntry*)other._mapArray.At(i))) ;
  }
}



RooMappedCategory::~RooMappedCategory() 
{
  // Destructor
  _mapArray.Delete() ;
}



Bool_t RooMappedCategory::map(const char* inKeyRegExp, const char* outKey)
{
  // Map input state names matching given regular expression to given output state name

  if (!inKeyRegExp || !outKey) return kTRUE ;  

  // Check if pattern is already registered
  if (_mapArray.FindObject(inKeyRegExp)) {
    cout << "RooMappedCategory::map(" << GetName() << "): ERROR expression " 
	 << inKeyRegExp << " already mapped" << endl ;
    return kTRUE ;
  }

  // Check if output type exists, if not register
  const RooCatType* outType = lookupType(outKey) ;
  if (!outType) outType = defineType(outKey) ;
  if (!outType) {
    cout << "RooMappedCategory::map(" << GetName() 
	 << "): ERROR, unable to output type " << outKey << endl ;
    return kTRUE ;    
  }

  // Create new map entry ;
  RooMapCatEntry *newMap = new RooMapCatEntry(inKeyRegExp,outType) ;
  if (!newMap->ok()) {
    cout << "RooMappedCategory::map(" << GetName() 
	 << "): ERROR, expression " << inKeyRegExp << " didn't compile" << endl ;
    delete newMap ;
    return kTRUE ;    
  }

  _mapArray.Add(newMap) ;  
  return kFALSE ;
}



RooCatType
RooMappedCategory::evaluate() const
{
  // Calculate the current value of the object
  const char* inKey = _inputCat ;

  // Scan array of regexps
  for (int i=0 ; i<_mapArray.GetEntries() ; i++) {
    RooMapCatEntry* map = (RooMapCatEntry*)_mapArray.At(i) ;
    if (map->match(inKey)) {
      return map->outCat() ;
    }
  }

  // Return default if nothing found
  return *_defCat ;
}


void RooMappedCategory::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this mapped category to the specified stream. In addition to the info
  // from RooAbsCategory::printToStream() we add:
  //
  //  Standard : input category
  //     Shape : default value
  //   Verbose : list of mapping rules

   RooAbsCategory::printToStream(os,opt,indent);

   if (opt > Standard) {
     os << indent << "--- RooMappedCategory ---" << endl
	<< indent << "  Maps from " ;
     _inputCat.arg().printToStream(os,Standard);
     
     os << indent << "  Default value is ";
     _defCat->printToStream(os,OneLine);

     os << indent << "  Mapping rules:" << endl;
     Int_t n= _mapArray.GetEntries();
     for(Int_t i= 0 ; i< n; i++) {
       RooMapCatEntry* map = (RooMapCatEntry*)_mapArray.At(i) ;
       os << indent << "  " << map->GetName() << " -> " << map->outCat().GetName() << endl ;
     }
   }
}


Bool_t RooMappedCategory::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read object contents from given stream
   if (compact) {
     cout << "RooMappedCategory::readFromSteam(" << GetName() << "): can't read in compact mode" << endl ;
     return kTRUE ;    
   } else {

     //Clear existing definitions, but preserve default output
     TString defCatName(_defCat->GetName()) ;
     _mapArray.Delete() ;
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
    for (int i=0 ; i<_mapArray.GetEntries() ; i++) {
      RooMapCatEntry* map = (RooMapCatEntry*)_mapArray.At(i) ;
      if (map->outCat().getVal()!=prevOutCat.getVal()) {
	if (!first) { os << " " ; }
	first=kFALSE ;

	os << map->outCat().GetName() << ":" << map->GetName() ;
	prevOutCat=map->outCat() ;
      } else {
	os << "," << map->GetName() ;
      }
    }
    
    if (!first) { os << " " ; }
    os << _defCat->GetName() << ":*" ;  
  }
}
