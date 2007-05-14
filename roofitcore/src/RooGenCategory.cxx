/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Name:  $:$Id: RooGenCategory.cxx,v 1.24 2007/05/11 09:11:58 verkerke Exp $
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
// RooGenCategory provides the most flexibe mapping of a series of input categories
// on a output category via a global function provided in the constructor
//
// The mapping function must have the form 'const char* mapFunc(const RooArgSet* catList)'
// and return the name of the output state for the list of categories supplied in the argument.
// The global function can be a ROOT interpreted function.
//
// RooGenCategory builds a numerical index-to-index map from the user function
// to achieve a high performance mapping.

#include "RooFit.h"

#include "Riostream.h"
#include "TMethodCall.h"
#include <stdlib.h>
#include <stdio.h>
#include "TString.h"
#include "Api.h"  
#include "RooGenCategory.h"
#include "RooStreamParser.h"
#include "RooMapCatEntry.h"
#include "RooErrorHandler.h"

ClassImp(RooGenCategory)



RooGenCategory::RooGenCategory(const char *name, const char *title, void *userFunc, RooArgSet& catList) :
  RooAbsCategory(name, title), 
  _superCat("superCat","Super Category",catList), 
  _superCatProxy("superCatProxy","Super Category Proxy",this,_superCat), 
  _map(0) 
{
  // Constructor with pointer to user mapping function and list of input categories
  
  // Convert the function pointer into a parse object using the CINT
  // dictionary in memory.
  _userFuncName = G__p2f2funcname(userFunc);
  if(_userFuncName.IsNull()) {
    cout << GetName() << ": cannot find dictionary info for (void*)"
         << (void*)userFunc << endl;
    return;
  }
  initialize() ;
}


RooGenCategory::RooGenCategory(const RooGenCategory& other, const char *name) :
  RooAbsCategory(other,name), _superCat(other._superCat), 
  _superCatProxy("superCatProxy","Super Category Proxy",this,_superCat),
  _map(0), _userFuncName(other._userFuncName)
{
  // Copy constructor
  removeServer((RooAbsArg&)other._superCat) ;
  initialize() ;
}




void RooGenCategory::initialize()
{
  // Initialization function

  // This is a static link, no need for redirection support
  addServer(_superCat,kTRUE,kTRUE) ;

  _userFunc= new TMethodCall();
  // We must use "RooArgSet*" instead of "RooArgSet&" here (why?)
  _userFunc->InitWithPrototype(_userFuncName.Data(),"RooArgSet*"); 

  updateIndexList() ;
}



RooGenCategory::~RooGenCategory() 
{
  // Destructor

  // Server no longer exists when RooAbsArg destructor is executing  
  if (_serverList.FindObject(&_superCat)) {
    removeServer(_superCat) ;
  }

  if (_map) delete[] _map ;
}


TString RooGenCategory::evalUserFunc(RooArgSet *vars) 
{
  // Utility function to evaluate (interpreted) user function
  assert(0 != _userFunc);
  Long_t result;
  _userArgs[0]= (Long_t)vars ;
  _userFunc->SetParamPtrs(_userArgs);
  _userFunc->Execute(result);
  const char *text= (const char *)result;
  return TString(text);
} 



void RooGenCategory::updateIndexList()
{
  // Update list of states and reevaluate input index code to output index code map

  // Recreate super-index to gen-index map ;
  if (_map) delete[] _map ;
  _map = new Int_t[_superCatProxy.arg().numTypes()] ;
  clearTypes() ;

  // DeepClone super category for iteration
  RooArgSet* tmp=(RooArgSet*) RooArgSet(_superCatProxy.arg()).snapshot(kTRUE) ;
  if (!tmp) {
    cout << "RooGenCategory::updateIndexList(" << GetName() << ") Couldn't deep-clone super category, abort," << endl ;
    RooErrorHandler::softAbort() ;
  }
  RooSuperCategory* superClone = (RooSuperCategory*) tmp->find(_superCatProxy.arg().GetName()) ;

  TIterator* sIter = superClone->typeIterator() ;
  RooArgSet *catList = superClone->getParameters((const RooArgSet*)0) ;
  RooCatType* type ;
  while ((type=(RooCatType*)sIter->Next())) {
    // Call user function
    superClone->setIndex(type->getVal()) ;

    TString typeName = evalUserFunc(catList) ;

    // Check if type exists for given name, register otherwise
    const RooCatType* type = lookupType(typeName,kFALSE) ;
    if (!type) type = defineType(typeName) ;

    // Fill map for this super-state
    _map[superClone->getIndex()] = type->getVal() ;
    //cout << "updateIndexList(" << GetName() << ") _map[" << superClone->getLabel() << "] = " << type->GetName() << endl ;
  }

  delete tmp ;
  delete catList ;
}


RooCatType
RooGenCategory::evaluate() const
{
  // Calculate current value of object
  if (isShapeDirty()) {
    const_cast<RooGenCategory*>(this)->updateIndexList() ;
  }

  const RooCatType* ret = lookupType(_map[(Int_t)_superCatProxy]) ;
  if (!ret) {
    cout << "RooGenCategory::evaluate(" << GetName() << ") ERROR: cannot lookup super index " << (Int_t) _superCatProxy << endl ;
    assert(0) ;
  }

  return *ret ;
}



void RooGenCategory::printToStream(ostream& os, PrintOption opt, TString indent) const
{
  // Print info about this mapped category to the specified stream. In addition to the info
  // from RooAbsCategory::printToStream() we add:
  //
  //  Standard : input category
  //     Shape : default value
  //   Verbose : list of mapping rules

   RooAbsCategory::printToStream(os,opt,indent);

   if (opt>=Verbose) {     
     os << indent << "--- RooGenCategory ---" << endl;
     os << indent << "  Input category list:" << endl ;
     TString moreIndent(indent) ;
     indent.Append("   ") ;
     ((RooSuperCategory&)_superCatProxy.arg()).inputCatList().printToStream(os,OneLine) ;
     os << indent << "  User mapping function is 'const char* " << _userFuncName << "(RooArgSet*)'" << endl ;
   }
}


Bool_t RooGenCategory::readFromStream(istream& /*is*/, Bool_t compact, Bool_t /*verbose*/) 
{
  // Read object contents from given stream
   if (compact) {
     cout << "RooGenCategory::readFromSteam(" << GetName() << "): can't read in compact mode" << endl ;
     return kTRUE ;    
   } else {
     return kFALSE ;
   }
   //return kFALSE ; //OSF: statement unreachable
}



void RooGenCategory::writeToStream(ostream& os, Bool_t compact) const
{
  // Write object contents to given stream 
  if (compact) {
    // Write value only
    os << getLabel() ;
  } else {
  }
}
