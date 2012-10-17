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
// RooArgList is a container object that can hold multiple RooAbsArg objects.
// The container has list semantics which means that:
//
//  - Contained objects are ordered, The iterator 
//    follows the object insertion order.
//
//  - Objects can be retrieved by name and index
//
//  - Multiple objects with the same name are allowed
//
// Ownership of contents. 
//
// Unowned objects are inserted with the add() method. Owned objects
// are added with addOwned() or addClone(). A RooArgSet either owns all 
// of it contents, or none, which is determined by the first <add>
// call. Once an ownership status is selected, inappropriate <add> calls
// will return error status. Clearing the list via removeAll() resets the 
// ownership status. Arguments supplied in the constructor are always added 
// as unowned elements.
//
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <iomanip>
#include <fstream>
#include "TClass.h"
#include "RooArgList.h"
#include "RooStreamParser.h"
#include "RooFormula.h"
#include "RooAbsRealLValue.h"
#include "RooAbsCategoryLValue.h"
#include "RooStringVar.h"
#include "RooTrace.h"
#include "RooMsgService.h"

ClassImp(RooArgList)
  ;


//_____________________________________________________________________________
RooArgList::RooArgList() :
  RooAbsCollection()
{
  // Default constructor
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooArgSet& set) :
  RooAbsCollection(set.GetName())
{
  // Constructor from a RooArgSet. 

  add(set) ;
}



//_____________________________________________________________________________
RooArgList::RooArgList(const char *name) :
  RooAbsCollection(name)
{
  // Empty list constructor
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooAbsArg& var1,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for list containing 1 initial object

  add(var1);
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 2 initial objects

  add(var1); add(var2);
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 3 initial objects

  add(var1); add(var2); add(var3);
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 4 initial objects

  add(var1); add(var2); add(var3); add(var4);
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooAbsArg& var1,
		     const RooAbsArg& var2, const RooAbsArg& var3,
		     const RooAbsArg& var4, const RooAbsArg& var5,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 5 initial objects

  add(var1); add(var2); add(var3); add(var4); add(var5);
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 6 initial objects

  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6);
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 7 initial objects

  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7, const RooAbsArg& var8,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 8 initial objects

  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7) ;add(var8) ;
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6, 
		     const RooAbsArg& var7, const RooAbsArg& var8,
		     const RooAbsArg& var9, const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 9 initial objects

  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6); add(var7); add(var8); add(var9);
}



//_____________________________________________________________________________
RooArgList::RooArgList(const TCollection& tcoll, const char* name) :
  RooAbsCollection(name)
{
  // Constructor from a root TCollection. Elements in the collection that
  // do not inherit from RooAbsArg will be skipped. A warning message
  // will be printed for every skipped item.

  TIterator* iter = tcoll.MakeIterator() ;
  TObject* obj ;
  while((obj=iter->Next())) {
    if (!dynamic_cast<RooAbsArg*>(obj)) {
      coutW(InputArguments) << "RooArgList::RooArgList(TCollection) element " << obj->GetName() 
			    << " is not a RooAbsArg, ignored" << endl ;
      continue ;
    }
    add(*(RooAbsArg*)obj) ;
  }
  delete iter ;
}



//_____________________________________________________________________________
RooArgList::RooArgList(const RooArgList& other, const char *name) 
  : RooAbsCollection(other,name)
{
  // Copy constructor. Note that a copy of a list is always non-owning,
  // even the source list is owning. To create an owning copy of
  // a list (owning or not), use the snaphot() method.
}



//_____________________________________________________________________________
RooArgList::~RooArgList() 
{
  // Destructor
}



//_____________________________________________________________________________
RooAbsArg& RooArgList::operator[](Int_t idx) const 
{     
  // Array operator. Element in slot 'idx' must already exist, otherwise
  // code will abort. 
  //
  // When used as lvalue in assignment operations, the element contained in
  // the list will not be changed, only the value of the existing element!

  RooAbsArg* arg = at(idx) ;
  if (!arg) {
    coutE(InputArguments) << "RooArgList::operator[](" << GetName() << ") ERROR: index " 
			  << idx << " out of range (0," << getSize() << ")" << endl ;
    RooErrorHandler::softAbort() ;
  }
  return *arg ; 
}



//_____________________________________________________________________________
void RooArgList::writeToStream(ostream& os, Bool_t compact) 
{
  // Write the contents of the argset in ASCII form to given stream.
  // 
  // All elements will be printed on a single line separated by a single 
  // white space. The contents of each element is written by the arguments' 
  // writeToStream() function

  if (!compact) {
    coutE(InputArguments) << "RooArgList::writeToStream(" << GetName() << ") non-compact mode not supported" << endl ;
    return ;
  }

  TIterator *iterat= createIterator();
  RooAbsArg *next = 0;
  while((0 != (next= (RooAbsArg*)iterat->Next()))) {
      next->writeToStream(os,kTRUE) ;
      os << " " ;
  }
  delete iterat;  
  os << endl ;
}



//_____________________________________________________________________________
Bool_t RooArgList::readFromStream(istream& is, Bool_t compact, Bool_t verbose) 
{
  // Read the contents of the argset in ASCII form from given stream.
  // 
  // A single line is read, and all elements are assumed to be separated 
  // by white space. The value of each argument is read by the arguments 
  // readFromStream function.

  if (!compact) {
    coutE(InputArguments) << "RooArgList::readFromStream(" << GetName() << ") non-compact mode not supported" << endl ;
    return kTRUE ;
  }    

  TIterator *iterat= createIterator();
  RooStreamParser parser(is) ;
  RooAbsArg *next = 0;
  while((0 != (next= (RooAbsArg*)iterat->Next()))) {
    if (!next->getAttribute("Dynamic")) {
      if (next->readFromStream(is,kTRUE,verbose)) {
	parser.zapToEnd() ;
	
	delete iterat ;
	return kTRUE ;
      }	
    } else {
    }
  }
  
  if (!parser.atEOL()) {
    TString rest = parser.readLine() ;
    if (verbose) {
      coutW(InputArguments) << "RooArgSet::readFromStream(" << GetName() 
			    << "): ignoring extra characters at end of line: '" << rest << "'" << endl ;
    }
  }
  
  delete iterat;    
  return kFALSE ;  
}

