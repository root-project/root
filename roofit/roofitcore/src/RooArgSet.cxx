/***************************************************************************** * Project: RooFit                                                           *
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
// RooArgSet is a container object that can hold multiple RooAbsArg objects.
// The container has set semantics which means that:
//
//  - Every object it contains must have a unique name returned by GetName().
//
//  - Contained objects are not ordered, although the set can be traversed
//    using an iterator returned by createIterator(). The iterator does not
//    necessarily follow the object insertion order.
//
//  - Objects can be retrieved by name only, and not by index.
//
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
#include <iomanip>
#include <fstream>
#include <list>
#include "TClass.h"
#include "RooArgSet.h"
#include "RooStreamParser.h"
#include "RooFormula.h"
#include "RooAbsRealLValue.h"
#include "RooAbsCategoryLValue.h"
#include "RooStringVar.h"
#include "RooTrace.h"
#include "RooArgList.h"
#include "RooSentinel.h"
#include "RooMsgService.h"

using namespace std ;

#if (__GNUC__==3&&__GNUC_MINOR__==2&&__GNUC_PATCHLEVEL__==3)
char* operator+( streampos&, char* );
#endif

ClassImp(RooArgSet)
  ;

char* RooArgSet::_poolBegin = 0 ;
char* RooArgSet::_poolCur = 0 ;
char* RooArgSet::_poolEnd = 0 ;
#define POOLSIZE 1048576

struct POOLDATA 
{
  void* _base ;
} ;

static std::list<POOLDATA> _memPoolList ;

//_____________________________________________________________________________
void RooArgSet::cleanup()
{
  // Clear memoery pool on exit to avoid reported memory leaks

  std::list<POOLDATA>::iterator iter = _memPoolList.begin() ;
  while(iter!=_memPoolList.end()) {
    free(iter->_base) ;
    iter->_base=0 ;
    iter++ ;
  }
  _memPoolList.clear() ;
}


#ifdef USEMEMPOOL

//_____________________________________________________________________________
void* RooArgSet::operator new (size_t bytes)
{
  // Overloaded new operator guarantees that all RooArgSets allocated with new
  // have a unique address, a property that is exploited in several places
  // in roofit to quickly index contents on normalization set pointers. 
  // The memory pool only allocates space for the class itself. The elements
  // stored in the set are stored outside the pool.

  //cout << " RooArgSet::operator new(" << bytes << ")" << endl ;

  if (!_poolBegin || _poolCur+(sizeof(RooArgSet)) >= _poolEnd) {

    if (_poolBegin!=0) {
      oocxcoutD((TObject*)0,Caching) << "RooArgSet::operator new(), starting new 1MB memory pool" << endl ;
    }

    // Start pruning empty memory pools if number exceeds 3
    if (_memPoolList.size()>3) {
      
      void* toFree(0) ;

      for (std::list<POOLDATA>::iterator poolIter =  _memPoolList.begin() ; poolIter!=_memPoolList.end() ; ++poolIter) {

	// If pool is empty, delete it and remove it from list
	if ((*(Int_t*)(poolIter->_base))==0) {
	  oocxcoutD((TObject*)0,Caching) << "RooArgSet::operator new(), pruning empty memory pool " << (void*)(poolIter->_base) << endl ;

	  toFree = poolIter->_base ;
	  _memPoolList.erase(poolIter) ;
	  break ;
	}
      }      

      free(toFree) ;      
    }
    
    void* mem = malloc(POOLSIZE) ;

    _poolBegin = (char*)mem ;
    // Reserve space for pool counter at head of pool
    _poolCur = _poolBegin+sizeof(Int_t) ;
    _poolEnd = _poolBegin+(POOLSIZE) ;

    // Clear pool counter
    *((Int_t*)_poolBegin)=0 ;
    
    POOLDATA p ;
    p._base=mem ;
    _memPoolList.push_back(p) ;

    RooSentinel::activate() ;
  }

  char* ptr = _poolCur ;
  _poolCur += bytes ;

  // Increment use counter of pool
  (*((Int_t*)_poolBegin))++ ;

  return ptr ;

}



//_____________________________________________________________________________
void RooArgSet::operator delete (void* ptr)
{
  // Memory is owned by pool, we need to do nothing to release it

  // Decrease use count in pool that ptr is on
  for (std::list<POOLDATA>::iterator poolIter =  _memPoolList.begin() ; poolIter!=_memPoolList.end() ; ++poolIter) {
    if ((char*)ptr > (char*)poolIter->_base && (char*)ptr < (char*)poolIter->_base + POOLSIZE) {
      (*(Int_t*)(poolIter->_base))-- ;
      break ;
    }
  }
  
}

#endif


//_____________________________________________________________________________
RooArgSet::RooArgSet() :
  RooAbsCollection()
{
  // Default constructor
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooArgList& list) :
  RooAbsCollection(list.GetName())
{
  // Constructor from a RooArgList. If the list contains multiple
  // objects with the same name, only the first is store in the set.
  // Warning messages will be printed for dropped items.

  add(list,kTRUE) ; // verbose to catch duplicate errors
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooArgList& list, const RooAbsArg* var1) :
  RooAbsCollection(list.GetName())
{
  // Constructor from a RooArgList. If the list contains multiple
  // objects with the same name, only the first is store in the set.
  // Warning messages will be printed for dropped items.

  if (var1) {
    add(*var1,kTRUE) ;
  }
  add(list,kTRUE) ; // verbose to catch duplicate errors
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const char *name) :
  RooAbsCollection(name)
{
  // Empty set constructor
}




//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooArgSet& set1, const RooArgSet& set2, const char *name) : RooAbsCollection(name)
{
  // Construct a set from two existing sets
  add(set1) ;
  add(set2) ;
    
}




//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooAbsArg& var1,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 1 initial object

  add(var1);
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 2 initial objects

  add(var1); add(var2);
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 3 initial objects

  add(var1); add(var2); add(var3);
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 4 initial objects

  add(var1); add(var2); add(var3); add(var4);
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooAbsArg& var1,
		     const RooAbsArg& var2, const RooAbsArg& var3,
		     const RooAbsArg& var4, const RooAbsArg& var5,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 5 initial objects

  add(var1); add(var2); add(var3); add(var4); add(var5);
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
		     const RooAbsArg& var3, const RooAbsArg& var4, 
		     const RooAbsArg& var5, const RooAbsArg& var6,
		     const char *name) :
  RooAbsCollection(name)
{
  // Constructor for set containing 6 initial objects

  add(var1); add(var2); add(var3); add(var4); add(var5); add(var6);
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
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
RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
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
RooArgSet::RooArgSet(const RooAbsArg& var1, const RooAbsArg& var2, 
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
RooArgSet::RooArgSet(const TCollection& tcoll, const char* name) :
  RooAbsCollection(name)
{
  // Constructor from a root TCollection. Elements in the collection that
  // do not inherit from RooAbsArg will be skipped. A warning message
  // will be printed for every skipped item.

  TIterator* iter = tcoll.MakeIterator() ;
  TObject* obj ;
  while((obj=iter->Next())) {
    if (!dynamic_cast<RooAbsArg*>(obj)) {
      coutW(InputArguments) << "RooArgSet::RooArgSet(TCollection) element " << obj->GetName() 
			    << " is not a RooAbsArg, ignored" << endl ;
      continue ;
    }
    add(*(RooAbsArg*)obj) ;
  }
  delete iter ;
}



//_____________________________________________________________________________
RooArgSet::RooArgSet(const RooArgSet& other, const char *name) 
  : RooAbsCollection(other,name)
{
  // Copy constructor. Note that a copy of a set is always non-owning,
  // even the source set is owning. To create an owning copy of
  // a set (owning or not), use the snaphot() method.
}



//_____________________________________________________________________________
RooArgSet::~RooArgSet() 
{
  // Destructor
  
}



//_____________________________________________________________________________
Bool_t RooArgSet::add(const RooAbsArg& var, Bool_t silent) 
{
  // Add element to non-owning set. The operation will fail if
  // a similarly named object already exists in the set, or
  // the set is specified to own its elements. Eventual error messages
  // can be suppressed with the silent flag

  return checkForDup(var,silent)? kFALSE : RooAbsCollection::add(var,silent) ;
}



//_____________________________________________________________________________
Bool_t RooArgSet::addOwned(RooAbsArg& var, Bool_t silent)
{
  // Add element to an owning set. The operation will fail if
  // a similarly named object already exists in the set, or
  // the set is not specified to own its elements. Eventual error messages
  // can be suppressed with the silent flag

  return checkForDup(var,silent)? kFALSE : RooAbsCollection::addOwned(var,silent) ;
}



//_____________________________________________________________________________
RooAbsArg* RooArgSet::addClone(const RooAbsArg& var, Bool_t silent) 
{
  // Add clone of specified element to an owning set. If sucessful, the
  // set will own the clone, not the original. The operation will fail if
  // a similarly named object already exists in the set, or
  // the set is not specified to own its elements. Eventual error messages
  // can be suppressed with the silent flag

  return checkForDup(var,silent)? 0 : RooAbsCollection::addClone(var,silent) ;
}



//_____________________________________________________________________________
RooAbsArg& RooArgSet::operator[](const char* name) const 
{     
  // Array operator. Named element must exist in set, otherwise
  // code will abort. 
  //
  // When used as lvalue in assignment operations, the element contained in
  // the list will not be changed, only the value of the existing element!

  RooAbsArg* arg = find(name) ;
  if (!arg) {
    coutE(InputArguments) << "RooArgSet::operator[](" << GetName() << ") ERROR: no element named " << name << " in set" << endl ;
    RooErrorHandler::softAbort() ;
  }
  return *arg ; 
}



//_____________________________________________________________________________
Bool_t RooArgSet::checkForDup(const RooAbsArg& var, Bool_t silent) const 
{
  // Check if element with var's name is already in set

  RooAbsArg *other = 0;
  if((other= find(var.GetName()))) {
    if(other != &var) {
      if (!silent)
	// print a warning if this variable is not the same one we
	// already have
	coutE(InputArguments) << "RooArgSet::checkForDup: ERROR argument with name " << var.GetName() << " is already in this set" << endl;
    }
    // don't add duplicates
    return kTRUE;
  }

  return kFALSE ;
}



//_____________________________________________________________________________
Double_t RooArgSet::getRealValue(const char* name, Double_t defVal, Bool_t verbose) const
{
  // Get value of a RooAbsReal stored in set with given name. If none is found, value of defVal is returned.
  // No error messages are printed unless the verbose flag is set

  RooAbsArg* raa = find(name) ;
  if (!raa) {
    if (verbose) coutE(InputArguments) << "RooArgSet::getRealValue(" << GetName() << ") ERROR no object with name '" << name << "' found" << endl ;
    return defVal ;
  }
  RooAbsReal* rar = dynamic_cast<RooAbsReal*>(raa) ;
  if (!rar) {
    if (verbose) coutE(InputArguments) << "RooArgSet::getRealValue(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsReal" << endl ;
    return defVal ;
  }
  return rar->getVal() ;
}



//_____________________________________________________________________________
Bool_t RooArgSet::setRealValue(const char* name, Double_t newVal, Bool_t verbose) 
{
  // Set value of a RooAbsRealLValye stored in set with given name to newVal
  // No error messages are printed unless the verbose flag is set

  RooAbsArg* raa = find(name) ;
  if (!raa) {
    if (verbose) coutE(InputArguments) << "RooArgSet::setRealValue(" << GetName() << ") ERROR no object with name '" << name << "' found" << endl ;
    return kTRUE ;
  }
  RooAbsRealLValue* rar = dynamic_cast<RooAbsRealLValue*>(raa) ;
  if (!rar) {
    if (verbose) coutE(InputArguments) << "RooArgSet::setRealValue(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsRealLValue" << endl ;
    return kTRUE;
  }
  rar->setVal(newVal) ;
  return kFALSE ;
}



//_____________________________________________________________________________
const char* RooArgSet::getCatLabel(const char* name, const char* defVal, Bool_t verbose) const
{
  // Get state name of a RooAbsCategory stored in set with given name. If none is found, value of defVal is returned.
  // No error messages are printed unless the verbose flag is set

  RooAbsArg* raa = find(name) ;
  if (!raa) {
    if (verbose) coutE(InputArguments) << "RooArgSet::getCatLabel(" << GetName() << ") ERROR no object with name '" << name << "' found" << endl ;
    return defVal ;
  }
  RooAbsCategory* rac = dynamic_cast<RooAbsCategory*>(raa) ;
  if (!rac) {
    if (verbose) coutE(InputArguments) << "RooArgSet::getCatLabel(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsCategory" << endl ;
    return defVal ;
  }
  return rac->getLabel() ;
}



//_____________________________________________________________________________
Bool_t RooArgSet::setCatLabel(const char* name, const char* newVal, Bool_t verbose) 
{
  // Set state name of a RooAbsCategoryLValue stored in set with given name to newVal.
  // No error messages are printed unless the verbose flag is set

  RooAbsArg* raa = find(name) ;
  if (!raa) {
    if (verbose) coutE(InputArguments) << "RooArgSet::setCatLabel(" << GetName() << ") ERROR no object with name '" << name << "' found" << endl ;
    return kTRUE ;
  }
  RooAbsCategoryLValue* rac = dynamic_cast<RooAbsCategoryLValue*>(raa) ;
  if (!rac) {
    if (verbose) coutE(InputArguments) << "RooArgSet::setCatLabel(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsCategory" << endl ;
    return kTRUE ;
  }
  rac->setLabel(newVal) ;
  return kFALSE ;
}



//_____________________________________________________________________________
Int_t RooArgSet::getCatIndex(const char* name, Int_t defVal, Bool_t verbose) const
{
  // Get index value of a RooAbsCategory stored in set with given name. If none is found, value of defVal is returned.
  // No error messages are printed unless the verbose flag is set

  RooAbsArg* raa = find(name) ;
  if (!raa) {
    if (verbose) coutE(InputArguments) << "RooArgSet::getCatLabel(" << GetName() << ") ERROR no object with name '" << name << "' found" << endl ;
    return defVal ;
  }
  RooAbsCategory* rac = dynamic_cast<RooAbsCategory*>(raa) ;
  if (!rac) {
    if (verbose) coutE(InputArguments) << "RooArgSet::getCatLabel(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsCategory" << endl ;
    return defVal ;
  }
  return rac->getIndex() ;
}



//_____________________________________________________________________________
Bool_t RooArgSet::setCatIndex(const char* name, Int_t newVal, Bool_t verbose) 
{
  // Set index value of a RooAbsCategoryLValue stored in set with given name to newVal.
  // No error messages are printed unless the verbose flag is set

  RooAbsArg* raa = find(name) ;
  if (!raa) {
    if (verbose) coutE(InputArguments) << "RooArgSet::setCatLabel(" << GetName() << ") ERROR no object with name '" << name << "' found" << endl ;
    return kTRUE ;
  }
  RooAbsCategoryLValue* rac = dynamic_cast<RooAbsCategoryLValue*>(raa) ;
  if (!rac) {
    if (verbose) coutE(InputArguments) << "RooArgSet::setCatLabel(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsCategory" << endl ;
    return kTRUE ;
  }
  rac->setIndex(newVal) ;
  return kFALSE ;
}



//_____________________________________________________________________________
const char* RooArgSet::getStringValue(const char* name, const char* defVal, Bool_t verbose) const
{
  // Get string value of a RooAbsString stored in set with given name. If none is found, value of defVal is returned.
  // No error messages are printed unless the verbose flag is set

  RooAbsArg* raa = find(name) ;
  if (!raa) {
    if (verbose) coutE(InputArguments) << "RooArgSet::getStringValue(" << GetName() << ") ERROR no object with name '" << name << "' found" << endl ;
    return defVal ;
  }
  RooAbsString* ras = dynamic_cast<RooAbsString*>(raa) ;
  if (!ras) {
    if (verbose) coutE(InputArguments) << "RooArgSet::getStringValue(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsString" << endl ;
    return defVal ;
  }
  return ras->getVal() ;
}



//_____________________________________________________________________________
Bool_t RooArgSet::setStringValue(const char* name, const char* newVal, Bool_t verbose) 
{
  // Set string value of a RooStringVar stored in set with given name to newVal.
  // No error messages are printed unless the verbose flag is set

  RooAbsArg* raa = find(name) ;
  if (!raa) {
    if (verbose) coutE(InputArguments) << "RooArgSet::setStringValue(" << GetName() << ") ERROR no object with name '" << name << "' found" << endl ;
    return kTRUE ;
  }
  RooStringVar* ras = dynamic_cast<RooStringVar*>(raa) ;
  if (!ras) {
    if (verbose) coutE(InputArguments) << "RooArgSet::setStringValue(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsString" << endl ;
    return kTRUE ;
  }
  ras->setVal(newVal) ;
  return kFALSE ;
}



//_____________________________________________________________________________
void RooArgSet::writeToFile(const char* fileName) const
{
  // Write contents of the argset to specified file.
  // See writeToStream() for details

  ofstream ofs(fileName) ;
  if (ofs.fail()) {
    coutE(InputArguments) << "RooArgSet::writeToFile(" << GetName() << ") error opening file " << fileName << endl ;
    return ;
  }
  writeToStream(ofs,kFALSE) ;
}



//_____________________________________________________________________________
Bool_t RooArgSet::readFromFile(const char* fileName, const char* flagReadAtt, const char* section, Bool_t verbose) 
{
  // Read contents of the argset from specified file.
  // See readFromStream() for details

  ifstream ifs(fileName) ;
  if (ifs.fail()) {
    coutE(InputArguments) << "RooArgSet::readFromFile(" << GetName() << ") error opening file " << fileName << endl ;
    return kTRUE ;
  }
  return readFromStream(ifs,kFALSE,flagReadAtt,section,verbose) ;
}




//_____________________________________________________________________________
void RooArgSet::writeToStream(ostream& os, Bool_t compact, const char* /*section*/) const
{
  // Write the contents of the argset in ASCII form to given stream.
  // 
  // A line is written for each element contained in the form
  // <argName> = <argValue>
  // 
  // The <argValue> part of each element is written by the arguments' 
  // writeToStream() function.

  if (compact) {
    coutE(InputArguments) << "RooArgSet::writeToStream(" << GetName() << ") compact mode not supported" << endl ;
    return ;
  }

  TIterator *iterator= createIterator();
  RooAbsArg *next = 0;
  while((0 != (next= (RooAbsArg*)iterator->Next()))) {
    os << next->GetName() << " = " ;
    next->writeToStream(os,kFALSE) ;
    os << endl ;
  }
  delete iterator;  
}




//_____________________________________________________________________________
Bool_t RooArgSet::readFromStream(istream& is, Bool_t compact, const char* flagReadAtt, const char* section, Bool_t verbose) 
{
  // Read the contents of the argset in ASCII form from given stream.
  // 
  // The stream is read to end-of-file and each line is assumed to be
  // of the form
  //
  // <argName> = <argValue>
  // 
  // Lines starting with argNames not matching any element in the list
  // will be ignored with a warning message. In addition limited C++ style 
  // preprocessing and flow control is provided. The following constructions 
  // are recognized:
  //
  // > #include "include.file"       
  // 
  // Include given file, recursive inclusion OK
  // 
  // > if (<boolean_expression>)
  // >   <name> = <value>
  // >   ....
  // > else if (<boolean_expression>)
  //     ....
  // > else
  //     ....
  // > endif
  //
  // All expressions are evaluated by RooFormula, and may involve any of
  // the sets variables. 
  //
  // > echo <Message>
  //
  // Print console message while reading from stream
  //
  // > abort
  //
  // Force termination of read sequence with error status 
  //
  // The value of each argument is read by the arguments readFromStream
  // function.

  if (compact) {
    coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << ") compact mode not supported" << endl ;
    return kTRUE ;
  }

  RooStreamParser parser(is) ;
  parser.setPunctuation("=") ;
  TString token ;
  Bool_t retVal(kFALSE) ;
  
  // Conditional stack and related state variables
  // coverity[UNINIT]
  Bool_t anyCondTrue[100] ;
  Bool_t condStack[100] ;
  Bool_t lastLineWasElse=kFALSE ;
  Int_t condStackLevel=0 ;
  condStack[0]=kTRUE ;
  
  // Prepare section processing
  TString sectionHdr("[") ;
  if (section) sectionHdr.Append(section) ;
  sectionHdr.Append("]") ;
  Bool_t inSection(section?kFALSE:kTRUE) ;

  Bool_t reprocessToken = kFALSE ;
  while (1) {

    if (is.eof() || is.fail() || parser.atEOF()) {
      break ;
    }
    
    // Read next token until end of file
    if (!reprocessToken) {
      token = parser.readToken() ;
    }
    reprocessToken = kFALSE ;

    // Skip empty lines 
    if (token.IsNull()) {
      continue ;
    }

    // Process include directives
    if (!token.CompareTo("include")) {
      if (parser.atEOL()) {
	coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() 
			      << "): no filename found after include statement" << endl ;
	return kTRUE ;
      }
      TString filename = parser.readLine() ;
      ifstream incfs(filename) ;
      if (!incfs.good()) {
	coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): cannot open include file " << filename << endl ;
	return kTRUE ;
      }
      coutI(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): processing include file " 
			    << filename << endl ;
      if (readFromStream(incfs,compact,flagReadAtt,inSection?0:section,verbose)) return kTRUE ;
      continue ;
    }

    // Process section headers if requested
    if (*token.Data()=='[') {
      TString hdr(token) ;
      const char* last = token.Data() + token.Length() -1 ;
      if (*last != ']') {
	hdr.Append(" ") ;
	hdr.Append(parser.readLine()) ;
      }
//       parser.putBackToken(token) ;
//       token = parser.readLine() ;
      if (section) {
	inSection = !sectionHdr.CompareTo(hdr) ;
      }
      continue ;
    }

    // If section is specified, ignore all data outside specified section
    if (!inSection) {
      parser.zapToEnd(kTRUE) ;
      continue ;
    }

    // Conditional statement evaluation
    if (!token.CompareTo("if")) {
      
      // Extract conditional expressions and check validity
      TString expr = parser.readLine() ;
      RooFormula form(expr,expr,*this) ;
      if (!form.ok()) return kTRUE ;
      
      // Evaluate expression
      Bool_t status = form.eval()?kTRUE:kFALSE ;
      if (lastLineWasElse) {
	anyCondTrue[condStackLevel] |= status ;
	lastLineWasElse=kFALSE ;
      } else {
	condStackLevel++ ;
	anyCondTrue[condStackLevel] = status ;
      }
      condStack[condStackLevel] = status ;
      
      if (verbose) cxcoutD(Eval) << "RooArgSet::readFromStream(" << GetName() 
				 << "): conditional expression " << expr << " = " 
				 << (condStack[condStackLevel]?"true":"false") << endl ;
      continue ; // go to next line
    }
    
    if (!token.CompareTo("else")) {
      // Must have seen an if statement before
      if (condStackLevel==0) {
	coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): unmatched 'else'" << endl ;
      }
      
      if (parser.atEOL()) {
	// simple else: process if nothing else was true
	condStack[condStackLevel] = !anyCondTrue[condStackLevel] ; 
	parser.zapToEnd(kFALSE) ;
	continue ;
      } else {
	// if anything follows it should be 'if'
	token = parser.readToken() ;
	if (token.CompareTo("if")) {
	  coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): syntax error: 'else " << token << "'" << endl ;
	  return kTRUE ;
	} else {
	  if (anyCondTrue[condStackLevel]) {
	    // No need for further checking, true conditional already processed
	    condStack[condStackLevel] = kFALSE ;
	    parser.zapToEnd(kFALSE) ;
	    continue ;
	  } else {
	    // Process as normal 'if' no true conditional was encountered 
	    reprocessToken = kTRUE ;
	    lastLineWasElse=kTRUE ;
	    continue ;
	  }
	}
      }	
    }
    
    if (!token.CompareTo("endif")) {
      // Must have seen an if statement before
      if (condStackLevel==0) {
	coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): unmatched 'endif'" << endl ;
	return kTRUE ;
      }
      
      // Decrease stack by one
      condStackLevel-- ;
      continue ;
    } 
    
    // If current conditional is true
    if (condStack[condStackLevel]) {
      
      // Process echo statements
      if (!token.CompareTo("echo")) {
	TString message = parser.readLine() ;
	coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): >> " << message << endl ;
	continue ;
      } 
      
      // Process abort statements
      if (!token.CompareTo("abort")) {
	TString message = parser.readLine() ;
	coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): USER ABORT" << endl ;
	return kTRUE ;
      } 
      
      // Interpret the rest as <arg> = <value_expr> 
      RooAbsArg *arg ;

      if ((arg = find(token)) && !arg->getAttribute("Dynamic")) {
	if (parser.expectToken("=",kTRUE)) {
	  parser.zapToEnd(kTRUE) ;
	  retVal=kTRUE ;
	  coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() 
				<< "): missing '=' sign: " << arg << endl ;
	  continue ;
	}
	Bool_t argRet = arg->readFromStream(is,kFALSE,verbose) ;	
	if (!argRet && flagReadAtt) arg->setAttribute(flagReadAtt,kTRUE) ;
	retVal |= argRet ;
      } else {
	if (verbose) {
	  coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): argument " 
				<< token << " not in list, ignored" << endl ;
	}
	parser.zapToEnd(kTRUE) ;
      }
    } else {
      parser.readLine() ;
    }
  }
  
  // Did we fully unwind the conditional stack?
  if (condStackLevel!=0) {
    coutE(InputArguments) << "RooArgSet::readFromStream(" << GetName() << "): missing 'endif'" << endl ;
    return kTRUE ;
  }
  
  return retVal ;
}


Bool_t RooArgSet::isInRange(const char* rangeSpec) 
{
  char buf[1024] ;
  strlcpy(buf,rangeSpec,1024) ;
  char* token = strtok(buf,",") ;
  
  TIterator* iter = createIterator() ;

  while(token) {

    Bool_t accept=kTRUE ;
    iter->Reset() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      RooAbsRealLValue* lvarg = dynamic_cast<RooAbsRealLValue*>(arg) ;
      if (lvarg) {
	if (!lvarg->inRange(token)) {
	  accept=kFALSE ;
	  break ;
	}
      }
      // WVE MUST HANDLE RooAbsCategoryLValue ranges as well
    }
    if (accept) {
      delete iter ;
      return kTRUE ;
    }

    token = strtok(0,",") ;
  }

  delete iter ;
  return kFALSE ;
}



