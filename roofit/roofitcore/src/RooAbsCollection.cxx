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
// RooAbsCollection is an abstract container object that can hold
// multiple RooAbsArg objects.  Collections are ordered and can
// contain multiple objects of the same name, (but a derived
// implementation can enforce unique names). The storage of objects in
// implement through class RooLinkedList, a doubly linked list with an
// an optional hash-table lookup mechanism for fast indexing of large
// collections. 
// END_HTML
//
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include "TClass.h"
#include "TStopwatch.h"
#include "TRegexp.h"
#include "RooAbsCollection.h"
#include "RooStreamParser.h"
#include "RooFormula.h"
#include "RooAbsRealLValue.h"
#include "RooAbsCategoryLValue.h"
#include "RooStringVar.h"
#include "RooTrace.h"
#include "RooArgList.h"
#include "RooLinkedListIter.h"
#include "RooCmdConfig.h"
#include "RooRealVar.h"
#include "RooGlobalFunc.h"
#include "RooMsgService.h"
#include <string>
#include <sstream>
using namespace std ;

#if (__GNUC__==3&&__GNUC_MINOR__==2&&__GNUC_PATCHLEVEL__==3)
char* operator+( streampos&, char* );
#endif

ClassImp(RooAbsCollection)
  ;

//_____________________________________________________________________________
RooAbsCollection::RooAbsCollection() :
  _list(0),
  _ownCont(kFALSE), 
  _name()
{
  // Default constructor

  RooTrace::create(this) ;
}



//_____________________________________________________________________________
RooAbsCollection::RooAbsCollection(const char *name) :
  _list(0),
  _ownCont(kFALSE), 
  _name(name)
{
  // Empty collection constructor

  RooTrace::create(this) ;
}



//_____________________________________________________________________________
RooAbsCollection::RooAbsCollection(const RooAbsCollection& other, const char *name) :
  TObject(other),
  RooPrintable(other),
  _list(other._list.getHashTableSize()) , 
  _ownCont(kFALSE), 
  _name(name)
{
  // Copy constructor. Note that a copy of a collection is always non-owning,
  // even the source collection is owning. To create an owning copy of
  // a collection (owning or not), use the snaphot() method.

  RooTrace::create(this) ;
  if (!name) setName(other.GetName()) ;
  
  // Transfer contents (not owned)
  TIterator *iterat= other.createIterator();
  RooAbsArg *arg = 0;
  while((arg= (RooAbsArg*)iterat->Next())) {
    add(*arg);
  }
  delete iterat;
}



//_____________________________________________________________________________
RooAbsCollection::~RooAbsCollection() 
{
  // Destructor

  // Delete all variables in our list if we own them
  if(_ownCont){ 
    safeDeleteList() ;
    //_list.Delete();
  }
  RooTrace::destroy(this) ;
}



//_____________________________________________________________________________
RooLinkedListIter RooAbsCollection::iterator(Bool_t dir) const 
{
    return _list.iterator(dir) ;
}


//_____________________________________________________________________________
void RooAbsCollection::safeDeleteList() 
{
  // Examine client server dependencies in list and
  // delete contents in safe order: any client
  // is deleted before a server is deleted

  // Handle trivial case here
  if (getSize()==1) {
    _list.Delete() ;
    return ;
  }
  
  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  Bool_t working = kTRUE ;

  while(working) {
    working = kFALSE ;
    iter->Reset() ;
    while((arg=(RooAbsArg*)iter->Next())) {

      // Check if arg depends on remainder of list      
      if (!arg->dependsOn(*this,arg)) {
	// Otherwise leave it our and delete it	
	remove(*arg) ;
	delete arg ;
	working = kTRUE ;
      } 
    }
    if (_list.GetSize()<2) break ;
  }
  delete iter ;

  // Check if there are any remaining elements
  if (getSize()>1) {    
    coutW(ObjectHandling) << "RooAbsCollection::safeDeleteList(" << GetName() 
			  << ") WARNING: unable to delete following elements in client-server order " ;
    Print("1") ;
  }

  // Built-in delete remaining elements
  _list.Delete() ;
}



//_____________________________________________________________________________
RooAbsCollection* RooAbsCollection::snapshot(Bool_t deepCopy) const
{
  // Take a snap shot of current collection contents:
  // An owning collection is returned containing clones of 
  // 
  //     - Elements in this collection 
  //     - External dependents of all elements
  //       and recursively any dependents of those dependents
  //       (if deepCopy flag is set)
  //
  // If deepCopy is specified, the client-server links between the cloned
  // list elements and the cloned external dependents are reconnected to
  // each other, making the snapshot a completely self-contained entity.
  //
  //

  // First create empty list
  TString snapName ;
  if (TString(GetName()).Length()>0) {
    snapName.Append("Snapshot of ") ;
    snapName.Append(GetName()) ;
  }
  RooAbsCollection* output = (RooAbsCollection*) create(snapName.Data()) ;
  if (deepCopy || getSize()>1000) {
    output->setHashTableSize(1000) ;
  }
  Bool_t error = snapshot(*output,deepCopy) ;
  if (error) {
    delete output ;
    return 0 ;
  }
  return output ;
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::snapshot(RooAbsCollection& output, Bool_t deepCopy) const 
{
  // Take a snap shot of current collection contents:
  // An owning collection is returned containing clones of 
  // 
  //     - Elements in this collection 
  //     - External dependents of all elements
  //       and recursively any dependents of those dependents
  //       (if deepCopy flag is set)
  //
  // If deepCopy is specified, the client-server links between the cloned
  // list elements and the cloned external dependents are reconnected to
  // each other, making the snapshot a completely self-contained entity.
  //
  //

  // Copy contents
  TIterator *iterat= createIterator();
  RooAbsArg *orig = 0;
  while((0 != (orig= (RooAbsArg*)iterat->Next()))) {
    RooAbsArg *copy= (RooAbsArg*)orig->Clone();
    output.add(*copy);
  }
  delete iterat;

  TIterator* vIter = output.createIterator() ;
  RooAbsArg* var ;

  // Add external dependents
  Bool_t error(kFALSE) ;
  if (deepCopy) {
    // Recursively add clones of all servers
    while ((var=(RooAbsArg*)vIter->Next())) {
      error |= output.addServerClonesToList(*var) ;
    }
  }

  // Handle eventual error conditions
  if (error) {
    coutE(ObjectHandling) << "RooAbsCollection::snapshot(): Errors occurred in deep clone process, snapshot not created" << endl ;
    output._ownCont = kTRUE ;    
    return kTRUE ;
  }

   // Redirect all server connections to internal list members
  vIter->Reset() ;
  while ((var=(RooAbsArg*)vIter->Next())) {
    var->redirectServers(output,deepCopy) ;
  }
  delete vIter ;


  // Transfer ownership of contents to list
  output._ownCont = kTRUE ;
  return kFALSE ;
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::addServerClonesToList(const RooAbsArg& var)
{
  // Add clones of servers of given argument to list

  Bool_t ret(kFALSE) ;

  TIterator* sIter = var.serverIterator() ;
  RooAbsArg* server ;
  while ((server=(RooAbsArg*)sIter->Next())) {
    RooAbsArg* tmp = find(server->GetName()) ;
    if (!tmp) {
      RooAbsArg* serverClone = (RooAbsArg*)server->Clone() ;      
      serverClone->setAttribute("SnapShot_ExtRefClone") ;
      _list.Add(serverClone) ;      
      ret |= addServerClonesToList(*server) ;
    } else {
    }
  }
  delete sIter ;
  return ret ;
}



//_____________________________________________________________________________
RooAbsCollection &RooAbsCollection::operator=(const RooAbsCollection& other) 
{
  // The assignment operator sets the value of any argument in our set
  // that also appears in the other set.

  if (&other==this) return *this ;

  RooAbsArg *elem, *theirs ;
  RooLinkedListIter iter = _list.iterator() ;
  while((elem=(RooAbsArg*)iter.Next())) {
    theirs= other.find(elem->GetName());
    if(!theirs) continue;
    theirs->syncCache() ;
    elem->copyCache(theirs) ;
    elem->setAttribute("Constant",theirs->isConstant()) ;
  }
  return *this;
}



//_____________________________________________________________________________
RooAbsCollection &RooAbsCollection::assignValueOnly(const RooAbsCollection& other) 
{
  // The assignment operator sets the value of any argument in our set
  // that also appears in the other set.

  if (&other==this) return *this ;

  RooAbsArg *elem, *theirs ;
  RooLinkedListIter iter = _list.iterator() ;
  while((elem=(RooAbsArg*)iter.Next())) {
    theirs= other.find(elem->GetName());
    if(!theirs) continue;
    theirs->syncCache() ;
    elem->copyCache(theirs) ;
  }
  return *this;
}



//_____________________________________________________________________________
RooAbsCollection &RooAbsCollection::assignFast(const RooAbsCollection& other) 
{
  // Functional equivalent of operator=() but assumes this and other collection
  // have same layout. Also no attributes are copied

  if (&other==this) return *this ;

  RooAbsArg *elem, *theirs ;
  RooLinkedListIter iter = _list.iterator() ;
  RooLinkedListIter iter2 = other._list.iterator() ;
  while((elem=(RooAbsArg*)iter.Next())) {

    // Identical size of iterators is documented assumption of method
    // coverity[NULL_RETURNS]
    theirs= (RooAbsArg*)iter2.Next() ;

    theirs->syncCache() ;
    elem->copyCache(theirs,kTRUE) ;
  }
  return *this;
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::addOwned(RooAbsArg& var, Bool_t silent) 
{
  // Add the specified argument to list. Returns kTRUE if successful, or
  // else kFALSE if a variable of the same name is already in the list.
  // This method can only be called on a list that is flagged as owning
  // all of its contents, or else on an empty list (which will force the
  // list into that mode).

  // check that we own our variables or else are empty
  if(!_ownCont && (getSize() > 0) && !silent) {
    coutE(ObjectHandling) << ClassName() << "::" << GetName() << "::addOwned: can only add to an owned list" << endl;
    return kFALSE;
  }
  _ownCont= kTRUE;

  _list.Add((RooAbsArg*)&var);
  return kTRUE;
}



//_____________________________________________________________________________
RooAbsArg *RooAbsCollection::addClone(const RooAbsArg& var, Bool_t silent) 
{
  // Add a clone of the specified argument to list. Returns a pointer to
  // the clone if successful, or else zero if a variable of the same name
  // is already in the list or the list does *not* own its variables (in
  // this case, try add() instead.) Calling addClone() on an empty list
  // forces it to take ownership of all its subsequent variables.

  // check that we own our variables or else are empty
  if(!_ownCont && (getSize() > 0) && !silent) {
    coutE(ObjectHandling) << ClassName() << "::" << GetName() << "::addClone: can only add to an owned list" << endl;
    return 0;
  }
  _ownCont= kTRUE;

  // add a pointer to a clone of this variable to our list (we now own it!)
  RooAbsArg *clone2= (RooAbsArg*)var.Clone();
  if(0 != clone2) _list.Add((RooAbsArg*)clone2);

  return clone2;
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::add(const RooAbsArg& var, Bool_t silent) 
{
  // Add the specified argument to list. Returns kTRUE if successful, or
  // else kFALSE if a variable of the same name is already in the list
  // or the list owns its variables (in this case, try addClone() or addOwned() instead).

  // check that this isn't a copy of a list
  if(_ownCont && !silent) {
    coutE(ObjectHandling) << ClassName() << "::" << GetName() << "::add: cannot add to an owned list" << endl;
    return kFALSE;
  }

  // add a pointer to this variable to our list (we don't own it!)
  _list.Add((RooAbsArg*)&var);
  return kTRUE;
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::add(const RooAbsCollection& list, Bool_t silent)
{
  // Add a collection of arguments to this collection by calling add()
  // for each element in the source collection

  Bool_t result(false) ;

  Int_t n= list.getSize() ;
  for(Int_t index= 0; index < n; index++) {
    result |= add((RooAbsArg&)*list._list.At(index),silent) ;
  }

  return result;  
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::addOwned(const RooAbsCollection& list, Bool_t silent)
{
  // Add a collection of arguments to this collection by calling addOwned()
  // for each element in the source collection

  Bool_t result(false) ;

  Int_t n= list.getSize() ;
  for(Int_t index= 0; index < n; index++) {
    result |= addOwned((RooAbsArg&)*list._list.At(index),silent) ;
  }

  return result;  
}



//_____________________________________________________________________________
void RooAbsCollection::addClone(const RooAbsCollection& list, Bool_t silent)
{
  // Add a collection of arguments to this collection by calling addOwned()
  // for each element in the source collection

  Int_t n= list.getSize() ;
  for(Int_t index= 0; index < n; index++) {
    addClone((RooAbsArg&)*list._list.At(index),silent) ;
  }
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::replace(const RooAbsCollection &other) 
{
  // Replace any args in our set with args of the same name from the other set
  // and return kTRUE for success. Fails if this list is a copy of another.

  // check that this isn't a copy of a list
  if(_ownCont) {
    coutE(ObjectHandling) << "RooAbsCollection: cannot replace variables in a copied list" << endl;
    return kFALSE;
  }

  // loop over elements in the other list
  TIterator *otherArgs= other.createIterator();
  const RooAbsArg *arg = 0;
  while((arg= (const RooAbsArg*)otherArgs->Next())) {

    // do we have an arg of the same name in our set?
    RooAbsArg *found= find(arg->GetName());
    if(found) replace(*found,*arg);
  }
  delete otherArgs;
  return kTRUE;
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::replace(const RooAbsArg& var1, const RooAbsArg& var2) 
{
  // Replace var1 with var2 and return kTRUE for success. Fails if
  // this list is a copy of another, if var1 is not already in this set,
  // or if var2 is already in this set. var1 and var2 do not need to have
  // the same name.

  // check that this isn't a copy of a list
  if(_ownCont) {
    coutE(ObjectHandling) << "RooAbsCollection: cannot replace variables in a copied list" << endl;
    return kFALSE;
  }

  // is var1 already in this list?
  const char *name= var1.GetName();

  Bool_t foundVar1(kFALSE) ;
  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (arg==&var1) foundVar1=kTRUE ;
  }
  delete iter ;
  if (!foundVar1) {
    coutE(ObjectHandling) << "RooAbsCollection: variable \"" << name << "\" is not in the list"
	 << " and cannot be replaced" << endl;
    return kFALSE;
  }

  RooAbsArg *other ;

  // is var2's name already in this list?
  if (dynamic_cast<RooArgSet*>(this)) {
    other= find(var2.GetName());
    if(other != 0 && other != &var1) {
      coutE(ObjectHandling) << "RooAbsCollection: cannot replace \"" << name
	   << "\" with already existing \"" << var2.GetName() << "\"" << endl;
      return kFALSE;
    }
  }

  // replace var1 with var2
  _list.Replace(&var1,&var2) ;
//   _list.AddBefore((RooAbsArg*)&var1,(RooAbsArg*)&var2);
//   _list.Remove((RooAbsArg*)&var1);
  return kTRUE;
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::remove(const RooAbsArg& var, Bool_t , Bool_t matchByNameOnly) 
{
  // Remove the specified argument from our list. Return kFALSE if
  // the specified argument is not found in our list. An exact pointer
  // match is required, not just a match by name. A variable can be
  // removed from a copied list and will be deleted at the same time.

  // is var already in this list?
  TString name(var.GetName()) ;
  Bool_t anyFound(kFALSE) ;

  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if ((&var)==arg) {
      _list.Remove(arg) ;
      anyFound=kTRUE ;
    } else if (matchByNameOnly) {
      if (!name.CompareTo(arg->GetName())) {
	_list.Remove(arg) ;
	anyFound=kTRUE ;
      }
    }
  }
  delete iter ;
  
  return anyFound ;
}



//_____________________________________________________________________________
Bool_t RooAbsCollection::remove(const RooAbsCollection& list, Bool_t silent, Bool_t matchByNameOnly) 
{
  // Remove each argument in the input list from our list using remove(const RooAbsArg&).
  // Return kFALSE in case of problems.

  Bool_t result(false) ;

  Int_t n= list.getSize() ;
  for(Int_t index= 0; index < n; index++) {
    result |= remove((RooAbsArg&)*list._list.At(index),silent,matchByNameOnly) ;
  }

  return result;
}



//_____________________________________________________________________________
void RooAbsCollection::removeAll() 
{
  // Remove all arguments from our set, deleting them if we own them.
  // This effectively restores our object to the state it would have
  // just after calling the RooAbsCollection(const char*) constructor.

  if(_ownCont) {
    safeDeleteList() ;
    _ownCont= kFALSE;
  }
  else {
    _list.Clear();
  }
}



//_____________________________________________________________________________
void RooAbsCollection::setAttribAll(const Text_t* name, Bool_t value) 
{
  // Set given attribute in each element of the collection by
  // calling each elements setAttribute() function.

  TIterator* iter= createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    arg->setAttribute(name,value) ;
  }
  delete iter ;
}




//_____________________________________________________________________________
RooAbsCollection* RooAbsCollection::selectByAttrib(const char* name, Bool_t value) const
{
  // Create a subset of the current collection, consisting only of those
  // elements with the specified attribute set. The caller is responsibe
  // for deleting the returned collection

  TString selName(GetName()) ;
  selName.Append("_selection") ;
  RooAbsCollection *sel = (RooAbsCollection*) create(selName.Data()) ;
  
  // Scan set contents for matching attribute
  TIterator* iter= createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    if (arg->getAttribute(name)==value)
      sel->add(*arg) ;
  }
  delete iter ;

  return sel ;
}




//_____________________________________________________________________________
RooAbsCollection* RooAbsCollection::selectCommon(const RooAbsCollection& refColl) const 
{
  // Create a subset of the current collection, consisting only of those
  // elements that are contained as well in the given reference collection.
  // The caller is responsible for deleting the returned collection

  // Create output set
  TString selName(GetName()) ;
  selName.Append("_selection") ;
  RooAbsCollection *sel = (RooAbsCollection*) create(selName.Data()) ; 

  // Scan set contents for matching attribute
  TIterator* iter= createIterator() ;
  RooAbsArg* arg ;
  while ((arg=(RooAbsArg*)iter->Next())) {
    if (refColl.find(arg->GetName()))
      sel->add(*arg) ;
  }
  delete iter ;

  return sel ;
}



//_____________________________________________________________________________
RooAbsCollection* RooAbsCollection::selectByName(const char* nameList, Bool_t verbose) const 
{
  // Create a subset of the current collection, consisting only of those
  // elements with names matching the wildcard expressions in nameList,
  // supplied as a comma separated list

  // Create output set
  TString selName(GetName()) ;
  selName.Append("_selection") ;
  RooAbsCollection *sel = (RooAbsCollection*) create(selName.Data()) ; 
  
  TIterator* iter = createIterator() ;

  char* buf = new char[strlen(nameList)+1] ;
  strlcpy(buf,nameList,strlen(nameList)+1) ;
  char* wcExpr = strtok(buf,",") ;
  while(wcExpr) {
    TRegexp rexp(wcExpr,kTRUE) ;
    if (verbose) {
      cxcoutD(ObjectHandling) << "RooAbsCollection::selectByName(" << GetName() << ") processing expression '" << wcExpr << "'" << endl ;
    }

    iter->Reset() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {
      if (TString(arg->GetName()).Index(rexp)>=0) {
	if (verbose) {
	  cxcoutD(ObjectHandling) << "RooAbsCollection::selectByName(" << GetName() << ") selected element " << arg->GetName() << endl ;
	}
	sel->add(*arg) ;
      }
    }
    wcExpr = strtok(0,",") ;
  }
  delete iter ;
  delete[] buf ;

  return sel ;
}




//_____________________________________________________________________________
Bool_t RooAbsCollection::equals(const RooAbsCollection& otherColl) const
{
  // Check if this and other collection have identically named contents

  // First check equal length 
  if (getSize() != otherColl.getSize()) return kFALSE ;

  // Then check that each element of our list also occurs in the other list
  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!otherColl.find(arg->GetName())) {
      delete iter ;
      return kFALSE ;
    }
  }
  delete iter ;
  return kTRUE ;
}




//_____________________________________________________________________________
Bool_t RooAbsCollection::overlaps(const RooAbsCollection& otherColl) const 
{
  // Check if this and other collection have common entries

  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (otherColl.find(arg->GetName())) {
      delete iter ;
      return kTRUE ;
    }
  }
  delete iter ;
  return kFALSE ;
}




//_____________________________________________________________________________
RooAbsArg *RooAbsCollection::find(const char *name) const 
{
  // Find object with given name in list. A null pointer 
  // is returned if no object with the given name is found

  return (RooAbsArg*) _list.find(name);
}



//_____________________________________________________________________________
string RooAbsCollection::contentsString() const 
{
  // Return comma separated list of contained object names as STL string

  string retVal ;
  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  Bool_t isFirst(kTRUE) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (isFirst) {
      isFirst=kFALSE ;
    } else {
      retVal += "," ;
    }
    retVal += arg->GetName() ;
  }
  delete iter ;
  return retVal ;
}



//_____________________________________________________________________________
void RooAbsCollection::printName(ostream& os) const 
{
  // Return collection name

  os << GetName() ;
}



//_____________________________________________________________________________
void RooAbsCollection::printTitle(ostream& os) const 
{
  // Return collection title

  os << GetTitle() ;
}



//_____________________________________________________________________________
void RooAbsCollection::printClassName(ostream& os) const 
{
  // Return collection class name

  os << IsA()->GetName() ;
}



//_____________________________________________________________________________
Int_t RooAbsCollection::defaultPrintContents(Option_t* opt) const 
{
  // Define default RooPrinable print options for given Print() flag string
  // For inline printing only show value of objects, for default print show
  // name,class name value and extras of each object. In verbose mode
  // also add object adress, argument and title
  
  if (opt && TString(opt)=="I") {
    return kValue ;
  }
  if (opt && TString(opt).Contains("v")) {
    return kAddress|kName|kArgs|kClassName|kValue|kTitle|kExtras ;
  }
  return kName|kClassName|kValue ;
}





//_____________________________________________________________________________
void RooAbsCollection::printValue(ostream& os) const
{
  // Print value of collection, i.e. a comma separated list of contained
  // object names

  Bool_t first2(kTRUE) ;
  os << "(" ;
  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    if (!first2) {
      os << "," ;
    } else {
      first2 = kFALSE ;
    }
    os << arg->GetName() ;
    
  }
  os << ")" ;  
  delete iter ;
}



//_____________________________________________________________________________
void RooAbsCollection::printMultiline(ostream&os, Int_t contents, Bool_t /*verbose*/, TString indent) const
{
  // Implement multiline printin of collection, one line for each ontained object showing
  // the requested content

  if (TString(GetName()).Length()>0 && (contents&kCollectionHeader)) {
    os << indent << ClassName() << "::" << GetName() << ":" << (_ownCont?" (Owning contents)":"") << endl;
  }

  TIterator *iterat= createIterator();
  int index= 0;
  RooAbsArg *next = 0;
  TString deeper(indent);
  deeper.Append("     ");
  
  // Adjust the with of the name field to fit the largest name, if requesed
  Int_t maxNameLen(1) ;
  Int_t nameFieldLengthSaved = RooPrintable::_nameLength ;
  if (nameFieldLengthSaved==0) {
    while((next=(RooAbsArg*)iterat->Next())) {
      Int_t len = strlen(next->GetName()) ;
      if (len>maxNameLen) maxNameLen = len ;
    }
    iterat->Reset() ;
    RooPrintable::nameFieldLength(maxNameLen+1) ;
  }
  
  while((0 != (next= (RooAbsArg*)iterat->Next()))) {
    os << indent << setw(3) << ++index << ") ";
    next->printStream(os,contents,kSingleLine,"");
  }
  delete iterat;
  
  // Reset name field length, if modified
  RooPrintable::nameFieldLength(nameFieldLengthSaved) ;
}



//_____________________________________________________________________________
void RooAbsCollection::dump() const 
{
  // Base contents dumper for debugging purposes

  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    cout << arg << " " << arg->IsA()->GetName() << "::" << arg->GetName() << " (" << arg->GetTitle() << ")" << endl ;
  }
  delete iter ;
}



//_____________________________________________________________________________
void RooAbsCollection::printLatex(const RooCmdArg& arg1, const RooCmdArg& arg2,
				  const RooCmdArg& arg3, const RooCmdArg& arg4,	
				  const RooCmdArg& arg5, const RooCmdArg& arg6,	
				  const RooCmdArg& arg7, const RooCmdArg& arg8) const
{
  // Output content of collection as LaTex table. By default a table with two columns is created: the left
  // column contains the name of each variable, the right column the value.
  //
  // The following optional named arguments can be used to modify the default behavior
  //
  //   Columns(Int_t ncol)                    -- Fold table into multiple columns, i.e. ncol=3 will result in 3 x 2 = 6 total columns
  //   Sibling(const RooAbsCollection& other) -- Define sibling list. The sibling list is assumed to have objects with the same
  //                                             name in the same order. If this is not the case warnings will be printed. If a single
  //                                             sibling list is specified, 3 columns will be output: the (common) name, the value of this
  //                                             list and the value in the sibling list. Multiple sibling lists can be specified by 
  //                                             repeating the Sibling() command. 
  //   Format(const char* str)                -- Classic format string, provided for backward compatibility
  //   Format(...)                            -- Formatting arguments, details are given below
  //   OutputFile(const char* fname)          -- Send output to file with given name rather than standard output
  //
  // The Format(const char* what,...) has the following structure
  //
  //   const char* what          -- Controls what is shown. "N" adds name, "E" adds error, 
  //                                "A" shows asymmetric error, "U" shows unit, "H" hides the value
  //   FixedPrecision(int n)     -- Controls precision, set fixed number of digits
  //   AutoPrecision(int n)      -- Controls precision. Number of shown digits is calculated from error 
  //                                + n specified additional digits (1 is sensible default)
  //   VerbatimName(Bool_t flag) -- Put variable name in a \verb+   + clause.
  //
  // Example use: list.printLatex(Columns(2), Format("NEU",AutoPrecision(1),VerbatimName()) ) ;


  
  // Define configuration for this method
  RooCmdConfig pc("RooAbsCollection::printLatex()") ;
  pc.defineInt("ncol","Columns",0,1) ;
  pc.defineString("outputFile","OutputFile",0,"") ;
  pc.defineString("format","Format",0,"NEYVU") ;
  pc.defineInt("sigDigit","Format",0,1) ;
  pc.defineObject("siblings","Sibling",0,0,kTRUE) ;
  pc.defineInt("dummy","FormatArgs",0,0) ;
  pc.defineMutex("Format","FormatArgs") ;
 
  // Stuff all arguments in a list
  RooLinkedList cmdList;
  cmdList.Add(const_cast<RooCmdArg*>(&arg1)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg2)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg3)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg4)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg5)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg6)) ;
  cmdList.Add(const_cast<RooCmdArg*>(&arg7)) ;  cmdList.Add(const_cast<RooCmdArg*>(&arg8)) ;

  // Process & check varargs 
  pc.process(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8) ;
  if (!pc.ok(kTRUE)) {
    return ;
  }

  const char* outFile = pc.getString("outputFile") ;
  if (outFile && strlen(outFile)) {
    ofstream ofs(outFile) ;
    if (pc.hasProcessed("FormatArgs")) {
      RooCmdArg* formatCmd = static_cast<RooCmdArg*>(cmdList.FindObject("FormatArgs")) ;
      formatCmd->addArg(RooFit::LatexTableStyle()) ;
      printLatex(ofs,pc.getInt("ncol"),0,0,pc.getObjectList("siblings"),formatCmd) ;    
    } else {
      printLatex(ofs,pc.getInt("ncol"),pc.getString("format"),pc.getInt("sigDigit"),pc.getObjectList("siblings")) ;
    }
  } else {
    if (pc.hasProcessed("FormatArgs")) {
      RooCmdArg* formatCmd = static_cast<RooCmdArg*>(cmdList.FindObject("FormatArgs")) ;
      formatCmd->addArg(RooFit::LatexTableStyle()) ;
      printLatex(cout,pc.getInt("ncol"),0,0,pc.getObjectList("siblings"),formatCmd) ;    
    } else {
      printLatex(cout,pc.getInt("ncol"),pc.getString("format"),pc.getInt("sigDigit"),pc.getObjectList("siblings")) ;
    }
  }
}




//_____________________________________________________________________________
void RooAbsCollection::printLatex(ostream& ofs, Int_t ncol, const char* option, Int_t sigDigit, const RooLinkedList& siblingList, const RooCmdArg* formatCmd) const 
{
  // Internal implementation function of printLatex

  // Count number of rows to print
  Int_t nrow = (Int_t) (getSize() / ncol + 0.99) ;
  Int_t i,j,k ;

  // Sibling list do not need to print their name as it is supposed to be the same  
  TString sibOption ;
  RooCmdArg sibFormatCmd ;
  if (option) {
    sibOption = option ;
    sibOption.ReplaceAll("N","") ;
    sibOption.ReplaceAll("n","") ;
  } else {
    sibFormatCmd = *formatCmd ;
    TString tmp = formatCmd->_s[0] ;
    tmp.ReplaceAll("N","") ;    
    tmp.ReplaceAll("n","") ;    
    static char buf[100] ;
    strlcpy(buf,tmp.Data(),100) ;
    sibFormatCmd._s[0] = buf ;
  }


  // Make list of lists ;
  RooLinkedList listList ;
  listList.Add((RooAbsArg*)this) ;
  TIterator* sIter = siblingList.MakeIterator() ;
  RooAbsCollection* col ;
  while((col=(RooAbsCollection*)sIter->Next())) {
    listList.Add(col) ;
  }
  delete sIter ;

  RooLinkedList listListRRV ;

  // Make list of RRV-only components
  TIterator* lIter = listList.MakeIterator() ;
  RooArgList* prevList = 0 ;
  while((col=(RooAbsCollection*)lIter->Next())) {
    RooArgList* list = new RooArgList ;
    TIterator* iter = col->createIterator() ;
    RooAbsArg* arg ;
    while((arg=(RooAbsArg*)iter->Next())) {    
      
      RooRealVar* rrv = dynamic_cast<RooRealVar*>(arg) ;
      if (rrv) {
	list->add(*rrv) ;
      } else {
	coutW(InputArguments) << "RooAbsCollection::printLatex: can only print RooRealVar in LateX, skipping non-RooRealVar object named "
	     << arg->GetName() << endl ;      
      }
      if (prevList && TString(rrv->GetName()).CompareTo(prevList->at(list->getSize()-1)->GetName())) {
	coutW(InputArguments) << "RooAbsCollection::printLatex: WARNING: naming and/or ordering of sibling list is different" << endl ;
      }
    }
    delete iter ;
    listListRRV.Add(list) ;
    if (prevList && list->getSize() != prevList->getSize()) {
      coutW(InputArguments) << "RooAbsCollection::printLatex: ERROR: sibling list(s) must have same length as self" << endl ;
      delete list ;
      listListRRV.Delete() ;
      return ;
    }
    prevList = list ;
  }

  // Construct table header
  Int_t nlist = listListRRV.GetSize() ;
  TString subheader = "l" ;
  for (k=0 ; k<nlist ; k++) subheader += "c" ;

  TString header = "\\begin{tabular}{" ;
  for (j=0 ; j<ncol ; j++) {
    if (j>0) header += "|" ;
    header += subheader ;
  }
  header += "}" ;
  ofs << header << endl ;


  // Print contents, delegating actual printing to RooRealVar::format()
  for (i=0 ; i<nrow ; i++) {
    for (j=0 ; j<ncol ; j++) {
      for (k=0 ; k<nlist ; k++) {
	RooRealVar* par = (RooRealVar*) ((RooArgList*)listListRRV.At(k))->at(i+j*nrow) ;
	if (par) {
	  if (option) {
	    TString* tmp = par->format(sigDigit,(k==0)?option:sibOption.Data()) ;
	    ofs << *tmp ;
	    delete tmp ;
	  } else {
	    TString* tmp = par->format((k==0)?*formatCmd:sibFormatCmd) ;
	    ofs << *tmp ;
	    delete tmp ;
	  }
	}
	if (!(j==ncol-1 && k==nlist-1)) {
	  ofs << " & " ;
	}
      }
    }
    ofs << "\\\\" << endl ;
  }
  
  ofs << "\\end{tabular}" << endl ;
  listListRRV.Delete() ;
}




//_____________________________________________________________________________
Bool_t RooAbsCollection::allInRange(const char* rangeSpec) const
{
  // Return true if all contained object report to have their
  // value inside the specified range

  if (!rangeSpec) return kTRUE ;

  // Parse rangeSpec specification
  vector<string> cutVec ;
  if (rangeSpec && strlen(rangeSpec)>0) {
    if (strchr(rangeSpec,',')==0) {
      cutVec.push_back(rangeSpec) ;
    } else {
      char* buf = new char[strlen(rangeSpec)+1] ;
      strlcpy(buf,rangeSpec,strlen(rangeSpec)+1) ;
      const char* oneRange = strtok(buf,",") ;
      while(oneRange) {
	cutVec.push_back(oneRange) ;
	oneRange = strtok(0,",") ;
      }
      delete[] buf ;
    }
  }


  RooLinkedListIter iter = _list.iterator() ;

  // Apply range based selection criteria
  Bool_t selectByRange = kTRUE ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter.Next())) {
    Bool_t selectThisArg = kFALSE ;
    UInt_t icut ;
    for (icut=0 ; icut<cutVec.size() ; icut++) {
      if (arg->inRange(cutVec[icut].c_str())) {
	selectThisArg = kTRUE ;
	break ;
      }
    }
    if (!selectThisArg) {
      selectByRange = kFALSE ;
      break ;
    }
  }

  return selectByRange ;
}




