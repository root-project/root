/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooAbsCollection.cc,v 1.11 2001/10/17 05:03:57 verkerke Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   07-Mar-2001 WV Created initial version
 *   23-Aug-2001 DK Enforce set semantics in the public interface
 *
 * Copyright (C) 2001 University of California
 *****************************************************************************/

// -- CLASS DESCRIPTION [CONT] --
// RooAbsCollection is an abstract container object that can hold multiple RooAbsArg objects.

#include <iostream.h>
#include <iomanip.h>
#include <fstream.h>
#include "TClass.h"
#include "RooFitCore/RooAbsCollection.hh"
#include "RooFitCore/RooStreamParser.hh"
#include "RooFitCore/RooFormula.hh"
#include "RooFitCore/RooAbsRealLValue.hh"
#include "RooFitCore/RooAbsCategoryLValue.hh"
#include "RooFitCore/RooStringVar.hh"
#include "RooFitCore/RooTrace.hh"
#include "RooFitCore/RooArgList.hh"

ClassImp(RooAbsCollection)
  ;

RooAbsCollection::RooAbsCollection() :
  _name(), _ownCont(kFALSE)
{
  // Default constructor
  RooTrace::create(this) ;
}

RooAbsCollection::RooAbsCollection(const char *name) :
  _name(name), _ownCont(kFALSE)
{
  // Empty collection constructor
  RooTrace::create(this) ;
}


RooAbsCollection::RooAbsCollection(const RooAbsCollection& other, const char *name) :
  _name(name), _ownCont(kFALSE)
{
  // Copy constructor. Note that a copy of a collection is always non-owning,
  // even the source collection is owning. To create an owning copy of
  // a collection (owning or not), use the snaphot() method.

  RooTrace::create(this) ;
  if (!name) setName(other.GetName()) ;
  
  // Transfer contents (not owned)
  TIterator *iterator= other.createIterator();
  RooAbsArg *arg(0);
  while(arg= (RooAbsArg*)iterator->Next()) {
    add(*arg);
  }
  delete iterator;
}



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
    while(arg=(RooAbsArg*)iter->Next()) {

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
    cout << "RooAbsCollection::safeDeleteList(" << GetName() 
	 << ") WARNING: unable to delete following elements in client-server order " ;
    Print("1") ;
  }

  // Built-in delete remaining elements
  _list.Delete() ;
}




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
  // Ff deepCopy is specified, the client-server links between the cloned
  // list elements and the cloned external dependents are reconnected to
  // each other, making the snapshot a completely self-contained entity.
  //
  //

  // First create empty list
  TString snapName("Snapshot of ") ;
  snapName.Append(GetName()) ;
  RooAbsCollection* snapshot = (RooAbsCollection*) create(snapName.Data()) ; //new RooAbsCollection(snapName.Data()) ;

  // Copy contents
  TIterator *iterator= createIterator();
  RooAbsArg *orig(0);
  while(0 != (orig= (RooAbsArg*)iterator->Next())) {
    RooAbsArg *copy= (RooAbsArg*)orig->Clone();
    snapshot->add(*copy);
  }
  delete iterator;

  TIterator* vIter = snapshot->createIterator() ;
  RooAbsArg* var ;

  // Add external dependents
  if (deepCopy) {
    // Recursively add clones of all servers
    while (var=(RooAbsArg*)vIter->Next()) {
      snapshot->addServerClonesToList(*var) ;
    }
  }

  // Redirect all server connections to internal list members
  vIter->Reset() ;
  while (var=(RooAbsArg*)vIter->Next()) {
    var->redirectServers(*snapshot,deepCopy) ;
  }
  delete vIter ;

  // Transfer ownership of contents to list
  snapshot->_ownCont = kTRUE ;
  return snapshot ;
}



void RooAbsCollection::addServerClonesToList(const RooAbsArg& var)
{
  // Add clones of servers of given argument to list

  TIterator* sIter = var.serverIterator() ;
  RooAbsArg* server ;
  while (server=(RooAbsArg*)sIter->Next()) {
    if (!find(server->GetName())) {
      RooAbsArg* serverClone = (RooAbsArg*)server->Clone() ;
      serverClone->setAttribute("SnapShot_ExtRefClone") ;
      add(*serverClone) ;      
      addServerClonesToList(*server) ;
    }
  }
  delete sIter ;
}

RooAbsCollection &RooAbsCollection::operator=(const RooAbsCollection& other) {

  // The assignment operator sets the value of any argument in our set
  // that also appears in the other set.
  if (&other==this) return *this ;

  RooAbsArg *elem, *theirs ;
  Int_t index(getSize());
  while(--index >= 0) {
    elem= (RooAbsArg*)_list.At(index);
    theirs= other.find(elem->GetName());
    if(!theirs) continue;

    theirs->syncCache() ;
    elem->copyCache(theirs) ;

  }
  return *this;
}

Bool_t RooAbsCollection::addOwned(RooAbsArg& var, Bool_t silent) {
  // Add the specified argument to list. Returns kTRUE if successful, or
  // else kFALSE if a variable of the same name is already in the list.
  // This method can only be called on a list that is flagged as owning
  // all of its contents, or else on an empty list (which will force the
  // list into that mode).

  // check that we own our variables or else are empty
  if(!_ownCont && (getSize() > 0)) {
    cout << ClassName() << "::" << GetName() << "::addOwned: can only add to an owned list" << endl;
    return kFALSE;
  }
  _ownCont= kTRUE;

  _list.Add((RooAbsArg*)&var);
  return kTRUE;
}


RooAbsArg *RooAbsCollection::addClone(const RooAbsArg& var, Bool_t silent) {
  // Add a clone of the specified argument to list. Returns a pointer to
  // the clone if successful, or else zero if a variable of the same name
  // is already in the list or the list does *not* own its variables (in
  // this case, try add() instead.) Calling addClone() on an empty list
  // forces it to take ownership of all its subsequent variables.

  // check that we own our variables or else are empty
  if(!_ownCont && (getSize() > 0)) {
    cout << ClassName() << "::" << GetName() << "::addClone: can only add to an owned list" << endl;
    return 0;
  }
  _ownCont= kTRUE;

  // add a pointer to a clone of this variable to our list (we now own it!)
  RooAbsArg *clone= (RooAbsArg*)var.Clone();
  if(0 != clone) _list.Add((RooAbsArg*)clone);

  return clone;
}



Bool_t RooAbsCollection::add(const RooAbsArg& var, Bool_t silent) {
  // Add the specified argument to list. Returns kTRUE if successful, or
  // else kFALSE if a variable of the same name is already in the list
  // or the list owns its variables (in this case, try addClone() or addOwned() instead).

  // check that this isn't a copy of a list
  if(_ownCont) {
    cout << ClassName() << "::" << GetName() << "::add: cannot add to an owned list" << endl;
    return kFALSE;
  }

  // add a pointer to this variable to our list (we don't own it!)
  _list.Add((RooAbsArg*)&var);
  return kTRUE;
}




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


Bool_t RooAbsCollection::replace(const RooAbsCollection &other) {
  // Replace any args in our set with args of the same name from the other set
  // and return kTRUE for success. Fails if this list is a copy of another.

  // check that this isn't a copy of a list
  if(_ownCont) {
    cout << "RooAbsCollection: cannot replace variables in a copied list" << endl;
    return kFALSE;
  }
  // loop over elements in the other list
  TIterator *otherArgs= other.createIterator();
  const RooAbsArg *arg(0);
  while(arg= (const RooAbsArg*)otherArgs->Next()) {
    // do we have an arg of the same name in our set?
    RooAbsArg *found= find(arg->GetName());
    if(found) replace(*found,*arg);
  }
  delete otherArgs;
  return kTRUE;
}

Bool_t RooAbsCollection::replace(const RooAbsArg& var1, const RooAbsArg& var2) 
{
  // Replace var1 with var2 and return kTRUE for success. Fails if
  // this list is a copy of another, if var1 is not already in this set,
  // or if var2 is already in this set. var1 and var2 do not need to have
  // the same name.

  // check that this isn't a copy of a list
  if(_ownCont) {
    cout << "RooAbsCollection: cannot replace variables in a copied list" << endl;
    return kFALSE;
  }
  // is var1 already in this list?
  const char *name= var1.GetName();

  Bool_t foundVar1(kFALSE) ;
  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
    if (arg==&var1) foundVar1=kTRUE ;
  }
  delete iter ;
  if (!foundVar1) {
    cout << "RooAbsCollection: variable \"" << name << "\" is not in the list"
	 << " and cannot be replaced" << endl;
    return kFALSE;
  }

  RooAbsArg *other= find(name);
  // is var2's name already in this list?
  if (dynamic_cast<RooArgSet*>(this)) {
    other= find(var2.GetName());
    if(other != 0 && other != &var1) {
      cout << "RooAbsCollection: cannot replace \"" << name
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



Bool_t RooAbsCollection::remove(const RooAbsArg& var, Bool_t silent, Bool_t matchByNameOnly) {
  // Remove the specified argument from our list. Return kFALSE if
  // the specified argument is not found in our list. An exact pointer
  // match is required, not just a match by name. A variable can be
  // removed from a copied list and will be deleted at the same time.

  // is var already in this list?
  TString name(var.GetName()) ;
  Bool_t anyFound(kFALSE) ;

  TIterator* iter = createIterator() ;
  RooAbsArg* arg ;
  while(arg=(RooAbsArg*)iter->Next()) {
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

Bool_t RooAbsCollection::remove(const RooAbsCollection& list, Bool_t silent, Bool_t matchByNameOnly) {
  // Remove each argument in the input list from our list using remove(const RooAbsArg&).
  // Return kFALSE in case of problems.

  Bool_t result(false) ;

  Int_t n= list.getSize() ;
  for(Int_t index= 0; index < n; index++) {
    result |= remove((RooAbsArg&)*list._list.At(index),silent,matchByNameOnly) ;
  }

  return result;
}

void RooAbsCollection::removeAll() {
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

void RooAbsCollection::setAttribAll(const Text_t* name, Bool_t value) 
{
  // Set given attribute in each element of the collection by
  // calling each elements setAttribute() function.

  TIterator* iter= createIterator() ;
  RooAbsArg* arg ;
  while (arg=(RooAbsArg*)iter->Next()) {
    arg->setAttribute(name,value) ;
  }
  delete iter ;
}


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
  while (arg=(RooAbsArg*)iter->Next()) {
    if (arg->getAttribute(name)==value)
      sel->add(*arg) ;
  }
  delete iter ;

  return sel ;
}


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
  while (arg=(RooAbsArg*)iter->Next()) {
    if (refColl.find(arg->GetName()))
      sel->add(*arg) ;
  }
  delete iter ;

  return sel ;
}




RooAbsArg *RooAbsCollection::find(const char *name) const 
{
  // Find object with given name in list. A null pointer 
  // is returned if no object with the given name is found

  return (RooAbsArg*) _list.FindObject(name);
}


void RooAbsCollection::printToStream(ostream& os, PrintOption opt, TString indent) const {
  // Print info about this argument set to the specified stream.
  //
  //  Standard: OneLine description of each argument
  //     Shape: Standard description of each argument
  //   Verbose: Shape description of each argument

  // we cannot use oneLinePrint() since we do not inherit from TNamed
  if (opt==OneLine) {
    os << "(" ;
    TIterator *iter= createIterator();
    RooAbsArg *arg ;
    Bool_t first(kTRUE) ;
    while(arg=(RooAbsArg*)iter->Next()) {
      if (!first) {
	os << "," ;
      } else {
	first = kFALSE ;
      }
      os << arg->GetName() ;
    }
    os << ")" << endl ;
    delete iter ;
    return ;
  }


  os << ClassName() << "::" << GetName() << ":" << (_ownCont?" (Owning contents)":"") << endl;
  if(opt >= Standard) {
    TIterator *iterator= createIterator();
    int index= 0;
    RooAbsArg *next(0);
    opt= lessVerbose(opt);
    TString deeper(indent);
    deeper.Append("     ");

    // Adjust the with of the name field to fit the largest name, if requesed
    Int_t maxNameLen(1) ;
    Int_t nameFieldLength = RooAbsArg::_nameLength ;
    if (nameFieldLength==0) {
      while(next=(RooAbsArg*)iterator->Next()) {
	Int_t len = strlen(next->GetName()) ;
	if (len>maxNameLen) maxNameLen = len ;
      }
      iterator->Reset() ;
      RooAbsArg::nameFieldLength(maxNameLen+1) ;
    }

    while(0 != (next= (RooAbsArg*)iterator->Next())) {
      os << indent << setw(3) << ++index << ") ";
      next->printToStream(os,opt,deeper);
    }
    delete iterator;

    // Reset name field length, if modified
    if (nameFieldLength==0) RooAbsArg::nameFieldLength(0) ;
  }
}
