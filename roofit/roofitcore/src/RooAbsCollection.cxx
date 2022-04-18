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

/**
\file RooAbsCollection.cxx
\class RooAbsCollection
\ingroup Roofitcore

RooAbsCollection is an abstract container object that can hold
multiple RooAbsArg objects.  Collections are ordered and can
contain multiple objects of the same name, (but a derived
implementation can enforce unique names). The storage of objects is
implemented using the container denoted by RooAbsCollection::Storage_t.
**/

#include "RooAbsCollection.h"

#include "TClass.h"
#include "TRegexp.h"
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
#include "strlcpy.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <memory>

ClassImp(RooAbsCollection);

namespace RooFit {
namespace Detail {

/**
 * Helper for hash-map-assisted finding of elements by name.
 * Create this helper if finding of elements by name is needed.
 * Upon creation, this object checks the global
 * RooNameReg::renameCounter()
 * and tracks elements of this collection by name. If an element
 * gets renamed, this counter will be increased, and the name to
 * object map becomes invalid. In this case, it has to be recreated.
 */
struct HashAssistedFind {

  /// Inititalise empty hash map for fast finding by name.
  template<typename It_t>
  HashAssistedFind(It_t first, It_t last) :
    currentRooNameRegCounter{ RooNameReg::instance().renameCounter() },
    rooNameRegCounterWhereMapWasValid{ currentRooNameRegCounter }
  {
    nameToItemMap.reserve(std::distance(first, last));
    for (auto it = first; it != last; ++it) {
      nameToItemMap.emplace((*it)->namePtr(), *it);
    }
  }

  bool isValid() const {
    return (currentRooNameRegCounter == rooNameRegCounterWhereMapWasValid);
  }

  RooAbsArg * find(const TNamed * nptr) const {
    assert(isValid());

    auto item = nameToItemMap.find(nptr);
    return item != nameToItemMap.end() ? const_cast<RooAbsArg *>(item->second) : nullptr;
  }

  void replace(const RooAbsArg * out, const RooAbsArg * in) {
    nameToItemMap.erase(out->namePtr());
    nameToItemMap.emplace(in->namePtr(), in);
  }

  void insert(const RooAbsArg * elm) {
    nameToItemMap.emplace(elm->namePtr(), elm);
  }

  void erase(const RooAbsArg * elm) {
    nameToItemMap.erase(elm->namePtr());
  }

  std::unordered_map<const TNamed *, const RooAbsArg * const> nameToItemMap;
  const std::size_t & currentRooNameRegCounter;
  std::size_t rooNameRegCounterWhereMapWasValid = 0;
};

}
}


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

RooAbsCollection::RooAbsCollection()
{
  _list.reserve(8);
}



////////////////////////////////////////////////////////////////////////////////
/// Empty collection constructor

RooAbsCollection::RooAbsCollection(const char *name) :
  _name(name)
{
  _list.reserve(8);
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor. Note that a copy of a collection is always non-owning,
/// even the source collection is owning. To create an owning copy of
/// a collection (owning or not), use the snapshot() method.

RooAbsCollection::RooAbsCollection(const RooAbsCollection& other, const char *name) :
  TObject(other),
  RooPrintable(other),
  _name(name),
  _allRRV(other._allRRV),
  _sizeThresholdForMapSearch(100)
{
  RooTrace::create(this) ;
  if (!name) setName(other.GetName()) ;

  _list.reserve(other._list.size());

  for (auto item : other._list) {
    insert(item);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Move constructor.

RooAbsCollection::RooAbsCollection(RooAbsCollection&& other) :
  TObject(other),
  RooPrintable(other),
  _list(std::move(other._list)),
  _ownCont(other._ownCont),
  _name(std::move(other._name)),
  _allRRV(other._allRRV),
  _sizeThresholdForMapSearch(other._sizeThresholdForMapSearch)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooAbsCollection::~RooAbsCollection()
{
  // Delete all variables in our list if we own them
  if(_ownCont){
    deleteList() ;
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Delete contents of the list.
/// The RooAbsArg destructor ensures clients and servers can be deleted in any
/// order.
/// Also cleans the hash-map for fast lookups if present.

void RooAbsCollection::deleteList()
{
  _hashAssistedFind = nullptr;

  // Built-in delete remaining elements
  for (auto item : _list) {
    delete item;
  }
  _list.clear();
}



////////////////////////////////////////////////////////////////////////////////
/// Take a snap shot of current collection contents.
/// An owning collection is returned containing clones of
/// - Elements in this collection
/// - External dependents of all elements and recursively any dependents of those dependents
///   (if deepCopy flag is set)
///
/// This is useful to save the values of variables or parameters. It doesn't require
/// deep copying if the parameters are direct members of the collection.
///
/// If deepCopy is specified, the client-server links between the cloned
/// list elements and the cloned external dependents are reconnected to
/// each other, making the snapshot a completely self-contained entity.
///
///

RooAbsCollection* RooAbsCollection::snapshot(bool deepCopy) const
{
  // First create empty list
  TString snapName ;
  if (TString(GetName()).Length()>0) {
    snapName.Append("Snapshot of ") ;
    snapName.Append(GetName()) ;
  }
  auto* output = static_cast<RooAbsCollection*>(create(snapName.Data())) ;

  if (snapshot(*output,deepCopy)) {
    delete output ;
    return nullptr ;
  }

  return output ;
}



////////////////////////////////////////////////////////////////////////////////
/// Take a snap shot of current collection contents:
/// A collection that owns its elements is returned containing clones of
///     - Elements in this collection
///     - External dependents of those elements
///       and recursively any dependents of those dependents
///       (if deepCopy flag is set)
///
/// If deepCopy is specified, the client-server links between the cloned
/// list elements and the cloned external dependents are reconnected to
/// each other, making the snapshot a completely self-contained entity.
///
///

bool RooAbsCollection::snapshot(RooAbsCollection& output, bool deepCopy) const
{
  // Copy contents
  output.reserve(_list.size());
  for (auto orig : _list) {
    output.add(*static_cast<RooAbsArg*>(orig->Clone()));
  }

  // Add external dependents
  bool error(false) ;
  if (deepCopy) {
    // Recursively add clones of all servers
    // Can only do index access because collection might reallocate when growing
    for (Storage_t::size_type i = 0; i < output._list.size(); ++i) {
      const auto var = output._list[i];
      error |= output.addServerClonesToList(*var);
    }
  }

  // Handle eventual error conditions
  if (error) {
    coutE(ObjectHandling) << "RooAbsCollection::snapshot(): Errors occurred in deep clone process, snapshot not created" << std::endl;
    output._ownCont = true ;
    return true ;
  }



   // Redirect all server connections to internal list members
  for (auto var : output) {
    var->redirectServers(output,deepCopy);
  }


  // Transfer ownership of contents to list
  output._ownCont = true ;
  return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Add clones of servers of given argument to end of list

bool RooAbsCollection::addServerClonesToList(const RooAbsArg& var)
{
  bool ret(false) ;

  // This can be a very heavy operation if existing elements depend on many others,
  // so make sure that we have the hash map available for faster finding.
  if (var.servers().size() > 20 || _list.size() > 30)
    useHashMapForFind(true);

  for (const auto server : var.servers()) {
    RooAbsArg* tmp = find(*server) ;

    if (!tmp) {
      auto* serverClone = static_cast<RooAbsArg*>(server->Clone());
      serverClone->setAttribute("SnapShot_ExtRefClone") ;
      insert(serverClone);
      ret |= addServerClonesToList(*server) ;
    }
  }

  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Assign values from the elements in `other` to our elements.
/// \warning This is not a conventional assignment operator. To avoid confusion, prefer using RooAbsCollection::assign().

RooAbsCollection &RooAbsCollection::operator=(const RooAbsCollection& other)
{
  assign(other);
  return *this;
}


////////////////////////////////////////////////////////////////////////////////
/// Sets the value, cache and constant attribute of any argument in our set
/// that also appears in the other set. Note that this function changes the
/// values of the elements in this collection, but is still marked `const` as
/// it does not change which elements this collection points to.

void RooAbsCollection::assign(const RooAbsCollection& other) const
{
  if (&other==this) return ;

  for (auto elem : _list) {
    auto theirs = other.find(*elem);
    if(!theirs) continue;
    theirs->syncCache() ;
    elem->copyCache(theirs) ;
    elem->setAttribute("Constant",theirs->isConstant()) ;
  }
  return ;
}


////////////////////////////////////////////////////////////////////////////////
/// Sets the value of any argument in our set that also appears in the other set.
/// \param[in] other Collection holding the arguments to syncronize values with.
/// \param[in] forceIfSizeOne If set to true and both our collection
///                and the other collection have a size of one, the arguments are
///                always syncronized without checking if they have the same name.

RooAbsCollection &RooAbsCollection::assignValueOnly(const RooAbsCollection& other, bool forceIfSizeOne)
{
  if (&other==this) return *this;

  // Short cut for 1 element assignment
  if (size()==1 && size() == other.size() && forceIfSizeOne) {
    other.first()->syncCache() ;
    first()->copyCache(other.first(),true) ;
    return *this;
  }

  for (auto elem : _list) {
    auto theirs = other.find(*elem);
    if(!theirs) continue;
    theirs->syncCache() ;
    elem->copyCache(theirs,true) ;
  }
  return *this;
}



////////////////////////////////////////////////////////////////////////////////
/// Functional equivalent of assign() but assumes this and other collection
/// have same layout. Also no attributes are copied

void RooAbsCollection::assignFast(const RooAbsCollection& other, bool setValDirty) const
{
  if (&other==this) return ;
  assert(hasSameLayout(other));

  auto iter2 = other._list.begin();
  for (auto iter1 = _list.begin();
      iter1 != _list.end() && iter2 != other._list.end();
      ++iter1, ++iter2) {
    // Identical size of iterators is documented assumption of method

    if (_allRRV) {
      // All contents are known to be RooRealVars - fast version of assignment
      auto ours   = static_cast<RooRealVar*>(*iter1);
      auto theirs = static_cast<RooRealVar*>(*iter2);
      ours->copyCacheFast(*theirs,setValDirty);
    } else {
      (*iter2)->syncCache() ;
      (*iter1)->copyCache(*iter2,true,setValDirty) ;
    }
  }

}


////////////////////////////////////////////////////////////////////////////////
/// Add an argument and transfer the ownership to the collection. Returns `true`
/// if successful, or `false` if the argument could not be added to the
/// collection (e.g. in the RooArgSet case when an argument with the same name
/// is already in the list). This method can only be called on a list that is
/// flagged as owning all of its contents, or else on an empty list (which will
/// force the list into that mode).
///
/// If the argument you want to add is owned by a `std::unique_ptr`, you should
/// prefer RooAbsCollection::addOwned(std::unique_ptr<RooAbsArg>, bool).

bool RooAbsCollection::addOwned(RooAbsArg& var, bool silent)
{
  if(!canBeAdded(var, silent)) return false;

  // check that we own our variables or else are empty
  if(!_ownCont && !empty() && !silent) {
    coutE(ObjectHandling) << ClassName() << "::" << GetName() << "::addOwned: can only add to an owned list" << std::endl;
    return false;
  }
  _ownCont= true;

  insert(&var);

  return true;
}


////////////////////////////////////////////////////////////////////////////////
/// Add an argument and transfer the ownership to the collection from a
/// `std::unique_ptr`. Always returns `true`. If the argument can not be added
/// to the collection (e.g. in the RooArgSet case when an argument with the
/// same name is already in the list), a `std::runtime_exception` will be
/// thrown, as nobody is owning the argument anymore. This method can only be
/// called on a list that is flagged as owning all of its contents, or else on
/// an empty list (which will force the list into that mode).
///
/// If you want to pass an argument that is not owned by a `std::unique_ptr`,
/// you can use RooAbsCollection::addOwned(RooAbsArg&, bool).

bool RooAbsCollection::addOwned(std::unique_ptr<RooAbsArg> var, bool silent) {
  bool result = addOwned(*var.release(), silent);
  if(!result) {
    throw std::runtime_error(std::string("RooAbsCollection::addOwned could not add the argument to the")
                             + " collection! The ownership would not be well defined if we ignore this.");
  }
  return result;
}



////////////////////////////////////////////////////////////////////////////////
/// Add a clone of the specified argument to list. Returns a pointer to
/// the clone if successful, or else zero if a variable of the same name
/// is already in the list or the list does *not* own its variables (in
/// this case, try add() instead.) Calling addClone() on an empty list
/// forces it to take ownership of all its subsequent variables.

RooAbsArg *RooAbsCollection::addClone(const RooAbsArg& var, bool silent)
{
  if(!canBeAdded(var, silent)) return nullptr;

  // check that we own our variables or else are empty
  if(!_ownCont && !empty() && !silent) {
    coutE(ObjectHandling) << ClassName() << "::" << GetName() << "::addClone: can only add to an owned list" << std::endl;
    return nullptr;
  }
  _ownCont= true;

  // add a pointer to a clone of this variable to our list (we now own it!)
  auto clone2 = static_cast<RooAbsArg*>(var.Clone());
  assert(clone2);

  insert(clone2);

  return clone2;
}



////////////////////////////////////////////////////////////////////////////////
/// Add the specified argument to list. Returns true if successful, or
/// else false if a variable of the same name is already in the list
/// or the list owns its variables (in this case, try addClone() or addOwned() instead).

bool RooAbsCollection::add(const RooAbsArg& var, bool silent)
{
  if(!canBeAdded(var, silent)) return false;

  // check that this isn't a copy of a list
  if(_ownCont && !silent) {
    coutE(ObjectHandling) << ClassName() << "::" << GetName() << "::add: cannot add to an owned list" << std::endl;
    return false;
  }

  // add a pointer to this variable to our list (we don't own it!)
  insert(const_cast<RooAbsArg*>(&var)); //FIXME const_cast

  return true;
}


////////////////////////////////////////////////////////////////////////////////
// Add a collection of arguments to this collection by calling addOwned()
/// for each element in the source collection. The input list can't be an
/// owning collection itself, otherwise the arguments would be owned by two
/// collections.
///
/// If you want to transfer arguments from one owning collection to another,
/// you have two options:
///  1. `std::move` the input collection and use
///     RooAbsCollection::addOwned(RooAbsCollection&&, bool) (preferred)
///  2. release the ownership of the input collection first, using
///     RooAbsCollection::releaseOwnership()

bool RooAbsCollection::addOwned(const RooAbsCollection& list, bool silent)
{
  if(list.isOwning()) {
    throw std::invalid_argument("Passing an owning RooAbsCollection by const& to"
            " RooAbsCollection::addOwned is forbidden because the ownership"
            " would be ambiguous! Please std::move() the RooAbsCollection in this case."
            " Note that the passed RooAbsCollection is invalid afterwards.");

  }

  bool result(false) ;
  _list.reserve(_list.size() + list._list.size());

  for (auto item : list._list) {
    result |= addOwned(*item, silent) ;
  }

  return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Add a collection of arguments to this collection by calling addOwned()
/// for each element in the source collection. Unlike
/// RooAbsCollection::addOwned(const RooAbsCollection&, bool), this function
/// also accepts owning source collections because their content will be
/// moved out.

bool RooAbsCollection::addOwned(RooAbsCollection&& list, bool silent)
{
  if(list.isOwning()) {
    list.releaseOwnership();
  }
  if(list.empty()) return false;

  bool result = addOwned(list, silent);

  if(!result) {
    throw std::runtime_error(std::string("RooAbsCollection::addOwned could not add the argument to the")
                             + " collection! The ownership would not be well defined if we ignore this.");
  }

  // So far, comps has only released the ownership, but it is still valid.
  // However, we don't want users to keep using objects after moving them, so
  // we make sure to keep our promise that the RooArgSet is really moved.
  // Just like a `std::unique_ptr` is also reset when moved.
  list.clear();

  return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Add a collection of arguments to this collection by calling addOwned()
/// for each element in the source collection

void RooAbsCollection::addClone(const RooAbsCollection& list, bool silent)
{
  _list.reserve(_list.size() + list._list.size());

  for (auto item : list._list) {
    addClone(*item, silent);
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Replace any args in our set with args of the same name from the other set
/// and return true for success. Fails if this list is a copy of another.

bool RooAbsCollection::replace(const RooAbsCollection &other)
{
  // check that this isn't a copy of a list
  if(_ownCont) {
    coutE(ObjectHandling) << "RooAbsCollection: cannot replace variables in a copied list" << std::endl;
    return false;
  }

  // loop over elements in the other list
  for (const auto * arg : other._list) {
    // do we have an arg of the same name in our set?
    auto found = find(*arg);
    if (found) replace(*found,*arg);
  }
  return true;
}



////////////////////////////////////////////////////////////////////////////////
/// Replace var1 with var2 and return true for success. Fails if
/// this list is a copy of another, if var1 is not already in this set,
/// or if var2 is already in this set. var1 and var2 do not need to have
/// the same name.

bool RooAbsCollection::replace(const RooAbsArg& var1, const RooAbsArg& var2)
{
  // check that this isn't a copy of a list
  if(_ownCont) {
    coutE(ObjectHandling) << "RooAbsCollection: cannot replace variables in a copied list" << std::endl;
    return false;
  }

  // is var1 already in this list?
  const char *name= var1.GetName();
  auto var1It = std::find(_list.begin(), _list.end(), &var1);

  if (var1It == _list.end()) {
    coutE(ObjectHandling) << "RooAbsCollection: variable \"" << name << "\" is not in the list"
    << " and cannot be replaced" << std::endl;
    return false;
  }


  // is var2's name already in this list?
  if (dynamic_cast<RooArgSet*>(this)) {
    RooAbsArg *other = find(var2);
    if(other != 0 && other != &var1) {
      coutE(ObjectHandling) << "RooAbsCollection: cannot replace \"" << name
      << "\" with already existing \"" << var2.GetName() << "\"" << std::endl;
      return false;
    }
  }

  // replace var1 with var2
  if (_hashAssistedFind) {
    _hashAssistedFind->replace(*var1It, &var2);
  }
  *var1It = const_cast<RooAbsArg*>(&var2); //FIXME try to get rid of const_cast

  if (_allRRV && dynamic_cast<const RooRealVar*>(&var2)==0) {
    _allRRV=false ;
  }

  return true;
}



////////////////////////////////////////////////////////////////////////////////
/// Remove the specified argument from our list. Return false if
/// the specified argument is not found in our list. An exact pointer
/// match is required, not just a match by name.
/// If `matchByNameOnly` is set, items will be looked up by name. In this case, if
/// the collection also owns the item, it will delete it.
bool RooAbsCollection::remove(const RooAbsArg& var, bool , bool matchByNameOnly)
{
  // is var already in this list?
  const auto sizeBefore = _list.size();

  if (matchByNameOnly) {
    const std::string name(var.GetName());
    auto nameMatch = [&name](const RooAbsArg* elm) {
      return elm->GetName() == name;
    };
    std::set<RooAbsArg*> toBeDeleted;

    if (_ownCont) {
      std::for_each(_list.begin(), _list.end(), [&toBeDeleted, nameMatch](RooAbsArg* elm){
        if (nameMatch(elm)) {
          toBeDeleted.insert(elm);
        }
      });
    }

    _list.erase(std::remove_if(_list.begin(), _list.end(), nameMatch), _list.end());

    for (auto arg : toBeDeleted)
      delete arg;
  } else {
    _list.erase(std::remove(_list.begin(), _list.end(), &var), _list.end());
  }

  if (_hashAssistedFind && sizeBefore != _list.size()) {
    _hashAssistedFind->erase(&var);
  }

  return sizeBefore != _list.size();
}



////////////////////////////////////////////////////////////////////////////////
/// Remove each argument in the input list from our list.
/// An exact pointer match is required, not just a match by name.
/// If `matchByNameOnly` is set, items will be looked up by name. In this case, if
/// the collection also owns the items, it will delete them.
/// Return false in case of problems.

bool RooAbsCollection::remove(const RooAbsCollection& list, bool /*silent*/, bool matchByNameOnly)
{

  auto oldSize = _list.size();
  std::vector<const RooAbsArg*> markedItems;

  if (matchByNameOnly) {

    // Instead of doing two passes on the list as in remove(RooAbsArg&), we do
    // everything in one pass, by using side effects of the predicate.
    auto nameMatchAndMark = [&list, &markedItems](const RooAbsArg* elm) {
      if( list.contains(*elm) ) {
        markedItems.push_back(elm);
        return true;
      }
      return false;
    };

    _list.erase(std::remove_if(_list.begin(), _list.end(), nameMatchAndMark), _list.end());

    std::set<const RooAbsArg*> toBeDeleted(markedItems.begin(), markedItems.end());
    if (_ownCont) {
      for (auto arg : toBeDeleted) {
        delete arg;
      }
    }
  }
  else {
    auto argMatchAndMark = [&list, &markedItems](const RooAbsArg* elm) {
      if( list.containsInstance(*elm) ) {
        markedItems.push_back(elm);
        return true;
      }
      return false;
    };

    _list.erase(std::remove_if(_list.begin(), _list.end(), argMatchAndMark), _list.end());
  }

  if (_hashAssistedFind && oldSize != _list.size()) {
    for( auto& var : markedItems ) {
      _hashAssistedFind->erase(var);
    }
  }

  return oldSize != _list.size();
}



////////////////////////////////////////////////////////////////////////////////
/// Remove all arguments from our set, deleting them if we own them.
/// This effectively restores our object to the state it would have
/// just after calling the RooAbsCollection(const char*) constructor.

void RooAbsCollection::removeAll()
{
  _hashAssistedFind = nullptr;

  if(_ownCont) {
    deleteList() ;
    _ownCont= false;
  }
  else {
    _list.clear();
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Set given attribute in each element of the collection by
/// calling each elements setAttribute() function.

void RooAbsCollection::setAttribAll(const Text_t* name, bool value)
{
  for (auto arg : _list) {
    arg->setAttribute(name, value);
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Create a subset of the current collection, consisting only of those
/// elements with the specified attribute set. The caller is responsibe
/// for deleting the returned collection

RooAbsCollection* RooAbsCollection::selectByAttrib(const char* name, bool value) const
{
  TString selName(GetName()) ;
  selName.Append("_selection") ;
  RooAbsCollection *sel = (RooAbsCollection*) create(selName.Data()) ;

  // Scan set contents for matching attribute
  for (auto arg : _list) {
    if (arg->getAttribute(name)==value)
      sel->add(*arg) ;
  }

  return sel ;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a subset of the current collection, consisting only of those
/// elements that are contained as well in the given reference collection.
/// Returns `true` only if something went wrong.
/// The complement of this function is getParameters().
/// \param[in] refColl The collection to check for common elements.
/// \param[out] outColl Output collection.

bool RooAbsCollection::selectCommon(const RooAbsCollection& refColl, RooAbsCollection& outColl) const
{
  outColl.clear();
  outColl.setName((std::string(GetName()) + "_selection").c_str());

  // Scan set contents for matching attribute
  for (auto arg : _list) {
    if (refColl.find(*arg))
      outColl.add(*arg) ;
  }

  return false;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a subset of the current collection, consisting only of those
/// elements that are contained as well in the given reference collection.
/// The caller is responsible for deleting the returned collection

RooAbsCollection* RooAbsCollection::selectCommon(const RooAbsCollection& refColl) const
{
  auto sel = static_cast<RooAbsCollection*>(create("")) ;
  selectCommon(refColl, *sel);
  return sel ;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a subset of the current collection, consisting only of those
/// elements with names matching the wildcard expressions in nameList,
/// supplied as a comma separated list

RooAbsCollection* RooAbsCollection::selectByName(const char* nameList, bool verbose) const
{
  // Create output set
  TString selName(GetName()) ;
  selName.Append("_selection") ;
  RooAbsCollection *sel = (RooAbsCollection*) create(selName.Data()) ;

  const size_t bufSize = strlen(nameList) + 1;
  std::vector<char> buf(bufSize);
  strlcpy(buf.data(),nameList,bufSize) ;
  char* wcExpr = strtok(buf.data(),",") ;
  while(wcExpr) {
    TRegexp rexp(wcExpr,true) ;
    if (verbose) {
      cxcoutD(ObjectHandling) << "RooAbsCollection::selectByName(" << GetName() << ") processing expression '" << wcExpr << "'" << std::endl;
    }

    RooFIter iter = fwdIterator() ;
    RooAbsArg* arg ;
    while((arg=iter.next())) {
      if (TString(arg->GetName()).Index(rexp)>=0) {
   if (verbose) {
     cxcoutD(ObjectHandling) << "RooAbsCollection::selectByName(" << GetName() << ") selected element " << arg->GetName() << std::endl;
   }
   sel->add(*arg) ;
      }
    }
    wcExpr = strtok(0,",") ;
  }

  return sel ;
}




////////////////////////////////////////////////////////////////////////////////
/// Check if this and other collection have identically-named contents

bool RooAbsCollection::equals(const RooAbsCollection& otherColl) const
{
  // First check equal length
  if (size() != otherColl.size()) return false ;

  // Then check that each element of our list also occurs in the other list
  auto compareByNamePtr = [](const RooAbsArg * left, const RooAbsArg * right) {
    return left->namePtr() == right->namePtr();
  };

  return std::is_permutation(_list.begin(), _list.end(),
      otherColl._list.begin(),
      compareByNamePtr);
}


namespace {
////////////////////////////////////////////////////////////////////////////////
/// Linear search through list of stored objects.
template<class Collection_t>
RooAbsArg* findUsingNamePointer(const Collection_t& coll, const TNamed* ptr) {
  auto findByNamePtr = [ptr](const RooAbsArg* elm) {
    return ptr == elm->namePtr();
  };

  auto item = std::find_if(coll.begin(), coll.end(), findByNamePtr);

  return item != coll.end() ? *item : nullptr;
}
}


////////////////////////////////////////////////////////////////////////////////
/// Find object with given name in list. A null pointer
/// is returned if no object with the given name is found.
RooAbsArg * RooAbsCollection::find(const char *name) const
{
  if (!name)
    return nullptr;

  // If an object with such a name exists, its name has been registered.
  const TNamed* nptr = RooNameReg::known(name);
  if (!nptr) return nullptr;

  if (_hashAssistedFind || _list.size() >= _sizeThresholdForMapSearch) {
    if (!_hashAssistedFind || !_hashAssistedFind->isValid()) {
      _hashAssistedFind = std::make_unique<HashAssistedFind>(_list.begin(), _list.end());
    }

    return _hashAssistedFind->find(nptr);
  }

  return findUsingNamePointer(_list, nptr);
}



////////////////////////////////////////////////////////////////////////////////
/// Find object with given name in list. A null pointer
/// is returned if no object with the given name is found.
RooAbsArg * RooAbsCollection::find(const RooAbsArg& arg) const
{
  const auto nptr = arg.namePtr();

  if (_hashAssistedFind || _list.size() >= _sizeThresholdForMapSearch) {
    if (!_hashAssistedFind || !_hashAssistedFind->isValid()) {
      _hashAssistedFind = std::make_unique<HashAssistedFind>(_list.begin(), _list.end());
    }

    return _hashAssistedFind->find(nptr);
  }

  return findUsingNamePointer(_list, nptr);
}


////////////////////////////////////////////////////////////////////////////////
/// Return index of item with given name, or -1 in case it's not in the collection.
Int_t RooAbsCollection::index(const char* name) const {
  const std::string theName(name);
  auto item = std::find_if(_list.begin(), _list.end(), [&theName](const RooAbsArg * elm){
    return elm->GetName() == theName;
  });
  return item != _list.end() ? item - _list.begin() : -1;
}


////////////////////////////////////////////////////////////////////////////////
/// Get value of a RooAbsReal stored in set with given name. If none is found, value of defVal is returned.
/// No error messages are printed unless the verbose flag is set

Double_t RooAbsCollection::getRealValue(const char* name, Double_t defVal, bool verbose) const
{
   RooAbsArg* raa = find(name) ;
   if (!raa) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::getRealValue(" << GetName() << ") ERROR no object with name '" << name << "' found" << std::endl;
      return defVal ;
   }
   RooAbsReal* rar = dynamic_cast<RooAbsReal*>(raa) ;
   if (!rar) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::getRealValue(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsReal" << std::endl;
      return defVal ;
   }
   return rar->getVal() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set value of a RooAbsRealLValye stored in set with given name to newVal
/// No error messages are printed unless the verbose flag is set

bool RooAbsCollection::setRealValue(const char* name, Double_t newVal, bool verbose)
{
   RooAbsArg* raa = find(name) ;
   if (!raa) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::setRealValue(" << GetName() << ") ERROR no object with name '" << name << "' found" << std::endl;
      return true ;
   }
   auto* rar = dynamic_cast<RooAbsRealLValue*>(raa) ;
   if (!rar) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::setRealValue(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsRealLValue" << std::endl;
      return true;
   }
   rar->setVal(newVal) ;
   return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Get state name of a RooAbsCategory stored in set with given name. If none is found, value of defVal is returned.
/// No error messages are printed unless the verbose flag is set

const char* RooAbsCollection::getCatLabel(const char* name, const char* defVal, bool verbose) const
{
   RooAbsArg* raa = find(name) ;
   if (!raa) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::getCatLabel(" << GetName() << ") ERROR no object with name '" << name << "' found" << std::endl;
      return defVal ;
   }
   auto* rac = dynamic_cast<RooAbsCategory*>(raa) ;
   if (!rac) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::getCatLabel(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsCategory" << std::endl;
      return defVal ;
   }
   return rac->getCurrentLabel() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set state name of a RooAbsCategoryLValue stored in set with given name to newVal.
/// No error messages are printed unless the verbose flag is set

bool RooAbsCollection::setCatLabel(const char* name, const char* newVal, bool verbose)
{
   RooAbsArg* raa = find(name) ;
   if (!raa) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::setCatLabel(" << GetName() << ") ERROR no object with name '" << name << "' found" << std::endl;
      return true ;
   }
   auto* rac = dynamic_cast<RooAbsCategoryLValue*>(raa) ;
   if (!rac) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::setCatLabel(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsCategory" << std::endl;
      return true ;
   }
   rac->setLabel(newVal) ;
   return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Get index value of a RooAbsCategory stored in set with given name. If none is found, value of defVal is returned.
/// No error messages are printed unless the verbose flag is set

Int_t RooAbsCollection::getCatIndex(const char* name, Int_t defVal, bool verbose) const
{
   RooAbsArg* raa = find(name) ;
   if (!raa) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::getCatLabel(" << GetName() << ") ERROR no object with name '" << name << "' found" << std::endl;
      return defVal ;
   }
   auto* rac = dynamic_cast<RooAbsCategory*>(raa) ;
   if (!rac) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::getCatLabel(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsCategory" << std::endl;
      return defVal ;
   }
   return rac->getCurrentIndex() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set index value of a RooAbsCategoryLValue stored in set with given name to newVal.
/// No error messages are printed unless the verbose flag is set

bool RooAbsCollection::setCatIndex(const char* name, Int_t newVal, bool verbose)
{
   RooAbsArg* raa = find(name) ;
   if (!raa) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::setCatLabel(" << GetName() << ") ERROR no object with name '" << name << "' found" << std::endl;
      return true ;
   }
   auto* rac = dynamic_cast<RooAbsCategoryLValue*>(raa) ;
   if (!rac) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::setCatLabel(" << GetName() << ") ERROR object '" << name << "' is not of type RooAbsCategory" << std::endl;
      return true ;
   }
   rac->setIndex(newVal) ;
   return false ;
}



////////////////////////////////////////////////////////////////////////////////
/// Get string value of a RooStringVar stored in set with given name. If none is found, value of defVal is returned.
/// No error messages are printed unless the verbose flag is set

const char* RooAbsCollection::getStringValue(const char* name, const char* defVal, bool verbose) const
{
   RooAbsArg* raa = find(name) ;
   if (!raa) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::getStringValue(" << GetName() << ") ERROR no object with name '" << name << "' found" << std::endl;
      return defVal ;
   }
   auto ras = dynamic_cast<const RooStringVar*>(raa) ;
   if (!ras) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::getStringValue(" << GetName() << ") ERROR object '" << name << "' is not of type RooStringVar" << std::endl;
      return defVal ;
   }

   return ras->getVal() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Set string value of a RooStringVar stored in set with given name to newVal.
/// No error messages are printed unless the verbose flag is set

bool RooAbsCollection::setStringValue(const char* name, const char* newVal, bool verbose)
{
   RooAbsArg* raa = find(name) ;
   if (!raa) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::setStringValue(" << GetName() << ") ERROR no object with name '" << name << "' found" << std::endl;
      return true ;
   }
   auto ras = dynamic_cast<RooStringVar*>(raa);
   if (!ras) {
      if (verbose) coutE(InputArguments) << "RooAbsCollection::setStringValue(" << GetName() << ") ERROR object '" << name << "' is not of type RooStringVar" << std::endl;
      return true ;
   }
   ras->setVal(newVal);

   return false;
}

////////////////////////////////////////////////////////////////////////////////
/// Return comma separated list of contained object names as STL string
std::string RooAbsCollection::contentsString() const
{
  std::string retVal ;
  for (auto arg : _list) {
    retVal += arg->GetName();
    retVal += ",";
  }

  retVal.erase(retVal.end()-1);

  return retVal;
}



////////////////////////////////////////////////////////////////////////////////
/// Return collection name

void RooAbsCollection::printName(std::ostream& os) const
{
  os << GetName() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return collection title

void RooAbsCollection::printTitle(std::ostream& os) const
{
  os << GetTitle() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return collection class name

void RooAbsCollection::printClassName(std::ostream& os) const
{
  os << IsA()->GetName() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Define default RooPrinable print options for given Print() flag string
/// For inline printing only show value of objects, for default print show
/// name,class name value and extras of each object. In verbose mode
/// also add object adress, argument and title

Int_t RooAbsCollection::defaultPrintContents(Option_t* opt) const
{
  if (opt && TString(opt)=="I") {
    return kValue ;
  }
  if (opt && TString(opt).Contains("v")) {
    return kAddress|kName|kArgs|kClassName|kValue|kTitle|kExtras ;
  }
  return kName|kClassName|kValue ;
}





////////////////////////////////////////////////////////////////////////////////
/// Print value of collection, i.e. a comma separated list of contained
/// object names

void RooAbsCollection::printValue(std::ostream& os) const
{
  bool first2(true) ;
  os << "(" ;
  for (auto arg : _list) {
    if (!first2) {
      os << "," ;
    } else {
      first2 = false ;
    }
    if (arg->IsA()->InheritsFrom(RooStringVar::Class())) {
       os << '\'' << ((RooStringVar *)arg)->getVal() << '\'';
    } else {
       os << arg->GetName();
    }
  }
  os << ")" ;
}



////////////////////////////////////////////////////////////////////////////////
/// Implement multiline printing of collection, one line for each contained object showing
/// the requested content

void RooAbsCollection::printMultiline(std::ostream&os, Int_t contents, bool /*verbose*/, TString indent) const
{
  if (TString(GetName()).Length()>0 && (contents&kCollectionHeader)) {
    os << indent << ClassName() << "::" << GetName() << ":" << (_ownCont?" (Owning contents)":"") << std::endl;
  }

  TString deeper(indent);
  deeper.Append("     ");

  // Adjust the width of the name field to fit the largest name, if requested
  Int_t maxNameLen(1) ;
  Int_t nameFieldLengthSaved = RooPrintable::_nameLength ;
  if (nameFieldLengthSaved==0) {
    for (auto next : _list) {
      Int_t len = strlen(next->GetName()) ;
      if (len>maxNameLen) maxNameLen = len ;
    }
    RooPrintable::nameFieldLength(maxNameLen+1) ;
  }

  unsigned int idx = 0;
  for (auto next : _list) {
    os << indent << std::setw(3) << ++idx << ") ";
    next->printStream(os,contents,kSingleLine,"");
  }

  // Reset name field length, if modified
  RooPrintable::nameFieldLength(nameFieldLengthSaved) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Base contents dumper for debugging purposes

void RooAbsCollection::dump() const
{
  for (auto arg : _list) {
    std::cout << arg << " " << arg->IsA()->GetName() << "::" << arg->GetName() << " (" << arg->GetTitle() << ")" << std::endl ;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Output content of collection as LaTex table. By default a table with two columns is created: the left
/// column contains the name of each variable, the right column the value.
///
/// The following optional named arguments can be used to modify the default behavior
/// <table>
/// <tr><th> Argument <th> Effect
/// <tr><td>   `Columns(Int_t ncol)`                    <td> Fold table into multiple columns, i.e. ncol=3 will result in 3 x 2 = 6 total columns
/// <tr><td>   `Sibling(const RooAbsCollection& other)` <td> Define sibling list.
///     The sibling list is assumed to have objects with the same
///     name in the same order. If this is not the case warnings will be printed. If a single
///     sibling list is specified, 3 columns will be output: the (common) name, the value of this
///     list and the value in the sibling list. Multiple sibling lists can be specified by
///     repeating the Sibling() command.
/// <tr><td>   `Format(const char* str)`                <td> Classic format string, provided for backward compatibility
/// <tr><td>   `Format()`                            <td> Formatting arguments.
///   <table>
///   <tr><td>   const char* what          <td> Controls what is shown. "N" adds name, "E" adds error,
///                                  "A" shows asymmetric error, "U" shows unit, "H" hides the value
///   <tr><td>   `FixedPrecision(int n)`     <td> Controls precision, set fixed number of digits
///   <tr><td>   `AutoPrecision(int n)`      <td> Controls precision. Number of shown digits is calculated from error
///                                               and n specified additional digits (1 is sensible default)
///   <tr><td>   `VerbatimName(bool flag)` <td> Put variable name in a \\verb+   + clause.
///   </table>
/// <tr><td>   `OutputFile(const char* fname)`          <td> Send output to file with given name rather than standard output
///
/// </table>
///
/// Example use:
/// ```
/// list.printLatex(Columns(2), Format("NEU",AutoPrecision(1),VerbatimName()) );
/// ```

void RooAbsCollection::printLatex(const RooCmdArg& arg1, const RooCmdArg& arg2,
              const RooCmdArg& arg3, const RooCmdArg& arg4,
              const RooCmdArg& arg5, const RooCmdArg& arg6,
              const RooCmdArg& arg7, const RooCmdArg& arg8) const
{


  // Define configuration for this method
  RooCmdConfig pc("RooAbsCollection::printLatex()") ;
  pc.defineInt("ncol","Columns",0,1) ;
  pc.defineString("outputFile","OutputFile",0,"") ;
  pc.defineString("format","Format",0,"NEYVU") ;
  pc.defineInt("sigDigit","Format",0,1) ;
  pc.defineObject("siblings","Sibling",0,0,true) ;
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
  if (!pc.ok(true)) {
    return ;
  }

  const char* outFile = pc.getString("outputFile") ;
  if (outFile && strlen(outFile)) {
    std::ofstream ofs(outFile) ;
    if (pc.hasProcessed("FormatArgs")) {
      auto* formatCmd = static_cast<RooCmdArg*>(cmdList.FindObject("FormatArgs")) ;
      formatCmd->addArg(RooFit::LatexTableStyle()) ;
      printLatex(ofs,pc.getInt("ncol"),0,0,pc.getObjectList("siblings"),formatCmd) ;
    } else {
      printLatex(ofs,pc.getInt("ncol"),pc.getString("format"),pc.getInt("sigDigit"),pc.getObjectList("siblings")) ;
    }
  } else {
    if (pc.hasProcessed("FormatArgs")) {
      auto* formatCmd = static_cast<RooCmdArg*>(cmdList.FindObject("FormatArgs")) ;
      formatCmd->addArg(RooFit::LatexTableStyle()) ;
      printLatex(std::cout,pc.getInt("ncol"),0,0,pc.getObjectList("siblings"),formatCmd) ;
    } else {
      printLatex(std::cout,pc.getInt("ncol"),pc.getString("format"),pc.getInt("sigDigit"),pc.getObjectList("siblings")) ;
    }
  }
}




////////////////////////////////////////////////////////////////////////////////
/// Internal implementation function of printLatex

void RooAbsCollection::printLatex(std::ostream& ofs, Int_t ncol, const char* option, Int_t sigDigit, const RooLinkedList& siblingList, const RooCmdArg* formatCmd) const
{
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
    TString tmp = formatCmd->getString(0) ;
    tmp.ReplaceAll("N","") ;
    tmp.ReplaceAll("n","") ;
    static char buf[100] ;
    strlcpy(buf,tmp.Data(),100) ;
    sibFormatCmd.setString(0, buf);
  }


  // Make list of lists ;
  RooLinkedList listList ;
  listList.Add((RooAbsArg*)this) ;
  for(auto * col : static_range_cast<RooAbsCollection*>(siblingList)) {
    listList.Add(col) ;
  }

  RooLinkedList listListRRV ;

  // Make list of RRV-only components
  RooArgList* prevList = 0 ;
  for(auto * col : static_range_cast<RooAbsCollection*>(listList)) {
    RooArgList* list = new RooArgList ;
    RooFIter iter = col->fwdIterator() ;
    RooAbsArg* arg ;
    while((arg=iter.next())) {

      auto* rrv = dynamic_cast<RooRealVar*>(arg) ;
      if (rrv) {
   list->add(*rrv) ;
      } else {
   coutW(InputArguments) << "RooAbsCollection::printLatex: can only print RooRealVar in LateX, skipping non-RooRealVar object named "
        << arg->GetName() << std::endl;
      }
      if (prevList && TString(rrv->GetName()).CompareTo(prevList->at(list->getSize()-1)->GetName())) {
   coutW(InputArguments) << "RooAbsCollection::printLatex: WARNING: naming and/or ordering of sibling list is different" << std::endl;
      }
    }
    listListRRV.Add(list) ;
    if (prevList && list->size() != prevList->size()) {
      coutW(InputArguments) << "RooAbsCollection::printLatex: ERROR: sibling list(s) must have same length as self" << std::endl;
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
  ofs << header << std::endl;


  // Print contents, delegating actual printing to RooRealVar::format()
  for (i=0 ; i<nrow ; i++) {
    for (j=0 ; j<ncol ; j++) {
      for (k=0 ; k<nlist ; k++) {
   RooRealVar* par = (RooRealVar*) ((RooArgList*)listListRRV.At(k))->at(i+j*nrow) ;
   if (par) {
     if (option) {
       ofs << *std::unique_ptr<TString>{par->format(sigDigit,(k==0)?option:sibOption.Data())};
     } else {
       ofs << *std::unique_ptr<TString>{par->format((k==0)?*formatCmd:sibFormatCmd)};
     }
   }
   if (!(j==ncol-1 && k==nlist-1)) {
     ofs << " & " ;
   }
      }
    }
    ofs << "\\\\" << std::endl;
  }

  ofs << "\\end{tabular}" << std::endl;
  listListRRV.Delete() ;
}




////////////////////////////////////////////////////////////////////////////////
/// Return true if all contained object report to have their
/// value inside the specified range

bool RooAbsCollection::allInRange(const char* rangeSpec) const
{
  if (!rangeSpec) return true ;

  // Parse rangeSpec specification
  std::vector<std::string> cutVec ;
  if (rangeSpec && strlen(rangeSpec)>0) {
    if (strchr(rangeSpec,',')==0) {
      cutVec.push_back(rangeSpec) ;
    } else {
      const size_t bufSize = strlen(rangeSpec)+1;
      std::vector<char> buf(bufSize);
      strlcpy(buf.data(),rangeSpec,bufSize) ;
      const char* oneRange = strtok(buf.data(),",") ;
      while(oneRange) {
   cutVec.push_back(oneRange) ;
   oneRange = strtok(0,",") ;
      }
    }
  }

  // Apply range based selection criteria
  bool selectByRange = true ;
  for (auto arg : _list) {
    bool selectThisArg = false ;
    UInt_t icut ;
    for (icut=0 ; icut<cutVec.size() ; icut++) {
      if (arg->inRange(cutVec[icut].c_str())) {
   selectThisArg = true ;
   break ;
      }
    }
    if (!selectThisArg) {
      selectByRange = false ;
      break ;
    }
  }

  return selectByRange ;
}

////////////////////////////////////////////////////////////////////////////////
/// If one of the TObject we have a referenced to is deleted, remove the
/// reference.

void RooAbsCollection::RecursiveRemove(TObject *obj)
{
   if (obj && obj->InheritsFrom(RooAbsArg::Class())) remove(*(RooAbsArg*)obj,false,false);
}

////////////////////////////////////////////////////////////////////////////////
/// Sort collection using std::sort and name comparison

void RooAbsCollection::sort(bool reverse) {
  //Windows seems to need an implementation where two different std::sorts are written
  //down in two different blocks. Switching between the two comparators using a ternary
  //operator does not compile on windows, although the signature is identical.
  if (reverse) {
    const auto cmpReverse = [](const RooAbsArg * l, const RooAbsArg * r) {
      return strcmp(l->GetName(), r->GetName()) > 0;
    };

    std::sort(_list.begin(), _list.end(), cmpReverse);
  }
  else {
    const auto cmp = [](const RooAbsArg * l, const RooAbsArg * r) {
      return strcmp(l->GetName(), r->GetName()) < 0;
    };

    std::sort(_list.begin(), _list.end(), cmp);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// Sort collection topologically: the servers of any RooAbsArg will be before
/// that RooAbsArg in the collection. Will throw an exception if servers
/// are missing in the collection.

void RooAbsCollection::sortTopologically() {
   std::unordered_set<TNamed const *> seenArgs;
   for (std::size_t iArg = 0; iArg < _list.size(); ++iArg) {
      RooAbsArg *arg = _list[iArg];
      bool movedArg = false;
      for (RooAbsArg *server : arg->servers()) {
         if (seenArgs.find(server->namePtr()) == seenArgs.end()) {
            auto found = std::find_if(_list.begin(), _list.end(),
                                      [server](RooAbsArg *elem) { return elem->namePtr() == server->namePtr(); });
            if (found == _list.end()) {
               std::stringstream ss;
               ss << "RooAbsArg \"" << arg->GetName() << "\" depends on \"" << server->GetName()
                  << "\", but this arg is missing in the collection!";
               throw std::runtime_error(ss.str());
            }
            _list.erase(found);
            _list.insert(_list.begin() + iArg, server);
            movedArg = true;
            break;
         }
      }
      if (movedArg) {
         --iArg;
         continue;
      }
      seenArgs.insert(arg->namePtr());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Factory for legacy iterators.

std::unique_ptr<RooAbsCollection::LegacyIterator_t> RooAbsCollection::makeLegacyIterator (bool forward) const {
  if (!forward)
    ccoutE(DataHandling) << "The legacy RooFit collection iterators don't support reverse iterations, any more. "
    << "Use begin() and end()" << std::endl;
  return std::make_unique<LegacyIterator_t>(_list);
}


////////////////////////////////////////////////////////////////////////////////
/// Insert an element into the owned collections.
void RooAbsCollection::insert(RooAbsArg* item) {
  _list.push_back(item);

  if (_allRRV && dynamic_cast<const RooRealVar*>(item)==0) {
    _allRRV= false;
  }

  if (_hashAssistedFind) {
    _hashAssistedFind->insert(item);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// \param[in] flag Switch hash map on or off.
void RooAbsCollection::useHashMapForFind(bool flag) const {
  if (flag && !_hashAssistedFind) _hashAssistedFind = std::make_unique<HashAssistedFind>(_list.begin(), _list.end());
  if (!flag) _hashAssistedFind = nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Check that all entries where the collections overlap have the same name.
bool RooAbsCollection::hasSameLayout(const RooAbsCollection& other) const {
  for (unsigned int i=0; i < std::min(_list.size(), other.size()); ++i) {
    if (_list[i]->namePtr() != other._list[i]->namePtr())
      return false;
  }

  return true;
}
