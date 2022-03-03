// Author: Stephan Hageboeck, CERN, 12/2018
/*****************************************************************************
 * Project: RooFit                                                           *
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

#ifndef ROOFIT_ROOFITCORE_INC_ROOSTLREFCOUNTLIST_H_
#define ROOFIT_ROOFITCORE_INC_ROOSTLREFCOUNTLIST_H_

#include "RooNameReg.h"

#include "Rtypes.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cassert>


/**
 * \class RooSTLRefCountList
 * The RooSTLRefCountList is a simple collection of **pointers** to the template objects with
 * reference counters.
 * The pointees are not owned, hence not deleted when removed from the collection.
 * Objects can be searched for either by pointer or by name (confusion possible when
 * objects with same name are present). This replicates the behaviour of the RooRefCountList.
 */

template <class T>
class RooSTLRefCountList {
  public:
    using Container_t = std::vector<T*>;

    static constexpr std::size_t minSizeForNamePointerOrdering = 7;

    RooSTLRefCountList() {
      // The static _renameCounter member gets connected to the RooNameReg as
      // soon as the first RooSTLRefCountList instance is constructed.
      if(_renameCounter == nullptr) _renameCounter =
        &RooNameReg::renameCounter();
    }
    RooSTLRefCountList(const RooSTLRefCountList&) = default;
    RooSTLRefCountList& operator=(const RooSTLRefCountList&) = default;
    RooSTLRefCountList& operator=(RooSTLRefCountList&&) = default;

    virtual ~RooSTLRefCountList() {}

    ///Add an object or increase refCount if it is already present. Only compares
    ///pointers to check for existing objects
    void Add(T * obj, std::size_t initialCount = 1) {
      auto foundItem = findByPointer(obj);

      if (foundItem != _storage.end()) {
        _refCount[foundItem - _storage.begin()] += initialCount;
      }
      else {
        if(!_orderedStorage.empty()) {
          _orderedStorage.insert(lowerBoundByNamePointer(obj), obj);
        }
        _storage.push_back(obj);
        _refCount.push_back(initialCount);
      }
    }


    ///Return ref count of item that iterator points to.
    std::size_t refCount(typename Container_t::const_iterator item) const {
      assert(_storage.size() == _refCount.size());

      return item != _storage.end() ? _refCount[item - _storage.begin()] : 0;
    }


    ///Return ref count of item with given address.
    template<typename Obj_t>
    std::size_t refCount(const Obj_t * obj) const {
      return refCount(findByPointer(obj));
    }

    ///Iterator over contained objects.
    typename Container_t::const_iterator begin() const {
      return _storage.begin();
    }

    ///End of contained objects.
    typename Container_t::const_iterator end() const {
      return _storage.end();
    }

    /// Retrieve an element from the list.
    typename Container_t::value_type operator[](std::size_t index) const {
      return _storage[index];
    }


    ///Direct reference to container of objects held by this list.
    const Container_t& containedObjects() const {
      return _storage;
    }


    ///Number of contained objects (neglecting the ref count).
    std::size_t size() const {
      assert(_storage.size() == _refCount.size());

      return _storage.size();
    }

    void reserve(std::size_t amount) {
      _storage.reserve(amount);
      _refCount.reserve(amount);
      _orderedStorage.reserve(amount);
    }


    ///Check if empty.
    bool empty() const {
      return _storage.empty();
    }


    ///Find an item by comparing its adress.
    template<typename Obj_t>
    typename Container_t::const_iterator findByPointer(const Obj_t * item) const {
      return std::find(_storage.begin(), _storage.end(), item);
    }


    ///Find an item by comparing strings returned by RooAbsArg::GetName()
    typename Container_t::const_iterator findByName(const char * name) const {
      //If this turns out to be a bottleneck,
      //one could use the RooNameReg to obtain the pointer to the arg's name and compare these
      const std::string theName(name);
      auto byName = [&theName](const T * element) {
        return element->GetName() == theName;
      };

      return std::find_if(_storage.begin(), _storage.end(), byName);
    }


    ///Find an item by comparing RooAbsArg::namePtr() adresses.
    T* findByNamePointer(const T * item) const {
      if(size() < minSizeForNamePointerOrdering) {
        auto nptr = item->namePtr();
        auto byNamePointer = [nptr](const T * element) {
          return element->namePtr() == nptr;
        };

        auto found = std::find_if(_storage.begin(), _storage.end(), byNamePointer);
        return found != _storage.end() ? *found : nullptr;
      } else {
        //As the collection is guaranteed to be sorted by namePtr() adress, we
        //can use a binary seach to look for `item` in this collection.
        auto first = lowerBoundByNamePointer(item);
        if(first == _orderedStorage.end()) return nullptr;
        if(item->namePtr() != (*first)->namePtr()) return nullptr;
        return *first;
      }
    }


    ///Check if list contains an item using findByPointer().
    template<typename Obj_t>
    bool containsByPointer(const Obj_t * obj) const {
      return findByPointer(obj) != _storage.end();
    }


    ///Check if list contains an item using findByNamePointer().
    bool containsByNamePtr(const T * obj) const {
      return findByNamePointer(obj);
    }


    ///Check if list contains an item using findByName().
    bool containsSameName(const char * name) const {
      return findByName(name) != _storage.end();
    }


    ///Decrease ref count of given object. Shrink list if ref count reaches 0.
    ///\param obj Decrease ref count of given object. Compare by pointer.
    ///\param force If true, remove irrespective of ref count.
    void Remove(const T * obj, bool force = false) {
      auto item = findByPointer(obj);

      if (item != _storage.end()) {
        const std::size_t pos = item - _storage.begin();

        if (force || --_refCount[pos] == 0) {
          //gcc4.x doesn't know how to erase at the position of a const_iterator
          //Therefore, erase at begin + pos instead of 'item'
          _storage.erase(_storage.begin() + pos);
          _refCount.erase(_refCount.begin() + pos);
          if(!_orderedStorage.empty()) {
            // For the ordered storage, we could find by name pointer addres
            // with binary search, but this will not work anymore if one of the
            // object pointers in this collection is dangling (can happen in
            // RooFit...). However, the linear search by object address is
            // acceptable, because we already do a linear search through
            // _storage at the beginning of Remove().
            _orderedStorage.erase(std::find(_orderedStorage.begin(), _orderedStorage.end(), obj));
          }
        }
      }
    }


    ///Remove from list irrespective of ref count.
    void RemoveAll(const T * obj) {
      Remove(obj, true);
    }


  private:
    //Return an iterator to the last element in this sorted collection with a
    //RooAbsArg::namePtr() adress smaller than for `item`.
    typename std::vector<T*>::const_iterator lowerBoundByNamePointer(const T * item) const {

      //If the _orderedStorage has not been initliazed yet or needs resorting
      //for other reasons, (re-)initialize it now.
      if(orderedStorageNeedsSorting() || _orderedStorage.size() != _storage.size()) initializeOrderedStorage();

      return std::lower_bound(_orderedStorage.begin(), _orderedStorage.end(), item->namePtr(),
             [](const auto& x, TNamed const* npt) -> bool {
                return x->namePtr() < npt;
              });
    }

    bool orderedStorageNeedsSorting() const {
      //If an RooAbsArg in this collection was renamed, the collection might
      //not be sorted correctly anymore! The solution: everytime any RooAbsArg
      //is renamed, the RooNameReg::renameCounter gets incremented. The
      //RooSTLRefCountList keeps track at which value of the counter it has
      //been sorted last. If the counter increased in the meantime, a
      //re-sorting is due.
      return _renameCounterForLastSorting != *_renameCounter;
    }

    void initializeOrderedStorage() const {
      _orderedStorage.clear();
      _orderedStorage.reserve(_storage.size());
      for(std::size_t i = 0; i < _storage.size(); ++i) {
        _orderedStorage.push_back(_storage[i]);
      }
      std::sort(_orderedStorage.begin(), _orderedStorage.end(),
              [](auto& a, auto& b) {
                return a->namePtr() != b->namePtr() ? a->namePtr() < b->namePtr() : a < b;
              });
      _renameCounterForLastSorting = *_renameCounter;
    }

    Container_t _storage;
    std::vector<UInt_t> _refCount;
    mutable std::vector<T*> _orderedStorage; //!
    mutable unsigned long _renameCounterForLastSorting = 0; //!

    // It is expensive to access the RooNameReg instance to get the counter for
    // the renameing operations. That's why we have out own static pointer to
    // the counter.
    static std::size_t const* _renameCounter;

    ClassDef(RooSTLRefCountList<T>,3);
};

template<class T>
std::size_t const* RooSTLRefCountList<T>::_renameCounter = nullptr;

class RooAbsArg;
class RooRefCountList;

namespace RooFit {
namespace STLRefCountListHelpers {
  /// Converter from the old RooRefCountList to RooSTLRefCountList.
  RooSTLRefCountList<RooAbsArg> convert(const RooRefCountList& old);
}
}

#endif /* ROOFIT_ROOFITCORE_INC_ROOSTLREFCOUNTLIST_H_ */
