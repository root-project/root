/*
 * RooRefCountListNew.h
 *
 *  Created on: 13 Dec 2018
 *      Author: shageboe
 */

#ifndef ROOFIT_ROOFITCORE_INC_ROOSTLREFCOUNTLIST_H_
#define ROOFIT_ROOFITCORE_INC_ROOSTLREFCOUNTLIST_H_
#include "Rtypes.h"

#include <vector>
#include <algorithm>
#include <assert.h>


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

    RooSTLRefCountList() {}
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
        _storage.emplace_back(obj);
        _refCount.emplace_back(initialCount);
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


    ///Direct reference to container of objects held by this list.
    const Container_t& containedObjects() const {
      return _storage;
    }


    ///Number of contained objects (neglecting the ref count).
    std::size_t size() const {
      assert(_storage.size() == _refCount.size());

      return _storage.size();
    }

    void reserve(std::size_t size) {
      _storage.reserve(size);
      _refCount.reserve(size);
    }


    ///Check if empty.
    bool empty() const {
      return _storage.empty();
    }


    ///Find an item by comparing its adress.
    template<typename Obj_t>
    typename Container_t::const_iterator findByPointer(const Obj_t * item) const {
      auto byPointer = [item](const T * listItem) {
        return listItem == item;
      };

      return std::find_if(_storage.begin(), _storage.end(), byPointer);
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
    typename Container_t::const_iterator findByNamePointer(const T * item) const {
      auto nptr = item->namePtr();
      auto byNamePointer = [nptr](const T * element) {
        return element->namePtr() == nptr;
      };

      return std::find_if(_storage.begin(), _storage.end(), byNamePointer);
    }


    ///Check if list contains an item using findByPointer().
    template<typename Obj_t>
    bool containsByPointer(const Obj_t * obj) const {
      return findByPointer(obj) != _storage.end();
    }


    ///Check if list contains an item using findByNamePointer().
    bool containsByNamePtr(const T * obj) const {
      return findByNamePointer(obj) != _storage.end();
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
          _storage.erase(item);
          _refCount.erase(_refCount.begin() + pos);
        }
      }
    }


    ///Remove from list irrespective of ref count.
    void RemoveAll(const T * obj) {
      Remove(obj, true);
    }


  private:
    Container_t _storage;
    std::vector<std::size_t> _refCount;

    ClassDef(RooSTLRefCountList<T>,1);
};



class RooAbsArg;
class RooRefCountList;

namespace STLRefCountListHelpers {
  /// Converter from the old RooRefCountList to RooSTLRefCountList.
  RooSTLRefCountList<RooAbsArg> convert(const RooRefCountList& old);
}

#endif /* ROOFIT_ROOFITCORE_INC_ROOSTLREFCOUNTLIST_H_ */
