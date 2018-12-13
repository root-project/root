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

class RooRefCountList;


template <class T>
class RooSTLRefCountList {
  public:
    using Container_t =  std::vector<T *>;

    RooSTLRefCountList() {}
    RooSTLRefCountList(const RooSTLRefCountList&) = default;
    RooSTLRefCountList& operator=(const RooSTLRefCountList&) = default;

    virtual ~RooSTLRefCountList() {}

    void Add(T * obj, std::size_t initialRefCount = 1) {
      auto foundItem = findByPointer(obj);

      if (foundItem != _storage.end()) {
        ++_refCount[foundItem - _storage.begin()];
      }
      else {
        _storage.emplace_back(obj);
        _refCount.emplace_back(initialRefCount);
      }
    }

    RooSTLRefCountList<T>& operator=(const RooRefCountList & other);

    std::size_t refCount(typename Container_t::const_iterator item) const {
      assert(_storage.size() == _refCount.size());

      return item != _storage.end() ? _refCount[item - _storage.begin()] : 0;
    }

    template<typename Obj_t>
    std::size_t refCount(const Obj_t * obj) const {
      return refCount(findByPointer(obj));
    }

    typename Container_t::const_iterator begin() const {
      return _storage.begin();
    }

    typename Container_t::const_iterator end() const {
      return _storage.end();
    }

    std::size_t size() const {
      assert(_storage.size() == _refCount.size());

      return _storage.size();
    }

//    template<typename Obj_t>
//    typename Container_t::iterator findByPointer(const Obj_t * item) {
//      auto byPointer = [item](const T * listItem) {
//        return listItem == item;
//      };
//
//      return std::find_if(_storage.begin(), _storage.end(), byPointer);
//    }

    template<typename Obj_t>
    typename Container_t::const_iterator findByPointer(const Obj_t * item) const {
      auto byPointer = [item](const T * listItem) {
        return listItem == item;
      };

      return std::find_if(_storage.begin(), _storage.end(), byPointer);
    }

    typename Container_t::const_iterator findByName(const char * name) const;

    typename Container_t::const_iterator findByNamePointer(const T * item) const;

    template<typename Obj_t>
    bool containsByPointer(const Obj_t * obj) const {
      return findByPointer(obj) != _storage.end();
    }

    bool containsByNamePtr(const T * obj) const {
      return findByNamePointer(obj) != _storage.end();
    }

    bool containsSameName(const char * name) const {
      return findByName(name) != _storage.end();
    }

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

    void RemoveAll(const T * obj) {
      Remove(obj, true);
    }


  private:
    Container_t _storage;
    std::vector<std::size_t> _refCount;

    ClassDef(RooSTLRefCountList<T>,1);
};



#endif /* ROOFIT_ROOFITCORE_INC_ROOSTLREFCOUNTLIST_H_ */
