/*
 * RooSTLRefCountList.cxx
 *
 *  Created on: 13 Dec 2018
 *      Author: shageboe
 */

#include "RooSTLRefCountList.h"

#include "RooRefCountList.h"
#include "RooLinkedListIter.h"
#include "RooAbsArg.h"
#include <string>


/**
 * \class RooSTLRefCountList
 * The RooSTLRefCountList is a simple collection of pointers and ref counters.
 * The pointers are not owned, hence not deleted when removed from the collection.
 * Objects can be searched for either by pointer or by name (confusion possible when
 * objects with same name are present). This replicates the behaviour of the RooRefCountList.
 */

ClassImp(RooSTLRefCountList<RooAbsArg>);


template<class T>
RooSTLRefCountList<T>& RooSTLRefCountList<T>::operator=(const RooRefCountList & other) {
  _storage.clear();
  _refCount.clear();

  _storage.reserve(other.GetSize());
  _refCount.reserve(other.GetSize());

  auto it = other.fwdIterator();
  for (RooAbsArg * elm = it.next(); elm != nullptr; elm = it.next()) {
    _storage.push_back(elm);
    _refCount.push_back(static_cast<typename decltype(_refCount)::value_type>(
        other.refCount(elm)));
  }

  return *this;
}

template<class T>
typename RooSTLRefCountList<T>::Container_t::const_iterator
RooSTLRefCountList<T>::findByName(const char * name) const {
  //If this turns out to be a bottleneck,
  //one could use the RooNameReg to obtain the pointer to the arg's and compare these
  const std::string theName(name);
  auto byName = [&theName](const T * element) {
    return element->GetName() == theName;
  };

  return std::find_if(_storage.begin(), _storage.end(), byName);
}

template<class T>
typename RooSTLRefCountList<T>::Container_t::const_iterator
RooSTLRefCountList<T>::findByNamePointer(const T * obj) const {
  auto nptr = obj->namePtr();
  auto byNamePointer = [nptr](const T * element) {
    return element->namePtr() == nptr;
  };

  return std::find_if(_storage.begin(), _storage.end(), byNamePointer);
}


template class RooSTLRefCountList<RooAbsArg>;
