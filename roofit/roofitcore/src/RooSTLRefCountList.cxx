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
 * The RooSTLRefCountList is a simple collection of **pointers** to the template objects and
 * corresponding ref counters.
 * The pointees are not owned, hence not deleted when removed from the collection.
 * Objects can be searched for either by pointer or by name (confusion possible when
 * objects with same name are present). This replicates the behaviour of the RooRefCountList.
 */

ClassImp(RooSTLRefCountList<RooAbsArg>);


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

// An explicit instantiation is needed because the two functions above
// cannot be in the header. They have a circular dependency with RooAbsArg,
// which contains a RooSTLRefCountList<RooAbsArg>.
template class RooSTLRefCountList<RooAbsArg>;
