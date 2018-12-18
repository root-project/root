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

/// Implementation used in RooAbsArg. Needs streamers.
ClassImp(RooSTLRefCountList<RooAbsArg>);

namespace STLRefCountListHelpers {
  /// This converter only gives out lists with T=RooAbsArg, because this
  /// is what the old ref count list can hold.
  RooSTLRefCountList<RooAbsArg> convert(const RooRefCountList& old) {
    RooSTLRefCountList<RooAbsArg> newList;
    newList.reserve(old.GetSize());

    auto it = old.fwdIterator();
    for (RooAbsArg * elm = it.next(); elm != nullptr; elm = it.next()) {
      newList.Add(elm, old.refCount(elm));
    }

    return newList;
  }
}

