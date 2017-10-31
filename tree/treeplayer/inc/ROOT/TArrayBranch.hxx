// Author: Enrico Guiraud CERN  10/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDF_TARRAYBRANCH
#define ROOT_TDF_TARRAYBRANCH

#include "TTreeReaderArray.h"
#include <iterator>

namespace ROOT {
namespace Experimental {
namespace TDF {
/// When using TDataFrame to read data from a ROOT file, users can specify that the type of a branch is TArrayBranch<T>
/// to indicate the branch is a c-style array, an STL array or any other type that can/must be accessed through a
/// TTreeReaderArray<T> (as opposed to a TTreeReaderValue<T>).
/// Column values of type TArrayBranch perform no copy of the underlying array data and offer a minimal array-like
/// interface to access the array elements: either via square brackets, or with C++11 range-based for loops.
template <typename T>
class TArrayBranch {
   TTreeReaderArray<T> *fReaderArray = nullptr; ///< Pointer to the TTreeReaderArray that actually owns the data
public:
   using iterator = typename TTreeReaderArray<T>::iterator;
   using const_iterator = typename TTreeReaderArray<T>::const_iterator;
   using value_type = T;

   TArrayBranch() {} // jitted types must be default-constructible
   TArrayBranch(TTreeReaderArray<T> &arr) : fReaderArray(&arr)
   {
      // trigger loading of entry into TTreeReaderArray
      // TODO: could we load more lazily? we need to guarantee that when a TArrayBranch is constructed all data
      // is loaded into the TTreeReaderArray because otherwise Snapshot never triggers this load.
      // Should we have Snapshot explicitly trigger the loading instead?
      // N.B. we would not need to trigger the load explicitly if Snapshot did not read the buffer returned by GetData
      // directly -- e.g. if we wrote a std::vector for each c-style array in input
      arr.At(0);
   }

   const T &operator[](std::size_t n) const { return fReaderArray->At(n); }

   // TODO: remove the need of GetData, e.g. by writing out std::vectors instead of c-style arrays
   T *GetData() { return static_cast<T *>(fReaderArray->GetAddress()); }

   iterator begin() { return fReaderArray->begin(); }
   iterator end() { return fReaderArray->end(); }
   const_iterator begin() const { return fReaderArray->cbegin(); }
   const_iterator end() const { return fReaderArray->cend(); }

   std::size_t size() const { return fReaderArray->GetSize(); }
};
}
}
}

#endif
