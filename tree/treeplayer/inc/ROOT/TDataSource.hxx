// Author: Enrico Guiraud, Danilo Piparo CERN  09/2017

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDATASOURCE
#define ROOT_TDATASOURCE

#include "RStringView.h"
#include "RtypesCore.h" // ULong64_t
#include <algorithm> // std::transform
#include <vector>
#include <typeinfo>

namespace ROOT {
namespace Experimental {
namespace TDF {

class TDataSource {
public:
   virtual ~TDataSource(){};
   virtual const std::vector<std::string> &GetColumnNames() const = 0;
   virtual bool HasColumn(std::string_view) const = 0;
   /// Type of a column as a string, e.g. `GetTypeName("x") == "double"`. Required for jitting e.g. `df.Filter("x>0")`.
   virtual std::string GetTypeName(std::string_view) const = 0;
   /// Called at most once per column by TDF. Return vector of pointers to pointers to column values - one per slot.
   template <typename T>
   std::vector<T **> GetColumnReaders(std::string_view name, unsigned int nSlots)
   {
      auto typeErasedVec = GetColumnReadersImpl(name, nSlots, typeid(T));
      std::vector<T **> typedVec(typeErasedVec.size());
      std::transform(typeErasedVec.begin(), typeErasedVec.end(), typedVec.begin(),
                     [](void *p) { return static_cast<T **>(p); });
      return typedVec;
   }
   /// Return chunks of entries to distribute to tasks. They are required to be continguous intervals with no entries
   /// skipped, starting at 0 and ending at nEntries, e.g. [0-5],[5-10] for 10 entries.
   virtual const std::vector<std::pair<ULong64_t, ULong64_t>> &GetEntryRanges() const = 0;
   /// Different threads will loop over different ranges and will pass different "slot" values.
   virtual void SetEntry(ULong64_t entry, unsigned int slot) = 0;
   /// Convenience method called at the start of each task, before processing a range of entries.
   /// DataSources can implement it if needed (does nothing by default).
   /// firstEntry is the first entry of the range that the task will process.
   virtual void InitSlot(unsigned int /*slot*/, ULong64_t /*firstEntry*/) {}

protected:
   /// type-erased vector of pointers to pointers to column values - one per slot
   virtual std::vector<void *>
   GetColumnReadersImpl(std::string_view name, unsigned int nSlots, const std::type_info &) = 0;
};

} // ns TDF
} // ns Experimental
} // ns ROOT

#endif // ROOT_TDATASOURCE
