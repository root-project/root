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

/**
\class ROOT::Experimental::TDF::TDataSource
\ingroup dataframe
\brief The TDataSource interface dictates how a TDataFrame interface to an arbitrary data format should look like.

A TDataSource allows to seamlessly provide an adaptor for any kind of data set
or data format to the TDataFrame. Another way to imagine it is a veritable
"cursor".
The data sources are not supposed to be used as they are from the user but
rather as a way to plug a dataset into a TDataFrame.
*/
class TDataSource {
public:
   virtual ~TDataSource() = default;

   /// \brief Returns a reference to the collection of the dataset's column names
   virtual const std::vector<std::string> &GetColumnNames() const = 0;

   /// \brief Checks if the dataset has a certain column
   /// \param[in] columnName The name of the column
   virtual bool HasColumn(std::string_view) const = 0;

   /// \brief Type of a column as a string, e.g. `GetTypeName("x") == "double"`. Required for jitting e.g. `df.Filter("x>0")`.
   /// \param[in] columnName The name of the column
   virtual std::string GetTypeName(std::string_view) const = 0;

   /// Called at most once per column by TDF. Return vector of pointers to pointers to column values - one per slot.
   /// \tparam T The type of the data stored in the column
   /// \param[in] columnName The name of the column
   ///
   /// These pointers are veritable cursors: it's a responsibility of the TDataSource implementation that they point to the
   /// "right" memory region.
   template <typename T>
   std::vector<T **> GetColumnReaders(std::string_view columnName)
   {
      auto typeErasedVec = GetColumnReadersImpl(columnName, typeid(T));
      std::vector<T **> typedVec(typeErasedVec.size());
      std::transform(typeErasedVec.begin(), typeErasedVec.end(), typedVec.begin(),
                     [](void *p) { return static_cast<T **>(p); });
      return typedVec;
   }

   /// \brief Return ranges of entries to distribute to tasks.
   /// They are required to be contiguous intervals with no entries skipped. Supposing a dataset with nEntries, the intervals
   /// must start at 0 and end at nEntries, e.g. [0-5],[5-10] for 10 entries.
   virtual const std::vector<std::pair<ULong64_t, ULong64_t>> &GetEntryRanges() const = 0;

   /// \brief Advance the "cursors" returned by GetColumnReaders to the selected entry for a particular slot.
   /// \param[in] slot The data processing slot that needs to be considered
   /// \param[in] entry The entry which needs to be pointed to by the reader pointers
   /// Slots are adopted to accommodate parallel data processing. Different workers will loop over different ranges and will
   /// be labelled by different "slot" values.
   virtual void SetEntry(unsigned int slot, ULong64_t entry) = 0;

   /// \brief Convenience method to set the number of slots
   /// For some implementation it's necessary to know the number of slots in advance for optimisation purposes.
   virtual void SetNSlots(unsigned int nSlots) = 0;

   /// \brief Convenience method called at the start of the data processing associated to a slot.
   /// \param[in] slot The data processing slot wihch needs to be initialised
   /// \param[in] firstEntry The first entry of the range that the task will process.
   virtual void InitSlot(unsigned int /*slot*/, ULong64_t /*firstEntry*/) {}

   /// \brief Convenience method called at the end of the data processing associated to a slot.
   /// \param[in] slot The data processing slot wihch needs to be finalised
   virtual void FinaliseSlot(unsigned int /*slot*/) {}

protected:
   /// type-erased vector of pointers to pointers to column values - one per slot
   virtual std::vector<void *>
   GetColumnReadersImpl(std::string_view name, const std::type_info &) = 0;
};

} // ns TDF
} // ns Experimental
} // ns ROOT

#endif // ROOT_TDATASOURCE
