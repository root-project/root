// Author: Enrico Guiraud, Danilo Piparo CERN  09/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDATASOURCE
#define ROOT_RDATASOURCE

#include "RDF/RColumnReaderBase.hxx"
#include "ROOT/RStringView.hxx"
#include "ROOT/RConfig.hxx" // R__DEPRECATED
#include "RtypesCore.h" // ULong64_t
#include "TError.h" // Warning
#include "TString.h"

#include <algorithm> // std::transform
#include <string>
#include <typeinfo>
#include <vector>

namespace ROOT {
namespace RDF {
class RDataSource;
}
}

/// Print a RDataSource at the prompt
namespace cling {
std::string printValue(ROOT::RDF::RDataSource *ds);
} // namespace cling

namespace ROOT {

namespace Internal {
namespace TDS {

/// Mother class of TTypedPointerHolder. The instances
/// of this class can be put in a container. Upon destruction,
/// the correct deletion of the pointer is performed in the
/// derived class.
class TPointerHolder {
protected:
   void *fPointer{nullptr};

public:
   TPointerHolder(void *ptr) : fPointer(ptr) {}
   void *GetPointer() { return fPointer; }
   void *GetPointerAddr() { return &fPointer; }
   virtual TPointerHolder *GetDeepCopy() = 0;
   virtual ~TPointerHolder(){};
};

/// Class to wrap a pointer and delete the memory associated to it
/// correctly
template <typename T>
class TTypedPointerHolder final : public TPointerHolder {
public:
   TTypedPointerHolder(T *ptr) : TPointerHolder((void *)ptr) {}

   TPointerHolder *GetDeepCopy() final
   {
      const auto typedPtr = static_cast<T *>(fPointer);
      return new TTypedPointerHolder(new T(*typedPtr));
   }

   ~TTypedPointerHolder() { delete static_cast<T *>(fPointer); }
};

} // ns TDS
} // ns Internal

namespace RDF {

// clang-format off
/**
\class ROOT::RDF::RDataSource
\ingroup dataframe
\brief RDataSource defines an API that RDataFrame can use to read arbitrary data formats.

A concrete RDataSource implementation (i.e. a class that inherits from RDataSource and implements all of its pure
methods) provides an adaptor that RDataFrame can leverage to read any kind of tabular data formats.
RDataFrame calls into RDataSource to retrieve information about the data, retrieve (thread-local) readers or "cursors"
for selected columns and to advance the readers to the desired data entry.

The sequence of calls that RDataFrame (or any other client of a RDataSource) performs is the following:

 - SetNSlots() : inform RDataSource of the desired level of parallelism
 - GetColumnReaders() : retrieve from RDataSource per-thread readers for the desired columns
 - Initialize() : inform RDataSource that an event-loop is about to start
 - GetEntryRanges() : retrieve from RDataSource a set of ranges of entries that can be processed concurrently
 - InitSlot() : inform RDataSource that a certain thread is about to start working on a certain range of entries
 - SetEntry() : inform RDataSource that a certain thread is about to start working on a certain entry
 - FinalizeSlot() : inform RDataSource that a certain thread finished working on a certain range of entries
 - Finalize() : inform RDataSource that an event-loop finished

RDataSource implementations must support running multiple event-loops consecutively (although sequentially) on the same dataset.
 - \b SetNSlots() is called once per RDataSource object, typically when it is associated to a RDataFrame.
 - \b GetColumnReaders() can be called several times, potentially with the same arguments, also in-between event-loops, but not during an event-loop.
 - \b GetEntryRanges() will be called several times, including during an event loop, as additional ranges are needed.  It will not be called concurrently.
 - \b Initialize() and \b Finalize() are called once per event-loop,  right before starting and right after finishing.
 - \b InitSlot(), \b SetEntry(), and \b FinalizeSlot() can be called concurrently from multiple threads, multiple times per event-loop.

 Advanced users that plan to implement a custom RDataSource can check out existing implementations, e.g. RCsvDS or RNTupleDS.
 See the inheritance diagram below for the full list of existing concrete implementations.
*/
class RDataSource {
   // clang-format on
private:
   /// \cond
   // Temporary boolean value used by the backwards compatibility code for the deprecated spellings Initialise,
   // Finalise and FinaliseSlot.
   bool fDeprecatedBaseCalled = false;
   /// \endcond

protected:
   using Record_t = std::vector<void *>;
   friend std::string cling::printValue(::ROOT::RDF::RDataSource *);

   virtual std::string AsString() { return "generic data source"; };

public:
   virtual ~RDataSource() = default;

   // clang-format off
   /// \brief Inform RDataSource of the number of processing slots (i.e. worker threads) used by the associated RDataFrame.
   /// Slots numbers are used to simplify parallel execution: RDataFrame guarantees that different threads will always
   /// pass different slot values when calling methods concurrently.
   // clang-format on
   virtual void SetNSlots(unsigned int nSlots) = 0;

   // clang-format off
   /// \brief Returns a reference to the collection of the dataset's column names
   // clang-format on
   virtual const std::vector<std::string> &GetColumnNames() const = 0;

   /// \brief Checks if the dataset has a certain column
   /// \param[in] colName The name of the column
   virtual bool HasColumn(std::string_view colName) const = 0;

   // clang-format off
   /// \brief Type of a column as a string, e.g. `GetTypeName("x") == "double"`. Required for jitting e.g. `df.Filter("x>0")`.
   /// \param[in] colName The name of the column
   // clang-format on
   virtual std::string GetTypeName(std::string_view colName) const = 0;

   // clang-format off
   /// Called at most once per column by RDF. Return vector of pointers to pointers to column values - one per slot.
   /// \tparam T The type of the data stored in the column
   /// \param[in] columnName The name of the column
   ///
   /// These pointers are veritable cursors: it's a responsibility of the RDataSource implementation that they point to
   /// the "right" memory region.
   // clang-format on
   template <typename T>
   std::vector<T **> GetColumnReaders(std::string_view columnName)
   {
      auto typeErasedVec = GetColumnReadersImpl(columnName, typeid(T));
      std::vector<T **> typedVec(typeErasedVec.size());
      std::transform(typeErasedVec.begin(), typeErasedVec.end(), typedVec.begin(),
                     [](void *p) { return static_cast<T **>(p); });
      return typedVec;
   }

   /// If the other GetColumnReaders overload returns an empty vector, this overload will be called instead.
   /// \param[in] slot The data processing slot that needs to be considered
   /// \param[in] name The name of the column for which a column reader needs to be returned
   /// \param[in] tid A type_info
   /// At least one of the two must return a non-empty/non-null value.
   virtual std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
   GetColumnReaders(unsigned int /*slot*/, std::string_view /*name*/, const std::type_info &)
   {
      return {};
   }

   // clang-format off
   /// \brief Return ranges of entries to distribute to tasks.
   /// They are required to be contiguous intervals with no entries skipped. Supposing a dataset with nEntries, the
   /// intervals must start at 0 and end at nEntries, e.g. [0-5],[5-10] for 10 entries.
   /// This function will be invoked repeatedly by RDataFrame as it needs additional entries to process.
   /// The same entry range should not be returned more than once.
   /// Returning an empty collection of ranges signals to RDataFrame that the processing can stop.
   // clang-format on
   virtual std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() = 0;

   // clang-format off
   /// \brief Advance the "cursors" returned by GetColumnReaders to the selected entry for a particular slot.
   /// \param[in] slot The data processing slot that needs to be considered
   /// \param[in] entry The entry which needs to be pointed to by the reader pointers
   /// Slots are adopted to accommodate parallel data processing. 
   /// Different workers will loop over different ranges and
   /// will be labelled by different "slot" values.
   /// Returns *true* if the entry has to be processed, *false* otherwise.
   // clang-format on
   virtual bool SetEntry(unsigned int slot, ULong64_t entry) = 0;

   // clang-format off
   /// \brief Convenience method called before starting an event-loop.
   /// This method might be called multiple times over the lifetime of a RDataSource, since
   /// users can run multiple event-loops with the same RDataFrame.
   /// Ideally, `Initialize` should set the state of the RDataSource so that multiple identical event-loops
   /// will produce identical results.
   // clang-format on
   virtual void Initialize() {}

   /// \cond
   // Unused deprecated struct, it's here to remind us to remove the deprecated spellings Initialise, Finalise and
   // FinaliseSlot. PR that removes the deprecated code: https://github.com/root-project/root/pull/9521 .
   struct R__DEPRECATED(6, 30,
                        "Use Initialize, Finalize and FinalizeSlot instead of the corresponding british spellings.")
      NeverUsedJustAReminder {
   };

   virtual void Initialise() { fDeprecatedBaseCalled = true; }

   void CallInitialize()
   {
      fDeprecatedBaseCalled = false;
      Initialise();
      if (!fDeprecatedBaseCalled) {
         Warning("RDataSource::Initialise", "Initialise is deprecated. Please rename it to \"Initialize\" (with a z).");
         return;
      }

      // `Initialise()` was not overridden, the data source uses the new spelling: good!
      Initialize();
   }
   /// \endcond

   // clang-format off
   /// \brief Convenience method called at the start of the data processing associated to a slot.
   /// \param[in] slot The data processing slot wihch needs to be initialized
   /// \param[in] firstEntry The first entry of the range that the task will process.
   /// This method might be called multiple times per thread per event-loop.
   // clang-format on
   virtual void InitSlot(unsigned int /*slot*/, ULong64_t /*firstEntry*/) {}

   // clang-format off
   /// \brief Convenience method called at the end of the data processing associated to a slot.
   /// \param[in] slot The data processing slot wihch needs to be finalized
   /// This method might be called multiple times per thread per event-loop.
   // clang-format on
   virtual void FinalizeSlot(unsigned int /*slot*/) {}

   /// \cond
   virtual void FinaliseSlot(unsigned int) { fDeprecatedBaseCalled = true; }

   void CallFinalizeSlot(unsigned int slot)
   {
      fDeprecatedBaseCalled = false;
      FinaliseSlot(slot);
      if (!fDeprecatedBaseCalled) {
         Warning("RDataSource::FinaliseSlot",
                 "FinaliseSlot is deprecated. Please implement FinalizeSlot (with a z) instead of FinaliseSlot.");
         return;
      }

      FinalizeSlot(slot);
   }
   /// \endcond

   // clang-format off
   /// \brief Convenience method called after concluding an event-loop.
   /// See Initialize for more details.
   // clang-format on
   virtual void Finalize() {}

   /// \cond
   virtual void Finalise() { fDeprecatedBaseCalled = true; }

   void CallFinalize()
   {
      fDeprecatedBaseCalled = false;
      Finalise();
      if (!fDeprecatedBaseCalled) {
         Warning("RDataSource::FinaliseSlot",
                 "Finalise is deprecated. Please implement Finalize (with a z) instead of Finalise.");
         return;
      }

      Finalize();
   }
   /// \endcond

   /// \brief Return a string representation of the datasource type.
   /// The returned string will be used by ROOT::RDF::SaveGraph() to represent
   /// the datasource in the visualization of the computation graph.
   /// Concrete datasources can override the default implementation.
   virtual std::string GetLabel() { return "Custom Datasource"; }

protected:
   /// type-erased vector of pointers to pointers to column values - one per slot
   virtual Record_t GetColumnReadersImpl(std::string_view name, const std::type_info &) = 0;
};

} // ns RDF

} // ns ROOT

/// Print a RDataSource at the prompt
namespace cling {
inline std::string printValue(ROOT::RDF::RDataSource *ds)
{
   return ds->AsString();
}
} // namespace cling

#endif // ROOT_TDATASOURCE
