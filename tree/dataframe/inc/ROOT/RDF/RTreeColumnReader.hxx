// Author: Enrico Guiraud CERN 09/2020

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RTREECOLUMNREADER
#define ROOT_RDF_RTREECOLUMNREADER

#include "RColumnReaderBase.hxx"
#include <ROOT/RVec.hxx>
#include <Rtypes.h>  // Long64_t, R__CLING_PTRCHECK
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

#include <memory>
#include <string>

namespace ROOT {
namespace Internal {
namespace RDF {

/// RTreeColumnReader specialization for TTree values read via TTreeReaderValues
template <typename T>
class R__CLING_PTRCHECK(off) RTreeColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   TTreeReader *fTreeReader = nullptr; ///< Non-owning pointer to the TTreeReader. Never null.
   std::unique_ptr<TTreeReaderValue<T>> fTreeValue;
   ROOT::RVec<T> fCachedValues; // RVec rather than std::vector to avoid vector<bool> shenanigans.
                                // We would not need this cache if the I/O layer already returned a contiguous buffer
                                // (or an iterator over the bulk).
   RMaskedEntryRange fMask;

   void LoadImpl(const Internal::RDF::RMaskedEntryRange &requestedMask, std::size_t bulkSize) final
   {
      if (requestedMask.FirstEntry() != fMask.FirstEntry()) { // new bulk
         fMask.SetAll(false);
         fMask.SetFirstEntry(requestedMask.FirstEntry());
      }

      for (std::size_t i = 0ul; i < bulkSize; ++i) {
         if (requestedMask[i] && !fMask[i]) { // we don't have a value for this entry yet
            fTreeReader->SetEntry(requestedMask.FirstEntry() + i);
            // TODO avoid copy with bulk I/O when possible?
            // need a copy-assign here (rather than a move-assign) because multiple TTreeReader{Value,Array} might
            // be using the same underlying object and if one moves the contents out the next will find it moved from.
            fCachedValues[i] = *fTreeValue->Get();
            fMask[i] = true;
         }
      }
   }

   void *GetImpl(std::size_t offset) final { return &fCachedValues[offset]; }

public:
   /// Construct the RTreeColumnReader. Actual initialization is performed lazily by the Init method.
   RTreeColumnReader(TTreeReader &r, const std::string &colName, std::size_t maxEventsPerBulk)
      : fTreeReader(&r), fTreeValue(std::make_unique<TTreeReaderValue<T>>(r, colName.c_str())),
        fCachedValues(maxEventsPerBulk), fMask(maxEventsPerBulk)
   {
   }

   /// The dtor resets the TTreeReaderValue object.
   //
   // Otherwise a race condition is present in which a TTreeReader
   // and its TTreeReader{Value,Array}s can be deleted concurrently:
   // - Thread #1) a task ends and pushes back processing slot
   // - Thread #2) a task starts and overwrites thread-local TTreeReaderValues
   // - Thread #1) first task deletes TTreeReader
   // See https://github.com/root-project/root/commit/26e8ace6e47de6794ac9ec770c3bbff9b7f2e945
   ~RTreeColumnReader() override { fTreeValue.reset(); }
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderArrays.
///
/// TTreeReaderArrays are used whenever the RDF column type is RVec<T>.
template <typename T>
class R__CLING_PTRCHECK(off) RTreeColumnReader<RVec<T>> final : public ROOT::Detail::RDF::RColumnReaderBase {
   TTreeReader *fTreeReader; ///< Non-owning pointer to the TTreeReader. Never null.
   std::unique_ptr<TTreeReaderArray<T>> fTreeArray;

   /// Enumerator for the memory layout of the branch
   enum class EStorageType : char { kContiguous, kUnknown, kSparse };

   /// We return references to the inner RVecs to clients, to guarantee a stable address and contiguous memory layout.
   /// fCachedValues (the outer vector) has size equal to maxEventsPerBulk.
   std::vector<RVec<T>> fCachedValues;

   /// Signal whether we ever checked that the branch we are reading with a TTreeReaderArray stores array elements
   /// in contiguous memory.
   EStorageType fStorageType = EStorageType::kUnknown;
   RMaskedEntryRange fMask;

   /// Whether we already printed a warning about performing a copy of the TTreeReaderArray contents
   bool fCopyWarningPrinted = false;

   void LoadImpl(const Internal::RDF::RMaskedEntryRange &requestedMask, std::size_t bulkSize) final
   {
      if (requestedMask.FirstEntry() != fMask.FirstEntry()) { // new bulk
         fMask.SetAll(false);
         fMask.SetFirstEntry(requestedMask.FirstEntry());
      }

      for (std::size_t i = 0ul; i < bulkSize; ++i) {
         if (requestedMask[i] && !fMask[i]) { // we don't have a value for this entry yet
            fTreeReader->SetEntry(requestedMask.FirstEntry() + i);

            // TODO I hate these copies. Could avoid them with bulk I/O when possible.
            RVec<T> rvec(fTreeArray->begin(), fTreeArray->end());
            swap(fCachedValues[i], rvec);
            fMask[i] = true;
         }
      }
   }

   void *GetImpl(std::size_t offset) final { return &fCachedValues[offset]; }

public:
   RTreeColumnReader(TTreeReader &r, const std::string &colName, std::size_t maxEventsPerBulk)
      : fTreeReader(&r), fTreeArray(std::make_unique<TTreeReaderArray<T>>(r, colName.c_str())),
        fCachedValues(maxEventsPerBulk), fMask(maxEventsPerBulk)
   {
   }

   /// See the other class template specializations for an explanation.
   ~RTreeColumnReader() override { fTreeArray.reset(); }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
