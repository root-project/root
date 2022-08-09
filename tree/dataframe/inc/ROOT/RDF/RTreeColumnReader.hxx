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
   std::unique_ptr<TTreeReaderValue<T>> fTreeValue;

   void *GetImpl(Long64_t) final { return fTreeValue->Get(); }
public:
   /// Construct the RTreeColumnReader. Actual initialization is performed lazily by the Init method.
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeValue(std::make_unique<TTreeReaderValue<T>>(r, colName.c_str()))
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
   ~RTreeColumnReader() { fTreeValue.reset(); }
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderArrays.
///
/// TTreeReaderArrays are used whenever the RDF column type is RVec<T>.
template <typename T>
class R__CLING_PTRCHECK(off) RTreeColumnReader<RVec<T>> final : public ROOT::Detail::RDF::RColumnReaderBase {
   std::unique_ptr<TTreeReaderArray<T>> fTreeArray;

   /// Enumerator for the memory layout of the branch
   enum class EStorageType : char { kContiguous, kUnknown, kSparse };

   /// We return a reference to this RVec to clients, to guarantee a stable address and contiguous memory layout.
   RVec<T> fRVec;

   /// Signal whether we ever checked that the branch we are reading with a TTreeReaderArray stores array elements
   /// in contiguous memory.
   EStorageType fStorageType = EStorageType::kUnknown;
   Long64_t fLastEntry = -1;

   /// Whether we already printed a warning about performing a copy of the TTreeReaderArray contents
   bool fCopyWarningPrinted = false;

   void *GetImpl(Long64_t entry) final
   {
      if (entry == fLastEntry)
         return &fRVec; // we already pointed our fRVec to the right address

      auto &readerArray = *fTreeArray;
      const auto readerArraySize = readerArray.GetSize();
      if (R__likely(fStorageType == EStorageType::kContiguous)) {
         // trigger loading of the contents of the TTreeReaderArray
         // the address of the first element in the reader array is not necessarily equal to
         // the address returned by the GetAddress method
         auto *readerArrayAddr = &readerArray.At(0);
         ROOT::Internal::VecOps::ResetAddress(fRVec, readerArrayAddr, readerArraySize);
      } else if (fStorageType == EStorageType::kUnknown && readerArraySize > 1) {
         // TODO Move this check to constructor once ROOT-10823 is fixed and TTreeReaderArray itself exposes this info
         fStorageType = EStorageType::kContiguous;
         for (auto i = 0u; i < readerArraySize - 1; ++i) {
            if ((char *)&readerArray[i + 1] - (char *)&readerArray[i] != sizeof(T)) {
               fStorageType = EStorageType::kSparse;
               break;
            }
         }
         if (fStorageType == EStorageType::kContiguous) {
            // must put the RVec in memory adoption mode
            auto readerArrayAddr = &readerArray.At(0);
            RVec<T> rvec(readerArrayAddr, readerArraySize);
            swap(fRVec, rvec);
         }
      } else {
         // The storage is not contiguous or we don't know yet: we cannot but copy into the rvec
#ifndef NDEBUG
         if (fStorageType == EStorageType::kSparse) {
            if (!fCopyWarningPrinted) {
               Warning("RTreeColumnReader::Get",
                       "Branch %s hangs from a non-split branch. A copy is being performed in order "
                       "to properly read the content.",
                       readerArray.GetBranchName());
               fCopyWarningPrinted = true;
            }
         }
#else
         (void)fCopyWarningPrinted;
#endif
         RVec<T> rvec(readerArray.begin(), readerArray.end());
         swap(fRVec, rvec);
      }
      fLastEntry = entry;
      return &fRVec;
   }

public:
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeArray(std::make_unique<TTreeReaderArray<T>>(r, colName.c_str()))
   {
   }

   /// See the other class template specializations for an explanation.
   ~RTreeColumnReader() { fTreeArray.reset(); }
};

/// RTreeColumnReader specialization for arrays of boolean values read via TTreeReaderArrays.
///
/// TTreeReaderArray<bool> is used whenever the RDF column type is RVec<bool>.
template <>
class R__CLING_PTRCHECK(off) RTreeColumnReader<RVec<bool>> final : public ROOT::Detail::RDF::RColumnReaderBase {

   std::unique_ptr<TTreeReaderArray<bool>> fTreeArray;

   /// We return a reference to this RVec to clients, to guarantee a stable address and contiguous memory layout
   RVec<bool> fRVec;

   // We always copy the contents of TTreeReaderArray<bool> into an RVec<bool> (never take a view into the memory
   // buffer) because the underlying memory buffer might be the one of a std::vector<bool>, which is not a contiguous
   // slab of bool values.
   // Note that this also penalizes the case in which the column type is actually bool[], but the possible performance
   // gains in this edge case is probably not worth the extra complication required to differentiate the two cases.
   void *GetImpl(Long64_t) final
   {
      auto &readerArray = *fTreeArray;
      const auto readerArraySize = readerArray.GetSize();
      if (readerArraySize > 0) {
         // always perform a copy
         RVec<bool> rvec(readerArray.begin(), readerArray.end());
         swap(fRVec, rvec);
      } else {
         RVec<bool> emptyVec{};
         swap(fRVec, emptyVec);
      }
      return &fRVec;
   }

public:
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeArray(std::make_unique<TTreeReaderArray<bool>>(r, colName.c_str()))
   {
   }

   /// See the other class template specializations for an explanation.
   ~RTreeColumnReader() { fTreeArray.reset(); }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
