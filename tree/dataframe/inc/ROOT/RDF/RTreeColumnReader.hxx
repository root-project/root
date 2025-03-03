// Author: Enrico Guiraud CERN 09/2020
// Author: Vincenzo Eduardo Padulano CERN 09/2024

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RTREECOLUMNREADER
#define ROOT_RDF_RTREECOLUMNREADER

#include "RColumnReaderBase.hxx"
#include <ROOT/RVec.hxx>
#include "ROOT/RDF/Utils.hxx"
#include <Rtypes.h>  // Long64_t, R__CLING_PTRCHECK
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

#include <array>
#include <memory>
#include <string>
#include <cstddef>

namespace ROOT {
namespace Internal {
namespace RDF {

/// RTreeColumnReader specialization for TTree values read via TTreeReaderValues
template <typename T>
class R__CLING_PTRCHECK(off) RTreeColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   std::unique_ptr<TTreeReaderUntypedValue> fTreeValue;

   void *GetImpl(Long64_t) final { return fTreeValue->Get(); }
public:
   /// Construct the RTreeColumnReader. Actual initialization is performed lazily by the Init method.
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeValue(std::make_unique<TTreeReaderUntypedValue>(r, colName.c_str(),
                                                             ROOT::Internal::RDF::TypeID2TypeName(typeid(T))))
   {
   }
};

class R__CLING_PTRCHECK(off) RTreeOpaqueColumnReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   std::unique_ptr<ROOT::Internal::TTreeReaderOpaqueValue> fTreeValue;

   void *GetImpl(Long64_t) final { return fTreeValue->GetAddress(); }

public:
   /// Construct the RTreeColumnReader. Actual initialization is performed lazily by the Init method.
   RTreeOpaqueColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeValue(std::make_unique<ROOT::Internal::TTreeReaderOpaqueValue>(r, colName.c_str()))
   {
   }
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderArrays.
///
/// TTreeReaderArrays are used whenever the RDF column type is RVec<T>.
template <typename T>
class R__CLING_PTRCHECK(off) RTreeColumnReader<RVec<T>> final : public ROOT::Detail::RDF::RColumnReaderBase {
   std::unique_ptr<TTreeReaderUntypedArray> fTreeArray;

   using Byte_t = std::byte;

   /// We return a reference to this RVec to clients, to guarantee a stable address and contiguous memory layout.
   RVec<Byte_t> fRVec;

   Long64_t fLastEntry = -1;

   /// Whether we already printed a warning about performing a copy of the TTreeReaderArray contents
   bool fCopyWarningPrinted = false;

   /// The size of the collection value type.
   std::size_t fValueSize{};

   void *GetImpl(Long64_t entry) final
   {
      if (entry == fLastEntry)
         return &fRVec; // we already pointed our fRVec to the right address

      auto &readerArray = *fTreeArray;
      const auto readerArraySize = readerArray.GetSize();

      // The reader could not read an array, signal this back to the node requesting the value
      if (R__unlikely(readerArray.GetReadStatus() == ROOT::Internal::TTreeReaderValueBase::EReadStatus::kReadError))
         return nullptr;

      if (readerArray.IsContiguous()) {
         if (readerArraySize > 0) {
            // trigger loading of the contents of the TTreeReaderArray
            // the address of the first element in the reader array is not necessarily equal to
            // the address returned by the GetAddress method
            RVec<Byte_t> rvec(readerArray.At(0), readerArraySize);
            swap(fRVec, rvec);
         } else {
            RVec<Byte_t> emptyVec{};
            swap(fRVec, emptyVec);
         }
      } else {
         // The storage is not contiguous or we don't know yet: we cannot but copy into the rvec
#ifndef NDEBUG
         if (!fCopyWarningPrinted) {
            Warning("RTreeColumnReader::Get",
                    "Branch %s hangs from a non-split branch. A copy is being performed in order "
                    "to properly read the content.",
                    readerArray.GetBranchName());
            fCopyWarningPrinted = true;
         }
#else
         (void)fCopyWarningPrinted;
#endif
         if (readerArraySize > 0) {
            // Caching the value type size since GetValueSize might be expensive.
            if (fValueSize == 0)
               fValueSize = readerArray.GetValueSize();
            assert(fValueSize > 0 && "Could not retrieve size of collection value type.");
            // Array is not contiguous, make a full copy of it.
            fRVec = RVec<Byte_t>();
            fRVec.reserve(readerArraySize * fValueSize);
            for (std::size_t i{0}; i < readerArraySize; i++) {
               auto val = readerArray.At(i);
               std::copy(val, val + fValueSize, std::back_inserter(fRVec));
            }
            fRVec.resize(readerArraySize);
         } else {
            RVec<Byte_t> emptyVec{};
            swap(fRVec, emptyVec);
         }
      }
      fLastEntry = entry;
      return &fRVec;
   }

public:
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeArray(
           std::make_unique<TTreeReaderUntypedArray>(r, colName, ROOT::Internal::RDF::TypeID2TypeName(typeid(T))))
   {
   }
};

/// RTreeColumnReader specialization for arrays of boolean values read via TTreeReaderArrays.
///
/// TTreeReaderArray<bool> is used whenever the RDF column type is RVec<bool>.
template <>
class R__CLING_PTRCHECK(off) RTreeColumnReader<RVec<bool>> final : public ROOT::Detail::RDF::RColumnReaderBase {

   using Byte_t = std::byte;

   std::unique_ptr<TTreeReaderUntypedArray> fTreeArray;

   /// We return a reference to this RVec to clients, to guarantee a stable address and contiguous memory layout
   RVec<Byte_t> fRVec;

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
         // Always perform a copy
         fRVec = RVec<Byte_t>();
         fRVec.reserve(readerArraySize * sizeof(bool));
         for (std::size_t i{0}; i < readerArraySize; i++) {
            auto val = readerArray.At(i);
            std::copy(val, val + sizeof(bool), std::back_inserter(fRVec));
         }
         fRVec.resize(readerArraySize);
      } else {
         RVec<Byte_t> emptyVec{};
         swap(fRVec, emptyVec);
      }
      return &fRVec;
   }

public:
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeArray(std::make_unique<TTreeReaderUntypedArray>(r, colName.c_str(),
                                                             ROOT::Internal::RDF::TypeID2TypeName(typeid(bool))))
   {
   }
};

/// RTreeColumnReader specialization for TTree values read via TTreeReaderArrays.
///
/// This specialization is used when the requested type for reading is std::array
template <typename T, std::size_t N>
class R__CLING_PTRCHECK(off) RTreeColumnReader<std::array<T, N>> final : public ROOT::Detail::RDF::RColumnReaderBase {
   std::unique_ptr<TTreeReaderUntypedArray> fTreeArray;

   Long64_t fLastEntry = -1;

   void *GetImpl(Long64_t) final { return fTreeArray->At(0); }

public:
   RTreeColumnReader(TTreeReader &r, const std::string &colName)
      : fTreeArray(std::make_unique<TTreeReaderUntypedArray>(r, colName.c_str(),
                                                             ROOT::Internal::RDF::TypeID2TypeName(typeid(T))))
   {
   }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif
