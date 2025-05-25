#include "ROOT/RDF/RTreeColumnReader.hxx"
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

void *ROOT::Internal::RDF::RTreeOpaqueColumnReader::GetImpl(Long64_t)
{
   return fTreeValue->GetAddress();
}

ROOT::Internal::RDF::RTreeOpaqueColumnReader::RTreeOpaqueColumnReader(TTreeReader &r, std::string_view colName)
   : fTreeValue(std::make_unique<ROOT::Internal::TTreeReaderOpaqueValue>(r, colName.data()))
{
}

ROOT::Internal::RDF::RTreeOpaqueColumnReader::~RTreeOpaqueColumnReader() = default;

void *ROOT::Internal::RDF::RTreeUntypedValueColumnReader::GetImpl(Long64_t)
{
   return fTreeValue->Get();
}

ROOT::Internal::RDF::RTreeUntypedValueColumnReader::RTreeUntypedValueColumnReader(TTreeReader &r,
                                                                                  std::string_view colName,
                                                                                  std::string_view typeName)
   : fTreeValue(std::make_unique<ROOT::Internal::TTreeReaderUntypedValue>(r, colName, typeName))
{
}

ROOT::Internal::RDF::RTreeUntypedValueColumnReader::~RTreeUntypedValueColumnReader() = default;

void *ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::GetImpl(Long64_t entry)
{
   if (entry == fLastEntry)
      return &fRVec; // we already pointed our fRVec to the right address

   auto &readerArray = *fTreeArray;
   const auto readerArraySize = readerArray.GetSize();

   // The reader could not read an array, signal this back to the node requesting the value
   if (R__unlikely(readerArray.GetReadStatus() == ROOT::Internal::TTreeReaderValueBase::EReadStatus::kReadError))
      return nullptr;

   if (readerArray.IsContiguous() && !(fCollectionType == ECollectionType::kRVecBool)) {
      if (readerArraySize > 0) {
         // trigger loading of the contents of the TTreeReaderArray
         // the address of the first element in the reader array is not necessarily equal to
         // the address returned by the GetAddress method
         RVec<Byte_t> rvec(readerArray.At(0), readerArraySize);
         swap(fRVec, rvec);
      } else {
         fRVec.clear();
      }
   } else {
      // The storage is not contiguous: we cannot but copy into the RVec
#ifndef NDEBUG
      if (!fCopyWarningPrinted && !(fCollectionType == ECollectionType::kRVecBool)) {
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
         fRVec.clear();
         fRVec.reserve(readerArraySize * fValueSize);
         for (std::size_t i{0}; i < readerArraySize; i++) {
            auto val = readerArray.At(i);
            std::copy(val, val + fValueSize, std::back_inserter(fRVec));
         }
         // RVec's `size()` method returns the value of the `fSize` data member, unlike std::vector's `size()`
         // which returns the distance between begin and end divided by the size of the collection value type.
         // This difference is important in this case: we reserved enough space in the RVec to fill
         // `readerArraySize * fValueSize` bytes, but the reader will need to read just `readerArraySize` elements
         // adjusted to the correct `fValueSize` bytes per element. Thus, we set the size of the RVec here to
         // represent the correct size of the user-requested RVec<T>. This leaves the RVec<Byte_t> in an invalid
         // state until it is cast to the correct type (by the `TryGet<T>` call).
         fRVec.set_size(readerArraySize);
      } else {
         fRVec.clear();
      }
   }
   fLastEntry = entry;
   if (fCollectionType == ECollectionType::kStdArray)
      return fRVec.data();
   else
      return &fRVec;
}

ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::RTreeUntypedArrayColumnReader(TTreeReader &r,
                                                                                  std::string_view colName,
                                                                                  std::string_view valueTypeName,
                                                                                  ECollectionType collType)
   : fTreeArray(std::make_unique<ROOT::Internal::TTreeReaderUntypedArray>(r, colName, valueTypeName)),
     fCollectionType(collType)
{
}

ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::~RTreeUntypedArrayColumnReader() = default;
