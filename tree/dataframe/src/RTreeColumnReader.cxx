#include "ROOT/RDF/RTreeColumnReader.hxx"
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

#include <bitset>
#include <list>

void *ROOT::Internal::RDF::RTreeOpaqueColumnReader::GetImpl(std::size_t entryInBulk)
{
   return fCachedValues[entryInBulk];
}

void ROOT::Internal::RDF::RTreeOpaqueColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   if (fLastEntry == mask.GetFirstEntry())
      return;

   const auto validIndices = mask.GetValidIndices();
   fLastEntry = mask.GetFirstEntry();
   if (validIndices.empty())
      return;

   std::fill(fCachedValues.begin(), fCachedValues.end(), nullptr);

   for (auto idx : validIndices)
      fCachedValues[idx] = fTreeValue->GetAddress();
}

ROOT::Internal::RDF::RTreeOpaqueColumnReader::RTreeOpaqueColumnReader(TTreeReader &r, std::string_view colName,
                                                                      std::size_t maxBulkSize)
   : fTreeValue(std::make_unique<ROOT::Internal::TTreeReaderOpaqueValue>(r, colName.data()))
{
   fCachedValues = ROOT::RVec<void *>(maxBulkSize);
}

ROOT::Internal::RDF::RTreeOpaqueColumnReader::~RTreeOpaqueColumnReader() = default;

void *ROOT::Internal::RDF::RTreeUntypedValueColumnReader::GetImpl(std::size_t entryInBulk)
{

   if (fAddrOfTypeThatCannotBeReadInBulk)
      return fAddrOfTypeThatCannotBeReadInBulk;

   if (!fIsCachedValueValidAtIdx[entryInBulk]) {
      // This entry was marked as invalid during loading, return nullptr to signal this to the caller
      return nullptr;
   }

   return fCachedValues.data() + entryInBulk * fValueSize;
}

void ROOT::Internal::RDF::RTreeUntypedValueColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   if (fLastEntry == mask.GetFirstEntry())
      return;

   // Early return for the case of TBranchObject with split level zero, which cannot be read in a bulk
   if (fTreeValue->IsBranchObjectUnsplit()) {
      fAddrOfTypeThatCannotBeReadInBulk = fTreeValue->Get();
      return;
   }

   // This can only happen at first loading time, when the TTreeReader has begun traversing entries and the branch
   // proxy is set.
   if (fValueSize == 0)
      fValueSize = fTreeValue->GetValueSize();
   assert(fValueSize > 0 && "Could not retrieve size of value type in RDataFrame TTree value column reader.");
   // First time setup for cache of bulk values
   if (fCachedValues.empty()) {
      fCachedValues.resize(fCachedBranchAddresses.size() * fValueSize);
   }

   // If we tried to load an entry and it was missing, there might not be enough information to get a deleter, so retry
   // as long as we don't have one
   if (!fObjDeleter)
      fObjDeleter = fTreeValue->GetDeleter();

   const auto validIndices = mask.GetValidIndices();
   if (validIndices.empty())
      return;

   std::fill(fIsCachedValueValidAtIdx.begin(), fIsCachedValueValidAtIdx.end(), true);
   std::fill(fCachedBranchAddressesRAII.begin(), fCachedBranchAddressesRAII.end(), nullptr);
   std::fill(fCachedBranchAddresses.begin(), fCachedBranchAddresses.end(), nullptr);
   std::fill(fCachedBranchAddressesOfAddresses.begin(), fCachedBranchAddressesOfAddresses.end(), nullptr);
   std::fill(fCachedValues.begin(), fCachedValues.end(), std::byte{});

   for (auto idx : validIndices) {
      fCachedBranchAddresses[idx] = fCachedValues.data() + idx * fValueSize;
      // Crucial for reading non-PODs. Also crucial that this specific address
      // survives until the end of the Get method which reads the actual value
      // from the branch into the memory location above
      fCachedBranchAddressesOfAddresses[idx] = &fCachedBranchAddresses[idx];
      fCachedBranchAddressesRAII[idx] = std::unique_ptr<void, std::function<void(void *)>>(
         fCachedBranchAddresses[idx], fObjDeleter ? fObjDeleter : [](void *) {});

      fTreeValue->SetAddress(fCachedBranchAddresses[idx], fCachedBranchAddressesOfAddresses[idx]);
      fIsCachedValueValidAtIdx[idx] = bool(fTreeValue->Get());
   }

   fLastEntry = mask.GetFirstEntry();
}

ROOT::Internal::RDF::RTreeUntypedValueColumnReader::RTreeUntypedValueColumnReader(TTreeReader &r,
                                                                                  std::string_view colName,
                                                                                  std::string_view typeName,
                                                                                  std::size_t maxBulkSize)
   : fTreeValue(std::make_unique<ROOT::Internal::TTreeReaderUntypedValue>(r, colName, typeName)),
     fCachedValues(),
     fCachedBranchAddresses(maxBulkSize),
     fCachedBranchAddressesOfAddresses(maxBulkSize),
     fObjDeleter(nullptr),
     fCachedBranchAddressesRAII(maxBulkSize),
     fIsCachedValueValidAtIdx(maxBulkSize),
     fValueSize(),
     fLastEntry(std::numeric_limits<std::uint64_t>::max()),
     fAddrOfTypeThatCannotBeReadInBulk(nullptr),
     fColName(colName)
{
}

ROOT::Internal::RDF::RTreeUntypedValueColumnReader::~RTreeUntypedValueColumnReader() = default;

void ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::LoadStdArray(std::size_t entryInBulk)
{
   auto &readerArray = *fTreeArray;
   // GetSize is called here to trigger actual reading of the branch proxy, which also sets the appropriate read status
   const auto readerArraySize = readerArray.GetSize();

   // The reader could not read an array, signal this back to the node requesting the value
   if (R__unlikely(readerArray.GetReadStatus() == ROOT::Internal::TTreeReaderValueBase::EReadStatus::kReadError)) {
      fCachedValuesInvalidIndices.push_back(entryInBulk);
      return;
   }

   // std::array storage should always be contiguous
   assert(readerArray.IsContiguous() && "std::array storage should always be contiguous");

   auto &cachedValues = std::get<0>(fCachedValues);

   if (readerArraySize > 0) {
      // trigger loading of the contents of the TTreeReaderArray
      // the address of the first element in the reader array is not necessarily equal to
      // the address returned by the GetAddress method
      RVec<std::byte> rvec(readerArray.At(0), readerArraySize);
      swap(cachedValues[entryInBulk], rvec);
   }
}

void ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::LoadStdVector(std::size_t entryInBulk)
{
   auto &readerArray = *fTreeArray;
   // GetSize is called here to trigger actual reading of the branch proxy, which also sets the appropriate read status
   const auto readerArraySize = readerArray.GetSize();

   // The reader could not read an array, signal this back to the node requesting the value
   if (R__unlikely(readerArray.GetReadStatus() == ROOT::Internal::TTreeReaderValueBase::EReadStatus::kReadError)) {
      fCachedValuesInvalidIndices.push_back(entryInBulk);
      return;
   }

   auto &cachedValues = std::get<1>(fCachedValues);

   // There is no zero-copy constructor for std::vector, so we need to copy the data in any case.
   if (readerArraySize > 0) {
      // Caching the value type size since GetValueSize might be expensive.
      if (fValueSize == 0)
         fValueSize = readerArray.GetValueSize();
      assert(fValueSize > 0 && "Could not retrieve size of collection value type.");
      // Array is not contiguous, make a full copy of it.
      auto &dst = cachedValues[entryInBulk];
      dst.reserve(readerArraySize * fValueSize);
      for (std::size_t i{0}; i < readerArraySize; i++) {
         auto val = readerArray.At(i);
         std::copy(val, val + fValueSize, std::back_inserter(dst));
      }
   }
}

void ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::LoadRVec(std::size_t entryInBulk)
{

   auto &readerArray = *fTreeArray;
   // GetSize is called here to trigger actual reading of the branch proxy, which also sets the appropriate read status
   const auto readerArraySize = readerArray.GetSize();

   // The reader could not read an array, signal this back to the node requesting the value
   if (R__unlikely(readerArray.GetReadStatus() == ROOT::Internal::TTreeReaderValueBase::EReadStatus::kReadError)) {
      fCachedValuesInvalidIndices.push_back(entryInBulk);
      return;
   }

   auto &cachedValues = std::get<0>(fCachedValues);

   if (readerArray.IsContiguous() && !(fCollectionType == ECollectionType::kRVecBool)) {
      if (readerArraySize > 0) {
         // trigger loading of the contents of the TTreeReaderArray
         // the address of the first element in the reader array is not necessarily equal to
         // the address returned by the GetAddress method
         RVec<std::byte> rvec(readerArray.At(0), readerArraySize);
         swap(cachedValues[entryInBulk], rvec);
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
         auto &dst = cachedValues[entryInBulk];
         dst.reserve(readerArraySize * fValueSize);
         for (std::size_t i{0}; i < readerArraySize; i++) {
            auto val = readerArray.At(i);
            std::copy(val, val + fValueSize, std::back_inserter(dst));
         }
         // RVec's `size()` method returns the value of the `fSize` data member, unlike std::vector's `size()`
         // which returns the distance between begin and end divided by the size of the collection value type.
         // This difference is important in this case: we reserved enough space in the RVec to fill
         // `readerArraySize * fValueSize` bytes, but the reader will need to read just `readerArraySize` elements
         // adjusted to the correct `fValueSize` bytes per element. Thus, we set the size of the RVec here to
         // represent the correct size of the user-requested RVec<T>. This leaves the RVec<std::byte> in an invalid
         // state until it is cast to the correct type (by the `TryGet<T>` call).
         dst.set_size(readerArraySize);
      }
   }
}

void ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{

   if (fLastEntry == mask.GetFirstEntry())
      return;

   const auto validIndices = mask.GetValidIndices();
   fLastEntry = mask.GetFirstEntry();
   if (validIndices.empty())
      return;

   fCachedValuesInvalidIndices.clear();
   fCachedValuesInvalidIndices.reserve(validIndices.size());

   std::visit(
      [](auto &&visited) {
         for (auto &coll : visited)
            coll.clear();
      },
      fCachedValues);

   for (auto idx : validIndices) {
      switch (fCollectionType) {
      case ECollectionType::kStdArray: LoadStdArray(idx); break;
      case ECollectionType::kRVec:
      case ECollectionType::kRVecBool: LoadRVec(idx); break;
      case ECollectionType::kStdVector: LoadStdVector(idx); break;
      default: throw std::runtime_error("RDataFrame: unrecognized value type to read a collection in a bulk.");
      }
   }
}

void *ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::GetImpl(std::size_t entryInBulk)
{
   if (fCachedValuesInvalidIndices.end() !=
       std::find(fCachedValuesInvalidIndices.begin(), fCachedValuesInvalidIndices.end(), entryInBulk)) {
      return nullptr;
   }

   switch (fCollectionType) {
   case ECollectionType::kStdArray: {
      return std::get<0>(fCachedValues)[entryInBulk].data();
   }
   case ECollectionType::kRVec:
   case ECollectionType::kRVecBool: return &std::get<0>(fCachedValues)[entryInBulk];
   case ECollectionType::kStdVector: return &std::get<1>(fCachedValues)[entryInBulk];
   default: throw std::runtime_error("RDataFrame: unrecognized value type to read a collection in a bulk.");
   }
}

ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::RTreeUntypedArrayColumnReader(TTreeReader &r,
                                                                                  std::string_view colName,
                                                                                  std::string_view valueTypeName,
                                                                                  ECollectionType collType,
                                                                                  std::size_t maxBulkSize)
   : fTreeArray(std::make_unique<ROOT::Internal::TTreeReaderUntypedArray>(r, colName, valueTypeName)),
     fCollectionType(collType)
{
   if (fCollectionType == ECollectionType::kStdVector)
      fCachedValues = std::vector<std::vector<std::byte>>(maxBulkSize);
   else
      fCachedValues = std::vector<ROOT::RVec<std::byte>>(maxBulkSize);
}

ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::~RTreeUntypedArrayColumnReader() = default;

ROOT::Internal::RDF::RMaskedColumnReader::RMaskedColumnReader(
   TTreeReader &r, std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase> valueReader, std::string_view maskName,
   unsigned int maskIndex, std::size_t maxBulkSize)
   : fValueReader{std::move(valueReader)},
     fTreeValueMask{std::make_unique<TTreeReaderValue<uint64_t>>(r, maskName.data())},
     fMaskIndex{maskIndex}
{
   fCachedValues = ROOT::RVec<void *>(maxBulkSize);
}

ROOT::Internal::RDF::RMaskedColumnReader::~RMaskedColumnReader() = default;

void *ROOT::Internal::RDF::RMaskedColumnReader::GetImpl(std::size_t entryInBulk)
{
   return fCachedValues[entryInBulk];
}

void ROOT::Internal::RDF::RMaskedColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   if (fLastEntry == mask.GetFirstEntry())
      return;

   const auto validIndices = mask.GetValidIndices();
   fLastEntry = mask.GetFirstEntry();
   if (validIndices.empty())
      return;

   std::fill(fCachedValues.begin(), fCachedValues.end(), nullptr);
   fValueReader->Load(mask);
   auto size = fCachedValues.size();
   for (decltype(size) i = 0; i < size; i++) {
      const std::bitset<64> treeMask{*fTreeValueMask->Get()};
      if (treeMask.test(fMaskIndex) == false)
         fCachedValues[i] = nullptr;
      else {
         fCachedValues[i] = fValueReader->TryGet<void>(i);
      }
   }
}
