#include "ROOT/RDF/RTreeColumnReader.hxx"
#include <TDictionary.h>

#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TTreeReaderArray.h>

#include <TBufferFile.h>
#include <TMathBase.h>

#include <bitset>
#include <list>

#include <TBranchElement.h>
#include <TFriendElement.h>

namespace {

bool DatasetContainsIndexedFriends(TTree *tree)
{
   if (!tree)
      return false;
   auto *friends = tree->GetListOfFriends();
   if (!friends || friends->GetEntries() == 0)
      return false;
   for (auto fr : ROOT::Detail::TRangeStaticCast<TFriendElement>(friends)) {
      auto frTree = fr->GetTree();
      if (frTree && frTree->GetTreeIndex())
         return true;
   }
   return false;
}

bool DatasetHasEntryList(TTreeReader &r)
{
   return r.GetEntryList();
}

template <typename UserRequestedType, typename OnDiskBranchType>
void LoadBulk(ROOT::RVec<std::byte> &cachedValues, TBranch &branch, std::size_t bulkSize, std::uint64_t bulkBegin,
              std::uint64_t chainOffset, std::unique_ptr<TBuffer> &branchBulkBuffer,
              std::size_t &branchBulkNumberOfEntries, std::uint64_t &entriesReadFromBranchBulkBuffer,
              Long64_t &lastBasketBegin, std::uint64_t &lastChainOffset)
{
   // We instantiate a TBufferFile once, the calls to GetEntriesSerialized may re-allocate if a basket is bigger than
   // the size of this buffer, so we decide on an arbitrary initial size equal to the average basket size advertised by
   // the branch multiplied by the size of the type of the value we're reading.
   if (branchBulkBuffer == nullptr) {
      const auto bufSize{branch.GetBasketSize() * sizeof(OnDiskBranchType)};
      assert(bufSize < std::numeric_limits<Int_t>::max()); // TBuffer constructor expects Int_t for the buffer size
      branchBulkBuffer = std::make_unique<TBufferFile>(TBuffer::EMode::kWrite, static_cast<Int_t>(bufSize));
   }

   std::size_t nLoaded{};
   while (nLoaded < bulkSize) {
      // assert we either have not loaded a bulk yet or we exhausted the last branch bulk
      // assert(fBranchBulkSize == 0u || fBranchBulkBegin + fBranchBulkSize == fLoadedEntriesBegin + nLoaded);

      // GetEntriesSerialized does not allow reading a bulk starting from the middle of a basket, so we have to find
      // the first entry of the basket
      const auto localEntry = bulkBegin - chainOffset + nLoaded;

      // TMath::BinarySearch calls std::lower_bound
      // See same pattern in TBranch::GetBasketAndFirsts
      const Long64_t idxOfBasketContainingLocalEntry = TMath::BinarySearch(
         /*number of written baskets in the branch*/ branch.GetWriteBasket() + 1,
         /*array of first entry of each basket*/ branch.GetBasketEntry(), static_cast<Long64_t>(localEntry));
      const Long64_t basketBegin = branch.GetBasketEntry()[idxOfBasketContainingLocalEntry];
      if (lastBasketBegin != basketBegin || lastChainOffset != chainOffset) {
         // beginning of basket
         lastChainOffset = chainOffset;
         lastBasketBegin = basketBegin;
         entriesReadFromBranchBulkBuffer = 0;
         // GetEntriesSerialized does not byte-swap, so later we use frombuf to byte-swap and copy in a single pass
         // Returns number of entries loaded from the basket
         auto ret = branch.GetBulkRead().GetEntriesSerialized(lastBasketBegin, *branchBulkBuffer);
         assert(ret != -1); // We should always be able to read the entries from the basket
         branchBulkNumberOfEntries = ret;
      }

      const std::size_t nToSkip = localEntry - basketBegin;
      const auto nToLoad = std::min(branchBulkNumberOfEntries - nToSkip, bulkSize - nLoaded);

      // Need a 'char *' for the call to frombuf
      auto data = reinterpret_cast<char *>(branchBulkBuffer->GetCurrent());
      data += nToSkip * sizeof(OnDiskBranchType); // advance to the first element we are interested in

      for (std::size_t i = 0u; i < nToLoad; ++i) {
         // `frombuf` also advances the `data` pointer
         if constexpr (std::is_same_v<UserRequestedType, OnDiskBranchType>) {
            // Save one extra copy when the user-requested type is the same as the on-disk branch type
            frombuf(data, reinterpret_cast<UserRequestedType *>(cachedValues.data() +
                                                                (nLoaded + i) * sizeof(UserRequestedType)));
         } else {
            // First fill a temporary value of the correct on-disk branch type, then cast it to the user-requested type
            // and copy it into the final address
            OnDiskBranchType temp{};
            frombuf(data, &temp);
            *reinterpret_cast<UserRequestedType *>(cachedValues.data() + (nLoaded + i) * sizeof(UserRequestedType)) =
               static_cast<UserRequestedType>(temp);
         }
      }

      nLoaded += nToLoad;
      entriesReadFromBranchBulkBuffer += nToLoad;
   }
}

using TypeInfoRef = std::reference_wrapper<const std::type_info>;
struct TypeInfoRefHash {
   std::size_t operator()(TypeInfoRef id) const { return id.get().hash_code(); }
};

struct TypeInfoRefEqualComp {
   bool operator()(TypeInfoRef left, TypeInfoRef right) const { return left.get() == right.get(); }
};

template <typename UserRequestedType>
ROOT::Internal::RDF::RTreeScalarBulkColumnReader::BulkLoadFunc_t GetBulkLoadFunctionImpl(EDataType dt)
{
   switch (dt) {
   case kChar_t: return &LoadBulk<UserRequestedType, Char_t>;
   case kShort_t: return &LoadBulk<UserRequestedType, Short_t>;
   case kInt_t: return &LoadBulk<UserRequestedType, Int_t>;
   case kLong_t: return &LoadBulk<UserRequestedType, Long_t>;
   case kFloat_t: return &LoadBulk<UserRequestedType, Float_t>;
   case kDouble_t: return &LoadBulk<UserRequestedType, Double_t>;
   case kUChar_t: return &LoadBulk<UserRequestedType, UChar_t>;
   case kUShort_t: return &LoadBulk<UserRequestedType, UShort_t>;
   case kUInt_t: return &LoadBulk<UserRequestedType, UInt_t>;
   case kULong_t: return &LoadBulk<UserRequestedType, ULong_t>;
   case kLong64_t: return &LoadBulk<UserRequestedType, Long64_t>;
   case kULong64_t: return &LoadBulk<UserRequestedType, ULong64_t>;
   case kBool_t: return &LoadBulk<UserRequestedType, Bool_t>;
   default: return nullptr;
   }
}

auto GetBulkLoadFunction(const std::type_info &userRequestedTypeInfo, const TDictionary &onDiskDict)
{

   using MapFunc_t = ROOT::Internal::RDF::RTreeScalarBulkColumnReader::BulkLoadFunc_t (*)(EDataType);
   // One pair per each type present in frombuf signatures in core/base/inc/Bytes.h
   const static std::unordered_map<TypeInfoRef, MapFunc_t, TypeInfoRefHash, TypeInfoRefEqualComp> typeInfoToLoadBulkFn{
      {typeid(Bool_t), &GetBulkLoadFunctionImpl<Bool_t>},     {typeid(UChar_t), &GetBulkLoadFunctionImpl<UChar_t>},
      {typeid(UShort_t), &GetBulkLoadFunctionImpl<UShort_t>}, {typeid(UInt_t), &GetBulkLoadFunctionImpl<UInt_t>},
      {typeid(ULong_t), &GetBulkLoadFunctionImpl<ULong_t>},   {typeid(ULong64_t), &GetBulkLoadFunctionImpl<ULong64_t>},
      {typeid(Float_t), &GetBulkLoadFunctionImpl<Float_t>},   {typeid(Double_t), &GetBulkLoadFunctionImpl<Double_t>},
      {typeid(Char_t), &GetBulkLoadFunctionImpl<Char_t>},     {typeid(Short_t), &GetBulkLoadFunctionImpl<Short_t>},
      {typeid(Int_t), &GetBulkLoadFunctionImpl<Int_t>},       {typeid(Long_t), &GetBulkLoadFunctionImpl<Long_t>},
      {typeid(Long64_t), &GetBulkLoadFunctionImpl<Long64_t>}};

   if (auto it = typeInfoToLoadBulkFn.find(userRequestedTypeInfo); it != typeInfoToLoadBulkFn.end()) {
      if (const auto *dataType = dynamic_cast<const TDataType *>(&onDiskDict)) {
         const auto eDataType = static_cast<EDataType>(dataType->GetType());
         if (auto fn = it->second(eDataType))
            return fn;
      }
   }

   throw std::runtime_error("Could not find appropriate function to read a bulk of entries for type '" +
                            ROOT::Internal::RDF::TypeID2TypeName(userRequestedTypeInfo) + "'.");
}

template <typename UserRequestedType, typename OnDiskBranchType, typename OnDiskLeafCountType>
void LoadBulkLeafCountArray(ROOT::RVec<std::byte> &cachedValues, ROOT::RVec<ROOT::RVec<std::byte>> &cachedViews,
                            TBranch &branch, std::size_t bulkSize, std::uint64_t bulkBegin, std::uint64_t chainOffset,
                            ROOT::RVec<std::byte> &typeErasedSizes, std::uint64_t &bufElementsOffset,
                            std::unique_ptr<TBuffer> &branchBulkBuffer, std::size_t &branchBulkNumberOfEntries,
                            std::uint64_t &entriesReadFromBranchBulkBuffer, Long64_t &lastBasketBegin,
                            std::uint64_t &lastChainOffset)
{

   // Set the size of the vector of sizes so it can be correctly interpreted as a vector of the leaf count type
   const auto oldSize = typeErasedSizes.size();
   typeErasedSizes.set_size(bulkSize);
   const auto &sizes = *reinterpret_cast<ROOT::RVec<OnDiskLeafCountType> *>(&typeErasedSizes);

   // We instantiate a TBufferFile once, the calls to GetEntriesSerialized may re-allocate if a basket is bigger than
   // the size of this buffer, so we decide on an arbitrary initial size equal to the average basket size advertised by
   // the branch multiplied by the size of the type of the value we're reading.
   if (branchBulkBuffer == nullptr) {
      const auto bufSize{branch.GetBasketSize() * sizeof(OnDiskBranchType)};
      assert(bufSize < std::numeric_limits<Int_t>::max()); // TBuffer constructor expects Int_t for the buffer size
      branchBulkBuffer = std::make_unique<TBufferFile>(TBuffer::EMode::kWrite, static_cast<Int_t>(bufSize));
   }

   const auto nTotalElements = std::accumulate(sizes.begin(), sizes.end(), OnDiskLeafCountType{0});
   cachedValues.resize(nTotalElements * sizeof(UserRequestedType));

   std::size_t nEntriesLoaded{};
   std::size_t nElementsLoaded{};
   while (nEntriesLoaded < bulkSize) {
      // assert we either have not loaded a bulk yet or we exhausted the last branch bulk
      // assert(fBranchBulkSize == 0u || fBranchBulkBegin + fBranchBulkSize == fLoadedEntriesBegin + nLoaded);

      // GetEntriesSerialized does not allow reading a bulk starting from the middle of a basket, so we have to find
      // the first entry of the basket
      const auto localEntry = bulkBegin - chainOffset + nEntriesLoaded;

      // TMath::BinarySearch calls std::lower_bound
      // See same pattern in TBranch::GetBasketAndFirsts
      const Long64_t idxOfBasketContainingLocalEntry = TMath::BinarySearch(
         /*number of written baskets in the branch*/ branch.GetWriteBasket() + 1,
         /*array of first entry of each basket*/ branch.GetBasketEntry(), static_cast<Long64_t>(localEntry));
      const Long64_t basketBegin = branch.GetBasketEntry()[idxOfBasketContainingLocalEntry];
      if (lastBasketBegin != basketBegin || lastChainOffset != chainOffset) {
         // beginning of basket
         lastChainOffset = chainOffset;
         lastBasketBegin = basketBegin;
         bufElementsOffset = 0;
         entriesReadFromBranchBulkBuffer = 0;
         // GetEntriesSerialized does not byte-swap, so later we use frombuf to byte-swap and copy in a single pass
         // Returns number of entries loaded from the basket
         auto ret = branch.GetBulkRead().GetEntriesSerialized(lastBasketBegin, *branchBulkBuffer);
         assert(ret != -1); // We should always be able to read the entries from the basket
         branchBulkNumberOfEntries = ret;
      }

      const std::size_t nToSkip = localEntry - basketBegin;

      // const auto bulkElementsOffset =
      //    std::accumulate(sizes.begin() + nEntriesLoaded, sizes.begin() + std::min<std::uint64_t>(nEntriesLoaded +
      //    nToSkip, sizes.size()), std::size_t{});

      const auto nEntriesToLoad = std::min(branchBulkNumberOfEntries - nToSkip, bulkSize - nEntriesLoaded);

      // // Need a 'char *' for the call to frombuf
      auto data = reinterpret_cast<char *>(branchBulkBuffer->GetCurrent());
      data += bufElementsOffset * sizeof(OnDiskBranchType); // advance to the first element we are interested in

      const auto nElementsToLoad = std::accumulate(sizes.begin() + nEntriesLoaded,
                                                   sizes.begin() + nEntriesLoaded + nEntriesToLoad, std::size_t{});

      for (std::size_t i = 0u; i < nElementsToLoad; ++i) {
         // `frombuf` also advances the `data` pointer
         if constexpr (std::is_same_v<UserRequestedType, OnDiskBranchType>) {
            // Save one extra copy when the user-requested type is the same as the on-disk branch type
            frombuf(data, reinterpret_cast<UserRequestedType *>(cachedValues.data() +
                                                                (nElementsLoaded + i) * sizeof(UserRequestedType)));
         } else {
            // First fill a temporary value of the correct on-disk branch type, then cast it to the user-requested type
            // and copy it into the final address
            OnDiskBranchType temp{};
            frombuf(data, &temp);
            *reinterpret_cast<UserRequestedType *>(cachedValues.data() +
                                                   (nElementsLoaded + i) * sizeof(UserRequestedType)) =
               static_cast<UserRequestedType>(temp);
         }
      }

      bufElementsOffset += nElementsToLoad;

      nElementsLoaded += nElementsToLoad;
      nEntriesLoaded += nEntriesToLoad;
      entriesReadFromBranchBulkBuffer += nEntriesToLoad;
   }

   // Prepare views for later reading
   std::size_t lastOffset{};
   for (std::size_t i = 0; i < bulkSize; i++) {
      auto curSize = sizes[i] * sizeof(UserRequestedType);
      ROOT::Internal::VecOps::ResetView(cachedViews[i], cachedValues.data() + lastOffset, curSize);
      cachedViews[i].set_size(sizes[i]);
      lastOffset += curSize;
   }

   typeErasedSizes.set_size(oldSize);
}

template <typename UserRequestedType, typename OnDiskBranchType>
ROOT::Internal::RDF::RTreeLeafCountArrayBulkColumnReader::BulkLoadFunc_t
GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType(const std::type_info &leafCountTypeInfo)
{
   const static std::unordered_map<TypeInfoRef,
                                   ROOT::Internal::RDF::RTreeLeafCountArrayBulkColumnReader::BulkLoadFunc_t,
                                   TypeInfoRefHash, TypeInfoRefEqualComp>
      typeInfoToLoadBulkFn{{typeid(Bool_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, Bool_t>},
                           {typeid(UChar_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, UChar_t>},
                           {typeid(UShort_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, UShort_t>},
                           {typeid(UInt_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, UInt_t>},
                           {typeid(ULong_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, ULong_t>},
                           {typeid(ULong64_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, ULong64_t>},
                           {typeid(Float_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, Float_t>},
                           {typeid(Double_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, Double_t>},
                           {typeid(Char_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, Char_t>},
                           {typeid(Short_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, Short_t>},
                           {typeid(Int_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, Int_t>},
                           {typeid(Long_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, Long_t>},
                           {typeid(Long64_t), &LoadBulkLeafCountArray<UserRequestedType, OnDiskBranchType, Long64_t>}};

   if (auto it = typeInfoToLoadBulkFn.find(leafCountTypeInfo); it != typeInfoToLoadBulkFn.end())
      return it->second;

   return nullptr;
}

template <typename UserRequestedType>
ROOT::Internal::RDF::RTreeLeafCountArrayBulkColumnReader::BulkLoadFunc_t
GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo(EDataType dt, const std::type_info &leafCountTypeInfo)
{
   switch (dt) {
   case kChar_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, Char_t>(
         leafCountTypeInfo);
   case kShort_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, Short_t>(
         leafCountTypeInfo);
   case kInt_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, Int_t>(
         leafCountTypeInfo);
   case kLong_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, Long_t>(
         leafCountTypeInfo);
   case kFloat_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, Float_t>(
         leafCountTypeInfo);
   case kDouble_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, Double_t>(
         leafCountTypeInfo);
   case kUChar_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, UChar_t>(
         leafCountTypeInfo);
   case kUShort_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, UShort_t>(
         leafCountTypeInfo);
   case kUInt_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, UInt_t>(
         leafCountTypeInfo);
   case kULong_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, ULong_t>(
         leafCountTypeInfo);
   case kLong64_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, Long64_t>(
         leafCountTypeInfo);
   case kULong64_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, ULong64_t>(
         leafCountTypeInfo);
   case kBool_t:
      return GetLeafCountArrayBulkLoadFunctionNoUserTypeInfoNoOnDiskBranchType<UserRequestedType, Bool_t>(
         leafCountTypeInfo);
   default: return nullptr;
   }
}

auto GetLeafCountArrayBulkLoadFunction(const std::type_info &userRequestedTypeInfo, const TDictionary &onDiskDict,
                                       const std::type_info &leafCountTypeInfo)
{

   using MapFunc_t =
      ROOT::Internal::RDF::RTreeLeafCountArrayBulkColumnReader::BulkLoadFunc_t (*)(EDataType, const std::type_info &);
   // One pair per each type present in frombuf signatures in core/base/inc/Bytes.h
   const static std::unordered_map<TypeInfoRef, MapFunc_t, TypeInfoRefHash, TypeInfoRefEqualComp> typeInfoToLoadBulkFn{
      {typeid(Bool_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<Bool_t>},
      {typeid(UChar_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<UChar_t>},
      {typeid(UShort_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<UShort_t>},
      {typeid(UInt_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<UInt_t>},
      {typeid(ULong_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<ULong_t>},
      {typeid(ULong64_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<ULong64_t>},
      {typeid(Float_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<Float_t>},
      {typeid(Double_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<Double_t>},
      {typeid(Char_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<Char_t>},
      {typeid(Short_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<Short_t>},
      {typeid(Int_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<Int_t>},
      {typeid(Long_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<Long_t>},
      {typeid(Long64_t), &GetLeafCountArrayBulkLoadFunctionNoUserTypeInfo<Long64_t>}};

   if (auto it = typeInfoToLoadBulkFn.find(userRequestedTypeInfo); it != typeInfoToLoadBulkFn.end()) {
      if (const auto *dataType = dynamic_cast<const TDataType *>(&onDiskDict)) {
         const auto eDataType = static_cast<EDataType>(dataType->GetType());
         if (auto fn = it->second(eDataType, leafCountTypeInfo))
            return fn;
      }
   }

   throw std::runtime_error("Could not find appropriate function to read a bulk of entries for type '" +
                            ROOT::Internal::RDF::TypeID2TypeName(userRequestedTypeInfo) + "'.");
}

} // namespace

void *ROOT::Internal::RDF::RTreeOpaqueColumnReader::GetImpl(std::size_t entryInBulk)
{
   return fIsCachedValueValidAtIdx[entryInBulk] ? &fIsCachedValueValidAtIdx[entryInBulk] : nullptr;
}

void ROOT::Internal::RDF::RTreeOpaqueColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   if (fLastEntry == mask.GetFirstEntry())
      return;

   const auto validIndices = mask.GetValidIndices();
   fLastEntry = mask.GetFirstEntry();
   if (validIndices.empty())
      return;

   std::fill(fIsCachedValueValidAtIdx.begin(), fIsCachedValueValidAtIdx.end(), false);

   for (auto idx : validIndices) {
      if (fTreeReader.GetCurrentEntry() != static_cast<Long64_t>(mask.GetFirstEntry() + idx))
         fTreeReader.SetEntry(mask.GetFirstEntry() + idx);
      fIsCachedValueValidAtIdx[idx] = fTreeValue->GetAddress();
   }
}

ROOT::Internal::RDF::RTreeOpaqueColumnReader::RTreeOpaqueColumnReader(TTreeReader &r, std::string_view colName,
                                                                      std::size_t maxBulkSize)
   : fTreeReader(r),
     fTreeValue(std::make_unique<ROOT::Internal::TTreeReaderOpaqueValue>(r, colName.data())),
     fLastEntry(std::numeric_limits<std::uint64_t>::max()),
     fIsCachedValueValidAtIdx(maxBulkSize, false)
{
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
      if (mask.GetBulkSize() > 1)
         throw std::runtime_error(
            "Cannot read TTree branch containing an unsplit TBranchObject with bulk size greater than 1.");
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

      if (fTreeReader.GetCurrentEntry() != static_cast<Long64_t>(mask.GetFirstEntry() + idx))
         fTreeReader.SetEntry(mask.GetFirstEntry() + idx);
      fTreeValue->SetAddress(fCachedBranchAddresses[idx], fCachedBranchAddressesOfAddresses[idx]);
      fIsCachedValueValidAtIdx[idx] = bool(fTreeValue->Get());
   }

   fLastEntry = mask.GetFirstEntry();
}

ROOT::Internal::RDF::RTreeUntypedValueColumnReader::RTreeUntypedValueColumnReader(TTreeReader &r,
                                                                                  std::string_view colName,
                                                                                  std::string_view typeName,
                                                                                  std::size_t maxBulkSize)
   : fTreeReader(r),
     fTreeValue(std::make_unique<ROOT::Internal::TTreeReaderUntypedValue>(r, colName, typeName)),
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
      fIsCachedValueValidAtIdx[entryInBulk] = false;
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
      fIsCachedValueValidAtIdx[entryInBulk] = false;
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
      fIsCachedValueValidAtIdx[entryInBulk] = false;
      return;
   }

   // The array is empty, avoid doing extra work
   if (readerArraySize == 0)
      return;

   auto &cachedValues = std::get<0>(fCachedValues);

   // Caching the value type size since GetValueSize might be expensive.
   if (fValueSize == 0)
      fValueSize = readerArray.GetValueSize();
   assert(fValueSize > 0 && "Could not retrieve size of collection value type.");
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

void ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{

   if (fLastEntry == mask.GetFirstEntry())
      return;

   const auto validIndices = mask.GetValidIndices();
   if (validIndices.empty())
      return;

   std::fill(fIsCachedValueValidAtIdx.begin(), fIsCachedValueValidAtIdx.end(), true);

   std::visit(
      [](auto &&visited) {
         for (auto &coll : visited)
            coll.clear();
      },
      fCachedValues);

   for (auto idx : validIndices) {
      if (fTreeReader.GetCurrentEntry() != static_cast<Long64_t>(mask.GetFirstEntry() + idx))
         fTreeReader.SetEntry(mask.GetFirstEntry() + idx);
      switch (fCollectionType) {
      case ECollectionType::kStdArray: LoadStdArray(idx); break;
      case ECollectionType::kRVec:
      case ECollectionType::kRVecBool: LoadRVec(idx); break;
      case ECollectionType::kStdVector: LoadStdVector(idx); break;
      default: throw std::runtime_error("RDataFrame: unrecognized value type to read a collection in a bulk.");
      }
   }

   fLastEntry = mask.GetFirstEntry();
}

void *ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::GetImpl(std::size_t entryInBulk)
{
   if (!fIsCachedValueValidAtIdx[entryInBulk]) {
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
   : fTreeReader(r),
     fTreeArray(std::make_unique<ROOT::Internal::TTreeReaderUntypedArray>(r, colName, valueTypeName)),
     fCollectionType(collType),
     fLastEntry(std::numeric_limits<std::uint64_t>::max()),
     fValueSize(0),
     fIsCachedValueValidAtIdx(maxBulkSize, true)
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
   : fTreeReader(r),
     fValueReader(std::move(valueReader)),
     fTreeValueMask{std::make_unique<TTreeReaderValue<uint64_t>>(r, maskName.data())},
     fMaskIndex(maskIndex),
     fLastEntry(std::numeric_limits<std::uint64_t>::max()),
     fIsCachedValueValidAtIdx(maxBulkSize, false)
{
}

ROOT::Internal::RDF::RMaskedColumnReader::~RMaskedColumnReader() = default;

void *ROOT::Internal::RDF::RMaskedColumnReader::GetImpl(std::size_t entryInBulk)
{
   if (fIsCachedValueValidAtIdx[entryInBulk])
      return fValueReader->TryGet<void>(entryInBulk);
   return nullptr;
}

void ROOT::Internal::RDF::RMaskedColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   if (fLastEntry == mask.GetFirstEntry())
      return;

   const auto validIndices = mask.GetValidIndices();
   fLastEntry = mask.GetFirstEntry();
   if (validIndices.empty())
      return;

   std::fill(fIsCachedValueValidAtIdx.begin(), fIsCachedValueValidAtIdx.end(), false);
   fValueReader->Load(mask);
   auto bulkSize = mask.GetBulkSize();
   for (decltype(bulkSize) idx = 0; idx < bulkSize; idx++) {
      if (fTreeReader.GetCurrentEntry() != static_cast<Long64_t>(mask.GetFirstEntry() + idx))
         fTreeReader.SetEntry(mask.GetFirstEntry() + idx);
      const std::bitset<64> treeMask{*fTreeValueMask->Get()};
      fIsCachedValueValidAtIdx[idx] = treeMask.test(fMaskIndex);
   }
}

ROOT::Internal::RDF::RTreeScalarBulkColumnReader::RTreeScalarBulkColumnReader(TTreeReader &r, std::string_view colName,
                                                                              std::string_view typeName,
                                                                              const std::type_info &ti,
                                                                              std::size_t maxBulkSize)
   : fTreeReader(r),
     fTreeValue(std::make_unique<ROOT::Internal::TTreeReaderUntypedValue>(r, colName, typeName)),
     fBulkSize(maxBulkSize),
     fValueSize(),
     fCachedValues(),
     fIsBulkValid(false),
     fLastEntry(std::numeric_limits<std::uint64_t>::max()),
     fUserRequestedTypeInfo(ti),
     fCachedBuffer(nullptr),
     fBranchBulkNumberOfEntries(0),
     fEntriesReadFromCachedBuffer(0),
     fLastBasketBegin(-1),
     fLastChainOffset(0)
{
}

ROOT::Internal::RDF::RTreeScalarBulkColumnReader::~RTreeScalarBulkColumnReader() = default;

void *ROOT::Internal::RDF::RTreeScalarBulkColumnReader::GetImpl(std::size_t entryInBulk)
{
   if (!fIsBulkValid)
      return nullptr;

   return fCachedValues.data() + entryInBulk * fValueSize;
}

void ROOT::Internal::RDF::RTreeScalarBulkColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   if (fLastEntry == mask.GetFirstEntry())
      return;

   // // Other readers may have changed the cursor
   if (fTreeReader.GetCurrentEntry() != static_cast<Long64_t>(mask.GetFirstEntry()))
      fTreeReader.SetEntry(mask.GetFirstEntry());

   // Branch might be unavailable (e.g. missing from the curent tree)
   auto branch = fTreeValue->GetBranch();
   if (branch) {
      fIsBulkValid = true;
   } else {
      fIsBulkValid = false;
      return;
   }

   auto *onDiskDict = fTreeValue->GetOnDiskDictionary();
   auto bulkLoadFunction = GetBulkLoadFunction(fUserRequestedTypeInfo, *onDiskDict);

   // This can only happen at first loading time, when the TTreeReader has begun traversing entries and the branch
   // proxy is set.
   if (fValueSize == 0)
      fValueSize = fTreeValue->GetValueSize();
   assert(fValueSize > 0 && "Could not retrieve size of value type in RDataFrame TTree value column reader.");
   // First time setup for cache of bulk values
   if (fCachedValues.empty()) {
      fCachedValues.resize(fBulkSize * fValueSize);
   }

   fLastEntry = mask.GetFirstEntry();

   bulkLoadFunction(fCachedValues, *branch, mask.GetBulkSize(), fLastEntry, branch->GetTree()->GetChainOffset(),
                    fCachedBuffer, fBranchBulkNumberOfEntries, fEntriesReadFromCachedBuffer, fLastBasketBegin,
                    fLastChainOffset);
}

void *ROOT::Internal::RDF::RTreeGenericValueColumnReader::GetImpl(std::size_t entryInBulk)
{
   return std::visit([&](auto &&arg) { return arg->GetImpl(entryInBulk); }, fReader);
}

void ROOT::Internal::RDF::RTreeGenericValueColumnReader::LoadImpl(const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   std::visit([&](auto &&arg) { arg->LoadImpl(mask); }, fReader);
}

ROOT::Internal::RDF::RTreeGenericValueColumnReader::~RTreeGenericValueColumnReader() = default;

ROOT::Internal::RDF::RTreeGenericValueColumnReader::RTreeGenericValueColumnReader(TTreeReader &r,
                                                                                  std::string_view colName,
                                                                                  std::string_view typeName,
                                                                                  const std::type_info &ti,
                                                                                  std::size_t maxBulkSize)
{
   if (DatasetContainsIndexedFriends(r.GetTree()) || DatasetHasEntryList(r)) {
      fReader = std::make_unique<RTreeUntypedValueColumnReader>(r, colName, typeName, maxBulkSize);
      return;
   }

   auto isValidBranchForBulkReading = [&]() {
      auto *tree = r.GetTree();
      if (!tree)
         return false;
      auto *branch = tree->GetBranch(colName.data());
      if (!branch)
         return false;
      // We can't read collections of split data members with the TBranch bulk reading API
      if (auto brEl = dynamic_cast<TBranchElement *>(branch);
          brEl &&
          (brEl->GetType() == TBranchElement::kClonesMemberNode || brEl->GetType() == TBranchElement::kSTLMemberNode))
         return false;

      return true;
   };

   if (isValidBranchForBulkReading()) {
      // One per each type present in frombuf signatures in core/base/inc/Bytes.h
      const static std::unordered_set<TypeInfoRef, TypeInfoRefHash, TypeInfoRefEqualComp> validTypesForBulkReading{
         typeid(Bool_t),    typeid(UChar_t), typeid(UShort_t), typeid(UInt_t), typeid(ULong_t),
         typeid(ULong64_t), typeid(Float_t), typeid(Double_t), typeid(Char_t), typeid(Short_t),
         typeid(Int_t),     typeid(Long_t),  typeid(Long64_t)};

      if (auto it = validTypesForBulkReading.find(ti); it != validTypesForBulkReading.end()) {
         fReader =
            std::make_unique<ROOT::Internal::RDF::RTreeScalarBulkColumnReader>(r, colName, typeName, ti, maxBulkSize);
         return;
      }
   }

   fReader = std::make_unique<RTreeUntypedValueColumnReader>(r, colName, typeName, maxBulkSize);
}

void *ROOT::Internal::RDF::RTreeLeafCountArrayBulkColumnReader::GetImpl(std::size_t entryInBulk)
{
   if (!fIsBulkValid)
      return nullptr;

   return &fCachedViews[entryInBulk];
}

void ROOT::Internal::RDF::RTreeLeafCountArrayBulkColumnReader::LoadImpl(
   const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   if (fLastEntry == mask.GetFirstEntry()) {
      return;
   }

   // Other readers may have changed the cursor
   if (fTreeReader.GetCurrentEntry() != static_cast<Long64_t>(mask.GetFirstEntry()))
      fTreeReader.SetEntry(mask.GetFirstEntry());

   // Branch might be unavailable (e.g. missing from the curent tree)
   auto branch = fTreeArray->GetBranch();
   if (branch) {
      fIsBulkValid = true;
   } else {
      fIsBulkValid = false;
      return;
   }

   fLastEntry = mask.GetFirstEntry();

   fLeafCountSizesReader->LoadImpl(mask);

   auto *onDiskDict = fTreeArray->GetOnDiskDictionary();
   auto bulkLoadFunction =
      GetLeafCountArrayBulkLoadFunction(fUserRequestedValueTypeInfo, *onDiskDict, fLeafCountTypeInfo);

   bulkLoadFunction(fCachedValues, fCachedViews, *branch, mask.GetBulkSize(), mask.GetFirstEntry(),
                    branch->GetTree()->GetChainOffset(), fLeafCountSizesReader->fCachedValues, fBufElementsOffset,
                    fCachedBuffer, fBranchBulkNumberOfEntries, fEntriesReadFromCachedBuffer, fLastBasketBegin,
                    fLastChainOffset);
}

ROOT::Internal::RDF::RTreeLeafCountArrayBulkColumnReader::RTreeLeafCountArrayBulkColumnReader(
   TTreeReader &r, std::string_view arrayColName, std::string_view valueTypeName, const std::type_info &valueTypeInfo,
   const std::type_info &leafCountTypeInfo, std::unique_ptr<RTreeScalarBulkColumnReader> sizeReader,
   std::size_t maxBulkSize)
   : fTreeReader(r),
     fTreeArray(std::make_unique<ROOT::Internal::TTreeReaderUntypedArray>(r, arrayColName, valueTypeName)),
     fCachedValues(1), // Initialize cached values just to set RVec views on them
     fIsBulkValid(false),
     fLastEntry(std::numeric_limits<std::uint64_t>::max()),
     fUserRequestedValueTypeInfo(valueTypeInfo),
     fLeafCountSizesReader(std::move(sizeReader)),
     fLeafCountTypeInfo(leafCountTypeInfo),
     fBufElementsOffset(0),
     fCachedBuffer(nullptr),
     fBranchBulkNumberOfEntries(0),
     fEntriesReadFromCachedBuffer(0),
     fLastBasketBegin(-1),
     fLastChainOffset(0)
{
   fCachedViews.reserve(maxBulkSize);
   // Initialize all RVec views, so later on we can just reset their data members without triggering owning checks
   for (std::size_t i = 0; i < maxBulkSize; i++)
      fCachedViews.emplace_back(fCachedValues.data(), 1);
}

ROOT::Internal::RDF::RTreeLeafCountArrayBulkColumnReader::~RTreeLeafCountArrayBulkColumnReader() = default;

void *ROOT::Internal::RDF::RTreeGenericCollectionColumnReader::GetImpl(std::size_t entryInBulk)
{
   return std::visit([&](auto &&arg) { return arg->GetImpl(entryInBulk); }, fReader);
}

void ROOT::Internal::RDF::RTreeGenericCollectionColumnReader::LoadImpl(
   const ROOT::Internal::RDF::RMaskedEntryRange &mask)
{
   std::visit([&](auto &&arg) { arg->LoadImpl(mask); }, fReader);
}

ROOT::Internal::RDF::RTreeGenericCollectionColumnReader::~RTreeGenericCollectionColumnReader() = default;

ROOT::Internal::RDF::RTreeGenericCollectionColumnReader::RTreeGenericCollectionColumnReader(
   TTreeReader &r, std::string_view arrayColName, std::string_view valueTypeName, const std::type_info &valueTypeInfo,
   ROOT::Internal::RDF::RTreeUntypedArrayColumnReader::ECollectionType collectionType, std::size_t maxBulkSize)
{
   if (DatasetContainsIndexedFriends(r.GetTree()) || DatasetHasEntryList(r)) {
      fReader =
         std::make_unique<RTreeUntypedArrayColumnReader>(r, arrayColName, valueTypeName, collectionType, maxBulkSize);
      return;
   }

   // One per each type present in frombuf signatures in core/base/inc/Bytes.h
   const static std::unordered_set<TypeInfoRef, TypeInfoRefHash, TypeInfoRefEqualComp> validTypesForBulkReading{
      typeid(Bool_t),    typeid(UChar_t), typeid(UShort_t), typeid(UInt_t), typeid(ULong_t),
      typeid(ULong64_t), typeid(Float_t), typeid(Double_t), typeid(Char_t), typeid(Short_t),
      typeid(Int_t),     typeid(Long_t),  typeid(Long64_t)};

   auto createLeafCountArrayReader = [&](bool &success) -> std::unique_ptr<RTreeLeafCountArrayBulkColumnReader> {
      auto *tree = r.GetTree();
      if (!tree)
         return nullptr;
      auto *branch = tree->GetBranch(arrayColName.data());
      if (!branch)
         return nullptr;
      // We can't read collections of split data members with the TBranch bulk reading API
      if (auto brEl = dynamic_cast<TBranchElement *>(branch);
          brEl &&
          (brEl->GetType() == TBranchElement::kClonesMemberNode || brEl->GetType() == TBranchElement::kSTLMemberNode))
         return nullptr;
      auto *listOfLeaves = branch->GetListOfLeaves();
      if (!listOfLeaves)
         return nullptr;
      if (listOfLeaves->GetEntries() != 1)
         return nullptr;
      auto *leaf = static_cast<TLeaf *>(listOfLeaves->UncheckedAt(0));
      if (!leaf)
         return nullptr;
      auto *leafCount = leaf->GetLeafCount();
      if (!leafCount)
         return nullptr;

      success = true;
      std::string leafCountTypeName = leafCount->GetTypeName();
      const auto &leafCountTypeInfo = ROOT::Internal::RDF::TypeName2TypeID(leafCountTypeName);
      auto leafCountSizeReader = std::make_unique<RTreeScalarBulkColumnReader>(
         r, leafCount->GetName(), leafCountTypeName, leafCountTypeInfo, maxBulkSize);
      return std::make_unique<RTreeLeafCountArrayBulkColumnReader>(
         r, arrayColName, valueTypeName, valueTypeInfo, leafCountTypeInfo, std::move(leafCountSizeReader), maxBulkSize);
   };
   if (auto it = validTypesForBulkReading.find(valueTypeInfo); it != validTypesForBulkReading.end()) {
      bool success{false};
      fReader = createLeafCountArrayReader(success);
      if (success)
         return;
   }

   fReader =
      std::make_unique<RTreeUntypedArrayColumnReader>(r, arrayColName, valueTypeName, collectionType, maxBulkSize);
}
