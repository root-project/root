/// \file RFieldSequenceContainer.cxx
/// \ingroup NTuple ROOT7
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include <ROOT/RField.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include "RFieldUtils.hxx"

#include <cstdlib> // for malloc, free
#include <memory>
#include <new> // hardware_destructive_interference_size

ROOT::Experimental::RArrayField::RArrayField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField,
                                             std::size_t arrayLength)
   : ROOT::Experimental::RFieldBase(fieldName,
                                    "std::array<" + itemField->GetTypeName() + "," +
                                       Internal::GetNormalizedInteger(static_cast<unsigned long long>(arrayLength)) +
                                       ">",
                                    ROOT::ENTupleStructure::kLeaf, false /* isSimple */, arrayLength),
     fItemSize(itemField->GetValueSize()),
     fArrayLength(arrayLength)
{
   fTraits |= itemField->GetTraits() & ~kTraitMappable;
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RArrayField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::make_unique<RArrayField>(newName, std::move(newItemField), fArrayLength);
}

std::size_t ROOT::Experimental::RArrayField::AppendImpl(const void *from)
{
   std::size_t nbytes = 0;
   if (fSubFields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubFields[0])->AppendV(from, fArrayLength);
      nbytes += fArrayLength * GetPrincipalColumnOf(*fSubFields[0])->GetElement()->GetPackedSize();
   } else {
      auto arrayPtr = static_cast<const unsigned char *>(from);
      for (unsigned i = 0; i < fArrayLength; ++i) {
         nbytes += CallAppendOn(*fSubFields[0], arrayPtr + (i * fItemSize));
      }
   }
   return nbytes;
}

void ROOT::Experimental::RArrayField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   if (fSubFields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubFields[0])->ReadV(globalIndex * fArrayLength, fArrayLength, to);
   } else {
      auto arrayPtr = static_cast<unsigned char *>(to);
      for (unsigned i = 0; i < fArrayLength; ++i) {
         CallReadOn(*fSubFields[0], globalIndex * fArrayLength + i, arrayPtr + (i * fItemSize));
      }
   }
}

void ROOT::Experimental::RArrayField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   if (fSubFields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubFields[0])
         ->ReadV(RNTupleLocalIndex(localIndex.GetClusterId(), localIndex.GetIndexInCluster() * fArrayLength),
                 fArrayLength, to);
   } else {
      auto arrayPtr = static_cast<unsigned char *>(to);
      for (unsigned i = 0; i < fArrayLength; ++i) {
         CallReadOn(*fSubFields[0],
                    RNTupleLocalIndex(localIndex.GetClusterId(), localIndex.GetIndexInCluster() * fArrayLength + i),
                    arrayPtr + (i * fItemSize));
      }
   }
}

void ROOT::Experimental::RArrayField::ConstructValue(void *where) const
{
   if (fSubFields[0]->GetTraits() & kTraitTriviallyConstructible)
      return;

   auto arrayPtr = reinterpret_cast<unsigned char *>(where);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      CallConstructValueOn(*fSubFields[0], arrayPtr + (i * fItemSize));
   }
}

void ROOT::Experimental::RArrayField::RArrayDeleter::operator()(void *objPtr, bool dtorOnly)
{
   if (fItemDeleter) {
      for (unsigned i = 0; i < fArrayLength; ++i) {
         fItemDeleter->operator()(reinterpret_cast<unsigned char *>(objPtr) + i * fItemSize, true /* dtorOnly */);
      }
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::RArrayField::GetDeleter() const
{
   if (!(fSubFields[0]->GetTraits() & kTraitTriviallyDestructible))
      return std::make_unique<RArrayDeleter>(fItemSize, fArrayLength, GetDeleterOf(*fSubFields[0]));
   return std::make_unique<RDeleter>();
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RArrayField::SplitValue(const RValue &value) const
{
   auto arrayPtr = value.GetPtr<unsigned char>().get();
   std::vector<RValue> result;
   result.reserve(fArrayLength);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      result.emplace_back(
         fSubFields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), arrayPtr + (i * fItemSize))));
   }
   return result;
}

void ROOT::Experimental::RArrayField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitArrayField(*this);
}

//------------------------------------------------------------------------------

namespace {

/// Retrieve the addresses of the data members of a generic RVec from a pointer to the beginning of the RVec object.
/// Returns pointers to fBegin, fSize and fCapacity in a std::tuple.
std::tuple<void **, std::int32_t *, std::int32_t *> GetRVecDataMembers(void *rvecPtr)
{
   void **begin = reinterpret_cast<void **>(rvecPtr);
   // int32_t fSize is the second data member (after 1 void*)
   std::int32_t *size = reinterpret_cast<std::int32_t *>(begin + 1);
   R__ASSERT(*size >= 0);
   // int32_t fCapacity is the third data member (1 int32_t after fSize)
   std::int32_t *capacity = size + 1;
   R__ASSERT(*capacity >= -1);
   return {begin, size, capacity};
}

std::tuple<const void *const *, const std::int32_t *, const std::int32_t *> GetRVecDataMembers(const void *rvecPtr)
{
   return {GetRVecDataMembers(const_cast<void *>(rvecPtr))};
}

std::size_t EvalRVecValueSize(std::size_t alignOfT, std::size_t sizeOfT, std::size_t alignOfRVecT)
{
   // the size of an RVec<T> is the size of its 4 data-members + optional padding:
   //
   // data members:
   // - void *fBegin
   // - int32_t fSize
   // - int32_t fCapacity
   // - the char[] inline storage, which is aligned like T
   //
   // padding might be present:
   // - between fCapacity and the char[] buffer aligned like T
   // - after the char[] buffer

   constexpr auto dataMemberSz = sizeof(void *) + 2 * sizeof(std::int32_t);

   // mimic the logic of RVecInlineStorageSize, but at runtime
   const auto inlineStorageSz = [&] {
#ifdef R__HAS_HARDWARE_INTERFERENCE_SIZE
      // hardware_destructive_interference_size is a C++17 feature but many compilers do not implement it yet
      constexpr unsigned cacheLineSize = std::hardware_destructive_interference_size;
#else
      constexpr unsigned cacheLineSize = 64u;
#endif
      const unsigned elementsPerCacheLine = (cacheLineSize - dataMemberSz) / sizeOfT;
      constexpr unsigned maxInlineByteSize = 1024;
      const unsigned nElements =
         elementsPerCacheLine >= 8 ? elementsPerCacheLine : (sizeOfT * 8 > maxInlineByteSize ? 0 : 8);
      return nElements * sizeOfT;
   }();

   // compute padding between first 3 datamembers and inline buffer
   // (there should be no padding between the first 3 data members)
   auto paddingMiddle = dataMemberSz % alignOfT;
   if (paddingMiddle != 0)
      paddingMiddle = alignOfT - paddingMiddle;

   // padding at the end of the object
   auto paddingEnd = (dataMemberSz + paddingMiddle + inlineStorageSz) % alignOfRVecT;
   if (paddingEnd != 0)
      paddingEnd = alignOfRVecT - paddingEnd;

   return dataMemberSz + inlineStorageSz + paddingMiddle + paddingEnd;
}

std::size_t EvalRVecAlignment(std::size_t alignOfSubField)
{
   // the alignment of an RVec<T> is the largest among the alignments of its data members
   // (including the inline buffer which has the same alignment as the RVec::value_type)
   return std::max({alignof(void *), alignof(std::int32_t), alignOfSubField});
}

void DestroyRVecWithChecks(std::size_t alignOfT, void **beginPtr, char *begin, std::int32_t *capacityPtr)
{
   // figure out if we are in the small state, i.e. begin == &inlineBuffer
   // there might be padding between fCapacity and the inline buffer, so we compute it here
   constexpr auto dataMemberSz = sizeof(void *) + 2 * sizeof(std::int32_t);
   auto paddingMiddle = dataMemberSz % alignOfT;
   if (paddingMiddle != 0)
      paddingMiddle = alignOfT - paddingMiddle;
   const bool isSmall = (begin == (reinterpret_cast<char *>(beginPtr) + dataMemberSz + paddingMiddle));

   const bool owns = (*capacityPtr != -1);
   if (!isSmall && owns)
      free(begin);
}

} // anonymous namespace

ROOT::Experimental::RRVecField::RRVecField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
   : ROOT::Experimental::RFieldBase(fieldName, "ROOT::VecOps::RVec<" + itemField->GetTypeName() + ">",
                                    ROOT::ENTupleStructure::kCollection, false /* isSimple */),
     fItemSize(itemField->GetValueSize()),
     fNWritten(0)
{
   if (!(itemField->GetTraits() & kTraitTriviallyDestructible))
      fItemDeleter = GetDeleterOf(*itemField);
   Attach(std::move(itemField));
   fValueSize = EvalRVecValueSize(fSubFields[0]->GetAlignment(), fSubFields[0]->GetValueSize(), GetAlignment());
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RRVecField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::make_unique<RRVecField>(newName, std::move(newItemField));
}

std::size_t ROOT::Experimental::RRVecField::AppendImpl(const void *from)
{
   auto [beginPtr, sizePtr, _] = GetRVecDataMembers(from);

   std::size_t nbytes = 0;
   if (fSubFields[0]->IsSimple() && *sizePtr) {
      GetPrincipalColumnOf(*fSubFields[0])->AppendV(*beginPtr, *sizePtr);
      nbytes += *sizePtr * GetPrincipalColumnOf(*fSubFields[0])->GetElement()->GetPackedSize();
   } else {
      auto begin = reinterpret_cast<const char *>(*beginPtr); // for pointer arithmetics
      for (std::int32_t i = 0; i < *sizePtr; ++i) {
         nbytes += CallAppendOn(*fSubFields[0], begin + i * fItemSize);
      }
   }

   fNWritten += *sizePtr;
   fPrincipalColumn->Append(&fNWritten);
   return nbytes + fPrincipalColumn->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RRVecField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   // TODO as a performance optimization, we could assign values to elements of the inline buffer:
   // if size < inline buffer size: we save one allocation here and usage of the RVec skips a pointer indirection

   auto [beginPtr, sizePtr, capacityPtr] = GetRVecDataMembers(to);

   // Read collection info for this entry
   ROOT::NTupleSize_t nItems;
   RNTupleLocalIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);
   char *begin = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics
   const std::size_t oldSize = *sizePtr;

   // See "semantics of reading non-trivial objects" in RNTuple's Architecture.md for details
   // on the element construction/destrution.
   const bool owns = (*capacityPtr != -1);
   const bool needsConstruct = !(fSubFields[0]->GetTraits() & kTraitTriviallyConstructible);
   const bool needsDestruct = owns && fItemDeleter;

   // Destroy excess elements, if any
   if (needsDestruct) {
      for (std::size_t i = nItems; i < oldSize; ++i) {
         fItemDeleter->operator()(begin + (i * fItemSize), true /* dtorOnly */);
      }
   }

   // Resize RVec (capacity and size)
   if (std::int32_t(nItems) > *capacityPtr) { // must reallocate
      // Destroy old elements: useless work for trivial types, but in case the element type's constructor
      // allocates memory we need to release it here to avoid memleaks (e.g. if this is an RVec<RVec<int>>)
      if (needsDestruct) {
         for (std::size_t i = 0u; i < oldSize; ++i) {
            fItemDeleter->operator()(begin + (i * fItemSize), true /* dtorOnly */);
         }
      }

      // TODO Increment capacity by a factor rather than just enough to fit the elements.
      if (owns) {
         // *beginPtr points to the array of item values (allocated in an earlier call by the following malloc())
         free(*beginPtr);
      }
      // We trust that malloc returns a buffer with large enough alignment.
      // This might not be the case if T in RVec<T> is over-aligned.
      *beginPtr = malloc(nItems * fItemSize);
      R__ASSERT(*beginPtr != nullptr);
      begin = reinterpret_cast<char *>(*beginPtr);
      *capacityPtr = nItems;

      // Placement new for elements that were already there before the resize
      if (needsConstruct) {
         for (std::size_t i = 0u; i < oldSize; ++i)
            CallConstructValueOn(*fSubFields[0], begin + (i * fItemSize));
      }
   }
   *sizePtr = nItems;

   // Placement new for new elements, if any
   if (needsConstruct) {
      for (std::size_t i = oldSize; i < nItems; ++i)
         CallConstructValueOn(*fSubFields[0], begin + (i * fItemSize));
   }

   if (fSubFields[0]->IsSimple() && nItems) {
      GetPrincipalColumnOf(*fSubFields[0])->ReadV(collectionStart, nItems, begin);
      return;
   }

   // Read the new values into the collection elements
   for (std::size_t i = 0; i < nItems; ++i) {
      CallReadOn(*fSubFields[0], collectionStart + i, begin + (i * fItemSize));
   }
}

std::size_t ROOT::Experimental::RRVecField::ReadBulkImpl(const RBulkSpec &bulkSpec)
{
   if (!fSubFields[0]->IsSimple())
      return RFieldBase::ReadBulkImpl(bulkSpec);

   if (bulkSpec.fAuxData->empty()) {
      /// Initialize auxiliary memory: the first sizeof(size_t) bytes store the value size of the item field.
      /// The following bytes store the item values, consecutively.
      bulkSpec.fAuxData->resize(sizeof(std::size_t));
      *reinterpret_cast<std::size_t *>(bulkSpec.fAuxData->data()) = fSubFields[0]->GetValueSize();
   }
   const auto itemValueSize = *reinterpret_cast<std::size_t *>(bulkSpec.fAuxData->data());
   unsigned char *itemValueArray = bulkSpec.fAuxData->data() + sizeof(std::size_t);
   auto [beginPtr, sizePtr, capacityPtr] = GetRVecDataMembers(bulkSpec.fValues);

   // Get size of the first RVec of the bulk
   RNTupleLocalIndex firstItemIndex;
   ROOT::NTupleSize_t collectionSize;
   this->GetCollectionInfo(bulkSpec.fFirstIndex, &firstItemIndex, &collectionSize);
   *beginPtr = itemValueArray;
   *sizePtr = collectionSize;
   *capacityPtr = -1;

   // Set the size of the remaining RVecs of the bulk, going page by page through the RNTuple offset column.
   // We optimistically assume that bulkSpec.fAuxData is already large enough to hold all the item values in the
   // given range. If not, we'll fix up the pointers afterwards.
   auto lastOffset = firstItemIndex.GetIndexInCluster() + collectionSize;
   ROOT::NTupleSize_t nRemainingValues = bulkSpec.fCount - 1;
   std::size_t nValues = 1;
   std::size_t nItems = collectionSize;
   while (nRemainingValues > 0) {
      ROOT::NTupleSize_t nElementsUntilPageEnd;
      const auto offsets =
         fPrincipalColumn->MapV<Internal::RColumnIndex>(bulkSpec.fFirstIndex + nValues, nElementsUntilPageEnd);
      const std::size_t nBatch = std::min(nRemainingValues, nElementsUntilPageEnd);
      for (std::size_t i = 0; i < nBatch; ++i) {
         const auto size = offsets[i] - lastOffset;
         std::tie(beginPtr, sizePtr, capacityPtr) =
            GetRVecDataMembers(reinterpret_cast<unsigned char *>(bulkSpec.fValues) + (nValues + i) * fValueSize);
         *beginPtr = itemValueArray + nItems * itemValueSize;
         *sizePtr = size;
         *capacityPtr = -1;

         nItems += size;
         lastOffset = offsets[i];
      }
      nRemainingValues -= nBatch;
      nValues += nBatch;
   }

   bulkSpec.fAuxData->resize(sizeof(std::size_t) + nItems * itemValueSize);
   // If the vector got reallocated, we need to fix-up the RVecs begin pointers.
   const auto delta = itemValueArray - (bulkSpec.fAuxData->data() + sizeof(std::size_t));
   if (delta != 0) {
      auto beginPtrAsUChar = reinterpret_cast<unsigned char *>(bulkSpec.fValues);
      for (std::size_t i = 0; i < bulkSpec.fCount; ++i) {
         *reinterpret_cast<unsigned char **>(beginPtrAsUChar) -= delta;
         beginPtrAsUChar += fValueSize;
      }
   }

   GetPrincipalColumnOf(*fSubFields[0])->ReadV(firstItemIndex, nItems, itemValueArray - delta);
   return RBulkSpec::kAllSet;
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RRVecField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RRVecField::GenerateColumns()
{
   GenerateColumnsImpl<Internal::RColumnIndex>();
}

void ROOT::Experimental::RRVecField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<Internal::RColumnIndex>(desc);
}

void ROOT::Experimental::RRVecField::ConstructValue(void *where) const
{
   // initialize data members fBegin, fSize, fCapacity
   // currently the inline buffer is left uninitialized
   void **beginPtr = new (where)(void *)(nullptr);
   std::int32_t *sizePtr = new (reinterpret_cast<void *>(beginPtr + 1)) std::int32_t(0);
   new (sizePtr + 1) std::int32_t(-1);
}

void ROOT::Experimental::RRVecField::RRVecDeleter::operator()(void *objPtr, bool dtorOnly)
{
   auto [beginPtr, sizePtr, capacityPtr] = GetRVecDataMembers(objPtr);

   char *begin = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics
   if (fItemDeleter) {
      for (std::int32_t i = 0; i < *sizePtr; ++i) {
         fItemDeleter->operator()(begin + i * fItemSize, true /* dtorOnly */);
      }
   }

   DestroyRVecWithChecks(fItemAlignment, beginPtr, begin, capacityPtr);
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::RRVecField::GetDeleter() const
{
   if (fItemDeleter)
      return std::make_unique<RRVecDeleter>(fSubFields[0]->GetAlignment(), fItemSize, GetDeleterOf(*fSubFields[0]));
   return std::make_unique<RRVecDeleter>(fSubFields[0]->GetAlignment());
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RRVecField::SplitValue(const RValue &value) const
{
   auto [beginPtr, sizePtr, _] = GetRVecDataMembers(value.GetPtr<void>().get());

   std::vector<RValue> result;
   char *begin = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics
   result.reserve(*sizePtr);
   for (std::int32_t i = 0; i < *sizePtr; ++i) {
      result.emplace_back(fSubFields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), begin + i * fItemSize)));
   }
   return result;
}

size_t ROOT::Experimental::RRVecField::GetValueSize() const
{
   return fValueSize;
}

size_t ROOT::Experimental::RRVecField::GetAlignment() const
{
   return EvalRVecAlignment(fSubFields[0]->GetAlignment());
}

void ROOT::Experimental::RRVecField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitRVecField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RVectorField::RVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField,
                                               bool isUntyped)
   : ROOT::Experimental::RFieldBase(fieldName, isUntyped ? "" : "std::vector<" + itemField->GetTypeName() + ">",
                                    ROOT::ENTupleStructure::kCollection, false /* isSimple */),
     fItemSize(itemField->GetValueSize()),
     fNWritten(0)
{
   if (!(itemField->GetTraits() & kTraitTriviallyDestructible))
      fItemDeleter = GetDeleterOf(*itemField);
   Attach(std::move(itemField));
}

ROOT::Experimental::RVectorField::RVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
   : RVectorField(fieldName, std::move(itemField), false)
{
}

std::unique_ptr<ROOT::Experimental::RVectorField>
ROOT::Experimental::RVectorField::CreateUntyped(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
{
   return std::unique_ptr<ROOT::Experimental::RVectorField>(new RVectorField(fieldName, std::move(itemField), true));
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RVectorField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::unique_ptr<ROOT::Experimental::RVectorField>(
      new RVectorField(newName, std::move(newItemField), GetTypeName().empty()));
}

std::size_t ROOT::Experimental::RVectorField::AppendImpl(const void *from)
{
   auto typedValue = static_cast<const std::vector<char> *>(from);
   // The order is important here: Profiling showed that the integer division is on the critical path. By moving the
   // computation of count before R__ASSERT, the compiler can use the result of a single instruction (on x86) also for
   // the modulo operation. Otherwise, it must perform the division twice because R__ASSERT expands to an external call
   // of Fatal() in case of failure, which could have side effects that the compiler cannot analyze.
   auto count = typedValue->size() / fItemSize;
   R__ASSERT((typedValue->size() % fItemSize) == 0);
   std::size_t nbytes = 0;

   if (fSubFields[0]->IsSimple() && count) {
      GetPrincipalColumnOf(*fSubFields[0])->AppendV(typedValue->data(), count);
      nbytes += count * GetPrincipalColumnOf(*fSubFields[0])->GetElement()->GetPackedSize();
   } else {
      for (unsigned i = 0; i < count; ++i) {
         nbytes += CallAppendOn(*fSubFields[0], typedValue->data() + (i * fItemSize));
      }
   }

   fNWritten += count;
   fPrincipalColumn->Append(&fNWritten);
   return nbytes + fPrincipalColumn->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RVectorField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   auto typedValue = static_cast<std::vector<char> *>(to);

   ROOT::NTupleSize_t nItems;
   RNTupleLocalIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   if (fSubFields[0]->IsSimple()) {
      typedValue->resize(nItems * fItemSize);
      if (nItems)
         GetPrincipalColumnOf(*fSubFields[0])->ReadV(collectionStart, nItems, typedValue->data());
      return;
   }

   // See "semantics of reading non-trivial objects" in RNTuple's Architecture.md
   R__ASSERT(fItemSize > 0);
   const auto oldNItems = typedValue->size() / fItemSize;
   const bool canRealloc = oldNItems < nItems;
   bool allDeallocated = false;
   if (fItemDeleter) {
      allDeallocated = canRealloc;
      for (std::size_t i = allDeallocated ? 0 : nItems; i < oldNItems; ++i) {
         fItemDeleter->operator()(typedValue->data() + (i * fItemSize), true /* dtorOnly */);
      }
   }
   typedValue->resize(nItems * fItemSize);
   if (!(fSubFields[0]->GetTraits() & kTraitTriviallyConstructible)) {
      for (std::size_t i = allDeallocated ? 0 : oldNItems; i < nItems; ++i) {
         CallConstructValueOn(*fSubFields[0], typedValue->data() + (i * fItemSize));
      }
   }

   for (std::size_t i = 0; i < nItems; ++i) {
      CallReadOn(*fSubFields[0], collectionStart + i, typedValue->data() + (i * fItemSize));
   }
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RVectorField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RVectorField::GenerateColumns()
{
   GenerateColumnsImpl<Internal::RColumnIndex>();
}

void ROOT::Experimental::RVectorField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<Internal::RColumnIndex>(desc);
}

void ROOT::Experimental::RVectorField::RVectorDeleter::operator()(void *objPtr, bool dtorOnly)
{
   auto vecPtr = static_cast<std::vector<char> *>(objPtr);
   if (fItemDeleter) {
      R__ASSERT(fItemSize > 0);
      R__ASSERT((vecPtr->size() % fItemSize) == 0);
      auto nItems = vecPtr->size() / fItemSize;
      for (std::size_t i = 0; i < nItems; ++i) {
         fItemDeleter->operator()(vecPtr->data() + (i * fItemSize), true /* dtorOnly */);
      }
   }
   std::destroy_at(vecPtr);
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::RVectorField::GetDeleter() const
{
   if (fItemDeleter)
      return std::make_unique<RVectorDeleter>(fItemSize, GetDeleterOf(*fSubFields[0]));
   return std::make_unique<RVectorDeleter>();
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RVectorField::SplitValue(const RValue &value) const
{
   auto vec = value.GetPtr<std::vector<char>>();
   R__ASSERT(fItemSize > 0);
   R__ASSERT((vec->size() % fItemSize) == 0);
   auto nItems = vec->size() / fItemSize;
   std::vector<RValue> result;
   result.reserve(nItems);
   for (unsigned i = 0; i < nItems; ++i) {
      result.emplace_back(
         fSubFields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), vec->data() + (i * fItemSize))));
   }
   return result;
}

void ROOT::Experimental::RVectorField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitVectorField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RField<std::vector<bool>>::RField(std::string_view name)
   : ROOT::Experimental::RFieldBase(name, "std::vector<bool>", ROOT::ENTupleStructure::kCollection,
                                    false /* isSimple */)
{
   Attach(std::make_unique<RField<bool>>("_0"));
}

std::size_t ROOT::Experimental::RField<std::vector<bool>>::AppendImpl(const void *from)
{
   auto typedValue = static_cast<const std::vector<bool> *>(from);
   auto count = typedValue->size();
   for (unsigned i = 0; i < count; ++i) {
      bool bval = (*typedValue)[i];
      CallAppendOn(*fSubFields[0], &bval);
   }
   fNWritten += count;
   fPrincipalColumn->Append(&fNWritten);
   return count + fPrincipalColumn->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RField<std::vector<bool>>::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   auto typedValue = static_cast<std::vector<bool> *>(to);

   ROOT::NTupleSize_t nItems;
   RNTupleLocalIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   typedValue->resize(nItems);
   for (unsigned i = 0; i < nItems; ++i) {
      bool bval;
      CallReadOn(*fSubFields[0], collectionStart + i, &bval);
      (*typedValue)[i] = bval;
   }
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::vector<bool>>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumns()
{
   GenerateColumnsImpl<Internal::RColumnIndex>();
}

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<Internal::RColumnIndex>(desc);
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RField<std::vector<bool>>::SplitValue(const RValue &value) const
{
   const auto &typedValue = value.GetRef<std::vector<bool>>();
   auto count = typedValue.size();
   std::vector<RValue> result;
   result.reserve(count);
   for (unsigned i = 0; i < count; ++i) {
      if (typedValue[i])
         result.emplace_back(fSubFields[0]->BindValue(std::shared_ptr<bool>(new bool(true))));
      else
         result.emplace_back(fSubFields[0]->BindValue(std::shared_ptr<bool>(new bool(false))));
   }
   return result;
}

void ROOT::Experimental::RField<std::vector<bool>>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitVectorBoolField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RArrayAsRVecField::RArrayAsRVecField(std::string_view fieldName,
                                                         std::unique_ptr<ROOT::Experimental::RFieldBase> itemField,
                                                         std::size_t arrayLength)
   : ROOT::Experimental::RFieldBase(fieldName, "ROOT::VecOps::RVec<" + itemField->GetTypeName() + ">",
                                    ROOT::ENTupleStructure::kCollection, false /* isSimple */),
     fItemSize(itemField->GetValueSize()),
     fArrayLength(arrayLength)
{
   Attach(std::move(itemField));
   fValueSize = EvalRVecValueSize(fSubFields[0]->GetAlignment(), fSubFields[0]->GetValueSize(), GetAlignment());
   if (!(fSubFields[0]->GetTraits() & kTraitTriviallyDestructible))
      fItemDeleter = GetDeleterOf(*fSubFields[0]);
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RArrayAsRVecField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::make_unique<RArrayAsRVecField>(newName, std::move(newItemField), fArrayLength);
}

void ROOT::Experimental::RArrayAsRVecField::ConstructValue(void *where) const
{
   // initialize data members fBegin, fSize, fCapacity
   void **beginPtr = new (where)(void *)(nullptr);
   std::int32_t *sizePtr = new (reinterpret_cast<void *>(beginPtr + 1)) std::int32_t(0);
   std::int32_t *capacityPtr = new (sizePtr + 1) std::int32_t(0);

   // Create the RVec with the known fixed size, do it once here instead of
   // every time the value is read in `Read*Impl` functions
   char *begin = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics

   // Early return if the RVec has already been allocated.
   if (*sizePtr == std::int32_t(fArrayLength))
      return;

   // Need to allocate the RVec if it is the first time the value is being created.
   // See "semantics of reading non-trivial objects" in RNTuple's Architecture.md for details
   // on the element construction.
   const bool owns = (*capacityPtr != -1); // RVec is adopting the memory
   const bool needsConstruct = !(fSubFields[0]->GetTraits() & kTraitTriviallyConstructible);
   const bool needsDestruct = owns && fItemDeleter;

   // Destroy old elements: useless work for trivial types, but in case the element type's constructor
   // allocates memory we need to release it here to avoid memleaks (e.g. if this is an RVec<RVec<int>>)
   if (needsDestruct) {
      for (std::int32_t i = 0; i < *sizePtr; ++i) {
         fItemDeleter->operator()(begin + (i * fItemSize), true /* dtorOnly */);
      }
   }

   // TODO: Isn't the RVec always owning in this case?
   if (owns) {
      // *beginPtr points to the array of item values (allocated in an earlier call by the following malloc())
      free(*beginPtr);
   }

   *beginPtr = malloc(fArrayLength * fItemSize);
   R__ASSERT(*beginPtr != nullptr);
   // Re-assign begin pointer after allocation
   begin = reinterpret_cast<char *>(*beginPtr);
   // Size and capacity are equal since the field data type is std::array
   *sizePtr = fArrayLength;
   *capacityPtr = fArrayLength;

   // Placement new for the array elements
   if (needsConstruct) {
      for (std::size_t i = 0; i < fArrayLength; ++i)
         CallConstructValueOn(*fSubFields[0], begin + (i * fItemSize));
   }
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::RArrayAsRVecField::GetDeleter() const
{
   if (fItemDeleter) {
      return std::make_unique<RRVecField::RRVecDeleter>(fSubFields[0]->GetAlignment(), fItemSize,
                                                        GetDeleterOf(*fSubFields[0]));
   }
   return std::make_unique<RRVecField::RRVecDeleter>(fSubFields[0]->GetAlignment());
}

void ROOT::Experimental::RArrayAsRVecField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{

   auto [beginPtr, _, __] = GetRVecDataMembers(to);
   auto rvecBeginPtr = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics

   if (fSubFields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubFields[0])->ReadV(globalIndex * fArrayLength, fArrayLength, rvecBeginPtr);
      return;
   }

   // Read the new values into the collection elements
   for (std::size_t i = 0; i < fArrayLength; ++i) {
      CallReadOn(*fSubFields[0], globalIndex * fArrayLength + i, rvecBeginPtr + (i * fItemSize));
   }
}

void ROOT::Experimental::RArrayAsRVecField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   auto [beginPtr, _, __] = GetRVecDataMembers(to);
   auto rvecBeginPtr = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics

   const auto &clusterId = localIndex.GetClusterId();
   const auto &indexInCluster = localIndex.GetIndexInCluster();

   if (fSubFields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubFields[0])
         ->ReadV(RNTupleLocalIndex(clusterId, indexInCluster * fArrayLength), fArrayLength, rvecBeginPtr);
      return;
   }

   // Read the new values into the collection elements
   for (std::size_t i = 0; i < fArrayLength; ++i) {
      CallReadOn(*fSubFields[0], RNTupleLocalIndex(clusterId, indexInCluster * fArrayLength + i),
                 rvecBeginPtr + (i * fItemSize));
   }
}

size_t ROOT::Experimental::RArrayAsRVecField::GetAlignment() const
{
   return EvalRVecAlignment(fSubFields[0]->GetAlignment());
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RArrayAsRVecField::SplitValue(const ROOT::Experimental::RFieldBase::RValue &value) const
{
   auto arrayPtr = value.GetPtr<unsigned char>().get();
   std::vector<ROOT::Experimental::RFieldBase::RValue> result;
   result.reserve(fArrayLength);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      result.emplace_back(
         fSubFields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), arrayPtr + (i * fItemSize))));
   }
   return result;
}

void ROOT::Experimental::RArrayAsRVecField::AcceptVisitor(ROOT::Experimental::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitArrayAsRVecField(*this);
}
