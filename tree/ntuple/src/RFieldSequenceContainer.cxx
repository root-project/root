/// \file RFieldSequenceContainer.cxx
/// \ingroup NTuple
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19

#include <ROOT/RField.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RFieldUtils.hxx>

#include <cstdlib> // for malloc, free
#include <limits>
#include <memory>
#include <new> // hardware_destructive_interference_size

namespace {

/// Retrieve the addresses of the data members of a generic RVec from a pointer to the beginning of the RVec object.
/// Returns pointers to fBegin, fSize and fCapacity in a std::tuple.
std::tuple<unsigned char **, std::int32_t *, std::int32_t *> GetRVecDataMembers(void *rvecPtr)
{
   unsigned char **beginPtr = reinterpret_cast<unsigned char **>(rvecPtr);
   // int32_t fSize is the second data member (after 1 void*)
   std::int32_t *size = reinterpret_cast<std::int32_t *>(beginPtr + 1);
   R__ASSERT(*size >= 0);
   // int32_t fCapacity is the third data member (1 int32_t after fSize)
   std::int32_t *capacity = size + 1;
   R__ASSERT(*capacity >= -1);
   return {beginPtr, size, capacity};
}

std::tuple<const unsigned char *const *, const std::int32_t *, const std::int32_t *>
GetRVecDataMembers(const void *rvecPtr)
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
      constexpr unsigned cacheLineSize = R__HARDWARE_INTERFERENCE_SIZE;
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

std::size_t EvalRVecAlignment(std::size_t alignOfSubfield)
{
   // the alignment of an RVec<T> is the largest among the alignments of its data members
   // (including the inline buffer which has the same alignment as the RVec::value_type)
   return std::max({alignof(void *), alignof(std::int32_t), alignOfSubfield});
}

void DestroyRVecWithChecks(std::size_t alignOfT, unsigned char **beginPtr, std::int32_t *capacityPtr)
{
   // figure out if we are in the small state, i.e. begin == &inlineBuffer
   // there might be padding between fCapacity and the inline buffer, so we compute it here
   constexpr auto dataMemberSz = sizeof(void *) + 2 * sizeof(std::int32_t);
   auto paddingMiddle = dataMemberSz % alignOfT;
   if (paddingMiddle != 0)
      paddingMiddle = alignOfT - paddingMiddle;
   const bool isSmall = (*beginPtr == (reinterpret_cast<unsigned char *>(beginPtr) + dataMemberSz + paddingMiddle));

   const bool owns = (*capacityPtr != -1);
   if (!isSmall && owns)
      free(*beginPtr);
}

std::vector<ROOT::RFieldBase::RValue> SplitVector(std::shared_ptr<void> valuePtr, ROOT::RFieldBase &itemField)
{
   auto *vec = static_cast<std::vector<char> *>(valuePtr.get());
   const auto itemSize = itemField.GetValueSize();
   R__ASSERT(itemSize > 0);
   R__ASSERT((vec->size() % itemSize) == 0);
   const auto nItems = vec->size() / itemSize;
   std::vector<ROOT::RFieldBase::RValue> result;
   result.reserve(nItems);
   for (unsigned i = 0; i < nItems; ++i) {
      result.emplace_back(itemField.BindValue(std::shared_ptr<void>(valuePtr, vec->data() + (i * itemSize))));
   }
   return result;
}

} // anonymous namespace

ROOT::RArrayField::RArrayField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField,
                               std::size_t arrayLength)
   : ROOT::RFieldBase(fieldName,
                      "std::array<" + itemField->GetTypeName() + "," +
                         Internal::GetNormalizedInteger(static_cast<unsigned long long>(arrayLength)) + ">",
                      ROOT::ENTupleStructure::kPlain, false /* isSimple */, arrayLength),
     fItemSize(itemField->GetValueSize()),
     fArrayLength(arrayLength)
{
   fTraits |= itemField->GetTraits() & ~kTraitMappable;
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RArrayField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::make_unique<RArrayField>(newName, std::move(newItemField), fArrayLength);
}

std::size_t ROOT::RArrayField::AppendImpl(const void *from)
{
   std::size_t nbytes = 0;
   if (fSubfields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubfields[0])->AppendV(from, fArrayLength);
      nbytes += fArrayLength * GetPrincipalColumnOf(*fSubfields[0])->GetElement()->GetPackedSize();
   } else {
      auto arrayPtr = static_cast<const unsigned char *>(from);
      for (unsigned i = 0; i < fArrayLength; ++i) {
         nbytes += CallAppendOn(*fSubfields[0], arrayPtr + (i * fItemSize));
      }
   }
   return nbytes;
}

void ROOT::RArrayField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   if (fSubfields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubfields[0])->ReadV(globalIndex * fArrayLength, fArrayLength, to);
   } else {
      auto arrayPtr = static_cast<unsigned char *>(to);
      for (unsigned i = 0; i < fArrayLength; ++i) {
         CallReadOn(*fSubfields[0], globalIndex * fArrayLength + i, arrayPtr + (i * fItemSize));
      }
   }
}

void ROOT::RArrayField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   if (fSubfields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubfields[0])->ReadV(localIndex * fArrayLength, fArrayLength, to);
   } else {
      auto arrayPtr = static_cast<unsigned char *>(to);
      for (unsigned i = 0; i < fArrayLength; ++i) {
         CallReadOn(*fSubfields[0], localIndex * fArrayLength + i, arrayPtr + (i * fItemSize));
      }
   }
}

std::size_t ROOT::RArrayField::ReadBulkImpl(const RBulkSpec &bulkSpec)
{
   if (!fSubfields[0]->IsSimple())
      return RFieldBase::ReadBulkImpl(bulkSpec);

   GetPrincipalColumnOf(*fSubfields[0])
      ->ReadV(bulkSpec.fFirstIndex * fArrayLength, bulkSpec.fCount * fArrayLength, bulkSpec.fValues);
   return RBulkSpec::kAllSet;
}

void ROOT::RArrayField::ReconcileOnDiskField(const RNTupleDescriptor &desc)
{
   static const std::vector<std::string> prefixes = {"std::array<"};

   EnsureMatchingOnDiskField(desc, kDiffTypeName).ThrowOnError();
   EnsureMatchingTypePrefix(desc, prefixes).ThrowOnError();
}

void ROOT::RArrayField::ConstructValue(void *where) const
{
   if (fSubfields[0]->GetTraits() & kTraitTriviallyConstructible)
      return;

   auto arrayPtr = reinterpret_cast<unsigned char *>(where);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      CallConstructValueOn(*fSubfields[0], arrayPtr + (i * fItemSize));
   }
}

void ROOT::RArrayField::RArrayDeleter::operator()(void *objPtr, bool dtorOnly)
{
   if (fItemDeleter) {
      for (unsigned i = 0; i < fArrayLength; ++i) {
         fItemDeleter->operator()(reinterpret_cast<unsigned char *>(objPtr) + i * fItemSize, true /* dtorOnly */);
      }
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::RArrayField::GetDeleter() const
{
   if (!(fSubfields[0]->GetTraits() & kTraitTriviallyDestructible))
      return std::make_unique<RArrayDeleter>(fItemSize, fArrayLength, GetDeleterOf(*fSubfields[0]));
   return std::make_unique<RDeleter>();
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RArrayField::SplitValue(const RValue &value) const
{
   auto valuePtr = value.GetPtr<void>();
   auto arrayPtr = static_cast<unsigned char *>(valuePtr.get());
   std::vector<RValue> result;
   result.reserve(fArrayLength);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      result.emplace_back(fSubfields[0]->BindValue(std::shared_ptr<void>(valuePtr, arrayPtr + (i * fItemSize))));
   }
   return result;
}

void ROOT::RArrayField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitArrayField(*this);
}

//------------------------------------------------------------------------------

ROOT::RRVecField::RRVecField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
   : ROOT::RFieldBase(fieldName, "ROOT::VecOps::RVec<" + itemField->GetTypeName() + ">",
                      ROOT::ENTupleStructure::kCollection, false /* isSimple */),
     fItemSize(itemField->GetValueSize()),
     fNWritten(0)
{
   if (!(itemField->GetTraits() & kTraitTriviallyDestructible))
      fItemDeleter = GetDeleterOf(*itemField);
   Attach(std::move(itemField));
   fValueSize = EvalRVecValueSize(fSubfields[0]->GetAlignment(), fSubfields[0]->GetValueSize(), GetAlignment());

   // Determine if we can optimimize bulk reading
   if (fSubfields[0]->IsSimple()) {
      fBulkSubfield = fSubfields[0].get();
   } else {
      if (auto f = dynamic_cast<RArrayField *>(fSubfields[0].get())) {
         auto grandChildFields = fSubfields[0]->GetMutableSubfields();
         if (grandChildFields[0]->IsSimple()) {
            fBulkSubfield = grandChildFields[0];
            fBulkNRepetition = f->GetLength();
         }
      }
   }
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RRVecField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::make_unique<RRVecField>(newName, std::move(newItemField));
}

std::size_t ROOT::RRVecField::AppendImpl(const void *from)
{
   auto [beginPtr, sizePtr, _] = GetRVecDataMembers(from);

   std::size_t nbytes = 0;
   if (fSubfields[0]->IsSimple() && *sizePtr) {
      GetPrincipalColumnOf(*fSubfields[0])->AppendV(*beginPtr, *sizePtr);
      nbytes += *sizePtr * GetPrincipalColumnOf(*fSubfields[0])->GetElement()->GetPackedSize();
   } else {
      for (std::int32_t i = 0; i < *sizePtr; ++i) {
         nbytes += CallAppendOn(*fSubfields[0], *beginPtr + i * fItemSize);
      }
   }

   fNWritten += *sizePtr;
   fPrincipalColumn->Append(&fNWritten);
   return nbytes + fPrincipalColumn->GetElement()->GetPackedSize();
}

unsigned char *ROOT::RRVecField::ResizeRVec(void *rvec, std::size_t nItems, std::size_t itemSize,
                                            const RFieldBase *itemField, RDeleter *itemDeleter)

{
   if (nItems > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      throw RException(R__FAIL("RVec too large: " + std::to_string(nItems)));
   }

   auto [beginPtr, sizePtr, capacityPtr] = GetRVecDataMembers(rvec);
   const std::size_t oldSize = *sizePtr;

   // See "semantics of reading non-trivial objects" in RNTuple's Architecture.md for details
   // on the element construction/destrution.
   const bool owns = (*capacityPtr != -1);
   const bool needsConstruct = !(itemField->GetTraits() & kTraitTriviallyConstructible);
   const bool needsDestruct = owns && itemDeleter;

   // Destroy excess elements, if any
   if (needsDestruct) {
      for (std::size_t i = nItems; i < oldSize; ++i) {
         itemDeleter->operator()(*beginPtr + (i * itemSize), true /* dtorOnly */);
      }
   }

   // Resize RVec (capacity and size)
   if (std::int32_t(nItems) > *capacityPtr) { // must reallocate
      // Destroy old elements: useless work for trivial types, but in case the element type's constructor
      // allocates memory we need to release it here to avoid memleaks (e.g. if this is an RVec<RVec<int>>)
      if (needsDestruct) {
         for (std::size_t i = 0u; i < oldSize; ++i) {
            itemDeleter->operator()(*beginPtr + (i * itemSize), true /* dtorOnly */);
         }
      }

      // TODO Increment capacity by a factor rather than just enough to fit the elements.
      if (owns) {
         // *beginPtr points to the array of item values (allocated in an earlier call by the following malloc())
         free(*beginPtr);
      }
      // We trust that malloc returns a buffer with large enough alignment.
      // This might not be the case if T in RVec<T> is over-aligned.
      *beginPtr = static_cast<unsigned char *>(malloc(nItems * itemSize));
      R__ASSERT(*beginPtr != nullptr);
      *capacityPtr = nItems;

      // Placement new for elements that were already there before the resize
      if (needsConstruct) {
         for (std::size_t i = 0u; i < oldSize; ++i)
            CallConstructValueOn(*itemField, *beginPtr + (i * itemSize));
      }
   }
   *sizePtr = nItems;

   // Placement new for new elements, if any
   if (needsConstruct) {
      for (std::size_t i = oldSize; i < nItems; ++i)
         CallConstructValueOn(*itemField, *beginPtr + (i * itemSize));
   }

   return *beginPtr;
}

void ROOT::RRVecField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   // TODO as a performance optimization, we could assign values to elements of the inline buffer:
   // if size < inline buffer size: we save one allocation here and usage of the RVec skips a pointer indirection

   // Read collection info for this entry
   ROOT::NTupleSize_t nItems;
   RNTupleLocalIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   auto begin = ResizeRVec(to, nItems, fItemSize, fSubfields[0].get(), fItemDeleter.get());

   if (fSubfields[0]->IsSimple() && nItems) {
      GetPrincipalColumnOf(*fSubfields[0])->ReadV(collectionStart, nItems, begin);
      return;
   }

   // Read the new values into the collection elements
   for (std::size_t i = 0; i < nItems; ++i) {
      CallReadOn(*fSubfields[0], collectionStart + i, begin + (i * fItemSize));
   }
}

std::size_t ROOT::RRVecField::ReadBulkImpl(const RBulkSpec &bulkSpec)
{
   if (!fBulkSubfield)
      return RFieldBase::ReadBulkImpl(bulkSpec);

   if (bulkSpec.fAuxData->empty()) {
      /// Initialize auxiliary memory: the first sizeof(size_t) bytes store the value size of the item field.
      /// The following bytes store the item values, consecutively.
      bulkSpec.fAuxData->resize(sizeof(std::size_t));
      *reinterpret_cast<std::size_t *>(bulkSpec.fAuxData->data()) = fBulkNRepetition * fBulkSubfield->GetValueSize();
   }
   const auto itemValueSize = *reinterpret_cast<std::size_t *>(bulkSpec.fAuxData->data());
   unsigned char *itemValueArray = bulkSpec.fAuxData->data() + sizeof(std::size_t);
   auto [beginPtr, sizePtr, capacityPtr] = GetRVecDataMembers(bulkSpec.fValues);

   // Get size of the first RVec of the bulk
   RNTupleLocalIndex firstItemIndex;
   ROOT::NTupleSize_t collectionSize;
   fPrincipalColumn->GetCollectionInfo(bulkSpec.fFirstIndex, &firstItemIndex, &collectionSize);
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
         fPrincipalColumn->MapV<ROOT::Internal::RColumnIndex>(bulkSpec.fFirstIndex + nValues, nElementsUntilPageEnd);
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

   GetPrincipalColumnOf(*fBulkSubfield)
      ->ReadV(firstItemIndex * fBulkNRepetition, nItems * fBulkNRepetition, itemValueArray - delta);
   return RBulkSpec::kAllSet;
}

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RRVecField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::RRVecField::GenerateColumns()
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>();
}

void ROOT::RRVecField::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(desc);
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RRVecField::BeforeConnectPageSource(Internal::RPageSource &pageSource)
{
   if (GetOnDiskId() == kInvalidDescriptorId)
      return nullptr;

   const auto descGuard = pageSource.GetSharedDescriptorGuard();
   const auto &fieldDesc = descGuard->GetFieldDescriptor(GetOnDiskId());
   if (fieldDesc.GetTypeName().rfind("std::array<", 0) == 0) {
      auto substitute = std::make_unique<RArrayAsRVecField>(
         GetFieldName(), fSubfields[0]->Clone(fSubfields[0]->GetFieldName()), fieldDesc.GetNRepetitions());
      substitute->SetOnDiskId(GetOnDiskId());
      return substitute;
   }
   return nullptr;
}

void ROOT::RRVecField::ReconcileOnDiskField(const RNTupleDescriptor &desc)
{
   EnsureMatchingOnDiskField(desc, kDiffTypeName).ThrowOnError();
}

void ROOT::RRVecField::ConstructValue(void *where) const
{
   // initialize data members fBegin, fSize, fCapacity
   // currently the inline buffer is left uninitialized
   void **beginPtr = new (where)(void *)(nullptr);
   std::int32_t *sizePtr = new (reinterpret_cast<void *>(beginPtr + 1)) std::int32_t(0);
   new (sizePtr + 1) std::int32_t(-1);
}

void ROOT::RRVecField::RRVecDeleter::operator()(void *objPtr, bool dtorOnly)
{
   auto [beginPtr, sizePtr, capacityPtr] = GetRVecDataMembers(objPtr);

   if (fItemDeleter) {
      for (std::int32_t i = 0; i < *sizePtr; ++i) {
         fItemDeleter->operator()(*beginPtr + i * fItemSize, true /* dtorOnly */);
      }
   }

   DestroyRVecWithChecks(fItemAlignment, beginPtr, capacityPtr);
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::RRVecField::GetDeleter() const
{
   if (fItemDeleter)
      return std::make_unique<RRVecDeleter>(fSubfields[0]->GetAlignment(), fItemSize, GetDeleterOf(*fSubfields[0]));
   return std::make_unique<RRVecDeleter>(fSubfields[0]->GetAlignment());
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RRVecField::SplitValue(const RValue &value) const
{
   auto [beginPtr, sizePtr, _] = GetRVecDataMembers(value.GetPtr<void>().get());

   std::vector<RValue> result;
   result.reserve(*sizePtr);
   for (std::int32_t i = 0; i < *sizePtr; ++i) {
      result.emplace_back(
         fSubfields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), *beginPtr + i * fItemSize)));
   }
   return result;
}

size_t ROOT::RRVecField::GetValueSize() const
{
   return fValueSize;
}

size_t ROOT::RRVecField::GetAlignment() const
{
   return EvalRVecAlignment(fSubfields[0]->GetAlignment());
}

void ROOT::RRVecField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitRVecField(*this);
}

//------------------------------------------------------------------------------

ROOT::RVectorField::RVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField,
                                 std::optional<std::string_view> emulatedFromType)
   : ROOT::RFieldBase(fieldName, emulatedFromType ? *emulatedFromType : "std::vector<" + itemField->GetTypeName() + ">",
                      ROOT::ENTupleStructure::kCollection, false /* isSimple */),
     fItemSize(itemField->GetValueSize()),
     fNWritten(0)
{
   if (emulatedFromType && !emulatedFromType->empty())
      fTraits |= kTraitEmulatedField;

   if (!(itemField->GetTraits() & kTraitTriviallyDestructible))
      fItemDeleter = GetDeleterOf(*itemField);
   Attach(std::move(itemField));
}

ROOT::RVectorField::RVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
   : RVectorField(fieldName, std::move(itemField), {})
{
}

std::unique_ptr<ROOT::RVectorField>
ROOT::RVectorField::CreateUntyped(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
{
   return std::unique_ptr<ROOT::RVectorField>(new RVectorField(fieldName, itemField->Clone("_0"), ""));
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RVectorField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   auto isUntyped = GetTypeName().empty() || ((fTraits & kTraitEmulatedField) != 0);
   auto emulatedFromType = isUntyped ? std::make_optional(GetTypeName()) : std::nullopt;
   return std::unique_ptr<ROOT::RVectorField>(new RVectorField(newName, std::move(newItemField), emulatedFromType));
}

std::size_t ROOT::RVectorField::AppendImpl(const void *from)
{
   auto typedValue = static_cast<const std::vector<char> *>(from);
   // The order is important here: Profiling showed that the integer division is on the critical path. By moving the
   // computation of count before R__ASSERT, the compiler can use the result of a single instruction (on x86) also for
   // the modulo operation. Otherwise, it must perform the division twice because R__ASSERT expands to an external call
   // of Fatal() in case of failure, which could have side effects that the compiler cannot analyze.
   auto count = typedValue->size() / fItemSize;
   R__ASSERT((typedValue->size() % fItemSize) == 0);
   std::size_t nbytes = 0;

   if (fSubfields[0]->IsSimple() && count) {
      GetPrincipalColumnOf(*fSubfields[0])->AppendV(typedValue->data(), count);
      nbytes += count * GetPrincipalColumnOf(*fSubfields[0])->GetElement()->GetPackedSize();
   } else {
      for (unsigned i = 0; i < count; ++i) {
         nbytes += CallAppendOn(*fSubfields[0], typedValue->data() + (i * fItemSize));
      }
   }

   fNWritten += count;
   fPrincipalColumn->Append(&fNWritten);
   return nbytes + fPrincipalColumn->GetElement()->GetPackedSize();
}

void ROOT::RVectorField::ResizeVector(void *vec, std::size_t nItems, std::size_t itemSize, const RFieldBase &itemField,
                                      RDeleter *itemDeleter)
{
   auto typedValue = static_cast<std::vector<char> *>(vec);

   // See "semantics of reading non-trivial objects" in RNTuple's Architecture.md
   R__ASSERT(itemSize > 0);
   const auto oldNItems = typedValue->size() / itemSize;
   const bool canRealloc = oldNItems < nItems;
   bool allDeallocated = false;
   if (itemDeleter) {
      allDeallocated = canRealloc;
      for (std::size_t i = allDeallocated ? 0 : nItems; i < oldNItems; ++i) {
         itemDeleter->operator()(typedValue->data() + (i * itemSize), true /* dtorOnly */);
      }
   }
   typedValue->resize(nItems * itemSize);
   if (!(itemField.GetTraits() & kTraitTriviallyConstructible)) {
      for (std::size_t i = allDeallocated ? 0 : oldNItems; i < nItems; ++i) {
         CallConstructValueOn(itemField, typedValue->data() + (i * itemSize));
      }
   }
}

void ROOT::RVectorField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   auto typedValue = static_cast<std::vector<char> *>(to);

   ROOT::NTupleSize_t nItems;
   RNTupleLocalIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   if (fSubfields[0]->IsSimple()) {
      typedValue->resize(nItems * fItemSize);
      if (nItems)
         GetPrincipalColumnOf(*fSubfields[0])->ReadV(collectionStart, nItems, typedValue->data());
      return;
   }

   ResizeVector(to, nItems, fItemSize, *fSubfields[0], fItemDeleter.get());

   for (std::size_t i = 0; i < nItems; ++i) {
      CallReadOn(*fSubfields[0], collectionStart + i, typedValue->data() + (i * fItemSize));
   }
}

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RVectorField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::RVectorField::GenerateColumns()
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>();
}

void ROOT::RVectorField::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(desc);
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RVectorField::BeforeConnectPageSource(Internal::RPageSource &pageSource)
{
   if (GetOnDiskId() == kInvalidDescriptorId)
      return nullptr;

   const auto descGuard = pageSource.GetSharedDescriptorGuard();
   const auto &fieldDesc = descGuard->GetFieldDescriptor(GetOnDiskId());
   if (fieldDesc.GetTypeName().rfind("std::array<", 0) == 0) {
      auto substitute = std::make_unique<RArrayAsVectorField>(
         GetFieldName(), fSubfields[0]->Clone(fSubfields[0]->GetFieldName()), fieldDesc.GetNRepetitions());
      substitute->SetOnDiskId(GetOnDiskId());
      return substitute;
   }
   return nullptr;
}

void ROOT::RVectorField::ReconcileOnDiskField(const RNTupleDescriptor &desc)
{
   EnsureMatchingOnDiskField(desc, kDiffTypeName).ThrowOnError();
}

void ROOT::RVectorField::RVectorDeleter::operator()(void *objPtr, bool dtorOnly)
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

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::RVectorField::GetDeleter() const
{
   if (fItemDeleter)
      return std::make_unique<RVectorDeleter>(fItemSize, GetDeleterOf(*fSubfields[0]));
   return std::make_unique<RVectorDeleter>();
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RVectorField::SplitValue(const RValue &value) const
{
   return SplitVector(value.GetPtr<void>(), *fSubfields[0]);
}

void ROOT::RVectorField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitVectorField(*this);
}

//------------------------------------------------------------------------------

ROOT::RField<std::vector<bool>>::RField(std::string_view name)
   : ROOT::RFieldBase(name, "std::vector<bool>", ROOT::ENTupleStructure::kCollection, false /* isSimple */)
{
   Attach(std::make_unique<RField<bool>>("_0"));
}

std::size_t ROOT::RField<std::vector<bool>>::AppendImpl(const void *from)
{
   auto typedValue = static_cast<const std::vector<bool> *>(from);
   auto count = typedValue->size();
   for (unsigned i = 0; i < count; ++i) {
      bool bval = (*typedValue)[i];
      CallAppendOn(*fSubfields[0], &bval);
   }
   fNWritten += count;
   fPrincipalColumn->Append(&fNWritten);
   return count + fPrincipalColumn->GetElement()->GetPackedSize();
}

void ROOT::RField<std::vector<bool>>::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   auto typedValue = static_cast<std::vector<bool> *>(to);

   if (fOnDiskNRepetitions == 0) {
      ROOT::NTupleSize_t nItems;
      RNTupleLocalIndex collectionStart;
      fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);
      typedValue->resize(nItems);
      for (std::size_t i = 0; i < nItems; ++i) {
         bool bval;
         CallReadOn(*fSubfields[0], collectionStart + i, &bval);
         (*typedValue)[i] = bval;
      }
   } else {
      typedValue->resize(fOnDiskNRepetitions);
      for (std::size_t i = 0; i < fOnDiskNRepetitions; ++i) {
         bool bval;
         CallReadOn(*fSubfields[0], globalIndex * fOnDiskNRepetitions + i, &bval);
         (*typedValue)[i] = bval;
      }
   }
}

void ROOT::RField<std::vector<bool>>::ReadInClusterImpl(ROOT::RNTupleLocalIndex localIndex, void *to)
{
   auto typedValue = static_cast<std::vector<bool> *>(to);

   if (fOnDiskNRepetitions == 0) {
      ROOT::NTupleSize_t nItems;
      RNTupleLocalIndex collectionStart;
      fPrincipalColumn->GetCollectionInfo(localIndex, &collectionStart, &nItems);
      typedValue->resize(nItems);
      for (std::size_t i = 0; i < nItems; ++i) {
         bool bval;
         CallReadOn(*fSubfields[0], collectionStart + i, &bval);
         (*typedValue)[i] = bval;
      }
   } else {
      typedValue->resize(fOnDiskNRepetitions);
      for (std::size_t i = 0; i < fOnDiskNRepetitions; ++i) {
         bool bval;
         CallReadOn(*fSubfields[0], localIndex * fOnDiskNRepetitions + i, &bval);
         (*typedValue)[i] = bval;
      }
   }
}

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RField<std::vector<bool>>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {{}});
   return representations;
}

void ROOT::RField<std::vector<bool>>::GenerateColumns()
{
   R__ASSERT(fOnDiskNRepetitions == 0); // fOnDiskNRepetitions must only be used for reading
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>();
}

void ROOT::RField<std::vector<bool>>::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   if (fOnDiskNRepetitions == 0)
      GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(desc);
}

void ROOT::RField<std::vector<bool>>::ReconcileOnDiskField(const RNTupleDescriptor &desc)
{
   const auto &fieldDesc = desc.GetFieldDescriptor(GetOnDiskId());

   if (fieldDesc.GetTypeName().rfind("std::array<", 0) == 0) {
      EnsureMatchingOnDiskField(desc, kDiffTypeName | kDiffStructure | kDiffNRepetitions).ThrowOnError();

      if (fieldDesc.GetNRepetitions() == 0) {
         throw RException(R__FAIL("fixed-size array --> std::vector<bool>: expected repetition count > 0\n" +
                                  Internal::GetTypeTraceReport(*this, desc)));
      }
      if (fieldDesc.GetStructure() != ENTupleStructure::kPlain) {
         throw RException(R__FAIL("fixed-size array --> std::vector<bool>: expected plain on-disk field\n" +
                                  Internal::GetTypeTraceReport(*this, desc)));
      }
      fOnDiskNRepetitions = fieldDesc.GetNRepetitions();
   } else {
      EnsureMatchingOnDiskField(desc, kDiffTypeName).ThrowOnError();
   }
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RField<std::vector<bool>>::SplitValue(const RValue &value) const
{
   const auto &typedValue = value.GetRef<std::vector<bool>>();
   auto count = typedValue.size();
   std::vector<RValue> result;
   result.reserve(count);
   for (unsigned i = 0; i < count; ++i) {
      if (typedValue[i]) {
         result.emplace_back(fSubfields[0]->BindValue(std::shared_ptr<bool>(new bool(true))));
      } else {
         result.emplace_back(fSubfields[0]->BindValue(std::shared_ptr<bool>(new bool(false))));
      }
   }
   return result;
}

void ROOT::RField<std::vector<bool>>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitVectorBoolField(*this);
}

//------------------------------------------------------------------------------

ROOT::RArrayAsRVecField::RArrayAsRVecField(std::string_view fieldName, std::unique_ptr<ROOT::RFieldBase> itemField,
                                           std::size_t arrayLength)
   : ROOT::RFieldBase(fieldName, "ROOT::VecOps::RVec<" + itemField->GetTypeName() + ">",
                      ROOT::ENTupleStructure::kCollection, false /* isSimple */),
     fItemSize(itemField->GetValueSize()),
     fArrayLength(arrayLength)
{
   Attach(std::move(itemField));
   fValueSize = EvalRVecValueSize(fSubfields[0]->GetAlignment(), fSubfields[0]->GetValueSize(), GetAlignment());
   if (!(fSubfields[0]->GetTraits() & kTraitTriviallyDestructible))
      fItemDeleter = GetDeleterOf(*fSubfields[0]);
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RArrayAsRVecField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::make_unique<RArrayAsRVecField>(newName, std::move(newItemField), fArrayLength);
}

void ROOT::RArrayAsRVecField::ConstructValue(void *where) const
{
   // initialize data members fBegin, fSize, fCapacity
   // currently the inline buffer is left uninitialized
   void **beginPtr = new (where)(void *)(nullptr);
   std::int32_t *sizePtr = new (static_cast<void *>(beginPtr + 1)) std::int32_t(0);
   new (sizePtr + 1) std::int32_t(-1);
}

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::RArrayAsRVecField::GetDeleter() const
{
   if (fItemDeleter) {
      return std::make_unique<RRVecField::RRVecDeleter>(fSubfields[0]->GetAlignment(), fItemSize,
                                                        GetDeleterOf(*fSubfields[0]));
   }
   return std::make_unique<RRVecField::RRVecDeleter>(fSubfields[0]->GetAlignment());
}

void ROOT::RArrayAsRVecField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   auto begin = RRVecField::ResizeRVec(to, fArrayLength, fItemSize, fSubfields[0].get(), fItemDeleter.get());

   if (fSubfields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubfields[0])->ReadV(globalIndex * fArrayLength, fArrayLength, begin);
      return;
   }

   // Read the new values into the collection elements
   for (std::size_t i = 0; i < fArrayLength; ++i) {
      CallReadOn(*fSubfields[0], globalIndex * fArrayLength + i, begin + (i * fItemSize));
   }
}

void ROOT::RArrayAsRVecField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   auto begin = RRVecField::ResizeRVec(to, fArrayLength, fItemSize, fSubfields[0].get(), fItemDeleter.get());

   if (fSubfields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubfields[0])->ReadV(localIndex * fArrayLength, fArrayLength, begin);
      return;
   }

   // Read the new values into the collection elements
   for (std::size_t i = 0; i < fArrayLength; ++i) {
      CallReadOn(*fSubfields[0], localIndex * fArrayLength + i, begin + (i * fItemSize));
   }
}

void ROOT::RArrayAsRVecField::ReconcileOnDiskField(const RNTupleDescriptor &desc)
{
   EnsureMatchingOnDiskField(desc, kDiffTypeName | kDiffTypeVersion | kDiffStructure | kDiffNRepetitions)
      .ThrowOnError();
   const auto &fieldDesc = desc.GetFieldDescriptor(GetOnDiskId());
   if (fieldDesc.GetTypeName().rfind("std::array<", 0) != 0) {
      throw RException(R__FAIL("RArrayAsRVecField " + GetQualifiedFieldName() + " expects an on-disk array field\n" +
                               Internal::GetTypeTraceReport(*this, desc)));
   }
}

size_t ROOT::RArrayAsRVecField::GetAlignment() const
{
   return EvalRVecAlignment(fSubfields[0]->GetAlignment());
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RArrayAsRVecField::SplitValue(const ROOT::RFieldBase::RValue &value) const
{
   auto arrayPtr = value.GetPtr<unsigned char>().get();
   std::vector<ROOT::RFieldBase::RValue> result;
   result.reserve(fArrayLength);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      result.emplace_back(
         fSubfields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), arrayPtr + (i * fItemSize))));
   }
   return result;
}

void ROOT::RArrayAsRVecField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitArrayAsRVecField(*this);
}

//------------------------------------------------------------------------------

ROOT::RArrayAsVectorField::RArrayAsVectorField(std::string_view fieldName, std::unique_ptr<ROOT::RFieldBase> itemField,
                                               std::size_t arrayLength)
   : ROOT::RFieldBase(fieldName, "std::vector<" + itemField->GetTypeName() + ">", ROOT::ENTupleStructure::kCollection,
                      false /* isSimple */),
     fItemSize(itemField->GetValueSize()),
     fArrayLength(arrayLength)
{
   Attach(std::move(itemField));
   if (!(fSubfields[0]->GetTraits() & kTraitTriviallyDestructible))
      fItemDeleter = GetDeleterOf(*fSubfields[0]);
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RArrayAsVectorField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::make_unique<RArrayAsVectorField>(newName, std::move(newItemField), fArrayLength);
}

void ROOT::RArrayAsVectorField::GenerateColumns()
{
   throw RException(R__FAIL("RArrayAsVectorField fields must only be used for reading"));
}

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::RArrayAsVectorField::GetDeleter() const
{
   if (fItemDeleter)
      return std::make_unique<RVectorField::RVectorDeleter>(fItemSize, GetDeleterOf(*fSubfields[0]));
   return std::make_unique<RVectorField::RVectorDeleter>();
}

void ROOT::RArrayAsVectorField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   auto typedValue = static_cast<std::vector<char> *>(to);

   if (fSubfields[0]->IsSimple()) {
      typedValue->resize(fArrayLength * fItemSize);
      GetPrincipalColumnOf(*fSubfields[0])->ReadV(globalIndex * fArrayLength, fArrayLength, typedValue->data());
      return;
   }

   RVectorField::ResizeVector(to, fArrayLength, fItemSize, *fSubfields[0], fItemDeleter.get());

   for (std::size_t i = 0; i < fArrayLength; ++i) {
      CallReadOn(*fSubfields[0], globalIndex * fArrayLength + i, typedValue->data() + (i * fItemSize));
   }
}

void ROOT::RArrayAsVectorField::ReadInClusterImpl(ROOT::RNTupleLocalIndex localIndex, void *to)
{
   auto typedValue = static_cast<std::vector<char> *>(to);

   if (fSubfields[0]->IsSimple()) {
      typedValue->resize(fArrayLength * fItemSize);
      GetPrincipalColumnOf(*fSubfields[0])->ReadV(localIndex * fArrayLength, fArrayLength, typedValue->data());
      return;
   }

   RVectorField::ResizeVector(to, fArrayLength, fItemSize, *fSubfields[0], fItemDeleter.get());

   for (std::size_t i = 0; i < fArrayLength; ++i) {
      CallReadOn(*fSubfields[0], localIndex * fArrayLength + i, typedValue->data() + (i * fItemSize));
   }
}

void ROOT::RArrayAsVectorField::ReconcileOnDiskField(const RNTupleDescriptor &desc)
{
   EnsureMatchingOnDiskField(desc, kDiffTypeName | kDiffTypeVersion | kDiffStructure | kDiffNRepetitions);

   const auto &fieldDesc = desc.GetFieldDescriptor(GetOnDiskId());
   if (fieldDesc.GetTypeName().rfind("std::array<", 0) != 0) {
      throw RException(R__FAIL("RArrayAsVectorField " + GetQualifiedFieldName() + " expects an on-disk array field\n" +
                               Internal::GetTypeTraceReport(*this, desc)));
   }
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RArrayAsVectorField::SplitValue(const ROOT::RFieldBase::RValue &value) const
{
   return SplitVector(value.GetPtr<void>(), *fSubfields[0]);
}

void ROOT::RArrayAsVectorField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitArrayAsVectorField(*this);
}
