/// \file RField.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RColumn.hxx>
#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <algorithm>
#include <cstdint>
#include <cstdlib> // for malloc, free
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <new> // hardware_destructive_interference_size
#include <type_traits>

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
   const bool isSmall = (reinterpret_cast<void *>(begin) == (beginPtr + dataMemberSz + paddingMiddle));

   const bool owns = (*capacityPtr != -1);
   if (!isSmall && owns)
      free(begin);
}

} // anonymous namespace

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RFieldZero::CloneImpl(std::string_view /*newName*/) const
{
   auto result = std::make_unique<RFieldZero>();
   for (auto &f : fSubFields)
      result->Attach(f->Clone(f->GetFieldName()));
   return result;
}

void ROOT::Experimental::RFieldZero::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitFieldZero(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RCardinalityField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RCardinalityField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ClusterSize_t>(desc);
}

void ROOT::Experimental::RCardinalityField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitCardinalityField(*this);
}

const ROOT::Experimental::RField<ROOT::RNTupleCardinality<std::uint32_t>> *
ROOT::Experimental::RCardinalityField::As32Bit() const
{
   return dynamic_cast<const RField<RNTupleCardinality<std::uint32_t>> *>(this);
}

const ROOT::Experimental::RField<ROOT::RNTupleCardinality<std::uint64_t>> *
ROOT::Experimental::RCardinalityField::As64Bit() const
{
   return dynamic_cast<const RField<RNTupleCardinality<std::uint64_t>> *>(this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<char>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<char>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kChar}}, {{EColumnType::kInt8},
                                                                          {EColumnType::kUInt8},
                                                                          {EColumnType::kInt16},
                                                                          {EColumnType::kUInt16},
                                                                          {EColumnType::kInt32},
                                                                          {EColumnType::kUInt32},
                                                                          {EColumnType::kInt64},
                                                                          {EColumnType::kUInt64},
                                                                          {EColumnType::kSplitInt16},
                                                                          {EColumnType::kSplitUInt16},
                                                                          {EColumnType::kSplitInt32},
                                                                          {EColumnType::kSplitUInt32},
                                                                          {EColumnType::kSplitInt64},
                                                                          {EColumnType::kSplitUInt64},
                                                                          {EColumnType::kBit}});
   return representations;
}

void ROOT::Experimental::RField<char>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitCharField(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<std::byte>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::byte>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kByte}}, {});
   return representations;
}

void ROOT::Experimental::RField<std::byte>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitByteField(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<int8_t>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RIntegralField<std::int8_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kInt8}}, {{EColumnType::kChar},
                                                                          {EColumnType::kUInt8},
                                                                          {EColumnType::kInt16},
                                                                          {EColumnType::kUInt16},
                                                                          {EColumnType::kInt32},
                                                                          {EColumnType::kUInt32},
                                                                          {EColumnType::kInt64},
                                                                          {EColumnType::kUInt64},
                                                                          {EColumnType::kSplitInt16},
                                                                          {EColumnType::kSplitUInt16},
                                                                          {EColumnType::kSplitInt32},
                                                                          {EColumnType::kSplitUInt32},
                                                                          {EColumnType::kSplitInt64},
                                                                          {EColumnType::kSplitUInt64},
                                                                          {EColumnType::kBit}});
   return representations;
}

void ROOT::Experimental::RIntegralField<std::int8_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt8Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<uint8_t>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RIntegralField<std::uint8_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kUInt8}}, {{EColumnType::kChar},
                                                                           {EColumnType::kInt8},
                                                                           {EColumnType::kInt16},
                                                                           {EColumnType::kUInt16},
                                                                           {EColumnType::kInt32},
                                                                           {EColumnType::kUInt32},
                                                                           {EColumnType::kInt64},
                                                                           {EColumnType::kUInt64},
                                                                           {EColumnType::kSplitInt16},
                                                                           {EColumnType::kSplitUInt16},
                                                                           {EColumnType::kSplitInt32},
                                                                           {EColumnType::kSplitUInt32},
                                                                           {EColumnType::kSplitInt64},
                                                                           {EColumnType::kSplitUInt64},
                                                                           {EColumnType::kBit}});
   return representations;
}

void ROOT::Experimental::RIntegralField<std::uint8_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt8Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<bool>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<bool>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kBit}}, {{EColumnType::kChar},
                                                                         {EColumnType::kInt8},
                                                                         {EColumnType::kUInt8},
                                                                         {EColumnType::kInt16},
                                                                         {EColumnType::kUInt16},
                                                                         {EColumnType::kInt32},
                                                                         {EColumnType::kUInt32},
                                                                         {EColumnType::kInt64},
                                                                         {EColumnType::kUInt64},
                                                                         {EColumnType::kSplitInt16},
                                                                         {EColumnType::kSplitUInt16},
                                                                         {EColumnType::kSplitInt32},
                                                                         {EColumnType::kSplitUInt32},
                                                                         {EColumnType::kSplitInt64},
                                                                         {EColumnType::kSplitUInt64}});
   return representations;
}

void ROOT::Experimental::RField<bool>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitBoolField(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<float>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<float>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitReal32},
                                                  {EColumnType::kReal32},
                                                  {EColumnType::kReal16},
                                                  {EColumnType::kReal32Trunc},
                                                  {EColumnType::kReal32Quant}},
                                                 {{EColumnType::kSplitReal64}, {EColumnType::kReal64}});
   return representations;
}

void ROOT::Experimental::RField<float>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitFloatField(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<double>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<double>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitReal64},
                                                  {EColumnType::kReal64},
                                                  {EColumnType::kSplitReal32},
                                                  {EColumnType::kReal32},
                                                  {EColumnType::kReal16},
                                                  {EColumnType::kReal32Trunc},
                                                  {EColumnType::kReal32Quant}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RField<double>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitDoubleField(*this);
}

void ROOT::Experimental::RField<double>::SetDouble32()
{
   fTypeAlias = "Double32_t";
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<int16_t>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RIntegralField<std::int16_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitInt16}, {EColumnType::kInt16}},
                                                 {{EColumnType::kChar},
                                                  {EColumnType::kInt8},
                                                  {EColumnType::kUInt8},
                                                  {EColumnType::kUInt16},
                                                  {EColumnType::kInt32},
                                                  {EColumnType::kUInt32},
                                                  {EColumnType::kInt64},
                                                  {EColumnType::kUInt64},
                                                  {EColumnType::kSplitUInt16},
                                                  {EColumnType::kSplitInt32},
                                                  {EColumnType::kSplitUInt32},
                                                  {EColumnType::kSplitInt64},
                                                  {EColumnType::kSplitUInt64},
                                                  {EColumnType::kBit}});
   return representations;
}

void ROOT::Experimental::RIntegralField<std::int16_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt16Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<uint16_t>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RIntegralField<std::uint16_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitUInt16}, {EColumnType::kUInt16}},
                                                 {{EColumnType::kChar},
                                                  {EColumnType::kInt8},
                                                  {EColumnType::kUInt8},
                                                  {EColumnType::kInt16},
                                                  {EColumnType::kInt32},
                                                  {EColumnType::kUInt32},
                                                  {EColumnType::kInt64},
                                                  {EColumnType::kUInt64},
                                                  {EColumnType::kSplitInt16},
                                                  {EColumnType::kSplitInt32},
                                                  {EColumnType::kSplitUInt32},
                                                  {EColumnType::kSplitInt64},
                                                  {EColumnType::kSplitUInt64},
                                                  {EColumnType::kBit}});
   return representations;
}

void ROOT::Experimental::RIntegralField<std::uint16_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt16Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<int32_t>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RIntegralField<std::int32_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitInt32}, {EColumnType::kInt32}},
                                                 {{EColumnType::kChar},
                                                  {EColumnType::kInt8},
                                                  {EColumnType::kUInt8},
                                                  {EColumnType::kInt16},
                                                  {EColumnType::kUInt16},
                                                  {EColumnType::kUInt32},
                                                  {EColumnType::kInt64},
                                                  {EColumnType::kUInt64},
                                                  {EColumnType::kSplitInt16},
                                                  {EColumnType::kSplitUInt16},
                                                  {EColumnType::kSplitUInt32},
                                                  {EColumnType::kSplitInt64},
                                                  {EColumnType::kSplitUInt64},
                                                  {EColumnType::kBit}});
   return representations;
}

void ROOT::Experimental::RIntegralField<std::int32_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt32Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<uint32_t>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RIntegralField<std::uint32_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitUInt32}, {EColumnType::kUInt32}},
                                                 {{EColumnType::kChar},
                                                  {EColumnType::kInt8},
                                                  {EColumnType::kUInt8},
                                                  {EColumnType::kInt16},
                                                  {EColumnType::kUInt16},
                                                  {EColumnType::kInt32},
                                                  {EColumnType::kInt64},
                                                  {EColumnType::kUInt64},
                                                  {EColumnType::kSplitInt16},
                                                  {EColumnType::kSplitUInt16},
                                                  {EColumnType::kSplitInt32},
                                                  {EColumnType::kSplitInt64},
                                                  {EColumnType::kSplitUInt64},
                                                  {EColumnType::kBit}});
   return representations;
}

void ROOT::Experimental::RIntegralField<std::uint32_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt32Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<uint64_t>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RIntegralField<std::uint64_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitUInt64}, {EColumnType::kUInt64}},
                                                 {{EColumnType::kChar},
                                                  {EColumnType::kInt8},
                                                  {EColumnType::kUInt8},
                                                  {EColumnType::kInt16},
                                                  {EColumnType::kUInt16},
                                                  {EColumnType::kInt32},
                                                  {EColumnType::kUInt32},
                                                  {EColumnType::kInt64},
                                                  {EColumnType::kSplitInt16},
                                                  {EColumnType::kSplitUInt16},
                                                  {EColumnType::kSplitInt32},
                                                  {EColumnType::kSplitUInt32},
                                                  {EColumnType::kSplitInt64},
                                                  {EColumnType::kBit}});
   return representations;
}

void ROOT::Experimental::RIntegralField<std::uint64_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt64Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::Experimental::RSimpleField<int64_t>;

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RIntegralField<std::int64_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitInt64}, {EColumnType::kInt64}},
                                                 {{EColumnType::kChar},
                                                  {EColumnType::kInt8},
                                                  {EColumnType::kUInt8},
                                                  {EColumnType::kInt16},
                                                  {EColumnType::kUInt16},
                                                  {EColumnType::kInt32},
                                                  {EColumnType::kUInt32},
                                                  {EColumnType::kUInt64},
                                                  {EColumnType::kSplitInt16},
                                                  {EColumnType::kSplitUInt16},
                                                  {EColumnType::kSplitInt32},
                                                  {EColumnType::kSplitUInt32},
                                                  {EColumnType::kSplitUInt64},
                                                  {EColumnType::kBit}});
   return representations;
}

void ROOT::Experimental::RIntegralField<std::int64_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt64Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::string>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitIndex64, EColumnType::kChar},
                                                  {EColumnType::kIndex64, EColumnType::kChar},
                                                  {EColumnType::kSplitIndex32, EColumnType::kChar},
                                                  {EColumnType::kIndex32, EColumnType::kChar}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RField<std::string>::GenerateColumns()
{
   GenerateColumnsImpl<ClusterSize_t, char>();
}

void ROOT::Experimental::RField<std::string>::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ClusterSize_t, char>(desc);
}

std::size_t ROOT::Experimental::RField<std::string>::AppendImpl(const void *from)
{
   auto typedValue = static_cast<const std::string *>(from);
   auto length = typedValue->length();
   fAuxiliaryColumn->AppendV(typedValue->data(), length);
   fIndex += length;
   fPrincipalColumn->Append(&fIndex);
   return length + fPrincipalColumn->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RField<std::string>::ReadGlobalImpl(ROOT::Experimental::NTupleSize_t globalIndex, void *to)
{
   auto typedValue = static_cast<std::string *>(to);
   RClusterIndex collectionStart;
   ClusterSize_t nChars;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nChars);
   if (nChars == 0) {
      typedValue->clear();
   } else {
      typedValue->resize(nChars);
      fAuxiliaryColumn->ReadV(collectionStart, nChars, const_cast<char *>(typedValue->data()));
   }
}

void ROOT::Experimental::RField<std::string>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitStringField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RRecordField::RRecordField(std::string_view name, const RRecordField &source)
   : ROOT::Experimental::RFieldBase(name, source.GetTypeName(), ENTupleStructure::kRecord, false /* isSimple */),
     fMaxAlignment(source.fMaxAlignment),
     fSize(source.fSize),
     fOffsets(source.fOffsets)
{
   for (const auto &f : source.GetSubFields())
      Attach(f->Clone(f->GetFieldName()));
   fTraits = source.fTraits;
}

ROOT::Experimental::RRecordField::RRecordField(std::string_view fieldName, std::string_view typeName)
   : ROOT::Experimental::RFieldBase(fieldName, typeName, ENTupleStructure::kRecord, false /* isSimple */)
{
}

void ROOT::Experimental::RRecordField::RRecordField::AttachItemFields(
   std::vector<std::unique_ptr<RFieldBase>> itemFields)
{
   fTraits |= kTraitTrivialType;
   for (auto &item : itemFields) {
      fMaxAlignment = std::max(fMaxAlignment, item->GetAlignment());
      fSize += GetItemPadding(fSize, item->GetAlignment()) + item->GetValueSize();
      fTraits &= item->GetTraits();
      Attach(std::move(item));
   }
   // Trailing padding: although this is implementation-dependent, most add enough padding to comply with the
   // requirements of the type with strictest alignment
   fSize += GetItemPadding(fSize, fMaxAlignment);
}

ROOT::Experimental::RRecordField::RRecordField(std::string_view fieldName,
                                               std::vector<std::unique_ptr<RFieldBase>> itemFields)
   : ROOT::Experimental::RFieldBase(fieldName, "", ENTupleStructure::kRecord, false /* isSimple */)
{
   fTraits |= kTraitTrivialType;
   fOffsets.reserve(itemFields.size());
   for (auto &item : itemFields) {
      fSize += GetItemPadding(fSize, item->GetAlignment());
      fOffsets.push_back(fSize);
      fMaxAlignment = std::max(fMaxAlignment, item->GetAlignment());
      fSize += item->GetValueSize();
      fTraits &= item->GetTraits();
      Attach(std::move(item));
   }
   // Trailing padding: although this is implementation-dependent, most add enough padding to comply with the
   // requirements of the type with strictest alignment
   fSize += GetItemPadding(fSize, fMaxAlignment);
}

std::size_t ROOT::Experimental::RRecordField::GetItemPadding(std::size_t baseOffset, std::size_t itemAlignment) const
{
   if (itemAlignment > 1) {
      auto remainder = baseOffset % itemAlignment;
      if (remainder != 0)
         return itemAlignment - remainder;
   }
   return 0;
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RRecordField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RRecordField>(new RRecordField(newName, *this));
}

std::size_t ROOT::Experimental::RRecordField::AppendImpl(const void *from)
{
   std::size_t nbytes = 0;
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      nbytes += CallAppendOn(*fSubFields[i], static_cast<const unsigned char *>(from) + fOffsets[i]);
   }
   return nbytes;
}

void ROOT::Experimental::RRecordField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      CallReadOn(*fSubFields[i], globalIndex, static_cast<unsigned char *>(to) + fOffsets[i]);
   }
}

void ROOT::Experimental::RRecordField::ReadInClusterImpl(RClusterIndex clusterIndex, void *to)
{
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      CallReadOn(*fSubFields[i], clusterIndex, static_cast<unsigned char *>(to) + fOffsets[i]);
   }
}

void ROOT::Experimental::RRecordField::ConstructValue(void *where) const
{
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      CallConstructValueOn(*fSubFields[i], static_cast<unsigned char *>(where) + fOffsets[i]);
   }
}

void ROOT::Experimental::RRecordField::RRecordDeleter::operator()(void *objPtr, bool dtorOnly)
{
   for (unsigned i = 0; i < fItemDeleters.size(); ++i) {
      fItemDeleters[i]->operator()(reinterpret_cast<unsigned char *>(objPtr) + fOffsets[i], true /* dtorOnly */);
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::RRecordField::GetDeleter() const
{
   std::vector<std::unique_ptr<RDeleter>> itemDeleters;
   itemDeleters.reserve(fOffsets.size());
   for (const auto &f : fSubFields) {
      itemDeleters.emplace_back(GetDeleterOf(*f));
   }
   return std::make_unique<RRecordDeleter>(std::move(itemDeleters), fOffsets);
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RRecordField::SplitValue(const RValue &value) const
{
   auto basePtr = value.GetPtr<unsigned char>().get();
   std::vector<RValue> result;
   result.reserve(fSubFields.size());
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      result.emplace_back(fSubFields[i]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), basePtr + fOffsets[i])));
   }
   return result;
}

void ROOT::Experimental::RRecordField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitRecordField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RVectorField::RVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField,
                                               bool isUntyped)
   : ROOT::Experimental::RFieldBase(fieldName, isUntyped ? "" : "std::vector<" + itemField->GetTypeName() + ">",
                                    ENTupleStructure::kCollection, false /* isSimple */),
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
   R__ASSERT((typedValue->size() % fItemSize) == 0);
   std::size_t nbytes = 0;
   auto count = typedValue->size() / fItemSize;

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

void ROOT::Experimental::RVectorField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   auto typedValue = static_cast<std::vector<char> *>(to);

   ClusterSize_t nItems;
   RClusterIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   if (fSubFields[0]->IsSimple()) {
      typedValue->resize(nItems * fItemSize);
      if (nItems)
         GetPrincipalColumnOf(*fSubFields[0])->ReadV(collectionStart, nItems, typedValue->data());
      return;
   }

   // See "semantics of reading non-trivial objects" in RNTuple's Architecture.md
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
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RVectorField::GenerateColumns()
{
   GenerateColumnsImpl<ClusterSize_t>();
}

void ROOT::Experimental::RVectorField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ClusterSize_t>(desc);
}

void ROOT::Experimental::RVectorField::RVectorDeleter::operator()(void *objPtr, bool dtorOnly)
{
   auto vecPtr = static_cast<std::vector<char> *>(objPtr);
   if (fItemDeleter) {
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

ROOT::Experimental::RRVecField::RRVecField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
   : ROOT::Experimental::RFieldBase(fieldName, "ROOT::VecOps::RVec<" + itemField->GetTypeName() + ">",
                                    ENTupleStructure::kCollection, false /* isSimple */),
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

void ROOT::Experimental::RRVecField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   // TODO as a performance optimization, we could assign values to elements of the inline buffer:
   // if size < inline buffer size: we save one allocation here and usage of the RVec skips a pointer indirection

   auto [beginPtr, sizePtr, capacityPtr] = GetRVecDataMembers(to);

   // Read collection info for this entry
   ClusterSize_t nItems;
   RClusterIndex collectionStart;
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
   RClusterIndex firstItemIndex;
   RClusterIndex collectionStart;
   ClusterSize_t collectionSize;
   this->GetCollectionInfo(bulkSpec.fFirstIndex, &firstItemIndex, &collectionSize);
   *beginPtr = itemValueArray;
   *sizePtr = collectionSize;
   *capacityPtr = -1;

   // Set the size of the remaining RVecs of the bulk, going page by page through the RNTuple offset column.
   // We optimistically assume that bulkSpec.fAuxData is already large enough to hold all the item values in the
   // given range. If not, we'll fix up the pointers afterwards.
   auto lastOffset = firstItemIndex.GetIndex() + collectionSize;
   ClusterSize_t::ValueType nRemainingValues = bulkSpec.fCount - 1;
   std::size_t nValues = 1;
   std::size_t nItems = collectionSize;
   while (nRemainingValues > 0) {
      NTupleSize_t nElementsUntilPageEnd;
      const auto offsets = fPrincipalColumn->MapV<ClusterSize_t>(bulkSpec.fFirstIndex + nValues, nElementsUntilPageEnd);
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
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RRVecField::GenerateColumns()
{
   GenerateColumnsImpl<ClusterSize_t>();
}

void ROOT::Experimental::RRVecField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ClusterSize_t>(desc);
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

ROOT::Experimental::RField<std::vector<bool>>::RField(std::string_view name)
   : ROOT::Experimental::RFieldBase(name, "std::vector<bool>", ENTupleStructure::kCollection, false /* isSimple */)
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

void ROOT::Experimental::RField<std::vector<bool>>::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   auto typedValue = static_cast<std::vector<bool> *>(to);

   ClusterSize_t nItems;
   RClusterIndex collectionStart;
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
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumns()
{
   GenerateColumnsImpl<ClusterSize_t>();
}

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ClusterSize_t>(desc);
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

ROOT::Experimental::RArrayField::RArrayField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField,
                                             std::size_t arrayLength)
   : ROOT::Experimental::RFieldBase(fieldName,
                                    "std::array<" + itemField->GetTypeName() + "," + std::to_string(arrayLength) + ">",
                                    ENTupleStructure::kLeaf, false /* isSimple */, arrayLength),
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
   auto arrayPtr = static_cast<const unsigned char *>(from);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      nbytes += CallAppendOn(*fSubFields[0], arrayPtr + (i * fItemSize));
   }
   return nbytes;
}

void ROOT::Experimental::RArrayField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   auto arrayPtr = static_cast<unsigned char *>(to);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      CallReadOn(*fSubFields[0], globalIndex * fArrayLength + i, arrayPtr + (i * fItemSize));
   }
}

void ROOT::Experimental::RArrayField::ReadInClusterImpl(RClusterIndex clusterIndex, void *to)
{
   auto arrayPtr = static_cast<unsigned char *>(to);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      CallReadOn(*fSubFields[0], RClusterIndex(clusterIndex.GetClusterId(), clusterIndex.GetIndex() * fArrayLength + i),
                 arrayPtr + (i * fItemSize));
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
// RArrayAsRVecField

ROOT::Experimental::RArrayAsRVecField::RArrayAsRVecField(std::string_view fieldName,
                                                         std::unique_ptr<ROOT::Experimental::RFieldBase> itemField,
                                                         std::size_t arrayLength)
   : ROOT::Experimental::RFieldBase(fieldName, "ROOT::VecOps::RVec<" + itemField->GetTypeName() + ">",
                                    ENTupleStructure::kCollection, false /* isSimple */),
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

void ROOT::Experimental::RArrayAsRVecField::ReadGlobalImpl(ROOT::Experimental::NTupleSize_t globalIndex, void *to)
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

void ROOT::Experimental::RArrayAsRVecField::ReadInClusterImpl(ROOT::Experimental::RClusterIndex clusterIndex, void *to)
{
   auto [beginPtr, _, __] = GetRVecDataMembers(to);
   auto rvecBeginPtr = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics

   const auto &clusterId = clusterIndex.GetClusterId();
   const auto &clusterIndexIndex = clusterIndex.GetIndex();

   if (fSubFields[0]->IsSimple()) {
      GetPrincipalColumnOf(*fSubFields[0])
         ->ReadV(RClusterIndex(clusterId, clusterIndexIndex * fArrayLength), fArrayLength, rvecBeginPtr);
      return;
   }

   // Read the new values into the collection elements
   for (std::size_t i = 0; i < fArrayLength; ++i) {
      CallReadOn(*fSubFields[0], RClusterIndex(clusterId, clusterIndexIndex * fArrayLength + i),
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

// RArrayAsRVecField
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------

ROOT::Experimental::RBitsetField::RBitsetField(std::string_view fieldName, std::size_t N)
   : ROOT::Experimental::RFieldBase(fieldName, "std::bitset<" + std::to_string(N) + ">", ENTupleStructure::kLeaf,
                                    false /* isSimple */, N),
     fN(N)
{
   fTraits |= kTraitTriviallyDestructible;
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RBitsetField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kBit}}, {});
   return representations;
}

void ROOT::Experimental::RBitsetField::GenerateColumns()
{
   GenerateColumnsImpl<bool>();
}

void ROOT::Experimental::RBitsetField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<bool>(desc);
}

std::size_t ROOT::Experimental::RBitsetField::AppendImpl(const void *from)
{
   const auto *asULongArray = static_cast<const Word_t *>(from);
   bool elementValue;
   std::size_t i = 0;
   for (std::size_t word = 0; word < (fN + kBitsPerWord - 1) / kBitsPerWord; ++word) {
      for (std::size_t mask = 0; (mask < kBitsPerWord) && (i < fN); ++mask, ++i) {
         elementValue = (asULongArray[word] & (static_cast<Word_t>(1) << mask)) != 0;
         fPrincipalColumn->Append(&elementValue);
      }
   }
   return fN;
}

void ROOT::Experimental::RBitsetField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   auto *asULongArray = static_cast<Word_t *>(to);
   bool elementValue;
   for (std::size_t i = 0; i < fN; ++i) {
      fPrincipalColumn->Read(globalIndex * fN + i, &elementValue);
      Word_t mask = static_cast<Word_t>(1) << (i % kBitsPerWord);
      Word_t bit = static_cast<Word_t>(elementValue) << (i % kBitsPerWord);
      asULongArray[i / kBitsPerWord] = (asULongArray[i / kBitsPerWord] & ~mask) | bit;
   }
}

void ROOT::Experimental::RBitsetField::ReadInClusterImpl(RClusterIndex clusterIndex, void *to)
{
   auto *asULongArray = static_cast<Word_t *>(to);
   bool elementValue;
   for (std::size_t i = 0; i < fN; ++i) {
      fPrincipalColumn->Read(RClusterIndex(clusterIndex.GetClusterId(), clusterIndex.GetIndex() * fN) + i,
                             &elementValue);
      Word_t mask = static_cast<Word_t>(1) << (i % kBitsPerWord);
      Word_t bit = static_cast<Word_t>(elementValue) << (i % kBitsPerWord);
      asULongArray[i / kBitsPerWord] = (asULongArray[i / kBitsPerWord] & ~mask) | bit;
   }
}

void ROOT::Experimental::RBitsetField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitBitsetField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNullableField::RNullableField(std::string_view fieldName, std::string_view typeName,
                                                   std::unique_ptr<RFieldBase> itemField)
   : ROOT::Experimental::RFieldBase(fieldName, typeName, ENTupleStructure::kCollection, false /* isSimple */)
{
   Attach(std::move(itemField));
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RNullableField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RNullableField::GenerateColumns()
{
   GenerateColumnsImpl<ClusterSize_t>();
}

void ROOT::Experimental::RNullableField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ClusterSize_t>(desc);
}

std::size_t ROOT::Experimental::RNullableField::AppendNull()
{
   fPrincipalColumn->Append(&fNWritten);
   return sizeof(ClusterSize_t);
}

std::size_t ROOT::Experimental::RNullableField::AppendValue(const void *from)
{
   auto nbytesItem = CallAppendOn(*fSubFields[0], from);
   fNWritten++;
   fPrincipalColumn->Append(&fNWritten);
   return sizeof(ClusterSize_t) + nbytesItem;
}

ROOT::Experimental::RClusterIndex ROOT::Experimental::RNullableField::GetItemIndex(NTupleSize_t globalIndex)
{
   RClusterIndex collectionStart;
   ClusterSize_t collectionSize;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &collectionSize);
   return (collectionSize == 0) ? RClusterIndex() : collectionStart;
}

void ROOT::Experimental::RNullableField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitNullableField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RUniquePtrField::RUniquePtrField(std::string_view fieldName, std::string_view typeName,
                                                     std::unique_ptr<RFieldBase> itemField)
   : RNullableField(fieldName, typeName, std::move(itemField)), fItemDeleter(GetDeleterOf(*fSubFields[0]))
{
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RUniquePtrField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::make_unique<RUniquePtrField>(newName, GetTypeName(), std::move(newItemField));
}

std::size_t ROOT::Experimental::RUniquePtrField::AppendImpl(const void *from)
{
   auto typedValue = static_cast<const std::unique_ptr<char> *>(from);
   if (*typedValue) {
      return AppendValue(typedValue->get());
   } else {
      return AppendNull();
   }
}

void ROOT::Experimental::RUniquePtrField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   auto ptr = static_cast<std::unique_ptr<char> *>(to);
   bool isValidValue = static_cast<bool>(*ptr);

   auto itemIndex = GetItemIndex(globalIndex);
   bool isValidItem = itemIndex.GetIndex() != kInvalidClusterIndex;

   void *valuePtr = nullptr;
   if (isValidValue)
      valuePtr = ptr->get();

   if (isValidValue && !isValidItem) {
      ptr->release();
      fItemDeleter->operator()(valuePtr, false /* dtorOnly */);
      return;
   }

   if (!isValidItem) // On-disk value missing; nothing else to do
      return;

   if (!isValidValue) {
      valuePtr = CallCreateObjectRawPtrOn(*fSubFields[0]);
      ptr->reset(reinterpret_cast<char *>(valuePtr));
   }

   CallReadOn(*fSubFields[0], itemIndex, valuePtr);
}

void ROOT::Experimental::RUniquePtrField::RUniquePtrDeleter::operator()(void *objPtr, bool dtorOnly)
{
   auto typedPtr = static_cast<std::unique_ptr<char> *>(objPtr);
   if (*typedPtr) {
      fItemDeleter->operator()(typedPtr->get(), false /* dtorOnly */);
      typedPtr->release();
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::RUniquePtrField::GetDeleter() const
{
   return std::make_unique<RUniquePtrDeleter>(GetDeleterOf(*fSubFields[0]));
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RUniquePtrField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   const auto &ptr = value.GetRef<std::unique_ptr<char>>();
   if (ptr) {
      result.emplace_back(fSubFields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), ptr.get())));
   }
   return result;
}

//------------------------------------------------------------------------------

ROOT::Experimental::ROptionalField::ROptionalField(std::string_view fieldName, std::string_view typeName,
                                                   std::unique_ptr<RFieldBase> itemField)
   : RNullableField(fieldName, typeName, std::move(itemField)), fItemDeleter(GetDeleterOf(*fSubFields[0]))
{
   if (fSubFields[0]->GetTraits() & kTraitTriviallyDestructible)
      fTraits |= kTraitTriviallyDestructible;
}

std::pair<void *, bool *> ROOT::Experimental::ROptionalField::GetValueAndEngagementPtrs(void *optionalPtr) const
{
   void *value = optionalPtr;
   bool *engagement =
      reinterpret_cast<bool *>(reinterpret_cast<unsigned char *>(optionalPtr) + fSubFields[0]->GetValueSize());
   return {value, engagement};
}

std::pair<const void *, const bool *>
ROOT::Experimental::ROptionalField::GetValueAndEngagementPtrs(const void *optionalPtr) const
{
   return GetValueAndEngagementPtrs(const_cast<void *>(optionalPtr));
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::ROptionalField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::make_unique<ROptionalField>(newName, GetTypeName(), std::move(newItemField));
}

std::size_t ROOT::Experimental::ROptionalField::AppendImpl(const void *from)
{
   const auto [valuePtr, engagementPtr] = GetValueAndEngagementPtrs(from);
   if (*engagementPtr) {
      return AppendValue(valuePtr);
   } else {
      return AppendNull();
   }
}

void ROOT::Experimental::ROptionalField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   auto [valuePtr, engagementPtr] = GetValueAndEngagementPtrs(to);
   auto itemIndex = GetItemIndex(globalIndex);
   if (itemIndex.GetIndex() == kInvalidClusterIndex) {
      *engagementPtr = false;
   } else {
      CallReadOn(*fSubFields[0], itemIndex, valuePtr);
      *engagementPtr = true;
   }
}

void ROOT::Experimental::ROptionalField::ConstructValue(void *where) const
{
   auto [valuePtr, engagementPtr] = GetValueAndEngagementPtrs(where);
   CallConstructValueOn(*fSubFields[0], valuePtr);
   *engagementPtr = false;
}

void ROOT::Experimental::ROptionalField::ROptionalDeleter::operator()(void *objPtr, bool dtorOnly)
{
   fItemDeleter->operator()(objPtr, true /* dtorOnly */);
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::ROptionalField::GetDeleter() const
{
   return std::make_unique<ROptionalDeleter>(GetDeleterOf(*fSubFields[0]));
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::ROptionalField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   const auto [valuePtr, engagementPtr] = GetValueAndEngagementPtrs(value.GetPtr<void>().get());
   if (*engagementPtr) {
      result.emplace_back(fSubFields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), valuePtr)));
   }
   return result;
}

size_t ROOT::Experimental::ROptionalField::GetValueSize() const
{
   const auto alignment = GetAlignment();
   // real size is the sum of the value size and the engagement boolean
   const auto actualSize = fSubFields[0]->GetValueSize() + sizeof(bool);
   auto padding = 0;
   if (alignment > 1) {
      auto remainder = actualSize % alignment;
      if (remainder != 0)
         padding = alignment - remainder;
   }
   return actualSize + padding;
}

size_t ROOT::Experimental::ROptionalField::GetAlignment() const
{
   return fSubFields[0]->GetAlignment();
}

//------------------------------------------------------------------------------

ROOT::Experimental::RAtomicField::RAtomicField(std::string_view fieldName, std::string_view typeName,
                                               std::unique_ptr<RFieldBase> itemField)
   : RFieldBase(fieldName, typeName, ENTupleStructure::kLeaf, false /* isSimple */)
{
   if (itemField->GetTraits() & kTraitTriviallyConstructible)
      fTraits |= kTraitTriviallyConstructible;
   if (itemField->GetTraits() & kTraitTriviallyDestructible)
      fTraits |= kTraitTriviallyDestructible;
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RAtomicField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::make_unique<RAtomicField>(newName, GetTypeName(), std::move(newItemField));
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RAtomicField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   result.emplace_back(fSubFields[0]->BindValue(value.GetPtr<void>()));
   return result;
}

void ROOT::Experimental::RAtomicField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitAtomicField(*this);
}
