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
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>

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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RCardinalityField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<Internal::RColumnIndex>(desc);
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
   static RColumnRepresentations representations({{ENTupleColumnType::kChar}}, {{ENTupleColumnType::kInt8},
                                                                                {ENTupleColumnType::kUInt8},
                                                                                {ENTupleColumnType::kInt16},
                                                                                {ENTupleColumnType::kUInt16},
                                                                                {ENTupleColumnType::kInt32},
                                                                                {ENTupleColumnType::kUInt32},
                                                                                {ENTupleColumnType::kInt64},
                                                                                {ENTupleColumnType::kUInt64},
                                                                                {ENTupleColumnType::kSplitInt16},
                                                                                {ENTupleColumnType::kSplitUInt16},
                                                                                {ENTupleColumnType::kSplitInt32},
                                                                                {ENTupleColumnType::kSplitUInt32},
                                                                                {ENTupleColumnType::kSplitInt64},
                                                                                {ENTupleColumnType::kSplitUInt64},
                                                                                {ENTupleColumnType::kBit}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kByte}}, {});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kInt8}}, {{ENTupleColumnType::kChar},
                                                                                {ENTupleColumnType::kUInt8},
                                                                                {ENTupleColumnType::kInt16},
                                                                                {ENTupleColumnType::kUInt16},
                                                                                {ENTupleColumnType::kInt32},
                                                                                {ENTupleColumnType::kUInt32},
                                                                                {ENTupleColumnType::kInt64},
                                                                                {ENTupleColumnType::kUInt64},
                                                                                {ENTupleColumnType::kSplitInt16},
                                                                                {ENTupleColumnType::kSplitUInt16},
                                                                                {ENTupleColumnType::kSplitInt32},
                                                                                {ENTupleColumnType::kSplitUInt32},
                                                                                {ENTupleColumnType::kSplitInt64},
                                                                                {ENTupleColumnType::kSplitUInt64},
                                                                                {ENTupleColumnType::kBit}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kUInt8}}, {{ENTupleColumnType::kChar},
                                                                                 {ENTupleColumnType::kInt8},
                                                                                 {ENTupleColumnType::kInt16},
                                                                                 {ENTupleColumnType::kUInt16},
                                                                                 {ENTupleColumnType::kInt32},
                                                                                 {ENTupleColumnType::kUInt32},
                                                                                 {ENTupleColumnType::kInt64},
                                                                                 {ENTupleColumnType::kUInt64},
                                                                                 {ENTupleColumnType::kSplitInt16},
                                                                                 {ENTupleColumnType::kSplitUInt16},
                                                                                 {ENTupleColumnType::kSplitInt32},
                                                                                 {ENTupleColumnType::kSplitUInt32},
                                                                                 {ENTupleColumnType::kSplitInt64},
                                                                                 {ENTupleColumnType::kSplitUInt64},
                                                                                 {ENTupleColumnType::kBit}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kBit}}, {{ENTupleColumnType::kChar},
                                                                               {ENTupleColumnType::kInt8},
                                                                               {ENTupleColumnType::kUInt8},
                                                                               {ENTupleColumnType::kInt16},
                                                                               {ENTupleColumnType::kUInt16},
                                                                               {ENTupleColumnType::kInt32},
                                                                               {ENTupleColumnType::kUInt32},
                                                                               {ENTupleColumnType::kInt64},
                                                                               {ENTupleColumnType::kUInt64},
                                                                               {ENTupleColumnType::kSplitInt16},
                                                                               {ENTupleColumnType::kSplitUInt16},
                                                                               {ENTupleColumnType::kSplitInt32},
                                                                               {ENTupleColumnType::kSplitUInt32},
                                                                               {ENTupleColumnType::kSplitInt64},
                                                                               {ENTupleColumnType::kSplitUInt64}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitReal32},
                                                  {ENTupleColumnType::kReal32},
                                                  {ENTupleColumnType::kReal16},
                                                  {ENTupleColumnType::kReal32Trunc},
                                                  {ENTupleColumnType::kReal32Quant}},
                                                 {{ENTupleColumnType::kSplitReal64}, {ENTupleColumnType::kReal64}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitReal64},
                                                  {ENTupleColumnType::kReal64},
                                                  {ENTupleColumnType::kSplitReal32},
                                                  {ENTupleColumnType::kReal32},
                                                  {ENTupleColumnType::kReal16},
                                                  {ENTupleColumnType::kReal32Trunc},
                                                  {ENTupleColumnType::kReal32Quant}},
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitInt16}, {ENTupleColumnType::kInt16}},
                                                 {{ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kInt8},
                                                  {ENTupleColumnType::kUInt8},
                                                  {ENTupleColumnType::kUInt16},
                                                  {ENTupleColumnType::kInt32},
                                                  {ENTupleColumnType::kUInt32},
                                                  {ENTupleColumnType::kInt64},
                                                  {ENTupleColumnType::kUInt64},
                                                  {ENTupleColumnType::kSplitUInt16},
                                                  {ENTupleColumnType::kSplitInt32},
                                                  {ENTupleColumnType::kSplitUInt32},
                                                  {ENTupleColumnType::kSplitInt64},
                                                  {ENTupleColumnType::kSplitUInt64},
                                                  {ENTupleColumnType::kBit}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitUInt16}, {ENTupleColumnType::kUInt16}},
                                                 {{ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kInt8},
                                                  {ENTupleColumnType::kUInt8},
                                                  {ENTupleColumnType::kInt16},
                                                  {ENTupleColumnType::kInt32},
                                                  {ENTupleColumnType::kUInt32},
                                                  {ENTupleColumnType::kInt64},
                                                  {ENTupleColumnType::kUInt64},
                                                  {ENTupleColumnType::kSplitInt16},
                                                  {ENTupleColumnType::kSplitInt32},
                                                  {ENTupleColumnType::kSplitUInt32},
                                                  {ENTupleColumnType::kSplitInt64},
                                                  {ENTupleColumnType::kSplitUInt64},
                                                  {ENTupleColumnType::kBit}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitInt32}, {ENTupleColumnType::kInt32}},
                                                 {{ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kInt8},
                                                  {ENTupleColumnType::kUInt8},
                                                  {ENTupleColumnType::kInt16},
                                                  {ENTupleColumnType::kUInt16},
                                                  {ENTupleColumnType::kUInt32},
                                                  {ENTupleColumnType::kInt64},
                                                  {ENTupleColumnType::kUInt64},
                                                  {ENTupleColumnType::kSplitInt16},
                                                  {ENTupleColumnType::kSplitUInt16},
                                                  {ENTupleColumnType::kSplitUInt32},
                                                  {ENTupleColumnType::kSplitInt64},
                                                  {ENTupleColumnType::kSplitUInt64},
                                                  {ENTupleColumnType::kBit}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitUInt32}, {ENTupleColumnType::kUInt32}},
                                                 {{ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kInt8},
                                                  {ENTupleColumnType::kUInt8},
                                                  {ENTupleColumnType::kInt16},
                                                  {ENTupleColumnType::kUInt16},
                                                  {ENTupleColumnType::kInt32},
                                                  {ENTupleColumnType::kInt64},
                                                  {ENTupleColumnType::kUInt64},
                                                  {ENTupleColumnType::kSplitInt16},
                                                  {ENTupleColumnType::kSplitUInt16},
                                                  {ENTupleColumnType::kSplitInt32},
                                                  {ENTupleColumnType::kSplitInt64},
                                                  {ENTupleColumnType::kSplitUInt64},
                                                  {ENTupleColumnType::kBit}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitUInt64}, {ENTupleColumnType::kUInt64}},
                                                 {{ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kInt8},
                                                  {ENTupleColumnType::kUInt8},
                                                  {ENTupleColumnType::kInt16},
                                                  {ENTupleColumnType::kUInt16},
                                                  {ENTupleColumnType::kInt32},
                                                  {ENTupleColumnType::kUInt32},
                                                  {ENTupleColumnType::kInt64},
                                                  {ENTupleColumnType::kSplitInt16},
                                                  {ENTupleColumnType::kSplitUInt16},
                                                  {ENTupleColumnType::kSplitInt32},
                                                  {ENTupleColumnType::kSplitUInt32},
                                                  {ENTupleColumnType::kSplitInt64},
                                                  {ENTupleColumnType::kBit}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitInt64}, {ENTupleColumnType::kInt64}},
                                                 {{ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kInt8},
                                                  {ENTupleColumnType::kUInt8},
                                                  {ENTupleColumnType::kInt16},
                                                  {ENTupleColumnType::kUInt16},
                                                  {ENTupleColumnType::kInt32},
                                                  {ENTupleColumnType::kUInt32},
                                                  {ENTupleColumnType::kUInt64},
                                                  {ENTupleColumnType::kSplitInt16},
                                                  {ENTupleColumnType::kSplitUInt16},
                                                  {ENTupleColumnType::kSplitInt32},
                                                  {ENTupleColumnType::kSplitUInt32},
                                                  {ENTupleColumnType::kSplitUInt64},
                                                  {ENTupleColumnType::kBit}});
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64, ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kIndex64, ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kSplitIndex32, ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kIndex32, ENTupleColumnType::kChar}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RField<std::string>::GenerateColumns()
{
   GenerateColumnsImpl<Internal::RColumnIndex, char>();
}

void ROOT::Experimental::RField<std::string>::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<Internal::RColumnIndex, char>(desc);
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
   RNTupleLocalIndex collectionStart;
   NTupleSize_t nChars;
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

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::Internal::CreateEmulatedField(std::string_view fieldName,
                                                  std::vector<std::unique_ptr<RFieldBase>> itemFields,
                                                  std::string_view emulatedFromType)
{
   return std::unique_ptr<RFieldBase>(new RRecordField(fieldName, std::move(itemFields), emulatedFromType));
}

ROOT::Experimental::RRecordField::RRecordField(std::string_view fieldName,
                                               std::vector<std::unique_ptr<RFieldBase>> itemFields,
                                               std::string_view emulatedFromType)
   : ROOT::Experimental::RFieldBase(fieldName, emulatedFromType, ENTupleStructure::kRecord, false /* isSimple */)
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
   fTraits |= !emulatedFromType.empty() * kTraitEmulatedField;
   // Trailing padding: although this is implementation-dependent, most add enough padding to comply with the
   // requirements of the type with strictest alignment
   fSize += GetItemPadding(fSize, fMaxAlignment);
}

ROOT::Experimental::RRecordField::RRecordField(std::string_view fieldName,
                                               std::vector<std::unique_ptr<RFieldBase>> itemFields)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields), "")
{
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

void ROOT::Experimental::RRecordField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      CallReadOn(*fSubFields[i], localIndex, static_cast<unsigned char *>(to) + fOffsets[i]);
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
   static RColumnRepresentations representations({{ENTupleColumnType::kBit}}, {});
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

void ROOT::Experimental::RBitsetField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   auto *asULongArray = static_cast<Word_t *>(to);
   bool elementValue;
   for (std::size_t i = 0; i < fN; ++i) {
      fPrincipalColumn->Read(RNTupleLocalIndex(localIndex.GetClusterId(), localIndex.GetIndexInCluster() * fN) + i,
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
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RNullableField::GenerateColumns()
{
   GenerateColumnsImpl<Internal::RColumnIndex>();
}

void ROOT::Experimental::RNullableField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<Internal::RColumnIndex>(desc);
}

std::size_t ROOT::Experimental::RNullableField::AppendNull()
{
   fPrincipalColumn->Append(&fNWritten);
   return sizeof(Internal::RColumnIndex);
}

std::size_t ROOT::Experimental::RNullableField::AppendValue(const void *from)
{
   auto nbytesItem = CallAppendOn(*fSubFields[0], from);
   fNWritten++;
   fPrincipalColumn->Append(&fNWritten);
   return sizeof(Internal::RColumnIndex) + nbytesItem;
}

ROOT::Experimental::RNTupleLocalIndex ROOT::Experimental::RNullableField::GetItemIndex(NTupleSize_t globalIndex)
{
   RNTupleLocalIndex collectionStart;
   NTupleSize_t collectionSize;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &collectionSize);
   return (collectionSize == 0) ? RNTupleLocalIndex() : collectionStart;
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
   bool isValidItem = itemIndex.GetIndexInCluster() != kInvalidNTupleIndex;

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

bool *ROOT::Experimental::ROptionalField::GetEngagementPtr(void *optionalPtr) const
{
   return reinterpret_cast<bool *>(reinterpret_cast<unsigned char *>(optionalPtr) + fSubFields[0]->GetValueSize());
}

const bool *ROOT::Experimental::ROptionalField::GetEngagementPtr(const void *optionalPtr) const
{
   return GetEngagementPtr(const_cast<void *>(optionalPtr));
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::ROptionalField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::make_unique<ROptionalField>(newName, GetTypeName(), std::move(newItemField));
}

std::size_t ROOT::Experimental::ROptionalField::AppendImpl(const void *from)
{
   if (*GetEngagementPtr(from)) {
      return AppendValue(from);
   } else {
      return AppendNull();
   }
}

void ROOT::Experimental::ROptionalField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   auto engagementPtr = GetEngagementPtr(to);
   auto itemIndex = GetItemIndex(globalIndex);
   if (itemIndex.GetIndexInCluster() == kInvalidNTupleIndex) {
      if (*engagementPtr && !(fSubFields[0]->GetTraits() & kTraitTriviallyDestructible))
         fItemDeleter->operator()(to, true /* dtorOnly */);
      *engagementPtr = false;
   } else {
      if (!(*engagementPtr) && !(fSubFields[0]->GetTraits() & kTraitTriviallyConstructible))
         CallConstructValueOn(*fSubFields[0], to);
      CallReadOn(*fSubFields[0], itemIndex, to);
      *engagementPtr = true;
   }
}

void ROOT::Experimental::ROptionalField::ConstructValue(void *where) const
{
   *GetEngagementPtr(where) = false;
}

void ROOT::Experimental::ROptionalField::ROptionalDeleter::operator()(void *objPtr, bool dtorOnly)
{
   if (fItemDeleter) {
      auto engagementPtr = reinterpret_cast<bool *>(reinterpret_cast<unsigned char *>(objPtr) + fEngagementPtrOffset);
      if (*engagementPtr)
         fItemDeleter->operator()(objPtr, true /* dtorOnly */);
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::ROptionalField::GetDeleter() const
{
   return std::make_unique<ROptionalDeleter>(
      (fSubFields[0]->GetTraits() & kTraitTriviallyDestructible) ? nullptr : GetDeleterOf(*fSubFields[0]),
      fSubFields[0]->GetValueSize());
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::ROptionalField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   const auto valuePtr = value.GetPtr<void>().get();
   if (*GetEngagementPtr(valuePtr)) {
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
