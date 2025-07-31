/// \file RField.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15

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
#include <ROOT/RNTupleTypes.hxx>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <type_traits>

std::unique_ptr<ROOT::RFieldBase> ROOT::RFieldZero::CloneImpl(std::string_view /*newName*/) const
{
   auto result = std::make_unique<RFieldZero>();
   for (auto &f : fSubfields)
      result->Attach(f->Clone(f->GetFieldName()));
   return result;
}

void ROOT::RFieldZero::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitFieldZero(*this);
}

//------------------------------------------------------------------------------

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RCardinalityField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::RCardinalityField::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(desc);
}

void ROOT::RCardinalityField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitCardinalityField(*this);
}

const ROOT::RField<ROOT::RNTupleCardinality<std::uint32_t>> *ROOT::RCardinalityField::As32Bit() const
{
   return dynamic_cast<const RField<RNTupleCardinality<std::uint32_t>> *>(this);
}

const ROOT::RField<ROOT::RNTupleCardinality<std::uint64_t>> *ROOT::RCardinalityField::As64Bit() const
{
   return dynamic_cast<const RField<RNTupleCardinality<std::uint64_t>> *>(this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<char>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RField<char>::GetColumnRepresentations() const
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

void ROOT::RField<char>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitCharField(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<std::byte>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RField<std::byte>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kByte}}, {});
   return representations;
}

void ROOT::RField<std::byte>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitByteField(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<int8_t>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RIntegralField<std::int8_t>::GetColumnRepresentations() const
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

void ROOT::RIntegralField<std::int8_t>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt8Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<uint8_t>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RIntegralField<std::uint8_t>::GetColumnRepresentations() const
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

void ROOT::RIntegralField<std::uint8_t>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt8Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<bool>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RField<bool>::GetColumnRepresentations() const
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

void ROOT::RField<bool>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitBoolField(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<float>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RField<float>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitReal32},
                                                  {ENTupleColumnType::kReal32},
                                                  {ENTupleColumnType::kReal16},
                                                  {ENTupleColumnType::kReal32Trunc},
                                                  {ENTupleColumnType::kReal32Quant}},
                                                 {{ENTupleColumnType::kSplitReal64}, {ENTupleColumnType::kReal64}});
   return representations;
}

void ROOT::RField<float>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitFloatField(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<double>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RField<double>::GetColumnRepresentations() const
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

void ROOT::RField<double>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitDoubleField(*this);
}

void ROOT::RField<double>::SetDouble32()
{
   fTypeAlias = "Double32_t";
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<int16_t>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RIntegralField<std::int16_t>::GetColumnRepresentations() const
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

void ROOT::RIntegralField<std::int16_t>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt16Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<uint16_t>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RIntegralField<std::uint16_t>::GetColumnRepresentations() const
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

void ROOT::RIntegralField<std::uint16_t>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt16Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<int32_t>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RIntegralField<std::int32_t>::GetColumnRepresentations() const
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

void ROOT::RIntegralField<std::int32_t>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt32Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<uint32_t>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RIntegralField<std::uint32_t>::GetColumnRepresentations() const
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

void ROOT::RIntegralField<std::uint32_t>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt32Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<uint64_t>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RIntegralField<std::uint64_t>::GetColumnRepresentations() const
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

void ROOT::RIntegralField<std::uint64_t>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt64Field(*this);
}

//------------------------------------------------------------------------------

template class ROOT::RSimpleField<int64_t>;

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RIntegralField<std::int64_t>::GetColumnRepresentations() const
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

void ROOT::RIntegralField<std::int64_t>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt64Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RField<std::string>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64, ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kIndex64, ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kSplitIndex32, ENTupleColumnType::kChar},
                                                  {ENTupleColumnType::kIndex32, ENTupleColumnType::kChar}},
                                                 {});
   return representations;
}

void ROOT::RField<std::string>::GenerateColumns()
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex, char>();
}

void ROOT::RField<std::string>::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex, char>(desc);
}

std::size_t ROOT::RField<std::string>::AppendImpl(const void *from)
{
   auto typedValue = static_cast<const std::string *>(from);
   auto length = typedValue->length();
   fAuxiliaryColumn->AppendV(typedValue->data(), length);
   fIndex += length;
   fPrincipalColumn->Append(&fIndex);
   return length + fPrincipalColumn->GetElement()->GetPackedSize();
}

void ROOT::RField<std::string>::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   auto typedValue = static_cast<std::string *>(to);
   RNTupleLocalIndex collectionStart;
   ROOT::NTupleSize_t nChars;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nChars);
   if (nChars == 0) {
      typedValue->clear();
   } else {
      typedValue->resize(nChars);
      fAuxiliaryColumn->ReadV(collectionStart, nChars, const_cast<char *>(typedValue->data()));
   }
}

void ROOT::RField<std::string>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitStringField(*this);
}

//------------------------------------------------------------------------------

ROOT::RRecordField::RRecordField(std::string_view name, const RRecordField &source)
   : ROOT::RFieldBase(name, source.GetTypeName(), ROOT::ENTupleStructure::kRecord, false /* isSimple */),
     fMaxAlignment(source.fMaxAlignment),
     fSize(source.fSize),
     fOffsets(source.fOffsets)
{
   for (const auto &f : source.GetConstSubfields())
      Attach(f->Clone(f->GetFieldName()));
   fTraits = source.fTraits;
}

ROOT::RRecordField::RRecordField(std::string_view fieldName, std::string_view typeName)
   : ROOT::RFieldBase(fieldName, typeName, ROOT::ENTupleStructure::kRecord, false /* isSimple */)
{
}

void ROOT::RRecordField::AttachItemFields(std::vector<std::unique_ptr<RFieldBase>> itemFields)
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

std::unique_ptr<ROOT::RFieldBase>
ROOT::Internal::CreateEmulatedRecordField(std::string_view fieldName,
                                          std::vector<std::unique_ptr<RFieldBase>> itemFields,
                                          std::string_view emulatedFromType)
{
   R__ASSERT(!emulatedFromType.empty());
   return std::unique_ptr<RFieldBase>(new RRecordField(fieldName, std::move(itemFields), emulatedFromType));
}

std::unique_ptr<ROOT::RFieldBase> ROOT::Internal::CreateEmulatedVectorField(std::string_view fieldName,
                                                                            std::unique_ptr<RFieldBase> itemField,
                                                                            std::string_view emulatedFromType)
{
   R__ASSERT(!emulatedFromType.empty());
   return std::unique_ptr<RFieldBase>(new RVectorField(fieldName, std::move(itemField), emulatedFromType));
}

ROOT::RRecordField::RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields,
                                 std::string_view emulatedFromType)
   : ROOT::RFieldBase(fieldName, emulatedFromType, ROOT::ENTupleStructure::kRecord, false /* isSimple */)
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

ROOT::RRecordField::RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields)
   : ROOT::RRecordField(fieldName, std::move(itemFields), "")
{
}

std::size_t ROOT::RRecordField::GetItemPadding(std::size_t baseOffset, std::size_t itemAlignment) const
{
   if (itemAlignment > 1) {
      auto remainder = baseOffset % itemAlignment;
      if (remainder != 0)
         return itemAlignment - remainder;
   }
   return 0;
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RRecordField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RRecordField>(new RRecordField(newName, *this));
}

std::size_t ROOT::RRecordField::AppendImpl(const void *from)
{
   std::size_t nbytes = 0;
   for (unsigned i = 0; i < fSubfields.size(); ++i) {
      nbytes += CallAppendOn(*fSubfields[i], static_cast<const unsigned char *>(from) + fOffsets[i]);
   }
   return nbytes;
}

void ROOT::RRecordField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   for (unsigned i = 0; i < fSubfields.size(); ++i) {
      CallReadOn(*fSubfields[i], globalIndex, static_cast<unsigned char *>(to) + fOffsets[i]);
   }
}

void ROOT::RRecordField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   for (unsigned i = 0; i < fSubfields.size(); ++i) {
      CallReadOn(*fSubfields[i], localIndex, static_cast<unsigned char *>(to) + fOffsets[i]);
   }
}

void ROOT::RRecordField::ConstructValue(void *where) const
{
   for (unsigned i = 0; i < fSubfields.size(); ++i) {
      CallConstructValueOn(*fSubfields[i], static_cast<unsigned char *>(where) + fOffsets[i]);
   }
}

void ROOT::RRecordField::RRecordDeleter::operator()(void *objPtr, bool dtorOnly)
{
   for (unsigned i = 0; i < fItemDeleters.size(); ++i) {
      fItemDeleters[i]->operator()(reinterpret_cast<unsigned char *>(objPtr) + fOffsets[i], true /* dtorOnly */);
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::RRecordField::GetDeleter() const
{
   std::vector<std::unique_ptr<RDeleter>> itemDeleters;
   itemDeleters.reserve(fOffsets.size());
   for (const auto &f : fSubfields) {
      itemDeleters.emplace_back(GetDeleterOf(*f));
   }
   return std::make_unique<RRecordDeleter>(std::move(itemDeleters), fOffsets);
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RRecordField::SplitValue(const RValue &value) const
{
   auto valuePtr = value.GetPtr<void>();
   auto charPtr = static_cast<unsigned char *>(valuePtr.get());
   std::vector<RValue> result;
   result.reserve(fSubfields.size());
   for (unsigned i = 0; i < fSubfields.size(); ++i) {
      result.emplace_back(fSubfields[i]->BindValue(std::shared_ptr<void>(valuePtr, charPtr + fOffsets[i])));
   }
   return result;
}

void ROOT::RRecordField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitRecordField(*this);
}

//------------------------------------------------------------------------------

ROOT::RBitsetField::RBitsetField(std::string_view fieldName, std::size_t N)
   : ROOT::RFieldBase(fieldName, "std::bitset<" + std::to_string(N) + ">", ROOT::ENTupleStructure::kLeaf,
                      false /* isSimple */, N),
     fN(N)
{
   fTraits |= kTraitTriviallyDestructible;
}

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RBitsetField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kBit}}, {});
   return representations;
}

void ROOT::RBitsetField::GenerateColumns()
{
   GenerateColumnsImpl<bool>();
}

void ROOT::RBitsetField::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<bool>(desc);
}

template <typename FUlong, typename FUlonglong, typename... Args>
void ROOT::RBitsetField::SelectWordSize(FUlong &&fUlong, FUlonglong &&fUlonglong, Args &&...args)
{
   if (WordSize() == sizeof(unsigned long)) {
      fUlong(std::forward<Args>(args)..., fN, *fPrincipalColumn);
   } else if (WordSize() == sizeof(unsigned long long)) {
      // NOTE: this can only happen on Windows; see the comment on the RBitsetField class.
      fUlonglong(std::forward<Args>(args)..., fN, *fPrincipalColumn);
   } else {
      R__ASSERT(false);
   }
}

template <typename Word_t>
static void BitsetAppendImpl(const void *from, size_t nBits, ROOT::Internal::RColumn &column)
{
   constexpr auto kBitsPerWord = sizeof(Word_t) * 8;

   const auto *asWordArray = static_cast<const Word_t *>(from);
   bool elementValue;
   std::size_t i = 0;
   for (std::size_t word = 0; word < (nBits + kBitsPerWord - 1) / kBitsPerWord; ++word) {
      for (std::size_t mask = 0; (mask < kBitsPerWord) && (i < nBits); ++mask, ++i) {
         elementValue = (asWordArray[word] & (static_cast<Word_t>(1) << mask)) != 0;
         column.Append(&elementValue);
      }
   }
}

std::size_t ROOT::RBitsetField::AppendImpl(const void *from)
{
   SelectWordSize(BitsetAppendImpl<unsigned long>, BitsetAppendImpl<unsigned long long>, from);
   return fN;
}

template <typename Word_t>
static void
BitsetReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to, size_t nBits, ROOT::Internal::RColumn &column)
{
   constexpr auto kBitsPerWord = sizeof(Word_t) * 8;

   auto *asWordArray = static_cast<Word_t *>(to);
   bool elementValue;
   for (std::size_t i = 0; i < nBits; ++i) {
      column.Read(globalIndex * nBits + i, &elementValue);
      Word_t mask = static_cast<Word_t>(1) << (i % kBitsPerWord);
      Word_t bit = static_cast<Word_t>(elementValue) << (i % kBitsPerWord);
      asWordArray[i / kBitsPerWord] = (asWordArray[i / kBitsPerWord] & ~mask) | bit;
   }
}

void ROOT::RBitsetField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   SelectWordSize(BitsetReadGlobalImpl<unsigned long>, BitsetReadGlobalImpl<unsigned long long>, globalIndex, to);
}

template <typename Word_t>
static void
BitsetReadInClusterImpl(ROOT::RNTupleLocalIndex localIndex, void *to, size_t nBits, ROOT::Internal::RColumn &column)
{
   constexpr auto kBitsPerWord = sizeof(Word_t) * 8;

   auto *asWordArray = static_cast<Word_t *>(to);
   bool elementValue;
   for (std::size_t i = 0; i < nBits; ++i) {
      column.Read(ROOT::RNTupleLocalIndex(localIndex.GetClusterId(), localIndex.GetIndexInCluster() * nBits) + i,
                  &elementValue);
      Word_t mask = static_cast<Word_t>(1) << (i % kBitsPerWord);
      Word_t bit = static_cast<Word_t>(elementValue) << (i % kBitsPerWord);
      asWordArray[i / kBitsPerWord] = (asWordArray[i / kBitsPerWord] & ~mask) | bit;
   }
}

void ROOT::RBitsetField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   SelectWordSize(BitsetReadInClusterImpl<unsigned long>, BitsetReadInClusterImpl<unsigned long long>, localIndex, to);
}

void ROOT::RBitsetField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitBitsetField(*this);
}

//------------------------------------------------------------------------------

ROOT::RNullableField::RNullableField(std::string_view fieldName, std::string_view typeName,
                                     std::unique_ptr<RFieldBase> itemField)
   : ROOT::RFieldBase(fieldName, typeName, ROOT::ENTupleStructure::kCollection, false /* isSimple */)
{
   Attach(std::move(itemField));
}

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RNullableField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::RNullableField::GenerateColumns()
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>();
}

void ROOT::RNullableField::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(desc);
}

std::size_t ROOT::RNullableField::AppendNull()
{
   fPrincipalColumn->Append(&fNWritten);
   return sizeof(ROOT::Internal::RColumnIndex);
}

std::size_t ROOT::RNullableField::AppendValue(const void *from)
{
   auto nbytesItem = CallAppendOn(*fSubfields[0], from);
   fNWritten++;
   fPrincipalColumn->Append(&fNWritten);
   return sizeof(ROOT::Internal::RColumnIndex) + nbytesItem;
}

ROOT::RNTupleLocalIndex ROOT::RNullableField::GetItemIndex(ROOT::NTupleSize_t globalIndex)
{
   RNTupleLocalIndex collectionStart;
   ROOT::NTupleSize_t collectionSize;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &collectionSize);
   return (collectionSize == 0) ? RNTupleLocalIndex() : collectionStart;
}

void ROOT::RNullableField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitNullableField(*this);
}

//------------------------------------------------------------------------------

ROOT::RUniquePtrField::RUniquePtrField(std::string_view fieldName, std::string_view typeName,
                                       std::unique_ptr<RFieldBase> itemField)
   : RNullableField(fieldName, typeName, std::move(itemField)), fItemDeleter(GetDeleterOf(*fSubfields[0]))
{
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RUniquePtrField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::make_unique<RUniquePtrField>(newName, GetTypeName(), std::move(newItemField));
}

std::size_t ROOT::RUniquePtrField::AppendImpl(const void *from)
{
   auto typedValue = static_cast<const std::unique_ptr<char> *>(from);
   if (*typedValue) {
      return AppendValue(typedValue->get());
   } else {
      return AppendNull();
   }
}

void ROOT::RUniquePtrField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   auto ptr = static_cast<std::unique_ptr<char> *>(to);
   bool isValidValue = static_cast<bool>(*ptr);

   auto itemIndex = GetItemIndex(globalIndex);
   bool isValidItem = itemIndex.GetIndexInCluster() != ROOT::kInvalidNTupleIndex;

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
      valuePtr = CallCreateObjectRawPtrOn(*fSubfields[0]);
      ptr->reset(reinterpret_cast<char *>(valuePtr));
   }

   CallReadOn(*fSubfields[0], itemIndex, valuePtr);
}

void ROOT::RUniquePtrField::RUniquePtrDeleter::operator()(void *objPtr, bool dtorOnly)
{
   auto typedPtr = static_cast<std::unique_ptr<char> *>(objPtr);
   if (*typedPtr) {
      fItemDeleter->operator()(typedPtr->get(), false /* dtorOnly */);
      typedPtr->release();
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::RUniquePtrField::GetDeleter() const
{
   return std::make_unique<RUniquePtrDeleter>(GetDeleterOf(*fSubfields[0]));
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RUniquePtrField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   auto valuePtr = value.GetPtr<void>();
   const auto &uniquePtr = *static_cast<std::unique_ptr<char> *>(valuePtr.get());
   if (uniquePtr) {
      result.emplace_back(fSubfields[0]->BindValue(std::shared_ptr<void>(valuePtr, uniquePtr.get())));
   }
   return result;
}

//------------------------------------------------------------------------------

ROOT::ROptionalField::ROptionalField(std::string_view fieldName, std::string_view typeName,
                                     std::unique_ptr<RFieldBase> itemField)
   : RNullableField(fieldName, typeName, std::move(itemField)), fItemDeleter(GetDeleterOf(*fSubfields[0]))
{
   if (fSubfields[0]->GetTraits() & kTraitTriviallyDestructible)
      fTraits |= kTraitTriviallyDestructible;
}

bool *ROOT::ROptionalField::GetEngagementPtr(void *optionalPtr) const
{
   return reinterpret_cast<bool *>(reinterpret_cast<unsigned char *>(optionalPtr) + fSubfields[0]->GetValueSize());
}

const bool *ROOT::ROptionalField::GetEngagementPtr(const void *optionalPtr) const
{
   return GetEngagementPtr(const_cast<void *>(optionalPtr));
}

std::unique_ptr<ROOT::RFieldBase> ROOT::ROptionalField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::make_unique<ROptionalField>(newName, GetTypeName(), std::move(newItemField));
}

std::size_t ROOT::ROptionalField::AppendImpl(const void *from)
{
   if (*GetEngagementPtr(from)) {
      return AppendValue(from);
   } else {
      return AppendNull();
   }
}

void ROOT::ROptionalField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   auto engagementPtr = GetEngagementPtr(to);
   auto itemIndex = GetItemIndex(globalIndex);
   if (itemIndex.GetIndexInCluster() == ROOT::kInvalidNTupleIndex) {
      if (*engagementPtr && !(fSubfields[0]->GetTraits() & kTraitTriviallyDestructible))
         fItemDeleter->operator()(to, true /* dtorOnly */);
      *engagementPtr = false;
   } else {
      if (!(*engagementPtr) && !(fSubfields[0]->GetTraits() & kTraitTriviallyConstructible))
         CallConstructValueOn(*fSubfields[0], to);
      CallReadOn(*fSubfields[0], itemIndex, to);
      *engagementPtr = true;
   }
}

void ROOT::ROptionalField::ConstructValue(void *where) const
{
   *GetEngagementPtr(where) = false;
}

void ROOT::ROptionalField::ROptionalDeleter::operator()(void *objPtr, bool dtorOnly)
{
   if (fItemDeleter) {
      auto engagementPtr = reinterpret_cast<bool *>(reinterpret_cast<unsigned char *>(objPtr) + fEngagementPtrOffset);
      if (*engagementPtr)
         fItemDeleter->operator()(objPtr, true /* dtorOnly */);
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::ROptionalField::GetDeleter() const
{
   return std::make_unique<ROptionalDeleter>(
      (fSubfields[0]->GetTraits() & kTraitTriviallyDestructible) ? nullptr : GetDeleterOf(*fSubfields[0]),
      fSubfields[0]->GetValueSize());
}

std::vector<ROOT::RFieldBase::RValue> ROOT::ROptionalField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   const auto valuePtr = value.GetPtr<void>().get();
   if (*GetEngagementPtr(valuePtr)) {
      result.emplace_back(fSubfields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), valuePtr)));
   }
   return result;
}

size_t ROOT::ROptionalField::GetValueSize() const
{
   const auto alignment = GetAlignment();
   // real size is the sum of the value size and the engagement boolean
   const auto actualSize = fSubfields[0]->GetValueSize() + sizeof(bool);
   auto padding = 0;
   if (alignment > 1) {
      auto remainder = actualSize % alignment;
      if (remainder != 0)
         padding = alignment - remainder;
   }
   return actualSize + padding;
}

size_t ROOT::ROptionalField::GetAlignment() const
{
   return fSubfields[0]->GetAlignment();
}

//------------------------------------------------------------------------------

ROOT::RAtomicField::RAtomicField(std::string_view fieldName, std::string_view typeName,
                                 std::unique_ptr<RFieldBase> itemField)
   : RFieldBase(fieldName, typeName, ROOT::ENTupleStructure::kLeaf, false /* isSimple */)
{
   if (itemField->GetTraits() & kTraitTriviallyConstructible)
      fTraits |= kTraitTriviallyConstructible;
   if (itemField->GetTraits() & kTraitTriviallyDestructible)
      fTraits |= kTraitTriviallyDestructible;
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RAtomicField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::make_unique<RAtomicField>(newName, GetTypeName(), std::move(newItemField));
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RAtomicField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   result.emplace_back(fSubfields[0]->BindValue(value.GetPtr<void>()));
   return result;
}

void ROOT::RAtomicField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitAtomicField(*this);
}
