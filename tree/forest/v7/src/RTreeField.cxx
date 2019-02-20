/// \file RTreeField.cxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RColumn.hxx"
#include "ROOT/RColumnModel.hxx"
#include "ROOT/RTreeField.hxx"
#include "ROOT/RTreeValue.hxx"

#include <TClass.h>
#include <TCollection.h>
#include <TDataMember.h>
#include <TError.h>
#include <TList.h>

#include <algorithm>
#include <cctype> // for isspace
#include <cstdlib> // for malloc, free
#include <exception>
#include <utility>

ROOT::Experimental::Detail::RTreeFieldBase::RTreeFieldBase(std::string_view name, std::string_view type, bool isSimple)
   : fName(name), fType(type), fIsSimple(isSimple), fParent(nullptr), fPrincipalColumn(nullptr)
{
}

ROOT::Experimental::Detail::RTreeFieldBase::~RTreeFieldBase()
{
}

ROOT::Experimental::Detail::RTreeFieldBase*
ROOT::Experimental::Detail::RTreeFieldBase::Create(const std::string &fieldName, const std::string &typeName)
{
   std::string normalizedType(typeName);
   normalizedType.erase(remove_if(normalizedType.begin(), normalizedType.end(), isspace), normalizedType.end());
   if (normalizedType == "string") normalizedType = "std::string";
   if (normalizedType.substr(0, 7) == "vector<") normalizedType = "std::" + normalizedType;

   if (normalizedType == "float") return new RTreeField<float>(fieldName);
   if (normalizedType == "std::string") return new RTreeField<std::string>(fieldName);
   if (normalizedType.substr(0, 12) == "std::vector<") {
      std::string itemTypeName = normalizedType.substr(12, normalizedType.length() - 13);
      auto itemField = Create(GetCollectionName(fieldName), itemTypeName);
      return new RTreeFieldVector(fieldName, std::unique_ptr<Detail::RTreeFieldBase>(itemField));
   }
   auto cl = TClass::GetClass(normalizedType.c_str());
   if (cl != nullptr) {
      return new RTreeFieldClass(fieldName, normalizedType);
   }
   R__ASSERT(false);
   return nullptr;
}

void ROOT::Experimental::Detail::RTreeFieldBase::DoAppend(const ROOT::Experimental::Detail::RTreeValueBase& /*value*/) {
   R__ASSERT(false);
}

void ROOT::Experimental::Detail::RTreeFieldBase::DoRead(
   ROOT::Experimental::TreeIndex_t /*index*/,
   RTreeValueBase* /*value*/)
{
   R__ASSERT(false);
}

void ROOT::Experimental::Detail::RTreeFieldBase::DoReadV(
   ROOT::Experimental::TreeIndex_t /*index*/,
   ROOT::Experimental::TreeIndex_t /*count*/,
   void* /*dst*/)
{
   R__ASSERT(false);
}

ROOT::Experimental::Detail::RTreeValueBase ROOT::Experimental::Detail::RTreeFieldBase::GenerateValue()
{
   void *where = malloc(GetValueSize());
   R__ASSERT(where != nullptr);
   return GenerateValue(where);
}

void ROOT::Experimental::Detail::RTreeFieldBase::DestroyValue(const RTreeValueBase &value, bool dtorOnly)
{
   if (!dtorOnly)
      free(value.GetRawPtr());
}

void ROOT::Experimental::Detail::RTreeFieldBase::Attach(
   std::unique_ptr<ROOT::Experimental::Detail::RTreeFieldBase> child)
{
   child->fParent = this;
   fSubFields.emplace_back(std::move(child));
}

std::string ROOT::Experimental::Detail::RTreeFieldBase::GetLeafName(const std::string &fullName)
{
   auto idx = fullName.find_last_of(kCollectionSeparator);
   return (idx == std::string::npos) ? fullName : fullName.substr(idx + 1);
}

std::string ROOT::Experimental::Detail::RTreeFieldBase::GetCollectionName(const std::string &parentName)
{
   std::string result(parentName);
   result.push_back(kCollectionSeparator);
   result.append(GetLeafName(parentName));
   return result;
}

void ROOT::Experimental::Detail::RTreeFieldBase::Flush() const
{
   for (auto& column : fColumns) {
      column->Flush();
   }
}

void ROOT::Experimental::Detail::RTreeFieldBase::ConnectColumns(RPageStorage *pageStorage)
{
   if (fColumns.empty()) DoGenerateColumns();
   for (auto& column : fColumns) {
      column->Connect(pageStorage);
   }
}

ROOT::Experimental::Detail::RTreeFieldBase::RIterator ROOT::Experimental::Detail::RTreeFieldBase::begin()
{
   if (fSubFields.empty()) return RIterator(this, -1);
   return RIterator(this->fSubFields[0].get(), 0);
}

ROOT::Experimental::Detail::RTreeFieldBase::RIterator ROOT::Experimental::Detail::RTreeFieldBase::end()
{
   return RIterator(this, -1);
}


//-----------------------------------------------------------------------------


void ROOT::Experimental::Detail::RTreeFieldBase::RIterator::Advance()
{
   auto itr = fStack.rbegin();
   if (!itr->fFieldPtr->fSubFields.empty()) {
      fStack.emplace_back(Position(itr->fFieldPtr->fSubFields[0].get(), 0));
      return;
   }

   unsigned int nextIdxInParent = ++(itr->fIdxInParent);
   while (nextIdxInParent >= itr->fFieldPtr->fParent->fSubFields.size()) {
      if (fStack.size() == 1) {
         itr->fFieldPtr = itr->fFieldPtr->fParent;
         itr->fIdxInParent = -1;
         return;
      }
      fStack.pop_back();
      itr = fStack.rbegin();
      nextIdxInParent = ++(itr->fIdxInParent);
   }
   itr->fFieldPtr = itr->fFieldPtr->fParent->fSubFields[nextIdxInParent].get();
}


//------------------------------------------------------------------------------


void ROOT::Experimental::RTreeField<float>::DoGenerateColumns()
{
   RColumnModel model(GetName(), EColumnType::kReal32, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(model));
   fPrincipalColumn = fColumns[0].get();
}


//------------------------------------------------------------------------------


void ROOT::Experimental::RTreeField<std::string>::DoGenerateColumns()
{
   RColumnModel modelIndex(GetName(), EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelIndex));

   RColumnModel modelChars(GetCollectionName(GetName()), EColumnType::kByte, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelChars));
   fPrincipalColumn = fColumns[0].get();
}

void ROOT::Experimental::RTreeField<std::string>::DoAppend(const ROOT::Experimental::Detail::RTreeValueBase& value)
{
   auto typedValue = reinterpret_cast<const ROOT::Experimental::RTreeValue<std::string>&>(value).Get();
   auto length = typedValue->length();
   Detail::RColumnElement<char, EColumnType::kByte> elemChars(const_cast<char*>(typedValue->data()));
   fColumns[1]->AppendV(elemChars, length);
   fIndex += length;
   fColumns[0]->Append(fElemIndex);
}

void ROOT::Experimental::RTreeField<std::string>::DoRead(
   ROOT::Experimental::TreeIndex_t index, ROOT::Experimental::Detail::RTreeValueBase* value)
{
   auto typedValue = reinterpret_cast<ROOT::Experimental::RTreeValue<std::string>*>(value)->Get();
   auto idxStart = (index == 0) ? 0 : *fColumns[0]->Map<TreeIndex_t, EColumnType::kIndex>(index - 1, &fElemIndex);
   auto idxEnd = *fColumns[0]->Map<TreeIndex_t, EColumnType::kIndex>(index, &fElemIndex);
   auto nChars = idxEnd - idxStart;
   typedValue->resize(nChars);
   Detail::RColumnElement<char, EColumnType::kByte> elemChars(const_cast<char*>(typedValue->data()));
   fColumns[1]->ReadV(idxStart, nChars, &elemChars);
}


//------------------------------------------------------------------------------


ROOT::Experimental::RTreeFieldClass::RTreeFieldClass(std::string_view fieldName, std::string_view className)
   : ROOT::Experimental::Detail::RTreeFieldBase(fieldName, className, false /* isSimple */)
   , fClass(TClass::GetClass(className.to_string().c_str()))
{
   if (fClass == nullptr) {
      throw std::runtime_error("RTreeField: no I/O support for type " + className.to_string());
   }
   TIter next(fClass->GetListOfDataMembers());
   while (auto dataMember = static_cast<TDataMember *>(next())) {
      printf("Now looking at %s %s\n", dataMember->GetName(), dataMember->GetFullTypeName());
      auto subField = Detail::RTreeFieldBase::Create(
         GetName() + "." + dataMember->GetName(), dataMember->GetFullTypeName());
      Attach(std::unique_ptr<Detail::RTreeFieldBase>(subField));
   }
}

void ROOT::Experimental::RTreeFieldClass::DoAppend(const Detail::RTreeValueBase& value) {
   TIter next(fClass->GetListOfDataMembers());
   unsigned i = 0;
   while (auto dataMember = static_cast<TDataMember *>(next())) {
      auto memberValue = fSubFields[i]->CaptureValue(
         reinterpret_cast<unsigned char*>(value.GetRawPtr()) + dataMember->GetOffset());
      fSubFields[i]->Append(memberValue);
      i++;
   }
}

void ROOT::Experimental::RTreeFieldClass::DoRead(TreeIndex_t index, Detail::RTreeValueBase* value) {
   TIter next(fClass->GetListOfDataMembers());
   unsigned i = 0;
   while (auto dataMember = static_cast<TDataMember *>(next())) {
      auto memberValue = fSubFields[i]->GenerateValue(
         reinterpret_cast<unsigned char*>(value->GetRawPtr()) + dataMember->GetOffset());
      fSubFields[i]->Read(index, &memberValue);
      i++;
   }
}

void ROOT::Experimental::RTreeFieldClass::DoGenerateColumns()
{
}

unsigned int ROOT::Experimental::RTreeFieldClass::GetNColumns() const
{
   return 0;
}

ROOT::Experimental::Detail::RTreeValueBase ROOT::Experimental::RTreeFieldClass::GenerateValue(void* where)
{
   return Detail::RTreeValueBase(this, fClass->New(where));
}

void ROOT::Experimental::RTreeFieldClass::DestroyValue(const Detail::RTreeValueBase& value, bool dtorOnly)
{
   fClass->Destructor(value.GetRawPtr(), true /* dtorOnly */);
   if (!dtorOnly)
      free(value.GetRawPtr());
}

ROOT::Experimental::Detail::RTreeValueBase ROOT::Experimental::RTreeFieldClass::CaptureValue(void* where)
{
   return Detail::RTreeValueBase(this, where);
}

size_t ROOT::Experimental::RTreeFieldClass::GetValueSize() const
{
   return fClass->GetClassSize();
}


//------------------------------------------------------------------------------


ROOT::Experimental::RTreeFieldVector::RTreeFieldVector(
   std::string_view fieldName, std::unique_ptr<Detail::RTreeFieldBase> itemField)
   : ROOT::Experimental::Detail::RTreeFieldBase(
      fieldName, "std::vector<" + itemField->GetType() + ">", false /* isSimple */)
   , fItemSize(itemField->GetValueSize()), fNWritten(0)
{
   Attach(std::move(itemField));
}

void ROOT::Experimental::RTreeFieldVector::DoAppend(const Detail::RTreeValueBase& value) {
   auto typedValue = reinterpret_cast<const RTreeValue<std::vector<char>>&>(value).Get();
   R__ASSERT((typedValue->size() % fItemSize) == 0);
   auto count = typedValue->size() / fItemSize;
   for (unsigned i = 0; i < count; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(typedValue->data() + (i * fItemSize));
      fSubFields[0]->Append(itemValue);
   }
   Detail::RColumnElement<TreeIndex_t, EColumnType::kIndex> elemIndex(&fNWritten);
   fNWritten += count;
   fColumns[0]->Append(elemIndex);
}

void ROOT::Experimental::RTreeFieldVector::DoRead(TreeIndex_t index, Detail::RTreeValueBase* value) {
   auto typedValue = reinterpret_cast<RTreeValue<std::vector<char>>*>(value)->Get();

   TreeIndex_t dummy;
   Detail::RColumnElement<TreeIndex_t, EColumnType::kIndex> elemIndex(&dummy);
   auto idxStart = (index == 0) ? 0
      : *fColumns[0]->template Map<TreeIndex_t, EColumnType::kIndex>(index - 1, &elemIndex);
   auto idxEnd = *fColumns[0]->template Map<TreeIndex_t, EColumnType::kIndex>(index, &elemIndex);
   auto nItems = idxEnd - idxStart;

   typedValue->resize(nItems * fItemSize);
   for (unsigned i = 0; i < nItems; ++i) {
      auto itemValue = fSubFields[0]->GenerateValue(typedValue->data() + (i * fItemSize));
      fSubFields[0]->Read(idxStart + i, &itemValue);
   }
}

void ROOT::Experimental::RTreeFieldVector::DoGenerateColumns()
{
   RColumnModel modelIndex(GetName(), EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelIndex));
   fPrincipalColumn = fColumns[0].get();
}

unsigned int ROOT::Experimental::RTreeFieldVector::GetNColumns() const
{
   return 1;
}

ROOT::Experimental::Detail::RTreeValueBase ROOT::Experimental::RTreeFieldVector::GenerateValue(void* where)
{
   // The memory location can be used as a vector of any type except bool (TODO)
   return Detail::RTreeValueBase(this, new (where) std::vector<char>());
}

void ROOT::Experimental::RTreeFieldVector::DestroyValue(const Detail::RTreeValueBase& value, bool dtorOnly)
{
   auto vec = static_cast<std::vector<char>*>(value.GetRawPtr());
   R__ASSERT((vec->size() % fItemSize) == 0);
   auto nItems = vec->size() / fItemSize;
   for (unsigned i = 0; i < nItems; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(vec->data() + (i * fItemSize));
      fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
   }
   vec->~vector();
   if (!dtorOnly)
      free(vec);
}

ROOT::Experimental::Detail::RTreeValueBase ROOT::Experimental::RTreeFieldVector::CaptureValue(void* where)
{
   return Detail::RTreeValueBase(this, where);
}

size_t ROOT::Experimental::RTreeFieldVector::GetValueSize() const
{
   return sizeof(std::vector<char>);
}

