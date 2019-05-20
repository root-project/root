/// \file RField.cxx
/// \ingroup Forest ROOT7
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
#include <ROOT/RColumnModel.hxx>
#include <ROOT/REntry.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldValue.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TClass.h>
#include <TCollection.h>
#include <TDataMember.h>
#include <TError.h>
#include <TList.h>

#include <algorithm>
#include <cctype> // for isspace
#include <cstdlib> // for malloc, free
#include <exception>
#include <iostream>
#include <utility>

ROOT::Experimental::Detail::RFieldBase::RFieldBase(
   std::string_view name, std::string_view type, EForestStructure structure, bool isSimple)
   : fName(name), fType(type), fStructure(structure), fIsSimple(isSimple), fParent(nullptr), fPrincipalColumn(nullptr)
{
}

ROOT::Experimental::Detail::RFieldBase::~RFieldBase()
{
}

ROOT::Experimental::Detail::RFieldBase*
ROOT::Experimental::Detail::RFieldBase::Create(const std::string &fieldName, const std::string &typeName)
{
   std::string normalizedType(typeName);
   normalizedType.erase(remove_if(normalizedType.begin(), normalizedType.end(), isspace), normalizedType.end());
   // TODO(jblomer): use a type translation map
   if (normalizedType == "Float_t") normalizedType = "float";
   if (normalizedType == "Double_t") normalizedType = "double";
   if (normalizedType == "Int_t") normalizedType = "std::int32_t";
   if (normalizedType == "int") normalizedType = "std::int32_t";
   if (normalizedType == "unsigned") normalizedType = "std::uint32_t";
   if (normalizedType == "unsigned int") normalizedType = "std::uint32_t";
   if (normalizedType == "UInt_t") normalizedType = "std::uint32_t";
   if (normalizedType == "ULong64_t") normalizedType = "std::uint64_t";
   if (normalizedType == "string") normalizedType = "std::string";
   if (normalizedType.substr(0, 7) == "vector<") normalizedType = "std::" + normalizedType;

   if (normalizedType == "ROOT::Experimental::ClusterSize_t") return new RField<ClusterSize_t>(fieldName);
   if (normalizedType == "std::int32_t") return new RField<std::int32_t>(fieldName);
   if (normalizedType == "std::uint32_t") return new RField<std::uint32_t>(fieldName);
   if (normalizedType == "std::uint64_t") return new RField<std::uint64_t>(fieldName);
   if (normalizedType == "float") return new RField<float>(fieldName);
   if (normalizedType == "double") return new RField<double>(fieldName);
   if (normalizedType == "std::string") return new RField<std::string>(fieldName);
   if (normalizedType.substr(0, 12) == "std::vector<") {
      std::string itemTypeName = normalizedType.substr(12, normalizedType.length() - 13);
      auto itemField = Create(GetCollectionName(fieldName), itemTypeName);
      return new RFieldVector(fieldName, std::unique_ptr<Detail::RFieldBase>(itemField));
   }
   // For the time being, we silently read RVec fields as std::vector
   if (normalizedType.substr(0, 19) == "ROOT::VecOps::RVec<") {
      std::string itemTypeName = normalizedType.substr(19, normalizedType.length() - 20);
      auto itemField = Create(GetCollectionName(fieldName), itemTypeName);
      return new RFieldVector(fieldName, std::unique_ptr<Detail::RFieldBase>(itemField));
   }
   // TODO: create an RFieldCollection?
   if (normalizedType == ":Collection:") return new RField<ClusterSize_t>(fieldName);
   auto cl = TClass::GetClass(normalizedType.c_str());
   if (cl != nullptr) {
      return new RFieldClass(fieldName, normalizedType);
   }
   R__ASSERT(false);
   return nullptr;
}

void ROOT::Experimental::Detail::RFieldBase::DoAppend(const ROOT::Experimental::Detail::RFieldValue& /*value*/) {
   R__ASSERT(false);
}

void ROOT::Experimental::Detail::RFieldBase::DoRead(
   ROOT::Experimental::ForestSize_t /*index*/,
   RFieldValue* /*value*/)
{
   R__ASSERT(false);
}

void ROOT::Experimental::Detail::RFieldBase::DoReadV(
   ROOT::Experimental::ForestSize_t /*index*/,
   ROOT::Experimental::ForestSize_t /*count*/,
   void* /*dst*/)
{
   R__ASSERT(false);
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::Detail::RFieldBase::GenerateValue()
{
   void *where = malloc(GetValueSize());
   R__ASSERT(where != nullptr);
   return GenerateValue(where);
}

void ROOT::Experimental::Detail::RFieldBase::DestroyValue(const RFieldValue &value, bool dtorOnly)
{
   if (!dtorOnly)
      free(value.GetRawPtr());
}

void ROOT::Experimental::Detail::RFieldBase::Attach(
   std::unique_ptr<ROOT::Experimental::Detail::RFieldBase> child)
{
   child->fParent = this;
   fSubFields.emplace_back(std::move(child));
}

std::string ROOT::Experimental::Detail::RFieldBase::GetLeafName(const std::string &fullName)
{
   auto idx = fullName.find_last_of(kCollectionSeparator);
   return (idx == std::string::npos) ? fullName : fullName.substr(idx + 1);
}

std::string ROOT::Experimental::Detail::RFieldBase::GetCollectionName(const std::string &parentName)
{
   std::string result(parentName);
   result.push_back(kCollectionSeparator);
   result.append(GetLeafName(parentName));
   return result;
}

void ROOT::Experimental::Detail::RFieldBase::Flush() const
{
   for (auto& column : fColumns) {
      column->Flush();
   }
}

void ROOT::Experimental::Detail::RFieldBase::ConnectColumns(RPageStorage *pageStorage)
{
   if (fColumns.empty()) DoGenerateColumns();
   for (auto& column : fColumns) {
      if ((fParent != nullptr) && (column->GetOffsetColumn() == nullptr))
         column->SetOffsetColumn(fParent->fPrincipalColumn);
      column->Connect(pageStorage);
   }
}

ROOT::Experimental::Detail::RFieldBase::RIterator ROOT::Experimental::Detail::RFieldBase::begin()
{
   if (fSubFields.empty()) return RIterator(this, -1);
   return RIterator(this->fSubFields[0].get(), 0);
}

ROOT::Experimental::Detail::RFieldBase::RIterator ROOT::Experimental::Detail::RFieldBase::end()
{
   return RIterator(this, -1);
}


//-----------------------------------------------------------------------------


void ROOT::Experimental::Detail::RFieldBase::RIterator::Advance()
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


ROOT::Experimental::Detail::RFieldBase* ROOT::Experimental::RFieldRoot::Clone(std::string_view /*newName*/)
{
   Detail::RFieldBase* result = new RFieldRoot();
   for (auto& f : fSubFields) {
      auto clone = f->Clone(f->GetName());
      result->Attach(std::unique_ptr<RFieldBase>(clone));
   }
   return result;
}


ROOT::Experimental::RForestEntry* ROOT::Experimental::RFieldRoot::GenerateEntry()
{
   auto entry = new RForestEntry();
   for (auto& f : fSubFields) {
      entry->AddValue(f->GenerateValue());
   }
   return entry;
}


//------------------------------------------------------------------------------


void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::DoGenerateColumns()
{
   RColumnModel model(GetName(), EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(model));
   fPrincipalColumn = fColumns[0].get();
}


//------------------------------------------------------------------------------


void ROOT::Experimental::RField<float>::DoGenerateColumns()
{
   RColumnModel model(GetName(), EColumnType::kReal32, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(model));
   fPrincipalColumn = fColumns[0].get();
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<double>::DoGenerateColumns()
{
   RColumnModel model(GetName(), EColumnType::kReal64, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(model));
   fPrincipalColumn = fColumns[0].get();
}


//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::int32_t>::DoGenerateColumns()
{
   RColumnModel model(GetName(), EColumnType::kInt32, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(model));
   fPrincipalColumn = fColumns[0].get();
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::uint32_t>::DoGenerateColumns()
{
   RColumnModel model(GetName(), EColumnType::kInt32, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(model));
   fPrincipalColumn = fColumns[0].get();
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::uint64_t>::DoGenerateColumns()
{
   RColumnModel model(GetName(), EColumnType::kInt64, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(model));
   fPrincipalColumn = fColumns[0].get();
}

//------------------------------------------------------------------------------


void ROOT::Experimental::RField<std::string>::DoGenerateColumns()
{
   RColumnModel modelIndex(GetName(), EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelIndex));

   RColumnModel modelChars(GetCollectionName(GetName()), EColumnType::kByte, false /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelChars));
   fPrincipalColumn = fColumns[0].get();
   fColumns[1]->SetOffsetColumn(fPrincipalColumn);
}

void ROOT::Experimental::RField<std::string>::DoAppend(const ROOT::Experimental::Detail::RFieldValue& value)
{
   auto typedValue = value.Get<std::string>();
   auto length = typedValue->length();
   Detail::RColumnElement<char, EColumnType::kByte> elemChars(const_cast<char*>(typedValue->data()));
   fColumns[1]->AppendV(elemChars, length);
   fIndex += length;
   fColumns[0]->Append(fElemIndex);
}

void ROOT::Experimental::RField<std::string>::DoRead(
   ROOT::Experimental::ForestSize_t index, ROOT::Experimental::Detail::RFieldValue* value)
{
   auto typedValue = value->Get<std::string>();
   ForestSize_t idxStart;
   ClusterSize_t nChars;
   fPrincipalColumn->GetCollectionInfo(index, &idxStart, &nChars);
   typedValue->resize(nChars);
   Detail::RColumnElement<char, EColumnType::kByte> elemChars(const_cast<char*>(typedValue->data()));
   fColumns[1]->ReadV(idxStart, nChars, &elemChars);
}

void ROOT::Experimental::RField<std::string>::CommitCluster()
{
   fIndex = 0;
}


//------------------------------------------------------------------------------


ROOT::Experimental::RFieldClass::RFieldClass(std::string_view fieldName, std::string_view className)
   : ROOT::Experimental::Detail::RFieldBase(fieldName, className, EForestStructure::kRecord, false /* isSimple */)
   , fClass(TClass::GetClass(std::string(className).c_str()))
{
   if (fClass == nullptr) {
      throw std::runtime_error("RField: no I/O support for type " + std::string(className));
   }
   TIter next(fClass->GetListOfDataMembers());
   while (auto dataMember = static_cast<TDataMember *>(next())) {
      //printf("Now looking at %s %s\n", dataMember->GetName(), dataMember->GetFullTypeName());
      auto subField = Detail::RFieldBase::Create(
         GetName() + "." + dataMember->GetName(), dataMember->GetFullTypeName());
      Attach(std::unique_ptr<Detail::RFieldBase>(subField));
   }
}

ROOT::Experimental::Detail::RFieldBase* ROOT::Experimental::RFieldClass::Clone(std::string_view newName)
{
   return new RFieldClass(newName, GetType());
}

void ROOT::Experimental::RFieldClass::DoAppend(const Detail::RFieldValue& value) {
   TIter next(fClass->GetListOfDataMembers());
   unsigned i = 0;
   while (auto dataMember = static_cast<TDataMember *>(next())) {
      auto memberValue = fSubFields[i]->CaptureValue(value.Get<unsigned char>() + dataMember->GetOffset());
      fSubFields[i]->Append(memberValue);
      i++;
   }
}

void ROOT::Experimental::RFieldClass::DoRead(ForestSize_t index, Detail::RFieldValue* value) {
   TIter next(fClass->GetListOfDataMembers());
   unsigned i = 0;
   while (auto dataMember = static_cast<TDataMember *>(next())) {
      auto memberValue = fSubFields[i]->GenerateValue(value->Get<unsigned char>() + dataMember->GetOffset());
      fSubFields[i]->Read(index, &memberValue);
      i++;
   }
}

void ROOT::Experimental::RFieldClass::DoGenerateColumns()
{
}

unsigned int ROOT::Experimental::RFieldClass::GetNColumns() const
{
   return 0;
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RFieldClass::GenerateValue(void* where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, fClass->New(where));
}

void ROOT::Experimental::RFieldClass::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   fClass->Destructor(value.GetRawPtr(), true /* dtorOnly */);
   if (!dtorOnly)
      free(value.GetRawPtr());
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RFieldClass::CaptureValue(void* where)
{
   return Detail::RFieldValue(true /* captureFlat */, this, where);
}

size_t ROOT::Experimental::RFieldClass::GetValueSize() const
{
   return fClass->GetClassSize();
}


//------------------------------------------------------------------------------


ROOT::Experimental::RFieldVector::RFieldVector(
   std::string_view fieldName, std::unique_ptr<Detail::RFieldBase> itemField)
   : ROOT::Experimental::Detail::RFieldBase(
      fieldName, "std::vector<" + itemField->GetType() + ">", EForestStructure::kCollection, false /* isSimple */)
   , fItemSize(itemField->GetValueSize()), fNWritten(0)
{
   Attach(std::move(itemField));
}

ROOT::Experimental::Detail::RFieldBase* ROOT::Experimental::RFieldVector::Clone(std::string_view newName)
{
   auto newItemField = fSubFields[0]->Clone(GetCollectionName(std::string(newName)));
   return new RFieldVector(newName, std::unique_ptr<Detail::RFieldBase>(newItemField));
}

void ROOT::Experimental::RFieldVector::DoAppend(const Detail::RFieldValue& value) {
   auto typedValue = value.Get<std::vector<char>>();
   R__ASSERT((typedValue->size() % fItemSize) == 0);
   auto count = typedValue->size() / fItemSize;
   for (unsigned i = 0; i < count; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(typedValue->data() + (i * fItemSize));
      fSubFields[0]->Append(itemValue);
   }
   Detail::RColumnElement<ClusterSize_t, EColumnType::kIndex> elemIndex(&fNWritten);
   fNWritten += count;
   fColumns[0]->Append(elemIndex);
}

void ROOT::Experimental::RFieldVector::DoRead(ForestSize_t index, Detail::RFieldValue* value) {
   auto typedValue = value->Get<std::vector<char>>();

   ClusterSize_t nItems;
   ForestSize_t idxStart;
   fPrincipalColumn->GetCollectionInfo(index, &idxStart, &nItems);

   typedValue->resize(nItems * fItemSize);
   for (unsigned i = 0; i < nItems; ++i) {
      auto itemValue = fSubFields[0]->GenerateValue(typedValue->data() + (i * fItemSize));
      fSubFields[0]->Read(idxStart + i, &itemValue);
   }
}

void ROOT::Experimental::RFieldVector::DoGenerateColumns()
{
   RColumnModel modelIndex(GetName(), EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelIndex));
   fPrincipalColumn = fColumns[0].get();
}

unsigned int ROOT::Experimental::RFieldVector::GetNColumns() const
{
   return 1;
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RFieldVector::GenerateValue(void* where)
{
   // The memory location can be used as a vector of any type except bool (TODO)
   return Detail::RFieldValue(this, reinterpret_cast<std::vector<char>*>(where));
}

void ROOT::Experimental::RFieldVector::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
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

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RFieldVector::CaptureValue(void* where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

size_t ROOT::Experimental::RFieldVector::GetValueSize() const
{
   return sizeof(std::vector<char>);
}

void ROOT::Experimental::RFieldVector::CommitCluster()
{
   fNWritten = 0;
}


//------------------------------------------------------------------------------


ROOT::Experimental::RFieldCollection::RFieldCollection(
   std::string_view name,
   std::shared_ptr<RCollectionForest> collectionForest,
   std::unique_ptr<RForestModel> collectionModel)
   : RFieldBase(name, ":Collection:", EForestStructure::kCollection, true /* isSimple */)
   , fCollectionForest(collectionForest)
{
   std::string namePrefix(name);
   namePrefix.push_back(kCollectionSeparator);
   for (unsigned i = 0; i < collectionModel->GetRootField()->fSubFields.size(); ++i) {
      auto& subField = collectionModel->GetRootField()->fSubFields[i];
      subField->fName = namePrefix + subField->fName;
      for (auto& grandChild : subField->fSubFields) {
         grandChild->fName = namePrefix + grandChild->fName;
      }
      Attach(std::move(subField));
   }
}


void ROOT::Experimental::RFieldCollection::DoGenerateColumns()
{
   RColumnModel modelIndex(GetName(), EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::make_unique<Detail::RColumn>(modelIndex));
   fPrincipalColumn = fColumns[0].get();
}


ROOT::Experimental::Detail::RFieldBase* ROOT::Experimental::RFieldCollection::Clone(std::string_view /*newName*/)
{
   // TODO(jblomer)
   return nullptr;
   //auto result = new RFieldCollection(newName, fCollectionForest, RForestModel::Create());
   //for (auto& f : fSubFields) {
   //   // switch the name prefix for the new parent name
   //   std::string cloneName = std::string(newName) + f->GetName().substr(GetName().length());
   //   auto clone = f->Clone(cloneName);
   //   result->Attach(std::unique_ptr<RFieldBase>(clone));
   //}
   //return result;
}

void ROOT::Experimental::RFieldCollection::CommitCluster() {
   *fCollectionForest->GetOffsetPtr() = 0;
}

