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
#include <ROOT/RColumnModel.hxx>
#include <ROOT/REntry.hxx>
#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldValue.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TBaseClass.h>
#include <TClass.h>
#include <TClassEdit.h>
#include <TCollection.h>
#include <TDataMember.h>
#include <TError.h>
#include <TList.h>

#include <algorithm>
#include <cctype> // for isspace
#include <cstdint>
#include <cstdlib> // for malloc, free
#include <cstring> // for memset
#include <exception>
#include <iostream>
#include <new> // hardware_destructive_interference_size
#include <type_traits>
#include <unordered_map>

namespace {

static const std::unordered_map<std::string_view, std::string_view> typeTranslationMap{
   {"Bool_t",   "bool"},
   {"Float_t",  "float"},
   {"Double_t", "double"},
   {"string",   "std::string"},

   {"Char_t",        "char"},
   {"int8_t",        "std::int8_t"},
   {"signed char",   "char"},
   {"UChar_t",       "std::uint8_t"},
   {"unsigned char", "std::uint8_t"},
   {"uint8_t",       "std::uint8_t"},

   {"Short_t",        "std::int16_t"},
   {"int16_t",        "std::int16_t"},
   {"short",          "std::int16_t"},
   {"UShort_t",       "std::uint16_t"},
   {"unsigned short", "std::uint16_t"},
   {"uint16_t",       "std::uint16_t"},

   {"Int_t",        "std::int32_t"},
   {"int32_t",      "std::int32_t"},
   {"int",          "std::int32_t"},
   {"UInt_t",       "std::uint32_t"},
   {"unsigned",     "std::uint32_t"},
   {"unsigned int", "std::uint32_t"},
   {"uint32_t",     "std::uint32_t"},

   {"Long_t",        "std::int64_t"},
   {"Long64_t",      "std::int64_t"},
   {"int64_t",       "std::int64_t"},
   {"long",          "std::int64_t"},
   {"ULong64_t",     "std::uint64_t"},
   {"unsigned long", "std::uint64_t"},
   {"uint64_t",      "std::uint64_t"}
};

/// Used in CreateField() in order to get the comma-separated list of template types
/// E.g., gets {"int", "std::variant<double,int>"} from "int,std::variant<double,int>"
std::vector<std::string> TokenizeTypeList(std::string templateType) {
   std::vector<std::string> result;
   if (templateType.empty())
      return result;

   const char *eol = templateType.data() + templateType.length();
   const char *typeBegin = templateType.data();
   const char *typeCursor = templateType.data();
   unsigned int nestingLevel = 0;
   while (typeCursor != eol) {
      switch (*typeCursor) {
      case '<':
         ++nestingLevel;
         break;
      case '>':
         --nestingLevel;
         break;
      case ',':
         if (nestingLevel == 0) {
            result.push_back(std::string(typeBegin, typeCursor - typeBegin));
            typeBegin = typeCursor + 1;
         }
         break;
      }
      typeCursor++;
   }
   result.push_back(std::string(typeBegin, typeCursor - typeBegin));
   return result;
}

std::string GetNormalizedType(const std::string &typeName) {
   std::string normalizedType(
      TClassEdit::ResolveTypedef(TClassEdit::CleanType(typeName.c_str(),
                                                       /*mode=*/2).c_str()));

   auto translatedType = typeTranslationMap.find(normalizedType);
   if (translatedType != typeTranslationMap.end())
      normalizedType = translatedType->second;

   if (normalizedType.substr(0, 7) == "vector<") normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 6) == "array<") normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 8) == "variant<") normalizedType = "std::" + normalizedType;

   return normalizedType;
}

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

} // anonymous namespace


//------------------------------------------------------------------------------


ROOT::Experimental::Detail::RFieldBase::RFieldBase(
   std::string_view name, std::string_view type, ENTupleStructure structure, bool isSimple, std::size_t nRepetitions)
   : fName(name), fType(type), fStructure(structure), fNRepetitions(nRepetitions), fIsSimple(isSimple),
     fParent(nullptr), fPrincipalColumn(nullptr)
{
}

ROOT::Experimental::Detail::RFieldBase::~RFieldBase()
{
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>>
ROOT::Experimental::Detail::RFieldBase::Create(const std::string &fieldName, const std::string &typeName)
{
   std::string normalizedType(GetNormalizedType(typeName));
   if (normalizedType.empty())
      return R__FAIL("no type name specified for Field " + fieldName);

   std::unique_ptr<ROOT::Experimental::Detail::RFieldBase> result;

   if (normalizedType == "ROOT::Experimental::ClusterSize_t") {
      result = std::make_unique<RField<ClusterSize_t>>(fieldName);
   } else if (normalizedType == "bool") {
      result = std::make_unique<RField<bool>>(fieldName);
   } else if (normalizedType == "char") {
      result = std::make_unique<RField<char>>(fieldName);
   } else if (normalizedType == "std::int8_t") {
      result = std::make_unique<RField<std::int8_t>>(fieldName);
   } else if (normalizedType == "std::uint8_t") {
      result = std::make_unique<RField<std::uint8_t>>(fieldName);
   } else if (normalizedType == "std::int16_t") {
      result = std::make_unique<RField<std::int16_t>>(fieldName);
   } else if (normalizedType == "std::uint16_t") {
      result = std::make_unique<RField<std::uint16_t>>(fieldName);
   } else if (normalizedType == "std::int32_t") {
      result = std::make_unique<RField<std::int32_t>>(fieldName);
   } else if (normalizedType == "std::uint32_t") {
      result = std::make_unique<RField<std::uint32_t>>(fieldName);
   } else if (normalizedType == "std::int64_t") {
      result = std::make_unique<RField<std::int64_t>>(fieldName);
   } else if (normalizedType == "std::uint64_t") {
      result = std::make_unique<RField<std::uint64_t>>(fieldName);
   } else if (normalizedType == "float") {
      result = std::make_unique<RField<float>>(fieldName);
   } else if (normalizedType == "double") {
      result = std::make_unique<RField<double>>(fieldName);
   } else if (normalizedType == "std::string") {
      result = std::make_unique<RField<std::string>>(fieldName);
   } else if (normalizedType == "std::vector<bool>") {
      result = std::make_unique<RField<std::vector<bool>>>(fieldName);
   } else if (normalizedType.substr(0, 12) == "std::vector<") {
      std::string itemTypeName = normalizedType.substr(12, normalizedType.length() - 13);
      auto itemField = Create("_0", itemTypeName);
      result = std::make_unique<RVectorField>(fieldName, itemField.Unwrap());
   } else if (normalizedType.substr(0, 19) == "ROOT::VecOps::RVec<") {
      std::string itemTypeName = normalizedType.substr(19, normalizedType.length() - 20);
      auto itemField = Create("_0", itemTypeName);
      result = std::make_unique<RRVecField>(fieldName, itemField.Unwrap());
   } else if (normalizedType.substr(0, 11) == "std::array<") {
      auto arrayDef = TokenizeTypeList(normalizedType.substr(11, normalizedType.length() - 12));
      R__ASSERT(arrayDef.size() == 2);
      auto arrayLength = std::stoi(arrayDef[1]);
      auto itemField = Create(GetNormalizedType(arrayDef[0]), arrayDef[0]);
      result = std::make_unique<RArrayField>(fieldName, itemField.Unwrap(), arrayLength);
   }
   if (normalizedType.substr(0, 13) == "std::variant<") {
      auto innerTypes = TokenizeTypeList(normalizedType.substr(13, normalizedType.length() - 14));
      std::vector<RFieldBase *> items;
      for (unsigned int i = 0; i < innerTypes.size(); ++i) {
         items.emplace_back(Create("_" + std::to_string(i), innerTypes[i]).Unwrap().release());
      }
      result = std::make_unique<RVariantField>(fieldName, items);
   }
   // TODO: create an RCollectionField?
   if (normalizedType == ":Collection:")
     result = std::make_unique<RField<ClusterSize_t>>(fieldName);

   if (!result) {
      auto cl = TClass::GetClass(normalizedType.c_str());
      if (cl != nullptr) {
         result = std::make_unique<RClassField>(fieldName, normalizedType);
      }
   }

   if (result)
      return result;
   return R__FAIL(std::string("Field ") + fieldName + " has unknown type " + normalizedType);
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::Detail::RFieldBase::EnsureValidFieldName(std::string_view fieldName)
{
   if (fieldName == "") {
      return R__FAIL("name cannot be empty string \"\"");
   } else if (fieldName.find(".") != std::string::npos) {
      return R__FAIL("name '" + std::string(fieldName) + "' cannot contain dot characters '.'");
   }
   return RResult<void>::Success();
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::Detail::RFieldBase::Clone(std::string_view newName) const
{
   auto clone = CloneImpl(newName);
   clone->fOnDiskId = fOnDiskId;
   clone->fDescription = fDescription;
   return clone;
}

std::size_t ROOT::Experimental::Detail::RFieldBase::AppendImpl(const ROOT::Experimental::Detail::RFieldValue& /*value*/)
{
   R__ASSERT(false && "A non-simple RField must implement its own AppendImpl");
   return 0;
}

void ROOT::Experimental::Detail::RFieldBase::ReadGlobalImpl(
   ROOT::Experimental::NTupleSize_t /*index*/,
   RFieldValue* /*value*/)
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

std::vector<ROOT::Experimental::Detail::RFieldValue>
ROOT::Experimental::Detail::RFieldBase::SplitValue(const RFieldValue & /*value*/) const
{
   return std::vector<RFieldValue>();
}

void ROOT::Experimental::Detail::RFieldBase::Attach(
   std::unique_ptr<ROOT::Experimental::Detail::RFieldBase> child)
{
   child->fParent = this;
   fSubFields.emplace_back(std::move(child));
}


std::vector<ROOT::Experimental::Detail::RFieldBase *> ROOT::Experimental::Detail::RFieldBase::GetSubFields() const
{
   std::vector<RFieldBase *> result;
   for (const auto &f : fSubFields) {
      result.emplace_back(f.get());
   }
   return result;
}


void ROOT::Experimental::Detail::RFieldBase::Flush() const
{
   for (auto& column : fColumns) {
      column->Flush();
   }
}


ROOT::Experimental::EColumnType ROOT::Experimental::Detail::RFieldBase::EnsureColumnType(
   const std::vector<EColumnType> &requestedTypes, unsigned int columnIndex, const RNTupleDescriptor &desc)
{
   R__ASSERT(!requestedTypes.empty());
   auto columnId = desc.FindColumnId(fOnDiskId, columnIndex);
   if (columnId == kInvalidDescriptorId) {
      throw RException(R__FAIL("Column missing: column #" + std::to_string(columnIndex) +
                               " for field " + fName));
   }

   const auto &columnDesc = desc.GetColumnDescriptor(columnId);
   for (auto type : requestedTypes) {
      if (type == columnDesc.GetModel().GetType())
         return type;
   }
   throw RException(R__FAIL(
      "On-disk type `" + RColumnElementBase::GetTypeName(columnDesc.GetModel().GetType()) +
         "` of column #" + std::to_string(columnIndex) + " for field `" + fName +
         "` is not convertible to the requested type" + [&]{
            std::string typeStr = requestedTypes.size() > 1 ? "s " : " ";
            for (std::size_t i = 0; i < requestedTypes.size(); i++) {
               typeStr += "`" + RColumnElementBase::GetTypeName(requestedTypes[i]) + "`";
               if (i != requestedTypes.size() - 1) {
                  typeStr += ", ";
               }
            }
            return typeStr;
         }()
   ));
   return columnDesc.GetModel().GetType();
}


void ROOT::Experimental::Detail::RFieldBase::ConnectPageSink(RPageSink &pageSink)
{
   R__ASSERT(fColumns.empty());
   GenerateColumnsImpl();
   if (!fColumns.empty())
      fPrincipalColumn = fColumns[0].get();
   for (auto& column : fColumns)
      column->Connect(fOnDiskId, &pageSink);
}


void ROOT::Experimental::Detail::RFieldBase::ConnectPageSource(RPageSource &pageSource)
{
   R__ASSERT(fColumns.empty());
   {
      const auto descriptorGuard = pageSource.GetSharedDescriptorGuard();
      GenerateColumnsImpl(descriptorGuard.GetRef());
   }
   if (!fColumns.empty())
      fPrincipalColumn = fColumns[0].get();
   for (auto& column : fColumns)
      column->Connect(fOnDiskId, &pageSource);
}


void ROOT::Experimental::Detail::RFieldBase::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitField(*this);
}


ROOT::Experimental::Detail::RFieldBase::RSchemaIterator ROOT::Experimental::Detail::RFieldBase::begin()
{
   if (fSubFields.empty()) return RSchemaIterator(this, -1);
   return RSchemaIterator(this->fSubFields[0].get(), 0);
}


ROOT::Experimental::Detail::RFieldBase::RSchemaIterator ROOT::Experimental::Detail::RFieldBase::end()
{
   return RSchemaIterator(this, -1);
}


//-----------------------------------------------------------------------------


void ROOT::Experimental::Detail::RFieldBase::RSchemaIterator::Advance()
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


std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RFieldZero::CloneImpl(std::string_view /*newName*/) const
{
   auto result = std::make_unique<RFieldZero>();
   for (auto &f : fSubFields)
      result->Attach(f->Clone(f->GetName()));
   return result;
}


void ROOT::Experimental::RFieldZero::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitFieldZero(*this);
}


//------------------------------------------------------------------------------


void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<ClusterSize_t, EColumnType::kIndex>(model, 0)));
}

void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kIndex}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitClusterSizeField(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<char>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kChar, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(Detail::RColumn::Create<
      char, EColumnType::kChar>(model, 0)));
}

void ROOT::Experimental::RField<char>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kChar}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<char>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitCharField(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::int8_t>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kInt8, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(Detail::RColumn::Create<
      std::int8_t, EColumnType::kInt8>(model, 0)));
}

void ROOT::Experimental::RField<std::int8_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kInt8}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<std::int8_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt8Field(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::uint8_t>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kInt8, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(Detail::RColumn::Create<
      std::uint8_t, EColumnType::kInt8>(model, 0)));
}

void ROOT::Experimental::RField<std::uint8_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kInt8}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<std::uint8_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt8Field(*this);
}

//------------------------------------------------------------------------------


void ROOT::Experimental::RField<bool>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kBit, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<bool, EColumnType::kBit>(model, 0)));
}

void ROOT::Experimental::RField<bool>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kBit}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<bool>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitBoolField(*this);
}

//------------------------------------------------------------------------------


void ROOT::Experimental::RField<float>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kReal32, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<float, EColumnType::kReal32>(model, 0)));
}

void ROOT::Experimental::RField<float>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kReal32}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<float>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitFloatField(*this);
}


//------------------------------------------------------------------------------

void ROOT::Experimental::RField<double>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kReal64, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<double, EColumnType::kReal64>(model, 0)));
}

void ROOT::Experimental::RField<double>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kReal64}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<double>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitDoubleField(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::int16_t>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kInt16, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(Detail::RColumn::Create<
      std::int16_t, EColumnType::kInt16>(model, 0)));
}

void ROOT::Experimental::RField<std::int16_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kInt16}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<std::int16_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt16Field(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::uint16_t>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kInt16, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(Detail::RColumn::Create<
      std::uint16_t, EColumnType::kInt16>(model, 0)));
}

void ROOT::Experimental::RField<std::uint16_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kInt16}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<std::uint16_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt16Field(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::int32_t>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kInt32, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(Detail::RColumn::Create<
      std::int32_t, EColumnType::kInt32>(model, 0)));
}

void ROOT::Experimental::RField<std::int32_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kInt32}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<std::int32_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitIntField(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::uint32_t>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kInt32, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<std::uint32_t, EColumnType::kInt32>(model, 0)));
}

void ROOT::Experimental::RField<std::uint32_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kInt32}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<std::uint32_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt32Field(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::uint64_t>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kInt64, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<std::uint64_t, EColumnType::kInt64>(model, 0)));
}

void ROOT::Experimental::RField<std::uint64_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kInt64}, 0, desc);
   GenerateColumnsImpl();
}

void ROOT::Experimental::RField<std::uint64_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt64Field(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::int64_t>::GenerateColumnsImpl()
{
   RColumnModel model(EColumnType::kInt64, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<std::int64_t, EColumnType::kInt64>(model, 0)));
}

void ROOT::Experimental::RField<std::int64_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto type = EnsureColumnType({EColumnType::kInt64, EColumnType::kInt32}, 0, desc);
   RColumnModel model(type, false /* isSorted*/);
   if (type == EColumnType::kInt64) {
      fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
         Detail::RColumn::Create<std::int64_t, EColumnType::kInt64>(model, 0)));
   } else {
      fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
         Detail::RColumn::Create<std::int64_t, EColumnType::kInt32>(model, 0)));
   }
}

void ROOT::Experimental::RField<std::int64_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt64Field(*this);
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RField<std::string>::GenerateColumnsImpl()
{
   RColumnModel modelIndex(EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<ClusterSize_t, EColumnType::kIndex>(modelIndex, 0)));

   RColumnModel modelChars(EColumnType::kChar, false /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<char, EColumnType::kChar>(modelChars, 1)));
}

void ROOT::Experimental::RField<std::string>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kIndex}, 0, desc);
   EnsureColumnType({EColumnType::kChar}, 1, desc);
   GenerateColumnsImpl();
}

std::size_t ROOT::Experimental::RField<std::string>::AppendImpl(const ROOT::Experimental::Detail::RFieldValue& value)
{
   auto typedValue = value.Get<std::string>();
   auto length = typedValue->length();
   Detail::RColumnElement<char> elemChars(const_cast<char*>(typedValue->data()));
   fColumns[1]->AppendV(elemChars, length);
   fIndex += length;
   fColumns[0]->Append(fElemIndex);
   return length + sizeof(fElemIndex);
}

void ROOT::Experimental::RField<std::string>::ReadGlobalImpl(
   ROOT::Experimental::NTupleSize_t globalIndex, ROOT::Experimental::Detail::RFieldValue *value)
{
   auto typedValue = value->Get<std::string>();
   RClusterIndex collectionStart;
   ClusterSize_t nChars;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nChars);
   if (nChars == 0) {
      typedValue->clear();
   } else {
      typedValue->resize(nChars);
      Detail::RColumnElement<char> elemChars(const_cast<char*>(typedValue->data()));
      fColumns[1]->ReadV(collectionStart, nChars, &elemChars);
   }
}

void ROOT::Experimental::RField<std::string>::CommitCluster()
{
   fIndex = 0;
}

void ROOT::Experimental::RField<std::string>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitStringField(*this);
}

//------------------------------------------------------------------------------


ROOT::Experimental::RClassField::RClassField(std::string_view fieldName, std::string_view className)
   : RClassField(fieldName, className, TClass::GetClass(std::string(className).c_str()))
{
}

ROOT::Experimental::RClassField::RClassField(std::string_view fieldName, std::string_view className, TClass *classp)
   : ROOT::Experimental::Detail::RFieldBase(fieldName, className, ENTupleStructure::kRecord, false /* isSimple */)
   , fClass(classp)
{
   if (fClass == nullptr) {
      throw RException(R__FAIL("RField: no I/O support for type " + std::string(className)));
   }
   // Avoid accidentally supporting std types through TClass.
   if (fClass->Property() & kIsDefinedInStd) {
      throw RException(R__FAIL(std::string(className) + " is not supported"));
   }

   int i = 0;
   for (auto baseClass : ROOT::Detail::TRangeStaticCast<TBaseClass>(*fClass->GetListOfBases())) {
      TClass *c = baseClass->GetClassPointer();
      auto subField = Detail::RFieldBase::Create(std::string(kPrefixInherited) + "_" + std::to_string(i),
                                                 c->GetName()).Unwrap();
      Attach(std::move(subField),
	     RSubFieldInfo{kBaseClass, static_cast<std::size_t>(baseClass->GetDelta())});
      i++;
   }
   for (auto dataMember : ROOT::Detail::TRangeStaticCast<TDataMember>(*fClass->GetListOfDataMembers())) {
      // Skip members explicitly marked as transient by user comment
      if (!dataMember->IsPersistent())
         continue;
      // Skip, for instance, unscoped enum constants defined in the class
      if (dataMember->Property() & kIsStatic)
         continue;
      auto subField = Detail::RFieldBase::Create(dataMember->GetName(), dataMember->GetFullTypeName()).Unwrap();
      Attach(std::move(subField),
	     RSubFieldInfo{kDataMember, static_cast<std::size_t>(dataMember->GetOffset())});
   }
}

void ROOT::Experimental::RClassField::Attach(std::unique_ptr<Detail::RFieldBase> child, RSubFieldInfo info)
{
   fMaxAlignment = std::max(fMaxAlignment, child->GetAlignment());
   fSubFieldsInfo.push_back(info);
   RFieldBase::Attach(std::move(child));
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RClassField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RClassField>(new RClassField(newName, GetType(), fClass));
}

std::size_t ROOT::Experimental::RClassField::AppendImpl(const Detail::RFieldValue& value) {
   std::size_t nbytes = 0;
   for (unsigned i = 0; i < fSubFields.size(); i++) {
      auto memberValue = fSubFields[i]->CaptureValue(value.Get<unsigned char>() + fSubFieldsInfo[i].fOffset);
      nbytes += fSubFields[i]->Append(memberValue);
   }
   return nbytes;
}

void ROOT::Experimental::RClassField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   for (unsigned i = 0; i < fSubFields.size(); i++) {
      auto memberValue = fSubFields[i]->CaptureValue(value->Get<unsigned char>() + fSubFieldsInfo[i].fOffset);
      fSubFields[i]->Read(globalIndex, &memberValue);
   }
}

void ROOT::Experimental::RClassField::ReadInClusterImpl(const RClusterIndex &clusterIndex, Detail::RFieldValue *value)
{
   for (unsigned i = 0; i < fSubFields.size(); i++) {
      auto memberValue = fSubFields[i]->CaptureValue(value->Get<unsigned char>() + fSubFieldsInfo[i].fOffset);
      fSubFields[i]->Read(clusterIndex, &memberValue);
   }
}

void ROOT::Experimental::RClassField::GenerateColumnsImpl()
{
}

void ROOT::Experimental::RClassField::GenerateColumnsImpl(const RNTupleDescriptor &)
{
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RClassField::GenerateValue(void* where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, fClass->New(where));
}

void ROOT::Experimental::RClassField::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   fClass->Destructor(value.GetRawPtr(), true /* dtorOnly */);
   if (!dtorOnly)
      free(value.GetRawPtr());
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RClassField::CaptureValue(void* where)
{
   return Detail::RFieldValue(true /* captureFlat */, this, where);
}


std::vector<ROOT::Experimental::Detail::RFieldValue>
ROOT::Experimental::RClassField::SplitValue(const Detail::RFieldValue &value) const
{
   std::vector<Detail::RFieldValue> result;
   for (unsigned i = 0; i < fSubFields.size(); i++) {
      auto memberValue = fSubFields[i]->CaptureValue(value.Get<unsigned char>() + fSubFieldsInfo[i].fOffset);
      result.emplace_back(memberValue);
   }
   return result;
}


size_t ROOT::Experimental::RClassField::GetValueSize() const
{
   return fClass->GetClassSize();
}

void ROOT::Experimental::RClassField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitClassField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RRecordField::RRecordField(
   std::string_view fieldName, std::vector<std::unique_ptr<Detail::RFieldBase>> &itemFields)
   : ROOT::Experimental::Detail::RFieldBase(fieldName, "", ENTupleStructure::kRecord, false /* isSimple */)
{
   for (auto &item : itemFields) {
      fMaxAlignment = std::max(fMaxAlignment, item->GetAlignment());
      fSize += GetItemPadding(fSize, item->GetAlignment()) + item->GetValueSize();
      Attach(std::move(item));
   }
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

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RRecordField::CloneImpl(std::string_view newName) const
{
   std::vector<std::unique_ptr<Detail::RFieldBase>> cloneItems;
   for (auto &item : fSubFields)
      cloneItems.emplace_back(item->Clone(item->GetName()));
   return std::make_unique<RRecordField>(newName, cloneItems);
}

std::size_t ROOT::Experimental::RRecordField::AppendImpl(const Detail::RFieldValue &value) {
   std::size_t nbytes = 0;
   std::size_t offset = 0;
   for (auto &item : fSubFields) {
      auto memberValue = item->CaptureValue(value.Get<unsigned char>() + offset);
      nbytes += item->Append(memberValue);
      offset += GetItemPadding(offset, item->GetAlignment()) + item->GetValueSize();
   }
   return nbytes;
}

void ROOT::Experimental::RRecordField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   std::size_t offset = 0;
   for (auto &item : fSubFields) {
      auto memberValue = item->CaptureValue(value->Get<unsigned char>() + offset);
      item->Read(globalIndex, &memberValue);
      offset += GetItemPadding(offset, item->GetAlignment()) + item->GetValueSize();
   }
}

void ROOT::Experimental::RRecordField::ReadInClusterImpl(const RClusterIndex &clusterIndex, Detail::RFieldValue *value)
{
   std::size_t offset = 0;
   for (auto &item : fSubFields) {
      auto memberValue = item->CaptureValue(value->Get<unsigned char>() + offset);
      item->Read(clusterIndex, &memberValue);
      offset += GetItemPadding(offset, item->GetAlignment()) + item->GetValueSize();
   }
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RRecordField::GenerateValue(void *where)
{
   std::size_t offset = 0;
   for (auto &item : fSubFields) {
      item->GenerateValue(static_cast<unsigned char *>(where) + offset);
      offset += GetItemPadding(offset, item->GetAlignment()) + item->GetValueSize();
   }
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

void ROOT::Experimental::RRecordField::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   std::size_t offset = 0;
   for (auto &item : fSubFields) {
      auto memberValue = item->CaptureValue(value.Get<unsigned char>() + offset);
      item->DestroyValue(memberValue, true /* dtorOnly */);
      offset += GetItemPadding(offset, item->GetAlignment()) + item->GetValueSize();
   }

   if (!dtorOnly)
      free(value.GetRawPtr());
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RRecordField::CaptureValue(void *where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}


std::vector<ROOT::Experimental::Detail::RFieldValue>
ROOT::Experimental::RRecordField::SplitValue(const Detail::RFieldValue &value) const
{
   std::size_t offset = 0;
   std::vector<Detail::RFieldValue> result;
   for (auto &item : fSubFields) {
      result.emplace_back(item->CaptureValue(value.Get<unsigned char>() + offset));
      offset += GetItemPadding(offset, item->GetAlignment()) + item->GetValueSize();
   }
   return result;
}


void ROOT::Experimental::RRecordField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitRecordField(*this);
}

//------------------------------------------------------------------------------


ROOT::Experimental::RVectorField::RVectorField(
   std::string_view fieldName, std::unique_ptr<Detail::RFieldBase> itemField)
   : ROOT::Experimental::Detail::RFieldBase(
      fieldName, "std::vector<" + itemField->GetType() + ">", ENTupleStructure::kCollection, false /* isSimple */)
   , fItemSize(itemField->GetValueSize()), fNWritten(0)
{
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RVectorField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetName());
   return std::make_unique<RVectorField>(newName, std::move(newItemField));
}

std::size_t ROOT::Experimental::RVectorField::AppendImpl(const Detail::RFieldValue& value) {
   auto typedValue = value.Get<std::vector<char>>();
   R__ASSERT((typedValue->size() % fItemSize) == 0);
   std::size_t nbytes = 0;
   auto count = typedValue->size() / fItemSize;
   for (unsigned i = 0; i < count; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(typedValue->data() + (i * fItemSize));
      nbytes += fSubFields[0]->Append(itemValue);
   }
   Detail::RColumnElement<ClusterSize_t> elemIndex(&fNWritten);
   fNWritten += count;
   fColumns[0]->Append(elemIndex);
   return nbytes + sizeof(elemIndex);
}

void ROOT::Experimental::RVectorField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   auto typedValue = value->Get<std::vector<char>>();

   ClusterSize_t nItems;
   RClusterIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   auto oldNItems = typedValue->size() / fItemSize;
   for (std::size_t i = nItems; i < oldNItems; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(typedValue->data() + (i * fItemSize));
      fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
   }
   typedValue->resize(nItems * fItemSize);
   for (std::size_t i = oldNItems; i < nItems; ++i) {
      fSubFields[0]->GenerateValue(typedValue->data() + (i * fItemSize));
   }

   for (std::size_t i = 0; i < nItems; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(typedValue->data() + (i * fItemSize));
      fSubFields[0]->Read(collectionStart + i, &itemValue);
   }
}

void ROOT::Experimental::RVectorField::GenerateColumnsImpl()
{
   RColumnModel modelIndex(EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<ClusterSize_t, EColumnType::kIndex>(modelIndex, 0)));
}

void ROOT::Experimental::RVectorField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kIndex}, 0, desc);
   GenerateColumnsImpl();
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RVectorField::GenerateValue(void* where)
{
   return Detail::RFieldValue(this, reinterpret_cast<std::vector<char>*>(where));
}

void ROOT::Experimental::RVectorField::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
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

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RVectorField::CaptureValue(void* where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

std::vector<ROOT::Experimental::Detail::RFieldValue>
ROOT::Experimental::RVectorField::SplitValue(const Detail::RFieldValue &value) const
{
   auto vec = static_cast<std::vector<char>*>(value.GetRawPtr());
   R__ASSERT((vec->size() % fItemSize) == 0);
   auto nItems = vec->size() / fItemSize;
   std::vector<Detail::RFieldValue> result;
   for (unsigned i = 0; i < nItems; ++i) {
      result.emplace_back(fSubFields[0]->CaptureValue(vec->data() + (i * fItemSize)));
   }
   return result;
}

void ROOT::Experimental::RVectorField::CommitCluster()
{
   fNWritten = 0;
}

void ROOT::Experimental::RVectorField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitVectorField(*this);
}


//------------------------------------------------------------------------------

ROOT::Experimental::RRVecField::RRVecField(std::string_view fieldName, std::unique_ptr<Detail::RFieldBase> itemField)
   : ROOT::Experimental::Detail::RFieldBase(fieldName, "ROOT::VecOps::RVec<" + itemField->GetType() + ">",
                                            ENTupleStructure::kCollection, false /* isSimple */),
     fItemSize(itemField->GetValueSize()), fNWritten(0)
{
   Attach(std::move(itemField));
   fValueSize = EvalValueSize(); // requires fSubFields to be populated
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RRVecField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetName());
   return std::make_unique<RRVecField>(newName, std::move(newItemField));
}

std::size_t ROOT::Experimental::RRVecField::AppendImpl(const Detail::RFieldValue &value)
{
   auto [beginPtr, sizePtr, _] = GetRVecDataMembers(value.GetRawPtr());

   std::size_t nbytes = 0;
   char *begin = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics
   for (std::int32_t i = 0; i < *sizePtr; ++i) {
      auto elementValue = fSubFields[0]->CaptureValue(begin + i * fItemSize);
      nbytes += fSubFields[0]->Append(elementValue);
   }

   Detail::RColumnElement<ClusterSize_t> elemIndex(&fNWritten);
   fNWritten += *sizePtr;
   fColumns[0]->Append(elemIndex);
   return nbytes + sizeof(elemIndex);
}

void ROOT::Experimental::RRVecField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   // TODO as a performance optimization, we could assign values to elements of the inline buffer:
   // if size < inline buffer size: we save one allocation here and usage of the RVec skips a pointer indirection

   auto [beginPtr, sizePtr, capacityPtr] = GetRVecDataMembers(value->GetRawPtr());

   // Read collection info for this entry
   ClusterSize_t nItems;
   RClusterIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);
   char *begin = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics
   const std::size_t oldSize = *sizePtr;

   // Destroy excess elements, if any
   for (std::size_t i = nItems; i < oldSize; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(begin + (i * fItemSize));
      fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
   }

   // Resize RVec (capacity and size)
   if (std::int32_t(nItems) > *capacityPtr) { // must reallocate
      // Destroy old elements: useless work for trivial types, but in case the element type's constructor
      // allocates memory we need to release it here to avoid memleaks (e.g. if this is an RVec<RVec<int>>)
      for (std::size_t i = 0u; i < oldSize; ++i) {
         auto itemValue = fSubFields[0]->CaptureValue(begin + (i * fItemSize));
         fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
      }

      // TODO Increment capacity by a factor rather than just enough to fit the elements.
      free(*beginPtr);
      // We trust that malloc returns a buffer with large enough alignment.
      // This might not be the case if T in RVec<T> is over-aligned.
      *beginPtr = malloc(nItems * fItemSize);
      R__ASSERT(*beginPtr != nullptr);
      begin = reinterpret_cast<char *>(*beginPtr);
      *capacityPtr = nItems;

      // Placement new for elements that were already there before the resize
      for (std::size_t i = 0u; i < oldSize; ++i)
         fSubFields[0]->GenerateValue(begin + (i * fItemSize));
   }
   *sizePtr = nItems;

   // Placement new for new elements, if any
   for (std::size_t i = oldSize; i < nItems; ++i)
      fSubFields[0]->GenerateValue(begin + (i * fItemSize));

   // Read the new values into the collection elements
   for (std::size_t i = 0; i < nItems; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(begin + (i * fItemSize));
      fSubFields[0]->Read(collectionStart + i, &itemValue);
   }
}

void ROOT::Experimental::RRVecField::GenerateColumnsImpl()
{
   RColumnModel modelIndex(EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(
      std::unique_ptr<Detail::RColumn>(Detail::RColumn::Create<ClusterSize_t, EColumnType::kIndex>(modelIndex, 0)));
}

void ROOT::Experimental::RRVecField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kIndex}, 0, desc);
   GenerateColumnsImpl();
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RRVecField::GenerateValue(void *where)
{
   // initialize data members fBegin, fSize, fCapacity
   // currently the inline buffer is left uninitialized
   void **beginPtr = new (where)(void *)(nullptr);
   std::int32_t *sizePtr = new (reinterpret_cast<void *>(beginPtr + 1)) std::int32_t(0);
   new (sizePtr + 1) std::int32_t(0);

   return Detail::RFieldValue(/*captureTag*/ true, this, where);
}

void ROOT::Experimental::RRVecField::DestroyValue(const Detail::RFieldValue &value, bool dtorOnly)
{
   auto [beginPtr, sizePtr, capacityPtr] = GetRVecDataMembers(value.GetRawPtr());

   char *begin = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics
   for (std::int32_t i = 0; i < *sizePtr; ++i) {
      auto elementValue = fSubFields[0]->CaptureValue(begin + i * fItemSize);
      fSubFields[0]->DestroyValue(elementValue, true /* dtorOnly */);
   }

   // figure out if we are in the small state, i.e. begin == &inlineBuffer
   // there might be padding between fCapacity and the inline buffer, so we compute it here
   constexpr auto dataMemberSz = sizeof(void *) + 2 * sizeof(std::int32_t);
   const auto alignOfT = fSubFields[0]->GetAlignment();
   auto paddingMiddle = dataMemberSz % alignOfT;
   if (paddingMiddle != 0)
      paddingMiddle = alignOfT - paddingMiddle;
   const bool isSmall = (reinterpret_cast<void *>(begin) == (beginPtr + dataMemberSz + paddingMiddle));

   const bool owns = (*capacityPtr != -1);
   if (!isSmall && owns)
      free(begin);

   if (!dtorOnly)
      free(beginPtr);
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RRVecField::CaptureValue(void *where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

std::vector<ROOT::Experimental::Detail::RFieldValue>
ROOT::Experimental::RRVecField::SplitValue(const Detail::RFieldValue &value) const
{
   auto [beginPtr, sizePtr, _] = GetRVecDataMembers(value.GetRawPtr());

   std::vector<Detail::RFieldValue> result;
   char *begin = reinterpret_cast<char *>(*beginPtr); // for pointer arithmetics
   for (std::int32_t i = 0; i < *sizePtr; ++i) {
      auto elementValue = fSubFields[0]->CaptureValue(begin + i * fItemSize);
      result.emplace_back(std::move(elementValue));
   }
   return result;
}

size_t ROOT::Experimental::RRVecField::EvalValueSize() const
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
   const auto alignOfT = fSubFields[0]->GetAlignment();
   const auto sizeOfT = fSubFields[0]->GetValueSize();

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
   const auto alignOfRVecT = GetAlignment();
   auto paddingEnd = (dataMemberSz + paddingMiddle + inlineStorageSz) % alignOfRVecT;
   if (paddingEnd != 0)
      paddingEnd = alignOfRVecT - paddingEnd;

   return dataMemberSz + inlineStorageSz + paddingMiddle + paddingEnd;
}

size_t ROOT::Experimental::RRVecField::GetValueSize() const
{
   return fValueSize;
}

size_t ROOT::Experimental::RRVecField::GetAlignment() const
{
   // the alignment of an RVec<T> is the largest among the alignments of its data members
   // (including the inline buffer which has the same alignment as the RVec::value_type)
   return std::max({alignof(void *), alignof(std::int32_t), fSubFields[0]->GetAlignment()});
}

void ROOT::Experimental::RRVecField::CommitCluster()
{
   fNWritten = 0;
}

void ROOT::Experimental::RRVecField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitRVecField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RField<std::vector<bool>>::RField(std::string_view name)
   : ROOT::Experimental::Detail::RFieldBase(name, "std::vector<bool>", ENTupleStructure::kCollection,
                                            false /* isSimple */)
{
   Attach(std::make_unique<RField<bool>>("_0"));
}

std::size_t ROOT::Experimental::RField<std::vector<bool>>::AppendImpl(const Detail::RFieldValue& value) {
   auto typedValue = value.Get<std::vector<bool>>();
   auto count = typedValue->size();
   for (unsigned i = 0; i < count; ++i) {
      bool bval = (*typedValue)[i];
      auto itemValue = fSubFields[0]->CaptureValue(&bval);
      fSubFields[0]->Append(itemValue);
   }
   Detail::RColumnElement<ClusterSize_t> elemIndex(&fNWritten);
   fNWritten += count;
   fColumns[0]->Append(elemIndex);
   return count + sizeof(elemIndex);
}

void ROOT::Experimental::RField<std::vector<bool>>::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue* value)
{
   auto typedValue = value->Get<std::vector<bool>>();

   ClusterSize_t nItems;
   RClusterIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   typedValue->resize(nItems);
   for (unsigned i = 0; i < nItems; ++i) {
      bool bval;
      auto itemValue = fSubFields[0]->GenerateValue(&bval);
      fSubFields[0]->Read(collectionStart + i, &itemValue);
      (*typedValue)[i] = bval;
   }
}

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumnsImpl()
{
   RColumnModel modelIndex(EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<ClusterSize_t, EColumnType::kIndex>(modelIndex, 0)));
}

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kIndex}, 0, desc);
   GenerateColumnsImpl();
}

std::vector<ROOT::Experimental::Detail::RFieldValue>
ROOT::Experimental::RField<std::vector<bool>>::SplitValue(const Detail::RFieldValue& value) const
{
   const static bool trueValue = true;
   const static bool falseValue = false;

   auto typedValue = value.Get<std::vector<bool>>();
   auto count = typedValue->size();
   std::vector<Detail::RFieldValue> result;
   for (unsigned i = 0; i < count; ++i) {
      if ((*typedValue)[i])
         result.emplace_back(fSubFields[0]->CaptureValue(const_cast<bool *>(&trueValue)));
      else
         result.emplace_back(fSubFields[0]->CaptureValue(const_cast<bool *>(&falseValue)));
   }
   return result;
}


void ROOT::Experimental::RField<std::vector<bool>>::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   auto vec = static_cast<std::vector<bool>*>(value.GetRawPtr());
   vec->~vector();
   if (!dtorOnly)
      free(vec);
}

void ROOT::Experimental::RField<std::vector<bool>>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitVectorBoolField(*this);
}


//------------------------------------------------------------------------------


ROOT::Experimental::RArrayField::RArrayField(
   std::string_view fieldName, std::unique_ptr<Detail::RFieldBase> itemField, std::size_t arrayLength)
   : ROOT::Experimental::Detail::RFieldBase(
      fieldName, "std::array<" + itemField->GetType() + "," + std::to_string(arrayLength) + ">",
      ENTupleStructure::kLeaf, false /* isSimple */, arrayLength)
   , fItemSize(itemField->GetValueSize()), fArrayLength(arrayLength)
{
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RArrayField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetName());
   return std::make_unique<RArrayField>(newName, std::move(newItemField), fArrayLength);
}

std::size_t ROOT::Experimental::RArrayField::AppendImpl(const Detail::RFieldValue& value) {
   std::size_t nbytes = 0;
   auto arrayPtr = value.Get<unsigned char>();
   for (unsigned i = 0; i < fArrayLength; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(arrayPtr + (i * fItemSize));
      nbytes += fSubFields[0]->Append(itemValue);
   }
   return nbytes;
}

void ROOT::Experimental::RArrayField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   auto arrayPtr = value->Get<unsigned char>();
   for (unsigned i = 0; i < fArrayLength; ++i) {
      auto itemValue = fSubFields[0]->GenerateValue(arrayPtr + (i * fItemSize));
      fSubFields[0]->Read(globalIndex * fArrayLength + i, &itemValue);
   }
}

void ROOT::Experimental::RArrayField::ReadInClusterImpl(const RClusterIndex &clusterIndex, Detail::RFieldValue *value)
{
   auto arrayPtr = value->Get<unsigned char>();
   for (unsigned i = 0; i < fArrayLength; ++i) {
      auto itemValue = fSubFields[0]->GenerateValue(arrayPtr + (i * fItemSize));
      fSubFields[0]->Read(RClusterIndex(clusterIndex.GetClusterId(), clusterIndex.GetIndex() * fArrayLength + i),
                          &itemValue);
   }
}

void ROOT::Experimental::RArrayField::GenerateColumnsImpl()
{
}

void ROOT::Experimental::RArrayField::GenerateColumnsImpl(const RNTupleDescriptor &)
{
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RArrayField::GenerateValue(void *where)
{
   auto arrayPtr = reinterpret_cast<unsigned char *>(where);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      fSubFields[0]->GenerateValue(arrayPtr + (i * fItemSize));
   }
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

void ROOT::Experimental::RArrayField::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   auto arrayPtr = value.Get<unsigned char>();
   for (unsigned i = 0; i < fArrayLength; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(arrayPtr + (i * fItemSize));
      fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
   }
   if (!dtorOnly)
      free(arrayPtr);
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RArrayField::CaptureValue(void *where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

std::vector<ROOT::Experimental::Detail::RFieldValue>
ROOT::Experimental::RArrayField::SplitValue(const Detail::RFieldValue &value) const
{
   auto arrayPtr = value.Get<unsigned char>();
   std::vector<Detail::RFieldValue> result;
   for (unsigned i = 0; i < fArrayLength; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(arrayPtr + (i * fItemSize));
      result.emplace_back(itemValue);
   }
   return result;
}

void ROOT::Experimental::RArrayField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitArrayField(*this);
}

//------------------------------------------------------------------------------

std::string ROOT::Experimental::RVariantField::GetTypeList(const std::vector<Detail::RFieldBase *> &itemFields)
{
   std::string result;
   for (size_t i = 0; i < itemFields.size(); ++i) {
      result += itemFields[i]->GetType() + ",";
   }
   R__ASSERT(!result.empty()); // there is always at least one variant
   result.pop_back(); // remove trailing comma
   return result;
}

ROOT::Experimental::RVariantField::RVariantField(
   std::string_view fieldName, const std::vector<Detail::RFieldBase *> &itemFields)
   : ROOT::Experimental::Detail::RFieldBase(fieldName,
      "std::variant<" + GetTypeList(itemFields) + ">", ENTupleStructure::kVariant, false /* isSimple */)
{
   auto nFields = itemFields.size();
   R__ASSERT(nFields > 0);
   fNWritten.resize(nFields, 0);
   for (unsigned int i = 0; i < nFields; ++i) {
      fMaxItemSize = std::max(fMaxItemSize, itemFields[i]->GetValueSize());
      fMaxAlignment = std::max(fMaxAlignment, itemFields[i]->GetAlignment());
      Attach(std::unique_ptr<Detail::RFieldBase>(itemFields[i]));
   }
   fTagOffset = (fMaxItemSize < fMaxAlignment) ? fMaxAlignment : fMaxItemSize;
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RVariantField::CloneImpl(std::string_view newName) const
{
   auto nFields = fSubFields.size();
   std::vector<Detail::RFieldBase *> itemFields;
   for (unsigned i = 0; i < nFields; ++i) {
      // TODO(jblomer): use unique_ptr in RVariantField constructor
      itemFields.emplace_back(fSubFields[i]->Clone(fSubFields[i]->GetName()).release());
   }
   return std::make_unique<RVariantField>(newName, itemFields);
}

std::uint32_t ROOT::Experimental::RVariantField::GetTag(void *variantPtr) const
{
   auto index = *(reinterpret_cast<char *>(variantPtr) + fTagOffset);
   return (index < 0) ? 0 : index + 1;
}

void ROOT::Experimental::RVariantField::SetTag(void *variantPtr, std::uint32_t tag) const
{
   auto index = reinterpret_cast<char *>(variantPtr) + fTagOffset;
   *index = static_cast<char>(tag - 1);
}

std::size_t ROOT::Experimental::RVariantField::AppendImpl(const Detail::RFieldValue& value)
{
   auto tag = GetTag(value.GetRawPtr());
   std::size_t nbytes = 0;
   auto index = 0;
   if (tag > 0) {
      auto itemValue = fSubFields[tag - 1]->CaptureValue(value.GetRawPtr());
      nbytes += fSubFields[tag - 1]->Append(itemValue);
      index = fNWritten[tag - 1]++;
   }
   RColumnSwitch varSwitch(ClusterSize_t(index), tag);
   Detail::RColumnElement<RColumnSwitch> elemSwitch(&varSwitch);
   fColumns[0]->Append(elemSwitch);
   return nbytes + sizeof(RColumnSwitch);
}

void ROOT::Experimental::RVariantField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   RClusterIndex variantIndex;
   std::uint32_t tag;
   fPrincipalColumn->GetSwitchInfo(globalIndex, &variantIndex, &tag);
   R__ASSERT(tag > 0); // TODO(jblomer): deal with invalid variants

   auto itemValue = fSubFields[tag - 1]->GenerateValue(value->GetRawPtr());
   fSubFields[tag - 1]->Read(variantIndex, &itemValue);
   SetTag(value->GetRawPtr(), tag);
}

void ROOT::Experimental::RVariantField::GenerateColumnsImpl()
{
   RColumnModel modelSwitch(EColumnType::kSwitch, false);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<RColumnSwitch, EColumnType::kSwitch>(modelSwitch, 0)));
}

void ROOT::Experimental::RVariantField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kSwitch}, 0, desc);
   GenerateColumnsImpl();
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RVariantField::GenerateValue(void *where)
{
   memset(where, 0, GetValueSize());
   fSubFields[0]->GenerateValue(where);
   SetTag(where, 1);
   return Detail::RFieldValue(this, reinterpret_cast<unsigned char *>(where));
}

void ROOT::Experimental::RVariantField::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   auto variantPtr = value.GetRawPtr();
   auto tag = GetTag(variantPtr);
   if (tag > 0) {
      auto itemValue = fSubFields[tag - 1]->CaptureValue(variantPtr);
      fSubFields[tag - 1]->DestroyValue(itemValue, true /* dtorOnly */);
   }
   if (!dtorOnly)
      free(variantPtr);
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RVariantField::CaptureValue(void *where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

size_t ROOT::Experimental::RVariantField::GetValueSize() const
{
   return fMaxItemSize + fMaxAlignment;  // TODO: fix for more than 255 items
}

void ROOT::Experimental::RVariantField::CommitCluster()
{
   std::fill(fNWritten.begin(), fNWritten.end(), 0);
}


//------------------------------------------------------------------------------


ROOT::Experimental::RCollectionField::RCollectionField(
   std::string_view name,
   std::shared_ptr<RCollectionNTupleWriter> collectionNTuple,
   std::unique_ptr<RNTupleModel> collectionModel)
   : RFieldBase(name, "", ENTupleStructure::kCollection, true /* isSimple */)
   , fCollectionNTuple(collectionNTuple)
{
   for (unsigned i = 0; i < collectionModel->GetFieldZero()->fSubFields.size(); ++i) {
      auto& subField = collectionModel->GetFieldZero()->fSubFields[i];
      Attach(std::move(subField));
   }
   SetDescription(collectionModel->GetDescription());
}


void ROOT::Experimental::RCollectionField::GenerateColumnsImpl()
{
   RColumnModel modelIndex(EColumnType::kIndex, true /* isSorted*/);
   fColumns.emplace_back(std::unique_ptr<Detail::RColumn>(
      Detail::RColumn::Create<ClusterSize_t, EColumnType::kIndex>(modelIndex, 0)));
}

void ROOT::Experimental::RCollectionField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   EnsureColumnType({EColumnType::kIndex}, 0, desc);
   GenerateColumnsImpl();
}


std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RCollectionField::CloneImpl(std::string_view newName) const
{
   auto result = std::make_unique<RCollectionField>(newName, fCollectionNTuple, RNTupleModel::Create());
   for (auto& f : fSubFields) {
      auto clone = f->Clone(f->GetName());
      result->Attach(std::move(clone));
   }
   return result;
}


void ROOT::Experimental::RCollectionField::CommitCluster() {
   *fCollectionNTuple->GetOffsetPtr() = 0;
}
