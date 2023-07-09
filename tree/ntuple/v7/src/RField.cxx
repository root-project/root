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
#include <TObjArray.h>
#include <TObjString.h>
#include <TRealData.h>
#include <TSchemaRule.h>
#include <TSchemaRuleSet.h>
#include <TVirtualObject.h>

#include <algorithm>
#include <cctype> // for isspace
#include <charconv>
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

/// Parse a type name of the form `T[n][m]...` and return the base type `T` and a vector that contains,
/// in order, the declared size for each dimension, e.g. for `unsigned char[1][2][3]` it returns the tuple
/// `{"unsigned char", {1, 2, 3}}`. Extra whitespace in `typeName` should be removed before calling this function.
///
/// If `typeName` is not an array type, it returns a tuple `{T, {}}`. On error, it returns a default-constructed tuple.
std::tuple<std::string, std::vector<size_t>> ParseArrayType(std::string_view typeName)
{
   std::vector<size_t> sizeVec;

   // Only parse outer array definition, i.e. the right `]` should be at the end of the type name
   while (typeName.back() == ']') {
      auto posRBrace = typeName.size() - 1;
      auto posLBrace = typeName.find_last_of("[", posRBrace);
      if (posLBrace == std::string_view::npos)
         return {};

      size_t size;
      if (std::from_chars(typeName.data() + posLBrace + 1, typeName.data() + posRBrace, size).ec != std::errc{})
         return {};
      sizeVec.insert(sizeVec.begin(), size);
      typeName.remove_suffix(typeName.size() - posLBrace);
   }
   return std::make_tuple(std::string{typeName}, sizeVec);
}

/// Returns the normalized resolved type and the normalized typedef before typedef resolution. If the given type
/// is not a typedef, the second element of the returned pair is the empty string.
std::pair<std::string, std::string> GetNormalizedType(const std::string &typeName) {
   std::string normalizedType(TClassEdit::CleanType(typeName.c_str(), /*mode=*/2));
   // The following types are asummed to be canonical names; thus, do not perform `typedef` resolution on those
   if (normalizedType == "ROOT::Experimental::ClusterSize_t")
      return std::make_pair(normalizedType, "");
   // We want the template parameter to stay in "std::uint32_t" / "std::uint64_t" form
   if (normalizedType.substr(0, 39) == "ROOT::Experimental::RNTupleCardinality<")
      return std::make_pair(normalizedType, "");

   std::string resolvedType = TClassEdit::ResolveTypedef(normalizedType.c_str());
   if (resolvedType == normalizedType)
      normalizedType = "";
   auto translatedType = typeTranslationMap.find(resolvedType);
   if (translatedType != typeTranslationMap.end())
      resolvedType = translatedType->second;

   if (resolvedType.substr(0, 7) == "vector<")
      resolvedType = "std::" + resolvedType;
   if (resolvedType.substr(0, 6) == "array<")
      resolvedType = "std::" + resolvedType;
   if (resolvedType.substr(0, 8) == "variant<")
      resolvedType = "std::" + resolvedType;
   if (resolvedType.substr(0, 5) == "pair<")
      resolvedType = "std::" + resolvedType;
   if (resolvedType.substr(0, 6) == "tuple<")
      resolvedType = "std::" + resolvedType;
   if (resolvedType.substr(0, 7) == "bitset<")
      resolvedType = "std::" + resolvedType;
   if (resolvedType.substr(0, 11) == "unique_ptr<")
      resolvedType = "std::" + resolvedType;

   return std::make_pair(resolvedType, normalizedType);
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

ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations::RColumnRepresentations()
{
   // A single representations with an empty set of columns
   fSerializationTypes.emplace_back(ColumnRepresentation_t());
   fDeserializationTypes.emplace_back(ColumnRepresentation_t());
}

ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations::RColumnRepresentations(
   const TypesList_t &serializationTypes, const TypesList_t &deserializationExtraTypes)
   : fSerializationTypes(serializationTypes), fDeserializationTypes(serializationTypes)
{
   fDeserializationTypes.insert(fDeserializationTypes.end(),
                                deserializationExtraTypes.begin(), deserializationExtraTypes.end());
}

//------------------------------------------------------------------------------

ROOT::Experimental::Detail::RFieldBase::RFieldBase(std::string_view name, std::string_view type,
                                                   ENTupleStructure structure, bool isSimple, std::size_t nRepetitions)
   : fName(name), fType(type), fStructure(structure), fNRepetitions(nRepetitions), fIsSimple(isSimple),
     fParent(nullptr), fPrincipalColumn(nullptr), fTraits(isSimple ? kTraitMappable : 0)
{
}

ROOT::Experimental::Detail::RFieldBase::~RFieldBase()
{
}

std::string ROOT::Experimental::Detail::RFieldBase::GetQualifiedFieldName() const
{
   std::string result = GetName();
   RFieldBase *parent = GetParent();
   while (parent && !parent->GetName().empty()) {
      result = parent->GetName() + "." + result;
      parent = parent->GetParent();
   }
   return result;
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>>
ROOT::Experimental::Detail::RFieldBase::Create(const std::string &fieldName, const std::string &typeName)
{
   auto [normalizedType, typeAlias] = GetNormalizedType(typeName);
   if (normalizedType.empty())
      return R__FAIL("no type name specified for Field " + fieldName);

   if (auto [arrayBaseType, arraySize] = ParseArrayType(normalizedType); !arraySize.empty()) {
      // TODO(jalopezg): support multi-dimensional row-major (C order) arrays in RArrayField
      if (arraySize.size() > 1)
         return R__FAIL("multi-dimensional array type not supported " + normalizedType);
      auto itemField = Create("_0", arrayBaseType).Unwrap();
      return {std::make_unique<RArrayField>(fieldName, std::move(itemField), arraySize[0])};
   }

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
   } else if (normalizedType == "Double32_t") {
      result = std::make_unique<RField<double>>(fieldName);
      static_cast<RField<double> *>(result.get())->SetDouble32();
      // Prevent the type alias from being reset by returning early
      return result;
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
      auto itemField = Create("_0", arrayDef[0]);
      result = std::make_unique<RArrayField>(fieldName, itemField.Unwrap(), arrayLength);
   } else if (normalizedType.substr(0, 13) == "std::variant<") {
      auto innerTypes = TokenizeTypeList(normalizedType.substr(13, normalizedType.length() - 14));
      std::vector<RFieldBase *> items;
      for (unsigned int i = 0; i < innerTypes.size(); ++i) {
         items.emplace_back(Create("_" + std::to_string(i), innerTypes[i]).Unwrap().release());
      }
      result = std::make_unique<RVariantField>(fieldName, items);
   } else if (normalizedType.substr(0, 10) == "std::pair<") {
      auto innerTypes = TokenizeTypeList(normalizedType.substr(10, normalizedType.length() - 11));
      if (innerTypes.size() != 2)
         return R__FAIL("the type list for std::pair must have exactly two elements");
      std::array<std::unique_ptr<RFieldBase>, 2> items{Create("_0", innerTypes[0]).Unwrap(),
                                                       Create("_1", innerTypes[1]).Unwrap()};
      result = std::make_unique<RPairField>(fieldName, items);
   } else if (normalizedType.substr(0, 11) == "std::tuple<") {
      auto innerTypes = TokenizeTypeList(normalizedType.substr(11, normalizedType.length() - 12));
      std::vector<std::unique_ptr<RFieldBase>> items;
      for (unsigned int i = 0; i < innerTypes.size(); ++i) {
         items.emplace_back(Create("_" + std::to_string(i), innerTypes[i]).Unwrap());
      }
      result = std::make_unique<RTupleField>(fieldName, items);
   } else if (normalizedType.substr(0, 12) == "std::bitset<") {
      auto size = std::stoull(normalizedType.substr(12, normalizedType.length() - 13));
      result = std::make_unique<RBitsetField>(fieldName, size);
   } else if (normalizedType.substr(0, 16) == "std::unique_ptr<") {
      std::string itemTypeName = normalizedType.substr(16, normalizedType.length() - 17);
      auto itemField = Create("_0", itemTypeName).Unwrap();
      auto normalizedInnerTypeName = itemField->GetType();
      result = std::make_unique<RUniquePtrField>(fieldName, "std::unique_ptr<" + normalizedInnerTypeName + ">",
                                                 std::move(itemField));
   } else if (normalizedType == ":Collection:") {
      // TODO: create an RCollectionField?
      result = std::make_unique<RField<ClusterSize_t>>(fieldName);
   } else if (normalizedType.substr(0, 39) == "ROOT::Experimental::RNTupleCardinality<") {
      auto innerTypes = TokenizeTypeList(normalizedType.substr(39, normalizedType.length() - 40));
      if (innerTypes.size() != 1)
         return R__FAIL(std::string("Field ") + fieldName + " has invalid cardinality template: " + normalizedType);
      if (innerTypes[0] == "std::uint32_t") {
         result = std::make_unique<RField<RNTupleCardinality<std::uint32_t>>>(fieldName);
      } else if (innerTypes[0] == "std::uint64_t") {
         result = std::make_unique<RField<RNTupleCardinality<std::uint64_t>>>(fieldName);
      } else {
         return R__FAIL(std::string("Field ") + fieldName + " has invalid cardinality template: " + normalizedType);
      }
   }

   if (!result) {
      auto cl = TClass::GetClass(normalizedType.c_str());
      if (cl != nullptr) {
         if (cl->GetCollectionProxy())
            result = std::make_unique<RCollectionClassField>(fieldName, normalizedType);
         else
            result = std::make_unique<RClassField>(fieldName, normalizedType);
      }
   }

   if (result) {
      result->fTypeAlias = typeAlias;
      return result;
   }
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

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::Detail::RFieldBase::GetColumnRepresentations() const
{
   static RColumnRepresentations representations;
   return representations;
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::Detail::RFieldBase::Clone(std::string_view newName) const
{
   auto clone = CloneImpl(newName);
   clone->fTypeAlias = fTypeAlias;
   clone->fOnDiskId = fOnDiskId;
   clone->fDescription = fDescription;
   // We can just copy the pointer because fColumnRepresentative points into a static structure
   clone->fColumnRepresentative = fColumnRepresentative;
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

ROOT::Experimental::NTupleSize_t
ROOT::Experimental::Detail::RFieldBase::EntryToColumnElementIndex(ROOT::Experimental::NTupleSize_t globalIndex) const
{
   std::size_t result = globalIndex;
   for (auto f = this; f != nullptr; f = f->GetParent()) {
      auto parent = f->GetParent();
      if (parent && (parent->GetStructure() == kCollection || parent->GetStructure() == kVariant))
         return 0U;
      result *= std::max(f->GetNRepetitions(), std::size_t{1U});
   }
   return result;
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

const ROOT::Experimental::Detail::RFieldBase::ColumnRepresentation_t &
ROOT::Experimental::Detail::RFieldBase::GetColumnRepresentative() const
{
   if (fColumnRepresentative)
      return *fColumnRepresentative;
   return GetColumnRepresentations().GetSerializationDefault();
}

void ROOT::Experimental::Detail::RFieldBase::SetColumnRepresentative(const ColumnRepresentation_t &representative)
{
   if (!fColumns.empty())
      throw RException(R__FAIL("cannot set column representative once field is connected"));
   const auto &validTypes = GetColumnRepresentations().GetSerializationTypes();
   auto itRepresentative = std::find(validTypes.begin(), validTypes.end(), representative);
   if (itRepresentative == std::end(validTypes))
      throw RException(R__FAIL("invalid column representative"));
   fColumnRepresentative = &(*itRepresentative);
}

const ROOT::Experimental::Detail::RFieldBase::ColumnRepresentation_t &
ROOT::Experimental::Detail::RFieldBase::EnsureCompatibleColumnTypes(const RNTupleDescriptor &desc) const
{
   if (fOnDiskId == kInvalidDescriptorId)
      throw RException(R__FAIL("No on-disk column information for field `" + GetQualifiedFieldName() + "`"));

   ColumnRepresentation_t onDiskTypes;
   for (const auto &c : desc.GetColumnIterable(fOnDiskId)) {
      onDiskTypes.emplace_back(c.GetModel().GetType());
   }
   for (const auto &t : GetColumnRepresentations().GetDeserializationTypes()) {
      if (t == onDiskTypes)
         return t;
   }

   std::string columnTypeNames;
   for (const auto &t : onDiskTypes) {
      if (!columnTypeNames.empty())
         columnTypeNames += ", ";
      columnTypeNames += RColumnElementBase::GetTypeName(t);
   }
   throw RException(R__FAIL("On-disk column types `" + columnTypeNames + "` for field `" + GetQualifiedFieldName() +
                            "` cannot be matched."));
}

size_t ROOT::Experimental::Detail::RFieldBase::AddReadCallback(ReadCallback_t func)
{
   fReadCallbacks.push_back(func);
   fIsSimple = false;
   return fReadCallbacks.size() - 1;
}

void ROOT::Experimental::Detail::RFieldBase::RemoveReadCallback(size_t idx)
{
   fReadCallbacks.erase(fReadCallbacks.begin() + idx);
   fIsSimple = (fTraits & kTraitMappable) && fReadCallbacks.empty();
}

void ROOT::Experimental::Detail::RFieldBase::AutoAdjustColumnTypes(const RNTupleWriteOptions &options)
{
   if ((options.GetCompression() == 0) && HasDefaultColumnRepresentative()) {
      ColumnRepresentation_t rep = GetColumnRepresentative();
      for (auto &colType : rep) {
         switch (colType) {
         case EColumnType::kSplitIndex64: colType = EColumnType::kIndex64; break;
         case EColumnType::kSplitIndex32: colType = EColumnType::kIndex32; break;
         case EColumnType::kSplitReal64: colType = EColumnType::kReal64; break;
         case EColumnType::kSplitReal32: colType = EColumnType::kReal32; break;
         case EColumnType::kSplitInt64: colType = EColumnType::kInt64; break;
         case EColumnType::kSplitInt32: colType = EColumnType::kInt32; break;
         case EColumnType::kSplitInt16: colType = EColumnType::kInt16; break;
         default: break;
         }
      }
      SetColumnRepresentative(rep);
   }

   if (options.GetHasSmallClusters()) {
      ColumnRepresentation_t rep = GetColumnRepresentative();
      for (auto &colType : rep) {
         switch (colType) {
         case EColumnType::kSplitIndex64: colType = EColumnType::kSplitIndex32; break;
         case EColumnType::kIndex64: colType = EColumnType::kIndex32; break;
         default: break;
         }
      }
      SetColumnRepresentative(rep);
   }

   if (fTypeAlias == "Double32_t")
      SetColumnRepresentative({EColumnType::kSplitReal32});
}

void ROOT::Experimental::Detail::RFieldBase::ConnectPageSink(RPageSink &pageSink, NTupleSize_t firstEntry)
{
   R__ASSERT(fColumns.empty());

   AutoAdjustColumnTypes(pageSink.GetWriteOptions());

   GenerateColumnsImpl();
   if (!fColumns.empty())
      fPrincipalColumn = fColumns[0].get();
   for (auto &column : fColumns) {
      auto firstElementIndex = (column.get() == fPrincipalColumn) ? EntryToColumnElementIndex(firstEntry) : 0;
      column->Connect(fOnDiskId, &pageSink, firstElementIndex);
   }
}


void ROOT::Experimental::Detail::RFieldBase::ConnectPageSource(RPageSource &pageSource)
{
   R__ASSERT(fColumns.empty());
   if (fColumnRepresentative)
      throw RException(R__FAIL("fixed column representative only valid when connecting to a page sink"));

   {
      const auto descriptorGuard = pageSource.GetSharedDescriptorGuard();
      const RNTupleDescriptor &desc = descriptorGuard.GetRef();
      GenerateColumnsImpl(desc);
      ColumnRepresentation_t onDiskColumnTypes;
      for (const auto &c : fColumns) {
         onDiskColumnTypes.emplace_back(c->GetModel().GetType());
      }
      for (const auto &t : GetColumnRepresentations().GetDeserializationTypes()) {
         if (t == onDiskColumnTypes)
            fColumnRepresentative = &t;
      }
      R__ASSERT(fColumnRepresentative);
      if (fOnDiskId != kInvalidDescriptorId)
         fOnDiskTypeVersion = desc.GetFieldDescriptor(fOnDiskId).GetTypeVersion();
   }
   if (!fColumns.empty())
      fPrincipalColumn = fColumns[0].get();
   for (auto& column : fColumns)
      column->Connect(fOnDiskId, &pageSource);
   OnConnectPageSource();
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

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitClusterSizeField(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RCardinalityField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RCardinalityField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RCardinalityField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitCardinalityField(*this);
}

const ROOT::Experimental::RField<ROOT::Experimental::RNTupleCardinality<std::uint32_t>> *
ROOT::Experimental::RCardinalityField::As32Bit() const
{
   return dynamic_cast<const RField<RNTupleCardinality<std::uint32_t>> *>(this);
}

const ROOT::Experimental::RField<ROOT::Experimental::RNTupleCardinality<std::uint64_t>> *
ROOT::Experimental::RCardinalityField::As64Bit() const
{
   return dynamic_cast<const RField<RNTupleCardinality<std::uint64_t>> *>(this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<char>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kChar}}, {{}});
   return representations;
}

void ROOT::Experimental::RField<char>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<char>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<char>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<char>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<char>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitCharField(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::int8_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kInt8}}, {{EColumnType::kUInt8}});
   return representations;
}

void ROOT::Experimental::RField<std::int8_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<std::int8_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::int8_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<std::int8_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::int8_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt8Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::uint8_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kUInt8}}, {{EColumnType::kInt8}});
   return representations;
}

void ROOT::Experimental::RField<std::uint8_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<std::uint8_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::uint8_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<std::uint8_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::uint8_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt8Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<bool>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kBit}}, {});
   return representations;
}

void ROOT::Experimental::RField<bool>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<bool>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<bool>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<bool>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<bool>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitBoolField(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<float>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitReal32}, {EColumnType::kReal32}}, {});
   return representations;
}

void ROOT::Experimental::RField<float>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<float>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<float>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<float>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<float>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitFloatField(*this);
}


//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<double>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitReal64}, {EColumnType::kReal64}, {EColumnType::kSplitReal32}, {EColumnType::kReal32}}, {});
   return representations;
}

void ROOT::Experimental::RField<double>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<double>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<double>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<double>(RColumnModel(onDiskTypes[0]), 0));
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

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::int16_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitInt16}, {EColumnType::kInt16}},
                                                 {{EColumnType::kSplitUInt16}, {EColumnType::kUInt16}});
   return representations;
}

void ROOT::Experimental::RField<std::int16_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<std::int16_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::int16_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<std::int16_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::int16_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt16Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::uint16_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitUInt16}, {EColumnType::kUInt16}},
                                                 {{EColumnType::kSplitInt16}, {EColumnType::kInt16}});
   return representations;
}

void ROOT::Experimental::RField<std::uint16_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<std::uint16_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::uint16_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<std::uint16_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::uint16_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt16Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::int32_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitInt32}, {EColumnType::kInt32}},
                                                 {{EColumnType::kSplitUInt32}, {EColumnType::kUInt32}});
   return representations;
}

void ROOT::Experimental::RField<std::int32_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<std::int32_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::int32_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<std::int32_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::int32_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitIntField(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::uint32_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitUInt32}, {EColumnType::kUInt32}},
                                                 {{EColumnType::kSplitInt32}, {EColumnType::kInt32}});
   return representations;
}

void ROOT::Experimental::RField<std::uint32_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<std::uint32_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::uint32_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<std::uint32_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::uint32_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt32Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::uint64_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitUInt64}, {EColumnType::kUInt64}},
                                                 {{EColumnType::kSplitInt64}, {EColumnType::kInt64}});
   return representations;
}

void ROOT::Experimental::RField<std::uint64_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<std::uint64_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::uint64_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<std::uint64_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::uint64_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt64Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::int64_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitInt64}, {EColumnType::kInt64}},
                                                 {{EColumnType::kSplitUInt64},
                                                  {EColumnType::kUInt64},
                                                  {EColumnType::kInt32},
                                                  {EColumnType::kSplitInt32},
                                                  {EColumnType::kUInt32},
                                                  {EColumnType::kSplitUInt32}});
   return representations;
}

void ROOT::Experimental::RField<std::int64_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<std::int64_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::int64_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<std::int64_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::int64_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt64Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::string>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitIndex64, EColumnType::kChar},
                                                  {EColumnType::kIndex64, EColumnType::kChar},
                                                  {EColumnType::kSplitIndex32, EColumnType::kChar},
                                                  {EColumnType::kIndex32, EColumnType::kChar}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RField<std::string>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
   fColumns.emplace_back(Detail::RColumn::Create<char>(RColumnModel(GetColumnRepresentative()[1]), 1));
}

void ROOT::Experimental::RField<std::string>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
   fColumns.emplace_back(Detail::RColumn::Create<char>(RColumnModel(onDiskTypes[1]), 1));
}

std::size_t ROOT::Experimental::RField<std::string>::AppendImpl(const ROOT::Experimental::Detail::RFieldValue& value)
{
   auto typedValue = value.Get<std::string>();
   auto length = typedValue->length();
   fColumns[1]->AppendV(typedValue->data(), length);
   fIndex += length;
   fColumns[0]->Append(fElemIndex);
   return length + fColumns[0]->GetElement()->GetPackedSize();
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
   if (fClass->GetCollectionProxy()) {
      throw RException(
         R__FAIL(std::string(className) + " has an associated collection proxy; use RCollectionClassField instead"));
   }

   if (!(fClass->ClassProperty() & kClassHasExplicitCtor))
      fTraits |= kTraitTriviallyConstructible;
   if (!(fClass->ClassProperty() & kClassHasExplicitDtor))
      fTraits |= kTraitTriviallyDestructible;

   int i = 0;
   for (auto baseClass : ROOT::Detail::TRangeStaticCast<TBaseClass>(*fClass->GetListOfBases())) {
      TClass *c = baseClass->GetClassPointer();
      auto subField = Detail::RFieldBase::Create(std::string(kPrefixInherited) + "_" + std::to_string(i),
                                                 c->GetName()).Unwrap();
      fTraits &= subField->GetTraits();
      Attach(std::move(subField),
	     RSubFieldInfo{kBaseClass, static_cast<std::size_t>(baseClass->GetDelta())});
      i++;
   }
   for (auto dataMember : ROOT::Detail::TRangeStaticCast<TDataMember>(*fClass->GetListOfDataMembers())) {
      // Skip, for instance, unscoped enum constants defined in the class
      if (dataMember->Property() & kIsStatic)
         continue;
      // Skip members explicitly marked as transient by user comment
      if (!dataMember->IsPersistent()) {
         // TODO(jblomer): we could do better
         fTraits &= ~(kTraitTriviallyConstructible | kTraitTriviallyDestructible);
         continue;
      }

      std::string typeName{dataMember->GetFullTypeName()};
      // For C-style arrays, complete the type name with the size for each dimension, e.g. `int[4][2]`
      if (dataMember->Property() & kIsArray) {
         for (int dim = 0, n = dataMember->GetArrayDim(); dim < n; ++dim)
            typeName += "[" + std::to_string(dataMember->GetMaxIndex(dim)) + "]";
      }
      auto subField = Detail::RFieldBase::Create(dataMember->GetName(), typeName).Unwrap();
      fTraits &= subField->GetTraits();
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

void ROOT::Experimental::RClassField::AddReadCallbacksFromIORules(const std::span<const ROOT::TSchemaRule *> rules,
                                                                  TClass *classp)
{
   for (const auto rule : rules) {
      if (rule->GetRuleType() != ROOT::TSchemaRule::kReadRule) {
         R__LOG_WARNING(NTupleLog()) << "ignoring I/O customization rule with unsupported type";
         continue;
      }
      auto func = rule->GetReadFunctionPointer();
      R__ASSERT(func != nullptr);
      fReadCallbacks.emplace_back([func, classp](Detail::RFieldValue &value) {
         TVirtualObject oldObj{nullptr};
         oldObj.fClass = classp;
         oldObj.fObject = value.GetRawPtr();
         func(static_cast<char *>(value.GetRawPtr()), &oldObj);
         oldObj.fClass = nullptr; // TVirtualObject does not own the value
      });
   }
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

void ROOT::Experimental::RClassField::OnConnectPageSource()
{
   // Add post-read callbacks for I/O customization rules; only rules that target transient members are allowed for now
   // TODO(jalopezg): revise after supporting schema evolution
   const auto ruleset = fClass->GetSchemaRules();
   if (!ruleset)
      return;
   auto referencesNonTransientMembers = [klass = fClass](const ROOT::TSchemaRule *rule) {
      if (rule->GetTarget() == nullptr)
         return false;
      for (auto target : ROOT::Detail::TRangeStaticCast<TObjString>(*rule->GetTarget())) {
         const auto dataMember = klass->GetDataMember(target->GetString());
         if (!dataMember || dataMember->IsPersistent()) {
            R__LOG_WARNING(NTupleLog()) << "ignoring I/O customization rule with non-transient member: "
                                        << dataMember->GetName();
            return true;
         }
      }
      return false;
   };

   auto rules = ruleset->FindRules(fClass->GetName(), static_cast<Int_t>(GetOnDiskTypeVersion()));
   rules.erase(std::remove_if(rules.begin(), rules.end(), referencesNonTransientMembers), rules.end());
   AddReadCallbacksFromIORules(rules, fClass);
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

std::uint32_t ROOT::Experimental::RClassField::GetTypeVersion() const
{
   return fClass->GetClassVersion();
}

void ROOT::Experimental::RClassField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitClassField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RCollectionClassField::RCollectionIterableOnce::RIteratorFuncs
ROOT::Experimental::RCollectionClassField::RCollectionIterableOnce::GetIteratorFuncs(TVirtualCollectionProxy *proxy,
                                                                                     bool readFromDisk)
{
   RIteratorFuncs ifuncs;
   ifuncs.fCreateIterators = proxy->GetFunctionCreateIterators(readFromDisk);
   ifuncs.fDeleteTwoIterators = proxy->GetFunctionDeleteTwoIterators(readFromDisk);
   ifuncs.fNext = proxy->GetFunctionNext(readFromDisk);
   R__ASSERT((ifuncs.fCreateIterators != nullptr) && (ifuncs.fDeleteTwoIterators != nullptr) &&
             (ifuncs.fNext != nullptr));
   return ifuncs;
}

ROOT::Experimental::RCollectionClassField::RCollectionClassField(std::string_view fieldName, std::string_view className)
   : RCollectionClassField(fieldName, className, TClass::GetClass(std::string(className).c_str()))
{
}

ROOT::Experimental::RCollectionClassField::RCollectionClassField(std::string_view fieldName, std::string_view className,
                                                                 TClass *classp)
   : ROOT::Experimental::Detail::RFieldBase(fieldName, className, ENTupleStructure::kCollection, false /* isSimple */),
     fNWritten(0)
{
   if (classp == nullptr)
      throw RException(R__FAIL("RField: no I/O support for collection proxy type " + std::string(className)));
   if (!classp->GetCollectionProxy())
      throw RException(R__FAIL(std::string(className) + " has no associated collection proxy"));

   fProxy.reset(classp->GetCollectionProxy()->Generate());
   fProperties = fProxy->GetProperties();
   fCollectionType = fProxy->GetCollectionType();
   if (fProxy->HasPointers())
      throw RException(R__FAIL("collection proxies whose value type is a pointer are not supported"));
   if (fProperties & TVirtualCollectionProxy::kIsAssociative)
      throw RException(R__FAIL("associative collections not supported"));

   fIFuncsRead = RCollectionIterableOnce::GetIteratorFuncs(fProxy.get(), true /* readFromDisk */);
   fIFuncsWrite = RCollectionIterableOnce::GetIteratorFuncs(fProxy.get(), false /* readFromDisk */);

   std::unique_ptr<ROOT::Experimental::Detail::RFieldBase> itemField;
   if (auto valueClass = fProxy->GetValueClass()) {
      // Element type is a class
      itemField = RFieldBase::Create("_0", valueClass->GetName()).Unwrap();
   } else {
      switch (fProxy->GetType()) {
      case EDataType::kChar_t:   itemField = std::make_unique<RField<char>>("_0"); break;
      case EDataType::kUChar_t:  itemField = std::make_unique<RField<std::uint8_t>>("_0"); break;
      case EDataType::kShort_t:  itemField = std::make_unique<RField<std::int16_t>>("_0"); break;
      case EDataType::kUShort_t: itemField = std::make_unique<RField<std::uint16_t>>("_0"); break;
      case EDataType::kInt_t:    itemField = std::make_unique<RField<std::int32_t>>("_0"); break;
      case EDataType::kUInt_t:   itemField = std::make_unique<RField<std::uint32_t>>("_0"); break;
      case EDataType::kLong_t:
      case EDataType::kLong64_t:
         itemField = std::make_unique<RField<std::int64_t>>("_0");
         break;
      case EDataType::kULong_t:
      case EDataType::kULong64_t:
         itemField = std::make_unique<RField<std::uint64_t>>("_0");
         break;
      case EDataType::kFloat_t:  itemField = std::make_unique<RField<float>>("_0"); break;
      case EDataType::kDouble_t: itemField = std::make_unique<RField<double>>("_0"); break;
      case EDataType::kBool_t:   itemField = std::make_unique<RField<bool>>("_0"); break;
      default:
         throw RException(R__FAIL("unsupported value type"));
      }
   }
   fItemSize = itemField->GetValueSize();
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RCollectionClassField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RCollectionClassField>(
      new RCollectionClassField(newName, GetType(), fProxy->GetCollectionClass()));
}

std::size_t ROOT::Experimental::RCollectionClassField::AppendImpl(const Detail::RFieldValue &value)
{
   std::size_t nbytes = 0;
   unsigned count = 0;
   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), value.GetRawPtr());
   for (auto ptr : RCollectionIterableOnce{value.GetRawPtr(), fIFuncsWrite, fProxy.get(),
                                           (fCollectionType == kSTLvector ? fItemSize : 0U)}) {
      auto itemValue = fSubFields[0]->CaptureValue(ptr);
      nbytes += fSubFields[0]->Append(itemValue);
      count++;
   }

   Detail::RColumnElement<ClusterSize_t> elemIndex(&fNWritten);
   fNWritten += count;
   fColumns[0]->Append(elemIndex);
   return nbytes + fColumns[0]->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RCollectionClassField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   ClusterSize_t nItems;
   RClusterIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), value->GetRawPtr());
   void *obj =
      fProxy->Allocate(static_cast<std::uint32_t>(nItems), (fProperties & TVirtualCollectionProxy::kNeedDelete));

   unsigned i = 0;
   for (auto elementPtr :
        RCollectionIterableOnce{obj, fIFuncsRead, fProxy.get(),
                                (fCollectionType == kSTLvector || obj != value->GetRawPtr() ? fItemSize : 0U)}) {
      auto itemValue = fSubFields[0]->CaptureValue(elementPtr);
      fSubFields[0]->Read(collectionStart + (i++), &itemValue);
   }
   if (obj != value->GetRawPtr())
      fProxy->Commit(obj);
}

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RCollectionClassField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RCollectionClassField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RCollectionClassField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RCollectionClassField::GenerateValue(void *where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, fProxy->New(where));
}

void ROOT::Experimental::RCollectionClassField::DestroyValue(const Detail::RFieldValue &value, bool dtorOnly)
{
   if (fProperties & TVirtualCollectionProxy::kNeedDelete) {
      TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), value.GetRawPtr());
      for (auto ptr : RCollectionIterableOnce{value.GetRawPtr(), fIFuncsWrite, fProxy.get(),
                                              (fCollectionType == kSTLvector ? fItemSize : 0U)}) {
         auto itemValue = fSubFields[0]->CaptureValue(ptr);
         fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
      }
   }
   fProxy->Destructor(value.GetRawPtr(), true /* dtorOnly */);
   if (!dtorOnly)
      free(value.GetRawPtr());
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RCollectionClassField::CaptureValue(void *where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

std::vector<ROOT::Experimental::Detail::RFieldValue>
ROOT::Experimental::RCollectionClassField::SplitValue(const Detail::RFieldValue &value) const
{
   std::vector<Detail::RFieldValue> result;
   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), value.GetRawPtr());
   for (auto ptr : RCollectionIterableOnce{value.GetRawPtr(), fIFuncsWrite, fProxy.get(),
                                           (fCollectionType == kSTLvector ? fItemSize : 0U)}) {
      result.emplace_back(fSubFields[0]->CaptureValue(ptr));
   }
   return result;
}

void ROOT::Experimental::RCollectionClassField::CommitCluster()
{
   fNWritten = 0;
}

void ROOT::Experimental::RCollectionClassField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitCollectionClassField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RRecordField::RRecordField(std::string_view fieldName,
                                               std::vector<std::unique_ptr<Detail::RFieldBase>> &&itemFields,
                                               const std::vector<std::size_t> &offsets, std::string_view typeName)
   : ROOT::Experimental::Detail::RFieldBase(fieldName, typeName, ENTupleStructure::kRecord, false /* isSimple */),
     fOffsets(offsets)
{
   fTraits |= kTraitTrivialType;
   for (auto &item : itemFields) {
      fMaxAlignment = std::max(fMaxAlignment, item->GetAlignment());
      fSize += GetItemPadding(fSize, item->GetAlignment()) + item->GetValueSize();
      fTraits &= item->GetTraits();
      Attach(std::move(item));
   }
}

ROOT::Experimental::RRecordField::RRecordField(std::string_view fieldName,
                                               std::vector<std::unique_ptr<Detail::RFieldBase>> &&itemFields)
   : ROOT::Experimental::Detail::RFieldBase(fieldName, "", ENTupleStructure::kRecord, false /* isSimple */)
{
   fTraits |= kTraitTrivialType;
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

ROOT::Experimental::RRecordField::RRecordField(std::string_view fieldName,
                                               std::vector<std::unique_ptr<Detail::RFieldBase>> &itemFields)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields))
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

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RRecordField::CloneImpl(std::string_view newName) const
{
   std::vector<std::unique_ptr<Detail::RFieldBase>> cloneItems;
   for (auto &item : fSubFields)
      cloneItems.emplace_back(item->Clone(item->GetName()));
   return std::unique_ptr<RRecordField>(new RRecordField(newName, std::move(cloneItems), fOffsets, GetType()));
}

std::size_t ROOT::Experimental::RRecordField::AppendImpl(const Detail::RFieldValue &value) {
   std::size_t nbytes = 0;
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      auto memberValue = fSubFields[i]->CaptureValue(value.Get<unsigned char>() + fOffsets[i]);
      nbytes += fSubFields[i]->Append(memberValue);
   }
   return nbytes;
}

void ROOT::Experimental::RRecordField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      auto memberValue = fSubFields[i]->CaptureValue(value->Get<unsigned char>() + fOffsets[i]);
      fSubFields[i]->Read(globalIndex, &memberValue);
   }
}

void ROOT::Experimental::RRecordField::ReadInClusterImpl(const RClusterIndex &clusterIndex, Detail::RFieldValue *value)
{
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      auto memberValue = fSubFields[i]->CaptureValue(value->Get<unsigned char>() + fOffsets[i]);
      fSubFields[i]->Read(clusterIndex, &memberValue);
   }
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RRecordField::GenerateValue(void *where)
{
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      fSubFields[i]->GenerateValue(static_cast<unsigned char *>(where) + fOffsets[i]);
   }
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

void ROOT::Experimental::RRecordField::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      auto memberValue = fSubFields[i]->CaptureValue(value.Get<unsigned char>() + fOffsets[i]);
      fSubFields[i]->DestroyValue(memberValue, true /* dtorOnly */);
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
   std::vector<Detail::RFieldValue> result;
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      result.emplace_back(fSubFields[i]->CaptureValue(value.Get<unsigned char>() + fOffsets[i]));
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
   return nbytes + fColumns[0]->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RVectorField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   auto typedValue = value->Get<std::vector<char>>();

   ClusterSize_t nItems;
   RClusterIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   if (fSubFields[0]->GetTraits() & kTraitTrivialType) {
      typedValue->resize(nItems * fItemSize);
   } else {
      // See "semantics of reading non-trivial objects" in RNTuple's architecture.md
      const auto oldNItems = typedValue->size() / fItemSize;
      const bool canRealloc = oldNItems < nItems;
      bool allDeallocated = false;
      if (!(fSubFields[0]->GetTraits() & kTraitTriviallyDestructible)) {
         allDeallocated = canRealloc;
         for (std::size_t i = allDeallocated ? 0 : nItems; i < oldNItems; ++i) {
            auto itemValue = fSubFields[0]->CaptureValue(typedValue->data() + (i * fItemSize));
            fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
         }
      }
      typedValue->resize(nItems * fItemSize);
      if (!(fSubFields[0]->GetTraits() & kTraitTriviallyConstructible)) {
         for (std::size_t i = allDeallocated ? 0 : oldNItems; i < nItems; ++i) {
            fSubFields[0]->GenerateValue(typedValue->data() + (i * fItemSize));
         }
      }
   }

   for (std::size_t i = 0; i < nItems; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(typedValue->data() + (i * fItemSize));
      fSubFields[0]->Read(collectionStart + i, &itemValue);
   }
}

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RVectorField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RVectorField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RVectorField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RVectorField::GenerateValue(void* where)
{
   return Detail::RFieldValue(this, reinterpret_cast<std::vector<char>*>(where));
}

void ROOT::Experimental::RVectorField::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   auto vec = static_cast<std::vector<char>*>(value.GetRawPtr());
   R__ASSERT((vec->size() % fItemSize) == 0);
   if (!(fSubFields[0]->GetTraits() & kTraitTriviallyDestructible)) {
      auto nItems = vec->size() / fItemSize;
      for (unsigned i = 0; i < nItems; ++i) {
         auto itemValue = fSubFields[0]->CaptureValue(vec->data() + (i * fItemSize));
         fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
      }
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
   return nbytes + fColumns[0]->GetElement()->GetPackedSize();
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

   // See "semantics of reading non-trivial objects" in RNTuple's architecture.md for details
   // on the element construction/destrution.
   const bool needsConstruct = !(fSubFields[0]->GetTraits() & kTraitTriviallyConstructible);
   const bool needsDestruct = !(fSubFields[0]->GetTraits() & kTraitTriviallyDestructible);

   // Destroy excess elements, if any
   if (needsDestruct) {
      for (std::size_t i = nItems; i < oldSize; ++i) {
         auto itemValue = fSubFields[0]->CaptureValue(begin + (i * fItemSize));
         fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
      }
   }

   // Resize RVec (capacity and size)
   if (std::int32_t(nItems) > *capacityPtr) { // must reallocate
      // Destroy old elements: useless work for trivial types, but in case the element type's constructor
      // allocates memory we need to release it here to avoid memleaks (e.g. if this is an RVec<RVec<int>>)
      if (needsDestruct) {
         for (std::size_t i = 0u; i < oldSize; ++i) {
            auto itemValue = fSubFields[0]->CaptureValue(begin + (i * fItemSize));
            fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
         }
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
      if (needsConstruct) {
         for (std::size_t i = 0u; i < oldSize; ++i)
            fSubFields[0]->GenerateValue(begin + (i * fItemSize));
      }
   }
   *sizePtr = nItems;

   // Placement new for new elements, if any
   if (needsConstruct) {
      for (std::size_t i = oldSize; i < nItems; ++i)
         fSubFields[0]->GenerateValue(begin + (i * fItemSize));
   }

   // Read the new values into the collection elements
   for (std::size_t i = 0; i < nItems; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(begin + (i * fItemSize));
      fSubFields[0]->Read(collectionStart + i, &itemValue);
   }
}

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RRVecField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RRVecField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RRVecField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
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
   if (!(fSubFields[0]->GetTraits() & kTraitTriviallyDestructible)) {
      for (std::int32_t i = 0; i < *sizePtr; ++i) {
         auto elementValue = fSubFields[0]->CaptureValue(begin + i * fItemSize);
         fSubFields[0]->DestroyValue(elementValue, true /* dtorOnly */);
      }
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
   return count + fColumns[0]->GetElement()->GetPackedSize();
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

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::vector<bool>>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
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
   fTraits |= itemField->GetTraits() & ~kTraitMappable;
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
      auto itemValue = fSubFields[0]->CaptureValue(arrayPtr + (i * fItemSize));
      fSubFields[0]->Read(globalIndex * fArrayLength + i, &itemValue);
   }
}

void ROOT::Experimental::RArrayField::ReadInClusterImpl(const RClusterIndex &clusterIndex, Detail::RFieldValue *value)
{
   auto arrayPtr = value->Get<unsigned char>();
   for (unsigned i = 0; i < fArrayLength; ++i) {
      auto itemValue = fSubFields[0]->CaptureValue(arrayPtr + (i * fItemSize));
      fSubFields[0]->Read(RClusterIndex(clusterIndex.GetClusterId(), clusterIndex.GetIndex() * fArrayLength + i),
                          &itemValue);
   }
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RArrayField::GenerateValue(void *where)
{
   if (fSubFields[0]->GetTraits() & kTraitTriviallyConstructible)
      return Detail::RFieldValue(true /* captureFlag */, this, where);

   auto arrayPtr = reinterpret_cast<unsigned char *>(where);
   for (unsigned i = 0; i < fArrayLength; ++i) {
      fSubFields[0]->GenerateValue(arrayPtr + (i * fItemSize));
   }
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

void ROOT::Experimental::RArrayField::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   auto arrayPtr = value.Get<unsigned char>();
   if (!(fSubFields[0]->GetTraits() & kTraitTriviallyDestructible)) {
      for (unsigned i = 0; i < fArrayLength; ++i) {
         auto itemValue = fSubFields[0]->CaptureValue(arrayPtr + (i * fItemSize));
         fSubFields[0]->DestroyValue(itemValue, true /* dtorOnly */);
      }
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

ROOT::Experimental::RBitsetField::RBitsetField(std::string_view fieldName, std::size_t N)
   : ROOT::Experimental::Detail::RFieldBase(fieldName, "std::bitset<" + std::to_string(N) + ">",
                                            ENTupleStructure::kLeaf, false /* isSimple */, N),
     fN(N)
{
   fTraits |= kTraitTriviallyDestructible;
}

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RBitsetField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kBit}}, {});
   return representations;
}

void ROOT::Experimental::RBitsetField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<bool>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RBitsetField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<bool>(RColumnModel(onDiskTypes[0]), 0));
}

std::size_t ROOT::Experimental::RBitsetField::AppendImpl(const Detail::RFieldValue &value)
{
   const auto *asULongArray = value.Get<Word_t>();
   bool elementValue;
   Detail::RColumnElement<bool> element(&elementValue);
   std::size_t i = 0;
   for (std::size_t word = 0; word < (fN + kBitsPerWord - 1) / kBitsPerWord; ++word) {
      for (std::size_t mask = 0; (mask < kBitsPerWord) && (i < fN); ++mask, ++i) {
         elementValue = (asULongArray[word] & (static_cast<Word_t>(1) << mask)) != 0;
         fColumns[0]->Append(element);
      }
   }
   return fN;
}

void ROOT::Experimental::RBitsetField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   auto *asULongArray = value->Get<Word_t>();
   bool elementValue;
   Detail::RColumnElement<bool> element(&elementValue);
   for (std::size_t i = 0; i < fN; ++i) {
      fColumns[0]->Read(globalIndex * fN + i, &element);
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
   // The variant needs to initialize its own tag member
   fTraits |= kTraitTriviallyDestructible & ~kTraitTriviallyConstructible;

   auto nFields = itemFields.size();
   R__ASSERT(nFields > 0);
   fNWritten.resize(nFields, 0);
   for (unsigned int i = 0; i < nFields; ++i) {
      fMaxItemSize = std::max(fMaxItemSize, itemFields[i]->GetValueSize());
      fMaxAlignment = std::max(fMaxAlignment, itemFields[i]->GetAlignment());
      fTraits &= itemFields[i]->GetTraits();
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

   // If `tag` equals 0, the variant is in the invalid state, i.e, it does not hold any of the valid alternatives in
   // the type list.  This happens, e.g., if the field was late added; in this case, keep the invalid tag, which makes
   // any `std::holds_alternative<T>` check fail later.
   if (R__likely(tag > 0)) {
      auto itemValue = fSubFields[tag - 1]->GenerateValue(value->GetRawPtr());
      fSubFields[tag - 1]->Read(variantIndex, &itemValue);
   }
   SetTag(value->GetRawPtr(), tag);
}

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RVariantField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSwitch}}, {{}});
   return representations;
}

void ROOT::Experimental::RVariantField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<RColumnSwitch>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RVariantField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<RColumnSwitch>(RColumnModel(onDiskTypes[0]), 0));
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

ROOT::Experimental::RNullableField::RNullableField(std::string_view fieldName, std::string_view typeName,
                                                   std::unique_ptr<Detail::RFieldBase> itemField)
   : ROOT::Experimental::Detail::RFieldBase(fieldName, typeName, ENTupleStructure::kCollection, false /* isSimple */)
{
   Attach(std::move(itemField));
}

ROOT::Experimental::RNullableField::~RNullableField()
{
   if (fDefaultItemValue.GetField()) {
      fDefaultItemValue.GetField()->DestroyValue(fDefaultItemValue);
   }
}

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RNullableField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32},
       {EColumnType::kBit}}, {});
   return representations;
}

void ROOT::Experimental::RNullableField::GenerateColumnsImpl()
{
   if (HasDefaultColumnRepresentative()) {
      if (fSubFields[0]->GetValueSize() < 4) {
         SetColumnRepresentative({EColumnType::kBit});
      }
   }
   if (IsDense()) {
      fDefaultItemValue = fSubFields[0]->GenerateValue();
      fColumns.emplace_back(Detail::RColumn::Create<bool>(RColumnModel(EColumnType::kBit), 0));
   } else {
      fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
   }
}

void ROOT::Experimental::RNullableField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   if (onDiskTypes[0] == EColumnType::kBit) {
      fColumns.emplace_back(Detail::RColumn::Create<bool>(RColumnModel(EColumnType::kBit), 0));
   } else {
      fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
   }
}

std::size_t ROOT::Experimental::RNullableField::AppendNull()
{
   if (IsDense()) {
      bool mask = false;
      Detail::RColumnElement<bool> maskElement(&mask);
      fPrincipalColumn->Append(maskElement);
      return 1 + fSubFields[0]->Append(fDefaultItemValue);
   } else {
      Detail::RColumnElement<ClusterSize_t> offsetElement(&fNWritten);
      fPrincipalColumn->Append(offsetElement);
      return sizeof(ClusterSize_t);
   }
}

std::size_t ROOT::Experimental::RNullableField::AppendValue(const Detail::RFieldValue &value)
{
   auto nbytesItem = fSubFields[0]->Append(value);
   if (IsDense()) {
      bool mask = true;
      Detail::RColumnElement<bool> maskElement(&mask);
      fPrincipalColumn->Append(maskElement);
      return 1 + nbytesItem;
   } else {
      fNWritten++;
      Detail::RColumnElement<ClusterSize_t> offsetElement(&fNWritten);
      fPrincipalColumn->Append(offsetElement);
      return sizeof(ClusterSize_t) + nbytesItem;
   }
}

ROOT::Experimental::RClusterIndex ROOT::Experimental::RNullableField::GetItemIndex(NTupleSize_t globalIndex)
{
   RClusterIndex nullIndex;
   if (IsDense()) {
      const bool isValidItem = *fPrincipalColumn->Map<bool>(globalIndex);
      return isValidItem ? fPrincipalColumn->GetClusterIndex(globalIndex) : nullIndex;
   } else {
      RClusterIndex collectionStart;
      ClusterSize_t collectionSize;
      fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &collectionSize);
      return (collectionSize == 0) ? nullIndex : collectionStart;
   }
}

void ROOT::Experimental::RNullableField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitNullableField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RUniquePtrField::RUniquePtrField(std::string_view fieldName, std::string_view typeName,
                                                     std::unique_ptr<Detail::RFieldBase> itemField)
   : RNullableField(fieldName, typeName, std::move(itemField))
{
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RUniquePtrField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetName());
   return std::make_unique<RUniquePtrField>(newName, GetType(), std::move(newItemField));
}

std::size_t ROOT::Experimental::RUniquePtrField::AppendImpl(const Detail::RFieldValue &value)
{
   auto typedValue = value.Get<std::unique_ptr<char>>();
   if (*typedValue) {
      auto itemValue = fSubFields[0]->CaptureValue(typedValue->get());
      return AppendValue(itemValue);
   } else {
      return AppendNull();
   }
}

void ROOT::Experimental::RUniquePtrField::ReadGlobalImpl(NTupleSize_t globalIndex, Detail::RFieldValue *value)
{
   auto ptr = value->Get<std::unique_ptr<char>>();
   bool isValidValue = static_cast<bool>(*ptr);

   auto itemIndex = GetItemIndex(globalIndex);
   bool isValidItem = itemIndex.GetIndex() != kInvalidClusterIndex;

   Detail::RFieldValue itemValue;
   if (isValidValue)
      itemValue = fSubFields[0]->CaptureValue(ptr->get());

   if (isValidValue && !isValidItem) {
      ptr->release();
      fSubFields[0]->DestroyValue(itemValue, false /* dtorOnly */);
      return;
   }

   if (!isValidItem) // On-disk value missing; nothing else to do
      return;

   if (!isValidValue) {
      itemValue = fSubFields[0]->GenerateValue();
      ptr->reset(itemValue.Get<char>());
   }

   fSubFields[0]->Read(itemIndex, &itemValue);
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RUniquePtrField::GenerateValue(void *where)
{
   return Detail::RFieldValue(this, reinterpret_cast<std::unique_ptr<char> *>(where));
}

void ROOT::Experimental::RUniquePtrField::DestroyValue(const Detail::RFieldValue &value, bool dtorOnly)
{
   auto ptr = value.Get<std::unique_ptr<char>>();
   if (*ptr) {
      auto itemValue = fSubFields[0]->CaptureValue(ptr->get());
      fSubFields[0]->DestroyValue(itemValue, false /* dtorOnly */);
      ptr->release();
   }
   if (!dtorOnly)
      free(ptr);
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RUniquePtrField::CaptureValue(void *where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, where);
}

std::vector<ROOT::Experimental::Detail::RFieldValue>
ROOT::Experimental::RUniquePtrField::SplitValue(const Detail::RFieldValue &value) const
{
   std::vector<Detail::RFieldValue> result;
   auto ptr = value.Get<std::unique_ptr<char>>();
   if (*ptr) {
      auto itemValue = fSubFields[0]->CaptureValue(ptr->get());
      result.emplace_back(itemValue);
   }
   return result;
}

//------------------------------------------------------------------------------

std::string ROOT::Experimental::RPairField::RPairField::GetTypeList(
   const std::array<std::unique_ptr<Detail::RFieldBase>, 2> &itemFields)
{
   return itemFields[0]->GetType() + "," + itemFields[1]->GetType();
}

ROOT::Experimental::RPairField::RPairField(std::string_view fieldName,
                                           std::array<std::unique_ptr<Detail::RFieldBase>, 2> &&itemFields,
                                           const std::array<std::size_t, 2> &offsets)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields), offsets,
                                      "std::pair<" + GetTypeList(itemFields) + ">")
{
}

ROOT::Experimental::RPairField::RPairField(std::string_view fieldName,
                                           std::array<std::unique_ptr<Detail::RFieldBase>, 2> &itemFields)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields), {},
                                      "std::pair<" + GetTypeList(itemFields) + ">")
{
   // ISO C++ does not guarantee any specific layout for `std::pair`; query TClass for the member offsets
   fClass = TClass::GetClass(GetType().c_str());
   if (!fClass)
      throw RException(R__FAIL("cannot get type information for " + GetType()));
   fSize = fClass->Size();
   fOffsets[0] = fClass->GetDataMember("first")->GetOffset();
   fOffsets[1] = fClass->GetDataMember("second")->GetOffset();
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RPairField::CloneImpl(std::string_view newName) const
{
   std::array<std::unique_ptr<Detail::RFieldBase>, 2> items{fSubFields[0]->Clone(fSubFields[0]->GetName()),
                                                            fSubFields[1]->Clone(fSubFields[1]->GetName())};

   std::unique_ptr<RPairField> result(new RPairField(newName, std::move(items), {fOffsets[0], fOffsets[1]}));
   result->fClass = fClass;
   return result;
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RPairField::GenerateValue(void *where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, fClass->New(where));
}

void ROOT::Experimental::RPairField::DestroyValue(const Detail::RFieldValue& value, bool dtorOnly)
{
   fClass->Destructor(value.GetRawPtr(), true /* dtorOnly */);
   if (!dtorOnly)
      free(value.GetRawPtr());
}

//------------------------------------------------------------------------------

std::string ROOT::Experimental::RTupleField::RTupleField::GetTypeList(
   const std::vector<std::unique_ptr<Detail::RFieldBase>> &itemFields)
{
   std::string result;
   if (itemFields.empty())
      throw RException(R__FAIL("the type list for std::tuple must have at least one element"));
   for (size_t i = 0; i < itemFields.size(); ++i) {
      result += itemFields[i]->GetType() + ",";
   }
   result.pop_back();          // remove trailing comma
   return result;
}

ROOT::Experimental::RTupleField::RTupleField(std::string_view fieldName,
                                             std::vector<std::unique_ptr<Detail::RFieldBase>> &&itemFields,
                                             const std::vector<std::size_t> &offsets)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields), offsets,
                                      "std::tuple<" + GetTypeList(itemFields) + ">")
{
}

ROOT::Experimental::RTupleField::RTupleField(std::string_view fieldName,
                                             std::vector<std::unique_ptr<Detail::RFieldBase>> &itemFields)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields), {},
                                      "std::tuple<" + GetTypeList(itemFields) + ">")
{
   fClass = TClass::GetClass(GetType().c_str());
   if (!fClass)
      throw RException(R__FAIL("cannot get type information for " + GetType()));
   fSize = fClass->Size();

   // ISO C++ does not guarantee neither specific layout nor member names for `std::tuple`.  However, most
   // implementations including libstdc++ (gcc), libc++ (llvm), and MSVC name members as `_0`, `_1`, ..., `_N-1`,
   // following the order of the type list.
   // Use TClass to get their offsets; in case a particular `std::tuple` implementation does not define such
   // members, the assertion below will fail.
   for (unsigned i = 0; i < fSubFields.size(); ++i) {
      std::string memberName("_" + std::to_string(i));
      auto member = fClass->GetRealData(memberName.c_str());
      if (!member)
         throw RException(R__FAIL(memberName + ": no such member"));
      fOffsets.push_back(member->GetThisOffset());
   }
}

std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>
ROOT::Experimental::RTupleField::CloneImpl(std::string_view newName) const
{
   std::vector<std::unique_ptr<Detail::RFieldBase>> items;
   for (const auto &item : fSubFields)
      items.push_back(item->Clone(item->GetName()));

   std::unique_ptr<RTupleField> result(new RTupleField(newName, std::move(items), fOffsets));
   result->fClass = fClass;
   return result;
}

ROOT::Experimental::Detail::RFieldValue ROOT::Experimental::RTupleField::GenerateValue(void *where)
{
   return Detail::RFieldValue(true /* captureFlag */, this, fClass->New(where));
}

void ROOT::Experimental::RTupleField::DestroyValue(const Detail::RFieldValue &value, bool dtorOnly)
{
   fClass->Destructor(value.GetRawPtr(), true /* dtorOnly */);
   if (!dtorOnly)
      free(value.GetRawPtr());
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

const ROOT::Experimental::Detail::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RCollectionField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RCollectionField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RCollectionField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Detail::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
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
