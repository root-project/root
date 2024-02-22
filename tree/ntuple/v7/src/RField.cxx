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
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleCollectionWriter.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TBaseClass.h>
#include <TClass.h>
#include <TClassEdit.h>
#include <TCollection.h>
#include <TDataMember.h>
#include <TEnum.h>
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
#include <memory>
#include <new> // hardware_destructive_interference_size
#include <type_traits>
#include <unordered_map>

namespace {

const std::unordered_map<std::string_view, std::string_view> typeTranslationMap{
   {"Bool_t",   "bool"},
   {"Float_t",  "float"},
   {"Double_t", "double"},
   {"string",   "std::string"},

   {"byte",          "std::byte"},
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

   // FIXME: Long_t and ULong_t are 32-bit on 64-bit Windows.
   {"Long_t",        "std::int64_t"},
   {"Long64_t",      "std::int64_t"},
   {"int64_t",       "std::int64_t"},
   {"long",          "std::int64_t"},
   {"ULong_t",       "std::uint64_t"},
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
      auto posLBrace = typeName.find_last_of('[', posRBrace);
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

/// Return the canonical name of a type, resolving typedefs to their underlying types if needed.  A canonical type has
/// typedefs stripped out from the type name.
std::string GetCanonicalTypeName(const std::string &typeName)
{
   // The following types are asummed to be canonical names; thus, do not perform `typedef` resolution on those
   if (typeName == "ROOT::Experimental::ClusterSize_t" || typeName.substr(0, 5) == "std::" ||
       typeName.substr(0, 39) == "ROOT::Experimental::RNTupleCardinality<")
      return typeName;

   return TClassEdit::ResolveTypedef(typeName.c_str());
}

/// Applies type name normalization rules that lead to the final name used to create a RField, e.g. transforms
/// `unsigned int` to `std::uint32_t` or `const vector<T>` to `std::vector<T>`.  Specifically, `const` / `volatile`
/// qualifiers are removed, integral types such as `unsigned int` or `long` are translated to fixed-length integer types
/// (e.g. `std::uint32_t`), and `std::` is added to fully qualify known types in the `std` namespace.  The same happens
/// to `ROOT::RVec` which is normalized to `ROOT::VecOps::RVec`.
std::string GetNormalizedTypeName(const std::string &typeName)
{
   std::string normalizedType{TClassEdit::CleanType(typeName.c_str(), /*mode=*/2)};

   if (auto it = typeTranslationMap.find(normalizedType); it != typeTranslationMap.end())
      normalizedType = it->second;

   if (normalizedType.substr(0, 7) == "vector<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 6) == "array<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 8) == "variant<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 5) == "pair<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 6) == "tuple<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 7) == "bitset<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 11) == "unique_ptr<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 4) == "set<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 14) == "unordered_set<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 4) == "map<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 14) == "unordered_map<")
      normalizedType = "std::" + normalizedType;
   if (normalizedType.substr(0, 7) == "atomic<")
      normalizedType = "std::" + normalizedType;

   if (normalizedType.substr(0, 11) == "ROOT::RVec<")
      normalizedType = "ROOT::VecOps::RVec<" + normalizedType.substr(11);

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

std::tuple<const void *const *, const std::int32_t *, const std::int32_t *> GetRVecDataMembers(const void *rvecPtr)
{
   return {GetRVecDataMembers(const_cast<void *>(rvecPtr))};
}

/// Applies the field IDs from 'from' to 'to', where from and to are expected to be each other's clones.
/// Used in RClassField and RCollectionClassField cloning. In these classes, we don't clone the subfields
/// but we recreate them. Therefore, the on-disk IDs need to be fixed up.
void SyncFieldIDs(const ROOT::Experimental::RFieldBase &from, ROOT::Experimental::RFieldBase &to)
{
   auto iFrom = from.cbegin();
   auto iTo = to.begin();
   for (; iFrom != from.cend(); ++iFrom, ++iTo) {
      iTo->SetOnDiskId(iFrom->GetOnDiskId());
   }
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

void ROOT::Experimental::Internal::CallCommitClusterOnField(RFieldBase &field)
{
   field.CommitCluster();
}
void ROOT::Experimental::Internal::CallConnectPageSinkOnField(RFieldBase &field, Internal::RPageSink &sink,
                                                              NTupleSize_t firstEntry)
{
   field.ConnectPageSink(sink, firstEntry);
}
void ROOT::Experimental::Internal::CallConnectPageSourceOnField(RFieldBase &field, Internal::RPageSource &source)
{
   field.ConnectPageSource(source);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RFieldBase::RColumnRepresentations::RColumnRepresentations()
{
   // A single representations with an empty set of columns
   fSerializationTypes.emplace_back(ColumnRepresentation_t());
   fDeserializationTypes.emplace_back(ColumnRepresentation_t());
}

ROOT::Experimental::RFieldBase::RColumnRepresentations::RColumnRepresentations(
   const TypesList_t &serializationTypes, const TypesList_t &deserializationExtraTypes)
   : fSerializationTypes(serializationTypes), fDeserializationTypes(serializationTypes)
{
   fDeserializationTypes.insert(fDeserializationTypes.end(),
                                deserializationExtraTypes.begin(), deserializationExtraTypes.end());
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RFieldBase::RValue::BindRawPtr(void *rawPtr)
{
   /// Empty shared pointer, used as the basis for the aliasing shared pointer constructor around rawPtr.
   /// Note that as a result of BindRawPtr, fObjPtr will be non-empty but have use count zero.
   static std::shared_ptr<void> fgRawPtrCtrlBlock;
   fObjPtr = std::shared_ptr<void>(fgRawPtrCtrlBlock, rawPtr);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RFieldBase::RBulk::RBulk(RBulk &&other)
   : fField(other.fField),
     fValueSize(other.fValueSize),
     fCapacity(other.fCapacity),
     fSize(other.fSize),
     fIsAdopted(other.fIsAdopted),
     fNValidValues(other.fNValidValues),
     fFirstIndex(other.fFirstIndex)
{
   std::swap(fDeleter, other.fDeleter);
   std::swap(fValues, other.fValues);
   std::swap(fMaskAvail, other.fMaskAvail);
}

ROOT::Experimental::RFieldBase::RBulk &ROOT::Experimental::RFieldBase::RBulk::operator=(RBulk &&other)
{
   std::swap(fField, other.fField);
   std::swap(fDeleter, other.fDeleter);
   std::swap(fValues, other.fValues);
   std::swap(fValueSize, other.fValueSize);
   std::swap(fCapacity, other.fCapacity);
   std::swap(fSize, other.fSize);
   std::swap(fIsAdopted, other.fIsAdopted);
   std::swap(fMaskAvail, other.fMaskAvail);
   std::swap(fNValidValues, other.fNValidValues);
   std::swap(fFirstIndex, other.fFirstIndex);
   return *this;
}

ROOT::Experimental::RFieldBase::RBulk::~RBulk()
{
   if (fValues)
      ReleaseValues();
}

void ROOT::Experimental::RFieldBase::RBulk::ReleaseValues()
{
   if (fIsAdopted)
      return;

   if (fField->GetTraits() & RFieldBase::kTraitTriviallyDestructible) {
      free(fValues);
      return;
   }

   for (std::size_t i = 0; i < fCapacity; ++i) {
      fDeleter->operator()(GetValuePtrAt(i), true /* dtorOnly */);
   }
   free(fValues);
}

void ROOT::Experimental::RFieldBase::RBulk::Reset(RClusterIndex firstIndex, std::size_t size)
{
   if (fCapacity < size) {
      if (fIsAdopted) {
         throw RException(R__FAIL("invalid attempt to bulk read beyond the adopted buffer"));
      }
      ReleaseValues();
      fValues = malloc(size * fValueSize);

      if (!(fField->GetTraits() & RFieldBase::kTraitTriviallyConstructible)) {
         for (std::size_t i = 0; i < size; ++i) {
            fField->ConstructValue(GetValuePtrAt(i));
         }
      }

      fMaskAvail = std::make_unique<bool[]>(size);
      fCapacity = size;
   }

   std::fill(fMaskAvail.get(), fMaskAvail.get() + size, false);
   fNValidValues = 0;

   fFirstIndex = firstIndex;
   fSize = size;
}

void ROOT::Experimental::RFieldBase::RBulk::CountValidValues()
{
   fNValidValues = 0;
   for (std::size_t i = 0; i < fSize; ++i)
      fNValidValues += static_cast<std::size_t>(fMaskAvail[i]);
}

void ROOT::Experimental::RFieldBase::RBulk::AdoptBuffer(void *buf, std::size_t capacity)
{
   ReleaseValues();
   fValues = buf;
   fCapacity = capacity;
   fSize = capacity;

   fMaskAvail = std::make_unique<bool[]>(capacity);

   fFirstIndex = RClusterIndex();

   fIsAdopted = true;
}

//------------------------------------------------------------------------------

void ROOT::Experimental::RFieldBase::RCreateObjectDeleter<void>::operator()(void *)
{
   R__LOG_WARNING(NTupleLog()) << "possibly leaking object from RField<T>::CreateObject<void>";
}

template <>
std::unique_ptr<void, typename ROOT::Experimental::RFieldBase::RCreateObjectDeleter<void>::deleter>
ROOT::Experimental::RFieldBase::CreateObject<void>() const
{
   static RCreateObjectDeleter<void>::deleter gDeleter;
   return std::unique_ptr<void, RCreateObjectDeleter<void>::deleter>(CreateObjectRawPtr(), gDeleter);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RFieldBase::RFieldBase(std::string_view name, std::string_view type, ENTupleStructure structure,
                                           bool isSimple, std::size_t nRepetitions)
   : fName(name),
     fType(type),
     fStructure(structure),
     fNRepetitions(nRepetitions),
     fIsSimple(isSimple),
     fParent(nullptr),
     fPrincipalColumn(nullptr),
     fTraits(isSimple ? kTraitMappable : 0)
{
}

std::string ROOT::Experimental::RFieldBase::GetQualifiedFieldName() const
{
   std::string result = GetFieldName();
   auto parent = GetParent();
   while (parent && !parent->GetFieldName().empty()) {
      result = parent->GetFieldName() + "." + result;
      parent = parent->GetParent();
   }
   return result;
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RFieldBase>>
ROOT::Experimental::RFieldBase::Create(const std::string &fieldName, const std::string &typeName)
{
   auto typeAlias = GetNormalizedTypeName(typeName);
   auto canonicalType = GetNormalizedTypeName(GetCanonicalTypeName(typeAlias));
   return R__FORWARD_RESULT(RFieldBase::Create(fieldName, canonicalType, typeAlias));
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RFieldBase>>
ROOT::Experimental::RFieldBase::Create(const std::string &fieldName, const std::string &canonicalType,
                                       const std::string &typeAlias)
{
   if (canonicalType.empty())
      return R__FAIL("no type name specified for Field " + fieldName);

   if (auto [arrayBaseType, arraySizes] = ParseArrayType(canonicalType); !arraySizes.empty()) {
      std::unique_ptr<RFieldBase> arrayField = Create("_0", arrayBaseType).Unwrap();
      for (int i = arraySizes.size() - 1; i >= 0; --i) {
         arrayField = std::make_unique<RArrayField>((i == 0) ? fieldName : "_0", std::move(arrayField), arraySizes[i]);
      }
      return arrayField;
   }

   std::unique_ptr<ROOT::Experimental::RFieldBase> result;

   if (canonicalType == "ROOT::Experimental::ClusterSize_t") {
      result = std::make_unique<RField<ClusterSize_t>>(fieldName);
   } else if (canonicalType == "bool") {
      result = std::make_unique<RField<bool>>(fieldName);
   } else if (canonicalType == "char") {
      result = std::make_unique<RField<char>>(fieldName);
   } else if (canonicalType == "std::byte") {
      result = std::make_unique<RField<std::byte>>(fieldName);
   } else if (canonicalType == "std::int8_t") {
      result = std::make_unique<RField<std::int8_t>>(fieldName);
   } else if (canonicalType == "std::uint8_t") {
      result = std::make_unique<RField<std::uint8_t>>(fieldName);
   } else if (canonicalType == "std::int16_t") {
      result = std::make_unique<RField<std::int16_t>>(fieldName);
   } else if (canonicalType == "std::uint16_t") {
      result = std::make_unique<RField<std::uint16_t>>(fieldName);
   } else if (canonicalType == "std::int32_t") {
      result = std::make_unique<RField<std::int32_t>>(fieldName);
   } else if (canonicalType == "std::uint32_t") {
      result = std::make_unique<RField<std::uint32_t>>(fieldName);
   } else if (canonicalType == "std::int64_t") {
      result = std::make_unique<RField<std::int64_t>>(fieldName);
   } else if (canonicalType == "std::uint64_t") {
      result = std::make_unique<RField<std::uint64_t>>(fieldName);
   } else if (canonicalType == "float") {
      result = std::make_unique<RField<float>>(fieldName);
   } else if (canonicalType == "double") {
      result = std::make_unique<RField<double>>(fieldName);
   } else if (canonicalType == "Double32_t") {
      result = std::make_unique<RField<double>>(fieldName);
      static_cast<RField<double> *>(result.get())->SetDouble32();
      // Prevent the type alias from being reset by returning early
      return result;
   } else if (canonicalType == "std::string") {
      result = std::make_unique<RField<std::string>>(fieldName);
   } else if (canonicalType == "std::vector<bool>") {
      result = std::make_unique<RField<std::vector<bool>>>(fieldName);
   } else if (canonicalType.substr(0, 12) == "std::vector<") {
      std::string itemTypeName = canonicalType.substr(12, canonicalType.length() - 13);
      auto itemField = Create("_0", itemTypeName);
      result = std::make_unique<RVectorField>(fieldName, itemField.Unwrap());
   } else if (canonicalType.substr(0, 19) == "ROOT::VecOps::RVec<") {
      std::string itemTypeName = canonicalType.substr(19, canonicalType.length() - 20);
      auto itemField = Create("_0", itemTypeName);
      result = std::make_unique<RRVecField>(fieldName, itemField.Unwrap());
   } else if (canonicalType.substr(0, 11) == "std::array<") {
      auto arrayDef = TokenizeTypeList(canonicalType.substr(11, canonicalType.length() - 12));
      R__ASSERT(arrayDef.size() == 2);
      auto arrayLength = std::stoi(arrayDef[1]);
      auto itemField = Create("_0", arrayDef[0]);
      result = std::make_unique<RArrayField>(fieldName, itemField.Unwrap(), arrayLength);
   } else if (canonicalType.substr(0, 13) == "std::variant<") {
      auto innerTypes = TokenizeTypeList(canonicalType.substr(13, canonicalType.length() - 14));
      std::vector<RFieldBase *> items;
      for (unsigned int i = 0; i < innerTypes.size(); ++i) {
         items.emplace_back(Create("_" + std::to_string(i), innerTypes[i]).Unwrap().release());
      }
      result = std::make_unique<RVariantField>(fieldName, items);
   } else if (canonicalType.substr(0, 10) == "std::pair<") {
      auto innerTypes = TokenizeTypeList(canonicalType.substr(10, canonicalType.length() - 11));
      if (innerTypes.size() != 2)
         return R__FAIL("the type list for std::pair must have exactly two elements");
      std::array<std::unique_ptr<RFieldBase>, 2> items{Create("_0", innerTypes[0]).Unwrap(),
                                                       Create("_1", innerTypes[1]).Unwrap()};
      result = std::make_unique<RPairField>(fieldName, items);
   } else if (canonicalType.substr(0, 11) == "std::tuple<") {
      auto innerTypes = TokenizeTypeList(canonicalType.substr(11, canonicalType.length() - 12));
      std::vector<std::unique_ptr<RFieldBase>> items;
      for (unsigned int i = 0; i < innerTypes.size(); ++i) {
         items.emplace_back(Create("_" + std::to_string(i), innerTypes[i]).Unwrap());
      }
      result = std::make_unique<RTupleField>(fieldName, items);
   } else if (canonicalType.substr(0, 12) == "std::bitset<") {
      auto size = std::stoull(canonicalType.substr(12, canonicalType.length() - 13));
      result = std::make_unique<RBitsetField>(fieldName, size);
   } else if (canonicalType.substr(0, 16) == "std::unique_ptr<") {
      std::string itemTypeName = canonicalType.substr(16, canonicalType.length() - 17);
      auto itemField = Create("_0", itemTypeName).Unwrap();
      auto normalizedInnerTypeName = itemField->GetTypeName();
      result = std::make_unique<RUniquePtrField>(fieldName, "std::unique_ptr<" + normalizedInnerTypeName + ">",
                                                 std::move(itemField));
   } else if (canonicalType.substr(0, 9) == "std::set<") {
      std::string itemTypeName = canonicalType.substr(9, canonicalType.length() - 10);
      auto itemField = Create("_0", itemTypeName).Unwrap();
      auto normalizedInnerTypeName = itemField->GetTypeName();
      result =
         std::make_unique<RSetField>(fieldName, "std::set<" + normalizedInnerTypeName + ">", std::move(itemField));
   } else if (canonicalType.substr(0, 19) == "std::unordered_set<") {
      std::string itemTypeName = canonicalType.substr(19, canonicalType.length() - 20);
      auto itemField = Create("_0", itemTypeName).Unwrap();
      auto normalizedInnerTypeName = itemField->GetTypeName();
      result = std::make_unique<RSetField>(fieldName, "std::unordered_set<" + normalizedInnerTypeName + ">",
                                           std::move(itemField));
   } else if (canonicalType.substr(0, 9) == "std::map<") {
      auto innerTypes = TokenizeTypeList(canonicalType.substr(9, canonicalType.length() - 10));
      if (innerTypes.size() != 2)
         return R__FAIL("the type list for std::map must have exactly two elements");

      auto normalizedKeyTypeName = GetNormalizedTypeName(innerTypes[0]);
      auto normalizedValueTypeName = GetNormalizedTypeName(innerTypes[1]);

      auto itemField =
         Create("_0", "std::pair<" + normalizedKeyTypeName + "," + normalizedValueTypeName + ">").Unwrap();
      result = std::make_unique<RMapField>(
         fieldName, "std::map<" + normalizedKeyTypeName + "," + normalizedValueTypeName + ">", std::move(itemField));
   } else if (canonicalType.substr(0, 19) == "std::unordered_map<") {
      auto innerTypes = TokenizeTypeList(canonicalType.substr(19, canonicalType.length() - 20));
      if (innerTypes.size() != 2)
         return R__FAIL("the type list for std::unordered_map must have exactly two elements");

      auto normalizedKeyTypeName = GetNormalizedTypeName(innerTypes[0]);
      auto normalizedValueTypeName = GetNormalizedTypeName(innerTypes[1]);

      auto itemField =
         Create("_0", "std::pair<" + normalizedKeyTypeName + "," + normalizedValueTypeName + ">").Unwrap();
      result = std::make_unique<RMapField>(
         fieldName, "std::unordered_map<" + normalizedKeyTypeName + "," + normalizedValueTypeName + ">",
         std::move(itemField));
   } else if (canonicalType.substr(0, 12) == "std::atomic<") {
      std::string itemTypeName = canonicalType.substr(12, canonicalType.length() - 13);
      auto itemField = Create("_0", itemTypeName).Unwrap();
      auto normalizedInnerTypeName = itemField->GetTypeName();
      result = std::make_unique<RAtomicField>(fieldName, "std::atomic<" + normalizedInnerTypeName + ">",
                                              std::move(itemField));
   } else if (canonicalType.substr(0, 39) == "ROOT::Experimental::RNTupleCardinality<") {
      auto innerTypes = TokenizeTypeList(canonicalType.substr(39, canonicalType.length() - 40));
      if (innerTypes.size() != 1)
         return R__FAIL(std::string("Field ") + fieldName + " has invalid cardinality template: " + canonicalType);
      if (innerTypes[0] == "std::uint32_t") {
         result = std::make_unique<RField<RNTupleCardinality<std::uint32_t>>>(fieldName);
      } else if (innerTypes[0] == "std::uint64_t") {
         result = std::make_unique<RField<RNTupleCardinality<std::uint64_t>>>(fieldName);
      } else {
         return R__FAIL(std::string("Field ") + fieldName + " has invalid cardinality template: " + canonicalType);
      }
   }

   if (!result) {
      auto e = TEnum::GetEnum(canonicalType.c_str());
      if (e != nullptr) {
         result = std::make_unique<REnumField>(fieldName, canonicalType);
      }
   }

   if (!result) {
      auto cl = TClass::GetClass(canonicalType.c_str());
      if (cl != nullptr) {
         if (cl->GetCollectionProxy())
            result = std::make_unique<RProxiedCollectionField>(fieldName, canonicalType);
         else
            result = std::make_unique<RClassField>(fieldName, canonicalType);
      }
   }

   if (result) {
      if (typeAlias != canonicalType)
         result->fTypeAlias = typeAlias;
      return result;
   }
   return R__FAIL(std::string("Field ") + fieldName + " has unknown type " + canonicalType);
}

ROOT::Experimental::RResult<void> ROOT::Experimental::RFieldBase::EnsureValidFieldName(std::string_view fieldName)
{
   if (fieldName.empty()) {
      return R__FAIL("name cannot be empty string \"\"");
   } else if (fieldName.find('.') != std::string::npos) {
      return R__FAIL("name '" + std::string(fieldName) + "' cannot contain dot characters '.'");
   }
   return RResult<void>::Success();
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RFieldBase::GetColumnRepresentations() const
{
   static RColumnRepresentations representations;
   return representations;
}

std::unique_ptr<ROOT::Experimental::RFieldBase> ROOT::Experimental::RFieldBase::Clone(std::string_view newName) const
{
   auto clone = CloneImpl(newName);
   clone->fTypeAlias = fTypeAlias;
   clone->fOnDiskId = fOnDiskId;
   clone->fDescription = fDescription;
   // We can just copy the pointer because fColumnRepresentative points into a static structure
   clone->fColumnRepresentative = fColumnRepresentative;
   return clone;
}

std::size_t ROOT::Experimental::RFieldBase::AppendImpl(const void * /* from */)
{
   R__ASSERT(false && "A non-simple RField must implement its own AppendImpl");
   return 0;
}

void ROOT::Experimental::RFieldBase::ReadGlobalImpl(ROOT::Experimental::NTupleSize_t /*index*/, void * /* to */)
{
   R__ASSERT(false);
}

std::size_t ROOT::Experimental::RFieldBase::ReadBulkImpl(const RBulkSpec &bulkSpec)
{
   const auto valueSize = GetValueSize();
   std::size_t nRead = 0;
   for (std::size_t i = 0; i < bulkSpec.fCount; ++i) {
      // Value not needed
      if (!bulkSpec.fMaskReq[i])
         continue;

      // Value already present
      if (bulkSpec.fMaskAvail[i])
         continue;

      Read(bulkSpec.fFirstIndex + i, reinterpret_cast<unsigned char *>(bulkSpec.fValues) + i * valueSize);
      bulkSpec.fMaskAvail[i] = true;
      nRead++;
   }
   return nRead;
}

void *ROOT::Experimental::RFieldBase::CreateObjectRawPtr() const
{
   void *where = operator new(GetValueSize());
   R__ASSERT(where != nullptr);
   ConstructValue(where);
   return where;
}

ROOT::Experimental::RFieldBase::RValue ROOT::Experimental::RFieldBase::CreateValue()
{
   void *obj = CreateObjectRawPtr();
   return RValue(this, std::shared_ptr<void>(obj, RSharedPtrDeleter(GetDeleter())));
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RFieldBase::SplitValue(const RValue & /*value*/) const
{
   return std::vector<RValue>();
}

void ROOT::Experimental::RFieldBase::Attach(std::unique_ptr<ROOT::Experimental::RFieldBase> child)
{
   // Note that during a model update, new fields will be attached to the zero field. The zero field, however,
   // does not change its inital state because only its sub fields get connected by RPageSink::UpdateSchema.
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("invalid attempt to attach subfield to already connected field"));
   child->fParent = this;
   fSubFields.emplace_back(std::move(child));
}

ROOT::Experimental::NTupleSize_t
ROOT::Experimental::RFieldBase::EntryToColumnElementIndex(NTupleSize_t globalIndex) const
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

std::vector<ROOT::Experimental::RFieldBase *> ROOT::Experimental::RFieldBase::GetSubFields()
{
   std::vector<RFieldBase *> result;
   result.reserve(fSubFields.size());
   for (const auto &f : fSubFields) {
      result.emplace_back(f.get());
   }
   return result;
}

std::vector<const ROOT::Experimental::RFieldBase *> ROOT::Experimental::RFieldBase::GetSubFields() const
{
   std::vector<const RFieldBase *> result;
   result.reserve(fSubFields.size());
   for (const auto &f : fSubFields) {
      result.emplace_back(f.get());
   }
   return result;
}

void ROOT::Experimental::RFieldBase::CommitCluster()
{
   for (auto& column : fColumns) {
      column->Flush();
   }
   CommitClusterImpl();
}

void ROOT::Experimental::RFieldBase::SetDescription(std::string_view description)
{
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("cannot set field description once field is connected"));
   fDescription = std::string(description);
}

void ROOT::Experimental::RFieldBase::SetOnDiskId(DescriptorId_t id)
{
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("cannot set field ID once field is connected"));
   fOnDiskId = id;
}

const ROOT::Experimental::RFieldBase::ColumnRepresentation_t &
ROOT::Experimental::RFieldBase::GetColumnRepresentative() const
{
   if (fColumnRepresentative)
      return *fColumnRepresentative;
   return GetColumnRepresentations().GetSerializationDefault();
}

void ROOT::Experimental::RFieldBase::SetColumnRepresentative(const ColumnRepresentation_t &representative)
{
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("cannot set column representative once field is connected"));
   const auto &validTypes = GetColumnRepresentations().GetSerializationTypes();
   auto itRepresentative = std::find(validTypes.begin(), validTypes.end(), representative);
   if (itRepresentative == std::end(validTypes))
      throw RException(R__FAIL("invalid column representative"));
   fColumnRepresentative = &(*itRepresentative);
}

const ROOT::Experimental::RFieldBase::ColumnRepresentation_t &
ROOT::Experimental::RFieldBase::EnsureCompatibleColumnTypes(const RNTupleDescriptor &desc) const
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
      columnTypeNames += Internal::RColumnElementBase::GetTypeName(t);
   }
   throw RException(R__FAIL("On-disk column types `" + columnTypeNames + "` for field `" + GetQualifiedFieldName() +
                            "` cannot be matched."));
}

size_t ROOT::Experimental::RFieldBase::AddReadCallback(ReadCallback_t func)
{
   fReadCallbacks.push_back(func);
   fIsSimple = false;
   return fReadCallbacks.size() - 1;
}

void ROOT::Experimental::RFieldBase::RemoveReadCallback(size_t idx)
{
   fReadCallbacks.erase(fReadCallbacks.begin() + idx);
   fIsSimple = (fTraits & kTraitMappable) && fReadCallbacks.empty();
}

void ROOT::Experimental::RFieldBase::AutoAdjustColumnTypes(const RNTupleWriteOptions &options)
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

void ROOT::Experimental::RFieldBase::ConnectPageSink(Internal::RPageSink &pageSink, NTupleSize_t firstEntry)
{
   if (dynamic_cast<ROOT::Experimental::RFieldZero *>(this))
      throw RException(R__FAIL("invalid attempt to connect zero field to page sink"));
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("invalid attempt to connect an already connected field to a page sink"));

   AutoAdjustColumnTypes(pageSink.GetWriteOptions());

   GenerateColumnsImpl();
   if (!fColumns.empty())
      fPrincipalColumn = fColumns[0].get();
   for (auto &column : fColumns) {
      auto firstElementIndex = (column.get() == fPrincipalColumn) ? EntryToColumnElementIndex(firstEntry) : 0;
      column->ConnectPageSink(fOnDiskId, pageSink, firstElementIndex);
   }

   fState = EState::kConnectedToSink;
}

void ROOT::Experimental::RFieldBase::ConnectPageSource(Internal::RPageSource &pageSource)
{
   if (dynamic_cast<ROOT::Experimental::RFieldZero *>(this))
      throw RException(R__FAIL("invalid attempt to connect zero field to page source"));
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("invalid attempt to connect an already connected field to a page source"));

   if (fColumnRepresentative)
      throw RException(R__FAIL("fixed column representative only valid when connecting to a page sink"));
   if (!fDescription.empty())
      throw RException(R__FAIL("setting description only valid when connecting to a page sink"));

   for (auto &f : fSubFields) {
      if (f->GetOnDiskId() == kInvalidDescriptorId) {
         f->SetOnDiskId(pageSource.GetSharedDescriptorGuard()->FindFieldId(f->GetFieldName(), GetOnDiskId()));
      }
      f->ConnectPageSource(pageSource);
   }

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
      column->ConnectPageSource(fOnDiskId, pageSource);
   OnConnectPageSource();

   fState = EState::kConnectedToSource;
}

void ROOT::Experimental::RFieldBase::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitField(*this);
}

//-----------------------------------------------------------------------------

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
ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<ROOT::Experimental::ClusterSize_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitClusterSizeField(*this);
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

void ROOT::Experimental::RCardinalityField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
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

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<char>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kChar}}, {{}});
   return representations;
}

void ROOT::Experimental::RField<char>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<char>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<char>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<char>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<char>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitCharField(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::byte>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kByte}}, {{}});
   return representations;
}

void ROOT::Experimental::RField<std::byte>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<char>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::byte>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<char>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::byte>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitByteField(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::int8_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kInt8}}, {{EColumnType::kUInt8}});
   return representations;
}

void ROOT::Experimental::RField<std::int8_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<std::int8_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::int8_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<std::int8_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::int8_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt8Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::uint8_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kUInt8}}, {{EColumnType::kInt8}});
   return representations;
}

void ROOT::Experimental::RField<std::uint8_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<std::uint8_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::uint8_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<std::uint8_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::uint8_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt8Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<bool>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kBit}}, {});
   return representations;
}

void ROOT::Experimental::RField<bool>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<bool>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<bool>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<bool>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<bool>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitBoolField(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<float>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitReal32}, {EColumnType::kReal32}, {EColumnType::kReal16}}, {});
   return representations;
}

void ROOT::Experimental::RField<float>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<float>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<float>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<float>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<float>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitFloatField(*this);
}

void ROOT::Experimental::RField<float>::SetHalfPrecision()
{
   SetColumnRepresentative({EColumnType::kReal16});
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<double>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitReal64}, {EColumnType::kReal64}, {EColumnType::kSplitReal32}, {EColumnType::kReal32}}, {});
   return representations;
}

void ROOT::Experimental::RField<double>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<double>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<double>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<double>(RColumnModel(onDiskTypes[0]), 0));
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

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::int16_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitInt16}, {EColumnType::kInt16}},
                                                 {{EColumnType::kSplitUInt16}, {EColumnType::kUInt16}});
   return representations;
}

void ROOT::Experimental::RField<std::int16_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<std::int16_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::int16_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<std::int16_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::int16_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitInt16Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::uint16_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitUInt16}, {EColumnType::kUInt16}},
                                                 {{EColumnType::kSplitInt16}, {EColumnType::kInt16}});
   return representations;
}

void ROOT::Experimental::RField<std::uint16_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<std::uint16_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::uint16_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<std::uint16_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::uint16_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt16Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::int32_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitInt32}, {EColumnType::kInt32}},
                                                 {{EColumnType::kSplitUInt32}, {EColumnType::kUInt32}});
   return representations;
}

void ROOT::Experimental::RField<std::int32_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<std::int32_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::int32_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<std::int32_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::int32_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitIntField(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::uint32_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitUInt32}, {EColumnType::kUInt32}},
                                                 {{EColumnType::kSplitInt32}, {EColumnType::kInt32}});
   return representations;
}

void ROOT::Experimental::RField<std::uint32_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<std::uint32_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::uint32_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<std::uint32_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::uint32_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt32Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RField<std::uint64_t>::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSplitUInt64}, {EColumnType::kUInt64}},
                                                 {{EColumnType::kSplitInt64}, {EColumnType::kInt64}});
   return representations;
}

void ROOT::Experimental::RField<std::uint64_t>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<std::uint64_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::uint64_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<std::uint64_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::uint64_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitUInt64Field(*this);
}

//------------------------------------------------------------------------------

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
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
   fColumns.emplace_back(Internal::RColumn::Create<std::int64_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::int64_t>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<std::int64_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RField<std::int64_t>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
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

void ROOT::Experimental::RField<std::string>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
   fColumns.emplace_back(Internal::RColumn::Create<char>(RColumnModel(GetColumnRepresentative()[1]), 1));
}

void ROOT::Experimental::RField<std::string>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
   fColumns.emplace_back(Internal::RColumn::Create<char>(RColumnModel(onDiskTypes[1]), 1));
}

std::size_t ROOT::Experimental::RField<std::string>::AppendImpl(const void *from)
{
   auto typedValue = static_cast<const std::string *>(from);
   auto length = typedValue->length();
   fColumns[1]->AppendV(typedValue->data(), length);
   fIndex += length;
   fColumns[0]->Append(&fIndex);
   return length + fColumns[0]->GetElement()->GetPackedSize();
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
      fColumns[1]->ReadV(collectionStart, nChars, const_cast<char *>(typedValue->data()));
   }
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
   : ROOT::Experimental::RFieldBase(fieldName, className, ENTupleStructure::kRecord, false /* isSimple */),
     fClass(classp)
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
         R__FAIL(std::string(className) + " has an associated collection proxy; use RProxiedCollectionField instead"));
   }

   if (!(fClass->ClassProperty() & kClassHasExplicitCtor))
      fTraits |= kTraitTriviallyConstructible;
   if (!(fClass->ClassProperty() & kClassHasExplicitDtor))
      fTraits |= kTraitTriviallyDestructible;

   int i = 0;
   for (auto baseClass : ROOT::Detail::TRangeStaticCast<TBaseClass>(*fClass->GetListOfBases())) {
      TClass *c = baseClass->GetClassPointer();
      auto subField =
         RFieldBase::Create(std::string(kPrefixInherited) + "_" + std::to_string(i), c->GetName()).Unwrap();
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

      std::string typeName{GetNormalizedTypeName(dataMember->GetTrueTypeName())};
      std::string typeAlias{GetNormalizedTypeName(dataMember->GetFullTypeName())};
      // For C-style arrays, complete the type name with the size for each dimension, e.g. `int[4][2]`
      if (dataMember->Property() & kIsArray) {
         for (int dim = 0, n = dataMember->GetArrayDim(); dim < n; ++dim)
            typeName += "[" + std::to_string(dataMember->GetMaxIndex(dim)) + "]";
      }
      auto subField = RFieldBase::Create(dataMember->GetName(), typeName, typeAlias).Unwrap();
      fTraits &= subField->GetTraits();
      Attach(std::move(subField),
	     RSubFieldInfo{kDataMember, static_cast<std::size_t>(dataMember->GetOffset())});
   }
}

void ROOT::Experimental::RClassField::Attach(std::unique_ptr<RFieldBase> child, RSubFieldInfo info)
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
      fReadCallbacks.emplace_back([func, classp](void *target) {
         TVirtualObject oldObj{nullptr};
         oldObj.fClass = classp;
         oldObj.fObject = target;
         func(static_cast<char *>(target), &oldObj);
         oldObj.fClass = nullptr; // TVirtualObject does not own the value
      });
   }
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RClassField::CloneImpl(std::string_view newName) const
{
   auto result = std::unique_ptr<RClassField>(new RClassField(newName, GetTypeName(), fClass));
   SyncFieldIDs(*this, *result);
   return result;
}

std::size_t ROOT::Experimental::RClassField::AppendImpl(const void *from)
{
   std::size_t nbytes = 0;
   for (unsigned i = 0; i < fSubFields.size(); i++) {
      nbytes += CallAppendOn(*fSubFields[i], static_cast<const unsigned char *>(from) + fSubFieldsInfo[i].fOffset);
   }
   return nbytes;
}

void ROOT::Experimental::RClassField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   for (unsigned i = 0; i < fSubFields.size(); i++) {
      CallReadOn(*fSubFields[i], globalIndex, static_cast<unsigned char *>(to) + fSubFieldsInfo[i].fOffset);
   }
}

void ROOT::Experimental::RClassField::ReadInClusterImpl(RClusterIndex clusterIndex, void *to)
{
   for (unsigned i = 0; i < fSubFields.size(); i++) {
      CallReadOn(*fSubFields[i], clusterIndex, static_cast<unsigned char *>(to) + fSubFieldsInfo[i].fOffset);
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

void ROOT::Experimental::RClassField::ConstructValue(void *where) const
{
   fClass->New(where);
}

void ROOT::Experimental::RClassField::RClassDeleter::operator()(void *objPtr, bool dtorOnly)
{
   fClass->Destructor(objPtr, true /* dtorOnly */);
   RDeleter::operator()(objPtr, dtorOnly);
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RClassField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   auto basePtr = value.GetPtr<unsigned char>().get();
   for (unsigned i = 0; i < fSubFields.size(); i++) {
      result.emplace_back(
         fSubFields[i]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), basePtr + fSubFieldsInfo[i].fOffset)));
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

ROOT::Experimental::REnumField::REnumField(std::string_view fieldName, std::string_view enumName)
   : REnumField(fieldName, enumName, TEnum::GetEnum(std::string(enumName).c_str()))
{
}

ROOT::Experimental::REnumField::REnumField(std::string_view fieldName, std::string_view enumName, TEnum *enump)
   : ROOT::Experimental::RFieldBase(fieldName, enumName, ENTupleStructure::kLeaf, false /* isSimple */)
{
   if (enump == nullptr) {
      throw RException(R__FAIL("RField: no I/O support for enum type " + std::string(enumName)));
   }
   // Avoid accidentally supporting std types through TEnum.
   if (enump->Property() & kIsDefinedInStd) {
      throw RException(R__FAIL(std::string(enumName) + " is not supported"));
   }

   switch (enump->GetUnderlyingType()) {
   case kChar_t: Attach(std::make_unique<RField<int8_t>>("_0")); break;
   case kUChar_t: Attach(std::make_unique<RField<uint8_t>>("_0")); break;
   case kShort_t: Attach(std::make_unique<RField<int16_t>>("_0")); break;
   case kUShort_t: Attach(std::make_unique<RField<uint16_t>>("_0")); break;
   case kInt_t: Attach(std::make_unique<RField<int32_t>>("_0")); break;
   case kUInt_t: Attach(std::make_unique<RField<uint32_t>>("_0")); break;
   case kLong_t:
   case kLong64_t: Attach(std::make_unique<RField<int64_t>>("_0")); break;
   case kULong_t:
   case kULong64_t: Attach(std::make_unique<RField<uint64_t>>("_0")); break;
   default: throw RException(R__FAIL("Unsupported underlying integral type for enum type " + std::string(enumName)));
   }

   fTraits |= kTraitTriviallyConstructible | kTraitTriviallyDestructible;
}

ROOT::Experimental::REnumField::REnumField(std::string_view fieldName, std::string_view enumName,
                                           std::unique_ptr<RFieldBase> intField)
   : ROOT::Experimental::RFieldBase(fieldName, enumName, ENTupleStructure::kLeaf, false /* isSimple */)
{
   Attach(std::move(intField));
   fTraits |= kTraitTriviallyConstructible | kTraitTriviallyDestructible;
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::REnumField::CloneImpl(std::string_view newName) const
{
   auto newIntField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::unique_ptr<REnumField>(new REnumField(newName, GetTypeName(), std::move(newIntField)));
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::REnumField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   result.emplace_back(fSubFields[0]->BindValue(value.GetPtr<void>()));
   return result;
}

void ROOT::Experimental::REnumField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitEnumField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RProxiedCollectionField::RCollectionIterableOnce::RIteratorFuncs
ROOT::Experimental::RProxiedCollectionField::RCollectionIterableOnce::GetIteratorFuncs(TVirtualCollectionProxy *proxy,
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

ROOT::Experimental::RProxiedCollectionField::RProxiedCollectionField(std::string_view fieldName,
                                                                     std::string_view typeName, TClass *classp)
   : RFieldBase(fieldName, typeName, ENTupleStructure::kCollection, false /* isSimple */), fNWritten(0)
{
   if (classp == nullptr)
      throw RException(R__FAIL("RField: no I/O support for collection proxy type " + std::string(typeName)));
   if (!classp->GetCollectionProxy())
      throw RException(R__FAIL(std::string(typeName) + " has no associated collection proxy"));

   fProxy.reset(classp->GetCollectionProxy()->Generate());
   fProperties = fProxy->GetProperties();
   fCollectionType = fProxy->GetCollectionType();
   if (fProxy->HasPointers())
      throw RException(R__FAIL("collection proxies whose value type is a pointer are not supported"));
   if (!fProxy->GetCollectionClass()->HasDictionary()) {
      throw RException(R__FAIL("dictionary not available for type " +
                               GetNormalizedTypeName(fProxy->GetCollectionClass()->GetName())));
   }

   fIFuncsRead = RCollectionIterableOnce::GetIteratorFuncs(fProxy.get(), true /* readFromDisk */);
   fIFuncsWrite = RCollectionIterableOnce::GetIteratorFuncs(fProxy.get(), false /* readFromDisk */);
}

ROOT::Experimental::RProxiedCollectionField::RProxiedCollectionField(std::string_view fieldName,
                                                                     std::string_view typeName,
                                                                     std::unique_ptr<RFieldBase> itemField)
   : RProxiedCollectionField(fieldName, typeName, TClass::GetClass(std::string(typeName).c_str()))
{
   fItemSize = itemField->GetValueSize();
   Attach(std::move(itemField));
}

ROOT::Experimental::RProxiedCollectionField::RProxiedCollectionField(std::string_view fieldName,
                                                                     std::string_view typeName)
   : RProxiedCollectionField(fieldName, typeName, TClass::GetClass(std::string(typeName).c_str()))
{
   // NOTE (fdegeus): std::map is supported, custom associative might be supported in the future if the need arises.
   if (fProperties & TVirtualCollectionProxy::kIsAssociative)
      throw RException(R__FAIL("custom associative collection proxies not supported"));

   std::unique_ptr<ROOT::Experimental::RFieldBase> itemField;

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

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RProxiedCollectionField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::unique_ptr<RProxiedCollectionField>(
      new RProxiedCollectionField(newName, GetTypeName(), std::move(newItemField)));
}

std::size_t ROOT::Experimental::RProxiedCollectionField::AppendImpl(const void *from)
{
   std::size_t nbytes = 0;
   unsigned count = 0;
   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), const_cast<void *>(from));
   for (auto ptr : RCollectionIterableOnce{const_cast<void *>(from), fIFuncsWrite, fProxy.get(),
                                           (fCollectionType == kSTLvector ? fItemSize : 0U)}) {
      nbytes += CallAppendOn(*fSubFields[0], ptr);
      count++;
   }

   fNWritten += count;
   fColumns[0]->Append(&fNWritten);
   return nbytes + fColumns[0]->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RProxiedCollectionField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   ClusterSize_t nItems;
   RClusterIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), to);
   void *obj =
      fProxy->Allocate(static_cast<std::uint32_t>(nItems), (fProperties & TVirtualCollectionProxy::kNeedDelete));

   unsigned i = 0;
   for (auto elementPtr : RCollectionIterableOnce{obj, fIFuncsRead, fProxy.get(),
                                                  (fCollectionType == kSTLvector || obj != to ? fItemSize : 0U)}) {
      CallReadOn(*fSubFields[0], collectionStart + (i++), elementPtr);
   }
   if (obj != to)
      fProxy->Commit(obj);
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RProxiedCollectionField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RProxiedCollectionField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RProxiedCollectionField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RProxiedCollectionField::ConstructValue(void *where) const
{
   fProxy->New(where);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter>
ROOT::Experimental::RProxiedCollectionField::GetDeleter() const
{
   if (fProperties & TVirtualCollectionProxy::kNeedDelete) {
      std::size_t itemSize = fCollectionType == kSTLvector ? fItemSize : 0U;
      return std::make_unique<RProxiedCollectionDeleter>(fProxy, GetDeleterOf(*fSubFields[0]), itemSize);
   }
   return std::make_unique<RProxiedCollectionDeleter>(fProxy);
}

void ROOT::Experimental::RProxiedCollectionField::RProxiedCollectionDeleter::operator()(void *objPtr, bool dtorOnly)
{
   if (fItemDeleter) {
      TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), objPtr);
      for (auto ptr : RCollectionIterableOnce{objPtr, fIFuncsWrite, fProxy.get(), fItemSize}) {
         fItemDeleter->operator()(ptr, true /* dtorOnly */);
      }
   }
   fProxy->Destructor(objPtr, true /* dtorOnly */);
   RDeleter::operator()(objPtr, dtorOnly);
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RProxiedCollectionField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   auto valueRawPtr = value.GetPtr<void>().get();
   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), valueRawPtr);
   for (auto ptr : RCollectionIterableOnce{valueRawPtr, fIFuncsWrite, fProxy.get(),
                                           (fCollectionType == kSTLvector ? fItemSize : 0U)}) {
      result.emplace_back(fSubFields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), ptr)));
   }
   return result;
}

void ROOT::Experimental::RProxiedCollectionField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitProxiedCollectionField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RRecordField::RRecordField(std::string_view fieldName,
                                               std::vector<std::unique_ptr<RFieldBase>> &&itemFields,
                                               const std::vector<std::size_t> &offsets, std::string_view typeName)
   : ROOT::Experimental::RFieldBase(fieldName, typeName, ENTupleStructure::kRecord, false /* isSimple */),
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
                                               std::vector<std::unique_ptr<RFieldBase>> &&itemFields)
   : ROOT::Experimental::RFieldBase(fieldName, "", ENTupleStructure::kRecord, false /* isSimple */)
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
                                               std::vector<std::unique_ptr<RFieldBase>> &itemFields)
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

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RRecordField::CloneImpl(std::string_view newName) const
{
   std::vector<std::unique_ptr<RFieldBase>> cloneItems;
   cloneItems.reserve(fSubFields.size());
   for (auto &item : fSubFields)
      cloneItems.emplace_back(item->Clone(item->GetFieldName()));
   return std::unique_ptr<RRecordField>(new RRecordField(newName, std::move(cloneItems), fOffsets, GetTypeName()));
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
   return std::make_unique<RRecordDeleter>(itemDeleters, fOffsets);
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RRecordField::SplitValue(const RValue &value) const
{
   auto basePtr = value.GetPtr<unsigned char>().get();
   std::vector<RValue> result;
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

ROOT::Experimental::RVectorField::RVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
   : ROOT::Experimental::RFieldBase(fieldName, "std::vector<" + itemField->GetTypeName() + ">",
                                    ENTupleStructure::kCollection, false /* isSimple */),
     fItemSize(itemField->GetValueSize()),
     fNWritten(0)
{
   if (!(itemField->GetTraits() & kTraitTriviallyDestructible))
      fItemDeleter = GetDeleterOf(*itemField);
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RVectorField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::make_unique<RVectorField>(newName, std::move(newItemField));
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
   fColumns[0]->Append(&fNWritten);
   return nbytes + fColumns[0]->GetElement()->GetPackedSize();
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

   // See "semantics of reading non-trivial objects" in RNTuple's architecture.md
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

void ROOT::Experimental::RVectorField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RVectorField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
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
   fColumns[0]->Append(&fNWritten);
   return nbytes + fColumns[0]->GetElement()->GetPackedSize();
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

   // See "semantics of reading non-trivial objects" in RNTuple's architecture.md for details
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

void ROOT::Experimental::RRVecField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RRVecField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
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
   fColumns[0]->Append(&fNWritten);
   return count + fColumns[0]->GetElement()->GetPackedSize();
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

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RField<std::vector<bool>>::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RField<std::vector<bool>>::SplitValue(const RValue &value) const
{
   const auto &typedValue = value.GetRef<std::vector<bool>>();
   auto count = typedValue.size();
   std::vector<RValue> result;
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
   // See "semantics of reading non-trivial objects" in RNTuple's architecture.md for details
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
      GetPrincipalColumnOf(*fSubFields[0])->ReadV(globalIndex*fArrayLength, fArrayLength, rvecBeginPtr);
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

void ROOT::Experimental::RBitsetField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<bool>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RBitsetField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<bool>(RColumnModel(onDiskTypes[0]), 0));
}

std::size_t ROOT::Experimental::RBitsetField::AppendImpl(const void *from)
{
   const auto *asULongArray = static_cast<const Word_t *>(from);
   bool elementValue;
   std::size_t i = 0;
   for (std::size_t word = 0; word < (fN + kBitsPerWord - 1) / kBitsPerWord; ++word) {
      for (std::size_t mask = 0; (mask < kBitsPerWord) && (i < fN); ++mask, ++i) {
         elementValue = (asULongArray[word] & (static_cast<Word_t>(1) << mask)) != 0;
         fColumns[0]->Append(&elementValue);
      }
   }
   return fN;
}

void ROOT::Experimental::RBitsetField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   auto *asULongArray = static_cast<Word_t *>(to);
   bool elementValue;
   for (std::size_t i = 0; i < fN; ++i) {
      fColumns[0]->Read(globalIndex * fN + i, &elementValue);
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

std::string ROOT::Experimental::RVariantField::GetTypeList(const std::vector<RFieldBase *> &itemFields)
{
   std::string result;
   for (size_t i = 0; i < itemFields.size(); ++i) {
      result += itemFields[i]->GetTypeName() + ",";
   }
   R__ASSERT(!result.empty()); // there is always at least one variant
   result.pop_back(); // remove trailing comma
   return result;
}

ROOT::Experimental::RVariantField::RVariantField(std::string_view fieldName,
                                                 const std::vector<RFieldBase *> &itemFields)
   : ROOT::Experimental::RFieldBase(fieldName, "std::variant<" + GetTypeList(itemFields) + ">",
                                    ENTupleStructure::kVariant, false /* isSimple */)
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
      Attach(std::unique_ptr<RFieldBase>(itemFields[i]));
   }
   fTagOffset = (fMaxItemSize < fMaxAlignment) ? fMaxAlignment : fMaxItemSize;
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RVariantField::CloneImpl(std::string_view newName) const
{
   auto nFields = fSubFields.size();
   std::vector<RFieldBase *> itemFields;
   for (unsigned i = 0; i < nFields; ++i) {
      // TODO(jblomer): use unique_ptr in RVariantField constructor
      itemFields.emplace_back(fSubFields[i]->Clone(fSubFields[i]->GetFieldName()).release());
   }
   return std::make_unique<RVariantField>(newName, itemFields);
}

std::uint32_t ROOT::Experimental::RVariantField::GetTag(const void *variantPtr, std::size_t tagOffset)
{
   auto index = *(reinterpret_cast<const char *>(variantPtr) + tagOffset);
   return (index < 0) ? 0 : index + 1;
}

void ROOT::Experimental::RVariantField::SetTag(void *variantPtr, std::size_t tagOffset, std::uint32_t tag)
{
   auto index = reinterpret_cast<char *>(variantPtr) + tagOffset;
   *index = static_cast<char>(tag - 1);
}

std::size_t ROOT::Experimental::RVariantField::AppendImpl(const void *from)
{
   auto tag = GetTag(from, fTagOffset);
   std::size_t nbytes = 0;
   auto index = 0;
   if (tag > 0) {
      nbytes += CallAppendOn(*fSubFields[tag - 1], from);
      index = fNWritten[tag - 1]++;
   }
   RColumnSwitch varSwitch(ClusterSize_t(index), tag);
   fColumns[0]->Append(&varSwitch);
   return nbytes + sizeof(RColumnSwitch);
}

void ROOT::Experimental::RVariantField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   RClusterIndex variantIndex;
   std::uint32_t tag;
   fPrincipalColumn->GetSwitchInfo(globalIndex, &variantIndex, &tag);

   // If `tag` equals 0, the variant is in the invalid state, i.e, it does not hold any of the valid alternatives in
   // the type list.  This happens, e.g., if the field was late added; in this case, keep the invalid tag, which makes
   // any `std::holds_alternative<T>` check fail later.
   if (R__likely(tag > 0)) {
      CallConstructValueOn(*fSubFields[tag - 1], to);
      CallReadOn(*fSubFields[tag - 1], variantIndex, to);
   }
   SetTag(to, fTagOffset, tag);
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RVariantField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{EColumnType::kSwitch}}, {{}});
   return representations;
}

void ROOT::Experimental::RVariantField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<RColumnSwitch>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RVariantField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<RColumnSwitch>(RColumnModel(onDiskTypes[0]), 0));
}

void ROOT::Experimental::RVariantField::ConstructValue(void *where) const
{
   memset(where, 0, GetValueSize());
   CallConstructValueOn(*fSubFields[0], where);
   SetTag(where, fTagOffset, 1);
}

void ROOT::Experimental::RVariantField::RVariantDeleter::operator()(void *objPtr, bool dtorOnly)
{
   auto tag = GetTag(objPtr, fTagOffset);
   if (tag > 0) {
      fItemDeleters[tag - 1]->operator()(objPtr, true /* dtorOnly */);
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::RVariantField::GetDeleter() const
{
   std::vector<std::unique_ptr<RDeleter>> itemDeleters;
   itemDeleters.reserve(fSubFields.size());
   for (const auto &f : fSubFields) {
      itemDeleters.emplace_back(GetDeleterOf(*f));
   }
   return std::make_unique<RVariantDeleter>(fTagOffset, itemDeleters);
}

size_t ROOT::Experimental::RVariantField::GetValueSize() const
{
   return fMaxItemSize + fMaxAlignment;  // TODO: fix for more than 255 items
}

void ROOT::Experimental::RVariantField::CommitClusterImpl()
{
   std::fill(fNWritten.begin(), fNWritten.end(), 0);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RSetField::RSetField(std::string_view fieldName, std::string_view typeName,
                                         std::unique_ptr<RFieldBase> itemField)
   : ROOT::Experimental::RProxiedCollectionField(fieldName, typeName, std::move(itemField))
{
}

std::unique_ptr<ROOT::Experimental::RFieldBase> ROOT::Experimental::RSetField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::make_unique<RSetField>(newName, GetTypeName(), std::move(newItemField));
}

//------------------------------------------------------------------------------

ROOT::Experimental::RMapField::RMapField(std::string_view fieldName, std::string_view typeName,
                                         std::unique_ptr<RFieldBase> itemField)
   : RProxiedCollectionField(fieldName, typeName, TClass::GetClass(std::string(typeName).c_str()))
{
   if (!dynamic_cast<RPairField *>(itemField.get()))
      throw RException(R__FAIL("RMapField inner field type must be of RPairField"));

   fItemClass = fProxy->GetValueClass();
   fItemSize = fItemClass->GetClassSize();

   Attach(std::move(itemField));
}

std::size_t ROOT::Experimental::RMapField::AppendImpl(const void *from)
{
   std::size_t nbytes = 0;
   unsigned count = 0;
   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), const_cast<void *>(from));
   for (auto ptr : RCollectionIterableOnce{const_cast<void *>(from), fIFuncsWrite, fProxy.get(), 0U}) {
      nbytes += CallAppendOn(*fSubFields[0], ptr);
      count++;
   }
   fNWritten += count;
   fColumns[0]->Append(&fNWritten);
   return nbytes + fColumns[0]->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RMapField::ReadGlobalImpl(NTupleSize_t globalIndex, void *to)
{
   ClusterSize_t nItems;
   RClusterIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), to);
   void *obj =
      fProxy->Allocate(static_cast<std::uint32_t>(nItems), (fProperties & TVirtualCollectionProxy::kNeedDelete));

   unsigned i = 0;
   for (auto ptr : RCollectionIterableOnce{obj, fIFuncsRead, fProxy.get(), fItemSize}) {
      CallReadOn(*fSubFields[0], collectionStart + i, ptr);
      i++;
   }

   if (obj != to)
      fProxy->Commit(obj);
}

std::vector<ROOT::Experimental::RFieldBase::RValue> ROOT::Experimental::RMapField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   auto valueRawPtr = value.GetPtr<void>().get();
   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), valueRawPtr);
   for (auto ptr : RCollectionIterableOnce{valueRawPtr, fIFuncsWrite, fProxy.get(), 0U}) {
      result.emplace_back(fSubFields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), ptr)));
   }
   return result;
}

std::unique_ptr<ROOT::Experimental::RFieldBase> ROOT::Experimental::RMapField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
   return std::unique_ptr<RMapField>(new RMapField(newName, GetTypeName(), std::move(newItemField)));
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
      fDefaultItemValue = std::make_unique<RValue>(fSubFields[0]->CreateValue());
      fColumns.emplace_back(Internal::RColumn::Create<bool>(RColumnModel(EColumnType::kBit), 0));
   } else {
      fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
   }
}

void ROOT::Experimental::RNullableField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   if (onDiskTypes[0] == EColumnType::kBit) {
      fColumns.emplace_back(Internal::RColumn::Create<bool>(RColumnModel(EColumnType::kBit), 0));
   } else {
      fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
   }
}

std::size_t ROOT::Experimental::RNullableField::AppendNull()
{
   if (IsDense()) {
      bool mask = false;
      fPrincipalColumn->Append(&mask);
      return 1 + CallAppendOn(*fSubFields[0], fDefaultItemValue->GetPtr<void>().get());
   } else {
      fPrincipalColumn->Append(&fNWritten);
      return sizeof(ClusterSize_t);
   }
}

std::size_t ROOT::Experimental::RNullableField::AppendValue(const void *from)
{
   auto nbytesItem = CallAppendOn(*fSubFields[0], from);
   if (IsDense()) {
      bool mask = true;
      fPrincipalColumn->Append(&mask);
      return 1 + nbytesItem;
   } else {
      fNWritten++;
      fPrincipalColumn->Append(&fNWritten);
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
      valuePtr = malloc(fSubFields[0]->GetValueSize());
      CallConstructValueOn(*fSubFields[0], valuePtr);
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

std::string
ROOT::Experimental::RPairField::RPairField::GetTypeList(const std::array<std::unique_ptr<RFieldBase>, 2> &itemFields)
{
   return itemFields[0]->GetTypeName() + "," + itemFields[1]->GetTypeName();
}

ROOT::Experimental::RPairField::RPairField(std::string_view fieldName,
                                           std::array<std::unique_ptr<RFieldBase>, 2> &&itemFields,
                                           const std::array<std::size_t, 2> &offsets)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields), offsets,
                                      "std::pair<" + GetTypeList(itemFields) + ">")
{
}

ROOT::Experimental::RPairField::RPairField(std::string_view fieldName,
                                           std::array<std::unique_ptr<RFieldBase>, 2> &itemFields)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields), {},
                                      "std::pair<" + GetTypeList(itemFields) + ">")
{
   // ISO C++ does not guarantee any specific layout for `std::pair`; query TClass for the member offsets
   fClass = TClass::GetClass(GetTypeName().c_str());
   if (!fClass)
      throw RException(R__FAIL("cannot get type information for " + GetTypeName()));
   fSize = fClass->Size();

   auto firstElem = fClass->GetRealData("first");
   if (!firstElem)
      throw RException(R__FAIL("first: no such member"));
   fOffsets[0] = firstElem->GetThisOffset();

   auto secondElem = fClass->GetRealData("second");
   if (!secondElem)
      throw RException(R__FAIL("second: no such member"));
   fOffsets[1] = secondElem->GetThisOffset();
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RPairField::CloneImpl(std::string_view newName) const
{
   std::array<std::unique_ptr<RFieldBase>, 2> items{fSubFields[0]->Clone(fSubFields[0]->GetFieldName()),
                                                    fSubFields[1]->Clone(fSubFields[1]->GetFieldName())};

   std::unique_ptr<RPairField> result(new RPairField(newName, std::move(items), {fOffsets[0], fOffsets[1]}));
   result->fClass = fClass;
   return result;
}

void ROOT::Experimental::RPairField::ConstructValue(void *where) const
{
   fClass->New(where);
}

void ROOT::Experimental::RPairField::RPairDeleter::operator()(void *objPtr, bool dtorOnly)
{
   fClass->Destructor(objPtr, true /* dtorOnly */);
   RDeleter::operator()(objPtr, dtorOnly);
}

//------------------------------------------------------------------------------

std::string
ROOT::Experimental::RTupleField::RTupleField::GetTypeList(const std::vector<std::unique_ptr<RFieldBase>> &itemFields)
{
   std::string result;
   if (itemFields.empty())
      throw RException(R__FAIL("the type list for std::tuple must have at least one element"));
   for (size_t i = 0; i < itemFields.size(); ++i) {
      result += itemFields[i]->GetTypeName() + ",";
   }
   result.pop_back();          // remove trailing comma
   return result;
}

ROOT::Experimental::RTupleField::RTupleField(std::string_view fieldName,
                                             std::vector<std::unique_ptr<RFieldBase>> &&itemFields,
                                             const std::vector<std::size_t> &offsets)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields), offsets,
                                      "std::tuple<" + GetTypeList(itemFields) + ">")
{
}

ROOT::Experimental::RTupleField::RTupleField(std::string_view fieldName,
                                             std::vector<std::unique_ptr<RFieldBase>> &itemFields)
   : ROOT::Experimental::RRecordField(fieldName, std::move(itemFields), {},
                                      "std::tuple<" + GetTypeList(itemFields) + ">")
{
   fClass = TClass::GetClass(GetTypeName().c_str());
   if (!fClass)
      throw RException(R__FAIL("cannot get type information for " + GetTypeName()));
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

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RTupleField::CloneImpl(std::string_view newName) const
{
   std::vector<std::unique_ptr<RFieldBase>> items;
   items.reserve(fSubFields.size());
   for (const auto &item : fSubFields)
      items.push_back(item->Clone(item->GetFieldName()));

   std::unique_ptr<RTupleField> result(new RTupleField(newName, std::move(items), fOffsets));
   result->fClass = fClass;
   return result;
}

void ROOT::Experimental::RTupleField::ConstructValue(void *where) const
{
   fClass->New(where);
}

void ROOT::Experimental::RTupleField::RTupleDeleter::operator()(void *objPtr, bool dtorOnly)
{
   fClass->Destructor(objPtr, true /* dtorOnly */);
   RDeleter::operator()(objPtr, dtorOnly);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RCollectionField::RCollectionField(std::string_view name,
                                                       std::shared_ptr<RNTupleCollectionWriter> collectionWriter,
                                                       std::unique_ptr<RFieldZero> collectionParent)
   : RFieldBase(name, "", ENTupleStructure::kCollection, false /* isSimple */), fCollectionWriter(collectionWriter)
{
   const std::size_t N = collectionParent->fSubFields.size();
   for (std::size_t i = 0; i < N; ++i) {
      Attach(std::move(collectionParent->fSubFields[i]));
   }
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RCollectionField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations(
      {{EColumnType::kSplitIndex64}, {EColumnType::kIndex64}, {EColumnType::kSplitIndex32}, {EColumnType::kIndex32}},
      {});
   return representations;
}

void ROOT::Experimental::RCollectionField::GenerateColumnsImpl()
{
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(GetColumnRepresentative()[0]), 0));
}

void ROOT::Experimental::RCollectionField::GenerateColumnsImpl(const RNTupleDescriptor &desc)
{
   auto onDiskTypes = EnsureCompatibleColumnTypes(desc);
   fColumns.emplace_back(Internal::RColumn::Create<ClusterSize_t>(RColumnModel(onDiskTypes[0]), 0));
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RCollectionField::CloneImpl(std::string_view newName) const
{
   auto parent = std::make_unique<RFieldZero>();
   for (auto& f : fSubFields) {
      parent->Attach(f->Clone(f->GetFieldName()));
   }
   return std::make_unique<RCollectionField>(newName, fCollectionWriter, std::move(parent));
}

std::size_t ROOT::Experimental::RCollectionField::AppendImpl(const void *from)
{
   // RCollectionFields are almost simple, but they return the bytes written by their subfields as accumulated by the
   // RNTupleCollectionWriter.
   std::size_t bytesWritten = fCollectionWriter->fBytesWritten;
   fCollectionWriter->fBytesWritten = 0;

   fColumns[0]->Append(from);
   return bytesWritten + fColumns[0]->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RCollectionField::ReadGlobalImpl(NTupleSize_t, void *)
{
   R__ASSERT(false && "should never read an RCollectionField");
}

void ROOT::Experimental::RCollectionField::CommitClusterImpl()
{
   *fCollectionWriter->GetOffsetPtr() = 0;
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
