/// \file RFieldBase.cxx
/// \ingroup NTuple
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RFieldUtils.hxx>

#include <TClass.h>
#include <TClassEdit.h>
#include <TEnum.h>

#include <string>
#include <vector>

namespace {

/// Used as a thread local context storage for Create(); steers the behavior of the Create() call stack
class CreateContextGuard;
class CreateContext {
   friend class CreateContextGuard;
   /// All classes that were defined by Create() calls higher up in the stack. Finds cyclic type definitions.
   std::vector<std::string> fClassesOnStack;
   /// If set to true, Create() will create an RInvalidField on error instead of throwing an exception.
   /// This is used in RFieldBase::Check() to identify unsupported sub fields.
   bool fContinueOnError = false;

public:
   CreateContext() = default;
   bool GetContinueOnError() const { return fContinueOnError; }
};

/// RAII for modifications of CreateContext
class CreateContextGuard {
   CreateContext &fCreateContext;
   std::size_t fNOriginalClassesOnStack;
   bool fOriginalContinueOnError;

public:
   CreateContextGuard(CreateContext &ctx)
      : fCreateContext(ctx),
        fNOriginalClassesOnStack(ctx.fClassesOnStack.size()),
        fOriginalContinueOnError(ctx.fContinueOnError)
   {
   }
   ~CreateContextGuard()
   {
      fCreateContext.fClassesOnStack.resize(fNOriginalClassesOnStack);
      fCreateContext.fContinueOnError = fOriginalContinueOnError;
   }

   void AddClassToStack(const std::string &cl)
   {
      if (std::find(fCreateContext.fClassesOnStack.begin(), fCreateContext.fClassesOnStack.end(), cl) !=
          fCreateContext.fClassesOnStack.end()) {
         throw ROOT::RException(R__FAIL("cyclic class definition: " + cl));
      }
      fCreateContext.fClassesOnStack.emplace_back(cl);
   }

   void SetContinueOnError(bool value) { fCreateContext.fContinueOnError = value; }
};

} // anonymous namespace

void ROOT::Internal::CallFlushColumnsOnField(RFieldBase &field)
{
   field.FlushColumns();
}
void ROOT::Internal::CallCommitClusterOnField(RFieldBase &field)
{
   field.CommitCluster();
}
void ROOT::Internal::CallConnectPageSinkOnField(RFieldBase &field, ROOT::Internal::RPageSink &sink,
                                                ROOT::NTupleSize_t firstEntry)
{
   field.ConnectPageSink(sink, firstEntry);
}
void ROOT::Internal::CallConnectPageSourceOnField(RFieldBase &field, ROOT::Internal::RPageSource &source)
{
   field.ConnectPageSource(source);
}

ROOT::RResult<std::unique_ptr<ROOT::RFieldBase>>
ROOT::Internal::CallFieldBaseCreate(const std::string &fieldName, const std::string &typeName,
                                    const ROOT::RCreateFieldOptions &options, const ROOT::RNTupleDescriptor *desc,
                                    ROOT::DescriptorId_t fieldId)
{
   return RFieldBase::Create(fieldName, typeName, options, desc, fieldId);
}

//------------------------------------------------------------------------------

ROOT::RFieldBase::RColumnRepresentations::RColumnRepresentations()
{
   // A single representations with an empty set of columns
   fSerializationTypes.emplace_back(ColumnRepresentation_t());
   fDeserializationTypes.emplace_back(ColumnRepresentation_t());
}

ROOT::RFieldBase::RColumnRepresentations::RColumnRepresentations(const Selection_t &serializationTypes,
                                                                 const Selection_t &deserializationExtraTypes)
   : fSerializationTypes(serializationTypes), fDeserializationTypes(serializationTypes)
{
   fDeserializationTypes.insert(fDeserializationTypes.end(), deserializationExtraTypes.begin(),
                                deserializationExtraTypes.end());
}

//------------------------------------------------------------------------------

void ROOT::RFieldBase::RValue::BindRawPtr(void *rawPtr)
{
   // Set fObjPtr to an aliased shared_ptr of the input raw pointer. Note that
   // fObjPtr will be non-empty but have use count zero.
   fObjPtr = ROOT::Internal::MakeAliasedSharedPtr(rawPtr);
}

//------------------------------------------------------------------------------

ROOT::RFieldBase::RBulkValues::RBulkValues(RBulkValues &&other)
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

ROOT::RFieldBase::RBulkValues &ROOT::RFieldBase::RBulkValues::operator=(RBulkValues &&other)
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

ROOT::RFieldBase::RBulkValues::~RBulkValues()
{
   if (fValues)
      ReleaseValues();
}

void ROOT::RFieldBase::RBulkValues::ReleaseValues()
{
   if (fIsAdopted)
      return;

   if (!(fField->GetTraits() & RFieldBase::kTraitTriviallyDestructible)) {
      for (std::size_t i = 0; i < fCapacity; ++i) {
         fDeleter->operator()(GetValuePtrAt(i), true /* dtorOnly */);
      }
   }

   operator delete(fValues);
}

void ROOT::RFieldBase::RBulkValues::Reset(RNTupleLocalIndex firstIndex, std::size_t size)
{
   if (fCapacity < size) {
      if (fIsAdopted) {
         throw RException(R__FAIL("invalid attempt to bulk read beyond the adopted buffer"));
      }
      ReleaseValues();
      fValues = operator new(size * fValueSize);

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

void ROOT::RFieldBase::RBulkValues::CountValidValues()
{
   fNValidValues = 0;
   for (std::size_t i = 0; i < fSize; ++i)
      fNValidValues += static_cast<std::size_t>(fMaskAvail[i]);
}

void ROOT::RFieldBase::RBulkValues::AdoptBuffer(void *buf, std::size_t capacity)
{
   ReleaseValues();
   fValues = buf;
   fCapacity = capacity;
   fSize = capacity;

   fMaskAvail = std::make_unique<bool[]>(capacity);

   fFirstIndex = RNTupleLocalIndex();

   fIsAdopted = true;
}

//------------------------------------------------------------------------------

void ROOT::RFieldBase::RCreateObjectDeleter<void>::operator()(void *)
{
   R__LOG_WARNING(ROOT::Internal::NTupleLog()) << "possibly leaking object from RField<T>::CreateObject<void>";
}

template <>
std::unique_ptr<void, typename ROOT::RFieldBase::RCreateObjectDeleter<void>::deleter>
ROOT::RFieldBase::CreateObject<void>() const
{
   static RCreateObjectDeleter<void>::deleter gDeleter;
   return std::unique_ptr<void, RCreateObjectDeleter<void>::deleter>(CreateObjectRawPtr(), gDeleter);
}

//------------------------------------------------------------------------------

ROOT::RFieldBase::RFieldBase(std::string_view name, std::string_view type, ROOT::ENTupleStructure structure,
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
   ROOT::Internal::EnsureValidNameForRNTuple(name, "Field");
}

std::string ROOT::RFieldBase::GetQualifiedFieldName() const
{
   std::string result = GetFieldName();
   auto parent = GetParent();
   while (parent && !parent->GetFieldName().empty()) {
      result = parent->GetFieldName() + "." + result;
      parent = parent->GetParent();
   }
   return result;
}

ROOT::RResult<std::unique_ptr<ROOT::RFieldBase>>
ROOT::RFieldBase::Create(const std::string &fieldName, const std::string &typeName)
{
   return R__FORWARD_RESULT(
      RFieldBase::Create(fieldName, typeName, ROOT::RCreateFieldOptions{}, nullptr, ROOT::kInvalidDescriptorId));
}

std::vector<ROOT::RFieldBase::RCheckResult>
ROOT::RFieldBase::Check(const std::string &fieldName, const std::string &typeName)
{
   RFieldZero fieldZero;
   ROOT::RCreateFieldOptions cfOpts{};
   cfOpts.SetReturnInvalidOnError(true);
   cfOpts.SetEmulateUnknownTypes(false);
   fieldZero.Attach(RFieldBase::Create(fieldName, typeName, cfOpts, nullptr, kInvalidDescriptorId).Unwrap());

   std::vector<RCheckResult> result;
   for (const auto &f : fieldZero) {
      const bool isInvalidField = f.GetTraits() & RFieldBase::kTraitInvalidField;
      if (!isInvalidField)
         continue;

      const auto &invalidField = static_cast<const RInvalidField &>(f);
      result.emplace_back(
         RCheckResult{invalidField.GetQualifiedFieldName(), invalidField.GetTypeName(), invalidField.GetError()});
   }
   return result;
}

ROOT::RResult<std::unique_ptr<ROOT::RFieldBase>>
ROOT::RFieldBase::Create(const std::string &fieldName, const std::string &typeName,
                         const ROOT::RCreateFieldOptions &options, const ROOT::RNTupleDescriptor *desc,
                         ROOT::DescriptorId_t fieldId)
{
   using ROOT::Internal::ParseArrayType;
   using ROOT::Internal::ParseUIntTypeToken;
   using ROOT::Internal::TokenizeTypeList;

   const auto resolvedType = ROOT::Internal::GetCanonicalTypePrefix(TClassEdit::ResolveTypedef(typeName.c_str()));

   thread_local CreateContext createContext;
   CreateContextGuard createContextGuard(createContext);
   if (options.GetReturnInvalidOnError())
      createContextGuard.SetContinueOnError(true);

   auto fnFail = [&fieldName,
                  &resolvedType](const std::string &errMsg,
                                 RInvalidField::RCategory cat =
                                    RInvalidField::RCategory::kTypeError) -> RResult<std::unique_ptr<RFieldBase>> {
      if (createContext.GetContinueOnError()) {
         return std::unique_ptr<RFieldBase>(std::make_unique<RInvalidField>(fieldName, resolvedType, errMsg, cat));
      } else {
         return R__FAIL(errMsg);
      }
   };

   if (resolvedType.empty())
      return R__FORWARD_RESULT(fnFail("no type name specified for field '" + fieldName + "'"));

   std::unique_ptr<ROOT::RFieldBase> result;

   const auto maybeGetChildId = [desc, fieldId](int childId) {
      if (desc) {
         const auto &fieldDesc = desc->GetFieldDescriptor(fieldId);
         return fieldDesc.GetLinkIds().at(childId);
      } else {
         return ROOT::kInvalidDescriptorId;
      }
   };

   // try-catch block to intercept any exception that may be thrown by Unwrap() so that this
   // function never throws but returns RResult::Error instead.
   try {
      if (auto [arrayBaseType, arraySizes] = ParseArrayType(resolvedType); !arraySizes.empty()) {
         std::unique_ptr<RFieldBase> arrayField = Create("_0", arrayBaseType, options, desc, fieldId).Unwrap();
         for (int i = arraySizes.size() - 1; i >= 0; --i) {
            arrayField =
               std::make_unique<RArrayField>((i == 0) ? fieldName : "_0", std::move(arrayField), arraySizes[i]);
         }
         return arrayField;
      }

      if (resolvedType == "bool") {
         result = std::make_unique<RField<bool>>(fieldName);
      } else if (resolvedType == "char") {
         result = std::make_unique<RField<char>>(fieldName);
      } else if (resolvedType == "std::byte") {
         result = std::make_unique<RField<std::byte>>(fieldName);
      } else if (resolvedType == "std::int8_t") {
         result = std::make_unique<RField<std::int8_t>>(fieldName);
      } else if (resolvedType == "std::uint8_t") {
         result = std::make_unique<RField<std::uint8_t>>(fieldName);
      } else if (resolvedType == "std::int16_t") {
         result = std::make_unique<RField<std::int16_t>>(fieldName);
      } else if (resolvedType == "std::uint16_t") {
         result = std::make_unique<RField<std::uint16_t>>(fieldName);
      } else if (resolvedType == "std::int32_t") {
         result = std::make_unique<RField<std::int32_t>>(fieldName);
      } else if (resolvedType == "std::uint32_t") {
         result = std::make_unique<RField<std::uint32_t>>(fieldName);
      } else if (resolvedType == "std::int64_t") {
         result = std::make_unique<RField<std::int64_t>>(fieldName);
      } else if (resolvedType == "std::uint64_t") {
         result = std::make_unique<RField<std::uint64_t>>(fieldName);
      } else if (resolvedType == "float") {
         result = std::make_unique<RField<float>>(fieldName);
      } else if (resolvedType == "double") {
         result = std::make_unique<RField<double>>(fieldName);
      } else if (resolvedType == "Double32_t") {
         result = std::make_unique<RField<double>>(fieldName);
         static_cast<RField<double> *>(result.get())->SetDouble32();
         // Prevent the type alias from being reset by returning early
         return result;
      } else if (resolvedType == "std::string") {
         result = std::make_unique<RField<std::string>>(fieldName);
      } else if (resolvedType == "TObject") {
         result = std::make_unique<RField<TObject>>(fieldName);
      } else if (resolvedType == "std::vector<bool>") {
         result = std::make_unique<RField<std::vector<bool>>>(fieldName);
      } else if (resolvedType.substr(0, 12) == "std::vector<") {
         std::string itemTypeName = resolvedType.substr(12, resolvedType.length() - 13);
         auto itemField = Create("_0", itemTypeName, options, desc, maybeGetChildId(0));
         result = std::make_unique<RVectorField>(fieldName, itemField.Unwrap());
      } else if (resolvedType.substr(0, 19) == "ROOT::VecOps::RVec<") {
         std::string itemTypeName = resolvedType.substr(19, resolvedType.length() - 20);
         auto itemField = Create("_0", itemTypeName, options, desc, maybeGetChildId(0));
         result = std::make_unique<RRVecField>(fieldName, itemField.Unwrap());
      } else if (resolvedType.substr(0, 11) == "std::array<") {
         auto arrayDef = TokenizeTypeList(resolvedType.substr(11, resolvedType.length() - 12));
         if (arrayDef.size() != 2) {
            return R__FORWARD_RESULT(fnFail("the template list for std::array must have exactly two elements"));
         }
         auto arrayLength = ParseUIntTypeToken(arrayDef[1]);
         auto itemField = Create("_0", arrayDef[0], options, desc, maybeGetChildId(0));
         result = std::make_unique<RArrayField>(fieldName, itemField.Unwrap(), arrayLength);
      } else if (resolvedType.substr(0, 13) == "std::variant<") {
         auto innerTypes = TokenizeTypeList(resolvedType.substr(13, resolvedType.length() - 14));
         std::vector<std::unique_ptr<RFieldBase>> items;
         items.reserve(innerTypes.size());
         for (unsigned int i = 0; i < innerTypes.size(); ++i) {
            items.emplace_back(
               Create("_" + std::to_string(i), innerTypes[i], options, desc, maybeGetChildId(i)).Unwrap());
         }
         result = std::make_unique<RVariantField>(fieldName, std::move(items));
      } else if (resolvedType.substr(0, 10) == "std::pair<") {
         auto innerTypes = TokenizeTypeList(resolvedType.substr(10, resolvedType.length() - 11));
         if (innerTypes.size() != 2) {
            return R__FORWARD_RESULT(fnFail("the type list for std::pair must have exactly two elements"));
         }
         std::array<std::unique_ptr<RFieldBase>, 2> items{
            Create("_0", innerTypes[0], options, desc, maybeGetChildId(0)).Unwrap(),
            Create("_1", innerTypes[1], options, desc, maybeGetChildId(1)).Unwrap()};
         result = std::make_unique<RPairField>(fieldName, std::move(items));
      } else if (resolvedType.substr(0, 11) == "std::tuple<") {
         auto innerTypes = TokenizeTypeList(resolvedType.substr(11, resolvedType.length() - 12));
         std::vector<std::unique_ptr<RFieldBase>> items;
         items.reserve(innerTypes.size());
         for (unsigned int i = 0; i < innerTypes.size(); ++i) {
            items.emplace_back(
               Create("_" + std::to_string(i), innerTypes[i], options, desc, maybeGetChildId(i)).Unwrap());
         }
         result = std::make_unique<RTupleField>(fieldName, std::move(items));
      } else if (resolvedType.substr(0, 12) == "std::bitset<") {
         auto size = ParseUIntTypeToken(resolvedType.substr(12, resolvedType.length() - 13));
         result = std::make_unique<RBitsetField>(fieldName, size);
      } else if (resolvedType.substr(0, 16) == "std::unique_ptr<") {
         std::string itemTypeName = resolvedType.substr(16, resolvedType.length() - 17);
         auto itemField = Create("_0", itemTypeName, options, desc, maybeGetChildId(0)).Unwrap();
         auto normalizedInnerTypeName = itemField->GetTypeName();
         result = std::make_unique<RUniquePtrField>(fieldName, "std::unique_ptr<" + normalizedInnerTypeName + ">",
                                                    std::move(itemField));
      } else if (resolvedType.substr(0, 14) == "std::optional<") {
         std::string itemTypeName = resolvedType.substr(14, resolvedType.length() - 15);
         auto itemField = Create("_0", itemTypeName, options, desc, maybeGetChildId(0)).Unwrap();
         auto normalizedInnerTypeName = itemField->GetTypeName();
         result = std::make_unique<ROptionalField>(fieldName, "std::optional<" + normalizedInnerTypeName + ">",
                                                   std::move(itemField));
      } else if (resolvedType.substr(0, 9) == "std::set<") {
         std::string itemTypeName = resolvedType.substr(9, resolvedType.length() - 10);
         auto itemField = Create("_0", itemTypeName, options, desc, maybeGetChildId(0)).Unwrap();
         auto normalizedInnerTypeName = itemField->GetTypeName();
         result =
            std::make_unique<RSetField>(fieldName, "std::set<" + normalizedInnerTypeName + ">", std::move(itemField));
      } else if (resolvedType.substr(0, 19) == "std::unordered_set<") {
         std::string itemTypeName = resolvedType.substr(19, resolvedType.length() - 20);
         auto itemField = Create("_0", itemTypeName, options, desc, maybeGetChildId(0)).Unwrap();
         auto normalizedInnerTypeName = itemField->GetTypeName();
         result = std::make_unique<RSetField>(fieldName, "std::unordered_set<" + normalizedInnerTypeName + ">",
                                              std::move(itemField));
      } else if (resolvedType.substr(0, 14) == "std::multiset<") {
         std::string itemTypeName = resolvedType.substr(14, resolvedType.length() - 15);
         auto itemField = Create("_0", itemTypeName, options, desc, maybeGetChildId(0)).Unwrap();
         auto normalizedInnerTypeName = itemField->GetTypeName();
         result = std::make_unique<RSetField>(fieldName, "std::multiset<" + normalizedInnerTypeName + ">",
                                              std::move(itemField));
      } else if (resolvedType.substr(0, 24) == "std::unordered_multiset<") {
         std::string itemTypeName = resolvedType.substr(24, resolvedType.length() - 25);
         auto itemField = Create("_0", itemTypeName, options, desc, maybeGetChildId(0)).Unwrap();
         auto normalizedInnerTypeName = itemField->GetTypeName();
         result = std::make_unique<RSetField>(fieldName, "std::unordered_multiset<" + normalizedInnerTypeName + ">",
                                              std::move(itemField));
      } else if (resolvedType.substr(0, 9) == "std::map<") {
         auto innerTypes = TokenizeTypeList(resolvedType.substr(9, resolvedType.length() - 10));
         if (innerTypes.size() != 2) {
            return R__FORWARD_RESULT(fnFail("the type list for std::map must have exactly two elements"));
         }

         auto itemField =
            Create("_0", "std::pair<" + innerTypes[0] + "," + innerTypes[1] + ">", options, desc, maybeGetChildId(0))
               .Unwrap();

         // We use the type names of subfields of the newly created item fields to create the map's type name to
         // ensure the inner type names are properly normalized.
         auto keyTypeName = itemField->GetConstSubfields()[0]->GetTypeName();
         auto valueTypeName = itemField->GetConstSubfields()[1]->GetTypeName();

         result = std::make_unique<RMapField>(fieldName, "std::map<" + keyTypeName + "," + valueTypeName + ">",
                                              std::move(itemField));
      } else if (resolvedType.substr(0, 19) == "std::unordered_map<") {
         auto innerTypes = TokenizeTypeList(resolvedType.substr(19, resolvedType.length() - 20));
         if (innerTypes.size() != 2)
            return R__FORWARD_RESULT(fnFail("the type list for std::unordered_map must have exactly two elements"));

         auto itemField =
            Create("_0", "std::pair<" + innerTypes[0] + "," + innerTypes[1] + ">", options, desc, maybeGetChildId(0))
               .Unwrap();

         // We use the type names of subfields of the newly created item fields to create the map's type name to
         // ensure the inner type names are properly normalized.
         auto keyTypeName = itemField->GetConstSubfields()[0]->GetTypeName();
         auto valueTypeName = itemField->GetConstSubfields()[1]->GetTypeName();

         result = std::make_unique<RMapField>(
            fieldName, "std::unordered_map<" + keyTypeName + "," + valueTypeName + ">", std::move(itemField));
      } else if (resolvedType.substr(0, 14) == "std::multimap<") {
         auto innerTypes = TokenizeTypeList(resolvedType.substr(14, resolvedType.length() - 15));
         if (innerTypes.size() != 2)
            return R__FORWARD_RESULT(fnFail("the type list for std::multimap must have exactly two elements"));

         auto itemField =
            Create("_0", "std::pair<" + innerTypes[0] + "," + innerTypes[1] + ">", options, desc, maybeGetChildId(0))
               .Unwrap();

         // We use the type names of subfields of the newly created item fields to create the map's type name to
         // ensure the inner type names are properly normalized.
         auto keyTypeName = itemField->GetConstSubfields()[0]->GetTypeName();
         auto valueTypeName = itemField->GetConstSubfields()[1]->GetTypeName();

         result = std::make_unique<RMapField>(fieldName, "std::multimap<" + keyTypeName + "," + valueTypeName + ">",
                                              std::move(itemField));
      } else if (resolvedType.substr(0, 24) == "std::unordered_multimap<") {
         auto innerTypes = TokenizeTypeList(resolvedType.substr(24, resolvedType.length() - 25));
         if (innerTypes.size() != 2)
            return R__FORWARD_RESULT(
               fnFail("the type list for std::unordered_multimap must have exactly two elements"));

         auto itemField =
            Create("_0", "std::pair<" + innerTypes[0] + "," + innerTypes[1] + ">", options, desc, maybeGetChildId(0))
               .Unwrap();

         // We use the type names of subfields of the newly created item fields to create the map's type name to
         // ensure the inner type names are properly normalized.
         auto keyTypeName = itemField->GetConstSubfields()[0]->GetTypeName();
         auto valueTypeName = itemField->GetConstSubfields()[1]->GetTypeName();

         result = std::make_unique<RMapField>(
            fieldName, "std::unordered_multimap<" + keyTypeName + "," + valueTypeName + ">", std::move(itemField));
      } else if (resolvedType.substr(0, 12) == "std::atomic<") {
         std::string itemTypeName = resolvedType.substr(12, resolvedType.length() - 13);
         auto itemField = Create("_0", itemTypeName, options, desc, maybeGetChildId(0)).Unwrap();
         auto normalizedInnerTypeName = itemField->GetTypeName();
         result = std::make_unique<RAtomicField>(fieldName, "std::atomic<" + normalizedInnerTypeName + ">",
                                                 std::move(itemField));
      } else if (resolvedType.substr(0, 25) == "ROOT::RNTupleCardinality<") {
         auto innerTypes = TokenizeTypeList(resolvedType.substr(25, resolvedType.length() - 26));
         if (innerTypes.size() != 1)
            return R__FORWARD_RESULT(fnFail("invalid cardinality template: " + resolvedType));
         const auto canonicalInnerType = ROOT::Internal::GetCanonicalTypePrefix(innerTypes[0]);
         if (canonicalInnerType == "std::uint32_t") {
            result = std::make_unique<RField<RNTupleCardinality<std::uint32_t>>>(fieldName);
         } else if (canonicalInnerType == "std::uint64_t") {
            result = std::make_unique<RField<RNTupleCardinality<std::uint64_t>>>(fieldName);
         } else {
            return R__FORWARD_RESULT(fnFail("invalid cardinality template: " + resolvedType));
         }
      }

      if (!result) {
         auto e = TEnum::GetEnum(resolvedType.c_str());
         if (e != nullptr) {
            result = std::make_unique<REnumField>(fieldName, typeName);
         }
      }

      if (!result) {
         auto cl = TClass::GetClass(typeName.c_str());
         // NOTE: if the class is not at least "Interpreted" we currently don't try to construct
         // the RClassField, as in that case we'd need to fetch the information from the StreamerInfo
         // rather than from TClass. This might be desirable in the future, but for now in this
         // situation we rely on field emulation instead.
         if (cl != nullptr && cl->GetState() >= TClass::kInterpreted) {
            createContextGuard.AddClassToStack(resolvedType);
            if (cl->GetCollectionProxy()) {
               result = std::make_unique<RProxiedCollectionField>(fieldName, typeName);
            } else {
               if (ROOT::Internal::GetRNTupleSerializationMode(cl) ==
                   ROOT::Internal::ERNTupleSerializationMode::kForceStreamerMode) {
                  result = std::make_unique<RStreamerField>(fieldName, typeName);
               } else {
                  result = std::make_unique<RClassField>(fieldName, typeName);
               }
            }
         } else if (options.GetEmulateUnknownTypes()) {
            assert(desc);
            const auto &fieldDesc = desc->GetFieldDescriptor(fieldId);

            std::vector<std::unique_ptr<RFieldBase>> memberFields;
            memberFields.reserve(fieldDesc.GetLinkIds().size());
            for (auto id : fieldDesc.GetLinkIds()) {
               const auto &memberDesc = desc->GetFieldDescriptor(id);
               auto field = Create(memberDesc.GetFieldName(), memberDesc.GetTypeName(), options, desc, id).Unwrap();
               memberFields.emplace_back(std::move(field));
            }
            R__ASSERT(typeName == fieldDesc.GetTypeName());
            auto recordField =
               Internal::CreateEmulatedField(fieldName, std::move(memberFields), fieldDesc.GetTypeName());
            recordField->fTypeAlias = fieldDesc.GetTypeAlias();
            return recordField;
         }
      }
   } catch (RException &e) {
      auto error = e.GetError();
      if (createContext.GetContinueOnError()) {
         return std::unique_ptr<RFieldBase>(std::make_unique<RInvalidField>(fieldName, typeName, error.GetReport(),
                                                                            RInvalidField::RCategory::kGeneric));
      } else {
         return error;
      }
   } catch (std::logic_error &e) {
      // Integer parsing error
      if (createContext.GetContinueOnError()) {
         return std::unique_ptr<RFieldBase>(
            std::make_unique<RInvalidField>(fieldName, typeName, e.what(), RInvalidField::RCategory::kGeneric));
      } else {
         return R__FAIL(e.what());
      }
   }

   if (result) {
      const auto normOrigType = ROOT::Internal::GetNormalizedUnresolvedTypeName(typeName);
      if (normOrigType != result->GetTypeName()) {
         result->fTypeAlias = normOrigType;
      }
      return result;
   }
   return R__FORWARD_RESULT(fnFail("unknown type: " + typeName, RInvalidField::RCategory::kUnknownType));
}

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RFieldBase::GetColumnRepresentations() const
{
   static RColumnRepresentations representations;
   return representations;
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RFieldBase::Clone(std::string_view newName) const
{
   auto clone = CloneImpl(newName);
   clone->fTypeAlias = fTypeAlias;
   clone->fOnDiskId = fOnDiskId;
   clone->fDescription = fDescription;
   // We can just copy the references because fColumnRepresentatives point into a static structure
   clone->fColumnRepresentatives = fColumnRepresentatives;
   return clone;
}

std::size_t ROOT::RFieldBase::AppendImpl(const void * /* from */)
{
   R__ASSERT(false && "A non-simple RField must implement its own AppendImpl");
   return 0;
}

void ROOT::RFieldBase::ReadGlobalImpl(ROOT::NTupleSize_t /*index*/, void * /* to */)
{
   R__ASSERT(false);
}

void ROOT::RFieldBase::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   ReadGlobalImpl(fPrincipalColumn->GetGlobalIndex(localIndex), to);
}

std::size_t ROOT::RFieldBase::ReadBulkImpl(const RBulkSpec &bulkSpec)
{
   const auto valueSize = GetValueSize();
   std::size_t nRead = 0;
   for (std::size_t i = 0; i < bulkSpec.fCount; ++i) {
      // Value not needed
      if (bulkSpec.fMaskReq && !bulkSpec.fMaskReq[i])
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

void *ROOT::RFieldBase::CreateObjectRawPtr() const
{
   void *where = operator new(GetValueSize());
   R__ASSERT(where != nullptr);
   ConstructValue(where);
   return where;
}

ROOT::RFieldBase::RValue ROOT::RFieldBase::CreateValue()
{
   void *obj = CreateObjectRawPtr();
   return RValue(this, std::shared_ptr<void>(obj, RSharedPtrDeleter(GetDeleter())));
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RFieldBase::SplitValue(const RValue & /*value*/) const
{
   return std::vector<RValue>();
}

void ROOT::RFieldBase::Attach(std::unique_ptr<ROOT::RFieldBase> child)
{
   // Note that during a model update, new fields will be attached to the zero field. The zero field, however,
   // does not change its inital state because only its sub fields get connected by RPageSink::UpdateSchema.
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("invalid attempt to attach subfield to already connected field"));
   child->fParent = this;
   fSubfields.emplace_back(std::move(child));
}

ROOT::NTupleSize_t ROOT::RFieldBase::EntryToColumnElementIndex(ROOT::NTupleSize_t globalIndex) const
{
   std::size_t result = globalIndex;
   for (auto f = this; f != nullptr; f = f->GetParent()) {
      auto parent = f->GetParent();
      if (parent && (parent->GetStructure() == ROOT::ENTupleStructure::kCollection ||
                     parent->GetStructure() == ROOT::ENTupleStructure::kVariant)) {
         return 0U;
      }
      result *= std::max(f->GetNRepetitions(), std::size_t{1U});
   }
   return result;
}

std::vector<ROOT::RFieldBase *> ROOT::RFieldBase::GetMutableSubfields()
{
   std::vector<RFieldBase *> result;
   result.reserve(fSubfields.size());
   for (const auto &f : fSubfields) {
      result.emplace_back(f.get());
   }
   return result;
}

std::vector<const ROOT::RFieldBase *> ROOT::RFieldBase::GetConstSubfields() const
{
   std::vector<const RFieldBase *> result;
   result.reserve(fSubfields.size());
   for (const auto &f : fSubfields) {
      result.emplace_back(f.get());
   }
   return result;
}

void ROOT::RFieldBase::FlushColumns()
{
   if (!fAvailableColumns.empty()) {
      const auto activeRepresentationIndex = fPrincipalColumn->GetRepresentationIndex();
      for (auto &column : fAvailableColumns) {
         if (column->GetRepresentationIndex() == activeRepresentationIndex) {
            column->Flush();
         }
      }
   }
}

void ROOT::RFieldBase::CommitCluster()
{
   if (!fAvailableColumns.empty()) {
      const auto activeRepresentationIndex = fPrincipalColumn->GetRepresentationIndex();
      for (auto &column : fAvailableColumns) {
         if (column->GetRepresentationIndex() == activeRepresentationIndex) {
            column->Flush();
         } else {
            column->CommitSuppressed();
         }
      }
   }
   CommitClusterImpl();
}

void ROOT::RFieldBase::SetDescription(std::string_view description)
{
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("cannot set field description once field is connected"));
   fDescription = std::string(description);
}

void ROOT::RFieldBase::SetOnDiskId(ROOT::DescriptorId_t id)
{
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("cannot set field ID once field is connected"));
   fOnDiskId = id;
}

/// Write the given value into columns. The value object has to be of the same type as the field.
/// Returns the number of uncompressed bytes written.
std::size_t ROOT::RFieldBase::Append(const void *from)
{
   if (~fTraits & kTraitMappable)
      return AppendImpl(from);

   fPrincipalColumn->Append(from);
   return fPrincipalColumn->GetElement()->GetPackedSize();
}

ROOT::RFieldBase::RBulkValues ROOT::RFieldBase::CreateBulk()
{
   return RBulkValues(this);
}

ROOT::RFieldBase::RValue ROOT::RFieldBase::BindValue(std::shared_ptr<void> objPtr)
{
   return RValue(this, objPtr);
}

std::size_t ROOT::RFieldBase::ReadBulk(const RBulkSpec &bulkSpec)
{
   if (fIsSimple) {
      /// For simple types, ignore the mask and memcopy the values into the destination
      fPrincipalColumn->ReadV(bulkSpec.fFirstIndex, bulkSpec.fCount, bulkSpec.fValues);
      std::fill(bulkSpec.fMaskAvail, bulkSpec.fMaskAvail + bulkSpec.fCount, true);
      return RBulkSpec::kAllSet;
   }

   return ReadBulkImpl(bulkSpec);
}

ROOT::RFieldBase::RSchemaIterator ROOT::RFieldBase::begin()
{
   return fSubfields.empty() ? RSchemaIterator(this, -1) : RSchemaIterator(fSubfields[0].get(), 0);
}

ROOT::RFieldBase::RSchemaIterator ROOT::RFieldBase::end()
{
   return RSchemaIterator(this, -1);
}

ROOT::RFieldBase::RConstSchemaIterator ROOT::RFieldBase::begin() const
{
   return fSubfields.empty() ? RConstSchemaIterator(this, -1) : RConstSchemaIterator(fSubfields[0].get(), 0);
}

ROOT::RFieldBase::RConstSchemaIterator ROOT::RFieldBase::end() const
{
   return RConstSchemaIterator(this, -1);
}

ROOT::RFieldBase::RConstSchemaIterator ROOT::RFieldBase::cbegin() const
{
   return fSubfields.empty() ? RConstSchemaIterator(this, -1) : RConstSchemaIterator(fSubfields[0].get(), 0);
}

ROOT::RFieldBase::RConstSchemaIterator ROOT::RFieldBase::cend() const
{
   return RConstSchemaIterator(this, -1);
}

ROOT::RFieldBase::RColumnRepresentations::Selection_t ROOT::RFieldBase::GetColumnRepresentatives() const
{
   if (fColumnRepresentatives.empty()) {
      return {GetColumnRepresentations().GetSerializationDefault()};
   }

   RColumnRepresentations::Selection_t result;
   result.reserve(fColumnRepresentatives.size());
   for (const auto &r : fColumnRepresentatives) {
      result.emplace_back(r.get());
   }
   return result;
}

void ROOT::RFieldBase::SetColumnRepresentatives(const RColumnRepresentations::Selection_t &representatives)
{
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("cannot set column representative once field is connected"));
   const auto &validTypes = GetColumnRepresentations().GetSerializationTypes();
   fColumnRepresentatives.clear();
   fColumnRepresentatives.reserve(representatives.size());
   for (const auto &r : representatives) {
      auto itRepresentative = std::find(validTypes.begin(), validTypes.end(), r);
      if (itRepresentative == std::end(validTypes))
         throw RException(R__FAIL("invalid column representative"));

      // don't add a duplicate representation
      if (std::find_if(fColumnRepresentatives.begin(), fColumnRepresentatives.end(),
                       [&r](const auto &rep) { return r == rep.get(); }) == fColumnRepresentatives.end())
         fColumnRepresentatives.emplace_back(*itRepresentative);
   }
}

const ROOT::RFieldBase::ColumnRepresentation_t &
ROOT::RFieldBase::EnsureCompatibleColumnTypes(const ROOT::RNTupleDescriptor &desc,
                                              std::uint16_t representationIndex) const
{
   static const ColumnRepresentation_t kEmpty;

   if (fOnDiskId == ROOT::kInvalidDescriptorId)
      throw RException(R__FAIL("No on-disk field information for `" + GetQualifiedFieldName() + "`"));

   ColumnRepresentation_t onDiskTypes;
   for (const auto &c : desc.GetColumnIterable(fOnDiskId)) {
      if (c.GetRepresentationIndex() == representationIndex)
         onDiskTypes.emplace_back(c.GetType());
   }
   if (onDiskTypes.empty()) {
      if (representationIndex == 0) {
         throw RException(R__FAIL("No on-disk column information for field `" + GetQualifiedFieldName() + "`"));
      }
      return kEmpty;
   }

   for (const auto &t : GetColumnRepresentations().GetDeserializationTypes()) {
      if (t == onDiskTypes)
         return t;
   }

   std::string columnTypeNames;
   for (const auto &t : onDiskTypes) {
      if (!columnTypeNames.empty())
         columnTypeNames += ", ";
      columnTypeNames += std::string("`") + ROOT::Internal::RColumnElementBase::GetColumnTypeName(t) + "`";
   }
   throw RException(R__FAIL("On-disk column types {" + columnTypeNames + "} for field `" + GetQualifiedFieldName() +
                            "` cannot be matched to its in-memory type `" + GetTypeName() + "` " +
                            "(representation index: " + std::to_string(representationIndex) + ")"));
}

size_t ROOT::RFieldBase::AddReadCallback(ReadCallback_t func)
{
   fReadCallbacks.push_back(func);
   fIsSimple = false;
   return fReadCallbacks.size() - 1;
}

void ROOT::RFieldBase::RemoveReadCallback(size_t idx)
{
   fReadCallbacks.erase(fReadCallbacks.begin() + idx);
   fIsSimple = (fTraits & kTraitMappable) && !fIsArtificial && fReadCallbacks.empty();
}

void ROOT::RFieldBase::AutoAdjustColumnTypes(const ROOT::RNTupleWriteOptions &options)
{
   if ((options.GetCompression() == 0) && HasDefaultColumnRepresentative()) {
      ColumnRepresentation_t rep = GetColumnRepresentations().GetSerializationDefault();
      for (auto &colType : rep) {
         switch (colType) {
         case ROOT::ENTupleColumnType::kSplitIndex64: colType = ROOT::ENTupleColumnType::kIndex64; break;
         case ROOT::ENTupleColumnType::kSplitIndex32: colType = ROOT::ENTupleColumnType::kIndex32; break;
         case ROOT::ENTupleColumnType::kSplitReal64: colType = ROOT::ENTupleColumnType::kReal64; break;
         case ROOT::ENTupleColumnType::kSplitReal32: colType = ROOT::ENTupleColumnType::kReal32; break;
         case ROOT::ENTupleColumnType::kSplitInt64: colType = ROOT::ENTupleColumnType::kInt64; break;
         case ROOT::ENTupleColumnType::kSplitInt32: colType = ROOT::ENTupleColumnType::kInt32; break;
         case ROOT::ENTupleColumnType::kSplitInt16: colType = ROOT::ENTupleColumnType::kInt16; break;
         case ROOT::ENTupleColumnType::kSplitUInt64: colType = ROOT::ENTupleColumnType::kUInt64; break;
         case ROOT::ENTupleColumnType::kSplitUInt32: colType = ROOT::ENTupleColumnType::kUInt32; break;
         case ROOT::ENTupleColumnType::kSplitUInt16: colType = ROOT::ENTupleColumnType::kUInt16; break;
         default: break;
         }
      }
      SetColumnRepresentatives({rep});
   }

   if (fTypeAlias == "Double32_t")
      SetColumnRepresentatives({{ROOT::ENTupleColumnType::kSplitReal32}});
}

void ROOT::RFieldBase::ConnectPageSink(ROOT::Internal::RPageSink &pageSink, ROOT::NTupleSize_t firstEntry)
{
   if (dynamic_cast<ROOT::RFieldZero *>(this))
      throw RException(R__FAIL("invalid attempt to connect zero field to page sink"));
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("invalid attempt to connect an already connected field to a page sink"));

   AutoAdjustColumnTypes(pageSink.GetWriteOptions());

   GenerateColumns();
   for (auto &column : fAvailableColumns) {
      // Only the first column of every representation can be a deferred column. In all column representations,
      // larger column indexes are data columns of collections (string, streamer) and thus
      // they have no elements on late model extension
      auto firstElementIndex = (column->GetIndex() == 0) ? EntryToColumnElementIndex(firstEntry) : 0;
      column->ConnectPageSink(fOnDiskId, pageSink, firstElementIndex);
   }

   if (HasExtraTypeInfo()) {
      pageSink.RegisterOnCommitDatasetCallback(
         [this](ROOT::Internal::RPageSink &sink) { sink.UpdateExtraTypeInfo(GetExtraTypeInfo()); });
   }

   fState = EState::kConnectedToSink;
}

void ROOT::RFieldBase::ConnectPageSource(ROOT::Internal::RPageSource &pageSource)
{
   if (dynamic_cast<ROOT::RFieldZero *>(this))
      throw RException(R__FAIL("invalid attempt to connect zero field to page source"));
   if (fState != EState::kUnconnected)
      throw RException(R__FAIL("invalid attempt to connect an already connected field to a page source"));

   if (!fColumnRepresentatives.empty())
      throw RException(R__FAIL("fixed column representative only valid when connecting to a page sink"));
   if (!fDescription.empty())
      throw RException(R__FAIL("setting description only valid when connecting to a page sink"));

   BeforeConnectPageSource(pageSource);

   for (auto &f : fSubfields) {
      if (f->GetOnDiskId() == ROOT::kInvalidDescriptorId) {
         f->SetOnDiskId(pageSource.GetSharedDescriptorGuard()->FindFieldId(f->GetFieldName(), GetOnDiskId()));
      }
      f->ConnectPageSource(pageSource);
   }

   // Do not generate columns nor set fColumnRepresentatives for artificial fields.
   if (!fIsArtificial) {
      const auto descriptorGuard = pageSource.GetSharedDescriptorGuard();
      const ROOT::RNTupleDescriptor &desc = descriptorGuard.GetRef();
      GenerateColumns(desc);
      if (fColumnRepresentatives.empty()) {
         // If we didn't get columns from the descriptor, ensure that we actually expect a field without columns
         for (const auto &t : GetColumnRepresentations().GetDeserializationTypes()) {
            if (t.empty()) {
               fColumnRepresentatives = {t};
               break;
            }
         }
      }
      R__ASSERT(!fColumnRepresentatives.empty());
      if (fOnDiskId != ROOT::kInvalidDescriptorId) {
         const auto &fieldDesc = desc.GetFieldDescriptor(fOnDiskId);
         fOnDiskTypeVersion = fieldDesc.GetTypeVersion();
         if (fieldDesc.GetTypeChecksum().has_value())
            fOnDiskTypeChecksum = *fieldDesc.GetTypeChecksum();
      }
   }
   for (auto &column : fAvailableColumns)
      column->ConnectPageSource(fOnDiskId, pageSource);

   AfterConnectPageSource();

   fState = EState::kConnectedToSource;
}

void ROOT::RFieldBase::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitField(*this);
}
