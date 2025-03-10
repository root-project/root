/// \file RFieldMeta.cxx
/// \ingroup NTuple ROOT7
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

// This file has concrete RField implementations that depend on ROOT Meta:
//  - RClassField
//  - REnumField
//  - RPairField
//  - RProxiedCollectionField
//    - RMapField
//    - RSetField
//  - RStreamerField
//  - RPairField
//  - RField<TObject>
//  - RVariantField

#include <ROOT/RField.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RFieldUtils.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RSpan.hxx>

#include <TBaseClass.h>
#include <TBufferFile.h>
#include <TClass.h>
#include <TDataMember.h>
#include <TEnum.h>
#include <TObject.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TRealData.h>
#include <TSchemaRule.h>
#include <TSchemaRuleSet.h>
#include <TStreamerElement.h>
#include <TVirtualObject.h>
#include <TVirtualStreamerInfo.h>

#include <algorithm>
#include <array>
#include <cstddef> // std::size_t
#include <cstdint> // std::uint32_t et al.
#include <cstring> // for memset
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>

namespace {

TClass *EnsureValidClass(std::string_view className)
{
   auto cl = TClass::GetClass(std::string(className).c_str());
   if (cl == nullptr) {
      throw ROOT::RException(R__FAIL("RField: no I/O support for type " + std::string(className)));
   }
   return cl;
}

TEnum *EnsureValidEnum(std::string_view enumName)
{
   auto e = TEnum::GetEnum(std::string(enumName).c_str());
   if (e == nullptr) {
      throw ROOT::RException(R__FAIL("RField: no I/O support for enum type " + std::string(enumName)));
   }
   return e;
}

} // anonymous namespace

ROOT::Experimental::RClassField::RClassField(std::string_view fieldName, const RClassField &source)
   : ROOT::Experimental::RFieldBase(fieldName, source.GetTypeName(), ROOT::ENTupleStructure::kRecord,
                                    false /* isSimple */),
     fClass(source.fClass),
     fSubfieldsInfo(source.fSubfieldsInfo),
     fMaxAlignment(source.fMaxAlignment)
{
   for (const auto &f : source.GetConstSubfields()) {
      RFieldBase::Attach(f->Clone(f->GetFieldName()));
   }
   fTraits = source.GetTraits();
}

ROOT::Experimental::RClassField::RClassField(std::string_view fieldName, std::string_view className)
   : RClassField(fieldName, EnsureValidClass(className))
{
}

ROOT::Experimental::RClassField::RClassField(std::string_view fieldName, TClass *classp)
   : ROOT::Experimental::RFieldBase(fieldName, Internal::GetRenormalizedTypeName(classp->GetName()),
                                    ROOT::ENTupleStructure::kRecord, false /* isSimple */),
     fClass(classp)
{
   if (fClass->GetState() < TClass::kInterpreted) {
      throw RException(R__FAIL(std::string("RField: RClassField \"") + classp->GetName() +
                               " cannot be constructed from a class that's not at least Interpreted"));
   }
   // Avoid accidentally supporting std types through TClass.
   if (fClass->Property() & kIsDefinedInStd) {
      throw RException(R__FAIL(std::string(GetTypeName()) + " is not supported"));
   }
   if (GetTypeName() == "TObject") {
      throw RException(R__FAIL("TObject is only supported through RField<TObject>"));
   }
   if (fClass->GetCollectionProxy()) {
      throw RException(R__FAIL(std::string(GetTypeName()) + " has an associated collection proxy; "
                                                            "use RProxiedCollectionField instead"));
   }
   // Classes with, e.g., custom streamers are not supported through this field. Empty classes, however, are.
   // Can be overwritten with the "rntuple.streamerMode=true" class attribute
   if (!fClass->CanSplit() && fClass->Size() > 1 &&
       Internal::GetRNTupleSerializationMode(fClass) != Internal::ERNTupleSerializationMode::kForceNativeMode) {
      throw RException(R__FAIL(GetTypeName() + " cannot be stored natively in RNTuple"));
   }
   if (Internal::GetRNTupleSerializationMode(fClass) == Internal::ERNTupleSerializationMode::kForceStreamerMode) {
      throw RException(R__FAIL(GetTypeName() + " has streamer mode enforced, not supported as native RNTuple class"));
   }

   if (!(fClass->ClassProperty() & kClassHasExplicitCtor))
      fTraits |= kTraitTriviallyConstructible;
   if (!(fClass->ClassProperty() & kClassHasExplicitDtor))
      fTraits |= kTraitTriviallyDestructible;

   int i = 0;
   const auto *bases = fClass->GetListOfBases();
   assert(bases);
   for (auto baseClass : ROOT::Detail::TRangeStaticCast<TBaseClass>(*bases)) {
      if (baseClass->GetDelta() < 0) {
         throw RException(R__FAIL(std::string("virtual inheritance is not supported: ") + GetTypeName() +
                                  " virtually inherits from " + baseClass->GetName()));
      }
      TClass *c = baseClass->GetClassPointer();
      auto subField =
         RFieldBase::Create(std::string(kPrefixInherited) + "_" + std::to_string(i), c->GetName()).Unwrap();
      fTraits &= subField->GetTraits();
      Attach(std::move(subField), RSubFieldInfo{kBaseClass, static_cast<std::size_t>(baseClass->GetDelta())});
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

      // NOTE: we use the already-resolved type name for the fields, otherwise TClass::GetClass may fail to resolve
      // context-dependent types (e.g. typedefs defined in the class itself - which will not be fully qualified in
      // the string returned by dataMember->GetFullTypeName())
      std::string typeName{dataMember->GetTrueTypeName()};
      // RFieldBase::Create() set subField->fTypeAlias based on the assumption that the user specified typeName, which
      // already went through one round of type resolution.
      std::string origTypeName{dataMember->GetFullTypeName()};

      // For C-style arrays, complete the type name with the size for each dimension, e.g. `int[4][2]`
      if (dataMember->Property() & kIsArray) {
         for (int dim = 0, n = dataMember->GetArrayDim(); dim < n; ++dim) {
            const auto addedStr = "[" + std::to_string(dataMember->GetMaxIndex(dim)) + "]";
            typeName += addedStr;
            origTypeName += addedStr;
         }
      }

      auto subField = RFieldBase::Create(dataMember->GetName(), typeName).Unwrap();

      const auto normTypeName = Internal::GetNormalizedUnresolvedTypeName(origTypeName);
      if (normTypeName == subField->GetTypeName()) {
         subField->fTypeAlias = "";
      } else {
         subField->fTypeAlias = normTypeName;
      }

      fTraits &= subField->GetTraits();
      Attach(std::move(subField), RSubFieldInfo{kDataMember, static_cast<std::size_t>(dataMember->GetOffset())});
   }
   fTraits |= kTraitTypeChecksum;
}

void ROOT::Experimental::RClassField::Attach(std::unique_ptr<RFieldBase> child, RSubFieldInfo info)
{
   fMaxAlignment = std::max(fMaxAlignment, child->GetAlignment());
   fSubfieldsInfo.push_back(info);
   RFieldBase::Attach(std::move(child));
}

std::vector<const ROOT::TSchemaRule *> ROOT::Experimental::RClassField::FindRules(const RFieldDescriptor *fieldDesc)
{
   ROOT::Detail::TSchemaRuleSet::TMatches rules;
   const auto ruleset = fClass->GetSchemaRules();
   if (!ruleset)
      return rules;

   if (!fieldDesc) {
      // If we have no on-disk information for the field, we still process the rules on the current in-memory version
      // of the class
      rules = ruleset->FindRules(fClass->GetName(), fClass->GetClassVersion(), fClass->GetCheckSum());
   } else {
      // We do have an on-disk field that correspond to the current RClassField instance. Ask for rules matching the
      // on-disk version of the field.
      if (fieldDesc->GetTypeChecksum()) {
         rules =
            ruleset->FindRules(fieldDesc->GetTypeName(), fieldDesc->GetTypeVersion(), *fieldDesc->GetTypeChecksum());
      } else {
         rules = ruleset->FindRules(fieldDesc->GetTypeName(), fieldDesc->GetTypeVersion());
      }
   }

   // Cleanup and sort rules
   // Check that any any given source member uses the same type in all rules
   std::unordered_map<std::string, std::string> sourceNameAndType;
   std::size_t nskip = 0; // skip whole-object-rules that were moved to the end of the rules vector
   for (auto itr = rules.begin(); itr != rules.end() - nskip;) {
      const auto rule = *itr;

      // Erase unknown rule types
      if (rule->GetRuleType() != ROOT::TSchemaRule::kReadRule) {
         R__LOG_WARNING(ROOT::Internal::NTupleLog())
            << "ignoring I/O customization rule with unsupported type: " << rule->GetRuleType();
         itr = rules.erase(itr);
         continue;
      }

      bool hasConflictingSourceMembers = false;
      for (auto source : TRangeDynCast<TSchemaRule::TSources>(rule->GetSource())) {
         auto memberType = source->GetTypeForDeclaration() + source->GetDimensions();
         auto [itrSrc, isNew] = sourceNameAndType.emplace(source->GetName(), memberType);
         if (!isNew && (itrSrc->second != memberType)) {
            R__LOG_WARNING(ROOT::Internal::NTupleLog())
               << "ignoring I/O customization rule due to conflicting source member type: " << itrSrc->second << " vs. "
               << memberType << " for member " << source->GetName();
            hasConflictingSourceMembers = true;
            break;
         }
      }
      if (hasConflictingSourceMembers) {
         itr = rules.erase(itr);
         continue;
      }

      // Rules targeting the entire object need to be executed at the end
      if (rule->GetTarget() == nullptr) {
         nskip++;
         if (itr != rules.end() - nskip)
            std::iter_swap(itr++, rules.end() - nskip);
         continue;
      }

      // For the time being, we only support rules targeting transient members
      bool hasPersistentTarget = false;
      for (auto target : ROOT::Detail::TRangeStaticCast<TObjString>(*rule->GetTarget())) {
         const auto dataMember = fClass->GetDataMember(target->GetString());
         if (!dataMember || dataMember->IsPersistent()) {
            R__LOG_WARNING(ROOT::Internal::NTupleLog())
               << "ignoring I/O customization rule with non-transient member: " << dataMember->GetName();
            hasPersistentTarget = true;
            break;
         }
      }
      if (hasPersistentTarget) {
         itr = rules.erase(itr);
         continue;
      }

      ++itr;
   }

   return rules;
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RClassField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RClassField>(new RClassField(newName, *this));
}

std::size_t ROOT::Experimental::RClassField::AppendImpl(const void *from)
{
   std::size_t nbytes = 0;
   for (unsigned i = 0; i < fSubfields.size(); i++) {
      nbytes += CallAppendOn(*fSubfields[i], static_cast<const unsigned char *>(from) + fSubfieldsInfo[i].fOffset);
   }
   return nbytes;
}

void ROOT::Experimental::RClassField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   for (const auto &[_, si] : fStagingItems) {
      CallReadOn(*si.fField, globalIndex, fStagingArea.get() + si.fOffset);
   }
   for (unsigned i = 0; i < fSubfields.size(); i++) {
      CallReadOn(*fSubfields[i], globalIndex, static_cast<unsigned char *>(to) + fSubfieldsInfo[i].fOffset);
   }
}

void ROOT::Experimental::RClassField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   for (const auto &[_, si] : fStagingItems) {
      CallReadOn(*si.fField, localIndex, fStagingArea.get() + si.fOffset);
   }
   for (unsigned i = 0; i < fSubfields.size(); i++) {
      CallReadOn(*fSubfields[i], localIndex, static_cast<unsigned char *>(to) + fSubfieldsInfo[i].fOffset);
   }
}

ROOT::DescriptorId_t ROOT::Experimental::RClassField::LookupMember(const RNTupleDescriptor &desc,
                                                                   std::string_view memberName,
                                                                   ROOT::DescriptorId_t classFieldId)
{
   auto idSourceMember = desc.FindFieldId(memberName, classFieldId);
   if (idSourceMember != ROOT::kInvalidDescriptorId)
      return idSourceMember;

   for (const auto &subFieldDesc : desc.GetFieldIterable(classFieldId)) {
      const auto subFieldName = subFieldDesc.GetFieldName();
      if (subFieldName.length() > 2 && subFieldName[0] == ':' && subFieldName[1] == '_') {
         idSourceMember = LookupMember(desc, memberName, subFieldDesc.GetId());
         if (idSourceMember != ROOT::kInvalidDescriptorId)
            return idSourceMember;
      }
   }

   return ROOT::kInvalidDescriptorId;
}

void ROOT::Experimental::RClassField::SetStagingClass(const std::string &className, unsigned int classVersion)
{
   TClass::GetClass(className.c_str())->GetStreamerInfo(classVersion);
   if (classVersion != GetTypeVersion()) {
      fStagingClass = TClass::GetClass((className + std::string("@@") + std::to_string(classVersion)).c_str());
   } else {
      fStagingClass = fClass;
   }
   R__ASSERT(fStagingClass);
}

void ROOT::Experimental::RClassField::PrepareStagingArea(const std::vector<const TSchemaRule *> &rules,
                                                         const RNTupleDescriptor &desc,
                                                         const RFieldDescriptor &classFieldDesc)
{
   std::size_t stagingAreaSize = 0;
   for (const auto rule : rules) {
      for (auto source : TRangeDynCast<TSchemaRule::TSources>(rule->GetSource())) {
         auto [itr, isNew] = fStagingItems.emplace(source->GetName(), RStagingItem());
         if (!isNew) {
            // This source member has already been processed by another rule (and we only support one type per member)
            continue;
         }
         RStagingItem &stagingItem = itr->second;

         const auto memberFieldId = LookupMember(desc, source->GetName(), classFieldDesc.GetId());
         if (memberFieldId == kInvalidDescriptorId) {
            throw RException(R__FAIL(std::string("cannot find on disk rule source member ") + GetTypeName() + "." +
                                     source->GetName()));
         }
         const auto &memberFieldDesc = desc.GetFieldDescriptor(memberFieldId);

         auto memberType = source->GetTypeForDeclaration() + source->GetDimensions();
         stagingItem.fField = Create("" /* we don't need a field name */, std::string(memberType)).Unwrap();
         stagingItem.fField->SetOnDiskId(memberFieldDesc.GetId());

         stagingItem.fOffset = fStagingClass->GetDataMemberOffset(source->GetName());
         // Since we successfully looked up the source member in the RNTuple on-disk meta-data, we expect it
         // to be present in the TClass instance, too.
         R__ASSERT(stagingItem.fOffset != TVirtualStreamerInfo::kMissing);
         stagingAreaSize = std::max(stagingAreaSize, stagingItem.fOffset + stagingItem.fField->GetValueSize());
      }
   }

   if (stagingAreaSize) {
      R__ASSERT(static_cast<Int_t>(stagingAreaSize) <= fStagingClass->Size()); // we may have removed rules
      fStagingArea = ROOT::Internal::MakeUninitArray<unsigned char>(stagingAreaSize);
   }
}

void ROOT::Experimental::RClassField::AddReadCallbacksFromIORule(const TSchemaRule *rule)
{
   auto func = rule->GetReadFunctionPointer();
   R__ASSERT(func != nullptr);
   fReadCallbacks.emplace_back([func, stagingClass = fStagingClass, stagingArea = fStagingArea.get()](void *target) {
      TVirtualObject onfileObj{nullptr};
      onfileObj.fClass = stagingClass;
      onfileObj.fObject = stagingArea;
      func(static_cast<char *>(target), &onfileObj);
      onfileObj.fObject = nullptr; // TVirtualObject does not own the value
   });
}

void ROOT::Experimental::RClassField::BeforeConnectPageSource(Internal::RPageSource &pageSource)
{
   std::vector<const TSchemaRule *> rules;
   std::unordered_set<std::string> knownSubfields;

   if (GetOnDiskId() == kInvalidDescriptorId) {
      // This can happen for added base classes or added members of class type
      rules = FindRules(nullptr);
      if (!rules.empty())
         SetStagingClass(GetTypeName(), GetTypeVersion());
   } else {
      const auto descriptorGuard = pageSource.GetSharedDescriptorGuard();
      const RNTupleDescriptor &desc = descriptorGuard.GetRef();
      const auto &fieldDesc = desc.GetFieldDescriptor(GetOnDiskId());

      // Check that we have the same type.
      // TODO(jblomer): relax for class rename rule
      if (GetTypeName() != fieldDesc.GetTypeName()) {
         throw RException(R__FAIL("incompatible type name for field " + GetFieldName() + ": " + GetTypeName() +
                                  " vs. " + fieldDesc.GetTypeName()));
      }

      for (auto linkId : fieldDesc.GetLinkIds()) {
         const auto &subFieldDesc = desc.GetFieldDescriptor(linkId);
         knownSubfields.insert(subFieldDesc.GetFieldName());
      }

      rules = FindRules(&fieldDesc);
      if (!rules.empty()) {
         SetStagingClass(fieldDesc.GetTypeName(), fieldDesc.GetTypeVersion());
         PrepareStagingArea(rules, desc, fieldDesc);
         for (auto &[_, si] : fStagingItems)
            Internal::CallConnectPageSourceOnField(*si.fField, pageSource);
      }
   }

   for (const auto rule : rules) {
      AddReadCallbacksFromIORule(rule);
   }

   // Iterate over all sub fields in memory and mark those as missing that are not in the descriptor.
   for (auto &field : fSubfields) {
      if (knownSubfields.count(field->GetFieldName()) == 0) {
         field->SetArtificial();
      }
   }
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
   result.reserve(fSubfields.size());
   for (unsigned i = 0; i < fSubfields.size(); i++) {
      result.emplace_back(
         fSubfields[i]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), basePtr + fSubfieldsInfo[i].fOffset)));
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

std::uint32_t ROOT::Experimental::RClassField::GetTypeChecksum() const
{
   return fClass->GetCheckSum();
}

void ROOT::Experimental::RClassField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitClassField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::REnumField::REnumField(std::string_view fieldName, std::string_view enumName)
   : REnumField(fieldName, EnsureValidEnum(enumName))
{
}

ROOT::Experimental::REnumField::REnumField(std::string_view fieldName, TEnum *enump)
   : ROOT::Experimental::RFieldBase(fieldName, Internal::GetRenormalizedTypeName(enump->GetQualifiedName()),
                                    ROOT::ENTupleStructure::kLeaf, false /* isSimple */)
{
   // Avoid accidentally supporting std types through TEnum.
   if (enump->Property() & kIsDefinedInStd) {
      throw RException(R__FAIL(GetTypeName() + " is not supported"));
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
   default: throw RException(R__FAIL("Unsupported underlying integral type for enum type " + GetTypeName()));
   }

   fTraits |= kTraitTriviallyConstructible | kTraitTriviallyDestructible;
}

ROOT::Experimental::REnumField::REnumField(std::string_view fieldName, std::string_view enumName,
                                           std::unique_ptr<RFieldBase> intField)
   : ROOT::Experimental::RFieldBase(fieldName, enumName, ROOT::ENTupleStructure::kLeaf, false /* isSimple */)
{
   Attach(std::move(intField));
   fTraits |= kTraitTriviallyConstructible | kTraitTriviallyDestructible;
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::REnumField::CloneImpl(std::string_view newName) const
{
   auto newIntField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::unique_ptr<REnumField>(new REnumField(newName, GetTypeName(), std::move(newIntField)));
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::REnumField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   result.emplace_back(fSubfields[0]->BindValue(value.GetPtr<void>()));
   return result;
}

void ROOT::Experimental::REnumField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitEnumField(*this);
}

//------------------------------------------------------------------------------

std::string
ROOT::Experimental::RPairField::RPairField::GetTypeList(const std::array<std::unique_ptr<RFieldBase>, 2> &itemFields)
{
   return itemFields[0]->GetTypeName() + "," + itemFields[1]->GetTypeName();
}

ROOT::Experimental::RPairField::RPairField(std::string_view fieldName,
                                           std::array<std::unique_ptr<RFieldBase>, 2> itemFields,
                                           const std::array<std::size_t, 2> &offsets)
   : ROOT::Experimental::RRecordField(fieldName, "std::pair<" + GetTypeList(itemFields) + ">")
{
   AttachItemFields(std::move(itemFields));
   fOffsets.push_back(offsets[0]);
   fOffsets.push_back(offsets[1]);
}

ROOT::Experimental::RPairField::RPairField(std::string_view fieldName,
                                           std::array<std::unique_ptr<RFieldBase>, 2> itemFields)
   : ROOT::Experimental::RRecordField(fieldName, "std::pair<" + GetTypeList(itemFields) + ">")
{
   AttachItemFields(std::move(itemFields));

   // ISO C++ does not guarantee any specific layout for `std::pair`; query TClass for the member offsets
   auto *c = TClass::GetClass(GetTypeName().c_str());
   if (!c)
      throw RException(R__FAIL("cannot get type information for " + GetTypeName()));
   fSize = c->Size();

   auto firstElem = c->GetRealData("first");
   if (!firstElem)
      throw RException(R__FAIL("first: no such member"));
   fOffsets.push_back(firstElem->GetThisOffset());

   auto secondElem = c->GetRealData("second");
   if (!secondElem)
      throw RException(R__FAIL("second: no such member"));
   fOffsets.push_back(secondElem->GetThisOffset());
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

ROOT::Experimental::RProxiedCollectionField::RProxiedCollectionField(std::string_view fieldName, TClass *classp)
   : RFieldBase(fieldName, Internal::GetRenormalizedTypeName(classp->GetName()), ROOT::ENTupleStructure::kCollection,
                false /* isSimple */),
     fNWritten(0)
{
   if (!classp->GetCollectionProxy())
      throw RException(R__FAIL(std::string(GetTypeName()) + " has no associated collection proxy"));

   fProxy.reset(classp->GetCollectionProxy()->Generate());
   fProperties = fProxy->GetProperties();
   fCollectionType = fProxy->GetCollectionType();
   if (fProxy->HasPointers())
      throw RException(R__FAIL("collection proxies whose value type is a pointer are not supported"));
   if (!fProxy->GetCollectionClass()->HasDictionary()) {
      throw RException(R__FAIL("dictionary not available for type " +
                               Internal::GetRenormalizedTypeName(fProxy->GetCollectionClass()->GetName())));
   }

   fIFuncsRead = RCollectionIterableOnce::GetIteratorFuncs(fProxy.get(), true /* readFromDisk */);
   fIFuncsWrite = RCollectionIterableOnce::GetIteratorFuncs(fProxy.get(), false /* readFromDisk */);
}

ROOT::Experimental::RProxiedCollectionField::RProxiedCollectionField(std::string_view fieldName,
                                                                     std::string_view typeName,
                                                                     std::unique_ptr<RFieldBase> itemField)
   : RProxiedCollectionField(fieldName, EnsureValidClass(typeName))
{
   fItemSize = itemField->GetValueSize();
   Attach(std::move(itemField));
}

ROOT::Experimental::RProxiedCollectionField::RProxiedCollectionField(std::string_view fieldName,
                                                                     std::string_view typeName)
   : RProxiedCollectionField(fieldName, EnsureValidClass(typeName))
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
      case EDataType::kChar_t: itemField = std::make_unique<RField<char>>("_0"); break;
      case EDataType::kUChar_t: itemField = std::make_unique<RField<std::uint8_t>>("_0"); break;
      case EDataType::kShort_t: itemField = std::make_unique<RField<std::int16_t>>("_0"); break;
      case EDataType::kUShort_t: itemField = std::make_unique<RField<std::uint16_t>>("_0"); break;
      case EDataType::kInt_t: itemField = std::make_unique<RField<std::int32_t>>("_0"); break;
      case EDataType::kUInt_t: itemField = std::make_unique<RField<std::uint32_t>>("_0"); break;
      case EDataType::kLong_t:
      case EDataType::kLong64_t: itemField = std::make_unique<RField<std::int64_t>>("_0"); break;
      case EDataType::kULong_t:
      case EDataType::kULong64_t: itemField = std::make_unique<RField<std::uint64_t>>("_0"); break;
      case EDataType::kFloat_t: itemField = std::make_unique<RField<float>>("_0"); break;
      case EDataType::kDouble_t: itemField = std::make_unique<RField<double>>("_0"); break;
      case EDataType::kBool_t: itemField = std::make_unique<RField<bool>>("_0"); break;
      default: throw RException(R__FAIL("unsupported value type"));
      }
   }

   fItemSize = itemField->GetValueSize();
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RProxiedCollectionField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
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
      nbytes += CallAppendOn(*fSubfields[0], ptr);
      count++;
   }

   fNWritten += count;
   fPrincipalColumn->Append(&fNWritten);
   return nbytes + fPrincipalColumn->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RProxiedCollectionField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   ROOT::NTupleSize_t nItems;
   RNTupleLocalIndex collectionStart;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);

   TVirtualCollectionProxy::TPushPop RAII(fProxy.get(), to);
   void *obj =
      fProxy->Allocate(static_cast<std::uint32_t>(nItems), (fProperties & TVirtualCollectionProxy::kNeedDelete));

   unsigned i = 0;
   for (auto elementPtr : RCollectionIterableOnce{obj, fIFuncsRead, fProxy.get(),
                                                  (fCollectionType == kSTLvector || obj != to ? fItemSize : 0U)}) {
      CallReadOn(*fSubfields[0], collectionStart + (i++), elementPtr);
   }
   if (obj != to)
      fProxy->Commit(obj);
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RProxiedCollectionField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RProxiedCollectionField::GenerateColumns()
{
   GenerateColumnsImpl<Internal::RColumnIndex>();
}

void ROOT::Experimental::RProxiedCollectionField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<Internal::RColumnIndex>(desc);
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
      return std::make_unique<RProxiedCollectionDeleter>(fProxy, GetDeleterOf(*fSubfields[0]), itemSize);
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
      result.emplace_back(fSubfields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), ptr)));
   }
   return result;
}

void ROOT::Experimental::RProxiedCollectionField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitProxiedCollectionField(*this);
}

//------------------------------------------------------------------------------

ROOT::Experimental::RMapField::RMapField(std::string_view fieldName, std::string_view typeName,
                                         std::unique_ptr<RFieldBase> itemField)
   : RProxiedCollectionField(fieldName, EnsureValidClass(typeName))
{
   if (!dynamic_cast<RPairField *>(itemField.get()))
      throw RException(R__FAIL("RMapField inner field type must be of RPairField"));

   auto *itemClass = fProxy->GetValueClass();
   fItemSize = itemClass->GetClassSize();

   Attach(std::move(itemField));
}

//------------------------------------------------------------------------------

ROOT::Experimental::RSetField::RSetField(std::string_view fieldName, std::string_view typeName,
                                         std::unique_ptr<RFieldBase> itemField)
   : ROOT::Experimental::RProxiedCollectionField(fieldName, typeName, std::move(itemField))
{
}

//------------------------------------------------------------------------------

namespace {

/// Used in RStreamerField::AppendImpl() in order to record the encountered streamer info records
class TBufferRecStreamer : public TBufferFile {
public:
   using RCallbackStreamerInfo = std::function<void(TVirtualStreamerInfo *)>;

private:
   RCallbackStreamerInfo fCallbackStreamerInfo;

public:
   TBufferRecStreamer(TBuffer::EMode mode, Int_t bufsiz, RCallbackStreamerInfo callbackStreamerInfo)
      : TBufferFile(mode, bufsiz), fCallbackStreamerInfo(callbackStreamerInfo)
   {
   }
   void TagStreamerInfo(TVirtualStreamerInfo *info) final { fCallbackStreamerInfo(info); }
};

} // anonymous namespace

ROOT::Experimental::RStreamerField::RStreamerField(std::string_view fieldName, std::string_view className,
                                                   std::string_view typeAlias)
   : RStreamerField(fieldName, EnsureValidClass(className))
{
   fTypeAlias = typeAlias;
}

ROOT::Experimental::RStreamerField::RStreamerField(std::string_view fieldName, TClass *classp)
   : ROOT::Experimental::RFieldBase(fieldName, Internal::GetRenormalizedTypeName(classp->GetName()),
                                    ROOT::ENTupleStructure::kStreamer, false /* isSimple */),
     fClass(classp),
     fIndex(0)
{
   fTraits |= kTraitTypeChecksum;
   if (!(fClass->ClassProperty() & kClassHasExplicitCtor))
      fTraits |= kTraitTriviallyConstructible;
   if (!(fClass->ClassProperty() & kClassHasExplicitDtor))
      fTraits |= kTraitTriviallyDestructible;
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RStreamerField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RStreamerField>(new RStreamerField(newName, GetTypeName(), GetTypeAlias()));
}

std::size_t ROOT::Experimental::RStreamerField::AppendImpl(const void *from)
{
   TBufferRecStreamer buffer(TBuffer::kWrite, GetValueSize(),
                             [this](TVirtualStreamerInfo *info) { fStreamerInfos[info->GetNumber()] = info; });
   fClass->Streamer(const_cast<void *>(from), buffer);

   auto nbytes = buffer.Length();
   fAuxiliaryColumn->AppendV(buffer.Buffer(), buffer.Length());
   fIndex += nbytes;
   fPrincipalColumn->Append(&fIndex);
   return nbytes + fPrincipalColumn->GetElement()->GetPackedSize();
}

void ROOT::Experimental::RStreamerField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   RNTupleLocalIndex collectionStart;
   ROOT::NTupleSize_t nbytes;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nbytes);

   TBufferFile buffer(TBuffer::kRead, nbytes);
   fAuxiliaryColumn->ReadV(collectionStart, nbytes, buffer.Buffer());
   fClass->Streamer(to, buffer);
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RStreamerField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64, ENTupleColumnType::kByte},
                                                  {ENTupleColumnType::kIndex64, ENTupleColumnType::kByte},
                                                  {ENTupleColumnType::kSplitIndex32, ENTupleColumnType::kByte},
                                                  {ENTupleColumnType::kIndex32, ENTupleColumnType::kByte}},
                                                 {});
   return representations;
}

void ROOT::Experimental::RStreamerField::GenerateColumns()
{
   GenerateColumnsImpl<Internal::RColumnIndex, std::byte>();
}

void ROOT::Experimental::RStreamerField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<Internal::RColumnIndex, std::byte>(desc);
}

void ROOT::Experimental::RStreamerField::ConstructValue(void *where) const
{
   fClass->New(where);
}

void ROOT::Experimental::RStreamerField::RStreamerFieldDeleter::operator()(void *objPtr, bool dtorOnly)
{
   fClass->Destructor(objPtr, true /* dtorOnly */);
   RDeleter::operator()(objPtr, dtorOnly);
}

ROOT::Experimental::RExtraTypeInfoDescriptor ROOT::Experimental::RStreamerField::GetExtraTypeInfo() const
{
   Internal::RExtraTypeInfoDescriptorBuilder extraTypeInfoBuilder;
   extraTypeInfoBuilder.ContentId(EExtraTypeInfoIds::kStreamerInfo)
      .TypeVersion(GetTypeVersion())
      .TypeName(GetTypeName())
      .Content(Internal::RNTupleSerializer::SerializeStreamerInfos(fStreamerInfos));
   return extraTypeInfoBuilder.MoveDescriptor().Unwrap();
}

std::size_t ROOT::Experimental::RStreamerField::GetAlignment() const
{
   return std::min(alignof(std::max_align_t), GetValueSize()); // TODO(jblomer): fix me
}

std::size_t ROOT::Experimental::RStreamerField::GetValueSize() const
{
   return fClass->GetClassSize();
}

std::uint32_t ROOT::Experimental::RStreamerField::GetTypeVersion() const
{
   return fClass->GetClassVersion();
}

std::uint32_t ROOT::Experimental::RStreamerField::GetTypeChecksum() const
{
   return fClass->GetCheckSum();
}

void ROOT::Experimental::RStreamerField::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitStreamerField(*this);
}

//------------------------------------------------------------------------------

std::size_t ROOT::Experimental::RField<TObject>::GetOffsetOfMember(const char *name)
{
   if (auto dataMember = TObject::Class()->GetDataMember(name)) {
      return dataMember->GetOffset();
   }
   throw RException(R__FAIL('\'' + std::string(name) + '\'' + " is an invalid data member"));
}

ROOT::Experimental::RField<TObject>::RField(std::string_view fieldName, const RField<TObject> &source)
   : ROOT::Experimental::RFieldBase(fieldName, "TObject", ROOT::ENTupleStructure::kRecord, false /* isSimple */)
{
   fTraits |= kTraitTypeChecksum;
   Attach(source.GetConstSubfields()[0]->Clone("fUniqueID"));
   Attach(source.GetConstSubfields()[1]->Clone("fBits"));
}

ROOT::Experimental::RField<TObject>::RField(std::string_view fieldName)
   : ROOT::Experimental::RFieldBase(fieldName, "TObject", ROOT::ENTupleStructure::kRecord, false /* isSimple */)
{
   assert(TObject::Class()->GetClassVersion() == 1);

   fTraits |= kTraitTypeChecksum;
   Attach(std::make_unique<RField<UInt_t>>("fUniqueID"));
   Attach(std::make_unique<RField<UInt_t>>("fBits"));
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RField<TObject>::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RField<TObject>>(new RField<TObject>(newName, *this));
}

std::size_t ROOT::Experimental::RField<TObject>::AppendImpl(const void *from)
{
   // Cf. TObject::Streamer()

   auto *obj = static_cast<const TObject *>(from);
   if (obj->TestBit(TObject::kIsReferenced)) {
      throw RException(R__FAIL("RNTuple I/O on referenced TObject is unsupported"));
   }

   std::size_t nbytes = 0;
   nbytes += CallAppendOn(*fSubfields[0], reinterpret_cast<const unsigned char *>(from) + GetOffsetUniqueID());

   UInt_t bits = *reinterpret_cast<const UInt_t *>(reinterpret_cast<const unsigned char *>(from) + GetOffsetBits());
   bits &= (~TObject::kIsOnHeap & ~TObject::kNotDeleted);
   nbytes += CallAppendOn(*fSubfields[1], &bits);

   return nbytes;
}

void ROOT::Experimental::RField<TObject>::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   // Cf. TObject::Streamer()

   auto *obj = static_cast<TObject *>(to);
   if (obj->TestBit(TObject::kIsReferenced)) {
      throw RException(R__FAIL("RNTuple I/O on referenced TObject is unsupported"));
   }

   CallReadOn(*fSubfields[0], globalIndex, static_cast<unsigned char *>(to) + GetOffsetUniqueID());

   const UInt_t bitIsOnHeap = obj->TestBit(TObject::kIsOnHeap) ? TObject::kIsOnHeap : 0;
   UInt_t bits;
   CallReadOn(*fSubfields[1], globalIndex, &bits);
   bits |= bitIsOnHeap | TObject::kNotDeleted;
   *reinterpret_cast<UInt_t *>(reinterpret_cast<unsigned char *>(to) + GetOffsetBits()) = bits;
}

void ROOT::Experimental::RField<TObject>::AfterConnectPageSource()
{
   if (GetOnDiskTypeVersion() != 1) {
      throw RException(R__FAIL("unsupported on-disk version of TObject: " + std::to_string(GetTypeVersion())));
   }
}

std::uint32_t ROOT::Experimental::RField<TObject>::GetTypeVersion() const
{
   return TObject::Class()->GetClassVersion();
}

std::uint32_t ROOT::Experimental::RField<TObject>::GetTypeChecksum() const
{
   return TObject::Class()->GetCheckSum();
}

void ROOT::Experimental::RField<TObject>::ConstructValue(void *where) const
{
   new (where) TObject();
}

std::vector<ROOT::Experimental::RFieldBase::RValue>
ROOT::Experimental::RField<TObject>::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   auto basePtr = value.GetPtr<unsigned char>().get();
   result.emplace_back(
      fSubfields[0]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), basePtr + GetOffsetUniqueID())));
   result.emplace_back(
      fSubfields[1]->BindValue(std::shared_ptr<void>(value.GetPtr<void>(), basePtr + GetOffsetBits())));
   return result;
}

size_t ROOT::Experimental::RField<TObject>::GetValueSize() const
{
   return sizeof(TObject);
}

size_t ROOT::Experimental::RField<TObject>::GetAlignment() const
{
   return alignof(TObject);
}

void ROOT::Experimental::RField<TObject>::AcceptVisitor(Detail::RFieldVisitor &visitor) const
{
   visitor.VisitTObjectField(*this);
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
   result.pop_back(); // remove trailing comma
   return result;
}

ROOT::Experimental::RTupleField::RTupleField(std::string_view fieldName,
                                             std::vector<std::unique_ptr<RFieldBase>> itemFields,
                                             const std::vector<std::size_t> &offsets)
   : ROOT::Experimental::RRecordField(fieldName, "std::tuple<" + GetTypeList(itemFields) + ">")
{
   AttachItemFields(std::move(itemFields));
   fOffsets = offsets;
}

ROOT::Experimental::RTupleField::RTupleField(std::string_view fieldName,
                                             std::vector<std::unique_ptr<RFieldBase>> itemFields)
   : ROOT::Experimental::RRecordField(fieldName, "std::tuple<" + GetTypeList(itemFields) + ">")
{
   AttachItemFields(std::move(itemFields));

   auto *c = TClass::GetClass(GetTypeName().c_str());
   if (!c)
      throw RException(R__FAIL("cannot get type information for " + GetTypeName()));
   fSize = c->Size();

   // ISO C++ does not guarantee neither specific layout nor member names for `std::tuple`.  However, most
   // implementations including libstdc++ (gcc), libc++ (llvm), and MSVC name members as `_0`, `_1`, ..., `_N-1`,
   // following the order of the type list.
   // Use TClass to get their offsets; in case a particular `std::tuple` implementation does not define such
   // members, the assertion below will fail.
   for (unsigned i = 0; i < fSubfields.size(); ++i) {
      std::string memberName("_" + std::to_string(i));
      auto member = c->GetRealData(memberName.c_str());
      if (!member)
         throw RException(R__FAIL(memberName + ": no such member"));
      fOffsets.push_back(member->GetThisOffset());
   }
}

//------------------------------------------------------------------------------

namespace {

// Depending on the compiler, the variant tag is stored either in a trailing char or in a trailing unsigned int
constexpr std::size_t GetVariantTagSize()
{
   // Should be all zeros except for the tag, which is 1
   std::variant<char> t;
   constexpr auto sizeOfT = sizeof(t);

   static_assert(sizeOfT == 2 || sizeOfT == 8, "unsupported std::variant layout");
   return sizeOfT == 2 ? 1 : 4;
}

template <std::size_t VariantSizeT>
struct RVariantTag {
   using ValueType_t = typename std::conditional_t<VariantSizeT == 1, std::uint8_t,
                                                   typename std::conditional_t<VariantSizeT == 4, std::uint32_t, void>>;
};

} // anonymous namespace

std::string ROOT::Experimental::RVariantField::GetTypeList(const std::vector<std::unique_ptr<RFieldBase>> &itemFields)
{
   std::string result;
   for (size_t i = 0; i < itemFields.size(); ++i) {
      result += itemFields[i]->GetTypeName() + ",";
   }
   R__ASSERT(!result.empty()); // there is always at least one variant
   result.pop_back();          // remove trailing comma
   return result;
}

ROOT::Experimental::RVariantField::RVariantField(std::string_view name, const RVariantField &source)
   : ROOT::Experimental::RFieldBase(name, source.GetTypeName(), ROOT::ENTupleStructure::kVariant, false /* isSimple */),
     fMaxItemSize(source.fMaxItemSize),
     fMaxAlignment(source.fMaxAlignment),
     fTagOffset(source.fTagOffset),
     fVariantOffset(source.fVariantOffset),
     fNWritten(source.fNWritten.size(), 0)
{
   for (const auto &f : source.GetConstSubfields())
      Attach(f->Clone(f->GetFieldName()));
   fTraits = source.fTraits;
}

ROOT::Experimental::RVariantField::RVariantField(std::string_view fieldName,
                                                 std::vector<std::unique_ptr<RFieldBase>> itemFields)
   : ROOT::Experimental::RFieldBase(fieldName, "std::variant<" + GetTypeList(itemFields) + ">",
                                    ROOT::ENTupleStructure::kVariant, false /* isSimple */)
{
   // The variant needs to initialize its own tag member
   fTraits |= kTraitTriviallyDestructible & ~kTraitTriviallyConstructible;

   auto nFields = itemFields.size();
   if (nFields == 0 || nFields > kMaxVariants) {
      throw RException(R__FAIL("invalid number of variant fields (outside [1.." + std::to_string(kMaxVariants) + ")"));
   }
   fNWritten.resize(nFields, 0);
   for (unsigned int i = 0; i < nFields; ++i) {
      fMaxItemSize = std::max(fMaxItemSize, itemFields[i]->GetValueSize());
      fMaxAlignment = std::max(fMaxAlignment, itemFields[i]->GetAlignment());
      fTraits &= itemFields[i]->GetTraits();
      Attach(std::move(itemFields[i]));
   }

   // With certain template parameters, the union of members of an std::variant starts at an offset > 0.
   // For instance, std::variant<std::optional<int>> on macOS.
   auto cl = TClass::GetClass(GetTypeName().c_str());
   assert(cl);
   auto dm = reinterpret_cast<TDataMember *>(cl->GetListOfDataMembers()->First());
   if (dm)
      fVariantOffset = dm->GetOffset();

   const auto tagSize = GetVariantTagSize();
   const auto padding = tagSize - (fMaxItemSize % tagSize);
   fTagOffset = fVariantOffset + fMaxItemSize + ((padding == tagSize) ? 0 : padding);
}

std::unique_ptr<ROOT::Experimental::RFieldBase>
ROOT::Experimental::RVariantField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RVariantField>(new RVariantField(newName, *this));
}

std::uint8_t ROOT::Experimental::RVariantField::GetTag(const void *variantPtr, std::size_t tagOffset)
{
   using TagType_t = RVariantTag<GetVariantTagSize()>::ValueType_t;
   auto tag = *reinterpret_cast<const TagType_t *>(reinterpret_cast<const unsigned char *>(variantPtr) + tagOffset);
   return (tag == TagType_t(-1)) ? 0 : tag + 1;
}

void ROOT::Experimental::RVariantField::SetTag(void *variantPtr, std::size_t tagOffset, std::uint8_t tag)
{
   using TagType_t = RVariantTag<GetVariantTagSize()>::ValueType_t;
   auto tagPtr = reinterpret_cast<TagType_t *>(reinterpret_cast<unsigned char *>(variantPtr) + tagOffset);
   *tagPtr = (tag == 0) ? TagType_t(-1) : static_cast<TagType_t>(tag - 1);
}

std::size_t ROOT::Experimental::RVariantField::AppendImpl(const void *from)
{
   auto tag = GetTag(from, fTagOffset);
   std::size_t nbytes = 0;
   auto index = 0;
   if (tag > 0) {
      nbytes += CallAppendOn(*fSubfields[tag - 1], reinterpret_cast<const unsigned char *>(from) + fVariantOffset);
      index = fNWritten[tag - 1]++;
   }
   Internal::RColumnSwitch varSwitch(index, tag);
   fPrincipalColumn->Append(&varSwitch);
   return nbytes + sizeof(Internal::RColumnSwitch);
}

void ROOT::Experimental::RVariantField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   RNTupleLocalIndex variantIndex;
   std::uint32_t tag;
   fPrincipalColumn->GetSwitchInfo(globalIndex, &variantIndex, &tag);
   R__ASSERT(tag < 256);

   // If `tag` equals 0, the variant is in the invalid state, i.e, it does not hold any of the valid alternatives in
   // the type list.  This happens, e.g., if the field was late added; in this case, keep the invalid tag, which makes
   // any `std::holds_alternative<T>` check fail later.
   if (R__likely(tag > 0)) {
      void *varPtr = reinterpret_cast<unsigned char *>(to) + fVariantOffset;
      CallConstructValueOn(*fSubfields[tag - 1], varPtr);
      CallReadOn(*fSubfields[tag - 1], variantIndex, varPtr);
   }
   SetTag(to, fTagOffset, tag);
}

const ROOT::Experimental::RFieldBase::RColumnRepresentations &
ROOT::Experimental::RVariantField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSwitch}}, {});
   return representations;
}

void ROOT::Experimental::RVariantField::GenerateColumns()
{
   GenerateColumnsImpl<Internal::RColumnSwitch>();
}

void ROOT::Experimental::RVariantField::GenerateColumns(const RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<Internal::RColumnSwitch>(desc);
}

void ROOT::Experimental::RVariantField::ConstructValue(void *where) const
{
   memset(where, 0, GetValueSize());
   CallConstructValueOn(*fSubfields[0], reinterpret_cast<unsigned char *>(where) + fVariantOffset);
   SetTag(where, fTagOffset, 1);
}

void ROOT::Experimental::RVariantField::RVariantDeleter::operator()(void *objPtr, bool dtorOnly)
{
   auto tag = GetTag(objPtr, fTagOffset);
   if (tag > 0) {
      fItemDeleters[tag - 1]->operator()(reinterpret_cast<unsigned char *>(objPtr) + fVariantOffset, true /*dtorOnly*/);
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::Experimental::RFieldBase::RDeleter> ROOT::Experimental::RVariantField::GetDeleter() const
{
   std::vector<std::unique_ptr<RDeleter>> itemDeleters;
   itemDeleters.reserve(fSubfields.size());
   for (const auto &f : fSubfields) {
      itemDeleters.emplace_back(GetDeleterOf(*f));
   }
   return std::make_unique<RVariantDeleter>(fTagOffset, fVariantOffset, std::move(itemDeleters));
}

size_t ROOT::Experimental::RVariantField::GetAlignment() const
{
   return std::max(fMaxAlignment, alignof(RVariantTag<GetVariantTagSize()>::ValueType_t));
}

size_t ROOT::Experimental::RVariantField::GetValueSize() const
{
   const auto alignment = GetAlignment();
   const auto actualSize = fTagOffset + GetVariantTagSize();
   const auto padding = alignment - (actualSize % alignment);
   return actualSize + ((padding == alignment) ? 0 : padding);
}

void ROOT::Experimental::RVariantField::CommitClusterImpl()
{
   std::fill(fNWritten.begin(), fNWritten.end(), 0);
}
