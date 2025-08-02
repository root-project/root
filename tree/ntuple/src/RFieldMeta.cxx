/// \file RFieldMeta.cxx
/// \ingroup NTuple
/// \author Jonas Hahnfeld <jonas.hahnfeld@cern.ch>
/// \date 2024-11-19

// This file has concrete RField implementations that depend on ROOT Meta:
//  - RClassField
//  - REnumField
//  - RPairField
//  - RProxiedCollectionField
//    - RMapField
//    - RSetField
//  - RStreamerField
//  - RField<TObject>
//  - RVariantField

#include <ROOT/RField.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RFieldUtils.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RSpan.hxx>

#include <TBaseClass.h>
#include <TBufferFile.h>
#include <TClass.h>
#include <TClassEdit.h>
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

using ROOT::Internal::GetRenormalizedTypeName;

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

ROOT::RClassField::RClassField(std::string_view fieldName, const RClassField &source)
   : ROOT::RFieldBase(fieldName, source.GetTypeName(), ROOT::ENTupleStructure::kRecord, false /* isSimple */),
     fClass(source.fClass),
     fSubfieldsInfo(source.fSubfieldsInfo),
     fMaxAlignment(source.fMaxAlignment)
{
   for (const auto &f : source.GetConstSubfields()) {
      RFieldBase::Attach(f->Clone(f->GetFieldName()));
   }
   fTraits = source.GetTraits();
}

ROOT::RClassField::RClassField(std::string_view fieldName, std::string_view className)
   : RClassField(fieldName, EnsureValidClass(className))
{
}

ROOT::RClassField::RClassField(std::string_view fieldName, TClass *classp)
   : ROOT::RFieldBase(fieldName, GetRenormalizedTypeName(classp->GetName()), ROOT::ENTupleStructure::kRecord,
                      false /* isSimple */),
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
       ROOT::Internal::GetRNTupleSerializationMode(fClass) !=
          ROOT::Internal::ERNTupleSerializationMode::kForceNativeMode) {
      throw RException(R__FAIL(GetTypeName() + " cannot be stored natively in RNTuple"));
   }
   if (ROOT::Internal::GetRNTupleSerializationMode(fClass) ==
       ROOT::Internal::ERNTupleSerializationMode::kForceStreamerMode) {
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

      const auto normTypeName = ROOT::Internal::GetNormalizedUnresolvedTypeName(origTypeName);
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

ROOT::RClassField::~RClassField()
{
   if (fStagingArea) {
      for (const auto &[_, si] : fStagingItems) {
         if (!(si.fField->GetTraits() & kTraitTriviallyDestructible)) {
            auto deleter = si.fField->GetDeleter();
            deleter->operator()(fStagingArea.get() + si.fOffset, true /* dtorOnly */);
         }
      }
   }
}

void ROOT::RClassField::Attach(std::unique_ptr<RFieldBase> child, RSubFieldInfo info)
{
   fMaxAlignment = std::max(fMaxAlignment, child->GetAlignment());
   fSubfieldsInfo.push_back(info);
   RFieldBase::Attach(std::move(child));
}

std::vector<const ROOT::TSchemaRule *> ROOT::RClassField::FindRules(const ROOT::RFieldDescriptor *fieldDesc)
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
      // We need to change (back) the name normalization from RNTuple to ROOT Meta
      std::string normalizedName;
      TClassEdit::GetNormalizedName(normalizedName, fieldDesc->GetTypeName());
      // We do have an on-disk field that correspond to the current RClassField instance. Ask for rules matching the
      // on-disk version of the field.
      if (fieldDesc->GetTypeChecksum()) {
         rules = ruleset->FindRules(normalizedName, fieldDesc->GetTypeVersion(), *fieldDesc->GetTypeChecksum());
      } else {
         rules = ruleset->FindRules(normalizedName, fieldDesc->GetTypeVersion());
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

      ++itr;
   }

   return rules;
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RClassField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RClassField>(new RClassField(newName, *this));
}

std::size_t ROOT::RClassField::AppendImpl(const void *from)
{
   std::size_t nbytes = 0;
   for (unsigned i = 0; i < fSubfields.size(); i++) {
      nbytes += CallAppendOn(*fSubfields[i], static_cast<const unsigned char *>(from) + fSubfieldsInfo[i].fOffset);
   }
   return nbytes;
}

void ROOT::RClassField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   for (const auto &[_, si] : fStagingItems) {
      CallReadOn(*si.fField, globalIndex, fStagingArea.get() + si.fOffset);
   }
   for (unsigned i = 0; i < fSubfields.size(); i++) {
      CallReadOn(*fSubfields[i], globalIndex, static_cast<unsigned char *>(to) + fSubfieldsInfo[i].fOffset);
   }
}

void ROOT::RClassField::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   for (const auto &[_, si] : fStagingItems) {
      CallReadOn(*si.fField, localIndex, fStagingArea.get() + si.fOffset);
   }
   for (unsigned i = 0; i < fSubfields.size(); i++) {
      CallReadOn(*fSubfields[i], localIndex, static_cast<unsigned char *>(to) + fSubfieldsInfo[i].fOffset);
   }
}

ROOT::DescriptorId_t ROOT::RClassField::LookupMember(const ROOT::RNTupleDescriptor &desc, std::string_view memberName,
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

void ROOT::RClassField::SetStagingClass(const std::string &className, unsigned int classVersion)
{
   TClass::GetClass(className.c_str())->GetStreamerInfo(classVersion);
   if (classVersion != GetTypeVersion()) {
      fStagingClass = TClass::GetClass((className + std::string("@@") + std::to_string(classVersion)).c_str());
      if (!fStagingClass) {
         // For a rename rule, we may simply ask for the old class name
         fStagingClass = TClass::GetClass(className.c_str());
      }
   } else {
      fStagingClass = fClass;
   }
   R__ASSERT(fStagingClass);
   R__ASSERT(static_cast<unsigned int>(fStagingClass->GetClassVersion()) == classVersion);
}

void ROOT::RClassField::PrepareStagingArea(const std::vector<const TSchemaRule *> &rules,
                                           const ROOT::RNTupleDescriptor &desc,
                                           const ROOT::RFieldDescriptor &classFieldDesc)
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
         // Since we successfully looked up the source member in the RNTuple on-disk metadata, we expect it
         // to be present in the TClass instance, too.
         R__ASSERT(stagingItem.fOffset != TVirtualStreamerInfo::kMissing);
         stagingAreaSize = std::max(stagingAreaSize, stagingItem.fOffset + stagingItem.fField->GetValueSize());
      }
   }

   if (stagingAreaSize) {
      R__ASSERT(static_cast<Int_t>(stagingAreaSize) <= fStagingClass->Size()); // we may have removed rules
      // We use std::make_unique instead of MakeUninitArray to zero-initialize the staging area.
      fStagingArea = std::make_unique<unsigned char[]>(stagingAreaSize);

      for (const auto &[_, si] : fStagingItems) {
         if (!(si.fField->GetTraits() & kTraitTriviallyConstructible)) {
            CallConstructValueOn(*si.fField, fStagingArea.get() + si.fOffset);
         }
      }
   }
}

void ROOT::RClassField::AddReadCallbacksFromIORule(const TSchemaRule *rule)
{
   auto func = rule->GetReadFunctionPointer();
   if (func == nullptr) {
      // Can happen for rename rules
      return;
   }
   fReadCallbacks.emplace_back([func, stagingClass = fStagingClass, stagingArea = fStagingArea.get()](void *target) {
      TVirtualObject onfileObj{nullptr};
      onfileObj.fClass = stagingClass;
      onfileObj.fObject = stagingArea;
      func(static_cast<char *>(target), &onfileObj);
      onfileObj.fObject = nullptr; // TVirtualObject does not own the value
   });
}

void ROOT::RClassField::BeforeConnectPageSource(ROOT::Internal::RPageSource &pageSource)
{
   std::vector<const TSchemaRule *> rules;
   // On-disk members that are not targeted by an I/O rule; all other sub fields of the in-memory class
   // will be marked as artificial (added member in a new class version or member set by rule).
   std::unordered_set<std::string> regularSubfields;

   if (GetOnDiskId() == kInvalidDescriptorId) {
      // This can happen for added base classes or added members of class type
      rules = FindRules(nullptr);
      if (!rules.empty())
         SetStagingClass(GetTypeName(), GetTypeVersion());
   } else {
      const auto descriptorGuard = pageSource.GetSharedDescriptorGuard();
      const ROOT::RNTupleDescriptor &desc = descriptorGuard.GetRef();
      const auto &fieldDesc = desc.GetFieldDescriptor(GetOnDiskId());

      for (auto linkId : fieldDesc.GetLinkIds()) {
         const auto &subFieldDesc = desc.GetFieldDescriptor(linkId);
         regularSubfields.insert(subFieldDesc.GetFieldName());
      }

      rules = FindRules(&fieldDesc);

      // If the field's type name is not the on-disk name but we found a rule, we know it is valid to read
      // on-disk data because we found the rule according to the on-disk (source) type name and version/checksum.
      if ((GetTypeName() != fieldDesc.GetTypeName()) && rules.empty()) {
         throw RException(R__FAIL("incompatible type name for field " + GetFieldName() + ": " + GetTypeName() +
                                  " vs. " + fieldDesc.GetTypeName()));
      }

      if (!rules.empty()) {
         SetStagingClass(fieldDesc.GetTypeName(), fieldDesc.GetTypeVersion());
         PrepareStagingArea(rules, desc, fieldDesc);
         for (auto &[_, si] : fStagingItems)
            Internal::CallConnectPageSourceOnField(*si.fField, pageSource);

         // Remove target member of read rules from the list of regular members of the underlying on-disk field
         for (const auto rule : rules) {
            if (!rule->GetTarget())
               continue;

            for (const auto target : ROOT::Detail::TRangeStaticCast<const TObjString>(*rule->GetTarget())) {
               regularSubfields.erase(std::string(target->GetString()));
            }
         }
      }
   }

   for (const auto rule : rules) {
      AddReadCallbacksFromIORule(rule);
   }

   // Iterate over all sub fields in memory and mark those as missing that are not in the descriptor.
   for (auto &field : fSubfields) {
      if (regularSubfields.count(field->GetFieldName()) == 0) {
         field->SetArtificial();
      }
   }
}

void ROOT::RClassField::ConstructValue(void *where) const
{
   fClass->New(where);
}

void ROOT::RClassField::RClassDeleter::operator()(void *objPtr, bool dtorOnly)
{
   fClass->Destructor(objPtr, true /* dtorOnly */);
   RDeleter::operator()(objPtr, dtorOnly);
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RClassField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   auto valuePtr = value.GetPtr<void>();
   auto charPtr = static_cast<unsigned char *>(valuePtr.get());
   result.reserve(fSubfields.size());
   for (unsigned i = 0; i < fSubfields.size(); i++) {
      result.emplace_back(
         fSubfields[i]->BindValue(std::shared_ptr<void>(valuePtr, charPtr + fSubfieldsInfo[i].fOffset)));
   }
   return result;
}

size_t ROOT::RClassField::GetValueSize() const
{
   return fClass->GetClassSize();
}

std::uint32_t ROOT::RClassField::GetTypeVersion() const
{
   return fClass->GetClassVersion();
}

std::uint32_t ROOT::RClassField::GetTypeChecksum() const
{
   return fClass->GetCheckSum();
}

void ROOT::RClassField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitClassField(*this);
}

//------------------------------------------------------------------------------

ROOT::REnumField::REnumField(std::string_view fieldName, std::string_view enumName)
   : REnumField(fieldName, EnsureValidEnum(enumName))
{
}

ROOT::REnumField::REnumField(std::string_view fieldName, TEnum *enump)
   : ROOT::RFieldBase(fieldName, GetRenormalizedTypeName(enump->GetQualifiedName()), ROOT::ENTupleStructure::kLeaf,
                      false /* isSimple */)
{
   // Avoid accidentally supporting std types through TEnum.
   if (enump->Property() & kIsDefinedInStd) {
      throw RException(R__FAIL(GetTypeName() + " is not supported"));
   }

   switch (enump->GetUnderlyingType()) {
   case kChar_t: Attach(std::make_unique<RField<Char_t>>("_0")); break;
   case kUChar_t: Attach(std::make_unique<RField<UChar_t>>("_0")); break;
   case kShort_t: Attach(std::make_unique<RField<Short_t>>("_0")); break;
   case kUShort_t: Attach(std::make_unique<RField<UShort_t>>("_0")); break;
   case kInt_t: Attach(std::make_unique<RField<Int_t>>("_0")); break;
   case kUInt_t: Attach(std::make_unique<RField<UInt_t>>("_0")); break;
   case kLong_t: Attach(std::make_unique<RField<Long_t>>("_0")); break;
   case kLong64_t: Attach(std::make_unique<RField<Long64_t>>("_0")); break;
   case kULong_t: Attach(std::make_unique<RField<ULong_t>>("_0")); break;
   case kULong64_t: Attach(std::make_unique<RField<ULong64_t>>("_0")); break;
   default: throw RException(R__FAIL("Unsupported underlying integral type for enum type " + GetTypeName()));
   }

   fTraits |= kTraitTriviallyConstructible | kTraitTriviallyDestructible;
}

ROOT::REnumField::REnumField(std::string_view fieldName, std::string_view enumName,
                             std::unique_ptr<RFieldBase> intField)
   : ROOT::RFieldBase(fieldName, enumName, ROOT::ENTupleStructure::kLeaf, false /* isSimple */)
{
   Attach(std::move(intField));
   fTraits |= kTraitTriviallyConstructible | kTraitTriviallyDestructible;
}

std::unique_ptr<ROOT::RFieldBase> ROOT::REnumField::CloneImpl(std::string_view newName) const
{
   auto newIntField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::unique_ptr<REnumField>(new REnumField(newName, GetTypeName(), std::move(newIntField)));
}

std::vector<ROOT::RFieldBase::RValue> ROOT::REnumField::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   result.emplace_back(fSubfields[0]->BindValue(value.GetPtr<void>()));
   return result;
}

void ROOT::REnumField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitEnumField(*this);
}

//------------------------------------------------------------------------------

std::string ROOT::RPairField::RPairField::GetTypeList(const std::array<std::unique_ptr<RFieldBase>, 2> &itemFields)
{
   return itemFields[0]->GetTypeName() + "," + itemFields[1]->GetTypeName();
}

ROOT::RPairField::RPairField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, 2> itemFields,
                             const std::array<std::size_t, 2> &offsets)
   : ROOT::RRecordField(fieldName, "std::pair<" + GetTypeList(itemFields) + ">")
{
   AttachItemFields(std::move(itemFields));
   fOffsets.push_back(offsets[0]);
   fOffsets.push_back(offsets[1]);
}

ROOT::RPairField::RPairField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, 2> itemFields)
   : ROOT::RRecordField(fieldName, "std::pair<" + GetTypeList(itemFields) + ">")
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

ROOT::RProxiedCollectionField::RCollectionIterableOnce::RIteratorFuncs
ROOT::RProxiedCollectionField::RCollectionIterableOnce::GetIteratorFuncs(TVirtualCollectionProxy *proxy,
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

ROOT::RProxiedCollectionField::RProxiedCollectionField(std::string_view fieldName, TClass *classp)
   : RFieldBase(fieldName, GetRenormalizedTypeName(classp->GetName()), ROOT::ENTupleStructure::kCollection,
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

   fIFuncsRead = RCollectionIterableOnce::GetIteratorFuncs(fProxy.get(), true /* readFromDisk */);
   fIFuncsWrite = RCollectionIterableOnce::GetIteratorFuncs(fProxy.get(), false /* readFromDisk */);
}

ROOT::RProxiedCollectionField::RProxiedCollectionField(std::string_view fieldName, std::string_view typeName,
                                                       std::unique_ptr<RFieldBase> itemField)
   : RProxiedCollectionField(fieldName, EnsureValidClass(typeName))
{
   fItemSize = itemField->GetValueSize();
   Attach(std::move(itemField));
}

ROOT::RProxiedCollectionField::RProxiedCollectionField(std::string_view fieldName, std::string_view typeName)
   : RProxiedCollectionField(fieldName, EnsureValidClass(typeName))
{
   // NOTE (fdegeus): std::map is supported, custom associative might be supported in the future if the need arises.
   if (fProperties & TVirtualCollectionProxy::kIsAssociative)
      throw RException(R__FAIL("custom associative collection proxies not supported"));

   std::unique_ptr<ROOT::RFieldBase> itemField;

   if (auto valueClass = fProxy->GetValueClass()) {
      // Element type is a class
      itemField = RFieldBase::Create("_0", valueClass->GetName()).Unwrap();
   } else {
      switch (fProxy->GetType()) {
      case EDataType::kChar_t: itemField = std::make_unique<RField<Char_t>>("_0"); break;
      case EDataType::kUChar_t: itemField = std::make_unique<RField<UChar_t>>("_0"); break;
      case EDataType::kShort_t: itemField = std::make_unique<RField<Short_t>>("_0"); break;
      case EDataType::kUShort_t: itemField = std::make_unique<RField<UShort_t>>("_0"); break;
      case EDataType::kInt_t: itemField = std::make_unique<RField<Int_t>>("_0"); break;
      case EDataType::kUInt_t: itemField = std::make_unique<RField<UInt_t>>("_0"); break;
      case EDataType::kLong_t: itemField = std::make_unique<RField<Long_t>>("_0"); break;
      case EDataType::kLong64_t: itemField = std::make_unique<RField<Long64_t>>("_0"); break;
      case EDataType::kULong_t: itemField = std::make_unique<RField<ULong_t>>("_0"); break;
      case EDataType::kULong64_t: itemField = std::make_unique<RField<ULong64_t>>("_0"); break;
      case EDataType::kFloat_t: itemField = std::make_unique<RField<Float_t>>("_0"); break;
      case EDataType::kDouble_t: itemField = std::make_unique<RField<Double_t>>("_0"); break;
      case EDataType::kBool_t: itemField = std::make_unique<RField<Bool_t>>("_0"); break;
      default: throw RException(R__FAIL("unsupported value type"));
      }
   }

   fItemSize = itemField->GetValueSize();
   Attach(std::move(itemField));
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RProxiedCollectionField::CloneImpl(std::string_view newName) const
{
   auto newItemField = fSubfields[0]->Clone(fSubfields[0]->GetFieldName());
   return std::unique_ptr<RProxiedCollectionField>(
      new RProxiedCollectionField(newName, GetTypeName(), std::move(newItemField)));
}

std::size_t ROOT::RProxiedCollectionField::AppendImpl(const void *from)
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

void ROOT::RProxiedCollectionField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
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

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RProxiedCollectionField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64},
                                                  {ENTupleColumnType::kIndex64},
                                                  {ENTupleColumnType::kSplitIndex32},
                                                  {ENTupleColumnType::kIndex32}},
                                                 {});
   return representations;
}

void ROOT::RProxiedCollectionField::GenerateColumns()
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>();
}

void ROOT::RProxiedCollectionField::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex>(desc);
}

void ROOT::RProxiedCollectionField::ConstructValue(void *where) const
{
   fProxy->New(where);
}

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::RProxiedCollectionField::GetDeleter() const
{
   if (fProperties & TVirtualCollectionProxy::kNeedDelete) {
      std::size_t itemSize = fCollectionType == kSTLvector ? fItemSize : 0U;
      return std::make_unique<RProxiedCollectionDeleter>(fProxy, GetDeleterOf(*fSubfields[0]), itemSize);
   }
   return std::make_unique<RProxiedCollectionDeleter>(fProxy);
}

void ROOT::RProxiedCollectionField::RProxiedCollectionDeleter::operator()(void *objPtr, bool dtorOnly)
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

std::vector<ROOT::RFieldBase::RValue> ROOT::RProxiedCollectionField::SplitValue(const RValue &value) const
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

void ROOT::RProxiedCollectionField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitProxiedCollectionField(*this);
}

//------------------------------------------------------------------------------

ROOT::RMapField::RMapField(std::string_view fieldName, std::string_view typeName, std::unique_ptr<RFieldBase> itemField)
   : RProxiedCollectionField(fieldName, EnsureValidClass(typeName))
{
   if (!dynamic_cast<RPairField *>(itemField.get()))
      throw RException(R__FAIL("RMapField inner field type must be of RPairField"));

   auto *itemClass = fProxy->GetValueClass();
   fItemSize = itemClass->GetClassSize();

   Attach(std::move(itemField));
}

//------------------------------------------------------------------------------

ROOT::RSetField::RSetField(std::string_view fieldName, std::string_view typeName, std::unique_ptr<RFieldBase> itemField)
   : ROOT::RProxiedCollectionField(fieldName, typeName, std::move(itemField))
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

ROOT::RStreamerField::RStreamerField(std::string_view fieldName, std::string_view className, std::string_view typeAlias)
   : RStreamerField(fieldName, EnsureValidClass(className))
{
   fTypeAlias = typeAlias;
}

ROOT::RStreamerField::RStreamerField(std::string_view fieldName, TClass *classp)
   : ROOT::RFieldBase(fieldName, GetRenormalizedTypeName(classp->GetName()), ROOT::ENTupleStructure::kStreamer,
                      false /* isSimple */),
     fClass(classp),
     fIndex(0)
{
   fTraits |= kTraitTypeChecksum;
   // For RClassField, we only check for explicit constructors and destructors and then recursively combine traits from
   // all member subfields. For RStreamerField, we treat the class as a black box and additionally need to check for
   // implicit constructors and destructors.
   if (!(fClass->ClassProperty() & (kClassHasExplicitCtor | kClassHasImplicitCtor)))
      fTraits |= kTraitTriviallyConstructible;
   if (!(fClass->ClassProperty() & (kClassHasExplicitDtor | kClassHasImplicitDtor)))
      fTraits |= kTraitTriviallyDestructible;
}

void ROOT::RStreamerField::BeforeConnectPageSource(ROOT::Internal::RPageSource &pageSource)
{
   pageSource.RegisterStreamerInfos();
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RStreamerField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RStreamerField>(new RStreamerField(newName, GetTypeName(), GetTypeAlias()));
}

std::size_t ROOT::RStreamerField::AppendImpl(const void *from)
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

void ROOT::RStreamerField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   RNTupleLocalIndex collectionStart;
   ROOT::NTupleSize_t nbytes;
   fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nbytes);

   TBufferFile buffer(TBuffer::kRead, nbytes);
   fAuxiliaryColumn->ReadV(collectionStart, nbytes, buffer.Buffer());
   fClass->Streamer(to, buffer);
}

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RStreamerField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSplitIndex64, ENTupleColumnType::kByte},
                                                  {ENTupleColumnType::kIndex64, ENTupleColumnType::kByte},
                                                  {ENTupleColumnType::kSplitIndex32, ENTupleColumnType::kByte},
                                                  {ENTupleColumnType::kIndex32, ENTupleColumnType::kByte}},
                                                 {});
   return representations;
}

void ROOT::RStreamerField::GenerateColumns()
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex, std::byte>();
}

void ROOT::RStreamerField::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ROOT::Internal::RColumnIndex, std::byte>(desc);
}

void ROOT::RStreamerField::ConstructValue(void *where) const
{
   fClass->New(where);
}

void ROOT::RStreamerField::RStreamerFieldDeleter::operator()(void *objPtr, bool dtorOnly)
{
   fClass->Destructor(objPtr, true /* dtorOnly */);
   RDeleter::operator()(objPtr, dtorOnly);
}

ROOT::RExtraTypeInfoDescriptor ROOT::RStreamerField::GetExtraTypeInfo() const
{
   ROOT::Internal::RExtraTypeInfoDescriptorBuilder extraTypeInfoBuilder;
   extraTypeInfoBuilder.ContentId(ROOT::EExtraTypeInfoIds::kStreamerInfo)
      .TypeVersion(GetTypeVersion())
      .TypeName(GetTypeName())
      .Content(ROOT::Internal::RNTupleSerializer::SerializeStreamerInfos(fStreamerInfos));
   return extraTypeInfoBuilder.MoveDescriptor().Unwrap();
}

std::size_t ROOT::RStreamerField::GetAlignment() const
{
   return std::min(alignof(std::max_align_t), GetValueSize()); // TODO(jblomer): fix me
}

std::size_t ROOT::RStreamerField::GetValueSize() const
{
   return fClass->GetClassSize();
}

std::uint32_t ROOT::RStreamerField::GetTypeVersion() const
{
   return fClass->GetClassVersion();
}

std::uint32_t ROOT::RStreamerField::GetTypeChecksum() const
{
   return fClass->GetCheckSum();
}

void ROOT::RStreamerField::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitStreamerField(*this);
}

//------------------------------------------------------------------------------

std::size_t ROOT::RField<TObject>::GetOffsetOfMember(const char *name)
{
   if (auto dataMember = TObject::Class()->GetDataMember(name)) {
      return dataMember->GetOffset();
   }
   throw RException(R__FAIL('\'' + std::string(name) + '\'' + " is an invalid data member"));
}

ROOT::RField<TObject>::RField(std::string_view fieldName, const RField<TObject> &source)
   : ROOT::RFieldBase(fieldName, "TObject", ROOT::ENTupleStructure::kRecord, false /* isSimple */)
{
   fTraits |= kTraitTypeChecksum;
   Attach(source.GetConstSubfields()[0]->Clone("fUniqueID"));
   Attach(source.GetConstSubfields()[1]->Clone("fBits"));
}

ROOT::RField<TObject>::RField(std::string_view fieldName)
   : ROOT::RFieldBase(fieldName, "TObject", ROOT::ENTupleStructure::kRecord, false /* isSimple */)
{
   assert(TObject::Class()->GetClassVersion() == 1);

   fTraits |= kTraitTypeChecksum;
   Attach(std::make_unique<RField<UInt_t>>("fUniqueID"));
   Attach(std::make_unique<RField<UInt_t>>("fBits"));
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RField<TObject>::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RField<TObject>>(new RField<TObject>(newName, *this));
}

std::size_t ROOT::RField<TObject>::AppendImpl(const void *from)
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

void ROOT::RField<TObject>::ReadTObject(void *to, UInt_t uniqueID, UInt_t bits)
{
   // Cf. TObject::Streamer()

   auto *obj = static_cast<TObject *>(to);
   if (obj->TestBit(TObject::kIsReferenced)) {
      throw RException(R__FAIL("RNTuple I/O on referenced TObject is unsupported"));
   }

   *reinterpret_cast<UInt_t *>(reinterpret_cast<unsigned char *>(to) + GetOffsetUniqueID()) = uniqueID;

   const UInt_t bitIsOnHeap = obj->TestBit(TObject::kIsOnHeap) ? TObject::kIsOnHeap : 0;
   bits |= bitIsOnHeap | TObject::kNotDeleted;
   *reinterpret_cast<UInt_t *>(reinterpret_cast<unsigned char *>(to) + GetOffsetBits()) = bits;
}

void ROOT::RField<TObject>::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
{
   UInt_t uniqueID, bits;
   CallReadOn(*fSubfields[0], globalIndex, &uniqueID);
   CallReadOn(*fSubfields[1], globalIndex, &bits);
   ReadTObject(to, uniqueID, bits);
}

void ROOT::RField<TObject>::ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to)
{
   UInt_t uniqueID, bits;
   CallReadOn(*fSubfields[0], localIndex, &uniqueID);
   CallReadOn(*fSubfields[1], localIndex, &bits);
   ReadTObject(to, uniqueID, bits);
}

void ROOT::RField<TObject>::AfterConnectPageSource()
{
   if (GetOnDiskTypeVersion() != 1) {
      throw RException(R__FAIL("unsupported on-disk version of TObject: " + std::to_string(GetTypeVersion())));
   }
}

std::uint32_t ROOT::RField<TObject>::GetTypeVersion() const
{
   return TObject::Class()->GetClassVersion();
}

std::uint32_t ROOT::RField<TObject>::GetTypeChecksum() const
{
   return TObject::Class()->GetCheckSum();
}

void ROOT::RField<TObject>::ConstructValue(void *where) const
{
   new (where) TObject();
}

std::vector<ROOT::RFieldBase::RValue> ROOT::RField<TObject>::SplitValue(const RValue &value) const
{
   std::vector<RValue> result;
   // Use GetPtr<TObject> to type-check
   std::shared_ptr<void> ptr = value.GetPtr<TObject>();
   auto charPtr = static_cast<unsigned char *>(ptr.get());
   result.emplace_back(fSubfields[0]->BindValue(std::shared_ptr<void>(ptr, charPtr + GetOffsetUniqueID())));
   result.emplace_back(fSubfields[1]->BindValue(std::shared_ptr<void>(ptr, charPtr + GetOffsetBits())));
   return result;
}

size_t ROOT::RField<TObject>::GetValueSize() const
{
   return sizeof(TObject);
}

size_t ROOT::RField<TObject>::GetAlignment() const
{
   return alignof(TObject);
}

void ROOT::RField<TObject>::AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const
{
   visitor.VisitTObjectField(*this);
}

//------------------------------------------------------------------------------

std::string ROOT::RTupleField::RTupleField::GetTypeList(const std::vector<std::unique_ptr<RFieldBase>> &itemFields)
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

ROOT::RTupleField::RTupleField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields,
                               const std::vector<std::size_t> &offsets)
   : ROOT::RRecordField(fieldName, "std::tuple<" + GetTypeList(itemFields) + ">")
{
   AttachItemFields(std::move(itemFields));
   fOffsets = offsets;
}

ROOT::RTupleField::RTupleField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields)
   : ROOT::RRecordField(fieldName, "std::tuple<" + GetTypeList(itemFields) + ">")
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

std::string ROOT::RVariantField::GetTypeList(const std::vector<std::unique_ptr<RFieldBase>> &itemFields)
{
   std::string result;
   for (size_t i = 0; i < itemFields.size(); ++i) {
      result += itemFields[i]->GetTypeName() + ",";
   }
   R__ASSERT(!result.empty()); // there is always at least one variant
   result.pop_back();          // remove trailing comma
   return result;
}

ROOT::RVariantField::RVariantField(std::string_view name, const RVariantField &source)
   : ROOT::RFieldBase(name, source.GetTypeName(), ROOT::ENTupleStructure::kVariant, false /* isSimple */),
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

ROOT::RVariantField::RVariantField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields)
   : ROOT::RFieldBase(fieldName, "std::variant<" + GetTypeList(itemFields) + ">", ROOT::ENTupleStructure::kVariant,
                      false /* isSimple */)
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

std::unique_ptr<ROOT::RFieldBase> ROOT::RVariantField::CloneImpl(std::string_view newName) const
{
   return std::unique_ptr<RVariantField>(new RVariantField(newName, *this));
}

std::uint8_t ROOT::RVariantField::GetTag(const void *variantPtr, std::size_t tagOffset)
{
   using TagType_t = RVariantTag<GetVariantTagSize()>::ValueType_t;
   auto tag = *reinterpret_cast<const TagType_t *>(reinterpret_cast<const unsigned char *>(variantPtr) + tagOffset);
   return (tag == TagType_t(-1)) ? 0 : tag + 1;
}

void ROOT::RVariantField::SetTag(void *variantPtr, std::size_t tagOffset, std::uint8_t tag)
{
   using TagType_t = RVariantTag<GetVariantTagSize()>::ValueType_t;
   auto tagPtr = reinterpret_cast<TagType_t *>(reinterpret_cast<unsigned char *>(variantPtr) + tagOffset);
   *tagPtr = (tag == 0) ? TagType_t(-1) : static_cast<TagType_t>(tag - 1);
}

std::size_t ROOT::RVariantField::AppendImpl(const void *from)
{
   auto tag = GetTag(from, fTagOffset);
   std::size_t nbytes = 0;
   auto index = 0;
   if (tag > 0) {
      nbytes += CallAppendOn(*fSubfields[tag - 1], reinterpret_cast<const unsigned char *>(from) + fVariantOffset);
      index = fNWritten[tag - 1]++;
   }
   ROOT::Internal::RColumnSwitch varSwitch(index, tag);
   fPrincipalColumn->Append(&varSwitch);
   return nbytes + sizeof(ROOT::Internal::RColumnSwitch);
}

void ROOT::RVariantField::ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to)
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

const ROOT::RFieldBase::RColumnRepresentations &ROOT::RVariantField::GetColumnRepresentations() const
{
   static RColumnRepresentations representations({{ENTupleColumnType::kSwitch}}, {});
   return representations;
}

void ROOT::RVariantField::GenerateColumns()
{
   GenerateColumnsImpl<ROOT::Internal::RColumnSwitch>();
}

void ROOT::RVariantField::GenerateColumns(const ROOT::RNTupleDescriptor &desc)
{
   GenerateColumnsImpl<ROOT::Internal::RColumnSwitch>(desc);
}

void ROOT::RVariantField::ConstructValue(void *where) const
{
   memset(where, 0, GetValueSize());
   CallConstructValueOn(*fSubfields[0], reinterpret_cast<unsigned char *>(where) + fVariantOffset);
   SetTag(where, fTagOffset, 1);
}

void ROOT::RVariantField::RVariantDeleter::operator()(void *objPtr, bool dtorOnly)
{
   auto tag = GetTag(objPtr, fTagOffset);
   if (tag > 0) {
      fItemDeleters[tag - 1]->operator()(reinterpret_cast<unsigned char *>(objPtr) + fVariantOffset, true /*dtorOnly*/);
   }
   RDeleter::operator()(objPtr, dtorOnly);
}

std::unique_ptr<ROOT::RFieldBase::RDeleter> ROOT::RVariantField::GetDeleter() const
{
   std::vector<std::unique_ptr<RDeleter>> itemDeleters;
   itemDeleters.reserve(fSubfields.size());
   for (const auto &f : fSubfields) {
      itemDeleters.emplace_back(GetDeleterOf(*f));
   }
   return std::make_unique<RVariantDeleter>(fTagOffset, fVariantOffset, std::move(itemDeleters));
}

size_t ROOT::RVariantField::GetAlignment() const
{
   return std::max(fMaxAlignment, alignof(RVariantTag<GetVariantTagSize()>::ValueType_t));
}

size_t ROOT::RVariantField::GetValueSize() const
{
   const auto alignment = GetAlignment();
   const auto actualSize = fTagOffset + GetVariantTagSize();
   const auto padding = alignment - (actualSize % alignment);
   return actualSize + ((padding == alignment) ? 0 : padding);
}

void ROOT::RVariantField::CommitClusterImpl()
{
   std::fill(fNWritten.begin(), fNWritten.end(), 0);
}
