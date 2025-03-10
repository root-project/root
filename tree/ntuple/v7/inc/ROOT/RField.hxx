/// \file ROOT/RField.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RField
#define ROOT7_RField

#include <ROOT/RError.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RSpan.hxx>
#include <string_view>
#include <ROOT/TypeTraits.hxx>

#include <TGenericClassInfo.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

class TClass;
class TEnum;
class TObject;
class TVirtualStreamerInfo;

namespace ROOT {

class TSchemaRule;

namespace Experimental {

class REntry;

namespace Detail {
class RFieldVisitor;
} // namespace Detail

/// The container field for an ntuple model, which itself has no physical representation.
/// Therefore, the zero field must not be connected to a page source or sink.
class RFieldZero final : public RFieldBase {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;
   void ConstructValue(void *) const final {}

public:
   RFieldZero() : RFieldBase("", "", ROOT::ENTupleStructure::kRecord, false /* isSimple */) {}

   using RFieldBase::Attach;
   size_t GetValueSize() const final { return 0; }
   size_t GetAlignment() const final { return 0; }

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// Used in RFieldBase::Check() to record field creation failures.
/// Also used when deserializing a field that contains unknown values that may come from
/// future RNTuple versions (e.g. an unknown Structure)
class RInvalidField final : public RFieldBase {
public:
   enum class RCategory {
      /// Generic unrecoverable error
      kGeneric,
      /// The type given to RFieldBase::Create was invalid
      kTypeError,
      /// The type given to RFieldBase::Create was unknown
      kUnknownType,
      /// The field could not be created because its descriptor had an unknown structural role
      kUnknownStructure
   };

private:
   std::string fError;
   RCategory fCategory;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RInvalidField>(newName, GetTypeName(), fError, fCategory);
   }
   void ConstructValue(void *) const final {}

public:
   RInvalidField(std::string_view name, std::string_view type, std::string_view error, RCategory category)
      : RFieldBase(name, type, ROOT::ENTupleStructure::kLeaf, false /* isSimple */), fError(error), fCategory(category)
   {
      fTraits |= kTraitInvalidField;
   }

   const std::string &GetError() const { return fError; }
   RCategory GetCategory() const { return fCategory; }

   size_t GetValueSize() const final { return 0; }
   size_t GetAlignment() const final { return 0; }
}; // RInvalidField

/// The field for a class with dictionary
class RClassField : public RFieldBase {
private:
   enum ESubFieldRole {
      kBaseClass,
      kDataMember,
   };
   struct RSubFieldInfo {
      ESubFieldRole fRole;
      std::size_t fOffset;
   };
   // Information to read into the staging area a field that is used as an input to an I/O customization rule
   struct RStagingItem {
      /// The field used to read the on-disk data. The fields type may be different from the on-disk type as long
      /// as the on-disk type can be converted to the fields type (through type cast / schema evolution).
      std::unique_ptr<RFieldBase> fField;
      std::size_t fOffset; ///< offset in fStagingArea
   };
   /// Prefix used in the subfield names generated for base classes
   static constexpr const char *kPrefixInherited{":"};

   class RClassDeleter : public RDeleter {
   private:
      TClass *fClass;

   public:
      explicit RClassDeleter(TClass *cl) : fClass(cl) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   TClass *fClass;
   /// Additional information kept for each entry in `fSubfields`
   std::vector<RSubFieldInfo> fSubfieldsInfo;
   std::size_t fMaxAlignment = 1;

   /// The staging area stores inputs to I/O rules according to the offsets given by the streamer info of
   /// "TypeName@@Version". The area is allocated depending on I/O rules resp. the source members of the I/O rules.
   std::unique_ptr<unsigned char[]> fStagingArea;
   /// The TClass instance that corresponds to the staging area.
   /// The staging class exists as <class name>@@<on-disk version> if the on-disk version is different from the
   /// current in-memory version, or it can be accessed by the first @@alloc streamer element of the current streamer
   /// info.
   TClass *fStagingClass = nullptr;
   std::unordered_map<std::string, RStagingItem> fStagingItems; ///< Lookup staging items by member name

private:
   RClassField(std::string_view fieldName, const RClassField &source); ///< Used by CloneImpl
   RClassField(std::string_view fieldName, TClass *classp);
   void Attach(std::unique_ptr<RFieldBase> child, RSubFieldInfo info);

   /// Returns the id of member 'name' in the class field given by 'fieldId', or kInvalidDescriptorId if no such
   /// member exist. Looks recursively in base classes.
   ROOT::DescriptorId_t
   LookupMember(const RNTupleDescriptor &desc, std::string_view memberName, ROOT::DescriptorId_t classFieldId);
   /// Sets fStagingClass according to the given name and version
   void SetStagingClass(const std::string &className, unsigned int classVersion);
   /// If there are rules with inputs (source members), create the staging area according to the TClass instance
   /// that corresponds to the on-disk field.
   void PrepareStagingArea(const std::vector<const TSchemaRule *> &rules, const RNTupleDescriptor &desc,
                           const RFieldDescriptor &classFieldId);
   /// Register post-read callback corresponding to a ROOT I/O customization rules.
   void AddReadCallbacksFromIORule(const TSchemaRule *rule);
   /// Given the on-disk information from the page source, find all the I/O customization rules that apply
   /// to the class field at hand, to which the fieldDesc descriptor, if provided, must correspond.
   /// Fields may not have an on-disk representation (e.g., when inserted by schema evolution), in which case the passed
   /// field descriptor is nullptr.
   std::vector<const TSchemaRule *> FindRules(const RFieldDescriptor *fieldDesc);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RClassDeleter>(fClass); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;
   void ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to) final;
   void BeforeConnectPageSource(Internal::RPageSource &pageSource) final;

public:
   RClassField(std::string_view fieldName, std::string_view className);
   RClassField(RClassField &&other) = default;
   RClassField &operator=(RClassField &&other) = default;
   ~RClassField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final;
   size_t GetAlignment() const final { return fMaxAlignment; }
   std::uint32_t GetTypeVersion() const final;
   std::uint32_t GetTypeChecksum() const final;
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// The field for a class using ROOT standard streaming
class RStreamerField final : public RFieldBase {
private:
   class RStreamerFieldDeleter : public RDeleter {
   private:
      TClass *fClass;

   public:
      explicit RStreamerFieldDeleter(TClass *cl) : fClass(cl) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   TClass *fClass = nullptr;
   Internal::RNTupleSerializer::StreamerInfoMap_t fStreamerInfos; ///< streamer info records seen during writing
   Internal::RColumnIndex fIndex;                                 ///< number of bytes written in the current cluster

private:
   RStreamerField(std::string_view fieldName, TClass *classp);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &) final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RStreamerFieldDeleter>(fClass); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;

   void CommitClusterImpl() final { fIndex = 0; }

   bool HasExtraTypeInfo() const final { return true; }
   // Returns the list of seen streamer infos
   RExtraTypeInfoDescriptor GetExtraTypeInfo() const final;

public:
   RStreamerField(std::string_view fieldName, std::string_view className, std::string_view typeAlias = "");
   RStreamerField(RStreamerField &&other) = default;
   RStreamerField &operator=(RStreamerField &&other) = default;
   ~RStreamerField() final = default;

   size_t GetValueSize() const final;
   size_t GetAlignment() const final;
   std::uint32_t GetTypeVersion() const final;
   std::uint32_t GetTypeChecksum() const final;
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// The field for an unscoped or scoped enum with dictionary
class REnumField : public RFieldBase {
private:
   REnumField(std::string_view fieldName, TEnum *enump);
   REnumField(std::string_view fieldName, std::string_view enumName, std::unique_ptr<RFieldBase> intField);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const final { CallConstructValueOn(*fSubfields[0], where); }

   std::size_t AppendImpl(const void *from) final { return CallAppendOn(*fSubfields[0], from); }
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final { CallReadOn(*fSubfields[0], globalIndex, to); }
   void ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to) final { CallReadOn(*fSubfields[0], localIndex, to); }

public:
   REnumField(std::string_view fieldName, std::string_view enumName);
   REnumField(REnumField &&other) = default;
   REnumField &operator=(REnumField &&other) = default;
   ~REnumField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final { return fSubfields[0]->GetValueSize(); }
   size_t GetAlignment() const final { return fSubfields[0]->GetAlignment(); }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// Classes with dictionaries that can be inspected by TClass
template <typename T, typename = void>
class RField final : public RClassField {
public:
   static std::string TypeName() { return ROOT::Internal::GetDemangledTypeName(typeid(T)); }
   RField(std::string_view name) : RClassField(name, TypeName())
   {
      static_assert(std::is_class_v<T>, "no I/O support for this basic C++ type");
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

template <typename T>
class RField<T, typename std::enable_if<std::is_enum_v<T>>::type> final : public REnumField {
public:
   static std::string TypeName() { return ROOT::Internal::GetDemangledTypeName(typeid(T)); }
   RField(std::string_view name) : REnumField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

/// An artificial field that transforms an RNTuple column that contains the offset of collections into
/// collection sizes. It is only used for reading, e.g. as projected field or as an artificial field that provides the
/// "number of" RDF columns for collections (e.g. `R_rdf_sizeof_jets` for a collection named `jets`).
/// It is used in the templated RField<RNTupleCardinality<SizeT>> form, which represents the collection sizes either
/// as 32bit unsigned int (std::uint32_t) or as 64bit unsigned int (std::uint64_t).
class RCardinalityField : public RFieldBase {
   friend class RNTupleCollectionView; // to access GetCollectionInfo()

private:
   void GetCollectionInfo(ROOT::NTupleSize_t globalIndex, RNTupleLocalIndex *collectionStart, ROOT::NTupleSize_t *size)
   {
      fPrincipalColumn->GetCollectionInfo(globalIndex, collectionStart, size);
   }
   void GetCollectionInfo(RNTupleLocalIndex localIndex, RNTupleLocalIndex *collectionStart, ROOT::NTupleSize_t *size)
   {
      fPrincipalColumn->GetCollectionInfo(localIndex, collectionStart, size);
   }

protected:
   RCardinalityField(std::string_view fieldName, std::string_view typeName)
      : RFieldBase(fieldName, typeName, ROOT::ENTupleStructure::kLeaf, false /* isSimple */)
   {
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;
   // Field is only used for reading
   void GenerateColumns() final { throw RException(R__FAIL("Cardinality fields must only be used for reading")); }
   void GenerateColumns(const RNTupleDescriptor &) final;

public:
   RCardinalityField(RCardinalityField &&other) = default;
   RCardinalityField &operator=(RCardinalityField &&other) = default;
   ~RCardinalityField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;

   const RField<RNTupleCardinality<std::uint32_t>> *As32Bit() const;
   const RField<RNTupleCardinality<std::uint64_t>> *As64Bit() const;
};

template <typename T>
class RSimpleField : public RFieldBase {
protected:
   void GenerateColumns() override { GenerateColumnsImpl<T>(); }
   void GenerateColumns(const RNTupleDescriptor &desc) override { GenerateColumnsImpl<T>(desc); }

   void ConstructValue(void *where) const final { new (where) T{0}; }

public:
   RSimpleField(std::string_view name, std::string_view type)
      : RFieldBase(name, type, ROOT::ENTupleStructure::kLeaf, true /* isSimple */)
   {
      fTraits |= kTraitTrivialType;
   }
   RSimpleField(RSimpleField &&other) = default;
   RSimpleField &operator=(RSimpleField &&other) = default;
   ~RSimpleField() override = default;

   T *Map(ROOT::NTupleSize_t globalIndex) { return fPrincipalColumn->Map<T>(globalIndex); }
   T *Map(RNTupleLocalIndex localIndex) { return fPrincipalColumn->Map<T>(localIndex); }
   T *MapV(ROOT::NTupleSize_t globalIndex, ROOT::NTupleSize_t &nItems)
   {
      return fPrincipalColumn->MapV<T>(globalIndex, nItems);
   }
   T *MapV(RNTupleLocalIndex localIndex, ROOT::NTupleSize_t &nItems)
   {
      return fPrincipalColumn->MapV<T>(localIndex, nItems);
   }

   size_t GetValueSize() const final { return sizeof(T); }
   size_t GetAlignment() const final { return alignof(T); }
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for concrete C++ types
////////////////////////////////////////////////////////////////////////////////

} // namespace Experimental
} // namespace ROOT

#include "RField/RFieldFundamental.hxx"
#include "RField/RFieldProxiedCollection.hxx"
#include "RField/RFieldRecord.hxx"
#include "RField/RFieldSequenceContainer.hxx"
#include "RField/RFieldSTLMisc.hxx"

namespace ROOT {
namespace Experimental {

template <typename SizeT>
class RField<RNTupleCardinality<SizeT>> final : public RCardinalityField {
protected:
   std::unique_ptr<ROOT::Experimental::RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField<RNTupleCardinality<SizeT>>>(newName);
   }
   void ConstructValue(void *where) const final { new (where) RNTupleCardinality<SizeT>(0); }

   /// Get the number of elements of the collection identified by globalIndex
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final
   {
      RNTupleLocalIndex collectionStart;
      ROOT::NTupleSize_t size;
      fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &size);
      *static_cast<RNTupleCardinality<SizeT> *>(to) = size;
   }

   /// Get the number of elements of the collection identified by clusterIndex
   void ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to) final
   {
      RNTupleLocalIndex collectionStart;
      ROOT::NTupleSize_t size;
      fPrincipalColumn->GetCollectionInfo(localIndex, &collectionStart, &size);
      *static_cast<RNTupleCardinality<SizeT> *>(to) = size;
   }

   std::size_t ReadBulkImpl(const RBulkSpec &bulkSpec) final
   {
      RNTupleLocalIndex collectionStart;
      ROOT::NTupleSize_t collectionSize;
      fPrincipalColumn->GetCollectionInfo(bulkSpec.fFirstIndex, &collectionStart, &collectionSize);

      auto typedValues = static_cast<RNTupleCardinality<SizeT> *>(bulkSpec.fValues);
      typedValues[0] = collectionSize;

      auto lastOffset = collectionStart.GetIndexInCluster() + collectionSize;
      ROOT::NTupleSize_t nRemainingEntries = bulkSpec.fCount - 1;
      std::size_t nEntries = 1;
      while (nRemainingEntries > 0) {
         ROOT::NTupleSize_t nItemsUntilPageEnd;
         auto offsets =
            fPrincipalColumn->MapV<Internal::RColumnIndex>(bulkSpec.fFirstIndex + nEntries, nItemsUntilPageEnd);
         std::size_t nBatch = std::min(nRemainingEntries, nItemsUntilPageEnd);
         for (std::size_t i = 0; i < nBatch; ++i) {
            typedValues[nEntries + i] = offsets[i] - lastOffset;
            lastOffset = offsets[i];
         }
         nRemainingEntries -= nBatch;
         nEntries += nBatch;
      }
      return RBulkSpec::kAllSet;
   }

public:
   static std::string TypeName() { return "ROOT::RNTupleCardinality<" + RField<SizeT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : RCardinalityField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;

   size_t GetValueSize() const final { return sizeof(RNTupleCardinality<SizeT>); }
   size_t GetAlignment() const final { return alignof(RNTupleCardinality<SizeT>); }
};

/// TObject requires special handling of the fBits and fUniqueID members
template <>
class RField<TObject> final : public RFieldBase {
   static std::size_t GetOffsetOfMember(const char *name);
   static std::size_t GetOffsetUniqueID() { return GetOffsetOfMember("fUniqueID"); }
   static std::size_t GetOffsetBits() { return GetOffsetOfMember("fBits"); }

private:
   RField(std::string_view fieldName, const RField<TObject> &source); ///< Used by CloneImpl()

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<TObject>>(); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;

   void AfterConnectPageSource() final;

public:
   static std::string TypeName() { return "TObject"; }

   RField(std::string_view fieldName);
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final;
   size_t GetAlignment() const final;
   std::uint32_t GetTypeVersion() const final;
   std::uint32_t GetTypeChecksum() const final;
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

// Has to be implemented after the definition of all RField<T> types
// The void type is specialized in RField.cxx

template <typename T>
std::unique_ptr<T, typename RFieldBase::RCreateObjectDeleter<T>::deleter> RFieldBase::CreateObject() const
{
   if (GetTypeName() != RField<T>::TypeName()) {
      throw RException(
         R__FAIL("type mismatch for field " + GetFieldName() + ": " + GetTypeName() + " vs. " + RField<T>::TypeName()));
   }
   return std::unique_ptr<T>(static_cast<T *>(CreateObjectRawPtr()));
}

template <>
struct RFieldBase::RCreateObjectDeleter<void> {
   using deleter = RCreateObjectDeleter<void>;
   void operator()(void *);
};

template <>
std::unique_ptr<void, typename RFieldBase::RCreateObjectDeleter<void>::deleter>
ROOT::Experimental::RFieldBase::CreateObject<void>() const;

} // namespace Experimental
} // namespace ROOT

#endif
