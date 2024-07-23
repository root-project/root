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
#include <ROOT/RVec.hxx>
#include <ROOT/TypeTraits.hxx>

#include <TGenericClassInfo.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <variant>
#include <vector>
#include <utility>

class TClass;
class TEnum;
class TObject;
class TVirtualStreamerInfo;

namespace ROOT {

class TSchemaRule;

namespace Experimental {

class RNTupleCollectionWriter;
class REntry;

namespace Detail {
class RFieldVisitor;
} // namespace Detail

/// The container field for an ntuple model, which itself has no physical representation.
/// Therefore, the zero field must not be connected to a page source or sink.
class RFieldZero final : public RFieldBase {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const override;
   void ConstructValue(void *) const final {}

public:
   RFieldZero() : RFieldBase("", "", ENTupleStructure::kRecord, false /* isSimple */) {}

   using RFieldBase::Attach;
   size_t GetValueSize() const final { return 0; }
   size_t GetAlignment() const final { return 0; }

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// Used in RFieldBase::Check() to record field creation failures.
class RInvalidField final : public RFieldBase {
   std::string fError;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RInvalidField>(newName, GetTypeName(), fError);
   }
   void ConstructValue(void *) const final {}

public:
   RInvalidField(std::string_view name, std::string_view type, std::string_view error)
      : RFieldBase(name, type, ENTupleStructure::kLeaf, false /* isSimple */), fError(error)
   {
   }

   const std::string &GetError() const { return fError; }

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
   /// Additional information kept for each entry in `fSubFields`
   std::vector<RSubFieldInfo> fSubFieldsInfo;
   std::size_t fMaxAlignment = 1;

private:
   RClassField(std::string_view fieldName, std::string_view className, TClass *classp);
   void Attach(std::unique_ptr<RFieldBase> child, RSubFieldInfo info);
   /// Register post-read callbacks corresponding to a list of ROOT I/O customization rules. `classp` is used to
   /// fill the `TVirtualObject` instance passed to the user function.
   void AddReadCallbacksFromIORules(const std::span<const TSchemaRule *> rules, TClass *classp = nullptr);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RClassDeleter>(fClass); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;
   void ReadInClusterImpl(RClusterIndex clusterIndex, void *to) final;
   void OnConnectPageSource() final;

public:
   RClassField(std::string_view fieldName, std::string_view className);
   RClassField(RClassField &&other) = default;
   RClassField &operator=(RClassField &&other) = default;
   ~RClassField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const override;
   size_t GetAlignment() const final { return fMaxAlignment; }
   std::uint32_t GetTypeVersion() const final;
   std::uint32_t GetTypeChecksum() const final;
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const override;
};

/// The field for a class in unsplit mode, which is using ROOT standard streaming
class RUnsplitField final : public RFieldBase {
private:
   class RUnsplitDeleter : public RDeleter {
   private:
      TClass *fClass;

   public:
      explicit RUnsplitDeleter(TClass *cl) : fClass(cl) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   TClass *fClass = nullptr;
   Internal::RNTupleSerializer::StreamerInfoMap_t fStreamerInfos; ///< streamer info records seen during writing
   ClusterSize_t fIndex;                                          ///< number of bytes written in the current cluster

private:
   // Note that className may be different from classp->GetName(), e.g. through different canonicalization of RNTuple
   // vs. TClass. Also, classp may be nullptr for types unsupported by the ROOT I/O.
   RUnsplitField(std::string_view fieldName, std::string_view className, TClass *classp);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &) final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RUnsplitDeleter>(fClass); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

   void CommitClusterImpl() final { fIndex = 0; }

   bool HasExtraTypeInfo() const final { return true; }
   // Returns the list of seen streamer infos
   RExtraTypeInfoDescriptor GetExtraTypeInfo() const final;

public:
   RUnsplitField(std::string_view fieldName, std::string_view className, std::string_view typeAlias = "");
   RUnsplitField(RUnsplitField &&other) = default;
   RUnsplitField &operator=(RUnsplitField &&other) = default;
   ~RUnsplitField() override = default;

   size_t GetValueSize() const final;
   size_t GetAlignment() const final;
   std::uint32_t GetTypeVersion() const final;
   std::uint32_t GetTypeChecksum() const final;
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// The field for an unscoped or scoped enum with dictionary
class REnumField : public RFieldBase {
private:
   REnumField(std::string_view fieldName, std::string_view enumName, TEnum *enump);
   REnumField(std::string_view fieldName, std::string_view enumName, std::unique_ptr<RFieldBase> intField);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const final { CallConstructValueOn(*fSubFields[0], where); }

   std::size_t AppendImpl(const void *from) final { return CallAppendOn(*fSubFields[0], from); }
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final { CallReadOn(*fSubFields[0], globalIndex, to); }
   void ReadInClusterImpl(RClusterIndex clusterIndex, void *to) final { CallReadOn(*fSubFields[0], clusterIndex, to); }

public:
   REnumField(std::string_view fieldName, std::string_view enumName);
   REnumField(REnumField &&other) = default;
   REnumField &operator=(REnumField &&other) = default;
   ~REnumField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final { return fSubFields[0]->GetValueSize(); }
   size_t GetAlignment() const final { return fSubFields[0]->GetAlignment(); }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// The field for an untyped record. The subfields are stored consequitively in a memory block, i.e.
/// the memory layout is identical to one that a C++ struct would have
class RRecordField : public RFieldBase {
private:
   class RRecordDeleter : public RDeleter {
   private:
      std::vector<std::unique_ptr<RDeleter>> fItemDeleters;
      std::vector<std::size_t> fOffsets;

   public:
      RRecordDeleter(std::vector<std::unique_ptr<RDeleter>> &itemDeleters, const std::vector<std::size_t> &offsets)
         : fItemDeleters(std::move(itemDeleters)), fOffsets(offsets)
      {
      }
      void operator()(void *objPtr, bool dtorOnly) final;
   };

protected:
   std::size_t fMaxAlignment = 1;
   std::size_t fSize = 0;
   std::vector<std::size_t> fOffsets;

   std::size_t GetItemPadding(std::size_t baseOffset, std::size_t itemAlignment) const;

   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const override;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const override;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;
   void ReadInClusterImpl(RClusterIndex clusterIndex, void *to) final;

   RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &&itemFields,
                const std::vector<std::size_t> &offsets, std::string_view typeName = "");

   template <std::size_t N>
   RRecordField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, N> &&itemFields,
                const std::array<std::size_t, N> &offsets, std::string_view typeName = "")
      : ROOT::Experimental::RFieldBase(fieldName, typeName, ENTupleStructure::kRecord, false /* isSimple */)
   {
      fTraits |= kTraitTrivialType;
      for (unsigned i = 0; i < N; ++i) {
         fOffsets.push_back(offsets[i]);
         fMaxAlignment = std::max(fMaxAlignment, itemFields[i]->GetAlignment());
         fSize += GetItemPadding(fSize, itemFields[i]->GetAlignment()) + itemFields[i]->GetValueSize();
         fTraits &= itemFields[i]->GetTraits();
         Attach(std::move(itemFields[i]));
      }
   }

public:
   /// Construct a RRecordField based on a vector of child fields. The ownership of the child fields is transferred
   /// to the RRecordField instance.
   RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &&itemFields);
   RRecordField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &itemFields);
   RRecordField(RRecordField &&other) = default;
   RRecordField &operator=(RRecordField &&other) = default;
   ~RRecordField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final { return fSize; }
   size_t GetAlignment() const final { return fMaxAlignment; }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// The generic field for a (nested) std::vector<Type> except for std::vector<bool>
class RVectorField : public RFieldBase {
private:
   class RVectorDeleter : public RDeleter {
   private:
      std::size_t fItemSize = 0;
      std::unique_ptr<RDeleter> fItemDeleter;

   public:
      RVectorDeleter() = default;
      RVectorDeleter(std::size_t itemSize, std::unique_ptr<RDeleter> itemDeleter)
         : fItemSize(itemSize), fItemDeleter(std::move(itemDeleter))
      {
      }
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   std::size_t fItemSize;
   ClusterSize_t fNWritten;
   std::unique_ptr<RDeleter> fItemDeleter;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const override { new (where) std::vector<char>(); }
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

   void CommitClusterImpl() final { fNWritten = 0; }

public:
   RVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField);
   RVectorField(RVectorField &&other) = default;
   RVectorField &operator=(RVectorField &&other) = default;
   ~RVectorField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const override { return sizeof(std::vector<char>); }
   size_t GetAlignment() const final { return std::alignment_of<std::vector<char>>(); }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
   void GetCollectionInfo(NTupleSize_t globalIndex, RClusterIndex *collectionStart, ClusterSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(globalIndex, collectionStart, size);
   }
   void GetCollectionInfo(RClusterIndex clusterIndex, RClusterIndex *collectionStart, ClusterSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(clusterIndex, collectionStart, size);
   }
};

/// The type-erased field for a RVec<Type>
class RRVecField : public RFieldBase {
public:
   /// the RRVecDeleter is also used by RArrayAsRVecField and therefore declared public
   class RRVecDeleter : public RDeleter {
   private:
      std::size_t fItemAlignment;
      std::size_t fItemSize = 0;
      std::unique_ptr<RDeleter> fItemDeleter;

   public:
      explicit RRVecDeleter(std::size_t itemAlignment) : fItemAlignment(itemAlignment) {}
      RRVecDeleter(std::size_t itemAlignment, std::size_t itemSize, std::unique_ptr<RDeleter> itemDeleter)
         : fItemAlignment(itemAlignment), fItemSize(itemSize), fItemDeleter(std::move(itemDeleter))
      {
      }
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   std::unique_ptr<RDeleter> fItemDeleter;

protected:
   std::size_t fItemSize;
   ClusterSize_t fNWritten;
   std::size_t fValueSize;

   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const override;
   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const override;

   std::size_t AppendImpl(const void *from) override;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) override;
   std::size_t ReadBulkImpl(const RBulkSpec &bulkSpec) final;

   void CommitClusterImpl() final { fNWritten = 0; }

public:
   RRVecField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField);
   RRVecField(RRVecField &&) = default;
   RRVecField &operator=(RRVecField &&) = default;
   RRVecField(const RRVecField &) = delete;
   RRVecField &operator=(RRVecField &) = delete;
   ~RRVecField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const override;
   size_t GetAlignment() const override;
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
   void GetCollectionInfo(NTupleSize_t globalIndex, RClusterIndex *collectionStart, ClusterSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(globalIndex, collectionStart, size);
   }
   void GetCollectionInfo(RClusterIndex clusterIndex, RClusterIndex *collectionStart, ClusterSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(clusterIndex, collectionStart, size);
   }
};

/// The generic field for fixed size arrays, which do not need an offset column
class RArrayField : public RFieldBase {
private:
   class RArrayDeleter : public RDeleter {
   private:
      std::size_t fItemSize = 0;
      std::size_t fArrayLength = 0;
      std::unique_ptr<RDeleter> fItemDeleter;

   public:
      RArrayDeleter(std::size_t itemSize, std::size_t arrayLength, std::unique_ptr<RDeleter> itemDeleter)
         : fItemSize(itemSize), fArrayLength(arrayLength), fItemDeleter(std::move(itemDeleter))
      {
      }
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   std::size_t fItemSize;
   std::size_t fArrayLength;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;
   void ReadInClusterImpl(RClusterIndex clusterIndex, void *to) final;

public:
   RArrayField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField, std::size_t arrayLength);
   RArrayField(RArrayField &&other) = default;
   RArrayField &operator=(RArrayField &&other) = default;
   ~RArrayField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetLength() const { return fArrayLength; }
   size_t GetValueSize() const final { return fItemSize * fArrayLength; }
   size_t GetAlignment() const final { return fSubFields[0]->GetAlignment(); }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/**
\class ROOT::Experimental::RArrayAsRVecField
\brief A field for fixed-size arrays that are represented as RVecs in memory.
\ingroup NTuple
This class is used only for reading. In particular, it helps exposing
arbitrarily-nested std::array on-disk fields as RVecs for usage in RDataFrame.
*/
class RArrayAsRVecField final : public RFieldBase {
private:
   std::unique_ptr<RDeleter> fItemDeleter; /// Sub field deleter or nullptr for simple fields
   std::size_t fItemSize;                  /// The size of a child field's item
   std::size_t fArrayLength;               /// The length of the arrays in this field
   std::size_t fValueSize;                 /// The size of a value of this field, i.e. an RVec

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void GenerateColumns() final { R__ASSERT(false && "RArrayAsRVec fields must only be used for reading"); }
   using RFieldBase::GenerateColumns;

   void ConstructValue(void *where) const final;
   /// Returns an RRVecField::RRVecDeleter
   std::unique_ptr<RDeleter> GetDeleter() const final;

   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;
   void ReadInClusterImpl(RClusterIndex clusterIndex, void *to) final;

public:
   /**
      Constructor of the field. the \p itemField argument represents the inner
      item of the on-disk array, i.e. for an `std::array<float>` it is the `float`
      field and not the `std::array` itself.
   */
   RArrayAsRVecField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField, std::size_t arrayLength);
   RArrayAsRVecField(const RArrayAsRVecField &other) = delete;
   RArrayAsRVecField &operator=(const RArrayAsRVecField &other) = delete;
   RArrayAsRVecField(RArrayAsRVecField &&other) = default;
   RArrayAsRVecField &operator=(RArrayAsRVecField &&other) = default;
   ~RArrayAsRVecField() final = default;

   std::size_t GetValueSize() const final { return fValueSize; }
   std::size_t GetAlignment() const final;

   std::vector<RFieldBase::RValue> SplitValue(const RFieldBase::RValue &value) const final;
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// The generic field an std::bitset<N>. All compilers we care about store the bits in an array of unsigned long.
/// TODO(jblomer): reading and writing efficiency should be improved; currently it is one bit at a time
/// with an array of bools on the page level.
class RBitsetField : public RFieldBase {
   using Word_t = unsigned long;
   static constexpr std::size_t kWordSize = sizeof(Word_t);
   static constexpr std::size_t kBitsPerWord = kWordSize * 8;

protected:
   std::size_t fN;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RBitsetField>(newName, fN);
   }
   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;
   void ConstructValue(void *where) const final { memset(where, 0, GetValueSize()); }
   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

public:
   RBitsetField(std::string_view fieldName, std::size_t N);
   RBitsetField(RBitsetField &&other) = default;
   RBitsetField &operator=(RBitsetField &&other) = default;
   ~RBitsetField() override = default;

   size_t GetValueSize() const final { return kWordSize * ((fN + kBitsPerWord - 1) / kBitsPerWord); }
   size_t GetAlignment() const final { return alignof(Word_t); }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;

   /// Get the number of bits in the bitset, i.e. the N in std::bitset<N>
   std::size_t GetN() const { return fN; }
};

/// The generic field for std::variant types
class RVariantField : public RFieldBase {
private:
   // Most compilers support at least 255 variants (256 - 1 value for the empty variant).
   // Some compilers switch to a two-byte tag field already with 254 variants.
   // MSVC only supports 163 variants in older versions, 250 in newer ones. It switches to a 2 byte
   // tag as of 128 variants (at least in debug mode), so for simplicity we set the limit to 125 variants.
   static constexpr std::size_t kMaxVariants = 125;

   class RVariantDeleter : public RDeleter {
   private:
      std::size_t fTagOffset;
      std::size_t fVariantOffset;
      std::vector<std::unique_ptr<RDeleter>> fItemDeleters;

   public:
      RVariantDeleter(std::size_t tagOffset, std::size_t variantOffset,
                      std::vector<std::unique_ptr<RDeleter>> &itemDeleters)
         : fTagOffset(tagOffset), fVariantOffset(variantOffset), fItemDeleters(std::move(itemDeleters))
      {
      }
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   size_t fMaxItemSize = 0;
   size_t fMaxAlignment = 1;
   /// In the std::variant memory layout, at which byte number is the index stored
   size_t fTagOffset = 0;
   /// In the std::variant memory layout, the actual union of types may start at an offset > 0
   size_t fVariantOffset = 0;
   std::vector<ClusterSize_t::ValueType> fNWritten;

   static std::string GetTypeList(const std::vector<RFieldBase *> &itemFields);
   /// Extracts the index from an std::variant and transforms it into the 1-based index used for the switch column
   /// The implementation supports two memory layouts that are in use: a trailing unsigned byte, zero-indexed,
   /// having the exception caused empty state encoded by the max tag value,
   /// or a trailing unsigned int instead of a char.
   static std::uint8_t GetTag(const void *variantPtr, std::size_t tagOffset);
   static void SetTag(void *variantPtr, std::size_t tagOffset, std::uint8_t tag);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

   void CommitClusterImpl() final;

public:
   // TODO(jblomer): use std::span in signature
   RVariantField(std::string_view fieldName, const std::vector<RFieldBase *> &itemFields);
   RVariantField(RVariantField &&other) = default;
   RVariantField &operator=(RVariantField &&other) = default;
   ~RVariantField() override = default;

   size_t GetValueSize() const final;
   size_t GetAlignment() const final;
};

/// The field for values that may or may not be present in an entry. Parent class for unique pointer field and
/// optional field. A nullable field cannot be instantiated itself but only its descendants.
/// The RNullableField takes care of the on-disk representation. Child classes are responsible for the in-memory
/// representation.  The on-disk representation can be "dense" or "sparse". Dense nullable fields have a bitmask
/// (true: item available, false: item missing) and serialize a default-constructed item for missing items.
/// Sparse nullable fields use a (Split)Index[64|32] column to point to the available items.
/// By default, items whose size is smaller or equal to 4 bytes (size of (Split)Index32 column element) are stored
/// densely.
class RNullableField : public RFieldBase {
   /// For a dense nullable field, used to write a default-constructed item for missing ones.
   std::unique_ptr<RValue> fDefaultItemValue;
   /// For a sparse nullable field, the number of written non-null items in this cluster
   ClusterSize_t fNWritten{0};

protected:
   const RFieldBase::RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &) final;

   std::size_t AppendNull();
   std::size_t AppendValue(const void *from);
   void CommitClusterImpl() final { fNWritten = 0; }

   // The default implementation that translates the request into a call to ReadGlobalImpl() cannot be used
   // because we don't team up the columns of the different column representations for this field.
   void ReadInClusterImpl(RClusterIndex clusterIndex, void *to) final;

   /// Given the index of the nullable field, returns the corresponding global index of the subfield or,
   /// if it is null, returns kInvalidClusterIndex
   RClusterIndex GetItemIndex(NTupleSize_t globalIndex);

   RNullableField(std::string_view fieldName, std::string_view typeName, std::unique_ptr<RFieldBase> itemField);

public:
   RNullableField(RNullableField &&other) = default;
   RNullableField &operator=(RNullableField &&other) = default;
   ~RNullableField() override = default;

   void SetDense() { SetColumnRepresentatives({{EColumnType::kBit}}); }
   void SetSparse() { SetColumnRepresentatives({{EColumnType::kSplitIndex64}}); }

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

class RUniquePtrField : public RNullableField {
   class RUniquePtrDeleter : public RDeleter {
   private:
      std::unique_ptr<RDeleter> fItemDeleter;

   public:
      explicit RUniquePtrDeleter(std::unique_ptr<RDeleter> itemDeleter) : fItemDeleter(std::move(itemDeleter)) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   std::unique_ptr<RDeleter> fItemDeleter;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const final { new (where) std::unique_ptr<char>(); }
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

public:
   RUniquePtrField(std::string_view fieldName, std::string_view typeName, std::unique_ptr<RFieldBase> itemField);
   RUniquePtrField(RUniquePtrField &&other) = default;
   RUniquePtrField &operator=(RUniquePtrField &&other) = default;
   ~RUniquePtrField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final { return sizeof(std::unique_ptr<char>); }
   size_t GetAlignment() const final { return alignof(std::unique_ptr<char>); }
};

class ROptionalField : public RNullableField {
   class ROptionalDeleter : public RDeleter {
   private:
      std::unique_ptr<RDeleter> fItemDeleter;

   public:
      explicit ROptionalDeleter(std::unique_ptr<RDeleter> itemDeleter) : fItemDeleter(std::move(itemDeleter)) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   std::unique_ptr<RDeleter> fItemDeleter;

   /// Given a pointer to an std::optional<T> in `optionalPtr`, extract a pointer to the value T* and a pointer
   /// to the engagement boolean. Assumes that an std::optional<T> is stored as
   /// `struct { T t; bool engagement; };`
   std::pair<const void *, const bool *> GetValueAndEngagementPtrs(const void *optionalPtr) const;
   std::pair<void *, bool *> GetValueAndEngagementPtrs(void *optionalPtr) const;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

public:
   ROptionalField(std::string_view fieldName, std::string_view typeName, std::unique_ptr<RFieldBase> itemField);
   ROptionalField(ROptionalField &&other) = default;
   ROptionalField &operator=(ROptionalField &&other) = default;
   ~ROptionalField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final;
   size_t GetAlignment() const final;
};

class RAtomicField : public RFieldBase {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const final { CallConstructValueOn(*fSubFields[0], where); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return GetDeleterOf(*fSubFields[0]); }

   std::size_t AppendImpl(const void *from) final { return CallAppendOn(*fSubFields[0], from); }
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final { CallReadOn(*fSubFields[0], globalIndex, to); }
   void ReadInClusterImpl(RClusterIndex clusterIndex, void *to) final { CallReadOn(*fSubFields[0], clusterIndex, to); }

public:
   RAtomicField(std::string_view fieldName, std::string_view typeName, std::unique_ptr<RFieldBase> itemField);
   RAtomicField(RAtomicField &&other) = default;
   RAtomicField &operator=(RAtomicField &&other) = default;
   ~RAtomicField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;

   size_t GetValueSize() const final { return fSubFields[0]->GetValueSize(); }
   size_t GetAlignment() const final { return fSubFields[0]->GetAlignment(); }

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// Classes with dictionaries that can be inspected by TClass
template <typename T, typename = void>
class RField final : public RClassField {
protected:
   void ConstructValue(void *where) const final
   {
      if constexpr (std::is_default_constructible_v<T>) {
         new (where) T();
      } else {
         // If there is no default constructor, try with the IO constructor
         new (where) T(static_cast<TRootIOCtor *>(nullptr));
      }
   }

public:
   static std::string TypeName() { return ROOT::Internal::GetDemangledTypeName(typeid(T)); }
   RField(std::string_view name) : RClassField(name, TypeName())
   {
      static_assert(std::is_class_v<T>, "no I/O support for this basic C++ type");
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

template <typename T>
class RField<T, typename std::enable_if<std::is_enum_v<T>>::type> : public REnumField {
public:
   static std::string TypeName() { return ROOT::Internal::GetDemangledTypeName(typeid(T)); }
   RField(std::string_view name) : REnumField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

/// The collection field is only used for writing; when reading, untyped collections are projected to an std::vector
class RCollectionField final : public ROOT::Experimental::RFieldBase {
private:
   /// Save the link to the collection ntuple in order to reset the offset counter when committing the cluster
   std::shared_ptr<RNTupleCollectionWriter> fCollectionWriter;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;
   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;
   void ConstructValue(void *) const final {}

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

   void CommitClusterImpl() final;

public:
   static std::string TypeName() { return ""; }
   RCollectionField(std::string_view name, std::shared_ptr<RNTupleCollectionWriter> collectionWriter,
                    std::unique_ptr<RFieldZero> collectionParent);
   RCollectionField(RCollectionField &&other) = default;
   RCollectionField &operator=(RCollectionField &&other) = default;
   ~RCollectionField() override = default;

   size_t GetValueSize() const final { return sizeof(ClusterSize_t); }
   size_t GetAlignment() const final { return alignof(ClusterSize_t); }
};

/// The generic field for `std::pair<T1, T2>` types
class RPairField : public RRecordField {
private:
   class RPairDeleter : public RDeleter {
   private:
      TClass *fClass;

   public:
      explicit RPairDeleter(TClass *cl) : fClass(cl) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   TClass *fClass = nullptr;
   static std::string GetTypeList(const std::array<std::unique_ptr<RFieldBase>, 2> &itemFields);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const override;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const override { return std::make_unique<RPairDeleter>(fClass); }

   RPairField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, 2> &&itemFields,
              const std::array<std::size_t, 2> &offsets);

public:
   RPairField(std::string_view fieldName, std::array<std::unique_ptr<RFieldBase>, 2> &itemFields);
   RPairField(RPairField &&other) = default;
   RPairField &operator=(RPairField &&other) = default;
   ~RPairField() override = default;
};

/// The generic field for `std::tuple<Ts...>` types
class RTupleField : public RRecordField {
private:
   class RTupleDeleter : public RDeleter {
   private:
      TClass *fClass;

   public:
      explicit RTupleDeleter(TClass *cl) : fClass(cl) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   TClass *fClass = nullptr;
   static std::string GetTypeList(const std::vector<std::unique_ptr<RFieldBase>> &itemFields);

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const override;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const override { return std::make_unique<RTupleDeleter>(fClass); }

   RTupleField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &&itemFields,
               const std::vector<std::size_t> &offsets);

public:
   RTupleField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> &itemFields);
   RTupleField(RTupleField &&other) = default;
   RTupleField &operator=(RTupleField &&other) = default;
   ~RTupleField() override = default;
};

/// An artificial field that transforms an RNTuple column that contains the offset of collections into
/// collection sizes. It is only used for reading, e.g. as projected field or as an artificial field that provides the
/// "number of" RDF columns for collections (e.g. `R_rdf_sizeof_jets` for a collection named `jets`).
/// It is used in the templated RField<RNTupleCardinality<SizeT>> form, which represents the collection sizes either
/// as 32bit unsigned int (std::uint32_t) or as 64bit unsigned int (std::uint64_t).
class RCardinalityField : public RFieldBase {
protected:
   RCardinalityField(std::string_view fieldName, std::string_view typeName)
      : RFieldBase(fieldName, typeName, ENTupleStructure::kLeaf, false /* isSimple */)
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
   void GenerateColumns() final { GenerateColumnsImpl<T>(); }
   void GenerateColumns(const RNTupleDescriptor &desc) final { GenerateColumnsImpl<T>(desc); }

   void ConstructValue(void *where) const final { new (where) T{0}; }

public:
   RSimpleField(std::string_view name, std::string_view type)
      : RFieldBase(name, type, ENTupleStructure::kLeaf, true /* isSimple */)
   {
      fTraits |= kTraitTrivialType;
   }
   RSimpleField(RSimpleField &&other) = default;
   RSimpleField &operator=(RSimpleField &&other) = default;
   ~RSimpleField() override = default;

   T *Map(NTupleSize_t globalIndex) { return fPrincipalColumn->Map<T>(globalIndex); }
   T *Map(RClusterIndex clusterIndex) { return fPrincipalColumn->Map<T>(clusterIndex); }
   T *MapV(NTupleSize_t globalIndex, NTupleSize_t &nItems) { return fPrincipalColumn->MapV<T>(globalIndex, nItems); }
   T *MapV(RClusterIndex clusterIndex, NTupleSize_t &nItems) { return fPrincipalColumn->MapV<T>(clusterIndex, nItems); }

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

namespace ROOT {
namespace Experimental {

template <>
class RField<ClusterSize_t> final : public RSimpleField<ClusterSize_t> {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "ROOT::Experimental::ClusterSize_t"; }
   explicit RField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   /// Special help for offset fields
   void GetCollectionInfo(NTupleSize_t globalIndex, RClusterIndex *collectionStart, ClusterSize_t *size)
   {
      fPrincipalColumn->GetCollectionInfo(globalIndex, collectionStart, size);
   }
   void GetCollectionInfo(RClusterIndex clusterIndex, RClusterIndex *collectionStart, ClusterSize_t *size)
   {
      fPrincipalColumn->GetCollectionInfo(clusterIndex, collectionStart, size);
   }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

template <typename SizeT>
class RField<RNTupleCardinality<SizeT>> final : public RCardinalityField {
protected:
   std::unique_ptr<ROOT::Experimental::RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField<RNTupleCardinality<SizeT>>>(newName);
   }
   void ConstructValue(void *where) const final { new (where) RNTupleCardinality<SizeT>(0); }

public:
   static std::string TypeName() { return "ROOT::Experimental::RNTupleCardinality<" + RField<SizeT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : RCardinalityField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   size_t GetValueSize() const final { return sizeof(RNTupleCardinality<SizeT>); }
   size_t GetAlignment() const final { return alignof(RNTupleCardinality<SizeT>); }

   /// Get the number of elements of the collection identified by globalIndex
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final
   {
      RClusterIndex collectionStart;
      ClusterSize_t size;
      fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &size);
      *static_cast<RNTupleCardinality<SizeT> *>(to) = size;
   }

   /// Get the number of elements of the collection identified by clusterIndex
   void ReadInClusterImpl(RClusterIndex clusterIndex, void *to) final
   {
      RClusterIndex collectionStart;
      ClusterSize_t size;
      fPrincipalColumn->GetCollectionInfo(clusterIndex, &collectionStart, &size);
      *static_cast<RNTupleCardinality<SizeT> *>(to) = size;
   }

   std::size_t ReadBulkImpl(const RBulkSpec &bulkSpec) final
   {
      RClusterIndex collectionStart;
      ClusterSize_t collectionSize;
      fPrincipalColumn->GetCollectionInfo(bulkSpec.fFirstIndex, &collectionStart, &collectionSize);

      auto typedValues = static_cast<RNTupleCardinality<SizeT> *>(bulkSpec.fValues);
      typedValues[0] = collectionSize;

      auto lastOffset = collectionStart.GetIndex() + collectionSize;
      ClusterSize_t::ValueType nRemainingEntries = bulkSpec.fCount - 1;
      std::size_t nEntries = 1;
      while (nRemainingEntries > 0) {
         NTupleSize_t nItemsUntilPageEnd;
         auto offsets = fPrincipalColumn->MapV<ClusterSize_t>(bulkSpec.fFirstIndex + nEntries, nItemsUntilPageEnd);
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
};

template <>
class RField<std::byte> final : public RSimpleField<std::byte> {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;

public:
   static std::string TypeName() { return "std::byte"; }
   explicit RField(std::string_view name) : RSimpleField(name, TypeName()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

template <>
class RField<std::string> final : public RFieldBase {
private:
   ClusterSize_t fIndex;

   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const final { new (where) std::string(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<std::string>>(); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::Experimental::NTupleSize_t globalIndex, void *to) final;

   void CommitClusterImpl() final { fIndex = 0; }

public:
   static std::string TypeName() { return "std::string"; }
   explicit RField(std::string_view name)
      : RFieldBase(name, TypeName(), ENTupleStructure::kLeaf, false /* isSimple */), fIndex(0)
   {
   }
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   size_t GetValueSize() const final { return sizeof(std::string); }
   size_t GetAlignment() const final { return std::alignment_of<std::string>(); }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

/// TObject requires special handling of the fBits and fUniqueID members
template <>
class RField<TObject> final : public RFieldBase {
   static std::size_t GetOffsetOfMember(const char *name);
   static std::size_t GetOffsetUniqueID() { return GetOffsetOfMember("fUniqueID"); }
   static std::size_t GetOffsetBits() { return GetOffsetOfMember("fBits"); }

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const override;
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<TObject>>(); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

   void OnConnectPageSource() final;

public:
   static std::string TypeName() { return "TObject"; }

   RField(std::string_view fieldName);
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final;
   size_t GetAlignment() const final;
   std::uint32_t GetTypeVersion() const final;
   std::uint32_t GetTypeChecksum() const final;
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

template <typename ItemT, std::size_t N>
class RField<std::array<ItemT, N>> : public RArrayField {
   using ContainerT = typename std::array<ItemT, N>;

protected:
   void ConstructValue(void *where) const final { new (where) ContainerT(); }

public:
   static std::string TypeName() { return "std::array<" + RField<ItemT>::TypeName() + "," + std::to_string(N) + ">"; }
   explicit RField(std::string_view name) : RArrayField(name, std::make_unique<RField<ItemT>>("_0"), N) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

template <typename ItemT, std::size_t N>
class RField<ItemT[N]> final : public RField<std::array<ItemT, N>> {
public:
   explicit RField(std::string_view name) : RField<std::array<ItemT, N>>(name) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

template <typename... ItemTs>
class RField<std::variant<ItemTs...>> final : public RVariantField {
   using ContainerT = typename std::variant<ItemTs...>;

private:
   template <typename HeadT, typename... TailTs>
   static std::string BuildItemTypes()
   {
      std::string result = RField<HeadT>::TypeName();
      if constexpr (sizeof...(TailTs) > 0)
         result += "," + BuildItemTypes<TailTs...>();
      return result;
   }

   template <typename HeadT, typename... TailTs>
   static std::vector<RFieldBase *> BuildItemFields(unsigned int index = 0)
   {
      std::vector<RFieldBase *> result;
      result.emplace_back(new RField<HeadT>("_" + std::to_string(index)));
      if constexpr (sizeof...(TailTs) > 0) {
         auto tailFields = BuildItemFields<TailTs...>(index + 1);
         result.insert(result.end(), tailFields.begin(), tailFields.end());
      }
      return result;
   }

protected:
   void ConstructValue(void *where) const final { new (where) ContainerT(); }

public:
   static std::string TypeName() { return "std::variant<" + BuildItemTypes<ItemTs...>() + ">"; }
   explicit RField(std::string_view name) : RVariantField(name, BuildItemFields<ItemTs...>()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

template <typename ItemT>
class RField<std::vector<ItemT>> final : public RVectorField {
   using ContainerT = typename std::vector<ItemT>;

protected:
   void ConstructValue(void *where) const final { new (where) ContainerT(); }

public:
   static std::string TypeName() { return "std::vector<" + RField<ItemT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : RVectorField(name, std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   size_t GetValueSize() const final { return sizeof(ContainerT); }
};

// std::vector<bool> is a template specialization and needs special treatment
template <>
class RField<std::vector<bool>> final : public RFieldBase {
private:
   ClusterSize_t fNWritten{0};

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const final { new (where) std::vector<bool>(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<std::vector<bool>>>(); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

   void CommitClusterImpl() final { fNWritten = 0; }

public:
   static std::string TypeName() { return "std::vector<bool>"; }
   explicit RField(std::string_view name);
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;

   size_t GetValueSize() const final { return sizeof(std::vector<bool>); }
   size_t GetAlignment() const final { return std::alignment_of<std::vector<bool>>(); }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
   void GetCollectionInfo(NTupleSize_t globalIndex, RClusterIndex *collectionStart, ClusterSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(globalIndex, collectionStart, size);
   }
   void GetCollectionInfo(RClusterIndex clusterIndex, RClusterIndex *collectionStart, ClusterSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(clusterIndex, collectionStart, size);
   }
};

template <typename ItemT>
class RField<ROOT::VecOps::RVec<ItemT>> final : public RRVecField {
   using ContainerT = typename ROOT::VecOps::RVec<ItemT>;

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      auto newItemField = fSubFields[0]->Clone(fSubFields[0]->GetFieldName());
      return std::make_unique<RField<ROOT::VecOps::RVec<ItemT>>>(newName, std::move(newItemField));
   }

   void ConstructValue(void *where) const final { new (where) ContainerT(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<ContainerT>>(); }

   std::size_t AppendImpl(const void *from) final
   {
      auto typedValue = static_cast<const ContainerT *>(from);
      auto nbytes = 0;
      auto count = typedValue->size();
      for (unsigned i = 0; i < count; ++i) {
         nbytes += CallAppendOn(*fSubFields[0], &typedValue->data()[i]);
      }
      this->fNWritten += count;
      fPrincipalColumn->Append(&this->fNWritten);
      return nbytes + fPrincipalColumn->GetElement()->GetPackedSize();
   }
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final
   {
      auto typedValue = static_cast<ContainerT *>(to);
      ClusterSize_t nItems;
      RClusterIndex collectionStart;
      fPrincipalColumn->GetCollectionInfo(globalIndex, &collectionStart, &nItems);
      typedValue->resize(nItems);
      for (unsigned i = 0; i < nItems; ++i) {
         CallReadOn(*fSubFields[0], collectionStart + i, &typedValue->data()[i]);
      }
   }

public:
   RField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
      : RRVecField(fieldName, std::move(itemField))
   {
   }

   explicit RField(std::string_view name) : RField(name, std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;

   static std::string TypeName() { return "ROOT::VecOps::RVec<" + RField<ItemT>::TypeName() + ">"; }

   size_t GetValueSize() const final { return sizeof(ContainerT); }
   size_t GetAlignment() const final { return std::alignment_of<ContainerT>(); }
};

template <typename T1, typename T2>
class RField<std::pair<T1, T2>> final : public RPairField {
   using ContainerT = typename std::pair<T1, T2>;

private:
   template <typename Ty1, typename Ty2>
   static std::array<std::unique_ptr<RFieldBase>, 2> BuildItemFields()
   {
      return {std::make_unique<RField<Ty1>>("_0"), std::make_unique<RField<Ty2>>("_1")};
   }

   static std::array<std::size_t, 2> BuildItemOffsets()
   {
      auto pair = ContainerT();
      auto offsetFirst = reinterpret_cast<std::uintptr_t>(&(pair.first)) - reinterpret_cast<std::uintptr_t>(&pair);
      auto offsetSecond = reinterpret_cast<std::uintptr_t>(&(pair.second)) - reinterpret_cast<std::uintptr_t>(&pair);
      return {offsetFirst, offsetSecond};
   }

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      std::array<std::unique_ptr<RFieldBase>, 2> items{fSubFields[0]->Clone(fSubFields[0]->GetFieldName()),
                                                       fSubFields[1]->Clone(fSubFields[1]->GetFieldName())};
      return std::make_unique<RField<std::pair<T1, T2>>>(newName, std::move(items));
   }

   void ConstructValue(void *where) const final { new (where) ContainerT(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<ContainerT>>(); }

public:
   static std::string TypeName() { return "std::pair<" + RField<T1>::TypeName() + "," + RField<T2>::TypeName() + ">"; }
   explicit RField(std::string_view name, std::array<std::unique_ptr<RFieldBase>, 2> &&itemFields)
      : RPairField(name, std::move(itemFields), BuildItemOffsets())
   {
      fMaxAlignment = std::max(alignof(T1), alignof(T2));
      fSize = sizeof(ContainerT);
   }
   explicit RField(std::string_view name) : RField(name, BuildItemFields<T1, T2>()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

template <typename... ItemTs>
class RField<std::tuple<ItemTs...>> final : public RTupleField {
   using ContainerT = typename std::tuple<ItemTs...>;

private:
   template <typename HeadT, typename... TailTs>
   static std::string BuildItemTypes()
   {
      std::string result = RField<HeadT>::TypeName();
      if constexpr (sizeof...(TailTs) > 0)
         result += "," + BuildItemTypes<TailTs...>();
      return result;
   }

   template <typename HeadT, typename... TailTs>
   static void _BuildItemFields(std::vector<std::unique_ptr<RFieldBase>> &itemFields, unsigned int index = 0)
   {
      itemFields.emplace_back(new RField<HeadT>("_" + std::to_string(index)));
      if constexpr (sizeof...(TailTs) > 0)
         _BuildItemFields<TailTs...>(itemFields, index + 1);
   }
   template <typename... Ts>
   static std::vector<std::unique_ptr<RFieldBase>> BuildItemFields()
   {
      std::vector<std::unique_ptr<RFieldBase>> result;
      _BuildItemFields<Ts...>(result);
      return result;
   }

   template <unsigned Index, typename HeadT, typename... TailTs>
   static void _BuildItemOffsets(std::vector<std::size_t> &offsets, const ContainerT &tuple)
   {
      auto offset =
         reinterpret_cast<std::uintptr_t>(&std::get<Index>(tuple)) - reinterpret_cast<std::uintptr_t>(&tuple);
      offsets.emplace_back(offset);
      if constexpr (sizeof...(TailTs) > 0)
         _BuildItemOffsets<Index + 1, TailTs...>(offsets, tuple);
   }
   template <typename... Ts>
   static std::vector<std::size_t> BuildItemOffsets()
   {
      std::vector<std::size_t> result;
      _BuildItemOffsets<0, Ts...>(result, ContainerT());
      return result;
   }

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      std::vector<std::unique_ptr<RFieldBase>> items;
      for (auto &item : fSubFields)
         items.push_back(item->Clone(item->GetFieldName()));
      return std::make_unique<RField<std::tuple<ItemTs...>>>(newName, std::move(items));
   }

   void ConstructValue(void *where) const final { new (where) ContainerT(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<ContainerT>>(); }

public:
   static std::string TypeName() { return "std::tuple<" + BuildItemTypes<ItemTs...>() + ">"; }
   explicit RField(std::string_view name, std::vector<std::unique_ptr<RFieldBase>> &&itemFields)
      : RTupleField(name, std::move(itemFields), BuildItemOffsets<ItemTs...>())
   {
      fMaxAlignment = std::max({alignof(ItemTs)...});
      fSize = sizeof(ContainerT);
   }
   explicit RField(std::string_view name) : RField(name, BuildItemFields<ItemTs...>()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

template <std::size_t N>
class RField<std::bitset<N>> final : public RBitsetField {
public:
   static std::string TypeName() { return "std::bitset<" + std::to_string(N) + ">"; }
   explicit RField(std::string_view name) : RBitsetField(name, N) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

template <typename ItemT>
class RField<std::unique_ptr<ItemT>> final : public RUniquePtrField {
public:
   static std::string TypeName() { return "std::unique_ptr<" + RField<ItemT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : RUniquePtrField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

template <typename ItemT>
class RField<std::optional<ItemT>> final : public ROptionalField {
public:
   static std::string TypeName() { return "std::optional<" + RField<ItemT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : ROptionalField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
};

template <typename ItemT>
class RField<std::atomic<ItemT>> final : public RAtomicField {
public:
   static std::string TypeName() { return "std::atomic<" + RField<ItemT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : RAtomicField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() override = default;
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
