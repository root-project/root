/// \file ROOT/RField/STLMisc.hxx
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

#ifndef ROOT7_RField_STLMisc
#define ROOT7_RField_STLMisc

#ifndef ROOT7_RField
#error "Please include RField.hxx!"
#endif

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleUtil.hxx>

#include <atomic>
#include <bitset>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace ROOT {
namespace Experimental {

namespace Detail {
class RFieldVisitor;
} // namespace Detail

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::atomic
////////////////////////////////////////////////////////////////////////////////

class RAtomicField : public RFieldBase {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void ConstructValue(void *where) const final { CallConstructValueOn(*fSubFields[0], where); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return GetDeleterOf(*fSubFields[0]); }

   std::size_t AppendImpl(const void *from) final { return CallAppendOn(*fSubFields[0], from); }
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final { CallReadOn(*fSubFields[0], globalIndex, to); }
   void ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to) final { CallReadOn(*fSubFields[0], localIndex, to); }

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

template <typename ItemT>
class RField<std::atomic<ItemT>> final : public RAtomicField {
public:
   static std::string TypeName() { return "std::atomic<" + RField<ItemT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : RAtomicField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::bitset
////////////////////////////////////////////////////////////////////////////////

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
   void ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to) final;

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

template <std::size_t N>
class RField<std::bitset<N>> final : public RBitsetField {
public:
   static std::string TypeName() { return "std::bitset<" + std::to_string(N) + ">"; }
   explicit RField(std::string_view name) : RBitsetField(name, N) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::byte
////////////////////////////////////////////////////////////////////////////////

extern template class RSimpleField<std::byte>;

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
   ~RField() final = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::optional and std::unique_ptr
////////////////////////////////////////////////////////////////////////////////

/// The field for values that may or may not be present in an entry. Parent class for unique pointer field and
/// optional field. A nullable field cannot be instantiated itself but only its descendants.
/// The RNullableField takes care of the on-disk representation. Child classes are responsible for the in-memory
/// representation.  Nullable fields use a (Split)Index[64|32] column to point to the available items.
class RNullableField : public RFieldBase {
   /// The number of written non-null items in this cluster
   Internal::RColumnIndex fNWritten{0};

protected:
   const RFieldBase::RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &) final;

   std::size_t AppendNull();
   std::size_t AppendValue(const void *from);
   void CommitClusterImpl() final { fNWritten = 0; }

   /// Given the index of the nullable field, returns the corresponding global index of the subfield or,
   /// if it is null, returns kInvalidNTupleIndex
   RNTupleLocalIndex GetItemIndex(NTupleSize_t globalIndex);

   RNullableField(std::string_view fieldName, std::string_view typeName, std::unique_ptr<RFieldBase> itemField);

public:
   RNullableField(RNullableField &&other) = default;
   RNullableField &operator=(RNullableField &&other) = default;
   ~RNullableField() override = default;

   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

class ROptionalField : public RNullableField {
   class ROptionalDeleter : public RDeleter {
   private:
      std::unique_ptr<RDeleter> fItemDeleter; // nullptr for trivially destructible items
      std::size_t fEngagementPtrOffset = 0;

   public:
      ROptionalDeleter(std::unique_ptr<RDeleter> itemDeleter, std::size_t engagementPtrOffset)
         : fItemDeleter(std::move(itemDeleter)), fEngagementPtrOffset(engagementPtrOffset) {}
      void operator()(void *objPtr, bool dtorOnly) final;
   };

   std::unique_ptr<RDeleter> fItemDeleter;

   /// Given a pointer to an std::optional<T> in `optionalPtr`, extract a pointer to the engagement boolean.
   /// Assumes that an std::optional<T> is stored as `struct { T t; bool engagement; };`
   const bool *GetEngagementPtr(const void *optionalPtr) const;
   bool *GetEngagementPtr(void *optionalPtr) const;
   std::size_t GetEngagementPtrOffset() const;

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

template <typename ItemT>
class RField<std::optional<ItemT>> final : public ROptionalField {
public:
   static std::string TypeName() { return "std::optional<" + RField<ItemT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : ROptionalField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
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

template <typename ItemT>
class RField<std::unique_ptr<ItemT>> final : public RUniquePtrField {
public:
   static std::string TypeName() { return "std::unique_ptr<" + RField<ItemT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : RUniquePtrField(name, TypeName(), std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::string
////////////////////////////////////////////////////////////////////////////////

template <>
class RField<std::string> final : public RFieldBase {
private:
   Internal::RColumnIndex fIndex;

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
   ~RField() final = default;

   size_t GetValueSize() const final { return sizeof(std::string); }
   size_t GetAlignment() const final { return std::alignment_of<std::string>(); }
   void AcceptVisitor(Detail::RFieldVisitor &visitor) const final;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::variant
////////////////////////////////////////////////////////////////////////////////

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
                      std::vector<std::unique_ptr<RDeleter>> itemDeleters)
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
   std::vector<Internal::RColumnIndex::ValueType> fNWritten;

   static std::string GetTypeList(const std::vector<std::unique_ptr<RFieldBase>> &itemFields);
   /// Extracts the index from an std::variant and transforms it into the 1-based index used for the switch column
   /// The implementation supports two memory layouts that are in use: a trailing unsigned byte, zero-indexed,
   /// having the exception caused empty state encoded by the max tag value,
   /// or a trailing unsigned int instead of a char.
   static std::uint8_t GetTag(const void *variantPtr, std::size_t tagOffset);
   static void SetTag(void *variantPtr, std::size_t tagOffset, std::uint8_t tag);

   RVariantField(std::string_view name, const RVariantField &source); // Used by CloneImpl()

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(NTupleSize_t globalIndex, void *to) final;

   void CommitClusterImpl() final;

public:
   RVariantField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields);
   RVariantField(RVariantField &&other) = default;
   RVariantField &operator=(RVariantField &&other) = default;
   ~RVariantField() override = default;

   size_t GetValueSize() const final;
   size_t GetAlignment() const final;
};

template <typename... ItemTs>
class RField<std::variant<ItemTs...>> final : public RVariantField {
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
   static std::vector<std::unique_ptr<RFieldBase>> BuildItemFields()
   {
      std::vector<std::unique_ptr<RFieldBase>> result;
      _BuildItemFields<ItemTs...>(result);
      return result;
   }

public:
   static std::string TypeName() { return "std::variant<" + BuildItemTypes<ItemTs...>() + ">"; }
   explicit RField(std::string_view name) : RVariantField(name, BuildItemFields()) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

} // namespace Experimental
} // namespace ROOT

#endif
