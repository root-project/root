/// \file ROOT/RField/SequenceContainer.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-09

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RField_SequenceContainer
#define ROOT_RField_SequenceContainer

#ifndef ROOT_RField
#error "Please include RField.hxx!"
#endif

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RVec.hxx>

#include <array>
#include <memory>
#include <vector>

namespace ROOT {

namespace Detail {
class RFieldVisitor;
} // namespace Detail

namespace Internal {
std::unique_ptr<RFieldBase> CreateEmulatedVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField,
                                                      std::string_view emulatedFromType);
}

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::array and C-style arrays
////////////////////////////////////////////////////////////////////////////////

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

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;
   void ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to) final;

   void ReconcileOnDiskField(const RNTupleDescriptor &desc) final;

public:
   RArrayField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField, std::size_t arrayLength);
   RArrayField(RArrayField &&other) = default;
   RArrayField &operator=(RArrayField &&other) = default;
   ~RArrayField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetLength() const { return fArrayLength; }
   size_t GetValueSize() const final { return fItemSize * fArrayLength; }
   size_t GetAlignment() const final { return fSubfields[0]->GetAlignment(); }
   void AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const final;
};

template <typename ItemT, std::size_t N>
class RField<std::array<ItemT, N>> : public RArrayField {
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
   ~RField() final = default;
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for ROOT's RVec
////////////////////////////////////////////////////////////////////////////////

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
   ROOT::Internal::RColumnIndex fNWritten;
   std::size_t fValueSize;

   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;
   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const ROOT::RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const final;
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;
   std::size_t ReadBulkImpl(const RBulkSpec &bulkSpec) final;

   void ReconcileOnDiskField(const RNTupleDescriptor &desc) final;

   void CommitClusterImpl() final { fNWritten = 0; }

public:
   RRVecField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField);
   RRVecField(RRVecField &&) = default;
   RRVecField &operator=(RRVecField &&) = default;
   RRVecField(const RRVecField &) = delete;
   RRVecField &operator=(RRVecField &) = delete;
   ~RRVecField() override = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final;
   size_t GetAlignment() const final;
   void AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const final;
   void
   GetCollectionInfo(ROOT::NTupleSize_t globalIndex, RNTupleLocalIndex *collectionStart, ROOT::NTupleSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(globalIndex, collectionStart, size);
   }
   void
   GetCollectionInfo(RNTupleLocalIndex localIndex, RNTupleLocalIndex *collectionStart, ROOT::NTupleSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(localIndex, collectionStart, size);
   }
};

template <typename ItemT>
class RField<ROOT::VecOps::RVec<ItemT>> final : public RRVecField {
public:
   RField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField)
      : RRVecField(fieldName, std::move(itemField))
   {
   }

   explicit RField(std::string_view name) : RField(name, std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;

   static std::string TypeName() { return "ROOT::VecOps::RVec<" + RField<ItemT>::TypeName() + ">"; }
};

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::vector
////////////////////////////////////////////////////////////////////////////////

/// The generic field for a (nested) `std::vector<Type>` except for `std::vector<bool>`
/// The field can be constructed as untyped collection through CreateUntyped().
class RVectorField : public RFieldBase {
   friend std::unique_ptr<RFieldBase> Internal::CreateEmulatedVectorField(std::string_view fieldName,
                                                                          std::unique_ptr<RFieldBase> itemField,
                                                                          std::string_view emulatedFromType);

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
   ROOT::Internal::RColumnIndex fNWritten;
   std::unique_ptr<RDeleter> fItemDeleter;

protected:
   /// Creates a possibly-untyped VectorField.
   /// If `emulatedFromType` is not nullopt, the field is untyped. If the string is empty, it is a "regular"
   /// untyped vector field; otherwise, it was created as an emulated field from the given type name.
   RVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField,
                std::optional<std::string_view> emulatedFromType);

   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const ROOT::RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const final { new (where) std::vector<char>(); }
   std::unique_ptr<RDeleter> GetDeleter() const final;

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;

   void ReconcileOnDiskField(const RNTupleDescriptor &desc) final;

   void CommitClusterImpl() final { fNWritten = 0; }

public:
   RVectorField(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField);
   RVectorField(RVectorField &&other) = default;
   RVectorField &operator=(RVectorField &&other) = default;
   ~RVectorField() override = default;

   static std::unique_ptr<RVectorField>
   CreateUntyped(std::string_view fieldName, std::unique_ptr<RFieldBase> itemField);

   std::vector<RValue> SplitValue(const RValue &value) const final;
   size_t GetValueSize() const final { return sizeof(std::vector<char>); }
   size_t GetAlignment() const final { return std::alignment_of<std::vector<char>>(); }
   void AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const final;
   void
   GetCollectionInfo(ROOT::NTupleSize_t globalIndex, RNTupleLocalIndex *collectionStart, ROOT::NTupleSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(globalIndex, collectionStart, size);
   }
   void
   GetCollectionInfo(RNTupleLocalIndex localIndex, RNTupleLocalIndex *collectionStart, ROOT::NTupleSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(localIndex, collectionStart, size);
   }
};

template <typename ItemT>
class RField<std::vector<ItemT>> final : public RVectorField {
public:
   static std::string TypeName() { return "std::vector<" + RField<ItemT>::TypeName() + ">"; }
   explicit RField(std::string_view name) : RVectorField(name, std::make_unique<RField<ItemT>>("_0")) {}
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;
};

// `std::vector<bool>` is a template specialization and needs special treatment
template <>
class RField<std::vector<bool>> final : public RFieldBase {
private:
   ROOT::Internal::RColumnIndex fNWritten{0};

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RField>(newName);
   }

   const RColumnRepresentations &GetColumnRepresentations() const final;
   void GenerateColumns() final;
   void GenerateColumns(const ROOT::RNTupleDescriptor &desc) final;

   void ConstructValue(void *where) const final { new (where) std::vector<bool>(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<std::vector<bool>>>(); }

   std::size_t AppendImpl(const void *from) final;
   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;

   void ReconcileOnDiskField(const RNTupleDescriptor &desc) final;

   void CommitClusterImpl() final { fNWritten = 0; }

public:
   static std::string TypeName() { return "std::vector<bool>"; }
   explicit RField(std::string_view name);
   RField(RField &&other) = default;
   RField &operator=(RField &&other) = default;
   ~RField() final = default;

   std::vector<RValue> SplitValue(const RValue &value) const final;

   size_t GetValueSize() const final { return sizeof(std::vector<bool>); }
   size_t GetAlignment() const final { return std::alignment_of<std::vector<bool>>(); }
   void AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const final;
   void
   GetCollectionInfo(ROOT::NTupleSize_t globalIndex, RNTupleLocalIndex *collectionStart, ROOT::NTupleSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(globalIndex, collectionStart, size);
   }
   void
   GetCollectionInfo(RNTupleLocalIndex localIndex, RNTupleLocalIndex *collectionStart, ROOT::NTupleSize_t *size) const
   {
      fPrincipalColumn->GetCollectionInfo(localIndex, collectionStart, size);
   }
};

////////////////////////////////////////////////////////////////////////////////
/// Additional classes related to sequence containers
////////////////////////////////////////////////////////////////////////////////

/**
\class ROOT::RArrayAsRVecField
\brief A field for fixed-size arrays that are represented as RVecs in memory.
\ingroup NTuple
This class is used only for reading. In particular, it helps exposing
arbitrarily-nested `std::array` on-disk fields as RVecs for usage in RDataFrame.
*/
class RArrayAsRVecField final : public RFieldBase {
private:
   std::unique_ptr<RDeleter> fItemDeleter; /// Sub field deleter or nullptr for simple fields
   std::size_t fItemSize;                  /// The size of a child field's item
   std::size_t fArrayLength;               /// The length of the arrays in this field
   std::size_t fValueSize;                 /// The size of a value of this field, i.e. an RVec

protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final;

   void GenerateColumns() final { throw RException(R__FAIL("RArrayAsRVec fields must only be used for reading")); }
   using RFieldBase::GenerateColumns;

   void ConstructValue(void *where) const final;
   /// Returns an RRVecField::RRVecDeleter
   std::unique_ptr<RDeleter> GetDeleter() const final;

   void ReadGlobalImpl(ROOT::NTupleSize_t globalIndex, void *to) final;
   void ReadInClusterImpl(RNTupleLocalIndex localIndex, void *to) final;

   void ReconcileOnDiskField(const RNTupleDescriptor &desc) final;

public:
   /**
      Constructor of the field. The `itemField` argument represents the inner
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
   void AcceptVisitor(ROOT::Detail::RFieldVisitor &visitor) const final;
};

} // namespace ROOT

#endif
