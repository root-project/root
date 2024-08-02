/// \file ROOT/RField/SequenceContainer.hxx
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

#ifndef ROOT7_RField_SequenceContainer
#define ROOT7_RField_SequenceContainer

#ifndef ROOT7_RField
#error "Please include RField.hxx!"
#endif

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RVec.hxx>

#include <array>
#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {

namespace Detail {
class RFieldVisitor;
} // namespace Detail

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

////////////////////////////////////////////////////////////////////////////////
/// Template specializations for C++ std::vector
////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////
/// Additional classes related to sequence containers
////////////////////////////////////////////////////////////////////////////////

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

} // namespace Experimental
} // namespace ROOT

#endif
