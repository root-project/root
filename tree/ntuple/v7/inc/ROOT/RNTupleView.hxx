/// \file ROOT/RNTupleView.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-05
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTupleView
#define ROOT7_RNTupleView

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <string_view>

#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <unordered_map>

namespace ROOT {
namespace Experimental {


// clang-format off
/**
\class ROOT::Experimental::RNTupleGlobalRange
\ingroup NTuple
\brief Used to loop over indexes (entries or collections) between start and end
*/
// clang-format on
class RNTupleGlobalRange {
private:
   NTupleSize_t fStart;
   NTupleSize_t fEnd;

public:
   class RIterator {
   private:
      NTupleSize_t fIndex = kInvalidNTupleIndex;
   public:
      using iterator = RIterator;
      using iterator_category = std::forward_iterator_tag;
      using value_type = NTupleSize_t;
      using difference_type = NTupleSize_t;
      using pointer = NTupleSize_t*;
      using reference = NTupleSize_t&;

      RIterator() = default;
      explicit RIterator(NTupleSize_t index) : fIndex(index) {}
      ~RIterator() = default;

      iterator  operator++(int) /* postfix */        { auto r = *this; fIndex++; return r; }
      iterator& operator++()    /* prefix */         { ++fIndex; return *this; }
      reference operator* ()                         { return fIndex; }
      pointer   operator->()                         { return &fIndex; }
      bool      operator==(const iterator& rh) const { return fIndex == rh.fIndex; }
      bool      operator!=(const iterator& rh) const { return fIndex != rh.fIndex; }
   };

   RNTupleGlobalRange(NTupleSize_t start, NTupleSize_t end) : fStart(start), fEnd(end) {}
   RIterator begin() { return RIterator(fStart); }
   RIterator end() { return RIterator(fEnd); }
   NTupleSize_t size() { return fEnd - fStart; }
   bool IsValid() const { return (fStart != kInvalidNTupleIndex) && (fEnd != kInvalidNTupleIndex); }
};


// clang-format off
/**
\class ROOT::Experimental::RNTupleClusterRange
\ingroup NTuple
\brief Used to loop over entries of collections in a single cluster
*/
// clang-format on
class RNTupleClusterRange {
private:
   const DescriptorId_t fClusterId;
   const ClusterSize_t::ValueType fStart;
   const ClusterSize_t::ValueType fEnd;
public:
   class RIterator {
   private:
      RClusterIndex fIndex;
   public:
      using iterator = RIterator;
      using iterator_category = std::forward_iterator_tag;
      using value_type = RClusterIndex;
      using difference_type = RClusterIndex;
      using pointer = RClusterIndex*;
      using reference = RClusterIndex&;

      RIterator() = default;
      explicit RIterator(RClusterIndex index) : fIndex(index) {}
      ~RIterator() = default;

      iterator  operator++(int) /* postfix */        { auto r = *this; fIndex++; return r; }
      iterator& operator++()    /* prefix */         { fIndex++; return *this; }
      reference operator* ()                         { return fIndex; }
      pointer   operator->()                         { return &fIndex; }
      bool      operator==(const iterator& rh) const { return fIndex == rh.fIndex; }
      bool      operator!=(const iterator& rh) const { return fIndex != rh.fIndex; }
   };

   RNTupleClusterRange(DescriptorId_t clusterId, ClusterSize_t::ValueType start, ClusterSize_t::ValueType end)
      : fClusterId(clusterId), fStart(start), fEnd(end) {}
   RIterator begin() { return RIterator(RClusterIndex(fClusterId, fStart)); }
   RIterator end() { return RIterator(RClusterIndex(fClusterId, fEnd)); }
};

namespace Internal {

/// Helper to get the iteration space of the given field that needs to be connected to the given page source.
/// The indexes are given by the number of elements of the principal column of the field or, if none exists,
/// by the number of elements of the first principal column found in the subfields searched by BFS.
/// If the field hierarchy is empty on columns, the returned field range is invalid (start and end set to
/// kInvalidNTupleIndex). An attempt to use such a field range in RNTupleViewBase::GetFieldRange will throw.
RNTupleGlobalRange GetFieldRange(const RFieldBase &field, const RPageSource &pageSource);

} // namespace Internal

// clang-format off
/**
\class ROOT::Experimental::RNTupleViewBase
\ingroup NTuple
\brief An RNTupleView provides read-only access to a single field of the ntuple

\tparam T The type of the object that will be read by the view; can be void if unknown at compile time.

The view owns a field and its underlying columns in order to fill an RField::RValue object with data. Data can be
accessed by index. For top-level fields, the index refers to the entry number. Fields that are part of
nested collections have global index numbers that are derived from their parent indexes.

View can only be created by a reader or by a collection view.
*/
// clang-format on
template <typename T>
class RNTupleViewBase {
protected:
   std::unique_ptr<RFieldBase> fField;
   RNTupleGlobalRange fFieldRange;
   RFieldBase::RValue fValue;

   static std::unique_ptr<RFieldBase> CreateField(DescriptorId_t fieldId, Internal::RPageSource &pageSource)
   {
      std::unique_ptr<RFieldBase> field;
      {
         const auto &desc = pageSource.GetSharedDescriptorGuard().GetRef();
         const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);
         if constexpr (std::is_void_v<T>) {
            field = fieldDesc.CreateField(desc);
         } else {
            field = std::make_unique<RField<T>>(fieldDesc.GetFieldName());
         }
      }
      field->SetOnDiskId(fieldId);
      Internal::CallConnectPageSourceOnField(*field, pageSource);
      return field;
   }

   RNTupleViewBase(std::unique_ptr<RFieldBase> field, RNTupleGlobalRange range)
      : fField(std::move(field)), fFieldRange(range), fValue(fField->CreateValue())
   {
   }

   RNTupleViewBase(std::unique_ptr<RFieldBase> field, RNTupleGlobalRange range, std::shared_ptr<T> objPtr)
      : fField(std::move(field)), fFieldRange(range), fValue(fField->BindValue(objPtr))
   {
   }

   RNTupleViewBase(std::unique_ptr<RFieldBase> field, RNTupleGlobalRange range, T *rawPtr)
      : fField(std::move(field)), fFieldRange(range), fValue(fField->BindValue(Internal::MakeAliasedSharedPtr(rawPtr)))
   {
   }

public:
   RNTupleViewBase(const RNTupleViewBase &other) = delete;
   RNTupleViewBase(RNTupleViewBase &&other) = default;
   RNTupleViewBase &operator=(const RNTupleViewBase &other) = delete;
   RNTupleViewBase &operator=(RNTupleViewBase &&other) = default;
   ~RNTupleViewBase() = default;

   const RFieldBase &GetField() const { return *fField; }
   const RFieldBase::RValue &GetValue() const { return fValue; }
   RNTupleGlobalRange GetFieldRange() const
   {
      if (!fFieldRange.IsValid()) {
         throw RException(R__FAIL("field iteration over empty fields is unsupported: " + fField->GetFieldName()));
      }
      return fFieldRange;
   }

   void Bind(std::shared_ptr<T> objPtr) { fValue.Bind(objPtr); }
   void BindRawPtr(T *rawPtr) { fValue.BindRawPtr(rawPtr); }
   void EmplaceNew() { fValue.EmplaceNew(); }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleView
\ingroup NTuple
\brief An RNTupleView for a known type. See RNTupleViewBase.
*/
// clang-format on
template <typename T>
class RNTupleView : public RNTupleViewBase<T> {
   friend class RNTupleReader;
   friend class RNTupleCollectionView;

protected:
   RNTupleView(std::unique_ptr<RFieldBase> field, RNTupleGlobalRange range)
      : RNTupleViewBase<T>(std::move(field), range)
   {
   }

   RNTupleView(std::unique_ptr<RFieldBase> field, RNTupleGlobalRange range, std::shared_ptr<T> objPtr)
      : RNTupleViewBase<T>(std::move(field), range, objPtr)
   {
   }

   RNTupleView(std::unique_ptr<RFieldBase> field, RNTupleGlobalRange range, T *rawPtr)
      : RNTupleViewBase<T>(std::move(field), range, rawPtr)
   {
   }

public:
   RNTupleView(const RNTupleView &other) = delete;
   RNTupleView(RNTupleView &&other) = default;
   RNTupleView &operator=(const RNTupleView &other) = delete;
   RNTupleView &operator=(RNTupleView &&other) = default;
   ~RNTupleView() = default;

   const T &operator()(NTupleSize_t globalIndex)
   {
      RNTupleViewBase<T>::fValue.Read(globalIndex);
      return RNTupleViewBase<T>::fValue.template GetRef<T>();
   }

   const T &operator()(RClusterIndex clusterIndex)
   {
      RNTupleViewBase<T>::fValue.Read(clusterIndex);
      return RNTupleViewBase<T>::fValue.template GetRef<T>();
   }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleView
\ingroup NTuple
\brief An RNTupleView that can be used when the type is unknown at compile time. See RNTupleViewBase.
*/
// clang-format on
template <>
class RNTupleView<void> final : public RNTupleViewBase<void> {
   friend class RNTupleReader;
   friend class RNTupleCollectionView;

protected:
   RNTupleView(std::unique_ptr<RFieldBase> field, RNTupleGlobalRange range)
      : RNTupleViewBase<void>(std::move(field), range)
   {
   }

   RNTupleView(std::unique_ptr<RFieldBase> field, RNTupleGlobalRange range, std::shared_ptr<void> objPtr)
      : RNTupleViewBase<void>(std::move(field), range, objPtr)
   {
   }

   RNTupleView(std::unique_ptr<RFieldBase> field, RNTupleGlobalRange range, void *rawPtr)
      : RNTupleViewBase<void>(std::move(field), range, rawPtr)
   {
   }

public:
   RNTupleView(const RNTupleView &other) = delete;
   RNTupleView(RNTupleView &&other) = default;
   RNTupleView &operator=(const RNTupleView &other) = delete;
   RNTupleView &operator=(RNTupleView &&other) = default;
   ~RNTupleView() = default;

   void operator()(NTupleSize_t globalIndex) { fValue.Read(globalIndex); }
   void operator()(RClusterIndex clusterIndex) { fValue.Read(clusterIndex); }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleDirectAccessView
\ingroup NTuple
\brief A view variant that provides direct access to the I/O buffers. Only works for mappable fields.
*/
// clang-format on
template <typename T>
class RNTupleDirectAccessView {
   friend class RNTupleReader;
   friend class RNTupleCollectionView;

protected:
   RField<T> fField;
   RNTupleGlobalRange fFieldRange;

   static RField<T> CreateField(DescriptorId_t fieldId, Internal::RPageSource &pageSource)
   {
      const auto &desc = pageSource.GetSharedDescriptorGuard().GetRef();
      const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);
      if (fieldDesc.GetTypeName() != RField<T>::TypeName()) {
         throw RException(R__FAIL("type mismatch for field " + fieldDesc.GetFieldName() + ": " +
                                  fieldDesc.GetTypeName() + " vs. " + RField<T>::TypeName()));
      }
      RField<T> field(fieldDesc.GetFieldName());
      field.SetOnDiskId(fieldId);
      Internal::CallConnectPageSourceOnField(field, pageSource);
      return field;
   }

   RNTupleDirectAccessView(RField<T> field, RNTupleGlobalRange range) : fField(std::move(field)), fFieldRange(range) {}

public:
   RNTupleDirectAccessView(const RNTupleDirectAccessView &other) = delete;
   RNTupleDirectAccessView(RNTupleDirectAccessView &&other) = default;
   RNTupleDirectAccessView &operator=(const RNTupleDirectAccessView &other) = delete;
   RNTupleDirectAccessView &operator=(RNTupleDirectAccessView &&other) = default;
   ~RNTupleDirectAccessView() = default;

   const RFieldBase &GetField() const { return fField; }
   RNTupleGlobalRange GetFieldRange() const { return fFieldRange; }

   const T &operator()(NTupleSize_t globalIndex) { return *fField.Map(globalIndex); }
   const T &operator()(RClusterIndex clusterIndex) { return *fField.Map(clusterIndex); }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleCollectionView
\ingroup NTuple
\brief A view for a collection, that can itself generate new ntuple views for its nested fields.
*/
// clang-format on
class RNTupleCollectionView {
   friend class RNTupleReader;

private:
   Internal::RPageSource *fSource;
   RField<RNTupleCardinality<std::uint64_t>> fField;
   RFieldBase::RValue fValue;

   RNTupleCollectionView(DescriptorId_t fieldId, const std::string &fieldName, Internal::RPageSource *source)
      : fSource(source), fField(fieldName), fValue(fField.CreateValue())
   {
      fField.SetOnDiskId(fieldId);
      Internal::CallConnectPageSourceOnField(fField, *source);
   }

   static RNTupleCollectionView Create(DescriptorId_t fieldId, Internal::RPageSource *source)
   {
      std::string fieldName;
      {
         const auto &desc = source->GetSharedDescriptorGuard().GetRef();
         const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);
         if (fieldDesc.GetStructure() != ENTupleStructure::kCollection) {
            throw RException(
               R__FAIL("invalid attemt to create collection view on non-collection field " + fieldDesc.GetFieldName()));
         }
         fieldName = fieldDesc.GetFieldName();
      }
      return RNTupleCollectionView(fieldId, fieldName, source);
   }

   DescriptorId_t GetFieldId(std::string_view fieldName)
   {
      auto descGuard = fSource->GetSharedDescriptorGuard();
      auto fieldId = descGuard->FindFieldId(fieldName, fField.GetOnDiskId());
      if (fieldId == kInvalidDescriptorId) {
         throw RException(R__FAIL("no field named '" + std::string(fieldName) + "' in collection '" +
                                  descGuard->GetQualifiedFieldName(fField.GetOnDiskId()) + "'"));
      }
      return fieldId;
   }

public:
   RNTupleCollectionView(const RNTupleCollectionView &other) = delete;
   RNTupleCollectionView(RNTupleCollectionView &&other) = default;
   RNTupleCollectionView &operator=(const RNTupleCollectionView &other) = delete;
   RNTupleCollectionView &operator=(RNTupleCollectionView &&other) = default;
   ~RNTupleCollectionView() = default;

   RNTupleClusterRange GetCollectionRange(NTupleSize_t globalIndex) {
      ClusterSize_t size;
      RClusterIndex collectionStart;
      fField.GetCollectionInfo(globalIndex, &collectionStart, &size);
      return RNTupleClusterRange(collectionStart.GetClusterId(), collectionStart.GetIndex(),
                                 collectionStart.GetIndex() + size);
   }
   RNTupleClusterRange GetCollectionRange(RClusterIndex clusterIndex)
   {
      ClusterSize_t size;
      RClusterIndex collectionStart;
      fField.GetCollectionInfo(clusterIndex, &collectionStart, &size);
      return RNTupleClusterRange(collectionStart.GetClusterId(), collectionStart.GetIndex(),
                                 collectionStart.GetIndex() + size);
   }

   /// Raises an exception if there is no field with the given name.
   template <typename T>
   RNTupleView<T> GetView(std::string_view fieldName)
   {
      auto field = RNTupleView<T>::CreateField(GetFieldId(fieldName), *fSource);
      auto range = Internal::GetFieldRange(*field, *fSource);
      return RNTupleView<T>(std::move(field), range);
   }

   /// Raises an exception if there is no field with the given name.
   template <typename T>
   RNTupleDirectAccessView<T> GetDirectAccessView(std::string_view fieldName)
   {
      auto field = RNTupleDirectAccessView<T>::CreateField(GetFieldId(fieldName), *fSource);
      auto range = Internal::GetFieldRange(field, *fSource);
      return RNTupleDirectAccessView<T>(std::move(field), range);
   }

   /// Raises an exception if there is no field with the given name.
   RNTupleCollectionView GetCollectionView(std::string_view fieldName)
   {
      return RNTupleCollectionView::Create(GetFieldId(fieldName), fSource);
   }

   std::uint64_t operator()(NTupleSize_t globalIndex)
   {
      fValue.Read(globalIndex);
      return fValue.GetRef<std::uint64_t>();
   }

   std::uint64_t operator()(RClusterIndex clusterIndex)
   {
      fValue.Read(clusterIndex);
      return fValue.GetRef<std::uint64_t>();
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
