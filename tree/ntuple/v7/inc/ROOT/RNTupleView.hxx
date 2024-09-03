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
   const NTupleSize_t fStart;
   const NTupleSize_t fEnd;

public:
   class RIterator {
   private:
      NTupleSize_t fIndex = kInvalidNTupleIndex;

   public:
      using iterator = RIterator;
      using iterator_category = std::forward_iterator_tag;
      using value_type = NTupleSize_t;
      using difference_type = NTupleSize_t;
      using pointer = NTupleSize_t *;
      using reference = NTupleSize_t &;

      RIterator() = default;
      explicit RIterator(NTupleSize_t index) : fIndex(index) {}
      ~RIterator() = default;

      iterator operator++(int) /* postfix */
      {
         auto r = *this;
         fIndex++;
         return r;
      }
      iterator &operator++() /* prefix */
      {
         ++fIndex;
         return *this;
      }
      reference operator*() { return fIndex; }
      pointer operator->() { return &fIndex; }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
   };

   RNTupleGlobalRange(NTupleSize_t start, NTupleSize_t end) : fStart(start), fEnd(end) {}
   RIterator begin() { return RIterator(fStart); }
   RIterator end() { return RIterator(fEnd); }
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
      using pointer = RClusterIndex *;
      using reference = RClusterIndex &;

      RIterator() = default;
      explicit RIterator(RClusterIndex index) : fIndex(index) {}
      ~RIterator() = default;

      iterator operator++(int) /* postfix */
      {
         auto r = *this;
         fIndex++;
         return r;
      }
      iterator &operator++() /* prefix */
      {
         fIndex++;
         return *this;
      }
      reference operator*() { return fIndex; }
      pointer operator->() { return &fIndex; }
      bool operator==(const iterator &rh) const { return fIndex == rh.fIndex; }
      bool operator!=(const iterator &rh) const { return fIndex != rh.fIndex; }
   };

   RNTupleClusterRange(DescriptorId_t clusterId, ClusterSize_t::ValueType start, ClusterSize_t::ValueType end)
      : fClusterId(clusterId), fStart(start), fEnd(end)
   {
   }
   RIterator begin() { return RIterator(RClusterIndex(fClusterId, fStart)); }
   RIterator end() { return RIterator(RClusterIndex(fClusterId, fEnd)); }
};

namespace Internal {
// TODO(bgruber): convert this trait into a requires clause in C++20
template <typename FieldT, typename SFINAE = void>
inline constexpr bool isMappable = false;

template <typename FieldT>
inline constexpr bool isMappable<FieldT, std::void_t<decltype(std::declval<FieldT>().Map(NTupleSize_t{}))>> = true;

// clang-format off
/**
\class ROOT::Experimental::RNTupleViewBase
\ingroup NTuple
\brief An RNTupleViewBase provides read-only access to a single field of the ntuple. This is an internal
       class: users are expected to use RNTupleUnownedView/RNTupleOwnedView.

\tparam T The type of the object that will be read by the view
\tparam UserProvidedAddress Whether the user provided an external memory location to read data into

The view owns a field and its underlying columns in order to fill an ntuple value object with data. Data can be
accessed by index. For top-level fields, the index refers to the entry number. Fields that are part of
nested collections have global index numbers that are derived from their parent indexes.

Fields of simple types with a Map() method will use that and thus expose zero-copy access.
*/
// clang-format on
template <typename T, bool UserProvidedAddress>
class RNTupleViewBase {
   using FieldT = RField<T>;

protected:
   /// fFieldId has fParent always set to null; views access nested fields without looking at the parent
   FieldT fField;
   /// Used as a Read() destination for fields that are not mappable
   RFieldBase::RValue fValue;

   void SetupField(DescriptorId_t fieldId, Internal::RPageSource *pageSource)
   {
      fField.SetOnDiskId(fieldId);
      Internal::CallConnectPageSourceOnField(fField, *pageSource);
      if constexpr (!UserProvidedAddress) {
         if ((fField.GetTraits() & RFieldBase::kTraitMappable) && fField.HasReadCallbacks())
            throw RException(R__FAIL("view disallowed on field with mappable type and read callback"));
      }
   }

   RNTupleViewBase(DescriptorId_t fieldId, Internal::RPageSource *pageSource)
      : fField(pageSource->GetSharedDescriptorGuard()->GetFieldDescriptor(fieldId).GetFieldName()),
        fValue(fField.CreateValue())
   {
      SetupField(fieldId, pageSource);
   }

   RNTupleViewBase(DescriptorId_t fieldId, Internal::RPageSource *pageSource, std::shared_ptr<T> objPtr)
      : fField(pageSource->GetSharedDescriptorGuard()->GetFieldDescriptor(fieldId).GetFieldName()),
        fValue(fField.BindValue(objPtr))
   {
      SetupField(fieldId, pageSource);
   }

   RNTupleViewBase(DescriptorId_t fieldId, Internal::RPageSource *pageSource, T *rawPtr)
      : fField(pageSource->GetSharedDescriptorGuard()->GetFieldDescriptor(fieldId).GetFieldName()),
        fValue(fField.BindValue(Internal::MakeAliasedSharedPtr(rawPtr)))
   {
      SetupField(fieldId, pageSource);
   }

   // Protected destructor to avoid instantiating this class directly
   ~RNTupleViewBase() = default;

public:
   RNTupleViewBase(const RNTupleViewBase &other) = delete;
   RNTupleViewBase(RNTupleViewBase &&other) = default;
   RNTupleViewBase &operator=(const RNTupleViewBase &other) = delete;
   RNTupleViewBase &operator=(RNTupleViewBase &&other) = default;

   const FieldT &GetField() const { return fField; }
   RNTupleGlobalRange GetFieldRange() const { return RNTupleGlobalRange(0, fField.GetNElements()); }

   const T &operator()(NTupleSize_t globalIndex)
   {
      if constexpr (Internal::isMappable<FieldT> && !UserProvidedAddress) {
         return *fField.Map(globalIndex);
      } else {
         fValue.Read(globalIndex);
         return fValue.GetRef<T>();
      }
   }

   const T &operator()(RClusterIndex clusterIndex)
   {
      if constexpr (Internal::isMappable<FieldT> && !UserProvidedAddress) {
         return *fField.Map(clusterIndex);
      } else {
         fValue.Read(clusterIndex);
         return fValue.GetRef<T>();
      }
   }

   // TODO(bgruber): turn enable_if into requires clause with C++20
   template <typename C = T, std::enable_if_t<Internal::isMappable<FieldT>, C *> = nullptr>
   const C *MapV(NTupleSize_t globalIndex, NTupleSize_t &nItems)
   {
      return fField.MapV(globalIndex, nItems);
   }

   // TODO(bgruber): turn enable_if into requires clause with C++20
   template <typename C = T, std::enable_if_t<Internal::isMappable<FieldT>, C *> = nullptr>
   const C *MapV(RClusterIndex clusterIndex, NTupleSize_t &nItems)
   {
      return fField.MapV(clusterIndex, nItems);
   }

   void Bind(std::shared_ptr<T> objPtr)
   {
      static_assert(
         UserProvidedAddress,
         "Only views which were created with an external memory location at construction time can be bound to a "
         "different memory location afterwards. Call the RNTupleReader::GetView overload with a shared_ptr.");
      fValue.Bind(objPtr);
   }

   void BindRawPtr(T *rawPtr)
   {
      static_assert(
         UserProvidedAddress,
         "Only views which were created with an external memory location at construction time can be bound to a "
         "different memory location afterwards. Call the RNTupleReader::GetView overload with a shared_ptr.");
      fValue.BindRawPtr(rawPtr);
   }

   void EmplaceNew()
   {
      // Let the user know they are misusing this function in case the field is
      // mappable and they are providing an external address after construction.
      // This would mean creating a new RValue member but not using it since
      // reading would be done by RField::Map directly.
      static_assert(!(Internal::isMappable<FieldT> && !UserProvidedAddress),
                    "Cannot emplace a new value into a view of a mappable type, unless an external memory location is "
                    "provided at construction time.");
      fValue.EmplaceNew();
   }
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleViewBase<void>
\ingroup NTuple
\brief An RNTupleView where the type is not known at compile time.

Can be used to read individual fields whose type is unknown. The void view gives access to the RValue
in addition to the field, so that the read object can be retrieved.
*/
// clang-format on
template <bool UserProvidedAddress>
class RNTupleViewBase<void, UserProvidedAddress> {
   friend class RNTupleReader;
   friend class RNTupleCollectionView;

protected:
   std::unique_ptr<RFieldBase> fField;
   RFieldBase::RValue fValue;

   static std::unique_ptr<RFieldBase> CreateField(DescriptorId_t fieldId, const RNTupleDescriptor &desc)
   {
      return desc.GetFieldDescriptor(fieldId).CreateField(desc);
   }

   void SetupField(DescriptorId_t fieldId, Internal::RPageSource *pageSource)
   {
      fField->SetOnDiskId(fieldId);
      Internal::CallConnectPageSourceOnField(*fField, *pageSource);
   }

   RNTupleViewBase(DescriptorId_t fieldId, Internal::RPageSource *pageSource)
      : fField(CreateField(fieldId, pageSource->GetSharedDescriptorGuard().GetRef())), fValue(fField->CreateValue())
   {
      SetupField(fieldId, pageSource);
   }

   RNTupleViewBase(DescriptorId_t fieldId, Internal::RPageSource *pageSource, std::shared_ptr<void> objPtr)
      : fField(CreateField(fieldId, pageSource->GetSharedDescriptorGuard().GetRef())), fValue(fField->BindValue(objPtr))
   {
      SetupField(fieldId, pageSource);
   }

   RNTupleViewBase(DescriptorId_t fieldId, Internal::RPageSource *pageSource, void *rawPtr)
      : fField(CreateField(fieldId, pageSource->GetSharedDescriptorGuard().GetRef())),
        fValue(fField->BindValue(Internal::MakeAliasedSharedPtr(rawPtr)))
   {
      SetupField(fieldId, pageSource);
   }

   // Protected destructor to avoid instantiating this class directly
   ~RNTupleViewBase() = default;

public:
   RNTupleViewBase(const RNTupleViewBase &other) = delete;
   RNTupleViewBase(RNTupleViewBase &&other) = default;
   RNTupleViewBase &operator=(const RNTupleViewBase &other) = delete;
   RNTupleViewBase &operator=(RNTupleViewBase &&other) = default;

   const RFieldBase &GetField() const { return *fField; }
   const RFieldBase::RValue &GetValue() const { return fValue; }
   RNTupleGlobalRange GetFieldRange() const { return RNTupleGlobalRange(0, fField->GetNElements()); }

   void operator()(NTupleSize_t globalIndex) { fValue.Read(globalIndex); }
   void operator()(RClusterIndex clusterIndex) { fValue.Read(clusterIndex); }

   void Bind(std::shared_ptr<void> objPtr)
   {
      static_assert(
         UserProvidedAddress,
         "Only views which were created with an external memory location at construction time can be bound to a "
         "different memory location afterwards. Call the RNTupleReader::GetView overload with a shared_ptr.");
      fValue.Bind(objPtr);
   }

   void BindRawPtr(void *rawPtr)
   {
      static_assert(
         UserProvidedAddress,
         "Only views which were created with an external memory location at construction time can be bound to a "
         "different memory location afterwards. Call the RNTupleReader::GetView overload with a shared_ptr.");
      fValue.BindRawPtr(rawPtr);
   }

   void EmplaceNew() { fValue.EmplaceNew(); }
};

} // namespace Internal

// clang-format off
/**
\class ROOT::Experimental::RNTupleUnownedView
\ingroup NTuple
\brief An RNTupleUnownedView provides read-only access to a single field of the ntuple. The memory is
       managed internally by the RNTupleUnownedView.

\tparam T The type of the object that will be read by the view

The view owns a field and its underlying columns in order to fill an ntuple value object with data. Data can be
accessed by index. For top-level fields, the index refers to the entry number. Fields that are part of
nested collections have global index numbers that are derived from their parent indexes.

Fields of simple types with a Map() method will use that and thus expose zero-copy access.
*/
// clang-format on
template <typename T>
class RNTupleUnownedView : public Internal::RNTupleViewBase<T, false> {
   friend class RNTupleReader;
   friend class RNTupleCollectionView;

public:
   using Internal::RNTupleViewBase<T, false>::RNTupleViewBase;
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleOwnedView
\ingroup NTuple
\brief An RNTupleOwnedView provides read-only access to a single field of the ntuple. The memory is
       provided and owned by the user.

\tparam T The type of the object that will be read by the view

The view owns a field and its underlying columns in order to fill an ntuple value object with data. Data can be
accessed by index. For top-level fields, the index refers to the entry number. Fields that are part of
nested collections have global index numbers that are derived from their parent indexes.

Fields of simple types with a Map() method will use that and thus expose zero-copy access.
*/
// clang-format on
template <typename T>
class RNTupleOwnedView : public Internal::RNTupleViewBase<T, true> {
   friend class RNTupleReader;
   friend class RNTupleCollectionView;

public:
   using Internal::RNTupleViewBase<T, true>::RNTupleViewBase;
};

// clang-format off
/**
\class ROOT::Experimental::RNTupleCollectionView
\ingroup NTuple
\brief A view for a collection, that can itself generate new ntuple views for its nested fields.
*/
// clang-format on
class RNTupleCollectionView : public RNTupleUnownedView<ClusterSize_t> {
   friend class RNTupleReader;

private:
   Internal::RPageSource *fSource;
   DescriptorId_t fCollectionFieldId;

   RNTupleCollectionView(DescriptorId_t fieldId, Internal::RPageSource *source)
      : RNTupleUnownedView<ClusterSize_t>(fieldId, source), fSource(source), fCollectionFieldId(fieldId)
   {
   }

public:
   RNTupleCollectionView(const RNTupleCollectionView &other) = delete;
   RNTupleCollectionView(RNTupleCollectionView &&other) = default;
   RNTupleCollectionView &operator=(const RNTupleCollectionView &other) = delete;
   RNTupleCollectionView &operator=(RNTupleCollectionView &&other) = default;
   ~RNTupleCollectionView() = default;

   RNTupleClusterRange GetCollectionRange(NTupleSize_t globalIndex)
   {
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
   RNTupleUnownedView<T> GetView(std::string_view fieldName)
   {
      auto fieldId = fSource->GetSharedDescriptorGuard()->FindFieldId(fieldName, fCollectionFieldId);
      if (fieldId == kInvalidDescriptorId) {
         throw RException(R__FAIL("no field named '" + std::string(fieldName) + "' in RNTuple '" +
                                  fSource->GetSharedDescriptorGuard()->GetName() + "'"));
      }
      return RNTupleUnownedView<T>(fieldId, fSource);
   }
   /// Raises an exception if there is no field with the given name.
   RNTupleCollectionView GetCollectionView(std::string_view fieldName)
   {
      auto fieldId = fSource->GetSharedDescriptorGuard()->FindFieldId(fieldName, fCollectionFieldId);
      if (fieldId == kInvalidDescriptorId) {
         throw RException(R__FAIL("no field named '" + std::string(fieldName) + "' in RNTuple '" +
                                  fSource->GetSharedDescriptorGuard()->GetName() + "'"));
      }
      return RNTupleCollectionView(fieldId, fSource);
   }

   ClusterSize_t operator()(NTupleSize_t globalIndex)
   {
      ClusterSize_t size;
      RClusterIndex collectionStart;
      fField.GetCollectionInfo(globalIndex, &collectionStart, &size);
      return size;
   }
   ClusterSize_t operator()(RClusterIndex clusterIndex)
   {
      ClusterSize_t size;
      RClusterIndex collectionStart;
      fField.GetCollectionInfo(clusterIndex, &collectionStart, &size);
      return size;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
