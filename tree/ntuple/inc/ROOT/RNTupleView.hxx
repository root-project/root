/// \file ROOT/RNTupleView.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-05

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleView
#define ROOT_RNTupleView

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleRange.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <string_view>

#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <unordered_map>

namespace ROOT {

class RNTupleReader;

namespace Internal {

/// Helper to get the iteration space of the given field that needs to be connected to the given page source.
/// The indexes are given by the number of elements of the principal column of the field or, if none exists,
/// by the number of elements of the first principal column found in the subfields searched by BFS.
/// If the field hierarchy is empty on columns, the returned field range is invalid (start and end set to
/// kInvalidNTupleIndex). An attempt to use such a field range in RNTupleViewBase::GetFieldRange will throw.
ROOT::RNTupleGlobalRange GetFieldRange(const ROOT::RFieldBase &field, const ROOT::Internal::RPageSource &pageSource);

} // namespace Internal

// clang-format off
/**
\class ROOT::RNTupleViewBase
\ingroup NTuple
\brief An RNTupleView provides read-only access to a single field of an RNTuple

\tparam T The type of the object that will be read by the view; can be void if unknown at compile time.

The view owns a field and its underlying columns in order to fill an RField::RValue object with data. Data can be
accessed by index. For top-level fields, the index refers to the entry number. Fields that are part of
nested collections have global index numbers that are derived from their parent indexes (\see GetFieldRange()).

View can only be created by a reader or by a collection view.

**Example: read an RNTuple's field with a view**
~~~ {.cpp}
auto reader = RNTupleReader::Open("myNtuple", "myntuple.root");
auto viewFoo = reader->GetView<float>("foo");
for (auto idx : reader->GetEntryRange()) {
   float foo = viewFoo(idx); // read field "foo" of the `idx`-th entry
   std::cout << foo << "\n";
}
~~~

**Example: read an RNTuple's collection subfield with a view**
~~~ {.cpp}
auto reader = RNTupleReader::Open("myNtuple", "myntuple.root");
// Assuming "v" is a std::vector<int>:
auto view = reader->GetView<int>("v._0");
// Effectively flattens all fields "v" in all entries and reads their elements.
for (auto idx : view.GetFieldRange()) {
   int x = view(idx);
   std::cout << x << "\n";
}
~~~
*/
// clang-format on
template <typename T>
class RNTupleViewBase {
protected:
   std::unique_ptr<ROOT::RFieldBase> fField;
   ROOT::RNTupleGlobalRange fFieldRange;
   ROOT::RFieldBase::RValue fValue;

   static std::unique_ptr<ROOT::RFieldBase>
   CreateField(ROOT::DescriptorId_t fieldId, Internal::RPageSource &pageSource, std::string_view typeName = "")
   {
      std::unique_ptr<ROOT::RFieldBase> field;
      {
         const auto &desc = pageSource.GetSharedDescriptorGuard().GetRef();
         const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);
         if constexpr (std::is_void_v<T>) {
            if (typeName.empty())
               field = fieldDesc.CreateField(desc);
            else
               field = ROOT::RFieldBase::Create(fieldDesc.GetFieldName(), std::string(typeName)).Unwrap();
         } else {
            field = std::make_unique<ROOT::RField<T>>(fieldDesc.GetFieldName());
         }
      }
      field->SetOnDiskId(fieldId);
      ROOT::Internal::CallConnectPageSourceOnField(*field, pageSource);
      return field;
   }

   RNTupleViewBase(std::unique_ptr<ROOT::RFieldBase> field, ROOT::RNTupleGlobalRange range)
      : fField(std::move(field)), fFieldRange(range), fValue(fField->CreateValue())
   {
   }

   RNTupleViewBase(std::unique_ptr<ROOT::RFieldBase> field, ROOT::RNTupleGlobalRange range, std::shared_ptr<T> objPtr)
      : fField(std::move(field)), fFieldRange(range), fValue(fField->BindValue(objPtr))
   {
   }

   RNTupleViewBase(std::unique_ptr<ROOT::RFieldBase> field, ROOT::RNTupleGlobalRange range, T *rawPtr)
      : fField(std::move(field)),
        fFieldRange(range),
        fValue(fField->BindValue(ROOT::Internal::MakeAliasedSharedPtr(rawPtr)))
   {
   }

public:
   RNTupleViewBase(const RNTupleViewBase &other) = delete;
   RNTupleViewBase(RNTupleViewBase &&other) = default;
   RNTupleViewBase &operator=(const RNTupleViewBase &other) = delete;
   RNTupleViewBase &operator=(RNTupleViewBase &&other) = default;
   ~RNTupleViewBase() = default;

   const ROOT::RFieldBase &GetField() const { return *fField; }
   ROOT::RFieldBase::RBulkValues CreateBulk() { return fField->CreateBulk(); }

   const ROOT::RFieldBase::RValue &GetValue() const { return fValue; }
   /// Returns the global field range of this view.
   /// This may differ from the RNTuple's entry range in case of subfields and can be used to iterate
   /// over all the concatenated elements of the subfield without caring which entry they belong to.
   /// Throws an RException if the underlying field of this view is empty, i.e. if it's a class or
   /// record field with no associated columns.
   ROOT::RNTupleGlobalRange GetFieldRange() const
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
\class ROOT::RNTupleView
\ingroup NTuple
\brief An RNTupleView for a known type. See RNTupleViewBase.
*/
// clang-format on
template <typename T>
class RNTupleView : public RNTupleViewBase<T> {
   friend class ROOT::RNTupleReader;
   friend class RNTupleCollectionView;

protected:
   RNTupleView(std::unique_ptr<ROOT::RFieldBase> field, ROOT::RNTupleGlobalRange range)
      : RNTupleViewBase<T>(std::move(field), range)
   {
   }

   RNTupleView(std::unique_ptr<ROOT::RFieldBase> field, ROOT::RNTupleGlobalRange range, std::shared_ptr<T> objPtr)
      : RNTupleViewBase<T>(std::move(field), range, objPtr)
   {
   }

   RNTupleView(std::unique_ptr<ROOT::RFieldBase> field, ROOT::RNTupleGlobalRange range, T *rawPtr)
      : RNTupleViewBase<T>(std::move(field), range, rawPtr)
   {
   }

public:
   RNTupleView(const RNTupleView &other) = delete;
   RNTupleView(RNTupleView &&other) = default;
   RNTupleView &operator=(const RNTupleView &other) = delete;
   RNTupleView &operator=(RNTupleView &&other) = default;
   ~RNTupleView() = default;

   /// Reads the value of this view for the entry with the provided `globalIndex`.
   const T &operator()(ROOT::NTupleSize_t globalIndex)
   {
      RNTupleViewBase<T>::fValue.Read(globalIndex);
      return RNTupleViewBase<T>::fValue.template GetRef<T>();
   }

   /// Reads the value of this view for the entry with the provided `localIndex`.
   /// See RNTupleLocalIndex for more details.
   const T &operator()(RNTupleLocalIndex localIndex)
   {
      RNTupleViewBase<T>::fValue.Read(localIndex);
      return RNTupleViewBase<T>::fValue.template GetRef<T>();
   }
};

// clang-format off
/**
\class ROOT::RNTupleView
\ingroup NTuple
\brief An RNTupleView that can be used when the type is unknown at compile time. See RNTupleViewBase.
*/
// clang-format on
template <>
class RNTupleView<void> final : public RNTupleViewBase<void> {
   friend class ROOT::RNTupleReader;
   friend class RNTupleCollectionView;

protected:
   RNTupleView(std::unique_ptr<ROOT::RFieldBase> field, ROOT::RNTupleGlobalRange range)
      : RNTupleViewBase<void>(std::move(field), range)
   {
   }

   RNTupleView(std::unique_ptr<ROOT::RFieldBase> field, ROOT::RNTupleGlobalRange range, std::shared_ptr<void> objPtr)
      : RNTupleViewBase<void>(std::move(field), range, objPtr)
   {
   }

   RNTupleView(std::unique_ptr<ROOT::RFieldBase> field, ROOT::RNTupleGlobalRange range, void *rawPtr)
      : RNTupleViewBase<void>(std::move(field), range, rawPtr)
   {
   }

public:
   RNTupleView(const RNTupleView &other) = delete;
   RNTupleView(RNTupleView &&other) = default;
   RNTupleView &operator=(const RNTupleView &other) = delete;
   RNTupleView &operator=(RNTupleView &&other) = default;
   ~RNTupleView() = default;

   /// \see RNTupleView::operator()(ROOT::NTupleSize_t)
   void operator()(ROOT::NTupleSize_t globalIndex) { fValue.Read(globalIndex); }
   /// \see RNTupleView::operator()(RNTupleLocalIndex)
   void operator()(RNTupleLocalIndex localIndex) { fValue.Read(localIndex); }
};

// clang-format off
/**
\class ROOT::RNTupleDirectAccessView
\ingroup NTuple
\brief A view variant that provides direct access to the I/O buffers. Only works for mappable fields.
*/
// clang-format on
template <typename T>
class RNTupleDirectAccessView {
   friend class ROOT::RNTupleReader;
   friend class RNTupleCollectionView;

protected:
   ROOT::RField<T> fField;
   ROOT::RNTupleGlobalRange fFieldRange;

   static ROOT::RField<T> CreateField(ROOT::DescriptorId_t fieldId, ROOT::Internal::RPageSource &pageSource)
   {
      const auto &desc = pageSource.GetSharedDescriptorGuard().GetRef();
      const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);
      if (!Internal::IsMatchingFieldType<T>(fieldDesc.GetTypeName())) {
         throw RException(R__FAIL("type mismatch for field " + fieldDesc.GetFieldName() + ": " +
                                  fieldDesc.GetTypeName() + " vs. " + ROOT::RField<T>::TypeName()));
      }
      ROOT::RField<T> field(fieldDesc.GetFieldName());
      field.SetOnDiskId(fieldId);
      ROOT::Internal::CallConnectPageSourceOnField(field, pageSource);
      return field;
   }

   RNTupleDirectAccessView(ROOT::RField<T> field, ROOT::RNTupleGlobalRange range)
      : fField(std::move(field)), fFieldRange(range)
   {
   }

public:
   RNTupleDirectAccessView(const RNTupleDirectAccessView &other) = delete;
   RNTupleDirectAccessView(RNTupleDirectAccessView &&other) = default;
   RNTupleDirectAccessView &operator=(const RNTupleDirectAccessView &other) = delete;
   RNTupleDirectAccessView &operator=(RNTupleDirectAccessView &&other) = default;
   ~RNTupleDirectAccessView() = default;

   const ROOT::RFieldBase &GetField() const { return fField; }
   /// \see RNTupleView::GetFieldRange()
   ROOT::RNTupleGlobalRange GetFieldRange() const { return fFieldRange; }

   /// \see RNTupleView::operator()(ROOT::NTupleSize_t)
   const T &operator()(ROOT::NTupleSize_t globalIndex) { return *fField.Map(globalIndex); }
   /// \see RNTupleView::operator()(RNTupleLocalIndex)
   const T &operator()(RNTupleLocalIndex localIndex) { return *fField.Map(localIndex); }
};

// clang-format off
/**
\class ROOT::RNTupleCollectionView
\ingroup NTuple
\brief A view for a collection, that can itself generate new ntuple views for its nested fields.
*/
// clang-format on
class RNTupleCollectionView {
   friend class ROOT::RNTupleReader;

private:
   ROOT::Internal::RPageSource *fSource;
   ROOT::RField<RNTupleCardinality<std::uint64_t>> fField;
   ROOT::RFieldBase::RValue fValue;

   RNTupleCollectionView(ROOT::DescriptorId_t fieldId, const std::string &fieldName,
                         ROOT::Internal::RPageSource *source)
      : fSource(source), fField(fieldName), fValue(fField.CreateValue())
   {
      fField.SetOnDiskId(fieldId);
      ROOT::Internal::CallConnectPageSourceOnField(fField, *source);
   }

   static RNTupleCollectionView Create(ROOT::DescriptorId_t fieldId, ROOT::Internal::RPageSource *source)
   {
      std::string fieldName;
      {
         const auto &desc = source->GetSharedDescriptorGuard().GetRef();
         const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);
         if (fieldDesc.GetStructure() != ROOT::ENTupleStructure::kCollection) {
            throw RException(
               R__FAIL("invalid attemt to create collection view on non-collection field " + fieldDesc.GetFieldName()));
         }
         fieldName = fieldDesc.GetFieldName();
      }
      return RNTupleCollectionView(fieldId, fieldName, source);
   }

   ROOT::DescriptorId_t GetFieldId(std::string_view fieldName)
   {
      auto descGuard = fSource->GetSharedDescriptorGuard();
      auto fieldId = descGuard->FindFieldId(fieldName, fField.GetOnDiskId());
      if (fieldId == ROOT::kInvalidDescriptorId) {
         throw RException(R__FAIL("no field named '" + std::string(fieldName) + "' in collection '" +
                                  descGuard->GetQualifiedFieldName(fField.GetOnDiskId()) + "'"));
      }
      return fieldId;
   }

   std::uint64_t GetCardinalityValue() const
   {
      // We created the RValue and know its type, avoid extra checks.
      void *ptr = fValue.GetPtr<void>().get();
      return *static_cast<RNTupleCardinality<std::uint64_t> *>(ptr);
   }

public:
   RNTupleCollectionView(const RNTupleCollectionView &other) = delete;
   RNTupleCollectionView(RNTupleCollectionView &&other) = default;
   RNTupleCollectionView &operator=(const RNTupleCollectionView &other) = delete;
   RNTupleCollectionView &operator=(RNTupleCollectionView &&other) = default;
   ~RNTupleCollectionView() = default;

   ROOT::RNTupleLocalRange GetCollectionRange(ROOT::NTupleSize_t globalIndex)
   {
      ROOT::NTupleSize_t size;
      RNTupleLocalIndex collectionStart;
      fField.GetCollectionInfo(globalIndex, &collectionStart, &size);
      return ROOT::RNTupleLocalRange(collectionStart.GetClusterId(), collectionStart.GetIndexInCluster(),
                                     collectionStart.GetIndexInCluster() + size);
   }

   ROOT::RNTupleLocalRange GetCollectionRange(RNTupleLocalIndex localIndex)
   {
      ROOT::NTupleSize_t size;
      RNTupleLocalIndex collectionStart;
      fField.GetCollectionInfo(localIndex, &collectionStart, &size);
      return ROOT::RNTupleLocalRange(collectionStart.GetClusterId(), collectionStart.GetIndexInCluster(),
                                     collectionStart.GetIndexInCluster() + size);
   }

   /// Provides access to an individual (sub)field.
   ///
   /// Raises an exception if there is no field with the given name.
   ///
   /// \sa ROOT::RNTupleReader::GetView(std::string_view)
   template <typename T>
   RNTupleView<T> GetView(std::string_view fieldName)
   {
      auto field = RNTupleView<T>::CreateField(GetFieldId(fieldName), *fSource);
      auto range = Internal::GetFieldRange(*field, *fSource);
      return RNTupleView<T>(std::move(field), range);
   }

   /// Provides direct access to the I/O buffers of a **mappable** (sub)field.
   ///
   /// Raises an exception if there is no field with the given name.
   /// Attempting to access the values of a direct-access view for non-mappable fields will yield compilation errors.
   ///
   /// \sa ROOT::RNTupleReader::DirectAccessView(std::string_view)
   template <typename T>
   RNTupleDirectAccessView<T> GetDirectAccessView(std::string_view fieldName)
   {
      auto field = RNTupleDirectAccessView<T>::CreateField(GetFieldId(fieldName), *fSource);
      auto range = Internal::GetFieldRange(field, *fSource);
      return RNTupleDirectAccessView<T>(std::move(field), range);
   }

   /// Provides access to a collection field, that can itself generate new RNTupleViews for its nested fields.
   ///
   /// Raises an exception if:
   /// * there is no field with the given name or,
   /// * the field is not a collection
   ///
   /// \sa ROOT::RNTupleReader::GetCollectionView(std::string_view)
   RNTupleCollectionView GetCollectionView(std::string_view fieldName)
   {
      return RNTupleCollectionView::Create(GetFieldId(fieldName), fSource);
   }

   /// \see RNTupleView::operator()(ROOT::NTupleSize_t)
   std::uint64_t operator()(ROOT::NTupleSize_t globalIndex)
   {
      fValue.Read(globalIndex);
      return GetCardinalityValue();
   }

   /// \see RNTupleView::operator()(RNTupleLocalIndex)
   std::uint64_t operator()(RNTupleLocalIndex localIndex)
   {
      fValue.Read(localIndex);
      return GetCardinalityValue();
   }
};

} // namespace ROOT

#endif
