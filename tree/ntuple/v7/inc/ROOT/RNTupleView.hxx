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
#include <ROOT/RStringView.hxx>

#include <iterator>
#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {


// clang-format off
/**
\class ROOT::Experimental::RNTupleViewRange
\ingroup NTuple
\brief Used to loop over indexes (entries or collections) between start and end
*/
// clang-format on
class RNTupleViewRange {
private:
   const NTupleSize_t fStart;
   const NTupleSize_t fEnd;
public:
   class RIterator : public std::iterator<std::forward_iterator_tag, NTupleSize_t> {
   private:
      using iterator = RIterator;
      NTupleSize_t fIndex = kInvalidNTupleIndex;
   public:
      RIterator() = default;
      explicit RIterator(NTupleSize_t index) : fIndex(index) {}
      ~RIterator() = default;

      iterator  operator++(int) /* postfix */        { auto r = *this; fIndex++; return r; }
      iterator& operator++()    /* prefix */         { fIndex++; return *this; }
      reference operator* ()                         { return fIndex; }
      pointer   operator->()                         { return &fIndex; }
      bool      operator==(const iterator& rh) const { return fIndex == rh.fIndex; }
      bool      operator!=(const iterator& rh) const { return fIndex != rh.fIndex; }
   };

   RNTupleViewRange(NTupleSize_t start, NTupleSize_t end) : fStart(start), fEnd(end) {}
   RIterator begin() { return RIterator(fStart); }
   RIterator end() { return RIterator(fEnd); }
};


// clang-format off
/**
\class ROOT::Experimental::RNTupleView
\ingroup NTuple
\brief An RNTupleView provides read-only access to a single field of the ntuple

(NB(jblomer): The ntuple view is very close to TTreeReader. Do we simply want to teach TTreeReader to deal with
RNTuple?)

The view owns a field and its underlying columns in order to fill an ntuple value object with data. Data can be
accessed by index. For top level fields, the index refers to the entry number. Fields that are part of
nested collections have global index numbers that are derived from their parent indexes.

The RNTupleView object is an iterable. That means, all field values in the tree can be sequentially read from begin()
to end().

For simple types, template specializations let the reading become a pure mapping into a page buffer.
*/
// clang-format on
template <typename T>
class RNTupleView {
   friend class RNTupleReader;
   friend class RNTupleViewCollection;

protected:
   RField<T> fField;
   Detail::RFieldValue fValue;
   RNTupleView(std::string_view fieldName, Detail::RPageSource* pageSource)
     : fField(fieldName), fValue(fField.GenerateValue())
   {
      Detail::RFieldFuse::Connect(*pageSource, fField);
      for (auto& f : fField) {
         Detail::RFieldFuse::Connect(*pageSource, f);
      }
   }

public:
   RNTupleView(const RNTupleView& other) = delete;
   RNTupleView(RNTupleView&& other) = default;
   RNTupleView& operator=(const RNTupleView& other) = delete;
   RNTupleView& operator=(RNTupleView&& other) = default;
   ~RNTupleView() { fField.DestroyValue(fValue); }

   const T& operator()(NTupleSize_t index) {
      fField.Read(index, &fValue);
      return *fValue.Get<T>();
   }
};

// Template specializations in order to directly map simple types into the page pool

template <>
class RNTupleView<float> {
   friend class RNTupleReader;
   friend class RNTupleViewCollection;

protected:
   RField<float> fField;
   RNTupleView(std::string_view fieldName, Detail::RPageSource* pageSource) : fField(fieldName) {
      Detail::RFieldFuse::Connect(*pageSource, fField);
   }

public:
   RNTupleView(const RNTupleView& other) = delete;
   RNTupleView(RNTupleView&& other) = default;
   RNTupleView& operator=(const RNTupleView& other) = delete;
   RNTupleView& operator=(RNTupleView&& other) = default;
   ~RNTupleView() = default;

   float operator()(NTupleSize_t index) { return *fField.Map(index); }
};


template <>
class RNTupleView<double> {
   friend class RNTupleReader;
   friend class RNTupleViewCollection;

protected:
   RField<double> fField;
   RNTupleView(std::string_view fieldName, Detail::RPageSource* pageSource) : fField(fieldName) {
      fField.ConnectColumns(pageSource);
   }

public:
   RNTupleView(const RNTupleView& other) = delete;
   RNTupleView(RNTupleView&& other) = default;
   RNTupleView& operator=(const RNTupleView& other) = delete;
   RNTupleView& operator=(RNTupleView&& other) = default;
   ~RNTupleView() = default;

   double operator()(NTupleSize_t index) { return *fField.Map(index); }
};


template <>
class RNTupleView<int> {
   friend class RNTupleReader;
   friend class RNTupleViewCollection;

protected:
   RField<int> fField;
   RNTupleView(std::string_view fieldName, Detail::RPageSource* pageSource) : fField(fieldName) {
      fField.ConnectColumns(pageSource);
   }

public:
   RNTupleView(const RNTupleView& other) = delete;
   RNTupleView(RNTupleView&& other) = default;
   RNTupleView& operator=(const RNTupleView& other) = delete;
   RNTupleView& operator=(RNTupleView&& other) = default;
   ~RNTupleView() = default;

   int operator()(NTupleSize_t index) { return *fField.Map(index); }
};


// clang-format off
/**
\class ROOT::Experimental::RNTupleViewCollection
\ingroup NTuple
\brief A view for a collection, that can itself generate new ntuple views for its nested fields.
*/
// clang-format on
class RNTupleViewCollection : public RNTupleView<ClusterSize_t> {
    friend class RNTupleReader;

private:
   std::string fCollectionName;
   Detail::RPageSource* fSource;

   RNTupleViewCollection(std::string_view fieldName, Detail::RPageSource* source)
      : RNTupleView<ClusterSize_t>(fieldName, source)
      , fCollectionName(fieldName)
      , fSource(source)
   {}

   std::string GetSubName(std::string_view name) {
      std::string prefix(fCollectionName);
      prefix.push_back(Detail::RFieldBase::kCollectionSeparator);
      return prefix + std::string(name);
   }

public:
   RNTupleViewCollection(const RNTupleViewCollection& other) = delete;
   RNTupleViewCollection(RNTupleViewCollection&& other) = default;
   RNTupleViewCollection& operator=(const RNTupleViewCollection& other) = delete;
   RNTupleViewCollection& operator=(RNTupleViewCollection&& other) = default;
   ~RNTupleViewCollection() = default;

   RNTupleViewRange GetViewRange(NTupleSize_t index) {
      ClusterSize_t size;
      NTupleSize_t idxStart;
      fField.GetCollectionInfo(index, &idxStart, &size);
      return RNTupleViewRange(idxStart, idxStart + size);
   }
   template <typename T>
   RNTupleView<T> GetView(std::string_view fieldName) { return RNTupleView<T>(GetSubName(fieldName), fSource); }
   RNTupleViewCollection GetViewCollection(std::string_view fieldName) {
      return RNTupleViewCollection(GetSubName(fieldName), fSource);
   }

   ClusterSize_t operator()(NTupleSize_t index) {
      ClusterSize_t size;
      NTupleSize_t idxStart;
      fField.GetCollectionInfo(index, &idxStart, &size);
      return size;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
