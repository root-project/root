/// \file ROOT/RForestView.hxx
/// \ingroup Forest ROOT7
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

#ifndef ROOT7_RForestView
#define ROOT7_RForestView

#include <ROOT/RForestUtil.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RStringView.hxx>

#include <iterator>
#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {


// clang-format off
/**
\class ROOT::Experimental::RForestViewContext
\ingroup Forest
\brief Used to loop over indexes (entries or collections) between start and end
*/
// clang-format on
class RForestViewRange {
private:
   const ForestSize_t fStart;
   const ForestSize_t fEnd;
public:
   class RIterator : public std::iterator<std::forward_iterator_tag, ForestSize_t> {
   private:
      using iterator = RIterator;
      ForestSize_t fIndex = kInvalidForestIndex;
   public:
      RIterator() = default;
      explicit RIterator(ForestSize_t index) : fIndex(index) {}
      ~RIterator() = default;

      iterator  operator++(int) /* postfix */        { auto r = *this; fIndex++; return r; }
      iterator& operator++()    /* prefix */         { fIndex++; return *this; }
      reference operator* ()                         { return fIndex; }
      pointer   operator->()                         { return &fIndex; }
      bool      operator==(const iterator& rh) const { return fIndex == rh.fIndex; }
      bool      operator!=(const iterator& rh) const { return fIndex != rh.fIndex; }
   };

   RForestViewRange(ForestSize_t start, ForestSize_t end) : fStart(start), fEnd(end) {}
   RIterator begin() { return RIterator(fStart); }
   RIterator end() { return RIterator(fEnd); }
};


// clang-format off
/**
\class ROOT::Experimental::RForestView
\ingroup Forest
\brief An RForestView provides read-only access to a single field of the forest

(NB(jblomer): The forest view is very close to TTreeReader. Do we simply want to teach TTreeReader to deal with Forest?)

The view owns a field and its underlying columns in order to fill a tree value object with data. Data can be
accessed by index. For top level fields, the index refers to the entry number. Fields that are part of
nested collections have global index numbers that are derived from their parent indexes.

The RForestView object is an iterable. That means, all field values in the tree can be sequentially read from begin() to end().

For simple types, template specializations let the reading become a pure mapping into a page buffer.
*/
// clang-format on
template <typename T>
class RForestView {
   friend class RInputForest;
   friend class RForestViewCollection;

protected:
   RField<T> fField;
   RFieldValue<T> fValue;
   RForestView(std::string_view fieldName, Detail::RPageSource* pageSource)
     : fField(fieldName), fValue(fField.GenerateValue())
   {
      fField.ConnectColumns(pageSource);
      for (auto& f : fField) {
         f.ConnectColumns(pageSource);
      }
   }

public:
   RForestView(const RForestView& other) = delete;
   RForestView(RForestView&& other) = default;
   RForestView& operator=(const RForestView& other) = delete;
   RForestView& operator=(RForestView&& other) = default;
   ~RForestView() { fField.DestroyValue(fValue); }

   const T& operator()(ForestSize_t index) {
      fField.Read(index, &fValue);
      return *fValue.Get();
   }
};

// Template specializations in order to directly map simple types into the page pool

template <>
class RForestView<float> {
   friend class RInputForest;
   friend class RForestViewCollection;

protected:
   RField<float> fField;
   RForestView(std::string_view fieldName, Detail::RPageSource* pageSource) : fField(fieldName) {
      fField.ConnectColumns(pageSource);
   }

public:
   RForestView(const RForestView& other) = delete;
   RForestView(RForestView&& other) = default;
   RForestView& operator=(const RForestView& other) = delete;
   RForestView& operator=(RForestView&& other) = default;
   ~RForestView() = default;

   float operator()(ForestSize_t index) { return *fField.Map(index); }
};


// clang-format off
/**
\class ROOT::Experimental::RForestViewCollection
\ingroup Forest
\brief A tree view for a collection, that can itself generate new tree views for its nested fields.
*/
// clang-format on
class RForestViewCollection : public RForestView<ClusterSize_t> {
    friend class RInputForest;

private:
   std::string fCollectionName;
   Detail::RPageSource* fSource;

   RForestViewCollection(std::string_view fieldName, Detail::RPageSource* source)
      : RForestView<ClusterSize_t>(fieldName, source)
      , fCollectionName(fieldName)
      , fSource(source)
   {}

   std::string GetSubName(std::string_view name) {
      std::string prefix(fCollectionName);
      prefix.push_back(Detail::RFieldBase::kCollectionSeparator);
      return prefix + std::string(name);
   }

public:
   RForestViewCollection(const RForestViewCollection& other) = delete;
   RForestViewCollection(RForestViewCollection&& other) = default;
   RForestViewCollection& operator=(const RForestViewCollection& other) = delete;
   RForestViewCollection& operator=(RForestViewCollection&& other) = default;
   ~RForestViewCollection() = default;

   RForestViewRange GetViewRange(ForestSize_t index) {
      ClusterSize_t size;
      ForestSize_t idxStart;
      fField.GetCollectionInfo(index, &idxStart, &size);
      return RForestViewRange(idxStart, idxStart + size);
   }
   template <typename T>
   RForestView<T> GetView(std::string_view fieldName) { return RForestView<T>(GetSubName(fieldName), fSource); }
   RForestViewCollection GetViewCollection(std::string_view fieldName) {
      return RForestViewCollection(GetSubName(fieldName), fSource);
   }

   ClusterSize_t operator()(ForestSize_t index) {
      ClusterSize_t size;
      ForestSize_t idxStart;
      fField.GetCollectionInfo(index, &idxStart, &size);
      return size;
   }
};

} // namespace Experimental
} // namespace ROOT

#endif
