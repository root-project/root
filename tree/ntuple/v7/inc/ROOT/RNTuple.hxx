/// \file ROOT/RNTuple.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RNTuple
#define ROOT7_RNTuple

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RStringView.hxx>

#include <iterator>
#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

class RForestEntry;
class RForestModel;

namespace Detail {
class RPageSink;
class RPageSource;
}

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::RForest
\ingroup NTuple
\brief The RForest represents a live dataset, whose structure is defined by an RForestModel

RForest connects the static information of the RForestModel to a source or sink on physical storage.
Reading and writing requires use of the corresponding derived class RInputForest or ROutputForest.
RForest writes only complete entries (rows of the data set).  The entry itself is not kept within the
RForest, which allows for multiple concurrent entries for the same RForest.  Besides reading an entire entry,
the RForest can expose views that read only specific fields.
*/
// clang-format on
class RForest {
protected:
   std::unique_ptr<RForestModel> fModel;
   /// The number of entries is constant for reading and reflects the sum of Fill() operations when writing
   ForestSize_t fNEntries;

   /// Only the derived RInputForest and ROutputForest can be instantiated
   explicit RForest(std::unique_ptr<RForestModel> model);

public:
   RForest(const RForest&) = delete;
   RForest& operator =(const RForest&) = delete;
   ~RForest();

   RForestModel* GetModel() { return fModel.get(); }
}; // RForest

} // namespace Detail


/**
 * Listing of the different options that can be returned by RInputForest::GetInfo()
 */
enum class EForestInfo {
   kSummary,  // Forest name, description, number of entries
};


// clang-format off
/**
\class ROOT::Experimental::RInputForest
\ingroup NTuple
\brief An RForest that is used to read data from storage

An input forest provides data from storage as C++ objects. The forest model can be created from the data on storage
or it can be imposed by the user. The latter case allows users to read into a specialized forest model that covers
only a subset of the fields in the forest. The forest model is used when reading complete entries.
Individual fields can be read as well by instantiating a tree view.
*/
// clang-format on
class RInputForest : public Detail::RForest {
private:
   std::unique_ptr<Detail::RPageSource> fSource;

public:
   // Browse through the entries
   class RIterator : public std::iterator<std::forward_iterator_tag, ForestSize_t> {
   private:
      using iterator = RIterator;
      ForestSize_t fIndex = kInvalidForestIndex;
   public:
      RIterator() = default;
      explicit RIterator(ForestSize_t index) : fIndex(index) {}
      ~RIterator() = default;

      iterator  operator++(int) /* postfix */        { auto r = *this; ++fIndex; return r; }
      iterator& operator++()    /* prefix */         { ++fIndex; return *this; }
      reference operator* ()                         { return fIndex; }
      pointer   operator->()                         { return &fIndex; }
      bool      operator==(const iterator& rh) const { return fIndex == rh.fIndex; }
      bool      operator!=(const iterator& rh) const { return fIndex != rh.fIndex; }
   };


   static std::unique_ptr<RInputForest> Open(std::unique_ptr<RForestModel> model,
                                             std::string_view forestName,
                                             std::string_view storage);
   static std::unique_ptr<RInputForest> Open(std::string_view forestName, std::string_view storage);

   /// The user imposes a forest model, which must be compatible with the model found in the data on storage
   RInputForest(std::unique_ptr<RForestModel> model, std::unique_ptr<Detail::RPageSource> source);
   /// The model is generated from the forest metadata on storage
   RInputForest(std::unique_ptr<Detail::RPageSource> source);
   ~RInputForest();

   ForestSize_t GetNEntries() { return fNEntries; }

   std::string GetInfo(const EForestInfo what = EForestInfo::kSummary);

   /// Analogous to Fill(), fills the default entry of the model. Returns false at the end of the forest.
   /// On I/O errors, raises an expection.
   void LoadEntry(ForestSize_t index) { LoadEntry(index, fModel->GetDefaultEntry()); }
   /// Fills a user provided entry after checking that the entry has been instantiated from the forest model
   void LoadEntry(ForestSize_t index, RForestEntry* entry) {
      for (auto& value : *entry) {
         value.GetField()->Read(index, &value);
      }
   }

   RForestViewRange GetViewRange() { return RForestViewRange(0, fNEntries); }

   /// Provides access to an individual field that can contain either a skalar value or a collection, e.g.
   /// GetView<double>("particles.pt") or GetView<std::vector<double>>("particle").  It can as well be the index
   /// field of a collection itself, like GetView<ForestSize_t>("particle")
   template <typename T>
   RForestView<T> GetView(std::string_view fieldName) { return RForestView<T>(fieldName, fSource.get()); }
   RForestViewCollection GetViewCollection(std::string_view fieldName) {
      return RForestViewCollection(fieldName, fSource.get());
   }

   RIterator begin() { return RIterator(0); }
   RIterator end() { return RIterator(fNEntries); }
};

// clang-format off
/**
\class ROOT::Experimental::ROutputForest
\ingroup NTuple
\brief An RForest that gets filled with entries (data) and writes them to storage

An output forest can be filled with entries. The caller has to make sure that the data that gets filled into a forest
is not modified for the time of the Fill() call. The fill call serializes the C++ object into the column format and
writes data into the corresponding column page buffers.  Writing of the buffers to storage is deferred and can be
triggered by Flush() or by destructing the forest.  On I/O errors, an exception is thrown.
*/
// clang-format on
class ROutputForest : public Detail::RForest {
private:
   static constexpr ForestSize_t kDefaultClusterSizeEntries = 8192;
   std::unique_ptr<Detail::RPageSink> fSink;
   ForestSize_t fClusterSizeEntries;
   ForestSize_t fLastCommitted;

public:
   static std::unique_ptr<ROutputForest> Recreate(std::unique_ptr<RForestModel> model,
                                                  std::string_view forestName,
                                                  std::string_view storage);
   ROutputForest(std::unique_ptr<RForestModel> model, std::unique_ptr<Detail::RPageSink> sink);
   ROutputForest(const ROutputForest&) = delete;
   ROutputForest& operator=(const ROutputForest&) = delete;
   ~ROutputForest();

   /// The simplest user interface if the default entry that comes with the forest model is used
   void Fill() { Fill(fModel->GetDefaultEntry()); }
   /// Multiple entries can have been instantiated from the forest model.  This method will perform
   /// a light check whether the entry comes from the forest's own model
   void Fill(RForestEntry *entry) {
      for (auto& treeValue : *entry) {
         treeValue.GetField()->Append(treeValue);
      }
      fNEntries++;
      if ((fNEntries % fClusterSizeEntries) == 0) CommitCluster();
   }
   /// Ensure that the data from the so far seen Fill calls has been written to storage
   void CommitCluster();
};

// clang-format off
/**
\class ROOT::Experimental::RCollectionForest
\ingroup NTuple
\brief A virtual forest for collections that can be used to some extent like a real forest
*
* This class is between a field and a forest.  It carries the offset column for the collection and the default entry
* taken from the collection model.  It does not, however, have a tree model because the collection model has been merged
* into the larger forest model.
*/
// clang-format on
class RCollectionForest {
private:
   ClusterSize_t fOffset;
   std::unique_ptr<RForestEntry> fDefaultEntry;
public:
   explicit RCollectionForest(std::unique_ptr<RForestEntry> defaultEntry);
   RCollectionForest(const RCollectionForest&) = delete;
   RCollectionForest& operator=(const RCollectionForest&) = delete;
   ~RCollectionForest() = default;

   void Fill() { Fill(fDefaultEntry.get()); }
   void Fill(RForestEntry *entry) {
      for (auto& treeValue : *entry) {
         treeValue.GetField()->Append(treeValue);
      }
      fOffset++;
   }

   ClusterSize_t* GetOffsetPtr() { return &fOffset; }
};

} // namespace Experimental
} // namespace ROOT

#endif
