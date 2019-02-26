/// \file ROOT/RForest.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RForest
#define ROOT7_RForest

#include <ROOT/RForestModel.hxx>
#include <ROOT/RForestUtil.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RTreeView.hxx>

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
\ingroup Forest
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
   /// All forests that use the same model share its ownership
   std::shared_ptr<RForestModel> fModel;
   /// The number of entries is constant for reading and reflects the sum of Fill() operations when writing
   ForestIndex_t fNEntries;

   /// Only the derived RInputForest and ROutputForest can be instantiated
   explicit RForest(std::shared_ptr<RForestModel> model);

public:
   RForest(const RForest&) = delete;
   RForest& operator =(const RForest&) = delete;
   ~RForest();
}; // RForest

} // namespace Detail


// clang-format off
/**
\class ROOT::Experimental::RInputForest
\ingroup Forest
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
   /// Encapsulates the entry number for the current iteration. All views share the same current
   /// entry number. Concurrent iterations need to use different contexts.
   std::unique_ptr<RTreeViewContext> fDefaultViewContext;
   ForestIndex_t fNEntries;

public:
   /// The user imposes a forest model, which must be compatible with the model found in the data on storage
   RInputForest(std::shared_ptr<RForestModel> model, std::unique_ptr<Detail::RPageSource> source);
   /// The model is generated from the forest metadata on storage
   RInputForest(std::unique_ptr<Detail::RPageSource> source);
   ~RInputForest();

   ForestIndex_t GetNEntries() { return fNEntries; }

   /// Analogous to Fill(), fills the default entry of the model. Returns false at the end of the forest.
   /// On I/O errors, raises an expection.
   void GetEntry(ForestIndex_t index) { GetEntry(index, fModel->GetDefaultEntry()); }
   /// Fills a user provided entry after checking that the entry has been instantiated from the forest model
   void GetEntry(ForestIndex_t index, RForestEntry* entry) {
      for (auto& value : *entry) {
         value.GetField()->Read(index, &value);
      }
   }

   /// Provides access to an individual field that can contain either a skalar value or a collection, e.g.
   /// GetView<double>("particles.pt") or GetView<RVec<double>>("particle").  It can as well be the index
   /// field of a collection itself, like GetView<ForestIndex_t>("particle")
   template <typename T>
   RTreeView<T> GetView(std::string_view fieldName, RTreeViewContext* context = nullptr) {
      if (context == nullptr)
         context = fDefaultViewContext.get();
      return RTreeView<T>(fieldName, context);
   }
   std::unique_ptr<RTreeViewContext> GetViewContext();
   void ViewReset() { fDefaultViewContext->Reset(); }
   bool ViewNext() { return fDefaultViewContext->Next(); }
};

// clang-format off
/**
\class ROOT::Experimental::ROutputForest
\ingroup Forest
\brief An RForest that gets filled with entries (data) and writes them to storage

An output forest can be filled with entries. The caller has to make sure that the data that gets filled into a forest
is not modified for the time of the Fill() call. The fill call serializes the C++ object into the column format and
writes data into the corresponding column page buffers.  Writing of the buffers to storage is deferred and can be
triggered by Flush() or by destructing the forest.  On I/O errors, an exception is thrown.
*/
// clang-format on
class ROutputForest : public Detail::RForest {
private:
   static constexpr ForestIndex_t kDefaultClusterSizeEntries = 8192;
   std::unique_ptr<Detail::RPageSink> fSink;
   ForestIndex_t fClusterSizeEntries;
   ForestIndex_t fLastCommitted;

public:
   ROutputForest(std::shared_ptr<RForestModel> model, std::unique_ptr<Detail::RPageSink> sink);
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

} // namespace Experimental
} // namespace ROOT

#endif
