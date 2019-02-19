/// \file ROOT/RTree.hxx
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

#ifndef ROOT7_RTree
#define ROOT7_RTree

#include <ROOT/RForestUtil.hxx>
#include <ROOT/RStringView.hxx>
#include <ROOT/RTreeModel.hxx>
#include <ROOT/RTreeView.hxx>

#include <memory>
#include <utility>

namespace ROOT {
namespace Experimental {

class RTreeEntry;
class RTreeModel;

namespace Detail {
class RPageSink;
class RPageSource;
}

namespace Detail {

// clang-format off
/**
\class ROOT::Experimental::RTree
\ingroup Forest
\brief The RTree represents a live dataset, whose structure is defined by an RTreeModel

RTree connects the static information of the RTreeModel to a source or sink on physical storage.
Reading and writing requires use of the corresponding derived class RInputTree or ROutputTree.
RTree writes only complete entries (rows of the data set).  The entry itself is not kept within the
RTree, which allows for multiple concurrent entries for the same RTree.  Besides reading an entire entry,
the RTree can expose tree views that read only specific fields.
*/
// clang-format on
class RTree {
protected:
   /// All trees that use the same model share its ownership
   std::shared_ptr<RTreeModel> fModel;
   /// The number of entries is constant for reading and reflects the sum of Fill() operations when writing
   TreeIndex_t fNEntries;

   /// Only the derived RInputTree and ROutputTree can be instantiated
   explicit RTree(std::shared_ptr<RTreeModel> model);

public:
   RTree(const RTree&) = delete;
   RTree& operator =(const RTree&) = delete;
   ~RTree();
}; // RTree

} // namespace Detail


// clang-format off
/**
\class ROOT::Experimental::RInputTree
\ingroup Forest
\brief An RTree that is used to read data from storage

An input tree provides data from storage as C++ objects. The tree model can be created from the data on storage
or it can be imposed by the user. The latter case allows users to read into a specialized tree model that covers
only a subset of the fields in the tree. The tree model is used when reading complete entries.
Individual fields can be read as well by instantiating a tree view.
*/
// clang-format on
class RInputTree : public Detail::RTree {
private:
   std::unique_ptr<Detail::RPageSource> fSource;
   TreeIndex_t fNEntries;

public:
   /// The user imposes a tree model, which must be compatible with the model found in the data on storage
   RInputTree(std::shared_ptr<RTreeModel> model, std::unique_ptr<Detail::RPageSource> source);
   /// The model is generated from the tree metadata on storage
   RInputTree(std::unique_ptr<Detail::RPageSource> source);
   ~RInputTree();

   /// Analogous to Fill(), fills the default entry of the model. Returns false at the end of the tree.
   /// On I/O errors, raises an expection.
   void GetEntry(TreeIndex_t index) { GetEntry(index, fModel->GetDefaultEntry()); }
   /// Fills a user provided entry after checking that the entry has been instantiated from the tree's model
   void GetEntry(TreeIndex_t index, RTreeEntry* entry) {
      for (auto& value : *entry) {
         value.GetField()->Read(index, &value);
      }
   }
   TreeIndex_t GetNEntries() { return fNEntries; }

   /// Provides access to an individual field that can contain either a skalar value or a collection, e.g.
   /// GetView<double>("particles.pt") or GetView<RVec<double>>("particle").  It can as well be the index
   /// field of a collection itself, like GetView<TreeIndex_t>("particle")
   template <typename T>
   RTreeView<T> GetView(std::string_view fieldName) {
      auto field = std::make_unique<RTreeField<T>>(fieldName, fSource.get());
      // ...
      return RTreeView<T>(std::move(field));
   }

   /// Returns a tree view on which one can call again GetView() and GetViewCollection.  The field name
   /// has refer to a collection
   RTreeViewCollection GetViewCollection(std::string_view fieldName);
};

// clang-format off
/**
\class ROOT::Experimental::ROutputTree
\ingroup Forest
\brief An RTree that gets filled with entries (data) and writes them to storage

An output tree can be filled with entries. The caller has to make sure that the data that gets filled into a tree
is not modified for the time of the Fill() call. The fill call serializes the C++ object into the column format and
writes data into the corresponding column page buffers.  Writing of the buffers to storage is deferred and can be
triggered by Flush() or by destructing the tree.  On I/O errors, an exception is thrown.
*/
// clang-format on
class ROutputTree : public Detail::RTree {
private:
   static constexpr TreeIndex_t kDefaultClusterSizeEntries = 8192;
   std::unique_ptr<Detail::RPageSink> fSink;
   TreeIndex_t fClusterSizeEntries;
   TreeIndex_t fLastCommitted;

public:
   ROutputTree(std::shared_ptr<RTreeModel> model, std::unique_ptr<Detail::RPageSink> sink);
   ROutputTree(const ROutputTree&) = delete;
   ROutputTree& operator=(const ROutputTree&) = delete;
   ~ROutputTree();

   /// The simplest user interface if the default entry that comes with the tree model is used
   void Fill() { Fill(fModel->GetDefaultEntry()); }
   /// Multiple tree entries can have been instantiated from the tree model.  This method will perform
   /// a light check whether the entry comes from the tree's own model
   void Fill(RTreeEntry *entry) {
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
