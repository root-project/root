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

#include <ROOT/RTreeUtil.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {

class RTreeModel;
class RTreeSink;
class RTreeSource;

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
the RTree can expose tree views that read only specific branches.
*/
// clang-format on
class RTree {
private:
   /// All trees that use the same model share its ownership
   std::shared_ptr<RTreeModel> fModel;
   /// The number of entries is constant for reading and reflects the current number of Fill() operations
   /// when writing
   TreeIndex_t fNentries;

protected:
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

More information
*/
// clang-format on
class RInputTree : public Detail::RTree {
private:
   std::unique_ptr<RTreeSource> fSource;

public:
   RInputTree(std::shared_ptr<RTreeModel> model, std::unique_ptr<RTreeSource> source);
   ~RInputTree();
};

// clang-format off
/**
\class ROOT::Experimental::RInputTree
\ingroup Forest
\brief An RTree that gets filled with entries (data) and writes them to storage

More information
*/
// clang-format on
class ROutputTree : public Detail::RTree {
private:
   std::unique_ptr<RTreeSink> fSink;

public:
   ROutputTree(std::shared_ptr<RTreeModel> model, std::unique_ptr<RTreeSink> sink);
   ~ROutputTree();
};

} // namespace Experimental
} // namespace ROOT

#endif
