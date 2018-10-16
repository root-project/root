/// \file ROOT/RPageStorage.hxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorage
#define ROOT7_RPageStorage

#include <ROOT/RPage.hxx>
#include <ROOT/RStringView.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {

// TODO(jblomer): factory methods to create tree sinks and sources outside Detail namespace

namespace Detail {

class RBranchBase;
class RColumn;

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageStorage
\ingroup Forest
\brief Manages tree meta-data, which is common for sinks and sources.

The tree meta-data contains of a list of branches, a unique identifier, and provenance information.
*/
// clang-format on
class RPageStorage {
   /// Register a new column.  When reading, the column must exist in the tree on disk corresponding to the meta-data.
   /// When writing, every column can only be attached once.
   virtual void AddColumn(RColumn *column) = 0;
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSink
\ingroup Forest
\brief Abstract interface to write data into a tree

The page sink takes the list of columns and afterwards a series of page commits and cluster commits.
The user is responsible to commit clusters at consistent point, i.e. when all pages corresponding to data
up to the given entry number are committed.
*/
// clang-format on
class RPageSink : public RPageStorage {
public:
   RPageSink(std::string_view treeName);
   virtual ~RPageSink();

   /// TODO(jblomer): keep abtract and let derived classed define
   virtual void AddColumn(RColumn * /*column*/) { }

   /// Physically creates the storage container to hold the tree (e.g., a directory in a TFile or a S3 bucket)
   virtual void Create() = 0;
   /// Write a page to the storage. The column attached to the page must have been added before.
   virtual void CommitPage(RPage *page) = 0;
   /// Finalize the current cluster and create a new one for the following data.
   virtual void CommitCluster(TreeIndex_t nEntries) = 0;
   /// Finalize the current cluster and the entrire data set.
   virtual void CommitDataset(TreeIndex_t nEntries) = 0;
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSource
\ingroup Forest
\brief Abstract interface to read data from a tree

The page source is initialized with the columns of interest. Pages from those columns can then be
mapped into pages. The page source also gives access to its meta-data.
*/
// clang-format on
class RPageSource : public RPageStorage {
public:
   RPageSource(std::string_view treeName);
   virtual ~RPageSource();
   /// TODO: copy/assignment for creating clones in multiple threads.

   /// TODO(jblomer): keep abtract and let derived classed define
   virtual void AddColumn(RColumn * /*column*/) { }

   /// Open the physical storage container for the tree
   virtual void Attach() {/* Make me abstract */}
   /// Return a top-level branch that can be iterated and contains as children all the branches stored in the tree.

   // TODO(jblomer): virtual std::unique_ptr<RBranchBase> ListBranches() {/* Make me abstract */ return nullptr;}
   // TODO(jblomer): ListClusters()

   /// Fills a page starting with index rangeStart; the corresponding column is taken from the page object
   void MapSlice(TreeIndex_t /*rangeStart*/, RPage * /*page*/) {/* Make me abstract */}
   virtual TreeIndex_t GetNentries() {/* Make me abstract */ return 0;}
   virtual TreeIndex_t GetNelements(RColumn * /*column*/) {/* Make me abstract */ return 0;}
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
