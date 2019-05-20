/// \file ROOT/RPageStorage.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-07-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPageStorage
#define ROOT7_RPageStorage

#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RStringView.hxx>

#include <atomic>
#include <memory>

namespace ROOT {
namespace Experimental {

class RForestModel;
// TODO(jblomer): factory methods to create tree sinks and sources outside Detail namespace

namespace Detail {

class RColumn;
class RPagePool;
class RFieldBase;

enum class EPageStorageType {
   kSink,
   kSource,
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageStorage
\ingroup NTuple
\brief Manages tree meta-data, which is common for sinks and sources.

The tree meta-data contains of a list of fields, a unique identifier, and provenance information.
*/
// clang-format on
class RPageStorage {
protected:
   /// All data is shipped to and from physical storage in pages, and moderated through a page pool
   std::unique_ptr<RPagePool> fPagePool;

public:
   RPageStorage();
   RPageStorage(const RPageStorage &other) = delete;
   RPageStorage& operator =(const RPageStorage &other) = delete;
   virtual ~RPageStorage();

   struct RColumnHandle {
      RColumnHandle() : fId(-1), fColumn(nullptr) {}
      RColumnHandle(int id, RColumn* column) : fId(id), fColumn(column) {}
      int fId;
      RColumn *fColumn;
   };
   /// The column handle identfies a column with the current open page storage
   using ColumnHandle_t = RColumnHandle;

   /// Register a new column.  When reading, the column must exist in the tree on disk corresponding to the meta-data.
   /// When writing, every column can only be attached once.
   virtual ColumnHandle_t AddColumn(RColumn *column) = 0;
   virtual EPageStorageType GetType() = 0;
   RPagePool* GetPagePool() const { return fPagePool.get(); }
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSink
\ingroup NTuple
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
   EPageStorageType GetType() final { return EPageStorageType::kSink; }

   /// Physically creates the storage container to hold the tree (e.g., a directory in a TFile or a S3 bucket)
   virtual void Create(RForestModel* model) = 0;
   /// Write a page to the storage. The column must have been added before.
   virtual void CommitPage(ColumnHandle_t columnHandle, const RPage &page) = 0;
   /// Finalize the current cluster and create a new one for the following data.
   virtual void CommitCluster(ForestSize_t nEntries) = 0;
   /// Finalize the current cluster and the entrire data set.
   virtual void CommitDataset() = 0;
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSource
\ingroup NTuple
\brief Abstract interface to read data from a tree

The page source is initialized with the columns of interest. Pages from those columns can then be
mapped into pages. The page source also gives access to its meta-data.
*/
// clang-format on
class RPageSource : public RPageStorage {
public:
   RPageSource(std::string_view treeName);
   virtual ~RPageSource();
   EPageStorageType GetType() final { return EPageStorageType::kSource; }
   /// TODO: copy/assignment for creating clones in multiple threads.

   /// Open the physical storage container for the tree
   virtual void Attach() = 0;

   // TODO(jblomer): virtual std::unique_ptr<RFieldBase> ListFields() {/* Make me abstract */ return nullptr;}
   // TODO(jblomer): ListClusters()
   virtual std::unique_ptr<ROOT::Experimental::RForestModel> GenerateModel() = 0;

   /// Fills a page starting with index rangeStart; the corresponding column is taken from the page object
   virtual void PopulatePage(ColumnHandle_t columnHandle, ForestSize_t index, RPage* page) = 0;
   virtual ForestSize_t GetNEntries() = 0;
   virtual ForestSize_t GetNElements(ColumnHandle_t columnHandle) = 0;
   virtual ColumnId_t GetColumnId(ColumnHandle_t columnHandle) = 0;
   virtual const RNTupleDescriptor& GetDescriptor() const = 0;
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
