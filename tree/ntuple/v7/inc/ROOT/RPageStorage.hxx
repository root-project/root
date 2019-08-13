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
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RStringView.hxx>

#include <atomic>
#include <cstddef>
#include <memory>

namespace ROOT {
namespace Experimental {

class RNTupleModel;
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
\brief Common functionality of an ntuple storage for both reading and writing

The RPageStore provides access to a storage container that keeps the bits of pages and clusters comprising
an ntuple.  Concrete implementations can use a TFile, a raw file, an object store, and so on.
*/
// clang-format on
class RPageStorage {
protected:
   std::string fNTupleName;

public:
   explicit RPageStorage(std::string_view name);
   RPageStorage(const RPageStorage &other) = delete;
   RPageStorage& operator =(const RPageStorage &other) = delete;
   virtual ~RPageStorage();

   struct RColumnHandle {
      RColumnHandle() : fId(-1), fColumn(nullptr) {}
      RColumnHandle(int id, const RColumn *column) : fId(id), fColumn(column) {}
      int fId;
      const RColumn *fColumn;
   };
   /// The column handle identifies a column with the current open page storage
   using ColumnHandle_t = RColumnHandle;

   /// Register a new column.  When reading, the column must exist in the ntuple on disk corresponding to the meta-data.
   /// When writing, every column can only be attached once.
   virtual ColumnHandle_t AddColumn(DescriptorId_t fieldId, const RColumn &column) = 0;
   /// Whether the concrete implementation is a sink or a source
   virtual EPageStorageType GetType() = 0;

   /// Every page store needs to be able to free pages it handed out.  But Sinks and sources have different means
   /// of allocating pages.
   virtual void ReleasePage(RPage &page) = 0;
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSink
\ingroup NTuple
\brief Abstract interface to write data into an ntuple

The page sink takes the list of columns and afterwards a series of page commits and cluster commits.
The user is responsible to commit clusters at a consistent point, i.e. when all pages corresponding to data
up to the given entry number are committed.
*/
// clang-format on
class RPageSink : public RPageStorage {
public:
   explicit RPageSink(std::string_view ntupleName);
   virtual ~RPageSink();
   EPageStorageType GetType() final { return EPageStorageType::kSink; }

   /// Physically creates the storage container to hold the ntuple (e.g., a keys a TFile or an S3 bucket)
   /// Create() associates column handles to the columns referenced by the model
   virtual void Create(RNTupleModel &model) = 0;
   /// Write a page to the storage. The column must have been added before.
   virtual void CommitPage(ColumnHandle_t columnHandle, const RPage &page) = 0;
   /// Finalize the current cluster and create a new one for the following data.
   virtual void CommitCluster(NTupleSize_t nEntries) = 0;
   /// Finalize the current cluster and the entrire data set.
   virtual void CommitDataset() = 0;

   /// Get a new, empty page for the given column that can be filled with up to nElements.  If nElements is zero,
   /// the page sink picks an appropriate size.
   virtual RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements = 0) = 0;
};

// clang-format off
/**
\class ROOT::Experimental::Detail::RPageSource
\ingroup NTuple
\brief Abstract interface to read data from an ntuple

The page source is initialized with the columns of interest. Pages from those columns can then be
mapped into memory. The page source also gives access to the ntuple's meta-data.
*/
// clang-format on
class RPageSource : public RPageStorage {
public:
   explicit RPageSource(std::string_view ntupleName);
   virtual ~RPageSource();
   EPageStorageType GetType() final { return EPageStorageType::kSource; }
   /// TODO: copy/assignment for creating clones in multiple threads.

   /// Open the physical storage container for the tree
   virtual void Attach() = 0;

   virtual NTupleSize_t GetNEntries() = 0;
   virtual NTupleSize_t GetNElements(ColumnHandle_t columnHandle) = 0;
   virtual ColumnId_t GetColumnId(ColumnHandle_t columnHandle) = 0;
   virtual const RNTupleDescriptor& GetDescriptor() const = 0;

   /// Allocates and fills a page that contains the index-th element
   virtual RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) = 0;
   /// Another version of PopulatePage that allows to specify cluster-relative indexes
   virtual RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) = 0;
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
