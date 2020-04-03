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
#include <ROOT/RNTupleOptions.hxx>
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

class RCluster;
class RColumn;
class RPagePool;
class RFieldBase;
class RNTupleMetrics;

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

   /// Page storage implementations usually have their own metrics
   virtual RNTupleMetrics &GetMetrics() = 0;
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
protected:
   const RNTupleWriteOptions fOptions;

   /// Building the ntuple descriptor while writing is done in the same way for all the storage sink implementations.
   /// Field, column, cluster ids and page indexes per cluster are issued sequentially starting with 0
   DescriptorId_t fLastFieldId = 0;
   DescriptorId_t fLastColumnId = 0;
   DescriptorId_t fLastClusterId = 0;
   NTupleSize_t fPrevClusterNEntries = 0;
   /// Keeps track of the number of elements in the currently open cluster. Indexed by column id.
   std::vector<RClusterDescriptor::RColumnRange> fOpenColumnRanges;
   /// Keeps track of the written pages in the currently open cluster. Indexed by column id.
   std::vector<RClusterDescriptor::RPageRange> fOpenPageRanges;
   RNTupleDescriptorBuilder fDescriptorBuilder;

   virtual void CreateImpl(const RNTupleModel &model) = 0;
   virtual RClusterDescriptor::RLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) = 0;
   virtual RClusterDescriptor::RLocator CommitClusterImpl(NTupleSize_t nEntries) = 0;
   virtual void CommitDatasetImpl() = 0;

public:
   RPageSink(std::string_view ntupleName, const RNTupleWriteOptions &options);
   virtual ~RPageSink();
   /// Guess the concrete derived page source from the file name (location)
   static std::unique_ptr<RPageSink> Create(std::string_view ntupleName, std::string_view location,
                                            const RNTupleWriteOptions &options = RNTupleWriteOptions());
   EPageStorageType GetType() final { return EPageStorageType::kSink; }

   ColumnHandle_t AddColumn(DescriptorId_t fieldId, const RColumn &column) final;

   /// Physically creates the storage container to hold the ntuple (e.g., a keys a TFile or an S3 bucket)
   /// To do so, Create() calls CreateImpl() after updating the descriptor.
   /// Create() associates column handles to the columns referenced by the model
   void Create(RNTupleModel &model);
   /// Write a page to the storage. The column must have been added before.
   void CommitPage(ColumnHandle_t columnHandle, const RPage &page);
   /// Finalize the current cluster and create a new one for the following data.
   void CommitCluster(NTupleSize_t nEntries);
   /// Finalize the current cluster and the entrire data set.
   void CommitDataset() { CommitDatasetImpl(); }

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
protected:
   const RNTupleReadOptions fOptions;
   RNTupleDescriptor fDescriptor;

   virtual RNTupleDescriptor AttachImpl() = 0;

public:
   RPageSource(std::string_view ntupleName, const RNTupleReadOptions &fOptions);
   virtual ~RPageSource();
   /// Guess the concrete derived page source from the file name (location)
   static std::unique_ptr<RPageSource> Create(std::string_view ntupleName, std::string_view location,
                                              const RNTupleReadOptions &options = RNTupleReadOptions());
   /// Open the same storage multiple time, e.g. for reading in multiple threads
   virtual std::unique_ptr<RPageSource> Clone() const = 0;

   EPageStorageType GetType() final { return EPageStorageType::kSource; }
   const RNTupleDescriptor &GetDescriptor() const { return fDescriptor; }
   ColumnHandle_t AddColumn(DescriptorId_t fieldId, const RColumn &column) final;

   /// Open the physical storage container for the tree
   void Attach() { fDescriptor = AttachImpl(); }
   NTupleSize_t GetNEntries();
   NTupleSize_t GetNElements(ColumnHandle_t columnHandle);
   ColumnId_t GetColumnId(ColumnHandle_t columnHandle);

   /// Allocates and fills a page that contains the index-th element
   virtual RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) = 0;
   /// Another version of PopulatePage that allows to specify cluster-relative indexes
   virtual RPage PopulatePage(ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex) = 0;

   /// Populates all the pages but only of the attached columns of the given cluster
   virtual std::unique_ptr<RCluster> LoadCluster(DescriptorId_t clusterId) = 0;
};

} // namespace Detail

} // namespace Experimental
} // namespace ROOT

#endif
