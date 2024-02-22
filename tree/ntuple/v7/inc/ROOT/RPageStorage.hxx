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

#include <ROOT/RCluster.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RSpan.hxx>
#include <string_view>

#include <atomic>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <vector>

namespace ROOT {
namespace Experimental {

class RFieldBase;
class RNTupleModel;

namespace Internal {
class RColumn;
class RColumnElementBase;
class RNTupleCompressor;
class RNTupleDecompressor;
struct RNTupleModelChangeset;

enum class EPageStorageType {
   kSink,
   kSource,
};

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageStorage
\ingroup NTuple
\brief Common functionality of an ntuple storage for both reading and writing

The RPageStore provides access to a storage container that keeps the bits of pages and clusters comprising
an ntuple.  Concrete implementations can use a TFile, a raw file, an object store, and so on.
*/
// clang-format on
class RPageStorage {
public:
   /// The interface of a task scheduler to schedule page (de)compression tasks
   class RTaskScheduler {
   public:
      virtual ~RTaskScheduler() = default;
      /// Start a new set of tasks
      virtual void Reset() = 0;
      /// Take a callable that represents a task
      virtual void AddTask(const std::function<void(void)> &taskFunc) = 0;
      /// Blocks until all scheduled tasks finished
      virtual void Wait() = 0;
   };

   /// A sealed page contains the bytes of a page as written to storage (packed & compressed).  It is used
   /// as an input to UnsealPages() as well as to transfer pages between different storage media.
   /// RSealedPage does _not_ own the buffer it is pointing to in order to not interfere with the memory management
   /// of concrete page sink and page source implementations.
   struct RSealedPage {
      const void *fBuffer = nullptr;
      std::uint32_t fSize = 0;
      std::uint32_t fNElements = 0;

      RSealedPage() = default;
      RSealedPage(const void *b, std::uint32_t s, std::uint32_t n) : fBuffer(b), fSize(s), fNElements(n) {}
      RSealedPage(const RSealedPage &other) = delete;
      RSealedPage& operator =(const RSealedPage &other) = delete;
      RSealedPage(RSealedPage &&other) = default;
      RSealedPage& operator =(RSealedPage &&other) = default;
   };

   using SealedPageSequence_t = std::deque<RSealedPage>;
   /// A range of sealed pages referring to the same column that can be used for vector commit
   struct RSealedPageGroup {
      DescriptorId_t fPhysicalColumnId;
      SealedPageSequence_t::const_iterator fFirst;
      SealedPageSequence_t::const_iterator fLast;

      RSealedPageGroup(DescriptorId_t d, SealedPageSequence_t::const_iterator b, SealedPageSequence_t::const_iterator e)
         : fPhysicalColumnId(d), fFirst(b), fLast(e)
      {
      }
   };

protected:
   Detail::RNTupleMetrics fMetrics;

   std::string fNTupleName;
   RTaskScheduler *fTaskScheduler = nullptr;
   void WaitForAllTasks()
   {
      if (!fTaskScheduler)
         return;
      fTaskScheduler->Wait();
      fTaskScheduler->Reset();
   }

public:
   explicit RPageStorage(std::string_view name);
   RPageStorage(const RPageStorage &other) = delete;
   RPageStorage& operator =(const RPageStorage &other) = delete;
   RPageStorage(RPageStorage &&other) = default;
   RPageStorage& operator =(RPageStorage &&other) = default;
   virtual ~RPageStorage();

   /// Whether the concrete implementation is a sink or a source
   virtual EPageStorageType GetType() = 0;

   struct RColumnHandle {
      DescriptorId_t fPhysicalId = kInvalidDescriptorId;
      const RColumn *fColumn = nullptr;

      /// Returns true for a valid column handle; fColumn and fPhysicalId should always either both
      /// be valid or both be invalid.
      explicit operator bool() const { return fPhysicalId != kInvalidDescriptorId && fColumn; }
   };
   /// The column handle identifies a column with the current open page storage
   using ColumnHandle_t = RColumnHandle;

   /// Register a new column.  When reading, the column must exist in the ntuple on disk corresponding to the meta-data.
   /// When writing, every column can only be attached once.
   virtual ColumnHandle_t AddColumn(DescriptorId_t fieldId, const RColumn &column) = 0;
   /// Unregisters a column.  A page source decreases the reference counter for the corresponding active column.
   /// For a page sink, dropping columns is currently a no-op.
   virtual void DropColumn(ColumnHandle_t columnHandle) = 0;

   /// Every page store needs to be able to free pages it handed out.  But Sinks and sources have different means
   /// of allocating pages.
   virtual void ReleasePage(RPage &page) = 0;

   /// Returns the default metrics object.  Subclasses might alternatively provide their own metrics object by
   /// overriding this.
   virtual Detail::RNTupleMetrics &GetMetrics() { return fMetrics; }

   /// Returns the NTuple name.
   const std::string &GetNTupleName() const { return fNTupleName; }

   void SetTaskScheduler(RTaskScheduler *taskScheduler) { fTaskScheduler = taskScheduler; }
}; // class RPageStorage

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageSink
\ingroup NTuple
\brief Abstract interface to write data into an ntuple

The page sink takes the list of columns and afterwards a series of page commits and cluster commits.
The user is responsible to commit clusters at a consistent point, i.e. when all pages corresponding to data
up to the given entry number are committed.

An object of this class may either be a wrapper (for example a RPageSinkBuf) or a "persistent" sink,
inheriting from RPagePersistentSink.
*/
// clang-format on
class RPageSink : public RPageStorage {
protected:
   std::unique_ptr<RNTupleWriteOptions> fOptions;

   /// Helper to zip pages and header/footer; includes a 16MB (kMAXZIPBUF) zip buffer.
   /// There could be concrete page sinks that don't need a compressor.  Therefore, and in order to stay consistent
   /// with the page source, we leave it up to the derived class whether or not the compressor gets constructed.
   std::unique_ptr<RNTupleCompressor> fCompressor;

   /// Helper for streaming a page. This is commonly used in derived, concrete page sinks. Note that if
   /// compressionSetting is 0 (uncompressed) and the page is mappable, the returned sealed page will
   /// point directly to the input page buffer.  Otherwise, the sealed page references an internal buffer
   /// of fCompressor.  Thus, the buffer pointed to by the RSealedPage should never be freed.
   /// Usage of this method requires construction of fCompressor.
   RSealedPage SealPage(const RPage &page, const RColumnElementBase &element, int compressionSetting);

   /// Seal a page using the provided buffer.
   static RSealedPage SealPage(const RPage &page, const RColumnElementBase &element, int compressionSetting, void *buf,
                               bool allowAlias = true);

public:
   RPageSink(std::string_view ntupleName, const RNTupleWriteOptions &options);

   RPageSink(const RPageSink&) = delete;
   RPageSink& operator=(const RPageSink&) = delete;
   RPageSink(RPageSink&&) = default;
   RPageSink& operator=(RPageSink&&) = default;
   ~RPageSink() override;

   EPageStorageType GetType() final { return EPageStorageType::kSink; }
   /// Returns the sink's write options.
   const RNTupleWriteOptions &GetWriteOptions() const { return *fOptions; }

   void DropColumn(ColumnHandle_t /*columnHandle*/) final {}

   /// Physically creates the storage container to hold the ntuple (e.g., a keys a TFile or an S3 bucket)
   /// Init() associates column handles to the columns referenced by the model
   virtual void Init(RNTupleModel &model) = 0;
   /// Incorporate incremental changes to the model into the ntuple descriptor. This happens, e.g. if new fields were
   /// added after the initial call to `RPageSink::Init(RNTupleModel &)`.
   /// `firstEntry` specifies the global index for the first stored element in the added columns.
   virtual void UpdateSchema(const RNTupleModelChangeset &changeset, NTupleSize_t firstEntry) = 0;

   /// Write a page to the storage. The column must have been added before.
   virtual void CommitPage(ColumnHandle_t columnHandle, const RPage &page) = 0;
   /// Write a preprocessed page to storage. The column must have been added before.
   virtual void CommitSealedPage(DescriptorId_t physicalColumnId, const RPageStorage::RSealedPage &sealedPage) = 0;
   /// Write a vector of preprocessed pages to storage. The corresponding columns must have been added before.
   virtual void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> ranges) = 0;
   /// Finalize the current cluster and create a new one for the following data.
   /// Returns the number of bytes written to storage (excluding meta-data).
   virtual std::uint64_t CommitCluster(NTupleSize_t nNewEntries) = 0;
   /// Write out the page locations (page list envelope) for all the committed clusters since the last call of
   /// CommitClusterGroup (or the beginning of writing).
   virtual void CommitClusterGroup() = 0;
   /// Finalize the current cluster and the entrire data set.
   virtual void CommitDataset() = 0;

   /// Get a new, empty page for the given column that can be filled with up to nElements.  If nElements is zero,
   /// the page sink picks an appropriate size.
   virtual RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) = 0;

   /// An RAII wrapper used to synchronize a page sink. See GetSinkGuard().
   class RSinkGuard {
      std::mutex *fLock;

   public:
      explicit RSinkGuard(std::mutex *lock) : fLock(lock)
      {
         if (fLock != nullptr) {
            fLock->lock();
         }
      }
      RSinkGuard(const RSinkGuard &) = delete;
      RSinkGuard &operator=(const RSinkGuard &) = delete;
      RSinkGuard(RSinkGuard &&) = delete;
      RSinkGuard &operator=(RSinkGuard &&) = delete;
      ~RSinkGuard()
      {
         if (fLock != nullptr) {
            fLock->unlock();
         }
      }
   };

   virtual RSinkGuard GetSinkGuard()
   {
      // By default, there is no lock and the guard does nothing.
      return RSinkGuard(nullptr);
   }
}; // class RPageSink

// clang-format off
/**
\class ROOT::Experimental::Internal::RPagePersistentSink
\ingroup NTuple
\brief Base class for a sink with a physical storage backend
*/
// clang-format on
class RPagePersistentSink : public RPageSink {
private:
   /// Used to map the IDs of the descriptor to the physical IDs issued during header/footer serialization
   RNTupleSerializer::RContext fSerializationContext;

   /// Remembers the starting cluster id for the next cluster group
   std::uint64_t fNextClusterInGroup = 0;
   /// Used to calculate the number of entries in the current cluster
   NTupleSize_t fPrevClusterNEntries = 0;
   /// Keeps track of the number of elements in the currently open cluster. Indexed by column id.
   std::vector<RClusterDescriptor::RColumnRange> fOpenColumnRanges;
   /// Keeps track of the written pages in the currently open cluster. Indexed by column id.
   std::vector<RClusterDescriptor::RPageRange> fOpenPageRanges;

protected:
   Internal::RNTupleDescriptorBuilder fDescriptorBuilder;

   /// Default I/O performance counters that get registered in fMetrics
   struct RCounters {
      Detail::RNTupleAtomicCounter &fNPageCommitted;
      Detail::RNTupleAtomicCounter &fSzWritePayload;
      Detail::RNTupleAtomicCounter &fSzZip;
      Detail::RNTupleAtomicCounter &fTimeWallWrite;
      Detail::RNTupleAtomicCounter &fTimeWallZip;
      Detail::RNTupleTickCounter<Detail::RNTupleAtomicCounter> &fTimeCpuWrite;
      Detail::RNTupleTickCounter<Detail::RNTupleAtomicCounter> &fTimeCpuZip;
   };
   std::unique_ptr<RCounters> fCounters;

   virtual void InitImpl(unsigned char *serializedHeader, std::uint32_t length) = 0;

   virtual RNTupleLocator CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page) = 0;
   virtual RNTupleLocator
   CommitSealedPageImpl(DescriptorId_t physicalColumnId, const RPageStorage::RSealedPage &sealedPage) = 0;
   /// Vector commit of preprocessed pages. The `ranges` array specifies a range of sealed pages to be
   /// committed for each column.  The returned vector contains, in order, the RNTupleLocator for each
   /// page on each range in `ranges`, i.e. the first N entries refer to the N pages in `ranges[0]`,
   /// followed by M entries that refer to the M pages in `ranges[1]`, etc.
   /// The default is to call `CommitSealedPageImpl` for each page; derived classes may provide an
   /// optimized implementation though.
   virtual std::vector<RNTupleLocator> CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges);
   /// Returns the number of bytes written to storage (excluding metadata)
   virtual std::uint64_t CommitClusterImpl() = 0;
   /// Returns the locator of the page list envelope of the given buffer that contains the serialized page list.
   /// Typically, the implementation takes care of compressing and writing the provided buffer.
   virtual RNTupleLocator CommitClusterGroupImpl(unsigned char *serializedPageList, std::uint32_t length) = 0;
   virtual void CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length) = 0;

   /// Enables the default set of metrics provided by RPageSink. `prefix` will be used as the prefix for
   /// the counters registered in the internal RNTupleMetrics object.
   /// This set of counters can be extended by a subclass by calling `fMetrics.MakeCounter<...>()`.
   ///
   /// A subclass using the default set of metrics is always responsible for updating the counters
   /// appropriately, e.g. `fCounters->fNPageCommited.Inc()`
   void EnableDefaultMetrics(const std::string &prefix);

public:
   RPagePersistentSink(std::string_view ntupleName, const RNTupleWriteOptions &options);

   RPagePersistentSink(const RPagePersistentSink &) = delete;
   RPagePersistentSink &operator=(const RPagePersistentSink &) = delete;
   RPagePersistentSink(RPagePersistentSink &&) = default;
   RPagePersistentSink &operator=(RPagePersistentSink &&) = default;
   ~RPagePersistentSink() override;

   /// Guess the concrete derived page source from the location
   static std::unique_ptr<RPageSink> Create(std::string_view ntupleName, std::string_view location,
                                            const RNTupleWriteOptions &options = RNTupleWriteOptions());

   ColumnHandle_t AddColumn(DescriptorId_t fieldId, const RColumn &column) final;

   /// Updates the descriptor and calls InitImpl() that handles the backend-specific details (file, DAOS, etc.)
   void Init(RNTupleModel &model) final;
   void UpdateSchema(const RNTupleModelChangeset &changeset, NTupleSize_t firstEntry) final;

   void CommitPage(ColumnHandle_t columnHandle, const RPage &page) final;
   void CommitSealedPage(DescriptorId_t physicalColumnId, const RPageStorage::RSealedPage &sealedPage) final;
   void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> ranges) final;
   std::uint64_t CommitCluster(NTupleSize_t nEntries) final;
   void CommitClusterGroup() final;
   void CommitDataset() final;
}; // class RPagePersistentSink

// clang-format off
/**
\class ROOT::Experimental::Internal::RPageSource
\ingroup NTuple
\brief Abstract interface to read data from an ntuple

The page source is initialized with the columns of interest. Alias columns from projected fields are mapped to the
corresponding physical columns. Pages from the columns of interest can then be mapped into memory.
The page source also gives access to the ntuple's meta-data.
*/
// clang-format on
class RPageSource : public RPageStorage {
public:
   /// Used in SetEntryRange / GetEntryRange
   struct REntryRange {
      NTupleSize_t fFirstEntry = kInvalidNTupleIndex;
      NTupleSize_t fNEntries = 0;

      /// Returns true if the given cluster has entries within the entry range
      bool IntersectsWith(const RClusterDescriptor &clusterDesc) const;
   };

   /// An RAII wrapper used for the read-only access to RPageSource::fDescriptor. See GetExclDescriptorGuard().
   class RSharedDescriptorGuard {
      const RNTupleDescriptor &fDescriptor;
      std::shared_mutex &fLock;

   public:
      RSharedDescriptorGuard(const RNTupleDescriptor &desc, std::shared_mutex &lock) : fDescriptor(desc), fLock(lock)
      {
         fLock.lock_shared();
      }
      RSharedDescriptorGuard(const RSharedDescriptorGuard &) = delete;
      RSharedDescriptorGuard &operator=(const RSharedDescriptorGuard &) = delete;
      RSharedDescriptorGuard(RSharedDescriptorGuard &&) = delete;
      RSharedDescriptorGuard &operator=(RSharedDescriptorGuard &&) = delete;
      ~RSharedDescriptorGuard() { fLock.unlock_shared(); }
      const RNTupleDescriptor *operator->() const { return &fDescriptor; }
      const RNTupleDescriptor &GetRef() const { return fDescriptor; }
   };

   /// An RAII wrapper used for the writable access to RPageSource::fDescriptor. See GetSharedDescriptorGuard().
   class RExclDescriptorGuard {
      RNTupleDescriptor &fDescriptor;
      std::shared_mutex &fLock;

   public:
      RExclDescriptorGuard(RNTupleDescriptor &desc, std::shared_mutex &lock) : fDescriptor(desc), fLock(lock)
      {
         fLock.lock();
      }
      RExclDescriptorGuard(const RExclDescriptorGuard &) = delete;
      RExclDescriptorGuard &operator=(const RExclDescriptorGuard &) = delete;
      RExclDescriptorGuard(RExclDescriptorGuard &&) = delete;
      RExclDescriptorGuard &operator=(RExclDescriptorGuard &&) = delete;
      ~RExclDescriptorGuard()
      {
         fDescriptor.IncGeneration();
         fLock.unlock();
      }
      RNTupleDescriptor *operator->() const { return &fDescriptor; }
      void MoveIn(RNTupleDescriptor &&desc) { fDescriptor = std::move(desc); }
   };

private:
   RNTupleDescriptor fDescriptor;
   mutable std::shared_mutex fDescriptorLock;
   REntryRange fEntryRange; ///< Used by the cluster pool to prevent reading beyond the given range

protected:
   /// Default I/O performance counters that get registered in fMetrics
   struct RCounters {
      Detail::RNTupleAtomicCounter &fNReadV;
      Detail::RNTupleAtomicCounter &fNRead;
      Detail::RNTupleAtomicCounter &fSzReadPayload;
      Detail::RNTupleAtomicCounter &fSzReadOverhead;
      Detail::RNTupleAtomicCounter &fSzUnzip;
      Detail::RNTupleAtomicCounter &fNClusterLoaded;
      Detail::RNTupleAtomicCounter &fNPageLoaded;
      Detail::RNTupleAtomicCounter &fNPagePopulated;
      Detail::RNTupleAtomicCounter &fTimeWallRead;
      Detail::RNTupleAtomicCounter &fTimeWallUnzip;
      Detail::RNTupleTickCounter<Detail::RNTupleAtomicCounter> &fTimeCpuRead;
      Detail::RNTupleTickCounter<Detail::RNTupleAtomicCounter> &fTimeCpuUnzip;
      Detail::RNTupleCalcPerf &fBandwidthReadUncompressed;
      Detail::RNTupleCalcPerf &fBandwidthReadCompressed;
      Detail::RNTupleCalcPerf &fBandwidthUnzip;
      Detail::RNTupleCalcPerf &fFractionReadOverhead;
      Detail::RNTupleCalcPerf &fCompressionRatio;
   };

   /// Keeps track of the requested physical column IDs. When using alias columns (projected fields), physical
   /// columns may be requested multiple times.
   class RActivePhysicalColumns {
   private:
      std::vector<DescriptorId_t> fIDs;
      std::vector<std::size_t> fRefCounters;

   public:
      void Insert(DescriptorId_t physicalColumnID);
      void Erase(DescriptorId_t physicalColumnID);
      RCluster::ColumnSet_t ToColumnSet() const;
   };

   std::unique_ptr<RCounters> fCounters;

   RNTupleReadOptions fOptions;
   /// The active columns are implicitly defined by the model fields or views
   RActivePhysicalColumns fActivePhysicalColumns;

   /// Helper to unzip pages and header/footer; comprises a 16MB (kMAXZIPBUF) unzip buffer.
   /// Not all page sources need a decompressor (e.g. virtual ones for chains and friends don't), thus we
   /// leave it up to the derived class whether or not the decompressor gets constructed.
   std::unique_ptr<RNTupleDecompressor> fDecompressor;

   virtual RNTupleDescriptor AttachImpl() = 0;
   // Only called if a task scheduler is set. No-op be default.
   virtual void UnzipClusterImpl(RCluster * /* cluster */)
      { }

   /// Helper for unstreaming a page. This is commonly used in derived, concrete page sources.  The implementation
   /// currently always makes a memory copy, even if the sealed page is uncompressed and in the final memory layout.
   /// The optimization of directly mapping pages is left to the concrete page source implementations.
   /// Usage of this method requires construction of fDecompressor. Memory is allocated via
   /// `RPageAllocatorHeap`; use `RPageAllocatorHeap::DeletePage()` to deallocate returned pages.
   RPage UnsealPage(const RSealedPage &sealedPage, const RColumnElementBase &element, DescriptorId_t physicalColumnId);

   /// Prepare a page range read for the column set in `clusterKey`.  Specifically, pages referencing the
   /// `kTypePageZero` locator are filled in `pageZeroMap`; otherwise, `perPageFunc` is called for each page. This is
   /// commonly used as part of `LoadClusters()` in derived classes.
   void PrepareLoadCluster(
      const RCluster::RKey &clusterKey, ROnDiskPageMap &pageZeroMap,
      std::function<void(DescriptorId_t, NTupleSize_t, const RClusterDescriptor::RPageRange::RPageInfo &)> perPageFunc);

   /// Enables the default set of metrics provided by RPageSource. `prefix` will be used as the prefix for
   /// the counters registered in the internal RNTupleMetrics object.
   /// A subclass using the default set of metrics is responsible for updating the counters
   /// appropriately, e.g. `fCounters->fNRead.Inc()`
   /// Alternatively, a subclass might provide its own RNTupleMetrics object by overriding the
   /// GetMetrics() member function.
   void EnableDefaultMetrics(const std::string &prefix);

   /// Note that the underlying lock is not recursive. See GetSharedDescriptorGuard() for further information.
   RExclDescriptorGuard GetExclDescriptorGuard() { return RExclDescriptorGuard(fDescriptor, fDescriptorLock); }

public:
   RPageSource(std::string_view ntupleName, const RNTupleReadOptions &fOptions);
   RPageSource(const RPageSource&) = delete;
   RPageSource& operator=(const RPageSource&) = delete;
   RPageSource(RPageSource &&) = delete;
   RPageSource &operator=(RPageSource &&) = delete;
   ~RPageSource() override;
   /// Guess the concrete derived page source from the file name (location)
   static std::unique_ptr<RPageSource> Create(std::string_view ntupleName, std::string_view location,
                                              const RNTupleReadOptions &options = RNTupleReadOptions());
   /// Open the same storage multiple time, e.g. for reading in multiple threads
   virtual std::unique_ptr<RPageSource> Clone() const = 0;

   EPageStorageType GetType() final { return EPageStorageType::kSource; }
   const RNTupleReadOptions &GetReadOptions() const { return fOptions; }

   /// Takes the read lock for the descriptor. Multiple threads can take the lock concurrently.
   /// The underlying std::shared_mutex, however, is neither read nor write recursive:
   /// within one thread, only one lock (shared or exclusive) must be acquired at the same time. This requires special
   /// care in sections protected by GetSharedDescriptorGuard() and GetExclDescriptorGuard() especially to avoid that
   /// the locks are acquired indirectly (e.g. by a call to GetNEntries()).
   /// As a general guideline, no other method of the page source should be called (directly or indirectly) in a
   /// guarded section.
   const RSharedDescriptorGuard GetSharedDescriptorGuard() const
   {
      return RSharedDescriptorGuard(fDescriptor, fDescriptorLock);
   }

   ColumnHandle_t AddColumn(DescriptorId_t fieldId, const RColumn &column) override;
   void DropColumn(ColumnHandle_t columnHandle) override;

   /// Open the physical storage container for the tree
   void Attach() { GetExclDescriptorGuard().MoveIn(AttachImpl()); }
   NTupleSize_t GetNEntries();
   NTupleSize_t GetNElements(ColumnHandle_t columnHandle);
   ColumnId_t GetColumnId(ColumnHandle_t columnHandle);

   /// Promise to only read from the given entry range. If set, prevents the cluster pool from reading-ahead beyond
   /// the given range. The range needs to be within [0, GetNEntries()).
   void SetEntryRange(const REntryRange &range);
   REntryRange GetEntryRange() const { return fEntryRange; }

   /// Allocates and fills a page that contains the index-th element
   virtual RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex) = 0;
   /// Another version of PopulatePage that allows to specify cluster-relative indexes
   virtual RPage PopulatePage(ColumnHandle_t columnHandle, RClusterIndex clusterIndex) = 0;

   /// Read the packed and compressed bytes of a page into the memory buffer provided by selaedPage. The sealed page
   /// can be used subsequently in a call to RPageSink::CommitSealedPage.
   /// The fSize and fNElements member of the sealedPage parameters are always set. If sealedPage.fBuffer is nullptr,
   /// no data will be copied but the returned size information can be used by the caller to allocate a large enough
   /// buffer and call LoadSealedPage again.
   virtual void
   LoadSealedPage(DescriptorId_t physicalColumnId, RClusterIndex clusterIndex, RSealedPage &sealedPage) = 0;

   /// Populates all the pages of the given cluster ids and columns; it is possible that some columns do not
   /// contain any pages.  The page source may load more columns than the minimal necessary set from `columns`.
   /// To indicate which columns have been loaded, LoadClusters() must mark them with SetColumnAvailable().
   /// That includes the ones from the `columns` that don't have pages; otherwise subsequent requests
   /// for the cluster would assume an incomplete cluster and trigger loading again.
   /// LoadClusters() is typically called from the I/O thread of a cluster pool, i.e. the method runs
   /// concurrently to other methods of the page source.
   virtual std::vector<std::unique_ptr<RCluster>> LoadClusters(std::span<RCluster::RKey> clusterKeys) = 0;

   /// Parallel decompression and unpacking of the pages in the given cluster. The unzipped pages are supposed
   /// to be preloaded in a page pool attached to the source. The method is triggered by the cluster pool's
   /// unzip thread. It is an optional optimization, the method can safely do nothing. In particular, the
   /// actual implementation will only run if a task scheduler is set. In practice, a task scheduler is set
   /// if implicit multi-threading is turned on.
   void UnzipCluster(RCluster *cluster);
}; // class RPageSource

} // namespace Internal

} // namespace Experimental
} // namespace ROOT

#endif
