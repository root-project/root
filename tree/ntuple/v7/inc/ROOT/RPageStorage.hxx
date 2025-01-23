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

#include <ROOT/RError.hxx>
#include <ROOT/RCluster.hxx>
#include <ROOT/RColumnElementBase.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RSpan.hxx>
#include <string_view>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ROOT {
namespace Experimental {

class RNTupleModel;

namespace Internal {
class RColumn;
class RNTupleCompressor;
struct RNTupleModelChangeset;
class RPageAllocator;

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
   /// The page checksum is a 64bit xxhash3
   static constexpr std::size_t kNBytesPageChecksum = sizeof(std::uint64_t);

   /// The interface of a task scheduler to schedule page (de)compression tasks
   class RTaskScheduler {
   public:
      virtual ~RTaskScheduler() = default;
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
   private:
      const void *fBuffer = nullptr;
      std::size_t fBufferSize = 0; ///< Size of the page payload and the trailing checksum (if available)
      std::uint32_t fNElements = 0;
      bool fHasChecksum = false; ///< If set, the last 8 bytes of the buffer are the xxhash of the rest of the buffer

   public:
      RSealedPage() = default;
      RSealedPage(const void *buffer, std::size_t bufferSize, std::uint32_t nElements, bool hasChecksum = false)
         : fBuffer(buffer), fBufferSize(bufferSize), fNElements(nElements), fHasChecksum(hasChecksum)
      {
      }
      RSealedPage(const RSealedPage &other) = default;
      RSealedPage &operator=(const RSealedPage &other) = default;
      RSealedPage(RSealedPage &&other) = default;
      RSealedPage &operator=(RSealedPage &&other) = default;

      const void *GetBuffer() const { return fBuffer; }
      void SetBuffer(const void *buffer) { fBuffer = buffer; }

      std::size_t GetDataSize() const
      {
         assert(fBufferSize >= fHasChecksum * kNBytesPageChecksum);
         return fBufferSize - fHasChecksum * kNBytesPageChecksum;
      }
      std::size_t GetBufferSize() const { return fBufferSize; }
      void SetBufferSize(std::size_t bufferSize) { fBufferSize = bufferSize; }

      std::uint32_t GetNElements() const { return fNElements; }
      void SetNElements(std::uint32_t nElements) { fNElements = nElements; }

      bool GetHasChecksum() const { return fHasChecksum; }
      void SetHasChecksum(bool hasChecksum) { fHasChecksum = hasChecksum; }

      void ChecksumIfEnabled();
      RResult<void> VerifyChecksumIfEnabled() const;
      /// Returns a failure if the sealed page has no checksum
      RResult<std::uint64_t> GetChecksum() const;
   };

   using SealedPageSequence_t = std::deque<RSealedPage>;
   /// A range of sealed pages referring to the same column that can be used for vector commit
   struct RSealedPageGroup {
      DescriptorId_t fPhysicalColumnId;
      SealedPageSequence_t::const_iterator fFirst;
      SealedPageSequence_t::const_iterator fLast;

      RSealedPageGroup() = default;
      RSealedPageGroup(DescriptorId_t d, SealedPageSequence_t::const_iterator b, SealedPageSequence_t::const_iterator e)
         : fPhysicalColumnId(d), fFirst(b), fLast(e)
      {
      }
   };

protected:
   Detail::RNTupleMetrics fMetrics;

   /// For the time being, we will use the heap allocator for all sources and sinks. This may change in the future.
   std::unique_ptr<RPageAllocator> fPageAllocator;

   std::string fNTupleName;
   RTaskScheduler *fTaskScheduler = nullptr;
   void WaitForAllTasks()
   {
      if (!fTaskScheduler)
         return;
      fTaskScheduler->Wait();
   }

public:
   explicit RPageStorage(std::string_view name);
   RPageStorage(const RPageStorage &other) = delete;
   RPageStorage &operator=(const RPageStorage &other) = delete;
   RPageStorage(RPageStorage &&other) = default;
   RPageStorage &operator=(RPageStorage &&other) = default;
   virtual ~RPageStorage();

   /// Whether the concrete implementation is a sink or a source
   virtual EPageStorageType GetType() = 0;

   struct RColumnHandle {
      DescriptorId_t fPhysicalId = kInvalidDescriptorId;
      RColumn *fColumn = nullptr;

      /// Returns true for a valid column handle; fColumn and fPhysicalId should always either both
      /// be valid or both be invalid.
      explicit operator bool() const { return fPhysicalId != kInvalidDescriptorId && fColumn; }
   };
   /// The column handle identifies a column with the current open page storage
   using ColumnHandle_t = RColumnHandle;

   /// Register a new column.  When reading, the column must exist in the ntuple on disk corresponding to the meta-data.
   /// When writing, every column can only be attached once.
   virtual ColumnHandle_t AddColumn(DescriptorId_t fieldId, RColumn &column) = 0;
   /// Unregisters a column.  A page source decreases the reference counter for the corresponding active column.
   /// For a page sink, dropping columns is currently a no-op.
   virtual void DropColumn(ColumnHandle_t columnHandle) = 0;
   DescriptorId_t GetColumnId(ColumnHandle_t columnHandle) const { return columnHandle.fPhysicalId; }

   /// Returns the default metrics object.  Subclasses might alternatively provide their own metrics object by
   /// overriding this.
   virtual Detail::RNTupleMetrics &GetMetrics() { return fMetrics; }

   /// Returns the NTuple name.
   const std::string &GetNTupleName() const { return fNTupleName; }

   void SetTaskScheduler(RTaskScheduler *taskScheduler) { fTaskScheduler = taskScheduler; }
}; // class RPageStorage

// clang-format off
/**
\class ROOT::Experimental::Internal::RWritePageMemoryManager
\ingroup NTuple
\brief Helper to maintain a memory budget for the write pages of a set of columns

The memory manager keeps track of the sum of bytes used by the write pages of a set of columns.
It will flush (and shrink) large pages of other columns on the attempt to expand a page.
*/
// clang-format on
class RWritePageMemoryManager {
private:
   struct RColumnInfo {
      RColumn *fColumn = nullptr;
      std::size_t fCurrentPageSize = 0;
      std::size_t fInitialPageSize = 0;

      bool operator>(const RColumnInfo &other) const;
   };

   /// Sum of all the write page sizes (their capacity) of the columns in `fColumnsSortedByPageSize`
   std::size_t fCurrentAllocatedBytes = 0;
   /// Maximum allowed value for `fCurrentAllocatedBytes`, set from RNTupleWriteOptions::fPageBufferBudget
   std::size_t fMaxAllocatedBytes = 0;
   /// All columns that called `ReservePage()` (hence `TryUpdate()`) at least once,
   /// sorted by their current write page size from large to small
   std::set<RColumnInfo, std::greater<RColumnInfo>> fColumnsSortedByPageSize;

   /// Flush columns in order of allocated write page size until the sum of all write page allocations
   /// leaves space for at least targetAvailableSize bytes. Only use columns with a write page size larger
   /// than pageSizeLimit.
   bool TryEvict(std::size_t targetAvailableSize, std::size_t pageSizeLimit);

public:
   explicit RWritePageMemoryManager(std::size_t maxAllocatedBytes) : fMaxAllocatedBytes(maxAllocatedBytes) {}

   /// Try to register the new write page size for the given column. Flush large columns to make space, if necessary.
   /// If not enough space is available after all (sum of write pages would be larger than fMaxAllocatedBytes),
   /// return false.
   bool TryUpdate(RColumn &column, std::size_t newWritePageSize);
};

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
public:
   using Callback_t = std::function<void(RPageSink &)>;

   /// Cluster that was staged, but not yet logically appended to the RNTuple
   struct RStagedCluster {
      std::uint64_t fNBytesWritten = 0;
      NTupleSize_t fNEntries = 0;

      struct RColumnInfo {
         RClusterDescriptor::RPageRange fPageRange;
         NTupleSize_t fNElements = kInvalidNTupleIndex;
         bool fIsSuppressed = false;
      };

      std::vector<RColumnInfo> fColumnInfos;
   };

protected:
   std::unique_ptr<RNTupleWriteOptions> fOptions;

   /// Helper to zip pages and header/footer; includes a 16MB (kMAXZIPBUF) zip buffer.
   /// There could be concrete page sinks that don't need a compressor.  Therefore, and in order to stay consistent
   /// with the page source, we leave it up to the derived class whether or not the compressor gets constructed.
   std::unique_ptr<RNTupleCompressor> fCompressor;

   /// Helper for streaming a page. This is commonly used in derived, concrete page sinks. Note that if
   /// compressionSetting is 0 (uncompressed) and the page is mappable and not checksummed, the returned sealed page
   /// will point directly to the input page buffer.  Otherwise, the sealed page references an internal buffer
   /// of fCompressor.  Thus, the buffer pointed to by the RSealedPage should never be freed.
   /// Usage of this method requires construction of fCompressor.
   RSealedPage SealPage(const RPage &page, const RColumnElementBase &element);

private:
   /// Flag if sink was initialized
   bool fIsInitialized = false;
   std::vector<Callback_t> fOnDatasetCommitCallbacks;
   std::vector<unsigned char> fSealPageBuffer; ///< Used as destination buffer in the simple SealPage overload

   /// Used in ReservePage to maintain the page buffer budget
   RWritePageMemoryManager fWritePageMemoryManager;

public:
   RPageSink(std::string_view ntupleName, const RNTupleWriteOptions &options);

   RPageSink(const RPageSink &) = delete;
   RPageSink &operator=(const RPageSink &) = delete;
   RPageSink(RPageSink &&) = default;
   RPageSink &operator=(RPageSink &&) = default;
   ~RPageSink() override;

   EPageStorageType GetType() final { return EPageStorageType::kSink; }
   /// Returns the sink's write options.
   const RNTupleWriteOptions &GetWriteOptions() const { return *fOptions; }

   void DropColumn(ColumnHandle_t /*columnHandle*/) final {}

   bool IsInitialized() const { return fIsInitialized; }

   /// Return the RNTupleDescriptor being constructed.
   virtual const RNTupleDescriptor &GetDescriptor() const = 0;

   virtual NTupleSize_t GetNEntries() const = 0;

   /// Physically creates the storage container to hold the ntuple (e.g., a keys a TFile or an S3 bucket)
   /// Init() associates column handles to the columns referenced by the model
   void Init(RNTupleModel &model)
   {
      if (fIsInitialized) {
         throw RException(R__FAIL("already initialized"));
      }
      fIsInitialized = true;
      InitImpl(model);
   }

protected:
   virtual void InitImpl(RNTupleModel &model) = 0;
   virtual void CommitDatasetImpl() = 0;

public:
   /// Parameters for the SealPage() method
   struct RSealPageConfig {
      const RPage *fPage = nullptr;                 ///< Input page to be sealed
      const RColumnElementBase *fElement = nullptr; ///< Corresponds to the page's elements, for size calculation etc.
      int fCompressionSetting = 0;                  ///< Compression algorithm and level to apply
      /// Adds a 8 byte little-endian xxhash3 checksum to the page payload. The buffer has to be large enough to
      /// to store the additional 8 bytes.
      bool fWriteChecksum = true;
      /// If false, the output buffer must not point to the input page buffer, which would otherwise be an option
      /// if the page is mappable and should not be compressed
      bool fAllowAlias = false;
      /// Location for sealed output. The memory buffer has to be large enough.
      void *fBuffer = nullptr;
   };

   /// Seal a page using the provided info.
   static RSealedPage SealPage(const RSealPageConfig &config);

   /// Incorporate incremental changes to the model into the ntuple descriptor. This happens, e.g. if new fields were
   /// added after the initial call to `RPageSink::Init(RNTupleModel &)`.
   /// `firstEntry` specifies the global index for the first stored element in the added columns.
   virtual void UpdateSchema(const RNTupleModelChangeset &changeset, NTupleSize_t firstEntry) = 0;
   /// Adds an extra type information record to schema. The extra type information will be written to the
   /// extension header. The information in the record will be merged with the existing information, e.g.
   /// duplicate streamer info records will be removed. This method is called by the "on commit dataset" callback
   /// registered by specific fields (e.g., streamer field) and during merging.
   virtual void UpdateExtraTypeInfo(const RExtraTypeInfoDescriptor &extraTypeInfo) = 0;

   /// Commits a suppressed column for the current cluster. Can be called anytime before CommitCluster().
   /// For any given column and cluster, there must be no calls to both CommitSuppressedColumn() and page commits.
   virtual void CommitSuppressedColumn(ColumnHandle_t columnHandle) = 0;
   /// Write a page to the storage. The column must have been added before.
   virtual void CommitPage(ColumnHandle_t columnHandle, const RPage &page) = 0;
   /// Write a preprocessed page to storage. The column must have been added before.
   virtual void CommitSealedPage(DescriptorId_t physicalColumnId, const RPageStorage::RSealedPage &sealedPage) = 0;
   /// Write a vector of preprocessed pages to storage. The corresponding columns must have been added before.
   virtual void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> ranges) = 0;
   /// Stage the current cluster and create a new one for the following data.
   /// Returns the object that must be passed to CommitStagedClusters to logically append the staged cluster to the
   /// ntuple descriptor.
   virtual RStagedCluster StageCluster(NTupleSize_t nNewEntries) = 0;
   /// Commit staged clusters, logically appending them to the ntuple descriptor.
   virtual void CommitStagedClusters(std::span<RStagedCluster> clusters) = 0;
   /// Finalize the current cluster and create a new one for the following data.
   /// Returns the number of bytes written to storage (excluding meta-data).
   virtual std::uint64_t CommitCluster(NTupleSize_t nNewEntries)
   {
      RStagedCluster stagedClusters[] = {StageCluster(nNewEntries)};
      CommitStagedClusters(stagedClusters);
      return stagedClusters[0].fNBytesWritten;
   }
   /// Write out the page locations (page list envelope) for all the committed clusters since the last call of
   /// CommitClusterGroup (or the beginning of writing).
   virtual void CommitClusterGroup() = 0;

   /// The registered callback is executed at the beginning of CommitDataset();
   void RegisterOnCommitDatasetCallback(Callback_t callback) { fOnDatasetCommitCallbacks.emplace_back(callback); }
   /// Run the registered callbacks and finalize the current cluster and the entrire data set.
   void CommitDataset();

   /// Get a new, empty page for the given column that can be filled with up to nElements;
   /// nElements must be larger than zero.
   virtual RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements);

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

   /// Union of the streamer info records that are sent from streamer fields to the sink before committing the dataset.
   RNTupleSerializer::StreamerInfoMap_t fStreamerInfos;

protected:
   /// Set of optional features supported by the persistent sink
   struct RFeatures {
      bool fCanMergePages = false;
   };

   RFeatures fFeatures;
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
   /// The mask allows to skip writing out certain pages. The vector has the size of all the pages.
   /// For every `false` value in the mask, the corresponding locator is skipped (missing) in the output vector.
   /// The default is to call `CommitSealedPageImpl` for each page; derived classes may provide an
   /// optimized implementation though.
   virtual std::vector<RNTupleLocator>
   CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges, const std::vector<bool> &mask);
   /// Returns the number of bytes written to storage (excluding metadata)
   virtual std::uint64_t StageClusterImpl() = 0;
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

   ColumnHandle_t AddColumn(DescriptorId_t fieldId, RColumn &column) final;

   const RNTupleDescriptor &GetDescriptor() const final { return fDescriptorBuilder.GetDescriptor(); }

   NTupleSize_t GetNEntries() const final { return fPrevClusterNEntries; }

   /// Updates the descriptor and calls InitImpl() that handles the backend-specific details (file, DAOS, etc.)
   void InitImpl(RNTupleModel &model) final;
   void UpdateSchema(const RNTupleModelChangeset &changeset, NTupleSize_t firstEntry) final;
   void UpdateExtraTypeInfo(const RExtraTypeInfoDescriptor &extraTypeInfo) final;

   /// Initialize sink based on an existing descriptor and fill into the descriptor builder.
   void InitFromDescriptor(const RNTupleDescriptor &descriptor);

   void CommitSuppressedColumn(ColumnHandle_t columnHandle) final;
   void CommitPage(ColumnHandle_t columnHandle, const RPage &page) final;
   void CommitSealedPage(DescriptorId_t physicalColumnId, const RPageStorage::RSealedPage &sealedPage) final;
   void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> ranges) final;
   RStagedCluster StageCluster(NTupleSize_t nNewEntries) final;
   void CommitStagedClusters(std::span<RStagedCluster> clusters) final;
   void CommitClusterGroup() final;
   void CommitDatasetImpl() final;
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

   /// An RAII wrapper used for the read-only access to `RPageSource::fDescriptor`. See `GetExclDescriptorGuard()``.
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

   /// An RAII wrapper used for the writable access to `RPageSource::fDescriptor`. See `GetSharedDescriptorGuard()`.
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
      void MoveIn(RNTupleDescriptor desc) { fDescriptor = std::move(desc); }
   };

private:
   RNTupleDescriptor fDescriptor;
   mutable std::shared_mutex fDescriptorLock;
   REntryRange fEntryRange;    ///< Used by the cluster pool to prevent reading beyond the given range
   bool fHasStructure = false; ///< Set to true once `LoadStructure()` is called
   bool fIsAttached = false;   ///< Set to true once `Attach()` is called

   /// Remembers the last cluster id from which a page was requested
   DescriptorId_t fLastUsedCluster = kInvalidDescriptorId;
   /// Clusters from where pages got preloaded in UnzipClusterImpl(), ordered by first entry number
   /// of the clusters. If the last used cluster changes in LoadPage(), all unused pages from
   /// previous clusters are evicted from the page pool.
   std::map<NTupleSize_t, DescriptorId_t> fPreloadedClusters;

   /// Does nothing if fLastUsedCluster == clusterId. Otherwise, updated fLastUsedCluster
   /// and evict unused paged from the page pool of all previous clusters.
   /// Must not be called when the descriptor guard is taken.
   void UpdateLastUsedCluster(DescriptorId_t clusterId);

protected:
   /// Default I/O performance counters that get registered in `fMetrics`
   struct RCounters {
      Detail::RNTupleAtomicCounter &fNReadV;
      Detail::RNTupleAtomicCounter &fNRead;
      Detail::RNTupleAtomicCounter &fSzReadPayload;
      Detail::RNTupleAtomicCounter &fSzReadOverhead;
      Detail::RNTupleAtomicCounter &fSzUnzip;
      Detail::RNTupleAtomicCounter &fNClusterLoaded;
      Detail::RNTupleAtomicCounter &fNPageRead;
      Detail::RNTupleAtomicCounter &fNPageUnsealed;
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

   /// Keeps track of the requested physical column IDs and their in-memory target type via a column element identifier.
   /// When using alias columns (projected fields), physical columns may be requested multiple times.
   class RActivePhysicalColumns {
   public:
      struct RColumnInfo {
         RColumnElementBase::RIdentifier fElementId;
         std::size_t fRefCounter = 0;
      };

   private:
      /// Maps physical column IDs to all the requested in-memory representations.
      /// A pair of physical column ID and in-memory representation can be requested multiple times, which is
      /// indicated by the reference counter.
      /// We can only have a handful of possible in-memory representations for a given column,
      /// so it is fine to search them linearly.
      std::unordered_map<DescriptorId_t, std::vector<RColumnInfo>> fColumnInfos;

   public:
      void Insert(DescriptorId_t physicalColumnId, RColumnElementBase::RIdentifier elementId);
      void Erase(DescriptorId_t physicalColumnId, RColumnElementBase::RIdentifier elementId);
      RCluster::ColumnSet_t ToColumnSet() const;
      bool HasColumnInfos(DescriptorId_t physicalColumnId) const { return fColumnInfos.count(physicalColumnId) > 0; }
      const std::vector<RColumnInfo> &GetColumnInfos(DescriptorId_t physicalColumnId) const
      {
         return fColumnInfos.at(physicalColumnId);
      }
   };

   /// Summarizes cluster-level information that are necessary to load a certain page.
   /// Used by LoadPageImpl().
   struct RClusterInfo {
      DescriptorId_t fClusterId = 0;
      /// Location of the page on disk
      RClusterDescriptor::RPageRange::RPageInfoExtended fPageInfo;
      /// The first element number of the page's column in the given cluster
      std::uint64_t fColumnOffset = 0;
   };

   std::unique_ptr<RCounters> fCounters;

   RNTupleReadOptions fOptions;
   /// The active columns are implicitly defined by the model fields or views
   RActivePhysicalColumns fActivePhysicalColumns;

   /// Pages that are unzipped with IMT are staged into the page pool
   RPagePool fPagePool;

   virtual void LoadStructureImpl() = 0;
   /// `LoadStructureImpl()` has been called before `AttachImpl()` is called
   virtual RNTupleDescriptor AttachImpl() = 0;
   /// Returns a new, unattached page source for the same data set
   virtual std::unique_ptr<RPageSource> CloneImpl() const = 0;
   // Only called if a task scheduler is set. No-op be default.
   virtual void UnzipClusterImpl(RCluster *cluster);
   // Returns a page from storage if not found in the page pool. Should be able to handle zero page locators.
   virtual RPageRef
   LoadPageImpl(ColumnHandle_t columnHandle, const RClusterInfo &clusterInfo, NTupleSize_t idxInCluster) = 0;

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
   /// `GetMetrics()` member function.
   void EnableDefaultMetrics(const std::string &prefix);

   /// Note that the underlying lock is not recursive. See `GetSharedDescriptorGuard()` for further information.
   RExclDescriptorGuard GetExclDescriptorGuard() { return RExclDescriptorGuard(fDescriptor, fDescriptorLock); }

public:
   RPageSource(std::string_view ntupleName, const RNTupleReadOptions &fOptions);
   RPageSource(const RPageSource &) = delete;
   RPageSource &operator=(const RPageSource &) = delete;
   RPageSource(RPageSource &&) = delete;
   RPageSource &operator=(RPageSource &&) = delete;
   ~RPageSource() override;
   /// Guess the concrete derived page source from the file name (location)
   static std::unique_ptr<RPageSource> Create(std::string_view ntupleName, std::string_view location,
                                              const RNTupleReadOptions &options = RNTupleReadOptions());
   /// Open the same storage multiple time, e.g. for reading in multiple threads.
   /// If the source is already attached, the clone will be attached, too. The clone will use, however,
   /// it's own connection to the underlying storage (e.g., file descriptor, XRootD handle, etc.)
   std::unique_ptr<RPageSource> Clone() const;

   /// Helper for unstreaming a page. This is commonly used in derived, concrete page sources.  The implementation
   /// currently always makes a memory copy, even if the sealed page is uncompressed and in the final memory layout.
   /// The optimization of directly mapping pages is left to the concrete page source implementations.
   RResult<RPage> static UnsealPage(const RSealedPage &sealedPage, const RColumnElementBase &element,
                                    RPageAllocator &pageAlloc);

   EPageStorageType GetType() final { return EPageStorageType::kSource; }
   const RNTupleReadOptions &GetReadOptions() const { return fOptions; }

   /// Takes the read lock for the descriptor. Multiple threads can take the lock concurrently.
   /// The underlying `std::shared_mutex`, however, is neither read nor write recursive:
   /// within one thread, only one lock (shared or exclusive) must be acquired at the same time. This requires special
   /// care in sections protected by `GetSharedDescriptorGuard()` and `GetExclDescriptorGuard()` especially to avoid
   /// that the locks are acquired indirectly (e.g. by a call to `GetNEntries()`). As a general guideline, no other
   /// method of the page source should be called (directly or indirectly) in a guarded section.
   const RSharedDescriptorGuard GetSharedDescriptorGuard() const
   {
      return RSharedDescriptorGuard(fDescriptor, fDescriptorLock);
   }

   ColumnHandle_t AddColumn(DescriptorId_t fieldId, RColumn &column) override;
   void DropColumn(ColumnHandle_t columnHandle) override;

   /// Loads header and footer without decompressing or deserializing them. This can be used to asynchronously open
   /// a file in the background. The method is idempotent and it is called as a first step in `Attach()`.
   /// Pages sources may or may not make use of splitting loading and processing meta-data.
   /// Therefore, `LoadStructure()` may do nothing and defer loading the meta-data to `Attach()`.
   void LoadStructure();
   /// Open the physical storage container and deserialize header and footer
   void Attach();
   NTupleSize_t GetNEntries();
   NTupleSize_t GetNElements(ColumnHandle_t columnHandle);

   /// Promise to only read from the given entry range. If set, prevents the cluster pool from reading-ahead beyond
   /// the given range. The range needs to be within `[0, GetNEntries())`.
   void SetEntryRange(const REntryRange &range);
   REntryRange GetEntryRange() const { return fEntryRange; }

   /// Allocates and fills a page that contains the index-th element. The default implementation searches
   /// the page and calls LoadPageImpl(). Returns a default-constructed RPage for suppressed columns.
   virtual RPageRef LoadPage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex);
   /// Another version of `LoadPage` that allows to specify cluster-relative indexes.
   /// Returns a default-constructed RPage for suppressed columns.
   virtual RPageRef LoadPage(ColumnHandle_t columnHandle, RNTupleLocalIndex localIndex);

   /// Read the packed and compressed bytes of a page into the memory buffer provided by `sealedPage`. The sealed page
   /// can be used subsequently in a call to `RPageSink::CommitSealedPage`.
   /// The `fSize` and `fNElements` member of the sealedPage parameters are always set. If `sealedPage.fBuffer` is
   /// `nullptr`, no data will be copied but the returned size information can be used by the caller to allocate a large
   /// enough buffer and call `LoadSealedPage` again.
   virtual void
   LoadSealedPage(DescriptorId_t physicalColumnId, RNTupleLocalIndex localIndex, RSealedPage &sealedPage) = 0;

   /// Populates all the pages of the given cluster ids and columns; it is possible that some columns do not
   /// contain any pages.  The page source may load more columns than the minimal necessary set from `columns`.
   /// To indicate which columns have been loaded, `LoadClusters()`` must mark them with `SetColumnAvailable()`.
   /// That includes the ones from the `columns` that don't have pages; otherwise subsequent requests
   /// for the cluster would assume an incomplete cluster and trigger loading again.
   /// `LoadClusters()` is typically called from the I/O thread of a cluster pool, i.e. the method runs
   /// concurrently to other methods of the page source.
   virtual std::vector<std::unique_ptr<RCluster>> LoadClusters(std::span<RCluster::RKey> clusterKeys) = 0;

   /// Parallel decompression and unpacking of the pages in the given cluster. The unzipped pages are supposed
   /// to be preloaded in a page pool attached to the source. The method is triggered by the cluster pool's
   /// unzip thread. It is an optional optimization, the method can safely do nothing. In particular, the
   /// actual implementation will only run if a task scheduler is set. In practice, a task scheduler is set
   /// if implicit multi-threading is turned on.
   void UnzipCluster(RCluster *cluster);

   // TODO(gparolini): for symmetry with SealPage(), we should either make this private or SealPage() public.
   RResult<RPage> UnsealPage(const RSealedPage &sealedPage, const RColumnElementBase &element);
}; // class RPageSource

} // namespace Internal

} // namespace Experimental
} // namespace ROOT

#endif
