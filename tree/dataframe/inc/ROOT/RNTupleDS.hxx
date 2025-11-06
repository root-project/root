/// \file RNTupleDS.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Enrico Guiraud <enrico.guiraud@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleDS
#define ROOT_RNTupleDS

#include <ROOT/RDataSource.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleTypes.hxx>
#include <string_view>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

// Follow RDF namespace convention
namespace ROOT {
class RDataFrame;
}
namespace ROOT::Internal::RDF {
/**
 * \brief Internal overload of the function that allows passing a range of entries
 *
 * The event range will be respected when processing this RNTuple. It is assumed
 * that processing happens within one thread only.
 */
ROOT::RDataFrame FromRNTuple(std::string_view ntupleName, const std::vector<std::string> &fileNames,
                             const std::pair<ULong64_t, ULong64_t> &range);
/**
 * \brief Retrieves the cluster boundaries and the number of entries for the input RNTuple
 *
 * \param[in] ntupleName The name of the RNTuple dataset
 * \param[in] location The location of the RNTuple dataset (e.g. a path to a file)
 *
 * \note This function is a helper for the Python side to avoid having to deal
 *       with the shared descriptor guard.
 */
std::pair<std::vector<ROOT::Internal::RNTupleClusterBoundaries>, ROOT::NTupleSize_t>
GetClustersAndEntries(std::string_view ntupleName, std::string_view location);
} // namespace ROOT::Internal::RDF

namespace ROOT {
class RFieldBase;
class RDataFrame;
class RNTuple;
} // namespace ROOT
namespace ROOT::Internal::RDF {
class RNTupleColumnReader;
}
namespace ROOT::Internal {
class RPageSource;
}

namespace ROOT::RDF {
class RNTupleDS final : public ROOT::RDF::RDataSource {
   friend class ROOT::Internal::RDF::RNTupleColumnReader;

   /// The PrepareNextRanges() method populates the fNextRanges list with REntryRangeDS records.
   /// The GetEntryRanges() swaps fNextRanges and fCurrentRanges and uses the list of
   /// REntryRangeDS records to return the list of ranges ready to use by the RDF loop manager.
   struct REntryRangeDS {
      std::unique_ptr<ROOT::Internal::RPageSource> fSource;
      ULong64_t fFirstEntry = 0; ///< First entry index in fSource
      /// End entry index in fSource, e.g. the number of entries in the range is fLastEntry - fFirstEntry
      ULong64_t fLastEntry = 0;
      std::string_view fFileName; ///< Storage location of the current RNTuple
   };

   /// A clone of the first pages source's descriptor.
   ROOT::RNTupleDescriptor fPrincipalDescriptor;

   /// The data source may be constructed with an ntuple name and a list of files
   std::string fNTupleName;
   std::vector<std::string> fFileNames;
   /// The staging area is relevant for chains of files, i.e. when fFileNames is not empty. In this case,
   /// files are opened in the background in batches of size `fNSlots` and kept in the staging area.
   /// The first file (chains or no chains) is always opened on construction in order to process the schema.
   /// For all subsequent files, the corresponding page sources in the staging area only executed `LoadStructure()`,
   /// i.e. they should have a compressed buffer of the meta-data available.
   /// Concretely:
   ///   1. We open the first file on construction to read the schema and then move the corresponding page source
   ///      in the staging area.
   ///   2. On `Initialize()`, we start the I/O background thread, which in turn opens the first batch of files.
   ///   3. At the beginning of `GetEntryRanges()`, we
   ///      a) wait for the I/O thread to finish,
   ///      b) call `PrepareNextRanges()` in the main thread to move the page sources from the staging area
   ///         into `fNextRanges`; this will also call `Attach()` on the page sources (i.e., deserialize the meta-data),
   ///         and
   ///      c) trigger staging of the next batch of files in the I/O background thread.
   ///   4. On `Finalize()`, the I/O background thread is stopped.
   std::vector<std::unique_ptr<ROOT::Internal::RPageSource>> fStagingArea;
   std::size_t fNextFileIndex = 0; ///< Index into fFileNames to the next file to process

   /// We prepare a prototype field for every column. If a column reader is actually requested
   /// in GetColumnReaders(), we move a clone of the field into a new column reader for RDataFrame.
   /// Only the clone connects to the backing page store and acquires I/O resources.
   /// The field IDs are set in the context of the first source and used as keys in fFieldId2QualifiedName.
   std::vector<std::unique_ptr<ROOT::RFieldBase>> fProtoFields;
   /// Columns may be requested with types other than with which they were initially added as proto fields. For example,
   /// a column with a `ROOT::RVec<float>` proto field may instead be requested as a `std::vector<float>`. In case this
   /// happens, we create an alternative proto field and store it here, with the original index in `fProtoFields` as
   /// key. A single column can have more than one alternative proto fields.
   std::unordered_map<std::size_t, std::vector<std::unique_ptr<ROOT::RFieldBase>>> fAlternativeProtoFields;
   /// Connects the IDs of active proto fields and their subfields to their fully qualified name (a.b.c.d).
   /// This enables the column reader to rewire the field IDs when the file changes (chain),
   /// using the fully qualified name as a search key in the descriptor of the other page sources.
   std::unordered_map<ROOT::DescriptorId_t, std::string> fFieldId2QualifiedName;
   std::vector<std::string> fColumnNames;
   std::vector<std::string> fColumnTypes;
   /// List of column readers returned by GetColumnReaders() organized by slot. Used to reconnect readers
   /// to new page sources when the files in the chain change.
   std::vector<std::vector<ROOT::Internal::RDF::RNTupleColumnReader *>> fActiveColumnReaders;

   ULong64_t fSeenEntriesNoGlobalRange = 0; ///< The number of entries seen so far in GetEntryRanges()

   std::vector<REntryRangeDS> fCurrentRanges; ///< Basis for the ranges returned by the last GetEntryRanges() call
   std::vector<REntryRangeDS> fNextRanges;    ///< Basis for the ranges populated by the PrepareNextRanges() call
   /// Maps the first entries from the ranges of the last GetEntryRanges() call to their corresponding index in
   /// the fCurrentRanges vectors.  This is necessary because the returned ranges get distributed arbitrarily
   /// onto slots.  In the InitSlot method, the column readers use this map to find the correct range to connect to.
   std::unordered_map<ULong64_t, std::size_t> fFirstEntry2RangeIdx;
   // Keep track of the scheduled entries - necessary for processing of GlobalEntries
   std::vector<std::pair<ULong64_t, ULong64_t>> fOriginalRanges;
   /// One element per slot, corresponding to the current range index for that slot, as filled by InitSlot
   std::vector<std::size_t> fSlotsToRangeIdxs;

   /// The background thread that runs StageNextSources()
   std::thread fThreadStaging;
   /// Protects the shared state between the main thread and the I/O thread
   std::mutex fMutexStaging;
   /// Signal for the state information of fIsReadyForStaging and fHasNextSources
   std::condition_variable fCvStaging;
   /// Is true when the staging thread should start working
   bool fIsReadyForStaging = false;
   /// Is true when the staging thread has populated the next batch of files to fStagingArea
   bool fHasNextSources = false;
   /// Is true when the I/O thread should quit
   bool fStagingThreadShouldTerminate = false;

   /// \brief Holds useful information about fields added to the RNTupleDS
   struct RFieldInfo {
      ROOT::DescriptorId_t fFieldId;
      std::size_t fNRepetitions;
      // Enable `std::vector::emplace_back` for this type
      RFieldInfo(ROOT::DescriptorId_t fieldId, std::size_t nRepetitions)
         : fFieldId(fieldId), fNRepetitions(nRepetitions)
      {
      }
   };

   /// Provides the RDF column "colName" given the field identified by fieldID. For records and collections,
   /// AddField recurses into the sub fields. The fieldInfos argument is a list of objects holding info
   /// about the fields of the outer collection(s) (w.r.t. fieldId). For instance, if fieldId refers to an
   /// `std::vector<Jet>`, with
   /// ~~~{.cpp}
   /// struct Jet {
   ///    float pt;
   ///    float eta;
   /// };
   /// ~~~
   /// AddField will recurse into `Jet.pt` and `Jet.eta` and provide the two inner fields as `ROOT::VecOps::RVec<float>`
   /// each.
   ///
   /// In case the field is a collection of type `ROOT::VecOps::RVec`, `std::vector` or `std::array`, its corresponding
   /// column is added as a `ROOT::VecOps::RVec`. Otherwise, the collection field's on-disk type is used. Note, however,
   /// that inner record members of such collections will still be added as `ROOT::VecOps::RVec` (e.g., `std::set<Jet>
   /// will be added as a `std::set`, but `Jet.[pt|eta] will be added as `ROOT::VecOps::RVec<float>).
   void AddField(const ROOT::RNTupleDescriptor &desc, std::string_view colName, ROOT::DescriptorId_t fieldId,
                 std::vector<RFieldInfo> fieldInfos, bool convertToRVec = true);

   /// The main function of the fThreadStaging background thread
   void ExecStaging();
   /// Starting from `fNextFileIndex`, opens the next `fNSlots` files. Calls `LoadStructure()` on the opened files.
   /// The very first file is already available from the constructor.
   void StageNextSources();
   /// Populates fNextRanges with the next set of entry ranges. Moves files from the staging area as necessary
   /// and aligns ranges with cluster boundaries for scheduling the tail of files.
   /// Upon return, the fNextRanges list is ordered.  It has usually fNSlots elements; fewer if there
   /// is not enough work to give at least one cluster to every slot.
   void PrepareNextRanges();

   explicit RNTupleDS(std::unique_ptr<ROOT::Internal::RPageSource> pageSource);

   ROOT::RFieldBase *GetFieldWithTypeChecks(std::string_view fieldName, const std::type_info &tid);

   friend ROOT::RDataFrame ROOT::Internal::RDF::FromRNTuple(std::string_view ntupleName,
                                                            const std::vector<std::string> &fileNames,
                                                            const std::pair<ULong64_t, ULong64_t> &range);

   explicit RNTupleDS(std::string_view ntupleName, const std::vector<std::string> &fileNames,
                      const std::pair<ULong64_t, ULong64_t> &range);

public:
   RNTupleDS(std::string_view ntupleName, std::string_view fileName);
   RNTupleDS(std::string_view ntupleName, const std::vector<std::string> &fileNames);
   // Rule of five
   RNTupleDS(const RNTupleDS &) = delete;
   RNTupleDS &operator=(const RNTupleDS &) = delete;
   RNTupleDS(RNTupleDS &&) = delete;
   RNTupleDS &operator=(RNTupleDS &&) = delete;
   ~RNTupleDS() final;

   void SetNSlots(unsigned int nSlots) final;
   std::size_t GetNFiles() const final { return fFileNames.empty() ? 1 : fFileNames.size(); }
   const std::vector<std::string> &GetColumnNames() const final { return fColumnNames; }
   bool HasColumn(std::string_view colName) const final;
   std::string GetTypeName(std::string_view colName) const final;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final;
   std::string GetLabel() final { return "RNTupleDS"; }

   void Initialize() final;
   void InitSlot(unsigned int slot, ULong64_t firstEntry) final;
   void FinalizeSlot(unsigned int slot) final;
   void Finalize() final;

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
   GetColumnReaders(unsigned int /*slot*/, std::string_view /*name*/, const std::type_info &) final;

   ROOT::RDF::RSampleInfo
   CreateSampleInfo(unsigned int,
                    const std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> &) const final;

   // Old API, unused
   bool SetEntry(unsigned int, ULong64_t) final { return true; }

protected:
   Record_t GetColumnReadersImpl(std::string_view name, const std::type_info &) final;
};
} // namespace ROOT::RDF

namespace ROOT::RDF {
RDataFrame FromRNTuple(std::string_view ntupleName, std::string_view fileName);
RDataFrame FromRNTuple(std::string_view ntupleName, const std::vector<std::string> &fileNames);
} // namespace ROOT::RDF

#endif
