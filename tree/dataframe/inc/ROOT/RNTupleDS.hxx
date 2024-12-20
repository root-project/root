/// \file RNTupleDS.hxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Enrico Guiraud <enrico.guiraud@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleDS
#define ROOT_RNTupleDS

#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDataSource.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <string_view>

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>

namespace ROOT {
class RNTuple;

namespace Experimental {
class RFieldBase;
class RNTupleDescriptor;

namespace Internal {
class RNTupleColumnReader;
class RPageSource;
}

class RNTupleDS final : public ROOT::RDF::RDataSource {
   friend class Internal::RNTupleColumnReader;

   /// The PrepareNextRanges() method populates the fNextRanges list with REntryRangeDS records.
   /// The GetEntryRanges() swaps fNextRanges and fCurrentRanges and uses the list of
   /// REntryRangeDS records to return the list of ranges ready to use by the RDF loop manager.
   struct REntryRangeDS {
      std::unique_ptr<ROOT::Experimental::Internal::RPageSource> fSource;
      ULong64_t fFirstEntry = 0; ///< First entry index in fSource
      /// End entry index in fSource, e.g. the number of entries in the range is fLastEntry - fFirstEntry
      ULong64_t fLastEntry = 0;
   };

   /// A clone of the first pages source's descriptor.
   std::unique_ptr<RNTupleDescriptor> fPrincipalDescriptor;

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
   std::vector<std::unique_ptr<ROOT::Experimental::Internal::RPageSource>> fStagingArea;
   std::size_t fNextFileIndex = 0; ///< Index into fFileNames to the next file to process

   /// We prepare a prototype field for every column. If a column reader is actually requested
   /// in GetColumnReaders(), we move a clone of the field into a new column reader for RDataFrame.
   /// Only the clone connects to the backing page store and acquires I/O resources.
   /// The field IDs are set in the context of the first source and used as keys in fFieldId2QualifiedName.
   std::vector<std::unique_ptr<ROOT::Experimental::RFieldBase>> fProtoFields;
   /// Connects the IDs of active proto fields and their subfields to their fully qualified name (a.b.c.d).
   /// This enables the column reader to rewire the field IDs when the file changes (chain),
   /// using the fully qualified name as a search key in the descriptor of the other page sources.
   std::unordered_map<ROOT::Experimental::DescriptorId_t, std::string> fFieldId2QualifiedName;
   std::vector<std::string> fColumnNames;
   std::vector<std::string> fColumnTypes;
   /// List of column readers returned by GetColumnReaders() organized by slot. Used to reconnect readers
   /// to new page sources when the files in the chain change.
   std::vector<std::vector<Internal::RNTupleColumnReader *>> fActiveColumnReaders;

   unsigned int fNSlots = 0;
   ULong64_t fSeenEntries = 0;                ///< The number of entries so far returned by GetEntryRanges()
   std::vector<REntryRangeDS> fCurrentRanges; ///< Basis for the ranges returned by the last GetEntryRanges() call
   std::vector<REntryRangeDS> fNextRanges;    ///< Basis for the ranges populated by the PrepareNextRanges() call
   /// Maps the first entries from the ranges of the last GetEntryRanges() call to their corresponding index in
   /// the fCurrentRanges vectors.  This is necessary because the returned ranges get distributed arbitrarily
   /// onto slots.  In the InitSlot method, the column readers use this map to find the correct range to connect to.
   std::unordered_map<ULong64_t, std::size_t> fFirstEntry2RangeIdx;

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
      DescriptorId_t fFieldId;
      std::size_t fNRepetitions;
      // Enable `std::vector::emplace_back` for this type
      RFieldInfo(DescriptorId_t fieldId, std::size_t nRepetitions) : fFieldId(fieldId), fNRepetitions(nRepetitions) {}
   };

   /// Provides the RDF column "colName" given the field identified by fieldID. For records and collections,
   /// AddField recurses into the sub fields. The fieldInfos argument is a list of objects holding info
   /// about the fields of the outer collection(s) (w.r.t. fieldId). For instance, if fieldId refers to an
   /// `std::vector<Jet>`, with
   /// struct Jet {
   ///    float pt;
   ///    float eta;
   /// };
   /// AddField will recurse into Jet.pt and Jet.eta and provide the two inner fields as std::vector<float> each.
   void AddField(const RNTupleDescriptor &desc, std::string_view colName, DescriptorId_t fieldId,
                 std::vector<RFieldInfo> fieldInfos);

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

   explicit RNTupleDS(std::unique_ptr<ROOT::Experimental::Internal::RPageSource> pageSource);

public:
   RNTupleDS(std::string_view ntupleName, std::string_view fileName);
   RNTupleDS(ROOT::RNTuple *ntuple);
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

   // Old API, unused
   bool SetEntry(unsigned int, ULong64_t) final { return true; }

protected:
   Record_t GetColumnReadersImpl(std::string_view name, const std::type_info &) final;
};

} // namespace Experimental

namespace RDF {
namespace Experimental {
RDataFrame FromRNTuple(std::string_view ntupleName, std::string_view fileName);
RDataFrame FromRNTuple(std::string_view ntupleName, const std::vector<std::string> &fileNames);
RDataFrame FromRNTuple(ROOT::RNTuple *ntuple);
} // namespace Experimental
} // namespace RDF

} // ns ROOT

#endif
