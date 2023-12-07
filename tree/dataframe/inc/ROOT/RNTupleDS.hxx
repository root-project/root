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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace ROOT {
namespace Experimental {

class RNTuple;
class RNTupleDescriptor;

namespace Detail {
class RFieldBase;
class RFieldValue;
class RPageSource;
} // namespace Detail

namespace Internal {
class RNTupleColumnReader;
}

class RNTupleDS final : public ROOT::RDF::RDataSource {
   friend class Internal::RNTupleColumnReader;

   /// The PrepareNextRanges() method populates the fNextRanges list with REntryRangeDS records.
   /// The GetEntryRanges() swaps fNextRanges and fCurrentRanges and uses the list of
   /// REntryRangeDS records to return the list of ranges ready to use by the RDF loop manager.
   struct REntryRangeDS {
      std::unique_ptr<ROOT::Experimental::Detail::RPageSource> fSource;
      ULong64_t fFirstEntry = 0; ///< First entry index in fSource
      /// End entry index in fSource, e.g. the number of entries in the range is fLastEntry - fFirstEntry
      ULong64_t fLastEntry = 0;
   };

   /// The first source is used to extract the schema and build the prototype fields. The page source
   /// is used to extract a clone of the descriptor to fPrincipalDescriptor. Afterwards it is moved
   /// into the first REntryRangeDS.
   std::unique_ptr<Detail::RPageSource> fPrincipalSource;
   /// A clone of the first pages source's descriptor.
   std::unique_ptr<RNTupleDescriptor> fPrincipalDescriptor;

   /// The data source may be constructed with an ntuple name and a list of files
   std::string fNTupleName;
   std::vector<std::string> fFileNames;
   std::size_t fNextFileIndex = 0; ///< Index into fFileNames to the next file to process

   /// We prepare a prototype field for every column. If a column reader is actually requested
   /// in GetColumnReaders(), we move a clone of the field into a new column reader for RDataFrame.
   /// Only the clone connects to the backing page store and acquires I/O resources.
   /// The field IDs are set in the context of the first source and used as keys in fFieldId2QualifiedName.
   std::vector<std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>> fProtoFields;
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

   /// Provides the RDF column "colName" given the field identified by fieldID. For records and collections,
   /// AddField recurses into the sub fields. The skeinIDs is the list of field IDs of the outer collections
   /// of fieldId. For instance, if fieldId refers to an `std::vector<Jet>`, with
   /// struct Jet {
   ///    float pt;
   ///    float eta;
   /// };
   /// AddField will recurse into Jet.pt and Jet.eta and provide the two inner fields as std::vector<float> each.
   void AddField(const RNTupleDescriptor &desc,
                 std::string_view colName,
                 DescriptorId_t fieldId,
                 std::vector<DescriptorId_t> skeinIDs);

   /// Populates fNextRanges with the next set of entry ranges. Opens files from the chain as necessary
   /// and aligns ranges with cluster boundaries for scheduling the tail of files.
   /// Upon return, the fNextRanges list is ordered.  It has usually fNSlots elements; fewer if there
   /// is not enough work to give at least one cluster to every slot.
   void PrepareNextRanges();

public:
   explicit RNTupleDS(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource);
   RNTupleDS(std::string_view ntupleName, const std::vector<std::string> &fileNames);
   ~RNTupleDS();

   void SetNSlots(unsigned int nSlots) final;
   const std::vector<std::string> &GetColumnNames() const final { return fColumnNames; }
   bool HasColumn(std::string_view colName) const final;
   std::string GetTypeName(std::string_view colName) const final;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final;
   std::string GetLabel() final { return "RNTupleDS"; }

   bool SetEntry(unsigned int slot, ULong64_t entry) final;

   void Initialize() final;
   void InitSlot(unsigned int slot, ULong64_t firstEntry) final;
   void FinalizeSlot(unsigned int slot) final;
   void Finalize() final;

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
   GetColumnReaders(unsigned int /*slot*/, std::string_view /*name*/, const std::type_info &) final;

protected:
   Record_t GetColumnReadersImpl(std::string_view name, const std::type_info &) final;
};

} // ns Experimental

namespace RDF {
namespace Experimental {
RDataFrame FromRNTuple(std::string_view ntupleName, std::string_view fileName);
RDataFrame FromRNTuple(ROOT::Experimental::RNTuple *ntuple);
RDataFrame FromRNTuple(std::string_view ntupleName, const std::vector<std::string> &fileNames);
} // namespace Experimental
} // namespace RDF

} // ns ROOT

#endif
