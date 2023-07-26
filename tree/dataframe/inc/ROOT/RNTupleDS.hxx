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
   /// Clones of the first source, one for each slot
   std::vector<std::unique_ptr<ROOT::Experimental::Detail::RPageSource>> fSources;

   /// The data source may be constructed with an ntuple name and a list of files
   std::string fNTupleName;
   std::vector<std::string> fFileNames;
   std::size_t fNextFileIndex = 0;

   /// We prepare a prototype field for every column. If a column reader is actually requested
   /// in GetColumnReaders(), we move a clone of the field into a new column reader for RDataFrame.
   /// Only the clone connects to the backing page store and acquires I/O resources.
   std::vector<std::unique_ptr<ROOT::Experimental::Detail::RFieldBase>> fFieldPrototypes;
   std::vector<std::string> fColumnNames;
   std::vector<std::string> fColumnTypes;
   /// List of column readers returned by GetColumnReaders()
   std::vector<ROOT::Experimental::Internal::RNTupleColumnReader *> fActiveColumnReaders;
   /// Connects the IDs of active fields and their subfields to their fully qualified name (a.b.c.d).
   /// This enables us to reset the field IDs when the file changes (chain).
   std::unordered_map<ROOT::Experimental::DescriptorId_t, std::string> fFieldId2QualifiedName;

   unsigned fNSlots = 0;
   bool fHasSeenAllRanges = false;
   ULong64_t fSeenEntries = 0; ///< The number of entries so far returned by GetEntryRanges()

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

   /// Re-attach all active column readers to fFileNames[fNextFileIndex]
   void SwitchFile();

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
} // namespace Experimental
} // namespace RDF

} // ns ROOT

#endif
