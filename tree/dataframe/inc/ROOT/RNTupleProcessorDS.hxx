/// \file RNTupleProcessorDS.hxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2025-06-18
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleProcessorDS
#define ROOT_RNTupleProcessorDS

#include <ROOT/RDataSource.hxx>
#include <ROOT/RNTupleProcessor.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <string_view>

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace ROOT {
class RDataFrame;
} // namespace ROOT

namespace ROOT::Experimental::RDF {
class RNTupleProcessorDS final : public ROOT::RDF::RDataSource {
   std::unique_ptr<RNTupleProcessor> fProcessor;

   /// Connects the IDs of active proto fields and their subfields to their fully qualified name (a.b.c.d).
   /// This enables the column reader to rewire the field IDs when the file changes (chain),
   /// using the fully qualified name as a search key in the descriptor of the other page sources.
   std::vector<std::string> fColumnNames;
   std::vector<std::string> fColumnTypes;

   std::vector<ROOT::Experimental::Internal::RDF::RNTupleProcessorColumnReader *> fActiveColumnReaders;

   /// \brief Holds useful information about fields added to the RNTupleProcessorDS
   struct RFieldInfo {
      ROOT::DescriptorId_t fFieldId;
      std::string fFieldName;
      std::string fTypeName;
      std::size_t fNRepetitions;
      // Enable `std::vector::emplace_back` for this type
      RFieldInfo(ROOT::DescriptorId_t fieldId, std::string_view fieldName, std::string_view typeName,
                 std::size_t nRepetitions)
         : fFieldId(fieldId), fFieldName(fieldName), fTypeName(typeName), fNRepetitions(nRepetitions)
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
   void AddField(const ROOT::RFieldBase &field, std::string_view colName, std::vector<RFieldInfo> fieldInfos,
                 bool convertToRVec = true);

public:
   RNTupleProcessorDS(std::unique_ptr<ROOT::Experimental::RNTupleProcessor> processor);
   // Rule of five
   RNTupleProcessorDS(const RNTupleProcessorDS &) = delete;
   RNTupleProcessorDS &operator=(const RNTupleProcessorDS &) = delete;
   RNTupleProcessorDS(RNTupleProcessorDS &&) = delete;
   RNTupleProcessorDS &operator=(RNTupleProcessorDS &&) = delete;
   ~RNTupleProcessorDS() final;

   void SetNSlots(unsigned int nSlots) final;
   // FIXME(fdegeus) get correct number of files (needs to be added in RNTupleProcessor)
   std::size_t GetNFiles() const final { return 1; }
   const std::vector<std::string> &GetColumnNames() const final { return fColumnNames; }
   bool HasColumn(std::string_view colName) const final;
   std::string GetTypeName(std::string_view colName) const final;
   std::vector<std::pair<ULong64_t, ULong64_t>> GetEntryRanges() final;
   std::string GetLabel() final { return "RNTupleProcessorDS"; }

   void Initialize() final;
   void InitSlot(unsigned int slot, ULong64_t firstEntry) final;
   void FinalizeSlot(unsigned int slot) final;
   void Finalize() final;

   std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
   GetColumnReaders(unsigned int slot, std::string_view name, const std::type_info &) final;

   ROOT::RDF::RSampleInfo
   CreateSampleInfo(unsigned int,
                    const std::unordered_map<std::string, ROOT::RDF::Experimental::RSample *> &) const final;

   // Old API, unused
   bool SetEntry(unsigned int, ULong64_t) final;

protected:
   Record_t GetColumnReadersImpl(std::string_view name, const std::type_info &) final;
};
} // namespace ROOT::Experimental::RDF

namespace ROOT::Experimental::RDF {
RDataFrame FromRNTupleProcessor(std::unique_ptr<ROOT::Experimental::RNTupleProcessor> processor);
} // namespace ROOT::Experimental::RDF

#endif // ROOT_RNTupleProcessorDS
