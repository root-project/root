/// \file RNTupleDS.cxx
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

#include <ROOT/RDF/RColumnReaderBase.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldValue.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleDS.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RStringView.hxx>

#include <TError.h>

#include <string>
#include <vector>
#include <typeinfo>
#include <utility>

namespace ROOT {
namespace Experimental {
namespace Detail {

/// An artifical field that transforms an RNTuple column that contains the offset of collections into
/// collection sizes. It is used to provide the "number of" RDF columns for collections, e.g.
/// `#jets` for a collection named `jets`.
class RRDFCardinalityField : public ROOT::Experimental::Detail::RFieldBase {
public:
   ROOT::Experimental::RField<ClusterSize_t> fOffsetField;

   static std::string TypeName() { return "ROOT::Experimental::ClusterSize_t::ValueType"; }
   RRDFCardinalityField()
     : Detail::RFieldBase("", TypeName(), ENTupleStructure::kLeaf, false /* isSimple */)
     , fOffsetField("")
   {
   }
   RRDFCardinalityField(RRDFCardinalityField&& other) = default;
   RRDFCardinalityField& operator =(RRDFCardinalityField&& other) = default;
   ~RRDFCardinalityField() = default;
   std::unique_ptr<Detail::RFieldBase> Clone(std::string_view /* newName */) const final {
      return std::make_unique<RRDFCardinalityField>();
   }

   /// Being virtual, the field doesn't have any columns of its own but it uses, indirectly, the columns
   /// of fOffsetField when reading
   void GenerateColumnsImpl() final { }

   ROOT::Experimental::Detail::RFieldValue GenerateValue(void* where) final
   {
      return Detail::RFieldValue(this, static_cast<ClusterSize_t *>(where));
   }
   Detail::RFieldValue CaptureValue(void *where) final
   {
      return Detail::RFieldValue(true /* captureFlag */, this, where);
   }
   size_t GetValueSize() const final { return sizeof(ClusterSize_t); }

   /// Get the number of elements of the collection identified by globalIndex
   void ReadGlobalImpl(ROOT::Experimental::NTupleSize_t globalIndex,
                       ROOT::Experimental::Detail::RFieldValue *value) final
   {
      RClusterIndex collectionStart;
      fOffsetField.GetCollectionInfo(globalIndex, &collectionStart, value->Get<ClusterSize_t>());
   }

   /// Get the number of elements of the collection identified by clusterIndex
   void ReadInClusterImpl(const ROOT::Experimental::RClusterIndex &clusterIndex,
                          ROOT::Experimental::Detail::RFieldValue *value) final
   {
      RClusterIndex collectionStart;
      fOffsetField.GetCollectionInfo(clusterIndex, &collectionStart, value->Get<ClusterSize_t>());
   }
};


/// Every RDF column is represented by exactly one RNTuple field
class RNTupleColumnReader : public ROOT::Detail::RDF::RColumnReaderBase {
protected:
   using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
   using RFieldValue = ROOT::Experimental::Detail::RFieldValue;
   using RPageSource = ROOT::Experimental::Detail::RPageSource;

   std::unique_ptr<RFieldBase> fField; ///< The field backing the RDF column
   std::vector<DescriptorId_t> fSkeinIDs; ///< For inner collections, the field IDs of the outer collections
   RFieldValue fValue; ///< The memory location used to read from fField
   Long64_t fLastEntry; ///< Last entry number that was read

public:
   RNTupleColumnReader(std::unique_ptr<RFieldBase> f, const std::vector<DescriptorId_t> &skeinIDs)
      : fField(std::move(f)), fSkeinIDs(skeinIDs), fValue(fField->GenerateValue()), fLastEntry(-1)
   {
   }
   virtual ~RNTupleColumnReader() { fField->DestroyValue(fValue); }

   /// Column readers are created as prtotypes and then cloned for every slot
   virtual std::unique_ptr<RNTupleColumnReader> Clone() const = 0;
   /// The RNTuple columns of fField and the outer collection fields given by fSkeinIDs are connected
   /// to the RNTuple page source only on demand.
   virtual void Connect(RPageSource &source) = 0;

   void *GetImpl(Long64_t entry) final
   {
      if (entry != fLastEntry) {
         fField->Read(entry, &fValue);
         fLastEntry = entry;
      }
      return fValue.GetRawPtr();
   }
};

/// Provides the size of a collection as an RDF column
class RNTupleCardinalityColumnReader : public RNTupleColumnReader {
public:
   RNTupleCardinalityColumnReader(std::unique_ptr<RFieldBase> f, const std::vector<DescriptorId_t> &skeinIDs)
      : RNTupleColumnReader(std::move(f), skeinIDs)
   {
   }

   std::unique_ptr<RNTupleColumnReader> Clone() const final {
      return std::make_unique<RNTupleCardinalityColumnReader>(fField->Clone(fField->GetName()), fSkeinIDs);
   }

   void Connect(RPageSource &source) final {
      auto f = fField.get();
      for (unsigned i = 0; i < fSkeinIDs.size() - 1; ++i) {
         Detail::RFieldFuse::Connect(fSkeinIDs[i], source, *f);
         f = f->GetSubFields()[0];
      }
      auto cardinalityField = dynamic_cast<RRDFCardinalityField *>(f);
      Detail::RFieldFuse::Connect(fSkeinIDs.back(), source, cardinalityField->fOffsetField);
   }
};

/// Provides a particular field of the RNTuple as RDF column
class RNTupleProjectionColumnReader : public RNTupleColumnReader {
public:
   RNTupleProjectionColumnReader(std::unique_ptr<RFieldBase> f, const std::vector<DescriptorId_t> &skeinIDs)
      : RNTupleColumnReader(std::move(f), skeinIDs)
   {
   }

   std::unique_ptr<RNTupleColumnReader> Clone() const final {
      return std::make_unique<RNTupleProjectionColumnReader>(fField->Clone(fField->GetName()), fSkeinIDs);
   }

   void Connect(RPageSource &source) final {
      auto f = fField.get();
      for (unsigned i = 0; i < fSkeinIDs.size() - 1; ++i) {
         Detail::RFieldFuse::Connect(fSkeinIDs[i], source, *f);
         f = f->GetSubFields()[0];
      }
      Detail::RFieldFuse::ConnectRecursively(fSkeinIDs.back(), source, *f);
   }
};

} // namespace Detail


RNTupleDS::~RNTupleDS() = default;


void RNTupleDS::AddProjection(
   const RNTupleDescriptor &desc, std::string_view colName, DescriptorId_t fieldId,
   std::vector<DescriptorId_t> skeinIDs)
{
   const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);
   if (fieldDesc.GetStructure() == ENTupleStructure::kCollection) {
      // Inner fields of collections are provided as projected collections of only that inner field
      skeinIDs.emplace_back(fieldId);
      // There should only be one sub field but it's easiest to access via the sub field range
      for (const auto& f : desc.GetFieldRange(fieldDesc.GetId())) {
         AddProjection(desc, colName, f.GetId(), skeinIDs);
      }
      // Note that at the end of the recursion, we handled the inner collections as well as the
      // collection as whole (e.g. we have RDF columns std::vector<jet.pt>, std::vector<float> jet.eta)
      // _and_ std::vector<Jet> jet. So we are done.
      return;
   } else if (fieldDesc.GetStructure() == ENTupleStructure::kRecord) {
      // Inner fields of records are provided as individual RDF columns
      for (const auto& f : desc.GetFieldRange(fieldDesc.GetId())) {
         auto innerName = colName.empty() ? f.GetFieldName() : (std::string(colName) + "." + f.GetFieldName());
         AddProjection(desc, innerName, f.GetId(), skeinIDs);
      }
   }

   // The class of fieldId might not be loaded in which case only the inner fields are made available
   auto fieldOrException = Detail::RFieldBase::Create("", fieldDesc.GetTypeName());
   if (!fieldOrException)
      return;
   auto valueField = fieldOrException.Unwrap();
   std::unique_ptr<Detail::RFieldBase> cardinalityField;
   // Collections get the additional "number of" RDF column (e.g. `#jets`)
   if (!skeinIDs.empty())
      cardinalityField = std::make_unique<Detail::RRDFCardinalityField>();

   std::string typeName;
   for (unsigned int i = 0; i < skeinIDs.size(); ++i) {
      valueField = std::make_unique<ROOT::Experimental::RVectorField>("", std::move(valueField));
      if (i < skeinIDs.size() - 1)
         cardinalityField = std::make_unique<ROOT::Experimental::RVectorField>("", std::move(cardinalityField));
   }

   if (cardinalityField) {
      fColumnNames.emplace_back(std::string("#") + std::string(colName));
      fColumnTypes.emplace_back(cardinalityField->GetType());
      auto cardColReader =
         std::make_unique<Detail::RNTupleCardinalityColumnReader>(std::move(cardinalityField), skeinIDs);
      fColumnReaderPrototypes.emplace_back(std::move(cardColReader));
   }

   skeinIDs.emplace_back(fieldId);
   fColumnNames.emplace_back(colName);
   fColumnTypes.emplace_back(valueField->GetType());
   auto valColReader = std::make_unique<Detail::RNTupleProjectionColumnReader>(std::move(valueField), skeinIDs);
   fColumnReaderPrototypes.emplace_back(std::move(valColReader));
}


RNTupleDS::RNTupleDS(std::unique_ptr<Detail::RPageSource> pageSource)
{
   pageSource->Attach();
   const auto &descriptor = pageSource->GetDescriptor();
   fSources.emplace_back(std::move(pageSource));

   AddProjection(descriptor, "", descriptor.GetFieldZeroId(), std::vector<DescriptorId_t>());
}

RDF::RDataSource::Record_t RNTupleDS::GetColumnReadersImpl(std::string_view /* name */, const std::type_info & /* ti */)
{
   // This datasource uses the GetColumnReaders2 API instead (better name in the works)
   return {};
}

std::unique_ptr<ROOT::Detail::RDF::RColumnReaderBase>
RNTupleDS::GetColumnReaders(unsigned int slot, std::string_view name, const std::type_info & /*tid*/)
{
   // TODO(jblomer): check incoming type
   const auto index = std::distance(fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), name));
   auto clone = fColumnReaderPrototypes[index]->Clone();
   clone->Connect(*fSources[slot]);
   return clone;
}

bool RNTupleDS::SetEntry(unsigned int, ULong64_t)
{
   return true;
}

std::vector<std::pair<ULong64_t, ULong64_t>> RNTupleDS::GetEntryRanges()
{
   // TODO(jblomer): use cluster boundaries for the entry ranges
   std::vector<std::pair<ULong64_t, ULong64_t>> ranges;
   if (fHasSeenAllRanges) return ranges;

   auto nEntries = fSources[0]->GetNEntries();
   const auto chunkSize = nEntries / fNSlots;
   const auto reminder = 1U == fNSlots ? 0 : nEntries % fNSlots;
   auto start = 0UL;
   auto end = 0UL;
   for (auto i : ROOT::TSeqU(fNSlots)) {
      start = end;
      end += chunkSize;
      ranges.emplace_back(start, end);
      (void)i;
   }
   ranges.back().second += reminder;
   fHasSeenAllRanges = true;
   return ranges;
}


std::string RNTupleDS::GetTypeName(std::string_view colName) const
{
   const auto index = std::distance(
      fColumnNames.begin(), std::find(fColumnNames.begin(), fColumnNames.end(), colName));
   return fColumnTypes[index];
}


bool RNTupleDS::HasColumn(std::string_view colName) const
{
   return std::find(fColumnNames.begin(), fColumnNames.end(), colName) !=
          fColumnNames.end();
}


void RNTupleDS::Initialise()
{
   fHasSeenAllRanges = false;
}


void RNTupleDS::Finalise()
{
}


void RNTupleDS::SetNSlots(unsigned int nSlots)
{
   R__ASSERT(fNSlots == 0);
   R__ASSERT(nSlots > 0);
   fNSlots = nSlots;

   for (unsigned int i = 1; i < fNSlots; ++i) {
      fSources.emplace_back(fSources[0]->Clone());
      R__ASSERT(i == (fSources.size() - 1));
      fSources[i]->Attach();
   }
}
} // ns Experimental
} // ns ROOT


ROOT::RDataFrame ROOT::Experimental::MakeNTupleDataFrame(std::string_view ntupleName, std::string_view fileName)
{
   auto pageSource = ROOT::Experimental::Detail::RPageSource::Create(ntupleName, fileName);
   ROOT::RDataFrame rdf(std::make_unique<RNTupleDS>(std::move(pageSource)));
   return rdf;
}
