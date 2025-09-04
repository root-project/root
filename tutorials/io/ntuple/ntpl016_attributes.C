/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Using Attributes to add meta-data to an RNTuple.
///
/// \macro_image
/// \macro_code
///
/// \date July 2025
/// \author The ROOT Team

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleAttributes.hxx>

#include <TFile.h>

#include <iostream>

static const char *fileName = "my_rntuple_with_attrs.root";
static const char *ntplName = "ntuple";
      
static void Write()
{
   // Step 1: create the RNTuple
   auto model = ROOT::RNTupleModel::Create();
   auto pInt = model->MakeField<int>("int");
   auto file = std::unique_ptr<TFile>(TFile::Open(fileName, "RECREATE"));
   auto writer = ROOT::RNTupleWriter::Append(std::move(model), ntplName, *file);

   // Step 2: create the model for the Attribute Set
   auto attrModel = ROOT::RNTupleModel::Create();
   auto pMyAttr = attrModel->MakeField<std::string>("myAttr");

   // Step 3: create the Attribute Set from the main writer
   auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "MyAttrSet");

   // Step 4: start an attribute range.
   // attrRange is a "pending range" which needs to be committed in order to write the attribute range to disk.
   // All entries added to the main writer will be associated to the set of attributes assigned
   // to the committed entry until the call to CommitRange().
   auto attrRange = attrSet->BeginRange();

   // Step 5: assign attribute values.
   // Values can be assigned anywhere between BeginRange() and CommitRange().
   *pMyAttr = "This is a custom attribute";

   // Step 6: fill the data inside the RNTuple
   for (int i = 0; i < 100; ++i) {
      *pInt = i;
      writer->Fill();
   }

   // Step 7: commit the attribute range.
   // CommitRange() with a single argument uses the values found in attrModel's default entry;
   // otherwise one may pass a reference to a REntry as the second argument in order to use its values instead
   // (similarly to RNTupleWriter::Fill()).
   attrSet->CommitRange(std::move(attrRange));
}

static void Read()
{
   auto reader = ROOT::RNTupleReader::Open(ntplName, fileName);

   // Read back the list of available attribute sets
   const auto attrSets = reader->GetDescriptor().GetAttributeSetNames();
   std::cout << "Number of Attribute Sets inside " << fileName << ":" << ntplName << ": " << attrSets.size() << "\n";
   for (const auto &name : attrSets) {
      std::cout << "  - " << name << "\n";
   }

   // Fetch a specific attribute set
   auto attrSet = reader->OpenAttributeSet("MyAttrSet");

   // Read attributes belonging to a specific entry index.
   // (Just read the first 5 values to avoid spamming)
   auto attrEntry = attrSet->CreateAttrEntry();
   auto pAttr = attrEntry->GetPtr<std::string>("myAttr");
   for (int i = 0; i < 5; ++i) {
      // Note that an entry may have multiple attributes associated, so we need to loop over them.
      for (auto idx : attrSet->GetAttributes(i)) {
         // LoadAttrEntry may be called with 1 or 2 arguments, similarly to RNTupleReader::LoadEntry().
         // If no entry is passed, the values will be available through the attrSet's model's default entry.
         attrSet->LoadAttrEntry(idx, *attrEntry);
         std::cout << "Entry " << i << " has attribute: myAttr = " << *pAttr << "\n";
      }
   }

   // Read all attributes inside the Attribute Set
   std::cout << "All attribute ranges:\n";
   for (auto idx : attrSet->GetAttributes()) {
      // LoadAttrEntry return the range covered by the idx-th attribute entry.
      const auto attrRange = attrSet->LoadAttrEntry(idx, *attrEntry);
      const auto [first, last] = attrRange.GetFirstLast().value();
      std::cout << "  - myAttr has value \"" << *pAttr << "\" over range [" << first << ", " << last << "]\n";
   }
}

void ntpl016_attributes()
{
   Write();
   Read();
}
