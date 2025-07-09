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
   auto pAttr = attrModel->MakeField<std::string>("myAttr");

   // Step 3: create the Attribute Set from the main writer
   auto attrSet = writer->CreateAttributeSet("MyAttrSet", std::move(attrModel));

   // Step 4: start an attribute range. attrRange has basically the same interface as REntry.
   // All entries added to the main writer will be associated to the set of attributes assigned
   // to attrEntry until the call to CommitRange().
   auto attrEntry = attrSet->BeginRange();

   auto &wModel = writer->GetModel();

   // Step 5: assign attribute values.
   // Values can be assigned anywhere between BeginRange() and CommitRange().
   auto pMyAttr = attrEntry.GetPtr<std::string>("myAttr");
   *pMyAttr = "This is a custom attribute";

   // Step 6: fill the data inside the RNTuple
   for (int i = 0; i < 100; ++i) {
      auto entry = wModel.CreateEntry();
      *pInt = i;
      writer->Fill(*entry);
   }

   // Step 7: commit the attribute range
   attrSet->CommitRange(std::move(attrEntry));
}

static void Read()
{
   auto reader = ROOT::RNTupleReader::Open(ntplName, fileName);

   // Read back the list of available attribute sets
   const auto attrSets = reader->GetDescriptor().GetAttributeSets();
   std::cout << "Number of Attribute Sets inside " << fileName << ":" << ntplName << ": " << (attrSets.size(), 1) << "\n";
   for (const auto &[name, _] : attrSets) {
      std::cout << "  - " << name << "\n";
   }

   // Fetch a specific attribute set
   auto attrSet = reader->OpenAttributeSet("MyAttrSet");

   // Read attributes belonging to a specific entry index.
   // (Just read the first 5 values to avoid spamming)
   for (int i = 0; i < 5; ++i) {
      // Note that an entry may have multiple attributes associated, so we need to loop over them.
      for (const auto &attrEntry : attrSet->GetAttributes(i)) {
         auto pAttr = attrEntry.GetPtr<std::string>("myAttr");
         std::cout << "Entry " << i << " has attribute: myAttr = " << *pAttr << "\n";
      }
   }

   // Read all attributes inside the Attribute Set
   std::cout << "All attribute ranges:\n";
   for (const auto &attrEntry : attrSet->GetAttributes()) {
      auto pAttr = attrEntry.GetPtr<std::string>("myAttr");
      const auto [first, last] = attrEntry.GetRange().GetFirstLast().value();
      std::cout << "  - myAttr has value \"" << *pAttr << "\" over range [" << first << ", " << last << "]\n";
   }
}

void ntpl016_attributes()
{
   Write();
   Read();
}
