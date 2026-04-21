/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example using RNTuple attributes
///
/// RNTuple Attributes are the way to store custom metadata in RNTuple.
///
/// They work by associating rows of user-defined metadata to ranges of entries in their parent RNTuple
/// (called the "main RNTuple"). These rows of metadata, called "attribute entries", are
/// defined much like a regular RNTuple by an RNTupleModel and they belong to an "attribute set".
///
/// An attribute set is a standalone collection of attributes which is linked to one and only one RNTuple.
/// Attribute sets are identified by their name and, similarly to RNTuples, they are created with
/// an associated Model and can be written and read via bespoke classes (RNTupleAttrSetWriter/Reader).
/// These classes are never used by themselves but are created from an existing RNTupleWriter or Reader.
///
/// Each main RNTuple can have an arbitrary number of associated attribute sets, though usually one is
/// enough for most purposes.
/// This tutorial shows how to create, write and read back attributes from an RNTuple.
///
/// NOTE: The RNTuple attributes are experimental at this point.
/// Functionality and interface are still subject to changes.
///
/// \macro_code
///
/// \date April 2026
/// \author The ROOT Team

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TFile.h>
#include <TRandom3.h>

#include <iostream>
#include <memory>
#include <utility>

constexpr const char *kFileName = "ntpl019_attributes.root";
constexpr const char *kNTupleName = "ntpl";

struct Event {
   // ... some event data here ...
   double pt = 0.;
};

static void Write()
{
   // Define the model of our main RNTuple.
   auto model = ROOT::RNTupleModel::Create();
   auto pEvent = model->MakeField<Event>("event");

   // Create our main RNTuple.
   // NOTE: currently attributes are only supported in TFile-based RNTuples, so we must
   // create the RNTupleWriter via Append(), not Recreate(). This will be relaxed in the future.
   auto file = std::unique_ptr<TFile>(TFile::Open(kFileName, "RECREATE"));
   auto writer = ROOT::RNTupleWriter::Append(std::move(model), kNTupleName, *file);

   // Define the model for the attribute set we want to create.
   auto attrModel = ROOT::RNTupleModel::Create();
   attrModel->SetDescription("Metadata containing the events' provenance"); // (this is optional)
   auto pRunNumber = attrModel->MakeField<std::int32_t>("runNumber");

   // Create the attribute set from the main RNTupleWriter. We name it "Provenance".
   auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "Provenance");

   // Writing attributes works like this:
   //    1. begin an attribute range
   //    2. fill your main RNTuple data as usual
   //    3. commit the attribute range.
   //
   // Between the beginning and committing of each attribute range you can set the values of all the
   // fields in the attribute model; all the main RNTuple data you filled in before committing the range
   // will have those values as metadata associated to it.
   //
   // Beginning an attribute range is done like this (note that attrEntry is an REntry, so filling
   // in the attribute values uses the same interface of the regular values):
   auto attrRange = attrSet->BeginRange();

   // Here you can assign values to your attributes. In this case we only have 1 attribute (the string "myAttr"),
   // so we only fill that.
   // Note that attribute values can be assigned anywhere between BeginRange() and CommitRange(), so we could do
   // this step even after the main data filling.
   *pRunNumber = 0;

   // Now fill in the main RNTuple data
   TRandom3 rng;
   for (int i = 0; i < 100; ++i) {
      *pEvent = {rng.Rndm()};
      writer->Fill();

      // For explanatory purpose, let's say every 10 entries we have a new run number.
      if (i % 10 == 0 && i > 0) {
         // Close the attribute range. To do this you need to move the attribute range into the CommitRange
         // method, so you cannot reuse `attrRange`.
         attrSet->CommitRange(std::move(attrRange));
         // If you need to open a new range, call BeginRange again.
         attrRange = attrSet->BeginRange();
         // Then we can assign the new run number.
         *pRunNumber += 1;
      }
   }

   // IMPORTANT: attributes are not written to storage until you call CommitRange and, differently from regular data,
   // they are NOT automatically written on destruction. If you don't call CommitRange, the attribute data will not
   // be stored.
   // In general, you can check if the attribute range was not committed via its operator bool():
   if (attrRange) {
      attrSet->CommitRange(std::move(attrRange));
   }
}

static bool IsGoodRunNumber(std::int32_t runNo)
{
   // For illustratory purposes, let's pretend we know that runNumber 4 is "good" for our analysis.
   return runNo == 4;
}

static void Read()
{
   // Open the main RNTuple for reading
   auto reader = ROOT::RNTupleReader::Open(kNTupleName, kFileName);

   // To read back the list of available attribute sets:
   std::cout << "Here are the attribute sets linked to the RNTuple '" << kNTupleName << "':\n";
   for (const auto &attrSetDesc : reader->GetDescriptor().GetAttrSetIterable()) {
      std::cout << "  " << attrSetDesc.GetName() << "\n";
   }

   // Open a specific attribute set
   auto attrSet = reader->OpenAttributeSet("Provenance");
   // Fetch pointers to all attributes you want to read.
   auto pRunNumber = attrSet->GetModel().GetDefaultEntry().GetPtr<std::int32_t>("runNumber");

   std::cout << "\nOpened attribute set '" << attrSet->GetDescriptor().GetName() << "' with description: \""
             << attrSet->GetDescriptor().GetDescription() << "\"\n";
   // Loop over the main entries and, for each, print its associated attribute "myAttr"
   for (auto mainIdx : reader->GetEntryRange()) {
      // There are various ways to access attributes from a main entry index (see the RNTupleAttrSetReader's
      // documentation). Here we use GetAttributes to get all attributes associated to the main entry `mainIdx`:
      std::cout << "Entry " << mainIdx << " has the following attributes associated to it:\n";
      for (auto attrIdx : attrSet->GetAttributes(mainIdx)) {
         auto range = attrSet->LoadEntry(attrIdx);
         std::cout << "  runNumber = " << *pRunNumber << " (valid for range [" << *range.GetFirst() << ", "
                   << *range.GetLast() << "])\n";
      }
   }

   // You can also do the opposite lookup, by looping over all attributes first...
   auto pEvent = reader->GetModel().GetDefaultEntry().GetPtr<Event>("event");
   for (auto attrIdx : attrSet->GetAttributes()) {
      auto range = attrSet->LoadEntry(attrIdx);
      // ...and then deciding whether you want to load the corresponding range or not.
      if (IsGoodRunNumber(*pRunNumber)) {
         std::cout << "\nRun " << *pRunNumber << " is good. Events:\n";
         for (auto mainIdx = range.GetStart(); mainIdx < range.GetEnd(); ++mainIdx) {
            reader->LoadEntry(mainIdx);
            std::cout << "  Event " << mainIdx << " with pt = " << pEvent->pt << "\n";
         }
      }
   }
}

void ntpl019_attributes()
{
   Write();
   Read();
}
