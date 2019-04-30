/// \file
/// \ingroup tutorial_forest
/// \notebook
/// Write and read tabular data with RForest.  Adapted from the cernbuild and cernstaff tree tutorials.
/// Illustrates the type-safe forest model interface, which is used to define a data model that is in a second step
/// taken by an input or an output forest.
///
/// \macro_image
/// \macro_code
///
/// \date April 2019
/// \author The ROOT Team

// NOTE: The RForest classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

R__LOAD_LIBRARY(libROOTForest)

#include <ROOT/RForest.hxx>
#include <ROOT/RForestModel.hxx>

#include <TCanvas.h>
#include <TH1I.h>
#include <TString.h>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>

// Import classes from experimental namespace for the time being
using RForestModel = ROOT::Experimental::RForestModel;
using RInputForest = ROOT::Experimental::RInputForest;
using ROutputForest = ROOT::Experimental::ROutputForest;

constexpr char const* kForestFile = "fst001_staff.root";

void Ingest() {
   // The input file cernstaff.dat is a copy of the CERN staff data base from 1988
   TString dir = gROOT->GetTutorialDir();
   dir.Append("/tree/");
   dir.ReplaceAll("/./","/");
   FILE *fp = fopen(Form("%scernstaff.dat", dir.Data()), "r");
   assert(fp != nullptr);

   // We create a unique pointer to an empty data model
   auto model = RForestModel::Create();

   // To define the data model, we create fields with a given C++ type and name.  Fields are roughly TTree branches.
   // MakeField returns a shared pointer to a memory location that we can populate to fill the forest with data
   auto fldCategory = model->MakeField<int>("Category");
   auto fldFlag     = model->MakeField<unsigned int>("Flag");
   auto fldAge      = model->MakeField<int>("Age");
   auto fldService  = model->MakeField<int>("Service");
   auto fldChildren = model->MakeField<int>("Children");
   auto fldGrade    = model->MakeField<int>("Grade");
   auto fldStep     = model->MakeField<int>("Step");
   auto fldHrweek   = model->MakeField<int>("Hrweek");
   auto fldCost     = model->MakeField<int>("Cost");
   auto fldDivision = model->MakeField<std::string>("Division");
   auto fldNation   = model->MakeField<std::string>("Nation");

   // Ensure any previously created files from this tutorial are removed
   std::remove(kForestFile);

   // We hand-over the data model to a newly created forest of name "Staff", stored in kForestFile
   // In return, we get a unique pointer to a forest that we can fill
     // --> Recreate
   auto forest = ROutputForest::Create(std::move(model), "Staff", kForestFile);

   char line[80];
   while (fgets(line,80,fp)) {
      // --> cin
      char division[32];
      char nation[32];
      sscanf(&line[0],"%d %d %d %d %d %d %d  %d %d %s %s",
         fldCategory.get(), fldFlag.get(), fldAge.get(), fldService.get(), fldChildren.get(), fldGrade.get(),
         fldStep.get(), fldHrweek.get(), fldCost.get(), division, nation);
      *fldDivision = std::string(division);
      *fldNation = std::string(nation);
      forest->Fill();
   }

   fclose(fp);
   // The forest unique pointer goes out of scope here.  On destruction, the forest flushes unwritten data to disk
   // and closes the attached ROOT file.
}

void Analyze() {
   // Create a forest without imposing a specific data model.  We could generate the data model from the forest
   // but here we prefer the view because we only want to access a single field
   auto forest = RInputForest::Open("Staff", kForestFile);

   // Quick overview of the forest's key meta-data
   /// --> GetDescription()
   std::cout << forest->Print();
   // In a future version of RForest, there will be support for forest->Show() and forest->Scan()

   // We resurrect the model from the forest on disk.
   // NB: performance-wise, this is not ideal because all the fields of the original model are populated as entries
   // are read.  See fst003_lhcbOpenData.C for an example with a slimmed down model for reading.
   // -> const
   RForestModel* model = forest->GetModel();
   // We get the memory location of the model's default entry, which is populated by forest->SetEntry()
   auto ptrAge = model->Get<int>("Age");

   TCanvas *c = new TCanvas("c", "Age Distribution CERN, 1988", 200, 10, 700, 500);
   TH1I *h = new TH1I("h", "Age Distribution CERN, 1988", 100, 0, 100);
   h->SetFillColor(48);

   // --> for (auto _ : *forest) ? what about the unused variable warning
   for (unsigned int i = 0; i < forest->GetNEntries(); ++i) {
      // --> LoadEntry()
      forest->LoadEntry(i);
      h->Fill(*ptrAge);
   }

   h->DrawCopy();
}

void fst001_staff() {
   Ingest();
   Analyze();
}
