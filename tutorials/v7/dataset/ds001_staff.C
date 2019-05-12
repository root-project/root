/// \file
/// \ingroup tutorial_dataset
/// \notebook
/// Write and read tabular data with RDataSet.  Adapted from the cernbuild and cernstaff tree tutorials.
/// Illustrates the type-safe data set model interface, which is used to define a data model that is in a second step
/// taken by an input or an output data set.
///
/// \macro_image
/// \macro_code
///
/// \date April 2019
/// \author The ROOT Team

// NOTE: The RDataSet classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

R__LOAD_LIBRARY(ROOTDataSet)

#include <ROOT/RDataSet.hxx>
#include <ROOT/RDataSetModel.hxx>

#include <TCanvas.h>
#include <TH1I.h>
#include <TROOT.h>
#include <TString.h>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <utility>

// Import classes from experimental namespace for the time being
// Also, hide the old RForest code name until sources are fully renamed.
using RDataSetModel = ROOT::Experimental::RForestModel;
using RDataSetReader = ROOT::Experimental::RInputForest;
using RDataSetWriter = ROOT::Experimental::ROutputForest;

constexpr char const* kDataSetFileName = "ds001_staff.root";

void Ingest() {
   // The input file cernstaff.dat is a copy of the CERN staff data base from 1988
   ifstream fin(gROOT->GetTutorialDir() + "/tree/cernstaff.dat");
   assert(fin.is_open());

   // We create a unique pointer to an empty data model
   auto model = RDataSetModel::Create();

   // To define the data model, we create fields with a given C++ type and name.  Fields are roughly TTree branches.
   // MakeField returns a shared pointer to a memory location that we can populate to fill the data set with data
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

   // We hand-over the data model to a newly created data set of name "Staff", stored in kDataSetFileName
   // In return, we get a unique pointer to a data set that we can fill
   auto dataSet = RDataSetWriter::Recreate(std::move(model), "Staff", kDataSetFileName);

   std::string record;
   while (std::getline(fin, record)) {
      std::istringstream iss(record);
      iss >> *fldCategory >> *fldFlag >> *fldAge >> *fldService >> *fldChildren >> *fldGrade >> *fldStep >> *fldHrweek
          >> *fldCost >> *fldDivision >> *fldNation;
      dataSet->Fill();
   }

   // The data set unique pointer goes out of scope here.  On destruction, the data set flushes unwritten data to disk
   // and closes the attached ROOT file.
}

void Analyze() {
   // Get a unique pointer to an empty RDataSet model
   auto model = RDataSetModel::Create();

   // We only define the fields that are needed for reading
   std::shared_ptr<int> fldAge = model->MakeField<int>("Age");

   // Create a data set and attach the read model to it
   auto dataSet = RDataSetReader::Open(std::move(model), "Staff", kDataSetFileName);

   // Quick overview of the data set's key meta-data
   std::cout << dataSet->GetInfo();
   // In a future version of RDataSet, there will be support for dataSet->Show() and dataSet->Scan()

   TCanvas *c = new TCanvas("c", "", 200, 10, 700, 500);
   TH1I *h = new TH1I("h", "Age Distribution CERN, 1988", 100, 0, 100);
   h->SetFillColor(48);

   for (auto entryId : *dataSet) {
      // Populate fldAge
      dataSet->LoadEntry(entryId);
      h->Fill(*fldAge);
   }

   h->DrawCopy();
}

void ds001_staff() {
   Ingest();
   Analyze();
}
