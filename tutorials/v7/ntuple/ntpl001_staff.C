/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Write and read tabular data with RNTuple.  Adapted from the cernbuild and cernstaff tree tutorials.
/// Illustrates the type-safe ntuple model interface, which is used to define a data model that is in a second step
/// taken by an ntuple reader or writer.
///
/// \macro_image
/// \macro_code
///
/// \date April 2019
/// \author The ROOT Team

// NOTE: The RNTuple classes are experimental at this point.
// Functionality, interface, and data format is still subject to changes.
// Do not use for real data!

// Until C++ runtime modules are universally used, we explicitly load the ntuple library.  Otherwise
// triggering autoloading from the use of templated types would require an exhaustive enumeration
// of "all" template instances in the LinkDef file.
R__LOAD_LIBRARY(ROOTNTuple)

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>

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

#include <nlohmann/json.hpp>

// Import classes from experimental namespace for the time being
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;

using RNTupleWriteOptions = ROOT::Experimental::RNTupleWriteOptions;

constexpr char const* kNTupleFileName = "ntpl001_staff.root";
//constexpr char const* kNTupleFileName = "test://ntpl001_staff.root";

void Ingest() {   
   
   // The input file cernstaff.dat is a copy of the CERN staff data base from 1988
   ifstream fin(gROOT->GetTutorialDir() + "/tree/cernstaff.dat");
   assert(fin.is_open());

   // We create a unique pointer to an empty data model
   auto model = RNTupleModel::Create();

   // To define the data model, we create fields with a given C++ type and name.  Fields are roughly TTree branches.
   // MakeField returns a shared pointer to a memory location that we can populate to fill the ntuple with data
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

   // We hand-over the data model to a newly created ntuple of name "Staff", stored in kNTupleFileName
   // In return, we get a unique pointer to an ntuple that we can fill

   // Compress the .root file
   RNTupleWriteOptions write_options;
   write_options.SetCompression(0);

   {
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff", kNTupleFileName, write_options);
      
      std::string record;
      while (std::getline(fin, record)) {
         std::istringstream iss(record);
         iss >> *fldCategory >> *fldFlag >> *fldAge >> *fldService >> *fldChildren >> *fldGrade >> *fldStep >> *fldHrweek
            >> *fldCost >> *fldDivision >> *fldNation;
         ntuple->Fill();
      }
   }

   // open the root file and retrieve rntuple object
   auto f = TFile::Open("/home/vporter/Documents/root/tutorials/v7/ntuple/ntpl001_staff.root","READ");
   
   auto ntpl = f->Get<ROOT::Experimental::RNTuple>("Staff");


   ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = ntpl->GetSeekHeader();
   ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = ntpl->GetSeekHeader() + ntpl->GetNBytesHeader();

   // set parameters for targeted bit flips in the header
   //ROOT::Internal::RRawFile::range_begin() = ntpl->GetSeekHeader(); // file offset of the header
   //ROOT::Internal::RRawFile::range_end() = ntpl->GetSeekHeader() + ntpl->GetNBytesHeader(); // size of the compressed ntuple header

   std::cout << ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin << std::endl;
   std::cout << ROOT::Internal::RRawFile::GetBitFlipParams().rng_end << std::endl;

   //ROOT::Internal::RRawFile::bitFlipParams.rng_begin = ntpl->GetSeekHeader();
   //ROOT::Internal::RRawFile::bitFlipParams.rng_end = ntpl->GetSeekHeader() + ntpl->GetNBytesHeader();

   // set parameters for targeted bit flips in the footer
   //ROOT::Internal::RRawFile::range_begin() = ntpl->GetSeekFooter();
   //ROOT::Internal::RRawFile::range_size() = ntpl->GetNBytesFooter();

   // Note: GetLenHeader() and GetLenFooter() return the size of the uncompressed ntuple header

   // The ntuple unique pointer goes out of scope here.  On destruction, the ntuple flushes unwritten data to disk
   // and closes the attached ROOT file.
}

void Analyze() {
   // Get a unique pointer to an empty RNTuple model
   auto model = RNTupleModel::Create();

   // We only define the fields that are needed for reading
   std::shared_ptr<int> fldAge = model->MakeField<int>("Age");

   // Create an ntuple and attach the read model to it
   auto ntuple = RNTupleReader::Open(std::move(model), "Staff", kNTupleFileName);
   //std::string("test://")+kNTupleFileName
   
   // Quick overview of the ntuple and list of fields.
   //ntuple->PrintInfo();
   // In a future version of RNTuple, there will be support for ntuple->Scan()

   //ntuple->Show(0);

   //specify and open output file
   std::ofstream file("run_3.txt");

   if(file.is_open())
   {
      for(int idx = 0; idx < ntuple->GetNEntries(); idx++) 
      {
         ntuple->Show(idx,file);
         file << std::endl;
      }

      file.close();
   }

   std::cout << "File Created!" << std::endl;

}

void ntpl001_staff() {
   Ingest();
   Analyze();
}
