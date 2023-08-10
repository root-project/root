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
#include <ROOT/RNTupleDescriptor.hxx>

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

using RNTupleDescriptor = ROOT::Experimental::RNTupleDescriptor;
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

   //std::cout << "begin " << ntpl->GetSeekHeader() << std::endl;
   //std::cout << "end " << ntpl->GetNBytesHeader() << std::endl;

   //ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_begin = ntpl->GetSeekHeader();
   //ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_end = ntpl->GetSeekHeader() + ntpl->GetNBytesHeader();

   // Create an RNTupleReader from the ntuple
   std::unique_ptr<RNTupleReader> ntupleReader = RNTupleReader::Open("Staff", "/home/vporter/Documents/root/tutorials/v7/ntuple/ntpl001_staff.root");

   // Get descriptor of the ntuple
   const auto descriptor = ntupleReader->GetDescriptor();
   
   // Get field id of "category" -> Xf
   auto fieldId = descriptor->FindFieldId("Category");

   // // Physical column id (Xf,0) -> Xcol
   //OLD auto columnId = descriptor->GetColumnId(fieldId,0);
   auto columnId = descriptor->FindPhysicalColumnId(fieldId,0);

   // // Get the cluster id (Xcol,0) -> Xcl
   auto clusterId = descriptor->FindClusterId(columnId,0);

   // // Get cluster descriptor of Xcl
   const auto &clusterDescriptor = descriptor->GetClusterDescriptor(clusterId);

   // // Get page range of Xcl
   auto &pageRangeXcl = clusterDescriptor.GetPageRange(columnId);

   const auto &pageInfo = pageRangeXcl.fPageInfos[0]; // page info for first page
   auto loc = pageInfo.fLocator; // locator for first page
   auto nelem = loc.fBytesOnStorage; // num of bytes
   auto offset = loc.GetPosition<std::uint64_t>(); // offset of first page

   //std::cout << "offset = " << offset << " nelem = " << nelem << std::endl;

   //ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_begin = offset;
   //ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_end = offset + nelem;

   //ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_begin = offset;
   //ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_end = offset + nelem;
   ROOT::Internal::RRawFile::GetFailureInjectionParams().failureType = ROOT::Internal::RRawFile::SetFailureType(ROOT::Internal::RRawFile::FailureType::ShortRead);
   
   // ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_begin = ntpl->GetSeekHeader();
   // ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_end = ntpl->GetSeekHeader() + ntpl->GetNBytesHeader();
   
   //std::cout << "rng begin " << ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_begin << std::endl;
   //std::cout << "rng end " << ROOT::Internal::RRawFile::GetFailureInjectionParams().rng_end << std::endl;
   //std::cout << "failure type " << ROOT::Internal::RRawFile::GetFailureInjectionParams().failureType << std::endl;

}

void Analyze() {
   // Get a unique pointer to an empty RNTuple model
   auto model = RNTupleModel::Create();

   // We only define the fields that are needed for reading
   std::shared_ptr<int> fldAge = model->MakeField<int>("Age");

   // Create an ntuple and attach the read model to it
   auto ntuple = RNTupleReader::Open(std::move(model), "Staff", kNTupleFileName);

   //specify and open output file
   std::ofstream file("short_read_0.txt");

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
