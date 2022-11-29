#include "gtest/gtest.h"

#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleImporter.hxx>

#include <TFile.h>
#include <TTree.h>

#include <cstdio>
#include <string>

using ROOT::Experimental::RNTupleImporter;
using ROOT::Experimental::RNTupleReader;

/**
 * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
 * goes out of scope.
 */
class FileRaii {
private:
   std::string fPath;

public:
   explicit FileRaii(const std::string &path) : fPath(path) {}
   FileRaii(const FileRaii &) = delete;
   FileRaii &operator=(const FileRaii &) = delete;
   ~FileRaii() { std::remove(fPath.c_str()); }
   std::string GetPath() const { return fPath; }
};

TEST(RNTupleImporter, Empty)
{
   FileRaii fileGuard("test_ntuple_importer_empty.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath()).Unwrap();
   importer->SetIsQuiet(true);
   EXPECT_THROW(importer->Import(), ROOT::Experimental::RException);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(0U, reader->GetNEntries());
   EXPECT_THROW(importer->Import(), ROOT::Experimental::RException);
}

TEST(RNTupleImporter, SimpleBranches)
{
   FileRaii fileGuard("test_ntuple_importer_simple_branches.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      bool myBool = true;
      Char_t myInt8 = -8;
      UChar_t myUInt8 = 8;
      Short_t myInt16 = -16;
      UShort_t myUInt16 = 16;
      Int_t myInt32 = -32;
      UInt_t myUInt32 = 32;
      Long64_t myInt64 = -64;
      ULong64_t myUInt64 = 64;
      Float_t myFloat = 32.0;
      Double_t myDouble = 64.0;
      // TODO(jblomer): Float16_t, Double32_t
      tree->Branch("myBool", &myBool);
      tree->Branch("myInt8", &myInt8);
      tree->Branch("myUInt8", &myUInt8);
      tree->Branch("myInt16", &myInt16);
      tree->Branch("myUInt16", &myUInt16);
      tree->Branch("myInt32", &myInt32);
      tree->Branch("myUInt32", &myUInt32);
      tree->Branch("myInt64", &myInt64);
      tree->Branch("myUInt64", &myUInt64);
      tree->Branch("myFloat", &myFloat);
      tree->Branch("myDouble", &myDouble);
      tree->Fill();
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath()).Unwrap();
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_TRUE(*reader->GetModel()->Get<bool>("myBool"));
   EXPECT_EQ(-8, *reader->GetModel()->Get<char>("myInt8"));
   EXPECT_EQ(8U, *reader->GetModel()->Get<std::uint8_t>("myUInt8"));
   EXPECT_EQ(-16, *reader->GetModel()->Get<std::int16_t>("myInt16"));
   EXPECT_EQ(16U, *reader->GetModel()->Get<std::uint16_t>("myUInt16"));
   EXPECT_EQ(-32, *reader->GetModel()->Get<std::int32_t>("myInt32"));
   EXPECT_EQ(32U, *reader->GetModel()->Get<std::uint32_t>("myUInt32"));
   EXPECT_EQ(-64, *reader->GetModel()->Get<std::int64_t>("myInt64"));
   EXPECT_EQ(64U, *reader->GetModel()->Get<std::uint64_t>("myUInt64"));
   EXPECT_FLOAT_EQ(32.0, *reader->GetModel()->Get<float>("myFloat"));
   EXPECT_FLOAT_EQ(64.0, *reader->GetModel()->Get<double>("myDouble"));
}

TEST(RNTupleImporter, CString)
{
   FileRaii fileGuard("test_ntuple_importer_cstring.root");
   {
      std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str(), "RECREATE"));
      auto tree = std::make_unique<TTree>("tree", "");
      const char *myString = "R";
      tree->Branch("myString", const_cast<char *>(myString), "myString/C");
      tree->Fill();
      myString = "";
      tree->SetBranchAddress("myString", const_cast<char *>(myString));
      tree->Fill();
      myString = "ROOT RNTuple";
      tree->SetBranchAddress("myString", const_cast<char *>(myString));
      tree->Fill();
      tree->Write();
   }

   auto importer = RNTupleImporter::Create(fileGuard.GetPath(), "tree", fileGuard.GetPath()).Unwrap();
   importer->SetIsQuiet(true);
   importer->SetNTupleName("ntuple");
   importer->Import();
   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(3U, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_EQ(std::string("R"), *reader->GetModel()->Get<std::string>("myString"));
   reader->LoadEntry(1);
   EXPECT_EQ(std::string(""), *reader->GetModel()->Get<std::string>("myString"));
   reader->LoadEntry(2);
   EXPECT_EQ(std::string("ROOT RNTuple"), *reader->GetModel()->Get<std::string>("myString"));
}
