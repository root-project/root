#include <ROOT/RDrawStorage.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TFile.h>

#include "gtest/gtest.h"

#include <string>
#include <vector>

using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
using RNTupleDraw = ROOT::Experimental::RNTupleDraw;
using RDrawStorage = ROOT::Experimental::Detail::RDrawStorage;

/**
 * It is hard to test the exact behaviour inside a TCanvas, so this test only checks if no
 * exception occurs and the most important member variables hold the correct value.
 */

namespace {
  /**
   * An RAII wrapper around an open temporary file on disk. It cleans up the guarded file when the wrapper object
   * goes out of scope.
   */
class FileRaii {
private:
   std::string fPath;
   TFile *file;
public:
   FileRaii(const std::string &path) :
      fPath(path),
      file(TFile::Open(fPath.c_str(), "RECREATE"))
      { }
   FileRaii(const FileRaii&) = delete;
   FileRaii& operator=(const FileRaii&) = delete;
   TFile* GetFile() const { return file; }
   std::string GetPath() const { return fPath; }
   ~FileRaii() {
      file->Close();
      std::remove(fPath.c_str());
   }
};
} // anonymous namespace

// Should print a message, that nothing was drawn.
TEST(PrintStorageLayout, emptyNTuple)
{
   FileRaii fileGuard("test_printStorageLayout_empty.root");
   std::string_view ntupleName = "emptyFile";
   {
      auto model = RNTupleModel::Create();
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard.GetPath());
   } // flushes content to file
   auto reader = RNTupleReader::Open(ntupleName, fileGuard.GetPath());
   auto draw = RNTupleDraw(reader);
   draw.Draw();
   
   FileRaii fileGuard2("test_printStorageLayout_empty2.root");
   std::string_view ntupleName2 = "emptyFile2";
   {
      auto model = RNTupleModel::Create();
      auto intfld = model->MakeField<int>("intfield");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName2, fileGuard2.GetPath());
   } // flushes content to file
   auto reader2 = RNTupleReader::Open(ntupleName2, fileGuard2.GetPath());
   auto draw2 = RNTupleDraw(reader2);
   draw2.Draw();
}

TEST(PrintStorageLayout, rootFile)
{
   FileRaii fileGuard("test_printStorageLayout_rootFile.root");
   std::string_view ntupleName = "rootFile";
   {
      auto model = RNTupleModel::Create();
      auto intfld = model->MakeField<std::int32_t>("intField");
      auto doublefld = model->MakeField<double>("doubleField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard.GetPath());
      *intfld = 9;
      *doublefld = 3.14;
      ntuple->Fill();
   } // flushes content to file
   auto model = RNTupleModel::Create();
   auto intptr = model->MakeField<std::int32_t>("intField");
   auto reader = RNTupleReader::Open(std::move(model), ntupleName, fileGuard.GetPath());
   auto drawStorage = RDrawStorage(reader.get());
   EXPECT_EQ(3ull, drawStorage.GetNFields());
   EXPECT_EQ(2ull, drawStorage.GetNColumns());
   EXPECT_EQ(1ull, drawStorage.GetNClusters());
   EXPECT_EQ(2ull, drawStorage.GetPageBoxSize());
   drawStorage.Draw();
}

TEST(PrintStorageLayout, rawFile)
{
   FileRaii fileGuard("test_printStorageLayout_rawFile.ntuple");
   std::string_view ntupleName = "rawFile";
   {
      auto model = RNTupleModel::Create();
      auto stringptr = model->MakeField<std::string>("stringField");
      auto doubleVecptr = model->MakeField<std::vector<double>>("doubleVectorField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard.GetPath());
      *stringptr = std::string("bus");
      *doubleVecptr = std::vector<double>(2, 2);
      ntuple->Fill();
   } // flushes content to file
   auto model = RNTupleModel::Create();
   auto stringptr = model->MakeField<std::string>("stringField");
   auto reader = RNTupleReader::Open(std::move(model), ntupleName, fileGuard.GetPath());
   auto drawStorage = RDrawStorage(reader.get());
   EXPECT_EQ(4ull, drawStorage.GetNFields());
   EXPECT_EQ(4ull, drawStorage.GetNColumns());
   EXPECT_EQ(1ull, drawStorage.GetNClusters());
   EXPECT_EQ(4ull, drawStorage.GetPageBoxSize());
   drawStorage.Draw();
   // To check if RNTupleReader is not dead after drawing
   std::stringstream os;
   reader->PrintInfo(ROOT::Experimental::ENTupleInfo::kSummary, os);
}

TEST(PrintStorageLayout, callTwice)
{
   FileRaii fileGuard("test_printStorageLayout_callTwice.root");
   std::string_view ntupleName = "rootFile";
   {
      auto model = RNTupleModel::Create();
      auto intfld = model->MakeField<std::array<std::int32_t, 2>>("intarrayField");
      auto vecvecfld = model->MakeField<std::vector<std::vector<double>>>("vecvecdoubleField");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard.GetPath());
      *intfld = { 1, 2 };
      *vecvecfld = { { 8.2, 8.3 }, { 3.4, 5,6 } };
      ntuple->Fill();
   } // flushes content to file
   auto model = RNTupleModel::Create();
   auto intfld = model->MakeField<std::array<std::int32_t, 2>>("intarrayField");
   auto reader = RNTupleReader::Open(std::move(model), ntupleName, fileGuard.GetPath());
   auto drawStorage = RDrawStorage(reader.get());
   drawStorage.Draw();
   drawStorage.Draw();
   EXPECT_EQ(6ull, drawStorage.GetNFields());
   EXPECT_EQ(4ull, drawStorage.GetNColumns());
   EXPECT_EQ(1ull, drawStorage.GetNClusters());
   EXPECT_EQ(4ull, drawStorage.GetPageBoxSize());
}
