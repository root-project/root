#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TFile.h>

#include "gtest/gtest.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

using RFieldBase = ROOT::Experimental::Detail::RFieldBase;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleDescriptor = ROOT::Experimental::RNTupleDescriptor;

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

// -------------------------- helper functions ---------------------------------

bool CompareFields (const RNTupleDescriptor &desc1, const RNTupleDescriptor &desc2) {
   if (desc1.GetNFields() != desc2.GetNFields())
      return false;
   for (std::size_t i = 0; i < desc1.GetNFields(); ++i) {
      if (desc1.GetFieldDescriptor(i) == desc2.GetFieldDescriptor(i))
         return true;
   }
   return false;
}

bool CompareColumns (const RNTupleDescriptor &desc1, const RNTupleDescriptor &desc2) {
   if(desc1.GetNColumns() != desc2.GetNColumns())
      return false;
   for (std::size_t i = 0; i < desc1.GetNColumns(); ++i) {
      if (desc1.GetColumnDescriptor(i) == desc2.GetColumnDescriptor(i))
         return true;
   }
   return false;
}

bool CompareClusters (const RNTupleDescriptor &desc1, const RNTupleDescriptor &desc2) {
   if(desc1.GetNClusters() != desc2.GetNClusters())
      return false;
   for (std::size_t i = 0; i < desc1.GetNClusters(); ++i) {
      if (desc1.GetClusterDescriptor(i) == desc2.GetClusterDescriptor(i))
         return true;
   }
   return false;
}

bool DescriptorsAreSame(const RNTupleDescriptor &desc1, const RNTupleDescriptor &desc2) {
   return ( CompareFields(desc1, desc2) && CompareColumns(desc1, desc2) && CompareClusters(desc1, desc2) );
}

// ------------------------------------------- Tests ------------------------------------------

TEST(RNTupleChain, noFiles)
{
   const std::string_view ntupleName = "noFilesNTuple";
   auto ntuple = RNTupleReader::Open(ntupleName,  std::vector<std::string>{ });
   EXPECT_EQ(ntuple, nullptr);
}

TEST(RNTupleChain, oneFile)
{
   const std::string_view ntupleName = "oneFileNTuple";
   FileRaii fileGuard("test_ntuple_chain_oneFile.root");
   {
      auto model = RNTupleModel::Create();
      auto dbField = model->MakeField<double>("db");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard.GetPath());
      
      for (int i = 0; i < 3; ++i) {
         *dbField = 5.0+i;
         *stField = "foo";
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open(ntupleName, fileGuard.GetPath());
   auto ntupleVec = RNTupleReader::Open(ntupleName, std::vector<std::string>{ fileGuard.GetPath() });
   EXPECT_TRUE(DescriptorsAreSame(ntuple->GetDescriptor(), ntupleVec->GetDescriptor()));
}

TEST(RNTupleChain, twiceSameFile)
{
   const std::string ntupleName = "twiceSameFileNTuple";
   FileRaii fileGuard("test_ntuple_chain_twiceSameFile.root");
   {
      auto model = RNTupleModel::Create();
      auto itField = model->MakeField<int>("it");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard.GetPath());

      for (int i = 0; i < 3; ++i) {
         *itField = 5+i;
         *stField = "boo";
         ntuple->Fill();
      }
   }

   const std::string ntupleName2 = "twiceSameFileNTupleMerged";
   FileRaii fileGuardmerged("test_ntuple_chain_twiceSameFile_merged.root");
   {
      auto model = RNTupleModel::Create();
      auto itField = model->MakeField<int>("it");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName2, fileGuardmerged.GetPath());
   
      for (int i = 0; i < 6; ++i) {
         *itField = 5+i;
         *stField = "boo";
         ntuple->Fill();
         if (i == 2) ntuple->CommitCluster();
      }
   }

   auto ntuple1 = RNTupleReader::Open(ntupleName, { fileGuard.GetPath(), fileGuard.GetPath() } );
   auto ntuple2 = RNTupleReader::Open(ntupleName2, fileGuardmerged.GetPath() );
   EXPECT_TRUE(DescriptorsAreSame(ntuple1->GetDescriptor(), ntuple2->GetDescriptor()));
}

TEST(RNTupleChain, threeFiles)
{
   const std::string_view ntupleName = "threeFilesNTuple";
   FileRaii fileGuard1("test_ntuple_chain_threeFiles_first_file.root");
   {
      auto model = RNTupleModel::Create();
      auto ftField = model->MakeField<float>("ft");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard1.GetPath());
      
      for (int i = 0; i < 7; ++i) {
         *ftField = 5.0f+(float)i;
         *stField = "foofoofoo";
         ntuple->Fill();
         if ( i == 3) ntuple->CommitCluster();
      }
   }

   // Note that a raw file is created here instead of a .root file.
   FileRaii fileGuard2("test_ntuple_chain_threeFiles_second_file.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto ftField = model->MakeField<float>("ft");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard2.GetPath());
      
      for (int i = 0; i < 5; ++i) {
         *ftField = 6.0f+(float)i;
         *stField = "foo";
         ntuple->Fill();
      }
   }

   FileRaii fileGuard3("test_ntuple_chain_threeFiles_third_file.root");
   {
      auto model = RNTupleModel::Create();
      auto ftField = model->MakeField<float>("ft");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard3.GetPath());
      
      for (int i = 0; i < 60000; ++i) {
         *ftField = 7.0f+(float)i;
         *stField = std::to_string(i+12);
         ntuple->Fill();
      }
   }

   FileRaii fileGuardmerged("test_ntuple_chain_threeFiles_mergedfile.root");
   {
      auto model = RNTupleModel::Create();
      auto ftField = model->MakeField<float>("ft");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuardmerged.GetPath());

      for (int i = 0; i < 7; ++i) {
         *ftField = 5.0f+(float)i;
         *stField = "foofoofoo";
         ntuple->Fill();
         if ( i == 3) ntuple->CommitCluster();
      }
      ntuple->CommitCluster();
      for (int i = 0; i < 5; ++i) {
         *ftField = 6.0f+(float)i;
         *stField = "foo";
         ntuple->Fill();
      }
      ntuple->CommitCluster();
      for (int i = 0; i < 60000; ++i) {
         *ftField = 7.0f+(float)i;
         *stField = std::to_string(i+12);
         ntuple->Fill();
      }
   }

   auto ntupleVec = RNTupleReader::Open(ntupleName, { fileGuard1.GetPath(), fileGuard2.GetPath(), fileGuard3.GetPath() });
   auto ntuple = RNTupleReader::Open(ntupleName, fileGuardmerged.GetPath());

   EXPECT_TRUE(DescriptorsAreSame(ntuple->GetDescriptor(), ntupleVec->GetDescriptor()));

   auto floatView = ntupleVec->GetView<float>("ft");
   auto stringView = ntupleVec->GetView<std::string>("st");

   auto mergedfloatView = ntuple->GetView<float>("ft");
   auto mergedStringView = ntuple->GetView<std::string>("st");

   // There are errors which only occur when the views are mixed up.
   EXPECT_FLOAT_EQ(5.0f, floatView(0));
   EXPECT_FLOAT_EQ(9.0f, floatView(14));
   EXPECT_FLOAT_EQ(10.0f, floatView(15));
   EXPECT_FLOAT_EQ(6.0f, floatView(1));
   EXPECT_FLOAT_EQ(6.0f, mergedfloatView(7));
   EXPECT_FLOAT_EQ(7.0f, floatView(2));
   EXPECT_FLOAT_EQ(8.0f, mergedfloatView(13));
   EXPECT_FLOAT_EQ(10.0f, floatView(5));
   EXPECT_FLOAT_EQ(11.0f, floatView(6));
   EXPECT_FLOAT_EQ(9.0f, floatView(10));
   EXPECT_FLOAT_EQ(10.0f, floatView(11));
   EXPECT_FLOAT_EQ(6.0f, floatView(7));
   EXPECT_FLOAT_EQ(7.0f, floatView(8));
   EXPECT_FLOAT_EQ(8.0f, floatView(9));
   EXPECT_FLOAT_EQ(9.0f, mergedfloatView(4));
   EXPECT_FLOAT_EQ(12004.0f, floatView(12009));
   EXPECT_FLOAT_EQ(8.0f, floatView(3));
   EXPECT_FLOAT_EQ(9.0f, floatView(4));
   EXPECT_FLOAT_EQ(8.0f, floatView(13));
   EXPECT_FLOAT_EQ(5.0f, mergedfloatView(0));
   EXPECT_FLOAT_EQ(11.0f, floatView(16));
   EXPECT_FLOAT_EQ(12.0f, floatView(17));
   EXPECT_FLOAT_EQ(49995.0f, floatView(50000));

   EXPECT_EQ("14", stringView(14));
   EXPECT_EQ("foofoofoo", stringView(0));
   EXPECT_EQ("foofoofoo", stringView(1));
   EXPECT_EQ("1034", stringView(1034));
   EXPECT_EQ("foofoofoo", stringView(2));
   EXPECT_EQ("foofoofoo", mergedStringView(6));
   EXPECT_EQ("foofoofoo", stringView(3));
   EXPECT_EQ("foo", stringView(8));
   EXPECT_EQ("foofoofoo", stringView(5));
   EXPECT_EQ("17", mergedStringView(17));
   EXPECT_EQ("foofoofoo", stringView(6));
   EXPECT_EQ("foo", mergedStringView(8));
   EXPECT_EQ("foo", stringView(7));
   EXPECT_EQ("13", stringView(13));
   EXPECT_EQ("foofoofoo", mergedStringView(0));
   EXPECT_EQ("foo", stringView(9));
   EXPECT_EQ("foo", stringView(10));
   EXPECT_EQ("foo", mergedStringView(7));
   EXPECT_EQ("foo", stringView(11));
   EXPECT_EQ("foofoofoo", mergedStringView(4));
   EXPECT_EQ("12", stringView(12));
   EXPECT_EQ("14", mergedStringView(14));
   EXPECT_EQ("foofoofoo", stringView(4));
   EXPECT_EQ("17003", stringView(17003));
   EXPECT_EQ("15", stringView(15));
   EXPECT_EQ("34567", stringView(34567));
   EXPECT_EQ("1680", stringView(1680));
   EXPECT_EQ("60010", stringView(60010));
   // TODO (lesimon): Check if RNTupleReader::Show() leads to desired output.
}


TEST(RNTupleChain, ChainOfChain)
{
   const std::string_view ntupleName = "ChainOfChainNTuple";
   FileRaii fileGuard("test_ntuple_chain_ChainOfChain.root");
   {
      auto model = RNTupleModel::Create();
      auto dbField = model->MakeField<double>("db");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard.GetPath());

      for (int i = 0; i < 3; ++i) {
         *dbField = 5.0+i;
         *stField = "foo" + std::to_string(i);
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open(ntupleName, fileGuard.GetPath());
   auto ntupleChain = RNTupleReader::ChainReader(ntupleName, ntuple, ntuple);
   auto ntupleChainOfChain = RNTupleReader::ChainReader(ntupleName, ntupleChain, ntupleChain);

   auto doubleView = ntuple->GetView<double>("db");
   auto stringView = ntuple->GetView<std::string>("st");

   auto doubleView2 = ntupleChain->GetView<double>("db");
   auto stringView2 = ntupleChain->GetView<std::string>("st");

   auto doubleView3 = ntupleChainOfChain->GetView<double>("db");
   auto stringView3 = ntupleChainOfChain->GetView<std::string>("st");

   EXPECT_DOUBLE_EQ(5.0f, doubleView(0));
   EXPECT_DOUBLE_EQ(6.0f, doubleView(1));
   EXPECT_DOUBLE_EQ(7.0f, doubleView(2));
   EXPECT_EQ("foo0", stringView(0));
   EXPECT_EQ("foo1", stringView(1));
   EXPECT_EQ("foo2", stringView(2));

   EXPECT_DOUBLE_EQ(5.0f, doubleView2(0));
   EXPECT_DOUBLE_EQ(6.0f, doubleView2(1));
   EXPECT_DOUBLE_EQ(7.0f, doubleView2(2));
   EXPECT_DOUBLE_EQ(5.0f, doubleView2(3));
   EXPECT_DOUBLE_EQ(6.0f, doubleView2(4));
   EXPECT_DOUBLE_EQ(7.0f, doubleView2(5));

   EXPECT_EQ("foo0", stringView2(0));
   EXPECT_EQ("foo1", stringView2(1));
   EXPECT_EQ("foo2", stringView2(2));
   EXPECT_EQ("foo0", stringView2(3));
   EXPECT_EQ("foo1", stringView2(4));
   EXPECT_EQ("foo2", stringView2(5));

   EXPECT_DOUBLE_EQ(5.0, doubleView3(0));
   EXPECT_DOUBLE_EQ(6.0, doubleView3(1));
   EXPECT_DOUBLE_EQ(7.0, doubleView3(2));
   EXPECT_DOUBLE_EQ(5.0, doubleView3(3));
   EXPECT_DOUBLE_EQ(6.0, doubleView3(4));
   EXPECT_DOUBLE_EQ(7.0, doubleView3(5));
   EXPECT_DOUBLE_EQ(5.0, doubleView3(6));
   EXPECT_DOUBLE_EQ(6.0, doubleView3(7));
   EXPECT_DOUBLE_EQ(7.0, doubleView3(8));
   EXPECT_DOUBLE_EQ(5.0, doubleView3(9));
   EXPECT_DOUBLE_EQ(6.0, doubleView3(10));
   EXPECT_DOUBLE_EQ(7.0, doubleView3(11));
   EXPECT_EQ("foo0", stringView3(0));
   EXPECT_EQ("foo1", stringView3(1));
   EXPECT_EQ("foo2", stringView3(2));
   EXPECT_EQ("foo0", stringView3(3));
   EXPECT_EQ("foo1", stringView3(4));
   EXPECT_EQ("foo2", stringView3(5));
   EXPECT_EQ("foo0", stringView3(6));
   EXPECT_EQ("foo1", stringView3(7));
   EXPECT_EQ("foo2", stringView3(8));
   EXPECT_EQ("foo0", stringView3(9));
   EXPECT_EQ("foo1", stringView3(10));
   EXPECT_EQ("foo2", stringView3(11));
}

TEST(RNTupleChain, ChainOfChainWithStdMove)
{
   const std::string_view ntupleName = "ChainOfChainNTuple";
   FileRaii fileGuard("test_ntuple_chain_ChainOfChain.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto dbField = model->MakeField<double>("db");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard.GetPath());
      
      for (int i = 0; i < 21000; ++i) {
         *dbField = 5.0+i;
         *stField = "foo" + std::to_string(i);
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open(ntupleName, fileGuard.GetPath());
   auto ntuple2 = RNTupleReader::Open(ntupleName, fileGuard.GetPath());
   auto ntuple3 = RNTupleReader::Open(ntupleName, fileGuard.GetPath());
   auto ntuple4 = RNTupleReader::Open(ntupleName, fileGuard.GetPath());
   auto ntupleChain = RNTupleReader::ChainReader(ntupleName, std::move(ntuple), std::move(ntuple2));
   auto ntupleChain2 = RNTupleReader::ChainReader(ntupleName, std::move(ntuple3), std::move(ntuple4));
   auto ntupleChainOfChain = RNTupleReader::ChainReader(ntupleName, std::move(ntupleChain), std::move(ntupleChain2));

   auto doubleView = ntupleChainOfChain->GetView<double>("db");
   auto stringView = ntupleChainOfChain->GetView<std::string>("st");

   EXPECT_DOUBLE_EQ(5.0, doubleView(0));
   EXPECT_DOUBLE_EQ(6.0, doubleView(1));
   EXPECT_DOUBLE_EQ(7.0, doubleView(2));
   EXPECT_DOUBLE_EQ(10005.0, doubleView(10000));
   EXPECT_DOUBLE_EQ(10006.0, doubleView(10001));
   EXPECT_DOUBLE_EQ(5.0, doubleView(21000));
   EXPECT_DOUBLE_EQ(6.0, doubleView(21001));
   EXPECT_DOUBLE_EQ(7.0, doubleView(21002));
   EXPECT_DOUBLE_EQ(10005.0, doubleView(31000));
   EXPECT_DOUBLE_EQ(6.0, doubleView(63001));
   EXPECT_DOUBLE_EQ(10006.0, doubleView(31001));
   EXPECT_DOUBLE_EQ(5.0, doubleView(42000));
   EXPECT_DOUBLE_EQ(6.0, doubleView(42001));
   EXPECT_DOUBLE_EQ(7.0, doubleView(42002));
   EXPECT_DOUBLE_EQ(10005.0, doubleView(52000));
   EXPECT_DOUBLE_EQ(10006.0, doubleView(52001));
   EXPECT_DOUBLE_EQ(5.0, doubleView(63000));
   EXPECT_DOUBLE_EQ(7.0, doubleView(63002));
   EXPECT_EQ("foo0", stringView(0));
   EXPECT_EQ("foo1", stringView(1));
   EXPECT_EQ("foo2", stringView(2));
   EXPECT_EQ("foo10000", stringView(10000));
   EXPECT_EQ("foo10001", stringView(10001));
   EXPECT_EQ("foo0", stringView(21000));
   EXPECT_EQ("foo1", stringView(21001));
   EXPECT_EQ("foo2", stringView(21002));
   EXPECT_EQ("foo0", stringView(63000));
   EXPECT_EQ("foo1", stringView(63001));
   EXPECT_EQ("foo10000", stringView(31000));
   EXPECT_EQ("foo10001", stringView(31001));
   EXPECT_EQ("foo0", stringView(42000));
   EXPECT_EQ("foo1", stringView(42001));
   EXPECT_EQ("foo2", stringView(42002));
   EXPECT_EQ("foo10000", stringView(52000));
   EXPECT_EQ("foo10001", stringView(52001));
   EXPECT_EQ("foo2", stringView(63002));
}

// When a cached page is returned in RPageSourceRoot::PopulatePage(), 2 pages with the same buffer will exist.
// This test should check if RNTupleChain can deal with such a situation.
TEST(RNTupleChain, CachedPage)
{
   const std::string ntupleName = "CachedPageNTuple";
   FileRaii fileGuard("test_ntuple_chain_cachedPage.root");
   {
      auto model = RNTupleModel::Create();
      auto boField = model->MakeField<bool>("bo");
      auto u64Field = model->MakeField<std::uint64_t>("u64");
      auto stField = model->MakeField<std::string>("st");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, fileGuard.GetPath());

      for (unsigned int i = 0; i < 10; ++i) {
         *boField = i%2;
         *u64Field = i;
         *stField = std::to_string(i) + "-th entry";
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open(ntupleName, { fileGuard.GetPath(), fileGuard.GetPath() });
   auto ntupleView1 = ntuple->GetView<bool>("bo");
   auto ntupleView2 = ntuple->GetView<bool>("bo");

   EXPECT_EQ(true, ntupleView1(11));
   EXPECT_EQ(false, ntupleView2(12));
}
