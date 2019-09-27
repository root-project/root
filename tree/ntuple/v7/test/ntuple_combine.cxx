#include <ROOT/RField.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>

#include <TFile.h>

#include "gtest/gtest.h"

#include <cassert>
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
   EXPECT_FLOAT_EQ(floatView(0), 5.0f);
   EXPECT_FLOAT_EQ(floatView(14), 9.0f);
   EXPECT_FLOAT_EQ(floatView(15), 10.0f);
   EXPECT_FLOAT_EQ(floatView(1), 6.0f);
   EXPECT_FLOAT_EQ(mergedfloatView(7), 6.0f);
   EXPECT_FLOAT_EQ(floatView(2), 7.0f);
   EXPECT_FLOAT_EQ(mergedfloatView(13), 8.0f);
   EXPECT_FLOAT_EQ(floatView(5), 10.0f);
   EXPECT_FLOAT_EQ(floatView(6), 11.0f);
   EXPECT_FLOAT_EQ(floatView(10), 9.0f);
   EXPECT_FLOAT_EQ(floatView(11), 10.0f);
   EXPECT_FLOAT_EQ(floatView(7), 6.0f);
   EXPECT_FLOAT_EQ(floatView(8), 7.0f);
   EXPECT_FLOAT_EQ(floatView(9), 8.0f);
   EXPECT_FLOAT_EQ(mergedfloatView(4), 9.0f);
   EXPECT_FLOAT_EQ(floatView(12009), 12004.0f);
   EXPECT_FLOAT_EQ(floatView(3), 8.0f);
   EXPECT_FLOAT_EQ(floatView(4), 9.0f);
   EXPECT_FLOAT_EQ(floatView(13), 8.0f);
   EXPECT_FLOAT_EQ(mergedfloatView(0), 5.0f);
   EXPECT_FLOAT_EQ(floatView(16), 11.0f);
   EXPECT_FLOAT_EQ(floatView(17), 12.0f);
   EXPECT_FLOAT_EQ(floatView(50000), 49995.0f);

   EXPECT_EQ(stringView(14), "14");
   EXPECT_EQ(stringView(0), "foofoofoo");
   EXPECT_EQ(stringView(1), "foofoofoo");
   EXPECT_EQ(stringView(1034), "1034");
   EXPECT_EQ(stringView(2), "foofoofoo");
   EXPECT_EQ(mergedStringView(6), "foofoofoo");
   EXPECT_EQ(stringView(3), "foofoofoo");
   EXPECT_EQ(stringView(8), "foo");
   EXPECT_EQ(stringView(5), "foofoofoo");
   EXPECT_EQ(mergedStringView(17), "17");
   EXPECT_EQ(stringView(6), "foofoofoo");
   EXPECT_EQ(mergedStringView(8), "foo");
   EXPECT_EQ(stringView(7), "foo");
   EXPECT_EQ(stringView(13), "13");
   EXPECT_EQ(mergedStringView(0), "foofoofoo");
   EXPECT_EQ(stringView(9), "foo");
   EXPECT_EQ(stringView(10), "foo");
   EXPECT_EQ(mergedStringView(7), "foo");
   EXPECT_EQ(stringView(11), "foo");
   EXPECT_EQ(mergedStringView(4), "foofoofoo");
   EXPECT_EQ(stringView(12), "12");
   EXPECT_EQ(mergedStringView(14), "14");
   EXPECT_EQ(stringView(4), "foofoofoo");
   EXPECT_EQ(stringView(17003), "17003");
   EXPECT_EQ(stringView(15), "15");
   EXPECT_EQ(stringView(34567), "34567");
   EXPECT_EQ(stringView(1680), "1680");
   EXPECT_EQ(stringView(60010), "60010");
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

   EXPECT_DOUBLE_EQ(doubleView(0), 5.0f);
   EXPECT_DOUBLE_EQ(doubleView(1), 6.0f);
   EXPECT_DOUBLE_EQ(doubleView(2), 7.0f);
   EXPECT_EQ(stringView(0), "foo0");
   EXPECT_EQ(stringView(1), "foo1");
   EXPECT_EQ(stringView(2), "foo2");

   EXPECT_DOUBLE_EQ(doubleView2(0), 5.0f);
   EXPECT_DOUBLE_EQ(doubleView2(1), 6.0f);
   EXPECT_DOUBLE_EQ(doubleView2(2), 7.0f);
   EXPECT_DOUBLE_EQ(doubleView2(3), 5.0f);
   EXPECT_DOUBLE_EQ(doubleView2(4), 6.0f);
   EXPECT_DOUBLE_EQ(doubleView2(5), 7.0f);

   EXPECT_EQ(stringView2(0), "foo0");
   EXPECT_EQ(stringView2(1), "foo1");
   EXPECT_EQ(stringView2(2), "foo2");
   EXPECT_EQ(stringView2(3), "foo0");
   EXPECT_EQ(stringView2(4), "foo1");
   EXPECT_EQ(stringView2(5), "foo2");

   EXPECT_DOUBLE_EQ(doubleView3(0), 5.0);
   EXPECT_DOUBLE_EQ(doubleView3(1), 6.0);
   EXPECT_DOUBLE_EQ(doubleView3(2), 7.0);
   EXPECT_DOUBLE_EQ(doubleView3(3), 5.0);
   EXPECT_DOUBLE_EQ(doubleView3(4), 6.0);
   EXPECT_DOUBLE_EQ(doubleView3(5), 7.0);
   EXPECT_DOUBLE_EQ(doubleView3(6), 5.0);
   EXPECT_DOUBLE_EQ(doubleView3(7), 6.0);
   EXPECT_DOUBLE_EQ(doubleView3(8), 7.0);
   EXPECT_DOUBLE_EQ(doubleView3(9), 5.0);
   EXPECT_DOUBLE_EQ(doubleView3(10), 6.0);
   EXPECT_DOUBLE_EQ(doubleView3(11), 7.0);
   EXPECT_EQ(stringView3(0), "foo0");
   EXPECT_EQ(stringView3(1), "foo1");
   EXPECT_EQ(stringView3(2), "foo2");
   EXPECT_EQ(stringView3(3), "foo0");
   EXPECT_EQ(stringView3(4), "foo1");
   EXPECT_EQ(stringView3(5), "foo2");
   EXPECT_EQ(stringView3(6), "foo0");
   EXPECT_EQ(stringView3(7), "foo1");
   EXPECT_EQ(stringView3(8), "foo2");
   EXPECT_EQ(stringView3(9), "foo0");
   EXPECT_EQ(stringView3(10), "foo1");
   EXPECT_EQ(stringView3(11), "foo2");
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

   EXPECT_DOUBLE_EQ(doubleView(0), 5.0);
   EXPECT_DOUBLE_EQ(doubleView(1), 6.0);
   EXPECT_DOUBLE_EQ(doubleView(2), 7.0);
   EXPECT_DOUBLE_EQ(doubleView(10000), 10005.0);
   EXPECT_DOUBLE_EQ(doubleView(10001), 10006.0);
   EXPECT_DOUBLE_EQ(doubleView(21000), 5.0);
   EXPECT_DOUBLE_EQ(doubleView(21001), 6.0);
   EXPECT_DOUBLE_EQ(doubleView(21002), 7.0);
   EXPECT_DOUBLE_EQ(doubleView(31000), 10005.0);
   EXPECT_DOUBLE_EQ(doubleView(31001), 10006.0);
   EXPECT_DOUBLE_EQ(doubleView(42000), 5.0);
   EXPECT_DOUBLE_EQ(doubleView(42001), 6.0);
   EXPECT_DOUBLE_EQ(doubleView(42002), 7.0);
   EXPECT_DOUBLE_EQ(doubleView(52000), 10005.0);
   EXPECT_DOUBLE_EQ(doubleView(52001), 10006.0);
   EXPECT_DOUBLE_EQ(doubleView(63000), 5.0);
   EXPECT_DOUBLE_EQ(doubleView(63001), 6.0);
   EXPECT_DOUBLE_EQ(doubleView(63002), 7.0);
   EXPECT_EQ(stringView(0), "foo0");
   EXPECT_EQ(stringView(1), "foo1");
   EXPECT_EQ(stringView(2), "foo2");
   EXPECT_EQ(stringView(10000), "foo10000");
   EXPECT_EQ(stringView(10001), "foo10001");
   EXPECT_EQ(stringView(21000), "foo0");
   EXPECT_EQ(stringView(21001), "foo1");
   EXPECT_EQ(stringView(21002), "foo2");
   EXPECT_EQ(stringView(31000), "foo10000");
   EXPECT_EQ(stringView(31001), "foo10001");
   EXPECT_EQ(stringView(42000), "foo0");
   EXPECT_EQ(stringView(42001), "foo1");
   EXPECT_EQ(stringView(42002), "foo2");
   EXPECT_EQ(stringView(52000), "foo10000");
   EXPECT_EQ(stringView(52001), "foo10001");
   EXPECT_EQ(stringView(63000), "foo0");
   EXPECT_EQ(stringView(63001), "foo1");
   EXPECT_EQ(stringView(63002), "foo2");
}

