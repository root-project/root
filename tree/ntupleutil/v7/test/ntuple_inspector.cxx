#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RNTupleOptions.hxx>

#include <TFile.h>
#include <TTree.h>

#include <cstdio>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "ntupleutil_test.hxx"

using ROOT::Experimental::RNTupleInspector;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::RNTupleReader;
using ROOT::Experimental::RNTupleWriteOptions;
using ROOT::Experimental::RNTupleWriter;

TEST(RNTupleInspector, Name)
{
   FileRaii fileGuard("test_ntuple_inspector_name.root");
   {
      auto model = RNTupleModel::Create();
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto inspector = RNTupleInspector::Create(ntuple).Unwrap();
   EXPECT_EQ("ntuple", inspector->GetName());
}

TEST(RNTupleInspector, CompressionSettings)
{
   FileRaii fileGuard("test_ntuple_inspector_size_single_int_field.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int32_t>("i");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      *nFldInt = 42;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());

   auto inspector = RNTupleInspector::Create(ntuple).Unwrap();
   EXPECT_EQ(505, inspector->GetCompressionSettings());
}

TEST(RNTupleInspector, SizeUncompressed)
{
   FileRaii fileGuard("test_ntuple_inspector_size_uncompressed.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int32_t>("i");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      *nFldInt = 42;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());

   auto inspector = RNTupleInspector::Create(ntuple).Unwrap();
   EXPECT_EQ(sizeof(int32_t), inspector->GetUncompressedSize());
   EXPECT_EQ(inspector->GetCompressedSize(), inspector->GetUncompressedSize());
}

TEST(RNTupleInspector, SizeCompressed)
{
   FileRaii fileGuard("test_ntuple_inspector_size_uncompressed.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int32_t>("i");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int32_t i = 0; i < 50; ++i) {
         *nFldInt = i;
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());

   auto inspector = RNTupleInspector::Create(ntuple).Unwrap();
   EXPECT_NE(inspector->GetCompressedSize(), inspector->GetUncompressedSize());
}

TEST(RNTupleInspector, SizeEmpty)
{
   FileRaii fileGuard("test_ntuple_inspector_size_empty.root");
   {
      auto model = RNTupleModel::Create();
      RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());

   auto inspector = RNTupleInspector::Create(ntuple).Unwrap();
   EXPECT_EQ(0, inspector->GetCompressedSize());
   EXPECT_EQ(0, inspector->GetUncompressedSize());
}

TEST(RNTupleInspector, SingleIntFieldCompression)
{
   FileRaii fileGuard("test_ntuple_inspector_size_single_int_field.root");
   {
      auto model = RNTupleModel::Create();
      auto nFldInt = model->MakeField<std::int32_t>("i");

      auto writeOptions = RNTupleWriteOptions();
      writeOptions.SetCompression(505);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), writeOptions);

      for (int32_t i = 0; i < 50; ++i) {
         *nFldInt = i;
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());

   auto inspector = RNTupleInspector::Create(ntuple).Unwrap();
   EXPECT_LT(0, inspector->GetCompressedSize());
   EXPECT_LT(inspector->GetCompressedSize(), inspector->GetUncompressedSize());
   EXPECT_GT(inspector->GetCompressionFactor(), 1);
}
