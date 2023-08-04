#include "ntuple_test.hxx"
#include <TFile.h>

#define RRAWFILE_TESTING_MODE

// bit flip in footer. compressed file
TEST(RNTuple, BitFlipInPageCompressedFile)
{
   constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
   // ensure file clean-up after test completes
   FileRaii fileGuard(kNTupleFileName);

   {
      // create an RNTuple instance with 2 fields and no compression and fill with data
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(505);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

      // TODO: add more content to the file than just one entry
      *ptrPt = 1.0;
      *ptrE = 2.0;
      writer->Fill();
   }

   // open file and inject bit flip
   {
      auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
      auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");

      std::unique_ptr<RNTupleReader> ntupleReader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
      const auto descriptor = ntupleReader->GetDescriptor();
      auto fieldId = descriptor->FindFieldId("pt");
      auto columnId = descriptor->FindPhysicalColumnId(fieldId,0);
      auto clusterId = descriptor->FindClusterId(columnId,0);
      const auto &clusterDescriptor = descriptor->GetClusterDescriptor(clusterId);
      auto &pageRangeXcl = clusterDescriptor.GetPageRange(clusterId);
      const auto &pageInfo = pageRangeXcl.fPageInfos[0];
      auto loc = pageInfo.fLocator;
      auto nelem = pageInfo.fNElements;
      auto offset = loc.GetPosition<std::uint64_t>();

      ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = offset;
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = nelem;
   }

   auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ_COMPRESS");

}



// bit flip in footer. uncompressed file
TEST(RNTuple, BitFlipInPageUncompressedFile)
{
   constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
   // ensure file clean-up after test completes
   FileRaii fileGuard(kNTupleFileName);

   {
      // create an RNTuple instance with 2 fields and no compression and fill with data
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

      // TODO: add more content to the file than just one entry
      *ptrPt = 1.0;
      *ptrE = 2.0;
      writer->Fill();
   }

   // open file and inject bit flip
   {
      auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
      auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");

      std::unique_ptr<RNTupleReader> ntupleReader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
      const auto descriptor = ntupleReader->GetDescriptor();
      auto fieldId = descriptor->FindFieldId("pt");
      auto columnId = descriptor->FindPhysicalColumnId(fieldId,0);
      auto clusterId = descriptor->FindClusterId(columnId,0);
      const auto &clusterDescriptor = descriptor->GetClusterDescriptor(clusterId);
      auto &pageRangeXcl = clusterDescriptor.GetPageRange(clusterId);
      const auto &pageInfo = pageRangeXcl.fPageInfos[0];
      auto loc = pageInfo.fLocator;
      auto nelem = pageInfo.fNElements;
      auto offset = loc.GetPosition<std::uint64_t>();

      ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = offset;
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = nelem;
   }

   // attempt to open file again
   try {
      auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
   }
   catch (const std::exception& ex){

      FAIL() << "Operation failed: " << ex.what();
   }
}



// bit flip in footer. compressed file
TEST(RNTuple, BitFlipInFooterCompressedFile)
{
   constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
   // ensure file clean-up after test completes
   FileRaii fileGuard(kNTupleFileName);

   {
      // create an RNTuple instance with 2 fields and no compression and fill with data
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(505);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

      // TODO: add more content to the file than just one entry
      *ptrPt = 1.0;
      *ptrE = 2.0;
      writer->Fill();
   }

   // open file and inject bit flip
   {
      auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
      auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = ntpl->GetSeekFooter();
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = ntpl->GetSeekFooter() + ntpl->GetSeekFooter();
   }

   auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ_COMPRESS");

}


// bit flip in footer. uncompressed file
TEST(RNTuple, BitFlipInFooterUncompressedFile)
{
   constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
   // ensure file clean-up after test completes
   FileRaii fileGuard(kNTupleFileName);

   {
      // create an RNTuple instance with 2 fields and no compression and fill with data
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

      // TODO: add more content to the file than just one entry
      *ptrPt = 1.0;
      *ptrE = 2.0;
      writer->Fill();
   }

   // open file and inject bit flip
   {
      auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
      auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = ntpl->GetSeekFooter();
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = ntpl->GetSeekFooter() + ntpl->GetNBytesFooter();
   }

   // attempt to open file again
   try {
      auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
   }
   catch (const std::exception& ex){

      FAIL() << "Operation failed: " << ex.what();
   }
}



// bit flip in header. compressed file
TEST(RNTuple, BitFlipInHeaderCompressedFile)
{
   constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
   // ensure file clean-up after test completes
   FileRaii fileGuard(kNTupleFileName);

   {
      // create an RNTuple instance with 2 fields and no compression and fill with data
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(505);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

      // TODO: add more content to the file than just one entry
      *ptrPt = 1.0;
      *ptrE = 2.0;
      writer->Fill();
   }

   // open file and inject bit flip
   {
      auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
      auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = ntpl->GetSeekHeader();
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = ntpl->GetSeekHeader() + ntpl->GetNBytesHeader();
   }

   auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ_COMPRESS");

}


// bit flip in header. uncompressed file
TEST(RNTuple, BitFlipInHeaderUncompressedFile)
{
   constexpr char const* kNTupleFileName = "test_ntuple_compression_bitflips.root";
   // ensure file clean-up after test completes
   FileRaii fileGuard(kNTupleFileName);

   {
      // create an RNTuple instance with 2 fields and no compression and fill with data
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

      // TODO: add more content to the file than just one entry
      *ptrPt = 1.0;
      *ptrE = 2.0;
      writer->Fill();
   }

   // open file and inject bit flip
   {
      auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
      auto ntpl = f->Get<ROOT::Experimental::RNTuple>("ntuple");
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_begin = ntpl->GetSeekHeader();
      ROOT::Internal::RRawFile::GetBitFlipParams().rng_end = ntpl->GetSeekHeader() + ntpl->GetNBytesHeader();
   }

   // attempt to open file again
   try {
      auto f = TFile::Open(fileGuard.GetPath().c_str(), "READ");
   }
   catch (const std::exception& ex){

      FAIL() << "Operation failed: " << ex.what();
   }
}



// no error injection. uncompressed file
TEST(RNTuple, UncompressedAndNoBitFlip)
{
   //test
   // ensure file clean-up after test completes
   FileRaii fileGuard("test_ntuple_uncompressed_bitflips.root");
   
   {
      // create an RNTuple instance with 2 fields and no compression and fill with data
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

      // TODO: add more content to the file than just one entry
      *ptrPt = 1.0;
      *ptrE = 2.0;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto viewPt = reader->GetView<float>("pt");

   for (auto i : reader->GetEntryRange()) {
     std::cout << viewPt(i) << std::endl;
   }

   EXPECT_EQ(1.0, viewPt(0));
}



// no error injection. compressed file
TEST(RNTuple, CompressedAndNoBitFlip)
{
   // ensure file clean-up after test completes
   FileRaii fileGuard("test_ntuple_compression_bitflips.root");
   
   {
      // create an RNTuple instance with 2 fields and no compression and fill with data
      auto model = RNTupleModel::Create();
      auto ptrPt = model->MakeField<float>("pt");
      auto ptrE = model->MakeField<float>("energy");

      RNTupleWriteOptions options;
      options.SetCompression(505);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);

      // TODO: add more content to the file than just one entry
      *ptrPt = 1.0;
      *ptrE = 2.0;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto viewPt = reader->GetView<float>("pt");

   for (auto i : reader->GetEntryRange()) {
     std::cout << viewPt(i) << std::endl;
   }

   EXPECT_EQ(1.0, viewPt(0));
}