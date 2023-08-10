#include "ntuple_test.hxx"
#include <TFile.h>

using RRawFile = ROOT::Internal::RRawFile;

TEST(RNTuple, UncompressedShortRead)
{
   // ensure file clean-up after test completes
   FileRaii fileGuard("test_ntuple.root");

   // enable short read v2
   RRawFile::GetFailureInjectionParams().failureType = RRawFile::SetFailureType(RRawFile::FailureType::ShortRead);
   
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