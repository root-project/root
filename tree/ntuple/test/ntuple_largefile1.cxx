#include <ROOT/RConfig.hxx>

#include "ntuple_test.hxx"

#include <TRandom3.h>

#include <algorithm>

TEST(RNTuple, LargeFile1)
{
#ifdef R__USE_IMT
   IMTRAII _;
#endif
   FileRaii fileGuard("test_large_file1.root");

   auto modelWrite = RNTupleModel::Create();
   auto &wrEnergy = *modelWrite->MakeField<double>("energy");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath(), options);
      constexpr std::uint64_t nEvents = 1024 * 1024 * 256; // Exceed 2GB file size
      for (std::uint64_t i = 0; i < nEvents; ++i) {
         wrEnergy = rnd.Rndm();
         chksumWrite += wrEnergy;
         ntuple->Fill();
      }
   }
#ifdef R__SEEK64
   FILE *file = fopen64(fileGuard.GetPath().c_str(), "rb");
   ASSERT_TRUE(file != nullptr);
   EXPECT_EQ(0, fseeko64(file, 0, SEEK_END));
   EXPECT_GT(ftello64(file), 2048LL * 1024LL * 1024LL);
#else
   FILE *file = fopen(fileGuard.GetPath().c_str(), "rb");
   ASSERT_TRUE(file != nullptr);
   EXPECT_EQ(0, fseek(file, 0, SEEK_END));
   EXPECT_GT(ftell(file), 2048LL * 1024LL * 1024LL);
#endif
   fclose(file);

   {
      auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
      auto rdEnergy = reader->GetView<double>("energy");

      double chksumRead = 0.0;
      for (auto i : reader->GetEntryRange()) {
         chksumRead += rdEnergy(i);
      }
      EXPECT_EQ(chksumRead, chksumWrite);
   }

   {
      auto f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      EXPECT_TRUE(f);
      auto ntuple = std::unique_ptr<ROOT::RNTuple>(f->Get<ROOT::RNTuple>("myNTuple"));
      auto reader = RNTupleReader::Open(*ntuple);
      auto rdEnergy = reader->GetView<double>("energy");

      double chksumRead = 0.0;
      for (auto i : reader->GetEntryRange()) {
         chksumRead += rdEnergy(i);
      }
      EXPECT_EQ(chksumRead, chksumWrite);
   }
}
