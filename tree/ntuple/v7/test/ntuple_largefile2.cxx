#include <ROOT/RConfig.hxx>

#include "ntuple_test.hxx"

#include <TRandom3.h>

#include <algorithm>
#include <random>

TEST(RNTuple, LargeFile2)
{
#ifdef R__USE_IMT
   IMTRAII _;
#endif
   FileRaii fileGuard("test_large_file2.root");

   // Start out with a mini-file created small file
   auto model = RNTupleModel::Create();
   *model->MakeField<float>("pt") = 42.0;
   auto writer = RNTupleWriter::Recreate(std::move(model), "small", fileGuard.GetPath());
   writer->Fill();
   writer = nullptr;

   // Update the file with another object
   auto f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));
   std::string str = "one";
   f->WriteObject(&str, "s1");

   // Turn it into a large file
   model = RNTupleModel::Create();
   auto E = model->MakeField<double>("E");
   RNTupleWriteOptions options;
   options.SetCompression(0);
   writer = RNTupleWriter::Append(std::move(model), "large", *f, options);

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   constexpr std::uint64_t nEvents = 1024 * 1024 * 256; // Exceed 2GB file size
   for (std::uint64_t i = 0; i < nEvents; ++i) {
      *E = rnd.Rndm();
      chksumWrite += *E;
      writer->Fill();
   }

   // Add one more object before the ntuple writer commits the footer
   str = "two";
   f->WriteObject(&str, "s2");
   writer = nullptr;
   f->Close();
   f = nullptr;

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

   f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
   {
      auto reader = RNTupleReader::Open("small", fileGuard.GetPath());
      reader->LoadEntry(0);
      EXPECT_EQ(42.0f, *reader->GetModel().GetDefaultEntry().GetPtr<float>("pt"));

      reader = RNTupleReader::Open("large", fileGuard.GetPath());
      auto viewE = reader->GetView<double>("E");
      double chksumRead = 0.0;
      for (auto i : reader->GetEntryRange()) {
         chksumRead += viewE(i);
      }
      EXPECT_EQ(chksumRead, chksumWrite);
   }

   {
      f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      EXPECT_TRUE(f);
      auto s1 = f->Get<std::string>("s1");
      EXPECT_EQ("one", *s1);
      auto s2 = f->Get<std::string>("s2");
      EXPECT_EQ("two", *s2);

      auto small = std::unique_ptr<ROOT::RNTuple>(f->Get<ROOT::RNTuple>("small"));
      auto reader = RNTupleReader::Open(*small);
      reader->LoadEntry(0);
      EXPECT_EQ(42.0f, *reader->GetModel().GetDefaultEntry().GetPtr<float>("pt"));

      auto large = std::unique_ptr<ROOT::RNTuple>(f->Get<ROOT::RNTuple>("large"));
      reader = RNTupleReader::Open(*large);
      auto viewE = reader->GetView<double>("E");
      double chksumRead = 0.0;
      for (auto i : reader->GetEntryRange()) {
         chksumRead += viewE(i);
      }
      EXPECT_EQ(chksumRead, chksumWrite);
   }
}
