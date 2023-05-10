#include <ROOT/RConfig.hxx>

#include "ntuple_test.hxx"

#include "TROOT.h"

TEST(RNTuple, RealWorld1)
{
   ROOT::EnableImplicitMT();
   FileRaii fileGuard("test_ntuple_realworld1.root");

   // See https://github.com/olifre/root-io-bench/blob/master/benchmark.cpp
   auto modelWrite = RNTupleModel::Create();
   auto wrEvent   = modelWrite->MakeField<std::uint32_t>("event");
   auto wrSignal  = modelWrite->MakeField<bool>("signal");
   auto wrEnergy  = modelWrite->MakeField<double>("energy");
   auto wrTimes   = modelWrite->MakeField<std::vector<double>>("times");
   auto wrIndices = modelWrite->MakeField<std::vector<std::uint32_t>>("indices");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      constexpr unsigned int nEvents = 60000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         *wrEvent = i;
         *wrEnergy = rnd.Rndm() * 1000.;
         *wrSignal = i % 2;

         chksumWrite += double(*wrEvent);
         chksumWrite += double(*wrSignal);
         chksumWrite += *wrEnergy;

         auto nTimes = 1 + floor(rnd.Rndm() * 1000.);
         wrTimes->resize(nTimes);
         for (unsigned int n = 0; n < nTimes; ++n) {
            wrTimes->at(n) = 1 + rnd.Rndm()*1000. - 500.;
            chksumWrite += wrTimes->at(n);
         }

         auto nIndices = 1 + floor(rnd.Rndm() * 1000.);
         wrIndices->resize(nIndices);
         for (unsigned int n = 0; n < nIndices; ++n) {
            wrIndices->at(n) = 1 + floor(rnd.Rndm() * 1000.);
            chksumWrite += double(wrIndices->at(n));
         }

         ntuple->Fill();
      }
   }

   auto modelRead  = RNTupleModel::Create();
   auto rdEvent   = modelRead->MakeField<std::uint32_t>("event");
   auto rdSignal  = modelRead->MakeField<bool>("signal");
   auto rdEnergy  = modelRead->MakeField<double>("energy");
   auto rdTimes   = modelRead->MakeField<std::vector<double>>("times");
   auto rdIndices = modelRead->MakeField<std::vector<std::uint32_t>>("indices");

   double chksumRead = 0.0;
   auto ntuple = RNTupleReader::Open(std::move(modelRead), "myNTuple", fileGuard.GetPath());
   for (auto entryId : *ntuple) {
      ntuple->LoadEntry(entryId);
      chksumRead += double(*rdEvent) + double(*rdSignal) + *rdEnergy;
      for (auto t : *rdTimes) chksumRead += t;
      for (auto ind : *rdIndices) chksumRead += double(ind);
   }

   // The floating point arithmetic should have been executed in the same order for reading and writing,
   // thus we expect the checksums to be bitwise identical
   EXPECT_EQ(chksumRead, chksumWrite);
}


// Stress test the asynchronous cluster pool by a deliberately unfavourable read pattern
TEST(RNTuple, RandomAccess)
{
   ROOT::EnableImplicitMT();
   FileRaii fileGuard("test_ntuple_random_access.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrValue   = modelWrite->MakeField<std::int32_t>("value", 42);

   constexpr unsigned int nEvents = 1000000;
   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      options.SetApproxZippedClusterSize(nEvents * sizeof(std::int32_t) / 10);
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath(), options);
      for (unsigned int i = 0; i < nEvents; ++i)
         ntuple->Fill();
   }

   RNTupleReadOptions options;
   options.SetClusterCache(RNTupleReadOptions::EClusterCache::kOn);
   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath(), options);
   EXPECT_EQ(10, ntuple->GetDescriptor()->GetNClusters());

   auto viewValue = ntuple->GetView<std::int32_t>("value");

   std::int32_t sum = 0;
   constexpr unsigned int nSamples = 1000;
   TRandom3 rnd(42);
   for (unsigned int i = 0; i < 1000; ++i) {
      auto entryId = floor(rnd.Rndm() * (nEvents - 1));
      sum += viewValue(entryId);
   }
   EXPECT_EQ(42 * nSamples, sum);
}


#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
TEST(RNTuple, LargeFile1)
{
   ROOT::EnableImplicitMT();
   FileRaii fileGuard("test_large_file1.root");

   auto modelWrite = RNTupleModel::Create();
   auto& wrEnergy  = *modelWrite->MakeField<double>("energy");

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
      auto rdEnergy  = reader->GetView<double>("energy");

      double chksumRead = 0.0;
      for (auto i : reader->GetEntryRange()) {
         chksumRead += rdEnergy(i);
      }
      EXPECT_EQ(chksumRead, chksumWrite);
   }

   {
      auto f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "READ"));
      EXPECT_TRUE(f);
      auto reader = RNTupleReader::Open(f->Get<RNTuple>("myNTuple"));
      auto rdEnergy  = reader->GetView<double>("energy");

      double chksumRead = 0.0;
      for (auto i : reader->GetEntryRange()) {
         chksumRead += rdEnergy(i);
      }
      EXPECT_EQ(chksumRead, chksumWrite);
   }
}


TEST(RNTuple, LargeFile2)
{
   ROOT::EnableImplicitMT();
   FileRaii fileGuard("test_large_file2.root");

   // Start out with a mini-file created small file
   auto model = RNTupleModel::Create();
   auto pt = model->MakeField<float>("pt", 42.0);
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
      EXPECT_EQ(42.0f, *reader->GetModel()->GetDefaultEntry()->Get<float>("pt"));

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

      auto reader = RNTupleReader::Open(f->Get<RNTuple>("small"));
      reader->LoadEntry(0);
      EXPECT_EQ(42.0f, *reader->GetModel()->GetDefaultEntry()->Get<float>("pt"));

      reader = RNTupleReader::Open(f->Get<RNTuple>("large"));
      auto viewE = reader->GetView<double>("E");
      double chksumRead = 0.0;
      for (auto i : reader->GetEntryRange()) {
         chksumRead += viewE(i);
      }
      EXPECT_EQ(chksumRead, chksumWrite);
   }
}
#endif

TEST(RNTuple, SmallClusters)
{
   FileRaii fileGuard("test_ntuple_small_clusters.root");

   {
      auto model = RNTupleModel::Create();
      auto fldVec = model->MakeField<std::vector<float>>("vec");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      fldVec->push_back(1.0);
      writer->Fill();
   }
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      auto desc = reader->GetDescriptor();
      auto colId = desc->FindLogicalColumnId(desc->FindFieldId("vec"), 0);
      EXPECT_EQ(EColumnType::kSplitIndex64, desc->GetColumnDescriptor(colId).GetModel().GetType());
      reader->LoadEntry(0);
      auto entry = reader->GetModel()->GetDefaultEntry();
      EXPECT_FLOAT_EQ(1u, entry->Get<std::vector<float>>("vec")->size());
      EXPECT_FLOAT_EQ(1.0, entry->Get<std::vector<float>>("vec")->at(0));
   }

   {
      auto model = RNTupleModel::Create();
      auto fldVec = model->MakeField<std::vector<float>>("vec");
      RNTupleWriteOptions options;
      options.SetHasSmallClusters(true);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      fldVec->push_back(1.0);
      writer->Fill();
   }
   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      auto desc = reader->GetDescriptor();
      auto colId = desc->FindLogicalColumnId(desc->FindFieldId("vec"), 0);
      EXPECT_EQ(EColumnType::kSplitIndex32, desc->GetColumnDescriptor(colId).GetModel().GetType());
      reader->LoadEntry(0);
      auto entry = reader->GetModel()->GetDefaultEntry();
      EXPECT_FLOAT_EQ(1u, entry->Get<std::vector<float>>("vec")->size());
      EXPECT_FLOAT_EQ(1.0, entry->Get<std::vector<float>>("vec")->at(0));
   }

   // Throw on attempt to commit cluster > 512MB
   auto model = RNTupleModel::Create();
   auto fldVec = model->MakeField<std::vector<float>>("vec");
   RNTupleWriteOptions options;
   options.SetHasSmallClusters(true);
   options.SetMaxUnzippedClusterSize(1000 * 1000 * 1000); // 1GB
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
   fldVec->push_back(1.0);
   // One float and one 32bit integer per entry
   for (unsigned int i = 0; i < (300 * 1000 * 1000) / 8; ++i) {
      writer->Fill();
   }
   writer->Fill();
   EXPECT_THROW(writer->CommitCluster(), ROOT::Experimental::RException);

   // On destruction of the writer, the exception in CommitCluster() produces an error log
   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kError, "[ROOT.NTuple]",
      "failure committing ntuple: invalid attempt to write a cluster > 512MiB", false /* matchFullMessage */);
   writer = nullptr;
}
