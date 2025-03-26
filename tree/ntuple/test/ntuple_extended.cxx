#include <ROOT/RConfig.hxx>

#include "ntuple_test.hxx"

#include <TRandom3.h>
#include <TROOT.h>

#include <algorithm>
#include <random>

TEST(RNTuple, RealWorld1)
{
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
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

TEST(RNTuple, Double32IMT)
{
   // Tests if parallel decompression correctly compresses the on-disk float to an in-memory double
#ifdef R__USE_IMT
   IMTRAII _;
#endif
   FileRaii fileGuard("test_ntuple_double32_imt.root");

   constexpr int kNEvents = 10;

   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("pt", "Double32_t").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      auto ptrPt = writer->GetModel().GetDefaultEntry().GetPtr<double>("pt");

      for (int i = 0; i < kNEvents; ++i) {
         *ptrPt = i;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewPt = reader->GetView<double>("pt");
   for (int i = 0; i < kNEvents; ++i) {
      EXPECT_DOUBLE_EQ(i, viewPt(i));
   }
}

TEST(RNTuple, MultiColumnExpansion)
{
   // Tests if on-disk columns that expand to multiple in-memory types are correctly handled
#ifdef R__USE_IMT
   IMTRAII _;
#endif
   FileRaii fileGuard("test_ntuple_multi_column_expansion.root");

   constexpr int kNEvents = 1000;

   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("pt", "Double32_t").Unwrap());
      RNTupleWriteOptions options;
      options.SetInitialUnzippedPageSize(8);
      options.SetMaxUnzippedPageSize(32);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);

      auto ptrPt = writer->GetModel().GetDefaultEntry().GetPtr<double>("pt");

      for (int i = 0; i < kNEvents; ++i) {
         *ptrPt = i;
         writer->Fill();
         if (i % 50 == 0)
            writer->CommitCluster();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewPt = reader->GetView<double>("pt");
   auto viewPtAsFloat = reader->GetView<float>("pt");

   std::random_device rd;
   std::mt19937 gen(rd());
   std::vector<unsigned int> indexes;
   indexes.reserve(kNEvents);
   for (unsigned int i = 0; i < kNEvents; ++i)
      indexes.emplace_back(i);
   std::shuffle(indexes.begin(), indexes.end(), gen);

   std::bernoulli_distribution dist(0.5);
   for (auto idx : indexes) {
      if (dist(gen)) {
         EXPECT_DOUBLE_EQ(idx, viewPt(idx));
         EXPECT_DOUBLE_EQ(idx, viewPtAsFloat(idx));
      } else {
         EXPECT_DOUBLE_EQ(idx, viewPtAsFloat(idx));
         EXPECT_DOUBLE_EQ(idx, viewPt(idx));
      }
   }
}

TEST(RNTuple, LargePages)
{
   FileRaii fileGuard("test_ntuple_large_pages.root");

   for (const auto useBufferedWrite : {true, false}) {
      {
         auto model = RNTupleModel::Create();
         auto fldRnd = model->MakeField<std::uint32_t>("rnd");
         RNTupleWriteOptions options;
         // Larger than the 16MB compression block limit
         options.SetMaxUnzippedPageSize(32 * 1024 * 1024);
         options.SetUseBufferedWrite(useBufferedWrite);
         auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);

         std::mt19937 gen;
         std::uniform_int_distribution<std::uint32_t> distrib;
         for (int i = 0; i < 25 * 1000 * 1000; ++i) { // 100 MB of int data
            *fldRnd = distrib(gen);
            writer->Fill();
         }
         writer.reset();
      }

      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      const auto &desc = reader->GetDescriptor();
      const auto rndColId = desc.FindPhysicalColumnId(desc.FindFieldId("rnd"), 0, 0);
      const auto &clusterDesc = desc.GetClusterDescriptor(desc.FindClusterId(rndColId, 0));
      EXPECT_GT(clusterDesc.GetPageRange(rndColId).Find(0).GetLocator().GetNBytesOnStorage(), kMAXZIPBUF);

      auto viewRnd = reader->GetView<std::uint32_t>("rnd");
      std::mt19937 gen;
      std::uniform_int_distribution<std::uint32_t> distrib;
      for (const auto i : reader->GetEntryRange()) {
         EXPECT_EQ(distrib(gen), viewRnd(i));
      }
   }
}
