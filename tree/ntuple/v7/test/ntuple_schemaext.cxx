#include "ntuple_test.hxx"

TEST(RNTuple, SchemaExtensionSimple)
{
   FileRaii fileGuard("test_ntuple_schemaext_simple.root");
   std::array<double, 2> refArray{1.0, 24.0};
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 42.0);

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      auto modelUpdater = ntuple->GetIncrementalModelUpdater();
      modelUpdater->BeginUpdate();
      std::array<double, 2> fieldArray = refArray;
      modelUpdater->AddField<std::array<double, 2>>("array", &fieldArray);
      modelUpdater->CommitUpdate();

      ntuple->Fill();
      *fieldPt = 12.0;
      fieldArray[1] = 1337.0;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(2U, ntuple->GetNEntries());
   EXPECT_EQ(4U, ntuple->GetDescriptor()->GetNFields());

   auto pt = ntuple->GetView<float>("pt");
   auto array = ntuple->GetView<std::array<double, 2>>("array");
   EXPECT_EQ(42.0, pt(0));
   EXPECT_EQ(12.0, pt(1));
   EXPECT_EQ(refArray, array(0));
   EXPECT_EQ(1337.0, array(1)[1]);
}

TEST(RNTuple, SchemaExtensionInvalidUse)
{
   FileRaii fileGuard("test_ntuple_schemaext_invalid.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 42.0);

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      auto entry = ntuple->GetModel()->CreateEntry();

      auto modelUpdater = ntuple->GetIncrementalModelUpdater();
      modelUpdater->BeginUpdate();
      double d;
      modelUpdater->AddField<double>("d", &d);
      // Cannot fill if the model is not frozen
      EXPECT_THROW(ntuple->Fill(), ROOT::Experimental::RException);
      // Trying to create an entry should throw if model is not frozen
      EXPECT_THROW((void)ntuple->GetModel()->CreateEntry(), ROOT::Experimental::RException);
      modelUpdater->CommitUpdate();

      // Using an entry that does not match the model should throw
      EXPECT_THROW(ntuple->Fill(*entry), ROOT::Experimental::RException);

      ntuple->Fill();
      auto entry2 = ntuple->GetModel()->CreateEntry();
      ntuple->Fill(*entry2);

      // Trying to update the schema after the first `Fill()` call should throw
      EXPECT_THROW(modelUpdater->BeginUpdate(), ROOT::Experimental::RException);

      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(3U, ntuple->GetNEntries());
   EXPECT_EQ(3U, ntuple->GetDescriptor()->GetNFields());
}

TEST(RNTuple, SchemaExtensionMultiple)
{
   FileRaii fileGuard("test_ntuple_schemaext_multiple.root");
   std::vector<std::uint32_t> refVec{0x00, 0xff, 0x55, 0xaa};
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 42.0);

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      auto modelUpdater = ntuple->GetIncrementalModelUpdater();
      modelUpdater->BeginUpdate();
      std::vector<std::uint32_t> fieldVec;
      modelUpdater->AddField<std::vector<std::uint32_t>>("vec", &fieldVec);
      modelUpdater->CommitUpdate();

      modelUpdater->BeginUpdate();
      float fieldFloat = 10.0;
      modelUpdater->AddField<float>("f", &fieldFloat);
      modelUpdater->CommitUpdate();

      modelUpdater->BeginUpdate();
      std::uint8_t u8 = 0x7f;
      modelUpdater->AddField<std::uint8_t>("u8", &u8);
      modelUpdater->CommitUpdate();

      ntuple->Fill();
      *fieldPt = 12.0;
      fieldFloat = 1.0;
      fieldVec = refVec;
      u8 = 0xaa;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(2U, ntuple->GetNEntries());
   EXPECT_EQ(6U, ntuple->GetDescriptor()->GetNFields());

   auto pt = ntuple->GetView<float>("pt");
   auto vec = ntuple->GetView<std::vector<std::uint32_t>>("vec");
   auto f = ntuple->GetView<float>("f");
   auto u8 = ntuple->GetView<std::uint8_t>("u8");
   EXPECT_EQ(42.0, pt(0));
   EXPECT_EQ(12.0, pt(1));
   EXPECT_EQ(std::vector<std::uint32_t>{}, vec(0));
   EXPECT_EQ(refVec, vec(1));
   EXPECT_EQ(10.0, f(0));
   EXPECT_EQ(1.0, f(1));
   EXPECT_EQ(0x7f, u8(0));
   EXPECT_EQ(0xaa, u8(1));
}

// Based on the RealWorld1 test in `ntuple_extended.cxx`, but here some fields are added after the fact
TEST(RNTuple, SchemaExtensionRealWorld1)
{
   ROOT::EnableImplicitMT();
   FileRaii fileGuard("test_ntuple_schemaext_realworld1.root");

   // See https://github.com/olifre/root-io-bench/blob/master/benchmark.cpp
   auto modelWrite = RNTupleModel::Create();
   auto wrEvent = modelWrite->MakeField<std::uint32_t>("event");
   auto wrSignal = modelWrite->MakeField<bool>("signal");
   auto wrTimes = modelWrite->MakeField<std::vector<double>>("times");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());

      auto modelUpdater = ntuple->GetIncrementalModelUpdater();
      modelUpdater->BeginUpdate();
      std::vector<std::uint32_t> fieldVec;
      double wrEnergy;
      std::vector<std::uint32_t> wrIndices;
      modelUpdater->AddField<double>("energy", &wrEnergy);
      modelUpdater->AddField<std::vector<std::uint32_t>>("indices", &wrIndices);
      modelUpdater->CommitUpdate();

      constexpr unsigned int nEvents = 60000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         *wrEvent = i;
         wrEnergy = rnd.Rndm() * 1000.;
         *wrSignal = i % 2;

         chksumWrite += double(*wrEvent);
         chksumWrite += double(*wrSignal);
         chksumWrite += wrEnergy;

         auto nTimes = 1 + floor(rnd.Rndm() * 1000.);
         wrTimes->resize(nTimes);
         for (unsigned int n = 0; n < nTimes; ++n) {
            wrTimes->at(n) = 1 + rnd.Rndm() * 1000. - 500.;
            chksumWrite += wrTimes->at(n);
         }

         auto nIndices = 1 + floor(rnd.Rndm() * 1000.);
         wrIndices.resize(nIndices);
         for (unsigned int n = 0; n < nIndices; ++n) {
            wrIndices.at(n) = 1 + floor(rnd.Rndm() * 1000.);
            chksumWrite += double(wrIndices.at(n));
         }

         ntuple->Fill();
      }
   }

   auto modelRead = RNTupleModel::Create();
   auto rdEvent = modelRead->MakeField<std::uint32_t>("event");
   auto rdSignal = modelRead->MakeField<bool>("signal");
   auto rdEnergy = modelRead->MakeField<double>("energy");
   auto rdTimes = modelRead->MakeField<std::vector<double>>("times");
   auto rdIndices = modelRead->MakeField<std::vector<std::uint32_t>>("indices");

   double chksumRead = 0.0;
   auto ntuple = RNTupleReader::Open(std::move(modelRead), "myNTuple", fileGuard.GetPath());
   for (auto entryId : *ntuple) {
      ntuple->LoadEntry(entryId);
      chksumRead += double(*rdEvent) + double(*rdSignal) + *rdEnergy;
      for (auto t : *rdTimes)
         chksumRead += t;
      for (auto ind : *rdIndices)
         chksumRead += double(ind);
   }

   // The floating point arithmetic should have been executed in the same order for reading and writing,
   // thus we expect the checksums to be bitwise identical
   EXPECT_EQ(chksumRead, chksumWrite);
}
