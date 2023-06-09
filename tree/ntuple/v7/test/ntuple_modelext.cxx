#include "ntuple_test.hxx"

namespace {
struct RFieldBaseTest : public ROOT::Experimental::Detail::RFieldBase {
   /// Returns the global index of the first entry that has a stored on-disk value.  For deferred fields, this allows
   /// for differentiating zero-initialized values read before the addition of the field from actual stored data.
   NTupleSize_t GetFirstEntry() const
   {
      auto fnColumnElementIndexToEntry = [&](NTupleSize_t columnElementIndex) -> std::size_t {
         std::size_t result = columnElementIndex;
         for (auto f = static_cast<const RFieldBase *>(this); f != nullptr; f = f->GetParent()) {
            auto parent = f->GetParent();
            if (parent && (parent->GetStructure() == ROOT::Experimental::kCollection ||
                           parent->GetStructure() == ROOT::Experimental::kVariant))
               return 0U;
            result /= std::max(f->GetNRepetitions(), std::size_t{1U});
         }
         return result;
      };

      if (fPrincipalColumn)
         return fnColumnElementIndexToEntry(fPrincipalColumn->GetFirstElementIndex());
      if (!fSubFields.empty())
         return static_cast<const RFieldBaseTest &>(*fSubFields[0]).GetFirstEntry();
      R__ASSERT(false);
      return 0;
   }
};

static std::size_t GetFirstEntry(const RFieldBase *f)
{
   return static_cast<const RFieldBaseTest *>(f)->GetFirstEntry();
}
} // anonymous namespace

TEST(RNTuple, ModelExtensionSimple)
{
   FileRaii fileGuard("test_ntuple_modelext_simple.root");
   std::array<double, 2> refArray{1.0, 24.0};
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 42.0);

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      ntuple->Fill();
      *fieldPt = 12.0;
      ntuple->Fill();

      auto modelUpdater = ntuple->CreateModelUpdater();
      modelUpdater->BeginUpdate();
      std::array<double, 2> fieldArray = refArray;
      modelUpdater->AddField<std::array<double, 2>>("array", &fieldArray);
      modelUpdater->CommitUpdate();

      ntuple->Fill();
      *fieldPt = 1.0;
      fieldArray[1] = 1337.0;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(4U, ntuple->GetNEntries());
   EXPECT_EQ(4U, ntuple->GetDescriptor()->GetNFields());
   EXPECT_EQ(0U, GetFirstEntry(ntuple->GetModel()->GetField("pt")));
   EXPECT_EQ(2U, GetFirstEntry(ntuple->GetModel()->GetField("array")));

   auto pt = ntuple->GetView<float>("pt");
   auto array = ntuple->GetView<std::array<double, 2>>("array");
   EXPECT_EQ(42.0, pt(0));
   EXPECT_EQ(12.0, pt(1));
   EXPECT_EQ(12.0, pt(2));
   EXPECT_EQ(1.0, pt(3));

   // There is no on-disk data for this field before entry 2, i.e., entry range [0..1] yields zeroes
   EXPECT_EQ((std::array<double, 2>{0.0, 0.0}), array(0));
   EXPECT_EQ((std::array<double, 2>{0.0, 0.0}), array(1));
   EXPECT_EQ(refArray, array(2));
   EXPECT_EQ(1337.0, array(3)[1]);
}

TEST(RNTuple, ModelExtensionInvalidUse)
{
   FileRaii fileGuard("test_ntuple_modelext_invalid.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 42.0);

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      auto entry = ntuple->GetModel()->CreateEntry();

      auto modelUpdater = ntuple->CreateModelUpdater();
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

      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(3U, ntuple->GetNEntries());
   EXPECT_EQ(3U, ntuple->GetDescriptor()->GetNFields());
}

TEST(RNTuple, ModelExtensionMultiple)
{
   FileRaii fileGuard("test_ntuple_modelext_multiple.root");
   std::vector<std::uint32_t> refVec{0x00, 0xff, 0x55, 0xaa};
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 42.0);

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      ntuple->Fill();
      *fieldPt = 2.0;
      ntuple->Fill();
      ntuple->CommitCluster();
      *fieldPt = 4.0;
      ntuple->Fill();

      auto modelUpdater = ntuple->CreateModelUpdater();
      modelUpdater->BeginUpdate();
      std::vector<std::uint32_t> fieldVec{0x11223344};
      modelUpdater->AddField<std::vector<std::uint32_t>>("vec", &fieldVec);
      modelUpdater->CommitUpdate();

      modelUpdater->BeginUpdate();
      float fieldFloat = 10.0;
      modelUpdater->AddField<float>("f", &fieldFloat);
      modelUpdater->CommitUpdate();

      modelUpdater->BeginUpdate();
      std::uint8_t u8 = 0x7f;
      std::string str{"default"};
      modelUpdater->AddField<std::uint8_t>("u8", &u8);
      modelUpdater->AddField<std::string>("str", &str);
      modelUpdater->CommitUpdate();

      ntuple->Fill();
      *fieldPt = 12.0;
      fieldFloat = 1.0;
      fieldVec = refVec;
      u8 = 0xaa;
      str = "abcdefABCDEF1234567890!@#$%^&*()";
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(5U, ntuple->GetNEntries());
   EXPECT_EQ(7U, ntuple->GetDescriptor()->GetNFields());
   EXPECT_EQ(0U, GetFirstEntry(ntuple->GetModel()->GetField("pt")));
   EXPECT_EQ(3U, GetFirstEntry(ntuple->GetModel()->GetField("vec")));
   EXPECT_EQ(3U, GetFirstEntry(ntuple->GetModel()->GetField("f")));
   EXPECT_EQ(3U, GetFirstEntry(ntuple->GetModel()->GetField("u8")));
   EXPECT_EQ(3U, GetFirstEntry(ntuple->GetModel()->GetField("str")));

   auto pt = ntuple->GetView<float>("pt");
   auto vec = ntuple->GetView<std::vector<std::uint32_t>>("vec");
   auto f = ntuple->GetView<float>("f");
   auto u8 = ntuple->GetView<std::uint8_t>("u8");
   auto str = ntuple->GetView<std::string>("str");
   EXPECT_EQ(42.0, pt(0));
   EXPECT_EQ(2.0, pt(1));
   EXPECT_EQ(4.0, pt(2));
   EXPECT_EQ(4.0, pt(3));
   EXPECT_EQ(12.0, pt(4));

   EXPECT_EQ(std::vector<std::uint32_t>{}, vec(0));
   EXPECT_EQ(std::vector<std::uint32_t>{}, vec(1));
   EXPECT_EQ(std::vector<std::uint32_t>{}, vec(2));
   EXPECT_EQ(std::vector<std::uint32_t>{0x11223344}, vec(3));
   EXPECT_EQ(refVec, vec(4));

   EXPECT_EQ(0.0, f(0));
   EXPECT_EQ(0.0, f(1));
   EXPECT_EQ(0.0, f(2));
   EXPECT_EQ(10.0, f(3));
   EXPECT_EQ(1.0, f(4));

   EXPECT_EQ(0x00, u8(0));
   EXPECT_EQ(0x00, u8(1));
   EXPECT_EQ(0x00, u8(2));
   EXPECT_EQ(0x7f, u8(3));
   EXPECT_EQ(0xaa, u8(4));

   EXPECT_TRUE(str(0).empty());
   EXPECT_TRUE(str(1).empty());
   EXPECT_TRUE(str(2).empty());
   EXPECT_EQ("default", str(3));
   EXPECT_EQ("abcdefABCDEF1234567890!@#$%^&*()", str(4));
}

TEST(RNTuple, ModelExtensionProject)
{
   FileRaii fileGuard("test_ntuple_modelext_project.root");
   std::vector<std::uint32_t> refVec{0x00, 0xff, 0x55, 0xaa};
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt", 42.0);

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      auto modelUpdater = ntuple->CreateModelUpdater();
      modelUpdater->BeginUpdate();
      std::vector<std::uint32_t> fieldVec;
      modelUpdater->AddField<std::vector<std::uint32_t>>("vec", &fieldVec);
      modelUpdater->CommitUpdate();

      modelUpdater->BeginUpdate();
      auto aliasVec = RFieldBase::Create("aliasVec", "std::vector<std::uint32_t>").Unwrap();
      modelUpdater->AddProjectedField(std::move(aliasVec), [](const std::string &fieldName) {
         if (fieldName == "aliasVec")
            return "vec";
         else
            return "vec._0";
      });
      modelUpdater->CommitUpdate();

      ntuple->Fill();
      *fieldPt = 12.0;
      fieldVec = refVec;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(2U, ntuple->GetNEntries());
   EXPECT_EQ(6U, ntuple->GetDescriptor()->GetNFields());

   auto pt = ntuple->GetView<float>("pt");
   auto aliasVec = ntuple->GetView<std::vector<std::uint32_t>>("aliasVec");
   EXPECT_EQ(42.0, pt(0));
   EXPECT_EQ(12.0, pt(1));
   EXPECT_EQ(std::vector<std::uint32_t>{}, aliasVec(0));
   EXPECT_EQ(refVec, aliasVec(1));
}

// Based on the RealWorld1 test in `ntuple_extended.cxx`, but here some fields are added after the fact
TEST(RNTuple, ModelExtensionRealWorld1)
{
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
   FileRaii fileGuard("test_ntuple_modelext_realworld1.root");

   // See https://github.com/olifre/root-io-bench/blob/master/benchmark.cpp
   auto modelWrite = RNTupleModel::Create();
   auto wrEvent = modelWrite->MakeField<std::uint32_t>("event");
   auto wrSignal = modelWrite->MakeField<bool>("signal");
   auto wrTimes = modelWrite->MakeField<std::vector<double>>("times");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());

      auto modelUpdater = ntuple->CreateModelUpdater();
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

TEST(RNTuple, ModelExtensionComplex)
{
   using doubleAoA_t = std::array<std::array<double, 2>, 2>;
   ROOT::EnableImplicitMT();
   FileRaii fileGuard("test_ntuple_modelext_complex.root");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   auto updateInitialFields = [&](unsigned entryId, std::uint32_t &u32, double &dbl) {
      u32 = entryId;
      dbl = rnd.Rndm() * 1000.;
      chksumWrite += static_cast<double>(u32);
      chksumWrite += dbl;
   };
   auto updateLateFields1 = [&](unsigned entryId, std::vector<double> &vec, doubleAoA_t &aoa) {
      auto size = 1 + floor(rnd.Rndm() * 1000.);
      vec.resize(size);
      for (unsigned i = 0; i < size; ++i) {
         vec[i] = 1 + rnd.Rndm() * 1000. - 500.;
         chksumWrite += vec[i];
      }
      aoa = {{{0.0, static_cast<double>(entryId)}, {42.0, static_cast<double>(entryId)}}};
      chksumWrite += (aoa[0][0] + aoa[0][1]) + (aoa[1][0] + aoa[1][1]);
   };
   auto updateLateFields2 = [&](unsigned entryId, std::variant<std::uint64_t, double> &var) {
      if ((entryId % 2) == 0) {
         std::uint64_t value = entryId + 1;
         var = value;
         chksumWrite += static_cast<double>(value);
      } else {
         double value = entryId;
         var = value;
         chksumWrite += value;
      }
   };

   {
      auto modelWrite = RNTupleModel::Create();
      auto u32 = modelWrite->MakeField<std::uint32_t>("u32");
      auto dbl = modelWrite->MakeField<double>("dbl");
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      // The model is to be extended below with new fields every 10000 entries
      auto modelUpdater = ntuple->CreateModelUpdater();
      for (unsigned int i = 0; i < 10000; ++i) {
         updateInitialFields(i, *u32, *dbl);
         ntuple->Fill();
      }

      // Force the serialization of a page list which will not know about the deferred columns coming later.
      // `RClusterDescriptorBuilder::AddDeferredColumnRanges()` should thus make up page ranges for the missing columns
      ntuple->CommitCluster(true /* commitClusterGroup */);

      modelUpdater->BeginUpdate();
      std::vector<double> dblVec;
      doubleAoA_t doubleAoA;
      modelUpdater->AddField<std::vector<double>>("dblVec", &dblVec);
      modelUpdater->AddField<doubleAoA_t>("doubleAoA", &doubleAoA);
      modelUpdater->CommitUpdate();
      for (unsigned int i = 10000; i < 20000; ++i) {
         updateInitialFields(i, *u32, *dbl);
         updateLateFields1(i, dblVec, doubleAoA);
         ntuple->Fill();
      }

      // A `R(Column|Page)Range` should also be generated in the previous cluster for the columns associated to the
      // `std::variant` field below
      ntuple->CommitCluster();

      modelUpdater->BeginUpdate();
      std::variant<std::uint64_t, double> var;
      modelUpdater->AddField<std::variant<std::uint64_t, double>>("var", &var);
      modelUpdater->CommitUpdate();
      for (unsigned int i = 20000; i < 30000; ++i) {
         updateInitialFields(i, *u32, *dbl);
         updateLateFields1(i, dblVec, doubleAoA);
         updateLateFields2(i, var);
         ntuple->Fill();
      }

      modelUpdater->BeginUpdate();
      bool b = false;
      EmptyStruct noColumns{};
      modelUpdater->AddField<bool>("b", &b);
      modelUpdater->AddField<EmptyStruct>("noColumns", &noColumns);
      modelUpdater->CommitUpdate();
      for (unsigned int i = 30000; i < 40000; ++i) {
         updateInitialFields(i, *u32, *dbl);
         updateLateFields1(i, dblVec, doubleAoA);
         updateLateFields2(i, var);
         b = !b;
         chksumWrite += static_cast<double>(b);
         ntuple->Fill();
      }
   }

   auto modelRead = RNTupleModel::Create();
   auto u32 = modelRead->MakeField<std::uint32_t>("u32");
   auto dbl = modelRead->MakeField<double>("dbl");
   auto dblVec = modelRead->MakeField<std::vector<double>>("dblVec");
   auto doubleAoA = modelRead->MakeField<doubleAoA_t>("doubleAoA");
   auto var = modelRead->MakeField<std::variant<std::uint64_t, double>>("var");
   auto b = modelRead->MakeField<bool>("b");
   auto noColumns = modelRead->MakeField<EmptyStruct>("noColumns");

   double chksumRead = 0.0;
   auto ntuple = RNTupleReader::Open(std::move(modelRead), "myNTuple", fileGuard.GetPath());
   EXPECT_EQ(40000U, ntuple->GetNEntries());
   for (auto entryId : *ntuple) {
      ntuple->LoadEntry(entryId);
      chksumRead += static_cast<double>(*u32) + *dbl;
      for (auto d : *dblVec)
         chksumRead += d;
      chksumRead += ((*doubleAoA)[0][0] + (*doubleAoA)[0][1]) + ((*doubleAoA)[1][0] + (*doubleAoA)[1][1]);
      if (entryId < 20000) {
         EXPECT_FALSE(std::holds_alternative<std::uint64_t>(*var) || std::holds_alternative<double>(*var));
         continue;
      }
      if ((entryId % 2) == 0) {
         chksumRead += std::get<std::uint64_t>(*var);
      } else {
         chksumRead += std::get<double>(*var);
      }
      chksumRead += static_cast<double>(*b);
      (void)*noColumns;
   }

   // The floating point arithmetic should have been executed in the same order for reading and writing,
   // thus we expect the checksums to be bitwise identical
   EXPECT_EQ(chksumRead, chksumWrite);
}
