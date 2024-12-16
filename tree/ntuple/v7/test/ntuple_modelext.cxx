#include "ntuple_test.hxx"
#include <TRandom3.h>

namespace {
struct RFieldBaseTest : public ROOT::Experimental::RFieldBase {
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

std::size_t GetFirstEntry(const RFieldBase &f)
{
   return static_cast<const RFieldBaseTest &>(f).GetFirstEntry();
}
} // anonymous namespace

TEST(RNTuple, ModelExtensionSimple)
{
   FileRaii fileGuard("test_ntuple_modelext_simple.root");
   std::array<double, 2> refArray{1.0, 24.0};
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());

      *fieldPt = 42.0;
      ntuple->Fill();
      *fieldPt = 12.0;
      ntuple->Fill();

      auto modelUpdater = ntuple->CreateModelUpdater();
      modelUpdater->BeginUpdate();
      auto fieldArray = modelUpdater->MakeField<std::array<double, 2>>("array");
      modelUpdater->CommitUpdate();

      *fieldArray = refArray;
      ntuple->Fill();
      *fieldPt = 1.0;
      fieldArray->at(1) = 1337.0;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(4U, ntuple->GetNEntries());
   EXPECT_EQ(4U, ntuple->GetDescriptor().GetNFields());
   EXPECT_EQ(0U, GetFirstEntry(ntuple->GetModel().GetConstField("pt")));
   EXPECT_EQ(2U, GetFirstEntry(ntuple->GetModel().GetConstField("array")));

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
      *model->MakeField<float>("pt") = 42.0;

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      auto entry = ntuple->GetModel().CreateEntry();

      auto modelUpdater = ntuple->CreateModelUpdater();
      modelUpdater->BeginUpdate();
      auto d = modelUpdater->MakeField<double>("d");
      // Cannot fill if the model is not frozen
      EXPECT_THROW(ntuple->Fill(), ROOT::RException);
      // Trying to create an entry should throw if model is not frozen
      EXPECT_THROW((void)ntuple->GetModel().CreateEntry(), ROOT::RException);
      modelUpdater->CommitUpdate();

      // Using an entry that does not match the model should throw
      EXPECT_THROW(ntuple->Fill(*entry), ROOT::RException);

      ntuple->Fill();
      auto entry2 = ntuple->GetModel().CreateEntry();
      ntuple->Fill(*entry2);

      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(3U, ntuple->GetNEntries());
   EXPECT_EQ(3U, ntuple->GetDescriptor().GetNFields());
}

TEST(RNTuple, ModelExtensionMultiple)
{
   FileRaii fileGuard("test_ntuple_modelext_multiple.root");
   std::vector<std::uint32_t> refVec{0x00, 0xff, 0x55, 0xaa};
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());

      *fieldPt = 42.0;
      ntuple->Fill();
      *fieldPt = 2.0;
      ntuple->Fill();
      ntuple->CommitCluster();
      *fieldPt = 4.0;
      ntuple->Fill();

      auto modelUpdater = ntuple->CreateModelUpdater();
      modelUpdater->BeginUpdate();
      auto fieldVec = modelUpdater->MakeField<std::vector<std::uint32_t>>("vec");
      fieldVec->emplace_back(0x11223344);
      modelUpdater->CommitUpdate();

      modelUpdater->BeginUpdate();
      auto fieldFloat = modelUpdater->MakeField<float>("f");
      *fieldFloat = 10;
      modelUpdater->CommitUpdate();

      modelUpdater->BeginUpdate();
      auto u8 = modelUpdater->MakeField<std::uint8_t>("u8");
      auto str = modelUpdater->MakeField<std::string>("str");
      modelUpdater->CommitUpdate();

      *u8 = 0x7f;
      *str = "default";
      ntuple->Fill();
      *fieldPt = 12.0;
      *fieldFloat = 1.0;
      *fieldVec = refVec;
      *u8 = 0xaa;
      *str = "abcdefABCDEF1234567890!@#$%^&*()";
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(5U, ntuple->GetNEntries());
   EXPECT_EQ(7U, ntuple->GetDescriptor().GetNFields());
   EXPECT_EQ(0U, GetFirstEntry(ntuple->GetModel().GetConstField("pt")));
   EXPECT_EQ(3U, GetFirstEntry(ntuple->GetModel().GetConstField("vec")));
   EXPECT_EQ(3U, GetFirstEntry(ntuple->GetModel().GetConstField("f")));
   EXPECT_EQ(3U, GetFirstEntry(ntuple->GetModel().GetConstField("u8")));
   EXPECT_EQ(3U, GetFirstEntry(ntuple->GetModel().GetConstField("str")));

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

TEST(RNTuple, ModelExtensionProjectSimple)
{
   FileRaii fileGuard("test_ntuple_modelext_project_simple.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt");
      model->AddProjectedField(std::make_unique<RField<float>>("aliasPt"), [](const std::string &) { return "pt"; });

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto modelUpdater = writer->CreateModelUpdater();
      modelUpdater->BeginUpdate();
      auto fieldE = modelUpdater->MakeField<float>("E");
      modelUpdater->CommitUpdate();

      *fieldPt = 42.0;
      *fieldE = 1.0;
      writer->Fill();
      *fieldPt = 137.0;
      *fieldE = 2.0;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(2U, reader->GetNEntries());
   EXPECT_EQ(4U, reader->GetDescriptor().GetNFields());

   auto pt = reader->GetView<float>("pt");
   auto aliasPt = reader->GetView<float>("aliasPt");
   auto E = reader->GetView<float>("E");
   EXPECT_FLOAT_EQ(42.0, pt(0));
   EXPECT_FLOAT_EQ(137.0, pt(1));
   EXPECT_FLOAT_EQ(42.0, aliasPt(0));
   EXPECT_FLOAT_EQ(137.0, aliasPt(1));
   EXPECT_FLOAT_EQ(1.0, E(0));
   EXPECT_FLOAT_EQ(2.0, E(1));
}

TEST(RNTuple, ModelExtensionProjectComplex)
{
   FileRaii fileGuard("test_ntuple_modelext_project_complex.root");
   std::vector<std::uint32_t> refVec{0x00, 0xff, 0x55, 0xaa};
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt");
      model->AddProjectedField(std::make_unique<RField<float>>("aliasPt"), [](const std::string &) { return "pt"; });
      model->AddProjectedField(std::make_unique<RField<float>>("al2Pt"), [](const std::string &) { return "pt"; });
      auto fieldE = model->MakeField<float>("E");
      model->AddProjectedField(std::make_unique<RField<float>>("aliasE"), [](const std::string &) { return "E"; });

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      auto modelUpdater = ntuple->CreateModelUpdater();
      modelUpdater->BeginUpdate();
      auto fieldVec = modelUpdater->MakeField<std::vector<std::uint32_t>>("vec");
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

      *fieldPt = 42.0;
      *fieldE = 1.0;
      ntuple->Fill();
      *fieldPt = 12.0;
      *fieldE = 2.0;
      *fieldVec = refVec;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(2U, ntuple->GetNEntries());
   EXPECT_EQ(10U, ntuple->GetDescriptor().GetNFields());

   auto pt = ntuple->GetView<float>("pt");
   auto aliasPt = ntuple->GetView<float>("aliasPt");
   auto al2Pt = ntuple->GetView<float>("al2Pt");
   auto aliasVec = ntuple->GetView<std::vector<std::uint32_t>>("aliasVec");
   auto E = ntuple->GetView<float>("E");
   auto aliasE = ntuple->GetView<float>("aliasE");
   EXPECT_FLOAT_EQ(42.0, pt(0));
   EXPECT_FLOAT_EQ(12.0, pt(1));
   EXPECT_FLOAT_EQ(42.0, aliasPt(0));
   EXPECT_FLOAT_EQ(12.0, aliasPt(1));
   EXPECT_FLOAT_EQ(42.0, al2Pt(0));
   EXPECT_FLOAT_EQ(12.0, al2Pt(1));
   EXPECT_FLOAT_EQ(1.0, E(0));
   EXPECT_FLOAT_EQ(2.0, E(1));
   EXPECT_FLOAT_EQ(1.0, aliasE(0));
   EXPECT_FLOAT_EQ(2.0, aliasE(1));
   EXPECT_EQ(std::vector<std::uint32_t>{}, aliasVec(0));
   EXPECT_EQ(refVec, aliasVec(1));
}

TEST(RNTuple, ModelExtensionSubFields)
{
   FileRaii fileGuard("test_ntuple_modelext_cluster_boundaries.root");
   CustomStruct refStruct{42.0, {1., 2., 3.}, {{4., 5.}, {}, {6.}}, "foo"};
   {
      auto model = RNTupleModel::Create();

      *model->MakeField<CustomStruct>("structFld") = refStruct;

      RNTupleWriteOptions opts;
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());

      for (unsigned i = 0; i < 2; ++i) {
         for (unsigned j = 0; j < 3; ++j) {
            ntuple->Fill();
         }
         ntuple->CommitCluster();
      }

      ntuple->Fill();

      auto modelUpdater = ntuple->CreateModelUpdater();

      modelUpdater->BeginUpdate();
      *modelUpdater->MakeField<CustomStruct>("extStructFld") = refStruct;
      modelUpdater->CommitUpdate();

      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto &desc = ntuple->GetDescriptor();

   EXPECT_EQ(3, desc.GetNClusters());
   EXPECT_EQ(8, desc.GetNEntries());

   auto structFldView = ntuple->GetView<CustomStruct>("structFld");
   auto extStructFldView = ntuple->GetView<CustomStruct>("extStructFld");

   EXPECT_EQ(structFldView(7), extStructFldView(7));

   // Check that the column ranges for model-extended subfields are properly constructed by iterating over their view.
   // For improper column ranges, the global field range would go until the value of kInvalidClusterIndex and result in
   // an out-of-bounds error.
   auto vecStructElemView = ntuple->GetView<float>("structFld.v2._0._0");
   auto extVecStructElemView = ntuple->GetView<float>("extStructFld.v2._0._0");
   for (const auto i : extVecStructElemView.GetFieldRange()) {
      EXPECT_EQ(vecStructElemView(i), extVecStructElemView(i));
   }
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
      auto wrEnergy = modelUpdater->MakeField<double>("energy");
      auto wrIndices = modelUpdater->MakeField<std::vector<std::uint32_t>>("indices");
      modelUpdater->CommitUpdate();

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
            wrTimes->at(n) = 1 + rnd.Rndm() * 1000. - 500.;
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
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
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

      // Force the serialization of a page list which will not know about the extended columns coming later.
      // `RClusterDescriptorBuilder::AddExtendedColumnRanges()` should thus make up page ranges for the missing columns
      ntuple->CommitCluster(true /* commitClusterGroup */);

      modelUpdater->BeginUpdate();
      auto dblVec = modelUpdater->MakeField<std::vector<double>>("dblVec");
      auto doubleAoA = modelUpdater->MakeField<doubleAoA_t>("doubleAoA");
      modelUpdater->CommitUpdate();
      for (unsigned int i = 10000; i < 20000; ++i) {
         updateInitialFields(i, *u32, *dbl);
         updateLateFields1(i, *dblVec, *doubleAoA);
         ntuple->Fill();
      }

      // A `R(Column|Page)Range` should also be generated in the previous cluster for the columns associated to the
      // `std::variant` field below
      ntuple->CommitCluster();

      modelUpdater->BeginUpdate();
      auto var = modelUpdater->MakeField<std::variant<std::uint64_t, double>>("var");
      modelUpdater->CommitUpdate();
      for (unsigned int i = 20000; i < 30000; ++i) {
         updateInitialFields(i, *u32, *dbl);
         updateLateFields1(i, *dblVec, *doubleAoA);
         updateLateFields2(i, *var);
         ntuple->Fill();
      }

      modelUpdater->BeginUpdate();
      auto b = modelUpdater->MakeField<bool>("b");
      auto noColumns = modelUpdater->MakeField<EmptyStruct>("noColumns");
      modelUpdater->CommitUpdate();
      for (unsigned int i = 30000; i < 40000; ++i) {
         updateInitialFields(i, *u32, *dbl);
         updateLateFields1(i, *dblVec, *doubleAoA);
         updateLateFields2(i, *var);
         *b = !(*b);
         chksumWrite += static_cast<double>(*b);
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

TEST(RNTuple, ModelExtensionNullableField)
{
   FileRaii fileGuard("test_ntuple_modelext_nullablefield.root");

   {
      auto writer = RNTupleWriter::Recreate(RNTupleModel::Create(), "ntpl", fileGuard.GetPath());
      writer->Fill();

      auto modelUpdater = writer->CreateModelUpdater();
      modelUpdater->BeginUpdate();
      auto fieldA = std::make_unique<RField<std::optional<float>>>("A");
      modelUpdater->AddField(std::move(fieldA));
      modelUpdater->CommitUpdate();

      auto ptrA = writer->GetModel().GetDefaultEntry().GetPtr<std::optional<float>>("A");

      *ptrA = 1.0;
      writer->Fill();

      ptrA->reset();
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(3u, reader->GetNEntries());
   auto viewA = reader->GetView<std::optional<float>>("A");

   EXPECT_FALSE(viewA(0).has_value());
   EXPECT_FLOAT_EQ(1.0, viewA(1).value());
   EXPECT_FALSE(viewA(2).has_value());
}
