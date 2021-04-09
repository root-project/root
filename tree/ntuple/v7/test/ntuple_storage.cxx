#include "ntuple_test.hxx"

TEST(RNTuple, Basics)
{
   FileRaii fileGuard("test_ntuple_barefile.ntuple");

   auto model = RNTupleModel::Create();
   auto wrPt = model->MakeField<float>("pt", 42.0);

   {
      RNTupleWriteOptions options;
      options.SetContainerFormat(ENTupleContainerFormat::kBare);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath(), options);
      ntuple->Fill();
      ntuple->CommitCluster();
      *wrPt = 24.0;
      ntuple->Fill();
      *wrPt = 12.0;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(3U, ntuple->GetNEntries());
   auto rdPt = ntuple->GetModel()->GetDefaultEntry()->Get<float>("pt");

   ntuple->LoadEntry(0);
   EXPECT_EQ(42.0, *rdPt);
   ntuple->LoadEntry(1);
   EXPECT_EQ(24.0, *rdPt);
   ntuple->LoadEntry(2);
   EXPECT_EQ(12.0, *rdPt);
}

TEST(RNTuple, Extended)
{
   FileRaii fileGuard("test_ntuple_barefile_ext.ntuple");

   auto model = RNTupleModel::Create();
   auto wrVector = model->MakeField<std::vector<double>>("vector");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      RNTupleWriteOptions options;
      options.SetContainerFormat(ENTupleContainerFormat::kBare);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath(), options);
      constexpr unsigned int nEvents = 32000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         auto nVec = 1 + floor(rnd.Rndm() * 1000.);
         wrVector->resize(nVec);
         for (unsigned int n = 0; n < nVec; ++n) {
            auto val = 1 + rnd.Rndm()*1000. - 500.;
            (*wrVector)[n] = val;
            chksumWrite += val;
         }
         ntuple->Fill();
         if (i % 1000 == 0)
            ntuple->CommitCluster();
      }
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   auto rdVector = ntuple->GetModel()->GetDefaultEntry()->Get<std::vector<double>>("vector");

   double chksumRead = 0.0;
   for (auto entryId : *ntuple) {
      ntuple->LoadEntry(entryId);
      for (auto v : *rdVector)
         chksumRead += v;
   }
   EXPECT_EQ(chksumRead, chksumWrite);
}

TEST(RPageSinkBuf, Basics)
{
   struct TestModel {
      std::unique_ptr<RNTupleModel> fModel;
      std::shared_ptr<float> fFloatField;
      std::shared_ptr<std::vector<CustomStruct>> fFieldKlassVec;
      TestModel() {
         fModel = RNTupleModel::Create();
         fFloatField = fModel->MakeField<float>("pt");
         fFieldKlassVec = fModel->MakeField<std::vector<CustomStruct>>("klassVec");
      }
   };

   FileRaii fileGuardBuf("test_ntuple_sinkbuf_basics_buf.root");
   FileRaii fileGuard("test_ntuple_sinkbuf_basics.root");
   {
      TestModel bufModel;
      // PageSinkBuf wraps a concrete page source
      auto ntupleBuf = std::make_unique<RNTupleWriter>(std::move(bufModel.fModel),
         std::make_unique<RPageSinkBuf>(std::make_unique<RPageSinkFile>(
            "buf", fileGuardBuf.GetPath(), RNTupleWriteOptions()
      )));

      TestModel unbufModel;
      auto ntuple = std::make_unique<RNTupleWriter>(std::move(unbufModel.fModel),
         std::make_unique<RPageSinkFile>("unbuf", fileGuard.GetPath(), RNTupleWriteOptions()
      ));

      for (int i = 0; i < 20000; i++) {
         *bufModel.fFloatField = static_cast<float>(i);
         *unbufModel.fFloatField = static_cast<float>(i);
         CustomStruct klass;
         klass.a = 42.0;
         klass.v1.emplace_back(static_cast<float>(i));
         klass.v2.emplace_back(std::vector<float>(3, static_cast<float>(i)));
         klass.s = "hi" + std::to_string(i);
         *bufModel.fFieldKlassVec = std::vector<CustomStruct>{klass};
         *unbufModel.fFieldKlassVec = std::vector<CustomStruct>{klass};

         ntupleBuf->Fill();
         ntuple->Fill();

         if (i % 15000 == 0) {
            ntupleBuf->CommitCluster();
            ntuple->CommitCluster();
         }
      }
   }

   auto ntupleBuf = RNTupleReader::Open("buf", fileGuardBuf.GetPath());
   auto ntuple = RNTupleReader::Open("unbuf", fileGuard.GetPath());
   EXPECT_EQ(ntuple->GetNEntries(), ntupleBuf->GetNEntries());

   auto viewPtBuf = ntupleBuf->GetView<float>("pt");
   auto viewKlassVecBuf = ntupleBuf->GetView<std::vector<CustomStruct>>("klassVec");
   auto viewPt = ntuple->GetView<float>("pt");
   auto viewKlassVec = ntuple->GetView<std::vector<CustomStruct>>("klassVec");
   for (auto i : ntupleBuf->GetEntryRange()) {
      EXPECT_EQ(static_cast<float>(i), viewPtBuf(i));
      EXPECT_EQ(viewPt(i), viewPtBuf(i));
      EXPECT_EQ(viewKlassVec(i).at(0).v1, viewKlassVecBuf(i).at(0).v1);
      EXPECT_EQ(viewKlassVec(i).at(0).v2, viewKlassVecBuf(i).at(0).v2);
      EXPECT_EQ(viewKlassVec(i).at(0).s, viewKlassVecBuf(i).at(0).s);
   }
}
