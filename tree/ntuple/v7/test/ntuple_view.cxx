#include "ntuple_test.hxx"

TEST(RNTuple, View)
{
   FileRaii fileGuard("test_ntuple_view.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);
   auto fieldTag = model->MakeField<std::string>("tag", "xyz");
   auto fieldJets = model->MakeField<std::vector<std::int32_t>>("jets");
   fieldJets->push_back(1);
   fieldJets->push_back(2);
   fieldJets->push_back(3);

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      fieldJets->clear();
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewPt = reader->GetView<float>("pt");
   int n = 0;
   for (auto i : reader->GetEntryRange()) {
      EXPECT_EQ(42.0, viewPt(i));
      n++;
   }
   EXPECT_EQ(2, n);

   auto viewJets = reader->GetView<std::vector<std::int32_t>>("jets");
   n = 0;
   for (auto i : reader->GetEntryRange()) {
      if (i == 0) {
         EXPECT_EQ(3U, viewJets(i).size());
         EXPECT_EQ(1, viewJets(i)[0]);
         EXPECT_EQ(2, viewJets(i)[1]);
         EXPECT_EQ(3, viewJets(i)[2]);
      } else {
         EXPECT_EQ(0U, viewJets(i).size());
      }
      n++;
   }
   EXPECT_EQ(2, n);

   auto viewJetElements = reader->GetView<std::int32_t>("jets._0");
   n = 0;
   for (auto i : viewJetElements.GetFieldRange()) {
      n++;
      EXPECT_EQ(n, viewJetElements(i));
   }
   EXPECT_EQ(3, n);
}

TEST(RNTuple, BulkView)
{
   FileRaii fileGuard("test_ntuple_bulk_view.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);
   auto eltsPerPage = 10'000;
   {
      RNTupleWriteOptions opt;
      opt.SetApproxUnzippedPageSize(eltsPerPage * sizeof(float));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple",
         fileGuard.GetPath(), opt);
      for (int i = 0; i < 100'000; i++) {
         ntuple->Fill();
      }
   }
   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewPt = ntuple->GetView<float>("pt");

   NTupleSize_t nPageItems = 0;
   const float *buf = viewPt.MapV(0, nPageItems);
   ASSERT_EQ(eltsPerPage, nPageItems);
   for (NTupleSize_t i = 0; i < nPageItems; i++) {
      ASSERT_EQ(42.0f, buf[i]) << i;
   }
   // second last element
   buf = viewPt.MapV(eltsPerPage - 2, nPageItems);
   ASSERT_EQ(2, nPageItems);
   for (NTupleSize_t i = 0; i < nPageItems; i++) {
      ASSERT_EQ(42.0f, buf[i]) << i;
   }
   // last element
   buf = viewPt.MapV(eltsPerPage - 1, nPageItems);
   ASSERT_EQ(1, nPageItems);
   for (NTupleSize_t i = 0; i < nPageItems; i++) {
      ASSERT_EQ(42.0f, buf[i]) << i;
   }
}

TEST(RNTuple, BulkCollectionView)
{
   FileRaii fileGuard("test_ntuple_bulk_view_collection.root");

   auto model = RNTupleModel::Create();
   auto fieldVec = model->MakeField<std::vector<double>>({"vec", "some data"});
   auto pageSize = 80 * 1024;
   {
      RNTupleWriteOptions opt;
      opt.SetApproxUnzippedPageSize(pageSize);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple",
         fileGuard.GetPath(), opt);
      for (int i = 0; i < 100'000; i++) {
         *fieldVec = std::vector<double>(i % 5, 100);
         ntuple->Fill();
      }
   }
   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewVecOffsets = ntuple->GetView<ClusterSize_t>("vec");
   auto viewVecData = ntuple->GetView<double>("vec._0");

   NTupleSize_t nPageItems = 0;
   const ClusterSize_t *offsets_buf = viewVecOffsets.MapV(0, nPageItems);
   ASSERT_EQ(pageSize / sizeof(ClusterSize_t), nPageItems);
   std::unique_ptr<ClusterSize_t[]> offsets = std::make_unique<ClusterSize_t[]>(nPageItems + 1);
   auto raw_offsets = offsets.get();
   // offsets implicitly start at zero, RNTuple does not store that information
   raw_offsets[0] = 0;
   memcpy(raw_offsets + 1, offsets_buf, sizeof(ClusterSize_t) * nPageItems);
   for (NTupleSize_t i = 1; i < nPageItems + 1; i++) {
      ASSERT_EQ((i - 1) % 5, raw_offsets[i] - raw_offsets[i-1]) << i;
   }
   const double *buf = viewVecData.MapV(0, nPageItems);
   ASSERT_EQ(pageSize / sizeof(double), nPageItems);
   for (NTupleSize_t i = 0; i < nPageItems; i++) {
      ASSERT_EQ(100.0f, buf[i]) << i;
   }
}

TEST(RNTuple, Composable)
{
   FileRaii fileGuard("test_ntuple_composable.root");

   auto eventModel = RNTupleModel::Create();
   auto fldPt = eventModel->MakeField<float>("pt", 0.0);

   auto hitModel = RNTupleModel::Create();
   auto fldHitX = hitModel->MakeField<float>("x", 0.0);
   auto fldHitY = hitModel->MakeField<float>("y", 0.0);

   auto trackModel = RNTupleModel::Create();
   auto fldTrackEnergy = trackModel->MakeField<float>("energy", 0.0);

   auto fldHits = trackModel->MakeCollection("hits", std::move(hitModel));
   auto fldTracks = eventModel->MakeCollection("tracks", std::move(trackModel));

   {
      auto ntuple = RNTupleWriter::Recreate(std::move(eventModel), "myNTuple", fileGuard.GetPath());

      for (unsigned i = 0; i < 8; ++i) {
         for (unsigned t = 0; t < 3; ++t) {
            for (unsigned h = 0; h < 2; ++h) {
               *fldHitX = 4.0;
               *fldHitY = 8.0;
               fldHits->Fill();
            }
            *fldTrackEnergy = i * t;
            fldTracks->Fill();
         }
         *fldPt = float(i);
         ntuple->Fill();
         if (i == 2)
            ntuple->CommitCluster();
      }
   }

   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewPt = ntuple->GetView<float>("pt");
   auto viewTracks = ntuple->GetCollectionView("tracks");
   auto viewTrackEnergy = viewTracks.GetView<float>("energy");
   auto viewHits = viewTracks.GetCollectionView("hits");
   auto viewHitX = viewHits.GetView<float>("x");
   auto viewHitY = viewHits.GetView<float>("y");

   int nEv = 0;
   for (auto e : ntuple->GetEntryRange()) {
      EXPECT_EQ(float(nEv), viewPt(e));
      EXPECT_EQ(3U, viewTracks(e));

      int nTr = 0;
      for (auto t : viewTracks.GetCollectionRange(e)) {
         EXPECT_EQ(nEv * nTr, viewTrackEnergy(t));

         EXPECT_EQ(2.0, viewHits(t));
         for (auto h : viewHits.GetCollectionRange(t)) {
            EXPECT_EQ(4.0, viewHitX(h));
            EXPECT_EQ(8.0, viewHitY(h));
         }
         nTr++;
      }
      EXPECT_EQ(3, nTr);

      nEv++;
   }
   EXPECT_EQ(8, nEv);
}

TEST(RNTuple, VoidView)
{
   FileRaii fileGuard("test_ntuple_voidview.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(1u, reader->GetNEntries());
   auto viewPt = reader->GetView<void>("pt");
   viewPt(0);
   EXPECT_FLOAT_EQ(42.0, viewPt.GetValue().GetRef<float>());
   EXPECT_STREQ("pt", viewPt.GetField().GetFieldName().c_str());
}

TEST(RNTuple, MissingViewNames)
{
   FileRaii fileGuard("test_ntuple_missing_view_names.root");
   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);
   auto muonModel = RNTupleModel::Create();
   auto muonPt = muonModel->MakeField<float>("pt", 42.0);
   auto muon = model->MakeCollection("Muon", std::move(muonModel));
   {
      RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
   }
   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewPt = ntuple->GetView<float>("pt");
   auto viewMuon = ntuple->GetCollectionView("Muon");
   try {
      auto badView = ntuple->GetView<float>("pT");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'pT' in RNTuple 'myNTuple'"));
   }
   try {
      auto badView = ntuple->GetCollectionView("Moun");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'Moun' in RNTuple 'myNTuple'"));
   }
   try {
      auto badView = viewMuon.GetView<float>("badField");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'badField' in RNTuple 'myNTuple'"));
   }
   try {
      auto badView = viewMuon.GetCollectionView("badC");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'badC' in RNTuple 'myNTuple'"));
   }
}

TEST(RNTuple, ViewWithExternalAddress)
{
   FileRaii fileGuard("test_ntuple_viewexternal.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt", 42.0);

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   // Typed shared_ptr
   auto data_1 = std::make_shared<float>();
   auto view_1 = reader->GetView("pt", data_1);
   view_1(0);
   EXPECT_FLOAT_EQ(42.0, *data_1);

   // Void shared_ptr
   std::shared_ptr<void> data_2{new float()};
   auto view_2 = reader->GetView("pt", data_2);
   view_2(0);
   EXPECT_FLOAT_EQ(42.0, *static_cast<float *>(data_2.get()));
}

TEST(RNTuple, BindEmplaceTyped)
{
   FileRaii fileGuard("test_ntuple_bindvalueemplacetyped.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt");

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *fieldPt = 11.f;
      writer->Fill();
      *fieldPt = 22.f;
      writer->Fill();
      *fieldPt = 33.f;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   // bind to shared_ptr
   auto value1 = std::make_shared<float>();
   auto view = reader->GetView<float>("pt", nullptr);
   view.Bind(value1);
   view(0);
   EXPECT_FLOAT_EQ(11.f, *value1);

   // bind to raw pointer
   float value2;
   view.BindRawPtr(&value2);
   view(1);
   EXPECT_FLOAT_EQ(22.f, value2);

   // emplace new value
   view.EmplaceNew();
   EXPECT_FLOAT_EQ(33.f, view(2));
   EXPECT_FLOAT_EQ(22.f, value2); // The previous value was not modified
}

TEST(RNTuple, BindEmplaceVoid)
{
   FileRaii fileGuard("test_ntuple_bindvalueemplacevoid.root");

   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt");

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *fieldPt = 11.f;
      writer->Fill();
      *fieldPt = 22.f;
      writer->Fill();
      *fieldPt = 33.f;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   // bind to shared_ptr
   std::shared_ptr<void> value1{new float()};
   auto view = reader->GetView<void>("pt", nullptr);
   view.Bind(value1);
   view(0);
   EXPECT_FLOAT_EQ(11.f, *reinterpret_cast<float *>(value1.get()));

   // bind to raw pointer
   float value2;
   view.BindRawPtr(&value2);
   view(1);
   EXPECT_FLOAT_EQ(22.f, value2);

   // emplace new value
   view.EmplaceNew();
   view(2);
   EXPECT_FLOAT_EQ(33.f, view.GetValue().GetRef<float>());
   EXPECT_FLOAT_EQ(22.f, value2); // The previous value was not modified
}
