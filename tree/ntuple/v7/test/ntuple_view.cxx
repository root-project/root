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
      RNTupleWriter ntuple(std::move(model),
         std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions()));
      ntuple.Fill();
      ntuple.CommitCluster();
      fieldJets->clear();
      ntuple.Fill();
   }

   RNTupleReader ntuple(std::make_unique<RPageSourceFile>("myNTuple", fileGuard.GetPath(), RNTupleReadOptions()));
   auto viewPt = ntuple.GetView<float>("pt");
   int n = 0;
   for (auto i : ntuple.GetEntryRange()) {
      EXPECT_EQ(42.0, viewPt(i));
      n++;
   }
   EXPECT_EQ(2, n);

   auto viewJets = ntuple.GetView<std::vector<std::int32_t>>("jets");
   n = 0;
   for (auto i : ntuple.GetEntryRange()) {
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

   auto viewJetElements = ntuple.GetView<std::int32_t>("jets.std::int32_t");
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
      opt.SetNElementsPerPage(eltsPerPage);
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

TEST(RNTuple, BulkViewCollection)
{
   FileRaii fileGuard("test_ntuple_bulk_view_collection.root");

   auto model = RNTupleModel::Create();
   auto fieldVec = model->MakeField<std::vector<double>>({"vec", "some data"});
   auto eltsPerPage = 10'000;
   {
      RNTupleWriteOptions opt;
      opt.SetNElementsPerPage(eltsPerPage);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "myNTuple",
         fileGuard.GetPath(), opt);
      for (int i = 0; i < 100'000; i++) {
         *fieldVec = std::vector<double>(i % 5, 100);
         ntuple->Fill();
      }
   }
   auto ntuple = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   auto viewVecOffsets = ntuple->GetView<ClusterSize_t>("vec");
   auto viewVecData = ntuple->GetView<double>("vec.double");

   NTupleSize_t nPageItems = 0;
   const ClusterSize_t *offsets_buf = viewVecOffsets.MapV(0, nPageItems);
   ASSERT_EQ(eltsPerPage, nPageItems);
   std::unique_ptr<ClusterSize_t[]> offsets = std::make_unique<ClusterSize_t[]>(nPageItems + 1);
   auto raw_offsets = offsets.get();
   // offsets implicitly start at zero, RNTuple does not store that information
   raw_offsets[0] = 0;
   memcpy(raw_offsets + 1, offsets_buf, sizeof(ClusterSize_t) * nPageItems);
   for (NTupleSize_t i = 1; i < nPageItems + 1; i++) {
      ASSERT_EQ((i - 1) % 5, raw_offsets[i] - raw_offsets[i-1]) << i;
   }
   const double *buf = viewVecData.MapV(0, nPageItems);
   ASSERT_EQ(eltsPerPage, nPageItems);
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
   auto viewTracks = ntuple->GetViewCollection("tracks");
   auto viewTrackEnergy = viewTracks.GetView<float>("energy");
   auto viewHits = viewTracks.GetViewCollection("hits");
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
   auto viewMuon = ntuple->GetViewCollection("Muon");
   try {
      auto badView = ntuple->GetView<float>("pT");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'pT' in RNTuple 'myNTuple'"));
   }
   try {
      auto badView = ntuple->GetViewCollection("Moun");
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
      auto badView = viewMuon.GetViewCollection("badC");
      FAIL() << "missing field names should throw";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("no field named 'badC' in RNTuple 'myNTuple'"));
   }
}
