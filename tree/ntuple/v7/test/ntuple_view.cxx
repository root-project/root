#include "ntuple_test.hxx"

#include <iterator>
#if __has_include(<ranges>)
#include <ranges>
#endif

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

   auto viewJetElements = ntuple.GetView<std::int32_t>("jets._0");
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

TEST(RNTuple, BulkViewCollection)
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

TEST(RNTuple, RNTupleGlobalRange)
{
#ifdef __cpp_lib_concepts
   static_require(std::random_access_iterator<RNTupleGlobalRange::Iterator>);
#endif
   const RNTupleGlobalRange r(0, 10);

   // semi-regularity of iterator
   {
#ifdef __cpp_lib_concepts
      static_assert(std::semiregular<RNTupleGlobalRange::Iterator>);
#endif

      RNTupleGlobalRange::Iterator it; // default ctor
      it = std::begin(r);              // copy assign
      auto it2 = it;                   // copy ctor
      EXPECT_EQ(it, it2);
   }

   // inc, dev, deref, subscript
   {
      auto it = std::begin(r);
      EXPECT_EQ(*it, 0);
      EXPECT_EQ(*(it++), 0);
      EXPECT_EQ(*it, 1);
      EXPECT_EQ(*(++it), 2);
      EXPECT_EQ(*it, 2);
      EXPECT_EQ(*(it--), 2);
      EXPECT_EQ(*it, 1);
      EXPECT_EQ(*(--it), 0);

      EXPECT_EQ(it[0], 0);
      EXPECT_EQ(it[5], 5);
      ++++it;
      EXPECT_EQ(it[-1], 1);
   }

   // arithmetic
   {
      auto it = std::begin(r);

      it += 4;
      EXPECT_EQ(*it, 4);
      it += -2;
      EXPECT_EQ(*it, 2);
      it -= 1;
      EXPECT_EQ(*it, 1);
      it -= -4;
      EXPECT_EQ(*it, 5);

      EXPECT_EQ(*(it + 2), 7);
      EXPECT_EQ(*(it + (-2)), 3);
      EXPECT_EQ(*(2 + it), 7);
      EXPECT_EQ(*((-2) + it), 3);
      EXPECT_EQ(*(it - 2), 3);
      EXPECT_EQ(*(it - (-2)), 7);

      auto it2 = r.begin() + 2;
      auto it3 = r.begin() + 7;
      EXPECT_EQ(it3 - it2, 5);
      EXPECT_EQ(it2 - it3, -5);
   }

   // relational
   {
      EXPECT_EQ(r.begin(), std::begin(r));
      EXPECT_EQ(r.end(), std::end(r));

      auto it2 = r.begin() + 2;
      auto it3 = r.begin() + 3;

      EXPECT_TRUE(it2 == it2);
      EXPECT_TRUE(it2 != it3);
      EXPECT_TRUE(it2 < it3);
      EXPECT_TRUE(it2 <= it2);
      EXPECT_TRUE(it3 > it2);
      EXPECT_TRUE(it2 >= it2);

      EXPECT_FALSE(it2 == it3);
      EXPECT_FALSE(it2 != it2);
      EXPECT_FALSE(it2 < it2);
      EXPECT_FALSE(it3 <= it2);
      EXPECT_FALSE(it2 > it3);
      EXPECT_FALSE(it2 >= it3);
   }

   // reachability of end
   {
      int i = 0;
      for (auto it = std::begin(r); it != std::end(r); it++)
         i++;
      EXPECT_EQ(i, 10);
   }

   // range-for
   {
      std::vector<NTupleSize_t> values;
      for (auto i : RNTupleGlobalRange{0, 3})
         values.push_back(i);
      EXPECT_THAT(values, testing::ElementsAre(0, 1, 2));
   }

#ifdef __cpp_lib_ranges
   static_require(std::random_access_range<ROOT::Experimental::RNTupleGlobalRange>);

   {
      std::vector<NTupleSize_t> values;
      for (auto i : ROOT::Experimental::RNTupleGlobalRange{0, 100} | std::views::take(3))
         values.push_back(i);
      EXPECT_THAT(values, testing::ElementsAre(0, 1, 2));
   }
#endif
}

TEST(RNTuple, RNTupleClusterRange)
{
   // most stuff is already tested via RNTupleGlobalRange, which uses the same infrastructure

#ifdef __cpp_lib_concepts
   static_require(std::random_access_iterator<RNTupleClusterRange::Iterator>);
#endif

   std::vector<RClusterIndex> values;
   for (auto i : RNTupleClusterRange{0xDEADBEEF, 0, 3})
      values.push_back(i);
   EXPECT_THAT(values, testing::ElementsAre(RClusterIndex{0xDEADBEEF, 0}, RClusterIndex{0xDEADBEEF, 1}, RClusterIndex{0xDEADBEEF, 2}));

#ifdef __cpp_lib_ranges
      static_require(std::random_access_range<ROOT::Experimental::RNTupleClusterRange>);
#endif
}
