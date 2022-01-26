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

TEST(RNTuple, InvalidWriteOptions) {
   RNTupleWriteOptions options;
   try {
      options.SetApproxUnzippedPageSize(0);
      FAIL() << "should not allow zero-sized page";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("page size"));
   }
   options.SetApproxUnzippedPageSize(10);
   options.SetApproxZippedClusterSize(50);
   try {
      options.SetMaxUnzippedClusterSize(40);
      FAIL() << "should not allow undersized cluster";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("must not be larger than"));
   }
   options.SetApproxZippedClusterSize(5);
   try {
      options.SetMaxUnzippedClusterSize(7);
      FAIL() << "should not allow undersized cluster";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("must not be larger than"));
   }

   FileRaii fileGuard("test_ntuple_invalid_write_options.root");
   auto model = RNTupleModel::Create();
   model->MakeField<std::int16_t>("x");
   options.SetApproxUnzippedPageSize(3);
   auto m2 = model->Clone();
   try {
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      FAIL() << "should not allow undersized pages";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("page size too small"));
   }
   options.SetApproxUnzippedPageSize(4);
   try {
      auto ntuple = RNTupleWriter::Recreate(std::move(m2), "ntpl", fileGuard.GetPath(), options);
   } catch (const RException &err) {
      FAIL() << "pages size should be just large enough for 2 elements";
   }
}

TEST(RNTuple, PageFilling) {
   FileRaii fileGuard("test_ntuple_page_filling.root");

   auto model = RNTupleModel::Create();
   auto fldX = model->MakeField<std::int16_t>("x");

   RNTupleWriteOptions options;
   // Exercises the page swapping algorithm with pages just big enough to hold 2 elements
   options.SetApproxUnzippedPageSize(4);

   {
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      for (std::int16_t i = 0; i < 8; ++i) {
         *fldX = i;
         ntuple->Fill();
         // Flush half-full pages
         if (i == 2)
            ntuple->CommitCluster();
         // Flush just after automatic flush
         if (i == 4)
            ntuple->CommitCluster();
      }
   }

   auto ntuple = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewX = ntuple->GetView<std::int16_t>("x");
   ASSERT_EQ(8u, ntuple->GetNEntries());
   for (std::int16_t i = 0; i < 8; ++i)
      EXPECT_EQ(i, viewX(i));

   const auto desc = ntuple->GetDescriptor();
   EXPECT_EQ(3u, desc->GetNClusters());
   const auto &cd1 = desc->GetClusterDescriptor(desc->FindClusterId(0, 0));
   const auto &pr1 = cd1.GetPageRange(0);
   ASSERT_EQ(2u, pr1.fPageInfos.size());
   EXPECT_EQ(2u, pr1.fPageInfos[0].fNElements);
   EXPECT_EQ(1u, pr1.fPageInfos[1].fNElements);
   const auto &cd2 = desc->GetClusterDescriptor(desc->FindNextClusterId(cd1.GetId()));
   const auto &pr2 = cd2.GetPageRange(0);
   ASSERT_EQ(1u, pr2.fPageInfos.size());
   EXPECT_EQ(2u, pr2.fPageInfos[0].fNElements);
   const auto &cd3 = desc->GetClusterDescriptor(desc->FindNextClusterId(cd2.GetId()));
   const auto &pr3 = cd3.GetPageRange(0);
   ASSERT_EQ(2u, pr3.fPageInfos.size());
   EXPECT_EQ(2u, pr3.fPageInfos[0].fNElements);
   EXPECT_EQ(1u, pr3.fPageInfos[1].fNElements);
}

TEST(RNTuple, PageFillingString) {
   FileRaii fileGuard("test_ntuple_page_filling_string.root");

   auto model = RNTupleModel::Create();
   // A string column exercises RColumn::AppendV
   auto fldX = model->MakeField<std::string>("x");

   RNTupleWriteOptions options;
   options.SetApproxUnzippedPageSize(16);

   {
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      // 1 page: 17 characters
      *fldX = "01234567890123456";
      ntuple->Fill();
      ntuple->CommitCluster();
      // 1 pages: 16 characters
      *fldX = "0123456789012345";
      ntuple->Fill();
      ntuple->CommitCluster();
      // 0 pages
      *fldX = "";
      ntuple->Fill();
      ntuple->CommitCluster();
      // 2 pages: 16 and 8 characters
      *fldX = "012345678901234567890123";
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewX = ntuple->GetView<std::string>("x");
   ASSERT_EQ(4u, ntuple->GetNEntries());
   EXPECT_EQ("01234567890123456",        viewX(0));
   EXPECT_EQ("0123456789012345",         viewX(1));
   EXPECT_EQ("",                         viewX(2));
   EXPECT_EQ("012345678901234567890123", viewX(3));

   const auto desc = ntuple->GetDescriptor();
   EXPECT_EQ(4u, desc->GetNClusters());
   const auto &cd1 = desc->GetClusterDescriptor(desc->FindClusterId(1, 0));
   const auto &pr1 = cd1.GetPageRange(1);
   ASSERT_EQ(1u, pr1.fPageInfos.size());
   EXPECT_EQ(17u, pr1.fPageInfos[0].fNElements);
   const auto &cd2 = desc->GetClusterDescriptor(desc->FindNextClusterId(cd1.GetId()));
   const auto &pr2 = cd2.GetPageRange(1);
   ASSERT_EQ(1u, pr2.fPageInfos.size());
   EXPECT_EQ(16u, pr2.fPageInfos[0].fNElements);
   const auto &cd3 = desc->GetClusterDescriptor(desc->FindNextClusterId(cd2.GetId()));
   const auto &pr3 = cd3.GetPageRange(1);
   ASSERT_EQ(0u, pr3.fPageInfos.size());
   const auto &cd4 = desc->GetClusterDescriptor(desc->FindNextClusterId(cd3.GetId()));
   const auto &pr4 = cd4.GetPageRange(1);
   ASSERT_EQ(2u, pr4.fPageInfos.size());
   EXPECT_EQ(16u, pr4.fPageInfos[0].fNElements);
   EXPECT_EQ(8u, pr4.fPageInfos[1].fNElements);
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
      ntupleBuf->EnableMetrics();

      TestModel unbufModel;
      auto ntuple = std::make_unique<RNTupleWriter>(std::move(unbufModel.fModel),
         std::make_unique<RPageSinkFile>("unbuf", fileGuard.GetPath(), RNTupleWriteOptions()
      ));

      for (int i = 0; i < 40000; i++) {
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

         if (i && i % 30000 == 0) {
            ntupleBuf->CommitCluster();
            ntuple->CommitCluster();
            auto *parallel_zip = ntupleBuf->GetMetrics().GetCounter(
               "RNTupleWriter.RPageSinkBuf.ParallelZip");
            ASSERT_FALSE(parallel_zip == nullptr);
            EXPECT_EQ(0, parallel_zip->GetValueAsInt());
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

   std::vector<std::pair<DescriptorId_t, std::int64_t>> pagePositions;
   std::size_t num_columns = 10;
   const auto &cluster0 = ntupleBuf->GetDescriptor()->GetClusterDescriptor(0);
   for (std::size_t i = 0; i < num_columns; i++) {
      const auto &columnPages = cluster0.GetPageRange(i);
      for (const auto &page: columnPages.fPageInfos) {
         pagePositions.push_back(std::make_pair(i, page.fLocator.fPosition));
      }
   }

   auto sortedPages = pagePositions;
   std::sort(begin(sortedPages), end(sortedPages),
      [](const auto &a, const auto &b) { return a.second < b.second; });

   // For this test, ensure at least some columns have multiple pages
   ASSERT_TRUE(sortedPages.size() > num_columns);
   // Buffered sink cluster column pages are written out together
   for (std::size_t i = 0; i < pagePositions.size() - 1; i++) {
      // if the next page belongs to another column, skip the check
      if (pagePositions.at(i+1).first != pagePositions.at(i).first) {
         continue;
      }
      auto page = std::find(begin(sortedPages), end(sortedPages), pagePositions[i]);
      ASSERT_TRUE(page != sortedPages.end());
      auto next_page = page + 1;
      auto column = pagePositions[i].first;
      ASSERT_EQ(column, next_page->first);
   }
}

TEST(RPageSinkBuf, ParallelZip) {
   ROOT::EnableImplicitMT();

   FileRaii fileGuard("test_ntuple_sinkbuf_pzip.root");
   {
      auto model = RNTupleModel::Create();
      auto floatField = model->MakeField<float>("pt");
      auto fieldKlassVec = model->MakeField<std::vector<CustomStruct>>("klassVec");
      auto ntuple = std::make_unique<RNTupleWriter>(std::move(model),
         std::make_unique<RPageSinkBuf>(std::make_unique<RPageSinkFile>(
            "buf_pzip", fileGuard.GetPath(), RNTupleWriteOptions()
      )));
      ntuple->EnableMetrics();
      for (int i = 0; i < 20000; i++) {
         *floatField = static_cast<float>(i);
         CustomStruct klass;
         klass.a = 42.0;
         klass.v1.emplace_back(static_cast<float>(i));
         klass.v2.emplace_back(std::vector<float>(3, static_cast<float>(i)));
         klass.s = "hi" + std::to_string(i);
         *fieldKlassVec = std::vector<CustomStruct>{klass};
         ntuple->Fill();
         if (i && i % 15000 == 0) {
            ntuple->CommitCluster();
            auto *parallel_zip = ntuple->GetMetrics().GetCounter(
               "RNTupleWriter.RPageSinkBuf.ParallelZip");
            ASSERT_FALSE(parallel_zip == nullptr);
            EXPECT_EQ(1, parallel_zip->GetValueAsInt());
         }
      }
   }

   auto ntuple = RNTupleReader::Open("buf_pzip", fileGuard.GetPath());
   EXPECT_EQ(20000, ntuple->GetNEntries());

   auto viewPt = ntuple->GetView<float>("pt");
   auto viewKlassVec = ntuple->GetView<std::vector<CustomStruct>>("klassVec");
   for (auto i : ntuple->GetEntryRange()) {
      float fi = static_cast<float>(i);
      EXPECT_EQ(fi, viewPt(i));
      EXPECT_EQ(std::vector<float>{fi}, viewKlassVec(i).at(0).v1);
      EXPECT_EQ((std::vector<float>(3, fi)), viewKlassVec(i).at(0).v2.at(0));
      EXPECT_EQ("hi" + std::to_string(i), viewKlassVec(i).at(0).s);
   }
}

TEST(RPageSink, Empty)
{
   FileRaii fileGuard("test_ntuple_empty.ntuple");

   auto model = RNTupleModel::Create();
   auto wrPt = model->MakeField<float>("pt", 42.0);

   {
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(0U, ntuple->GetNEntries());
   EXPECT_EQ(0U, ntuple->GetDescriptor()->GetNClusterGroups());
   EXPECT_EQ(0U, ntuple->GetDescriptor()->GetNClusters());
}

TEST(RPageSink, MultipleClusterGroups)
{
   FileRaii fileGuard("test_ntuple_multi_cluster_groups.ntuple");

   auto model = RNTupleModel::Create();
   auto wrPt = model->MakeField<float>("pt", 42.0);

   {
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      ntuple->Fill();
      ntuple->CommitCluster();
      // This pattern should work: CommitCluster(false) followed by CommitCluster(true) and
      // be identical to a single call to CommitCluster(true)
      ntuple->CommitCluster(true);
      *wrPt = 24.0;
      ntuple->Fill();
      ntuple->CommitCluster();
      *wrPt = 12.0;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(2U, ntuple->GetDescriptor()->GetNClusterGroups());
   EXPECT_EQ(3U, ntuple->GetDescriptor()->GetNClusters());
   EXPECT_EQ(3U, ntuple->GetNEntries());
   auto rdPt = ntuple->GetModel()->GetDefaultEntry()->Get<float>("pt");

   ntuple->LoadEntry(0);
   EXPECT_EQ(42.0, *rdPt);
   ntuple->LoadEntry(1);
   EXPECT_EQ(24.0, *rdPt);
   ntuple->LoadEntry(2);
   EXPECT_EQ(12.0, *rdPt);
}
