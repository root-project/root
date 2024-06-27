#include "ntuple_test.hxx"

#include <ROOT/RPageNullSink.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#ifdef R__USE_IMT
#include <ROOT/TThreadExecutor.hxx>
#endif

#include <TRandom3.h>
#include <TMemFile.h>

#include <limits>

using ROOT::Experimental::Internal::RNTupleWriteOptionsManip;
using ROOT::Experimental::Internal::RPageNullSink;

namespace {
/// An RPageSink that keeps counters of (vector) commit of (sealed) pages; used to test RPageSinkBuf
class RPageSinkMock : public RPageSink {
public:
   struct {
      size_t fNCommitPage = 0;
      size_t fNCommitSealedPage = 0;
      size_t fNCommitSealedPageV = 0;
   } fCounters{};

protected:
   RPageAllocatorHeap fPageAllocator{};

   ColumnHandle_t AddColumn(ROOT::Experimental::DescriptorId_t, const ROOT::Experimental::Internal::RColumn &) final
   {
      return {};
   }

   const RNTupleDescriptor &GetDescriptor() const final
   {
      static RNTupleDescriptor descriptor;
      return descriptor;
   }

   void InitImpl(RNTupleModel &) final {}
   void UpdateSchema(const ROOT::Experimental::Internal::RNTupleModelChangeset &, NTupleSize_t) final {}
   void UpdateExtraTypeInfo(const ROOT::Experimental::RExtraTypeInfoDescriptor &) final {}
   void CommitPage(ColumnHandle_t /*columnHandle*/, const RPage & /*page*/) final { fCounters.fNCommitPage++; }
   void CommitSealedPage(ROOT::Experimental::DescriptorId_t, const RPageStorage::RSealedPage &) final
   {
      fCounters.fNCommitSealedPage++;
   }
   void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup>) override { fCounters.fNCommitSealedPageV++; }
   std::uint64_t CommitCluster(NTupleSize_t) final { return 0; }
   void CommitClusterGroup() final {}
   void CommitDatasetImpl() final {}

   RPage ReservePage(ColumnHandle_t columnHandle, std::size_t nElements) final
   {
      auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
      return fPageAllocator.NewPage(columnHandle.fPhysicalId, elementSize, nElements);
   }
   void ReleasePage(RPage &page) final { fPageAllocator.DeletePage(page); }

public:
   RPageSinkMock(const ROOT::Experimental::RNTupleWriteOptions &options) : RPageSink("test", options) {}
};
} // namespace

TEST(RNTuple, Basics)
{
   FileRaii fileGuard("test_ntuple_basics.ntuple");

   {
      auto model = RNTupleModel::Create();
      auto wrPt = model->MakeField<float>("pt", 42.0);

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      EXPECT_EQ(ntuple->GetNEntries(), 0);
      ntuple->Fill();
      EXPECT_EQ(ntuple->GetNEntries(), 1);
      EXPECT_EQ(ntuple->GetLastCommitted(), 0);
      ntuple->CommitCluster();
      EXPECT_EQ(ntuple->GetLastCommitted(), 1);
      *wrPt = 24.0;
      ntuple->Fill();
      *wrPt = 12.0;
      ntuple->Fill();
      EXPECT_EQ(ntuple->GetNEntries(), 3);
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(3U, ntuple->GetNEntries());
   auto rdPt = ntuple->GetModel().GetDefaultEntry().GetPtr<float>("pt");

   ntuple->LoadEntry(0);
   EXPECT_EQ(42.0, *rdPt);
   ntuple->LoadEntry(1);
   EXPECT_EQ(24.0, *rdPt);
   ntuple->LoadEntry(2);
   EXPECT_EQ(12.0, *rdPt);

   try {
      ntuple->LoadEntry(3);
      FAIL() << "loading a non-existing entry should throw";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("entry with index 3 out of bounds"));
   }
}

TEST(RNTuple, Extended)
{
   FileRaii fileGuard("test_ntuple_ext.ntuple");

   TRandom3 rnd(42);
   double chksumWrite = 0.0;
   {
      auto model = RNTupleModel::Create();
      auto wrVector = model->MakeField<std::vector<double>>("vector");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      constexpr unsigned int nEvents = 32000;
      for (unsigned int i = 0; i < nEvents; ++i) {
         auto nVec = 1 + floor(rnd.Rndm() * 1000.);
         wrVector->resize(nVec);
         for (unsigned int n = 0; n < nVec; ++n) {
            auto val = 1 + rnd.Rndm() * 1000. - 500.;
            (*wrVector)[n] = val;
            chksumWrite += val;
         }
         ntuple->Fill();
         if (i % 1000 == 0)
            ntuple->CommitCluster();
      }
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   auto rdVector = ntuple->GetModel().GetDefaultEntry().GetPtr<std::vector<double>>("vector");

   double chksumRead = 0.0;
   for (auto entryId : *ntuple) {
      ntuple->LoadEntry(entryId);
      for (auto v : *rdVector)
         chksumRead += v;
   }
   EXPECT_EQ(chksumRead, chksumWrite);
}

TEST(RNTuple, InvalidWriteOptions)
{
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

TEST(RNTuple, PageFilling)
{
   FileRaii fileGuard("test_ntuple_page_filling.root");

   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<std::int16_t>("x");

      RNTupleWriteOptions options;
      // Exercises the page swapping algorithm with pages just big enough to hold 2 elements
      options.SetApproxUnzippedPageSize(4);
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

   const auto &desc = ntuple->GetDescriptor();
   EXPECT_EQ(3u, desc.GetNClusters());
   const auto &cd1 = desc.GetClusterDescriptor(desc.FindClusterId(0, 0));
   const auto &pr1 = cd1.GetPageRange(0);
   ASSERT_EQ(2u, pr1.fPageInfos.size());
   EXPECT_EQ(2u, pr1.fPageInfos[0].fNElements);
   EXPECT_EQ(1u, pr1.fPageInfos[1].fNElements);
   const auto &cd2 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd1.GetId()));
   const auto &pr2 = cd2.GetPageRange(0);
   ASSERT_EQ(1u, pr2.fPageInfos.size());
   EXPECT_EQ(2u, pr2.fPageInfos[0].fNElements);
   const auto &cd3 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd2.GetId()));
   const auto &pr3 = cd3.GetPageRange(0);
   ASSERT_EQ(2u, pr3.fPageInfos.size());
   EXPECT_EQ(2u, pr3.fPageInfos[0].fNElements);
   EXPECT_EQ(1u, pr3.fPageInfos[1].fNElements);
}

TEST(RNTuple, PageFillingTail)
{
   FileRaii fileGuard("test_ntuple_page_filling_tail.root");

   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<std::int16_t>("x");

      RNTupleWriteOptions options;
      // Exercises the tail page optimization with pages to hold 4 elements
      options.SetApproxUnzippedPageSize(8);
      options.SetUseTailPageOptimization(true);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      for (std::int16_t i = 0; i < 16; ++i) {
         *fldX = i;
         ntuple->Fill();
         // Trigger tail page optimization; the page has 4 + 1 elements
         if (i == 4)
            ntuple->CommitCluster();
         // Trigger another tail page optimization: the first page in this cluster had 4 elements and was flushed
         // automatically; the second page has 4 + 1 elements
         if (i == 13)
            ntuple->CommitCluster();
      }
   }

   auto ntuple = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewX = ntuple->GetView<std::int16_t>("x");
   ASSERT_EQ(16u, ntuple->GetNEntries());
   for (std::int16_t i = 0; i < 16; ++i)
      EXPECT_EQ(i, viewX(i));

   const auto &desc = ntuple->GetDescriptor();
   EXPECT_EQ(3u, desc.GetNClusters());
   const auto &cd1 = desc.GetClusterDescriptor(desc.FindClusterId(0, 0));
   const auto &pr1 = cd1.GetPageRange(0);
   ASSERT_EQ(1u, pr1.fPageInfos.size());
   EXPECT_EQ(5u, pr1.fPageInfos[0].fNElements);
   const auto &cd2 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd1.GetId()));
   const auto &pr2 = cd2.GetPageRange(0);
   ASSERT_EQ(2u, pr2.fPageInfos.size());
   EXPECT_EQ(4u, pr2.fPageInfos[0].fNElements);
   EXPECT_EQ(5u, pr2.fPageInfos[1].fNElements);
   const auto &cd3 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd2.GetId()));
   const auto &pr3 = cd3.GetPageRange(0);
   ASSERT_EQ(1u, pr3.fPageInfos.size());
   EXPECT_EQ(2u, pr3.fPageInfos[0].fNElements);
}

TEST(RNTuple, PageFillingTailOff)
{
   FileRaii fileGuard("test_ntuple_page_filling_tail_off.root");

   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<std::int16_t>("x");

      RNTupleWriteOptions options;
      // Exercises the (disabled) tail page optimization with pages to hold 4 elements
      options.SetApproxUnzippedPageSize(8);
      options.SetUseTailPageOptimization(false);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      for (std::int16_t i = 0; i < 16; ++i) {
         *fldX = i;
         ntuple->Fill();
         // Tail page optimization should not be active: the pages have 4 and 1 elements
         if (i == 4)
            ntuple->CommitCluster();
         // Two pages of 4 elements and one undersized tail page with only 1 element
         if (i == 13)
            ntuple->CommitCluster();
      }
   }

   auto ntuple = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewX = ntuple->GetView<std::int16_t>("x");
   ASSERT_EQ(16u, ntuple->GetNEntries());
   for (std::int16_t i = 0; i < 16; ++i)
      EXPECT_EQ(i, viewX(i));

   const auto &desc = ntuple->GetDescriptor();
   EXPECT_EQ(3u, desc.GetNClusters());
   const auto &cd1 = desc.GetClusterDescriptor(desc.FindClusterId(0, 0));
   const auto &pr1 = cd1.GetPageRange(0);
   ASSERT_EQ(2u, pr1.fPageInfos.size());
   EXPECT_EQ(4u, pr1.fPageInfos[0].fNElements);
   EXPECT_EQ(1u, pr1.fPageInfos[1].fNElements);
   const auto &cd2 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd1.GetId()));
   const auto &pr2 = cd2.GetPageRange(0);
   ASSERT_EQ(3u, pr2.fPageInfos.size());
   EXPECT_EQ(4u, pr2.fPageInfos[0].fNElements);
   EXPECT_EQ(4u, pr2.fPageInfos[1].fNElements);
   EXPECT_EQ(1u, pr2.fPageInfos[2].fNElements);
   const auto &cd3 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd2.GetId()));
   const auto &pr3 = cd3.GetPageRange(0);
   ASSERT_EQ(1u, pr3.fPageInfos.size());
   EXPECT_EQ(2u, pr3.fPageInfos[0].fNElements);
}

TEST(RNTuple, PageFillingString)
{
   FileRaii fileGuard("test_ntuple_page_filling_string.root");

   {
      auto model = RNTupleModel::Create();
      // A string column exercises RColumn::AppendV
      auto fldX = model->MakeField<std::string>("x");

      RNTupleWriteOptions options;
      options.SetApproxUnzippedPageSize(16);
      options.SetUseTailPageOptimization(true);
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
      // 2 pages: 16 and 10 characters
      *fldX = "01234567890123456789012";
      ntuple->Fill();
      *fldX = "012";
      ntuple->Fill(); // main write page is half full here; RColumn::AppendV should flush the shadow page
   }

   auto ntuple = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewX = ntuple->GetView<std::string>("x");
   ASSERT_EQ(5u, ntuple->GetNEntries());
   EXPECT_EQ("01234567890123456", viewX(0));
   EXPECT_EQ("0123456789012345", viewX(1));
   EXPECT_EQ("", viewX(2));
   EXPECT_EQ("01234567890123456789012", viewX(3));
   EXPECT_EQ("012", viewX(4));

   const auto &desc = ntuple->GetDescriptor();
   EXPECT_EQ(4u, desc.GetNClusters());
   const auto &cd1 = desc.GetClusterDescriptor(desc.FindClusterId(1, 0));
   const auto &pr1 = cd1.GetPageRange(1);
   ASSERT_EQ(1u, pr1.fPageInfos.size());
   EXPECT_EQ(17u, pr1.fPageInfos[0].fNElements);
   const auto &cd2 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd1.GetId()));
   const auto &pr2 = cd2.GetPageRange(1);
   ASSERT_EQ(1u, pr2.fPageInfos.size());
   EXPECT_EQ(16u, pr2.fPageInfos[0].fNElements);
   const auto &cd3 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd2.GetId()));
   const auto &pr3 = cd3.GetPageRange(1);
   ASSERT_EQ(0u, pr3.fPageInfos.size());
   const auto &cd4 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd3.GetId()));
   const auto &pr4 = cd4.GetPageRange(1);
   ASSERT_EQ(2u, pr4.fPageInfos.size());
   EXPECT_EQ(16u, pr4.fPageInfos[0].fNElements);
   EXPECT_EQ(10u, pr4.fPageInfos[1].fNElements);
}

#ifdef R__HAS_DAVIX
TEST(RNTuple, OpenHTTP)
{
   std::unique_ptr<TFile> file(TFile::Open("http://root.cern/files/tutorials/ntpl004_dimuon_v1rc2.root"));
   auto reader = RNTupleReader::Open(file->Get<RNTuple>("Events"));
   reader->LoadEntry(0);
}
#endif

TEST(RNTuple, TMemFile)
{
   TMemFile file("memfile.root", "RECREATE");

   {
      auto model = RNTupleModel::Create();
      auto pt = model->MakeField<float>("pt", 42.0);

      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", file);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open(file.Get<RNTuple>("ntpl"));
   auto pt = reader->GetModel().GetDefaultEntry().GetPtr<float>("pt");
   reader->LoadEntry(0);
   EXPECT_EQ(*pt, 42.0);
}

TEST(RPageSinkBuf, Basics)
{
   struct TestModel {
      std::unique_ptr<RNTupleModel> fModel;
      std::shared_ptr<float> fFloatField;
      std::shared_ptr<std::vector<CustomStruct>> fFieldKlassVec;
      TestModel()
      {
         fModel = RNTupleModel::Create();
         fFloatField = fModel->MakeField<float>("pt");
         fFieldKlassVec = fModel->MakeField<std::vector<CustomStruct>>("klassVec");
      }
   };

   FileRaii fileGuardBuf("test_ntuple_sinkbuf_basics_buf.root");
   FileRaii fileGuard("test_ntuple_sinkbuf_basics.root");
   {
      RNTupleWriteOptions options;
      options.SetUseBufferedWrite(true);
      TestModel bufModel;
      // PageSinkBuf wraps a concrete page source
      auto ntupleBuf = RNTupleWriter::Recreate(std::move(bufModel.fModel), "buf", fileGuardBuf.GetPath(), options);
      ntupleBuf->EnableMetrics();

      options.SetUseBufferedWrite(false);
      TestModel unbufModel;
      auto ntuple = RNTupleWriter::Recreate(std::move(unbufModel.fModel), "unbuf", fileGuard.GetPath(), options);

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
            auto *parallel_zip = ntupleBuf->GetMetrics().GetCounter("RNTupleWriter.RPageSinkBuf.ParallelZip");
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
   const auto &cluster0 = ntupleBuf->GetDescriptor().GetClusterDescriptor(0);
   for (std::size_t i = 0; i < num_columns; i++) {
      const auto &columnPages = cluster0.GetPageRange(i);
      for (const auto &page : columnPages.fPageInfos) {
         pagePositions.push_back(std::make_pair(i, page.fLocator.GetPosition<std::uint64_t>()));
      }
   }

   auto sortedPages = pagePositions;
   std::sort(begin(sortedPages), end(sortedPages), [](const auto &a, const auto &b) { return a.second < b.second; });

   // For this test, ensure at least some columns have multiple pages
   ASSERT_TRUE(sortedPages.size() > num_columns);
   // Buffered sink cluster column pages are written out together
   for (std::size_t i = 0; i < pagePositions.size() - 1; i++) {
      // if the next page belongs to another column, skip the check
      if (pagePositions.at(i + 1).first != pagePositions.at(i).first) {
         continue;
      }
      auto page = std::find(begin(sortedPages), end(sortedPages), pagePositions[i]);
      ASSERT_TRUE(page != sortedPages.end());
      auto next_page = page + 1;
      auto column = pagePositions[i].first;
      ASSERT_EQ(column, next_page->first);
   }
}

TEST(RPageSinkBuf, ParallelZip)
{
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif

   FileRaii fileGuard("test_ntuple_sinkbuf_pzip.root");
   {
      auto model = RNTupleModel::Create();
      auto floatField = model->MakeField<float>("pt");
      auto fieldKlassVec = model->MakeField<std::vector<CustomStruct>>("klassVec");
      auto writer = RNTupleWriter::Recreate(std::move(model), "buf_pzip", fileGuard.GetPath());
      writer->EnableMetrics();
      for (int i = 0; i < 20000; i++) {
         *floatField = static_cast<float>(i);
         CustomStruct klass;
         klass.a = 42.0;
         klass.v1.emplace_back(static_cast<float>(i));
         klass.v2.emplace_back(std::vector<float>(3, static_cast<float>(i)));
         klass.s = "hi" + std::to_string(i);
         *fieldKlassVec = std::vector<CustomStruct>{klass};
         writer->Fill();
         if (i && i % 15000 == 0) {
            writer->CommitCluster();
            auto *parallel_zip = writer->GetMetrics().GetCounter("RNTupleWriter.RPageSinkBuf.ParallelZip");
            ASSERT_FALSE(parallel_zip == nullptr);
#ifdef R__USE_IMT
            EXPECT_EQ(1, parallel_zip->GetValueAsInt());
#else
            EXPECT_EQ(0, parallel_zip->GetValueAsInt());
#endif
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

#ifdef R__USE_IMT
   ROOT::DisableImplicitMT();
#endif
}

#ifdef R__USE_IMT
TEST(RPageSinkBuf, ParallelZipIMT)
{
   ROOT::EnableImplicitMT(2);

   ROOT::TThreadExecutor ex(2);
   ex.Foreach(
      [&](int i) {
         std::string filename = "test_ntuple_sinkbuf_pzip.";
         filename += std::to_string(i);
         filename += ".root";
         FileRaii fileGuard(filename);

         auto model = ROOT::Experimental::RNTupleModel::Create();
         auto wrPt = model->MakeField<float>("pt", 42.0);

         RNTupleWriteOptions options;
         options.SetApproxUnzippedPageSize(8);
         options.SetUseBufferedWrite(true);
         auto writer =
            ROOT::Experimental::RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), options);
         for (int c = 0; c < 2; c++) {
            writer->Fill();
            writer->Fill();
            // The first page is full now.
            writer->Fill();
            writer->Fill();
            // The second page is full now.
            writer->Fill();
            writer->CommitCluster();
         }
      },
      {1, 2});

   ROOT::DisableImplicitMT();
}
#endif

TEST(RPageSinkBuf, CommitSealedPageV)
{
   RNTupleWriteOptions options;
   options.SetApproxUnzippedPageSize(16);

   {
      std::unique_ptr<RPageSink> sink(new RPageSinkMock(options));
      auto &counters = static_cast<RPageSinkMock *>(sink.get())->fCounters;

      auto model = RNTupleModel::Create();
      auto u64Field = model->MakeField<std::uint64_t>("u64");
      auto u32Field = model->MakeField<std::uint16_t>("u32");
      auto strField = model->MakeField<std::string>("str");
      auto ntuple = ROOT::Experimental::Internal::CreateRNTupleWriter(std::move(model),
                                                                      std::make_unique<RPageSinkBuf>(std::move(sink)));
      ntuple->Fill();
      ntuple->Fill();
      ntuple->Fill();
      ntuple->CommitCluster();
      // Parallel zip not available, but pages are still vector-commited
      EXPECT_EQ(0, counters.fNCommitPage);
      EXPECT_EQ(0, counters.fNCommitSealedPage);
      EXPECT_EQ(1, counters.fNCommitSealedPageV);
   }
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif
   {
      std::unique_ptr<RPageSink> sink(new RPageSinkMock(options));
      auto &counters = static_cast<RPageSinkMock *>(sink.get())->fCounters;

      auto model = RNTupleModel::Create();
      auto u64Field = model->MakeField<std::uint64_t>("u64");
      auto u32Field = model->MakeField<std::uint32_t>("u32");
      auto strField = model->MakeField<std::string>("str");
      auto ntuple = ROOT::Experimental::Internal::CreateRNTupleWriter(std::move(model),
                                                                      std::make_unique<RPageSinkBuf>(std::move(sink)));
      ntuple->Fill();
      ntuple->Fill();
      ntuple->CommitCluster();
      // All pages in all columns committed via a single call to `CommitSealedPageV()`
      EXPECT_EQ(0, counters.fNCommitPage);
      EXPECT_EQ(0, counters.fNCommitSealedPage);
      EXPECT_EQ(1, counters.fNCommitSealedPageV);
   }

#ifdef R__USE_IMT
   ROOT::DisableImplicitMT();
#endif
}

TEST(RPageSink, Empty)
{
   FileRaii fileGuard("test_ntuple_empty.ntuple");

   {
      auto model = RNTupleModel::Create();
      auto wrPt = model->MakeField<float>("pt", 42.0);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(0U, ntuple->GetNEntries());
   EXPECT_EQ(0U, ntuple->GetDescriptor().GetNClusterGroups());
   EXPECT_EQ(0U, ntuple->GetDescriptor().GetNClusters());
}

TEST(RPageSink, MultipleClusterGroups)
{
   FileRaii fileGuard("test_ntuple_multi_cluster_groups.ntuple");

   {
      auto model = RNTupleModel::Create();
      auto wrPt = model->MakeField<float>("pt", 42.0);
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
   EXPECT_EQ(2U, ntuple->GetDescriptor().GetNClusterGroups());
   EXPECT_EQ(3U, ntuple->GetDescriptor().GetNClusters());
   EXPECT_EQ(3U, ntuple->GetNEntries());
   auto rdPt = ntuple->GetModel().GetDefaultEntry().GetPtr<float>("pt");

   ntuple->LoadEntry(0);
   EXPECT_EQ(42.0, *rdPt);
   ntuple->LoadEntry(1);
   EXPECT_EQ(24.0, *rdPt);
   ntuple->LoadEntry(2);
   EXPECT_EQ(12.0, *rdPt);
}

TEST(RPageNullSink, Basics)
{
   auto model = RNTupleModel::Create();
   auto wrPt = model->MakeField<float>("pt");

   auto sink = std::make_unique<RPageNullSink>("null", RNTupleWriteOptions());
   auto ntuple = ROOT::Experimental::Internal::CreateRNTupleWriter(std::move(model), std::move(sink));

   *wrPt = 42.0;
   ntuple->Fill();
   EXPECT_EQ(ntuple->GetNEntries(), 1);
}

TEST(RPageStorageFile, MultiKeyBlob)
{
   FileRaii fileGuard("test_ntuple_storage_multi_key_blob.root");

   const auto kMaxKeySize = 10 * 1024 * 1024; // 10 MiB
   const auto dataSize = kMaxKeySize * 2;
   auto data = std::make_unique<unsigned char[]>(dataSize);
   std::uint64_t blobOffset;

   {
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), 0, EContainerFormat::kTFile, kMaxKeySize);
      memset(data.get(), 0x99, dataSize);
      data[42] = 0x42;
      data[dataSize - 42] = 0x11;
      blobOffset = writer->WriteBlob(data.get(), dataSize, dataSize);
      writer->Commit();
   }
   {
      memset(data.get(), 0, dataSize);

      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      // Force reader to read the max key size
      (void)reader.GetNTuple("ntpl");
      reader.ReadBuffer(data.get(), dataSize, blobOffset);

      EXPECT_EQ(data[0], 0x99);
      EXPECT_EQ(data[dataSize / 2], 0x99);
      EXPECT_EQ(data[2 * dataSize / 3], 0x99);
      EXPECT_EQ(data[dataSize - 1], 0x99);
      EXPECT_EQ(data[42], 0x42);
      EXPECT_EQ(data[dataSize - 42], 0x11);
   }
}

TEST(RPageStorageFile, MultiKeyBlob_ExactlyMax)
{
   // Write a payload that's exactly `maxKeySize` long and verify it doesn't split the key.

   FileRaii fileGuard("test_ntuple_storage_multi_key_exact.root");

   const auto kMaxKeySize = 100 * 1024; // 100 KiB
   const auto dataSize = kMaxKeySize;
   auto data = std::make_unique<unsigned char[]>(dataSize);
   std::uint64_t blobOffset;

   {
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), 0, EContainerFormat::kTFile, kMaxKeySize);
      memset(data.get(), 0, dataSize);
      blobOffset = writer->WriteBlob(data.get(), dataSize, dataSize);
      writer->Commit();
   }
   {
      // Fill read buffer with sentinel data (they should be overwritten by zeroes)
      memset(data.get(), 0x99, dataSize);

      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      // Force reader to read the max key size
      (void)reader.GetNTuple("ntpl");
      rawFile->ReadAt(data.get(), dataSize, blobOffset);

      // If we didn't split the key, we expect to find all zeroes at the end of `data`.
      // Otherwise we will have some non-zero bytes, since it will host the next chunk offset.
      uint64_t lastU64 = *reinterpret_cast<uint64_t *>(&data[dataSize - sizeof(uint64_t)]);
      EXPECT_EQ(lastU64, 0);
   }
}

TEST(RPageStorageFile, MultiKeyBlob_SmallKey)
{
   FileRaii fileGuard("test_ntuple_storage_multi_key_blob_small_key.root");

   const auto kMaxKeySize = 50 * 1024; // 50 KiB
   const auto dataSize = kMaxKeySize * 1000;
   auto data = std::make_unique<unsigned char[]>(dataSize);
   std::uint64_t blobOffset;

   {
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), 0, EContainerFormat::kTFile, kMaxKeySize);
      memset(data.get(), 0x99, dataSize);
      data[42] = 0x42;
      data[dataSize - 42] = 0x84;
      blobOffset = writer->WriteBlob(data.get(), dataSize, dataSize);
      writer->Commit();
   }
   {
      memset(data.get(), 0, dataSize);

      auto rawFile = RRawFile::Create(fileGuard.GetPath());
      auto reader = RMiniFileReader{rawFile.get()};
      // Force reader to read the max key size
      (void)reader.GetNTuple("ntpl");
      reader.ReadBuffer(data.get(), dataSize, blobOffset);

      EXPECT_EQ(data[0], 0x99);
      EXPECT_EQ(data[dataSize / 2], 0x99);
      EXPECT_EQ(data[2 * dataSize / 3], 0x99);
      EXPECT_EQ(data[dataSize - 1], 0x99);
      EXPECT_EQ(data[42], 0x42);
      EXPECT_EQ(data[dataSize - 42], 0x84);
   }
}

TEST(RPageStorageFile, MultiKeyBlob_TooManyChunks)
{
   // Try writing more than the max possible number of chunks for a split key and verify it fails

#ifdef GTEST_FLAG_SET
   // Death tests must run single-threaded:
   // https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
   GTEST_FLAG_SET(death_test_style, "threadsafe");
#endif

   FileRaii fileGuard("test_ntuple_storage_multi_key_blob_small_key.root");

   const auto kMaxKeySize = 128;

   {
      const auto kOkayDataSize = 1024;
      const auto data = std::make_unique<unsigned char[]>(kOkayDataSize);
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), 0, EContainerFormat::kTFile, kMaxKeySize);
      memset(data.get(), 0x99, kOkayDataSize);
      writer->WriteBlob(data.get(), kOkayDataSize, kOkayDataSize);
      writer->Commit();
   }

   {
      const auto kTooBigDataSize = 5000;
      const auto data = std::make_unique<unsigned char[]>(kTooBigDataSize);
      auto writer = RNTupleFileWriter::Recreate("ntpl", fileGuard.GetPath(), 0, EContainerFormat::kTFile, kMaxKeySize);
      memset(data.get(), 0x99, kTooBigDataSize);
      EXPECT_DEATH(writer->WriteBlob(data.get(), kTooBigDataSize, kTooBigDataSize), "");
      writer->Commit();
   }
}

template <int... N>
static void GenerateLotsOfFields(RNTupleModel &model, int firstIdx, std::integer_sequence<int, N...>)
{
   (model.MakeField<double>(std::to_string(N + firstIdx)), ...);
}

TEST(RPageStorageFile, MultiKeyBlob_Header)
{
   // Write split header and read it back

   FileRaii fileGuardComp("test_ntuple_storage_multi_key_blob_header_comp.root");
   FileRaii fileGuardUcmp("test_ntuple_storage_multi_key_blob_header_ucmp.root");

   constexpr auto kMaxKeySize = 1024;
   constexpr auto kNFields = 1024;

   auto modelComp = RNTupleModel::Create();
   // NOTE: must call this multiple times not to exceed the max fold expansion limit
   for (int i = 0; i < 4; ++i)
      GenerateLotsOfFields(*modelComp, i * kNFields / 4, std::make_integer_sequence<int, kNFields / 4>{});
   auto modelUcmp = modelComp->Clone();

   auto optionsComp = RNTupleWriteOptions{};
   RNTupleWriteOptionsManip::SetMaxKeySize(optionsComp, kMaxKeySize);
   auto optionsUcmp = optionsComp.Clone();
   optionsUcmp->SetCompression(0);

   {
      auto writerComp = RNTupleWriter::Recreate(std::move(modelComp), "myNTuple", fileGuardComp.GetPath(), optionsComp);
      auto writerUcmp =
         RNTupleWriter::Recreate(std::move(modelUcmp), "myNTuple", fileGuardUcmp.GetPath(), *optionsUcmp);
      writerComp->Fill();
      writerUcmp->Fill();
   }

   {
      auto ntupleComp = RNTupleReader::Open("myNTuple", fileGuardComp.GetPath());
      auto ntupleUcmp = RNTupleReader::Open("myNTuple", fileGuardUcmp.GetPath());

      EXPECT_GE(ntupleComp->GetDescriptor().GetOnDiskHeaderSize(), kMaxKeySize);
      EXPECT_GE(ntupleUcmp->GetDescriptor().GetOnDiskHeaderSize(), kMaxKeySize);

      for (int i = 0; i < kNFields; ++i) {
         ntupleComp->GetModel().GetField(std::to_string(i));
         ntupleUcmp->GetModel().GetField(std::to_string(i));
      }
   }
}

TEST(RPageStorageFile, MultiKeyBlob_Pages)
{
   // Write split pages and read them back

   FileRaii fileGuardComp("test_ntuple_storage_multi_key_blob_pages_comp.root");
   FileRaii fileGuardUcmp("test_ntuple_storage_multi_key_blob_pages_ucmp.root");

   constexpr auto kMaxKeySize = 1024;

   auto optionsComp = RNTupleWriteOptions{};
   optionsComp.SetApproxUnzippedPageSize(kMaxKeySize * 10);
   RNTupleWriteOptionsManip::SetMaxKeySize(optionsComp, kMaxKeySize);
   auto optionsUcmp = optionsComp.Clone();
   optionsUcmp->SetCompression(0);

   {
      auto modelUcmp = RNTupleModel::Create();
      auto modelComp = RNTupleModel::Create();
      auto fU = modelUcmp->MakeField<double>("f");
      auto fC = modelComp->MakeField<double>("f");
      auto writerUcmp =
         RNTupleWriter::Recreate(std::move(modelUcmp), "myNTuple", fileGuardUcmp.GetPath(), *optionsUcmp);
      auto writerComp = RNTupleWriter::Recreate(std::move(modelComp), "myNTuple", fileGuardComp.GetPath(), optionsComp);
      TRandom3 rnd(42);
      for (int i = 0; i < 100000; ++i) {
         *fC = rnd.Rndm() * std::numeric_limits<double>::max();
         *fU = rnd.Rndm() * std::numeric_limits<double>::max();
         writerComp->Fill();
         writerUcmp->Fill();
      }
   }

   {
      auto modelUcmp = RNTupleModel::Create();
      auto modelComp = RNTupleModel::Create();
      auto fU = modelUcmp->MakeField<double>("f");
      auto fC = modelComp->MakeField<double>("f");
      auto ntupleUcmp = RNTupleReader::Open(std::move(modelUcmp), "myNTuple", fileGuardUcmp.GetPath());
      auto ntupleComp = RNTupleReader::Open(std::move(modelComp), "myNTuple", fileGuardComp.GetPath());

      // Verify that the pages are larger than maxKeySize
      EXPECT_GT(
         ntupleComp->GetDescriptor().GetClusterDescriptor(0).GetPageRange(0).fPageInfos[0].fLocator.fBytesOnStorage,
         kMaxKeySize);
      EXPECT_GT(
         ntupleUcmp->GetDescriptor().GetClusterDescriptor(0).GetPageRange(0).fPageInfos[0].fLocator.fBytesOnStorage,
         kMaxKeySize);

      TRandom3 rnd(42);
      for (int i = 0; i < 100000; ++i) {
         ntupleUcmp->LoadEntry(i);
         ntupleComp->LoadEntry(i);
         double valC = *fC;
         double expectC = rnd.Rndm() * std::numeric_limits<double>::max();
         double valU = *fU;
         double expectU = rnd.Rndm() * std::numeric_limits<double>::max();
         EXPECT_DOUBLE_EQ(valC, expectC);
         EXPECT_DOUBLE_EQ(valU, expectU);
      }
   }
}

TEST(RPageStorageFile, MultiKeyBlob_PageList)
{
   // Write a split page list and read it back

   FileRaii fileGuardComp("test_ntuple_storage_multi_key_blob_page_list_comp.root");
   FileRaii fileGuardUcmp("test_ntuple_storage_multi_key_blob_page_list_ucmp.root");

   constexpr auto kMaxKeySize = 1024;

   auto optionsComp = RNTupleWriteOptions{};
   optionsComp.SetApproxUnzippedPageSize(128);
   RNTupleWriteOptionsManip::SetMaxKeySize(optionsComp, kMaxKeySize);
   auto optionsUcmp = optionsComp.Clone();
   optionsUcmp->SetCompression(0);

   {
      auto modelComp = RNTupleModel::Create();
      auto modelUcmp = RNTupleModel::Create();
      auto fC = modelComp->MakeField<double>("f");
      auto fU = modelUcmp->MakeField<double>("f");

      auto writerComp = RNTupleWriter::Recreate(std::move(modelComp), "myNTuple", fileGuardComp.GetPath(), optionsComp);
      auto writerUcmp =
         RNTupleWriter::Recreate(std::move(modelUcmp), "myNTuple", fileGuardUcmp.GetPath(), *optionsUcmp);
      TRandom3 rnd(84);
      for (int i = 0; i < 100000; ++i) {
         *fC = rnd.Rndm() * std::numeric_limits<double>::max();
         *fU = rnd.Rndm() * std::numeric_limits<double>::max();
         writerComp->Fill();
         writerUcmp->Fill();
      }
   }

   {
      auto modelComp = RNTupleModel::Create();
      auto modelUcmp = RNTupleModel::Create();
      auto fC = modelComp->MakeField<double>("f");
      auto fU = modelUcmp->MakeField<double>("f");
      auto ntupleComp = RNTupleReader::Open(std::move(modelComp), "myNTuple", fileGuardComp.GetPath());
      auto ntupleUcmp = RNTupleReader::Open(std::move(modelUcmp), "myNTuple", fileGuardUcmp.GetPath());

      // Verify that the page list is larger than maxKeySize
      EXPECT_GT(ntupleComp->GetDescriptor().GetClusterGroupDescriptor(0).GetPageListLength(), kMaxKeySize);
      EXPECT_GT(ntupleUcmp->GetDescriptor().GetClusterGroupDescriptor(0).GetPageListLength(), kMaxKeySize);

      TRandom3 rnd(84);
      for (int i = 0; i < 100000; ++i) {
         ntupleComp->LoadEntry(i);
         ntupleUcmp->LoadEntry(i);
         double valC = *fC;
         double expectC = rnd.Rndm() * std::numeric_limits<double>::max();
         double valU = *fU;
         double expectU = rnd.Rndm() * std::numeric_limits<double>::max();
         EXPECT_DOUBLE_EQ(valC, expectC);
         EXPECT_DOUBLE_EQ(valU, expectU);
      }
   }
}

TEST(RPageStorageFile, MultiKeyBlob_Footer)
{
   // Write a split footer and read it back

   FileRaii fileGuardComp("test_ntuple_storage_multi_key_blob_footer_comp.root");
   FileRaii fileGuardUcmp("test_ntuple_storage_multi_key_blob_footer_ucmp.root");

   constexpr auto kMaxKeySize = 1024;

   auto optionsComp = RNTupleWriteOptions{};
   RNTupleWriteOptionsManip::SetMaxKeySize(optionsComp, kMaxKeySize);
   auto optionsUcmp = optionsComp.Clone();
   optionsUcmp->SetCompression(0);

   {
      auto modelComp = RNTupleModel::Create();
      auto modelUcmp = RNTupleModel::Create();
      auto fC = modelComp->MakeField<double>("f");
      auto fU = modelUcmp->MakeField<double>("f");

      auto writerComp = RNTupleWriter::Recreate(std::move(modelComp), "myNTuple", fileGuardComp.GetPath(), optionsComp);
      auto writerUcmp =
         RNTupleWriter::Recreate(std::move(modelUcmp), "myNTuple", fileGuardUcmp.GetPath(), *optionsUcmp);
      TRandom3 rnd(84);
      for (int i = 0; i < 100000; ++i) {
         *fC = rnd.Rndm() * std::numeric_limits<double>::max();
         *fU = rnd.Rndm() * std::numeric_limits<double>::max();
         writerComp->Fill();
         writerUcmp->Fill();
         if (i % 100 == 0) {
            writerComp->CommitCluster(true);
            writerUcmp->CommitCluster(true);
         }
      }
   }

   {
      auto modelComp = RNTupleModel::Create();
      auto modelUcmp = RNTupleModel::Create();
      auto fC = modelComp->MakeField<double>("f");
      auto fU = modelUcmp->MakeField<double>("f");
      auto ntupleComp = RNTupleReader::Open(std::move(modelComp), "myNTuple", fileGuardComp.GetPath());
      auto ntupleUcmp = RNTupleReader::Open(std::move(modelUcmp), "myNTuple", fileGuardUcmp.GetPath());

      // Verify that the footer is actually bigger than the max key size
      EXPECT_GT(ntupleComp->GetDescriptor().GetOnDiskFooterSize(), kMaxKeySize);
      EXPECT_GT(ntupleUcmp->GetDescriptor().GetOnDiskFooterSize(), kMaxKeySize);

      TRandom3 rnd(84);
      for (int i = 0; i < 100000; ++i) {
         ntupleComp->LoadEntry(i);
         ntupleUcmp->LoadEntry(i);
         double valC = *fC;
         double expectC = rnd.Rndm() * std::numeric_limits<double>::max();
         double valU = *fU;
         double expectU = rnd.Rndm() * std::numeric_limits<double>::max();
         EXPECT_DOUBLE_EQ(valC, expectC);
         EXPECT_DOUBLE_EQ(valU, expectU);
      }
   }
}
