#include "ntuple_test.hxx"
#include <TRandom3.h>
#include <TMemFile.h>
#include <TVirtualStreamerInfo.h>

#include <ROOT/RPageNullSink.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#ifdef R__USE_IMT
#include <ROOT/TThreadExecutor.hxx>
#endif

#include <TRandom3.h>
#include <TMemFile.h>

#include <limits>

using ROOT::Experimental::Internal::RPageNullSink;
using ROOT::Internal::RNTupleWriteOptionsManip;

#include <cstring>

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
   ColumnHandle_t AddColumn(ROOT::DescriptorId_t, ROOT::Internal::RColumn &) final { return {}; }

   const RNTupleDescriptor &GetDescriptor() const final
   {
      static RNTupleDescriptor descriptor;
      return descriptor;
   }

   ROOT::NTupleSize_t GetNEntries() const final { return 0; }

   void InitImpl(RNTupleModel &) final {}
   void UpdateSchema(const ROOT::Internal::RNTupleModelChangeset &, ROOT::NTupleSize_t) final {}
   void UpdateExtraTypeInfo(const ROOT::RExtraTypeInfoDescriptor &) final {}
   void CommitSuppressedColumn(ColumnHandle_t) final {}
   void CommitPage(ColumnHandle_t /*columnHandle*/, const RPage & /*page*/) final { fCounters.fNCommitPage++; }
   void CommitSealedPage(ROOT::DescriptorId_t, const RPageStorage::RSealedPage &) final
   {
      fCounters.fNCommitSealedPage++;
   }
   void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup>) override { fCounters.fNCommitSealedPageV++; }
   RStagedCluster StageCluster(ROOT::NTupleSize_t) final { return {}; }
   void CommitStagedClusters(std::span<RStagedCluster>) final {}
   void CommitClusterGroup() final {}
   void CommitDatasetImpl() final {}

public:
   RPageSinkMock(const ROOT::RNTupleWriteOptions &options) : RPageSink("test", options) {}
};
} // namespace

TEST(RNTuple, Basics)
{
   FileRaii fileGuard("test_ntuple_basics.ntuple");

   {
      auto model = RNTupleModel::Create();
      auto wrPt = model->MakeField<float>("pt");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      EXPECT_EQ(ntuple->GetNEntries(), 0);
      *wrPt = 42.0;
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
   } catch (const ROOT::RException &err) {
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
      options.SetInitialUnzippedPageSize(0);
      FAIL() << "should not allow zero initial page size";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("initial page size"));
   }
   try {
      options.SetMaxUnzippedPageSize(0);
      FAIL() << "should not allow zero-sized page";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("maximum page size"));
   }
   try {
      options.SetMaxUnzippedPageSize(10);
      FAIL() << "should not allow undersized pages";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("must not be larger than"));
   }
   try {
      options.SetMaxUnzippedClusterSize(40);
      FAIL() << "should not allow undersized cluster";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("must not be larger than"));
   }
   options.SetApproxZippedClusterSize(5);
   try {
      options.SetMaxUnzippedClusterSize(7);
      FAIL() << "should not allow undersized cluster";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("must not be larger than"));
   }

   FileRaii fileGuard("test_ntuple_invalid_write_options.root");
   auto model = RNTupleModel::Create();
   model->MakeField<std::int16_t>("x");
   options.SetInitialUnzippedPageSize(1);
   auto m2 = model->Clone();
   try {
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      FAIL() << "should not allow undersized pages";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("too small for at least one element"));
   }
   options.SetApproxZippedClusterSize(10 * 1000 * 1000);
}

TEST(RNTuple, PageFilling)
{
   FileRaii fileGuard("test_ntuple_page_filling.root");

   // We want to reach the maximum page size but not overshoot
   {
      auto model = RNTupleModel::Create();
      auto fldX = model->MakeField<float>("x");
      auto fldY = model->MakeField<double>("y");

      RNTupleWriteOptions options;
      options.SetInitialUnzippedPageSize(24);
      options.SetMaxUnzippedPageSize(24);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      for (int i = 0; i < 6; ++i)
         writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(1u, desc.GetNClusters());
   const auto colIdX = desc.FindLogicalColumnId(desc.FindFieldId("x"), 0, 0);
   const auto colIdY = desc.FindLogicalColumnId(desc.FindFieldId("y"), 0, 0);

   const auto &clusterDesc = desc.GetClusterDescriptor(desc.FindClusterId(0, 0));
   const auto &prX = clusterDesc.GetPageRange(colIdX);
   const auto &prY = clusterDesc.GetPageRange(colIdY);
   ASSERT_EQ(1u, prX.GetPageInfos().size());
   EXPECT_EQ(6u, prX.GetPageInfos()[0].GetNElements());
   ASSERT_EQ(2u, prY.GetPageInfos().size());
   EXPECT_EQ(3u, prY.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(3u, prY.GetPageInfos()[1].GetNElements());
}

TEST(RNTuple, PageFillingString)
{
   FileRaii fileGuard("test_ntuple_page_filling_string.root");

   {
      auto model = RNTupleModel::Create();
      // A string column exercises RColumn::AppendV
      auto fldX = model->MakeField<std::string>("x");

      RNTupleWriteOptions options;
      options.SetInitialUnzippedPageSize(8);
      options.SetMaxUnzippedPageSize(16);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      // 2 page: 16 + 1 characters
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
   ASSERT_EQ(2u, pr1.GetPageInfos().size());
   EXPECT_EQ(16u, pr1.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(1u, pr1.GetPageInfos()[1].GetNElements());
   const auto &cd2 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd1.GetId()));
   const auto &pr2 = cd2.GetPageRange(1);
   ASSERT_EQ(1u, pr2.GetPageInfos().size());
   EXPECT_EQ(16u, pr2.GetPageInfos()[0].GetNElements());
   const auto &cd3 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd2.GetId()));
   const auto &pr3 = cd3.GetPageRange(1);
   ASSERT_EQ(0u, pr3.GetPageInfos().size());
   const auto &cd4 = desc.GetClusterDescriptor(desc.FindNextClusterId(cd3.GetId()));
   const auto &pr4 = cd4.GetPageRange(1);
   ASSERT_EQ(2u, pr4.GetPageInfos().size());
   EXPECT_EQ(16u, pr4.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(10u, pr4.GetPageInfos()[1].GetNElements());
}

TEST(RNTuple, FlushColumns)
{
   FileRaii fileGuard("test_ntuple_flush.root");

   {
      auto model = RNTupleModel::Create();
      auto pt = model->MakeField<float>("pt");

      auto writer = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());

      *pt = 1.0;
      writer->Fill();

      writer->FlushColumns();

      *pt = 2.0;
      writer->Fill();
   }

   // If FlushColumns() worked, there will be two pages with one element each.
   auto reader = RNTupleReader::Open("f", fileGuard.GetPath());
   const auto &descriptor = reader->GetDescriptor();

   auto fieldId = descriptor.FindFieldId("pt");
   auto columnId = descriptor.FindPhysicalColumnId(fieldId, 0, 0);
   auto &pageInfos = descriptor.GetClusterDescriptor(0).GetPageRange(columnId).GetPageInfos();
   ASSERT_EQ(pageInfos.size(), 2);
   EXPECT_EQ(pageInfos[0].GetNElements(), 1);
   EXPECT_EQ(pageInfos[1].GetNElements(), 1);
}

TEST(RNTuple, WritePageBudgetLimit)
{
   FileRaii fileGuard("test_ntuple_write_page_budget_limit.root");

   auto model = RNTupleModel::Create();
   model->MakeField<float>("x");
   model->Freeze();
   auto modelClone = model->Clone();

   RNTupleWriteOptions options;
   options.SetUseBufferedWrite(false);
   options.SetInitialUnzippedPageSize(4);
   options.SetPageBufferBudget(3);
   try {
      RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      FAIL() << "too small page buffer budget should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("page buffer memory budget too small"));
   }

   options.SetPageBufferBudget(4);
   auto writer = RNTupleWriter::Recreate(std::move(modelClone), "ntpl", fileGuard.GetPath(), options);
   auto ptrX = writer->GetModel().GetDefaultEntry().GetPtr<float>("x");
   *ptrX = 137.0;
   writer->Fill();
   writer.reset();

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewX = reader->GetView<float>("x");
   EXPECT_FLOAT_EQ(137., viewX(0));
}

TEST(RNTuple, WritePageBudget)
{
   FileRaii fileGuard("test_ntuple_write_page_budget.root");

   auto model = RNTupleModel::Create();
   auto ptrA = model->MakeField<std::vector<double>>("a");
   auto ptrB = model->MakeField<std::vector<double>>("b");
   auto ptrC = model->MakeField<std::vector<double>>("c");
   auto ptrD = model->MakeField<std::vector<double>>("d");

   RNTupleWriteOptions options;
   options.SetUseBufferedWrite(false);
   options.SetInitialUnzippedPageSize(8);
   options.SetPageBufferBudget(50 * 8);
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);

   *ptrC = std::vector<double>(20, 1.0);
   writer->Fill();
   // Page sizes (sum: 39):
   //   a     -- 1
   //   a._0  -- 1
   //   b     -- 1
   //   b._0  -- 1
   //   c     -- 1
   //   c._0  -- 32
   //   d     -- 1
   //   d._0  -- 1
   ptrC->clear();
   *ptrA = std::vector<double>(4, 2.0);
   writer->Fill();
   // Page sizes (sum: 47):
   //   a     -- 2
   //   a._0  -- 4
   //   b     -- 2
   //   b._0  -- 1
   //   c     -- 2
   //   c._0  -- 32
   //   d     -- 2
   //   d._0  -- 1
   ptrA->clear();
   *ptrB = std::vector<double>(40, 3.0);
   *ptrD = std::vector<double>(5, 4.0);
   writer->Fill();
   // Page sizes (sum: 37):
   //   a     -- 4
   //   a._0  -- 4
   //   b     -- 4
   //   b._0  -- 8 <-- flushed at attempt to expand from 16 to 32
   //   c     -- 4
   //   c._0  -- 1 <-- flushed to make room for b._0
   //   d     -- 4
   //   d._0  -- 8
   ptrB->clear();
   *ptrD = std::vector<double>(34, 4.0);
   *ptrC = std::vector<double>(4, 1.0);
   writer->Fill();
   // Page sizes (sum: 34):
   //   a     -- 4
   //   a._0  -- 4
   //   b     -- 4
   //   b._0  -- 8
   //   c     -- 4
   //   c._0  -- 4
   //   d     -- 4
   //   d._0  -- 2 <-- flushed twice at attempt to expand from 16 to 32
   writer.reset();

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto viewA = reader->GetView<std::vector<double>>("a");
   auto viewB = reader->GetView<std::vector<double>>("b");
   auto viewC = reader->GetView<std::vector<double>>("c");
   auto viewD = reader->GetView<std::vector<double>>("d");
   EXPECT_EQ(20u, viewC(0).size());
   EXPECT_EQ(0u, viewC(1).size());
   EXPECT_EQ(0u, viewC(2).size());
   EXPECT_EQ(4u, viewC(3).size());
   EXPECT_EQ(0u, viewA(0).size());
   EXPECT_EQ(4u, viewA(1).size());
   EXPECT_EQ(0u, viewA(2).size());
   EXPECT_EQ(0u, viewA(3).size());
   EXPECT_EQ(0u, viewB(0).size());
   EXPECT_EQ(0u, viewB(1).size());
   EXPECT_EQ(40u, viewB(2).size());
   EXPECT_EQ(0u, viewB(3).size());
   EXPECT_EQ(0u, viewD(0).size());
   EXPECT_EQ(0u, viewD(1).size());
   EXPECT_EQ(5u, viewD(2).size());
   EXPECT_EQ(34u, viewD(3).size());

   const auto &desc = reader->GetDescriptor();
   const auto fldIdA = desc.FindFieldId("a");
   const auto fldIdB = desc.FindFieldId("b");
   const auto fldIdC = desc.FindFieldId("c");
   const auto fldIdD = desc.FindFieldId("d");
   const auto colIdA = desc.FindLogicalColumnId(fldIdA, 0, 0);
   const auto colIdB = desc.FindLogicalColumnId(fldIdB, 0, 0);
   const auto colIdC = desc.FindLogicalColumnId(fldIdC, 0, 0);
   const auto colIdD = desc.FindLogicalColumnId(fldIdD, 0, 0);
   const auto colIdAD = desc.FindLogicalColumnId(desc.FindFieldId("_0", fldIdA), 0, 0);
   const auto colIdBD = desc.FindLogicalColumnId(desc.FindFieldId("_0", fldIdB), 0, 0);
   const auto colIdCD = desc.FindLogicalColumnId(desc.FindFieldId("_0", fldIdC), 0, 0);
   const auto colIdDD = desc.FindLogicalColumnId(desc.FindFieldId("_0", fldIdD), 0, 0);

   const auto &clusterDesc = desc.GetClusterDescriptor(0);
   const auto &prA = clusterDesc.GetPageRange(colIdA);
   const auto &prB = clusterDesc.GetPageRange(colIdB);
   const auto &prC = clusterDesc.GetPageRange(colIdC);
   const auto &prD = clusterDesc.GetPageRange(colIdD);
   const auto &prAD = clusterDesc.GetPageRange(colIdAD);
   const auto &prBD = clusterDesc.GetPageRange(colIdBD);
   const auto &prCD = clusterDesc.GetPageRange(colIdCD);
   const auto &prDD = clusterDesc.GetPageRange(colIdDD);

   EXPECT_EQ(1u, prA.GetPageInfos().size());
   EXPECT_EQ(4u, prA.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(1u, prB.GetPageInfos().size());
   EXPECT_EQ(4u, prB.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(1u, prC.GetPageInfos().size());
   EXPECT_EQ(4u, prC.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(1u, prD.GetPageInfos().size());
   EXPECT_EQ(4u, prD.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(1u, prAD.GetPageInfos().size());
   EXPECT_EQ(4u, prAD.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(2u, prBD.GetPageInfos().size());
   EXPECT_EQ(32u, prBD.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(8u, prBD.GetPageInfos()[1].GetNElements());
   EXPECT_EQ(2u, prCD.GetPageInfos().size());
   EXPECT_EQ(20u, prCD.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(4u, prCD.GetPageInfos()[1].GetNElements());
   EXPECT_EQ(3u, prDD.GetPageInfos().size());
   EXPECT_EQ(16u, prDD.GetPageInfos()[0].GetNElements());
   EXPECT_EQ(16u, prDD.GetPageInfos()[1].GetNElements());
   EXPECT_EQ(7u, prDD.GetPageInfos()[2].GetNElements());
}

#ifdef R__HAS_DAVIX
TEST(RNTuple, OpenHTTP)
{
   std::unique_ptr<TFile> file(TFile::Open("http://root.cern/files/tutorials/ntpl004_dimuon_v1.root"));
   auto Events = std::unique_ptr<ROOT::RNTuple>(file->Get<ROOT::RNTuple>("Events"));
   auto model = RNTupleModel::Create();
   model->MakeField<ROOT::RNTupleCardinality<std::uint32_t>>("nMuon");
   auto reader = RNTupleReader::Open(std::move(model), *Events);
   reader->LoadEntry(0);
}
#endif

TEST(RNTuple, TMemFile)
{
   TMemFile file("memfile.root", "RECREATE");

   {
      auto model = RNTupleModel::Create();
      *model->MakeField<float>("pt") = 42.0;

      auto writer = RNTupleWriter::Append(std::move(model), "ntpl", file);
      writer->Fill();
   }

   auto ntpl = std::unique_ptr<ROOT::RNTuple>(file.Get<ROOT::RNTuple>("ntpl"));
   auto reader = RNTupleReader::Open(*ntpl);
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
      options.SetEnableSamePageMerging(false);
      options.SetMaxUnzippedPageSize(32 * 1024);
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

   std::vector<std::pair<ROOT::DescriptorId_t, std::int64_t>> pagePositions;
   std::size_t num_columns = 10;
   const auto &cluster0 = ntupleBuf->GetDescriptor().GetClusterDescriptor(0);
   for (std::size_t i = 0; i < num_columns; i++) {
      const auto &columnPages = cluster0.GetPageRange(i);
      for (const auto &page : columnPages.GetPageInfos()) {
         pagePositions.push_back(std::make_pair(i, page.GetLocator().GetPosition<std::uint64_t>()));
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

         auto model = ROOT::RNTupleModel::Create();
         *model->MakeField<float>("pt") = 42.0;

         RNTupleWriteOptions options;
         options.SetInitialUnzippedPageSize(4);
         options.SetMaxUnzippedPageSize(8);
         options.SetUseBufferedWrite(true);
         auto writer = RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath(), options);
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
   options.SetInitialUnzippedPageSize(8);
   options.SetMaxUnzippedPageSize(16);

   {
      std::unique_ptr<RPageSink> sink(new RPageSinkMock(options));
      auto &counters = static_cast<RPageSinkMock *>(sink.get())->fCounters;

      auto model = RNTupleModel::Create();
      auto u64Field = model->MakeField<std::uint64_t>("u64");
      auto u32Field = model->MakeField<std::uint16_t>("u32");
      auto strField = model->MakeField<std::string>("str");
      auto ntuple =
         ROOT::Internal::CreateRNTupleWriter(std::move(model), std::make_unique<RPageSinkBuf>(std::move(sink)));
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
      auto ntuple =
         ROOT::Internal::CreateRNTupleWriter(std::move(model), std::make_unique<RPageSinkBuf>(std::move(sink)));
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
      model->MakeField<float>("pt");
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
      auto wrPt = model->MakeField<float>("pt");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      *wrPt = 42.0;
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
   auto ntuple = ROOT::Internal::CreateRNTupleWriter(std::move(model), std::move(sink));

   *wrPt = 42.0;
   ntuple->Fill();
   EXPECT_EQ(ntuple->GetNEntries(), 1);
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
         ntupleComp->GetModel().GetConstField(std::to_string(i));
         ntupleUcmp->GetModel().GetConstField(std::to_string(i));
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
   optionsComp.SetMaxUnzippedPageSize(kMaxKeySize * 10);
   RNTupleWriteOptionsManip::SetMaxKeySize(optionsComp, kMaxKeySize);
   auto optionsUcmp = optionsComp.Clone();
   optionsUcmp->SetCompression(0);

   {
      auto modelUcmp = RNTupleModel::Create();
      auto modelComp = RNTupleModel::Create();
      auto fU = modelUcmp->MakeField<double>("f");
      auto iU = modelUcmp->MakeField<std::uint32_t>("i");
      auto fC = modelComp->MakeField<double>("f");
      auto iC = modelComp->MakeField<std::uint32_t>("i");
      auto writerUcmp =
         RNTupleWriter::Recreate(std::move(modelUcmp), "myNTuple", fileGuardUcmp.GetPath(), *optionsUcmp);
      auto writerComp = RNTupleWriter::Recreate(std::move(modelComp), "myNTuple", fileGuardComp.GetPath(), optionsComp);
      TRandom3 rnd(42);
      for (int i = 0; i < 100000; ++i) {
         *fC = rnd.Rndm() * std::numeric_limits<double>::max();
         *iC = rnd.Integer(std::numeric_limits<std::int32_t>::max());
         *fU = rnd.Rndm() * std::numeric_limits<double>::max();
         *iU = rnd.Integer(std::numeric_limits<std::int32_t>::max());
         writerComp->Fill();
         writerUcmp->Fill();
      }
   }

   {
      auto modelUcmp = RNTupleModel::Create();
      auto modelComp = RNTupleModel::Create();
      auto fU = modelUcmp->MakeField<double>("f");
      auto iU = modelUcmp->MakeField<std::uint32_t>("i");
      auto fC = modelComp->MakeField<double>("f");
      auto iC = modelComp->MakeField<std::uint32_t>("i");
      auto ntupleUcmp = RNTupleReader::Open(std::move(modelUcmp), "myNTuple", fileGuardUcmp.GetPath());
      auto ntupleComp = RNTupleReader::Open(std::move(modelComp), "myNTuple", fileGuardComp.GetPath());

      // Verify that the pages are larger than maxKeySize
      EXPECT_GT(ntupleComp->GetDescriptor()
                   .GetClusterDescriptor(0)
                   .GetPageRange(0)
                   .GetPageInfos()[0]
                   .GetLocator()
                   .GetNBytesOnStorage(),
                kMaxKeySize);
      EXPECT_GT(ntupleUcmp->GetDescriptor()
                   .GetClusterDescriptor(0)
                   .GetPageRange(0)
                   .GetPageInfos()[0]
                   .GetLocator()
                   .GetNBytesOnStorage(),
                kMaxKeySize);

      TRandom3 rnd(42);
      for (int i = 0; i < 100000; ++i) {
         ntupleUcmp->LoadEntry(i);
         ntupleComp->LoadEntry(i);
         double valC = *fC;
         double expectC = rnd.Rndm() * std::numeric_limits<double>::max();
         std::int32_t valIC = *iC;
         std::int32_t expectIC = rnd.Integer(std::numeric_limits<std::int32_t>::max());
         double valU = *fU;
         double expectU = rnd.Rndm() * std::numeric_limits<double>::max();
         std::int32_t valIU = *iU;
         std::int32_t expectIU = rnd.Integer(std::numeric_limits<std::int32_t>::max());
         EXPECT_DOUBLE_EQ(valC, expectC);
         EXPECT_DOUBLE_EQ(valIC, expectIC);
         EXPECT_DOUBLE_EQ(valU, expectU);
         EXPECT_DOUBLE_EQ(valIU, expectIU);
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
   optionsComp.SetInitialUnzippedPageSize(128);
   optionsComp.SetMaxUnzippedPageSize(128);
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

TEST(RPageSink, SamePageMerging)
{
   FileRaii fileGuard("test_same_page_merging.root");

   for (auto enable : {true, false}) {
      auto model = RNTupleModel::Create();
      *model->MakeField<float>("px") = 1.0;
      *model->MakeField<float>("py") = 1.0;
      RNTupleWriteOptions options;
      options.SetEnableSamePageMerging(enable);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      writer->Fill();
      writer.reset();

      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      EXPECT_EQ(1u, reader->GetNEntries());

      const auto &desc = reader->GetDescriptor();
      const auto pxColId = desc.FindPhysicalColumnId(desc.FindFieldId("px"), 0, 0);
      const auto pyColId = desc.FindPhysicalColumnId(desc.FindFieldId("py"), 0, 0);
      const auto clusterId = desc.FindClusterId(pxColId, 0);
      const auto &clusterDesc = desc.GetClusterDescriptor(clusterId);
      EXPECT_EQ(enable, clusterDesc.GetPageRange(pxColId).Find(0).GetLocator().GetPosition<std::uint64_t>() ==
                           clusterDesc.GetPageRange(pyColId).Find(0).GetLocator().GetPosition<std::uint64_t>());

      auto viewPx = reader->GetView<float>("px");
      auto viewPy = reader->GetView<float>("py");
      EXPECT_FLOAT_EQ(1.0, viewPx(0));
      EXPECT_FLOAT_EQ(1.0, viewPy(0));
   }
}

TEST(RPageSinkFile, StreamerInfo)
{
   FileRaii fileGuard("test_ntuple_page_sink_file_streamer_info.ntuple");

   auto model = RNTupleModel::Create();
   model->MakeField<CustomStruct>("f1");
   model->AddField(std::make_unique<ROOT::RStreamerField>("f2", "StructWithArrays"));
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   writer->Fill(); // need one entry to trigger streamer info record for streamer field
   writer.reset();

   auto file = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str()));
   bool found[2] = {false, false};
   for (auto info : TRangeDynCast<TVirtualStreamerInfo>(*file->GetStreamerInfoList())) {
      if (strcmp(info->GetName(), "CustomStruct") == 0)
         found[0] = true;
      if (strcmp(info->GetName(), "StructWithArrays") == 0)
         found[1] = true;
      if (found[0] && found[1])
         return;
   }
   FAIL() << "not all streamer infos found! ";
}
