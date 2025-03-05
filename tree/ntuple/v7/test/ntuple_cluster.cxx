#include "ntuple_test.hxx"
#include "gtest/gtest.h"

#include <ROOT/RCluster.hxx>
#include <ROOT/RClusterPool.hxx>
#include <ROOT/RColumn.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>
#ifdef R__USE_IMT
#include <TROOT.h>
#include <ROOT/TThreadExecutor.hxx>
#endif

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

using ROOT::Experimental::RNTupleDescriptor;
using ROOT::Experimental::Internal::RCluster;
using ROOT::Experimental::Internal::RClusterPool;
using ROOT::Experimental::Internal::ROnDiskPage;
using ROOT::Experimental::Internal::RPageSource;
using ROOT::Internal::RPageRef;

namespace {

/**
 * Used to track LoadClusters calls triggered by ClusterPool::GetCluster
 */
class RPageSourceMock : public RPageSource {
protected:
   void LoadStructureImpl() final {}
   RNTupleDescriptor AttachImpl(RNTupleSerializer::EDescriptorDeserializeMode) final { return RNTupleDescriptor(); }
   std::unique_ptr<RPageSource> CloneImpl() const final { return nullptr; }
   RPageRef LoadPageImpl(ColumnHandle_t, const RClusterInfo &, ROOT::NTupleSize_t) final { return RPageRef(); }

public:
   /// Records the cluster IDs requests by LoadClusters() calls
   std::vector<ROOT::DescriptorId_t> fReqsClusterIds;
   std::vector<ROOT::Experimental::Internal::RCluster::ColumnSet_t> fReqsColumns;

   RPageSourceMock() : RPageSource("test", RNTupleReadOptions())
   {
      ROOT::Experimental::Internal::RNTupleDescriptorBuilder descBuilder;
      descBuilder.SetNTuple("ntpl", "");
      for (unsigned i = 0; i <= 5; ++i) {
         descBuilder.AddCluster(ROOT::Experimental::Internal::RClusterDescriptorBuilder()
                                   .ClusterId(i)
                                   .FirstEntryIndex(i)
                                   .NEntries(1)
                                   .MoveDescriptor()
                                   .Unwrap());
      }
      ROOT::Experimental::Internal::RClusterGroupDescriptorBuilder cgBuilder;
      cgBuilder.ClusterGroupId(0).MinEntry(0).EntrySpan(6).NClusters(6);
      cgBuilder.AddSortedClusters({0, 1, 2, 3, 4, 5});
      descBuilder.AddClusterGroup(cgBuilder.MoveDescriptor().Unwrap());
      auto descriptorGuard = GetExclDescriptorGuard();
      descriptorGuard.MoveIn(descBuilder.MoveDescriptor());
   }
   void LoadSealedPage(ROOT::DescriptorId_t, ROOT::RNTupleLocalIndex, RSealedPage &) final {}
   std::vector<std::unique_ptr<RCluster>> LoadClusters(std::span<RCluster::RKey> clusterKeys) final
   {
      std::vector<std::unique_ptr<RCluster>> result;
      for (auto key : clusterKeys) {
         fReqsClusterIds.emplace_back(key.fClusterId);
         fReqsColumns.emplace_back(key.fPhysicalColumnSet);
         auto cluster = std::make_unique<RCluster>(key.fClusterId);
         auto pageMap = std::make_unique<ROOT::Experimental::Internal::ROnDiskPageMap>();
         for (auto colId : key.fPhysicalColumnSet) {
            pageMap->Register(ROnDiskPage::Key(colId, 0), ROnDiskPage(nullptr, 0));
            cluster->SetColumnAvailable(colId);
         }
         cluster->Adopt(std::move(pageMap));
         result.emplace_back(std::move(cluster));
      }
      return result;
   }
};

} // anonymous namespace

TEST(Cluster, Allocate)
{
   auto cluster = new ROOT::Experimental::Internal::ROnDiskPageMapHeap(nullptr);
   delete cluster;

   cluster = new ROOT::Experimental::Internal::ROnDiskPageMapHeap(std::make_unique<unsigned char[]>(1));
   delete cluster;
}

TEST(Cluster, Basics)
{
   auto memory = new unsigned char[3];
   auto pageMap =
      std::make_unique<ROOT::Experimental::Internal::ROnDiskPageMapHeap>(std::unique_ptr<unsigned char[]>(memory));
   pageMap->Register(ROnDiskPage::Key(5, 0), ROnDiskPage(&memory[0], 1));
   pageMap->Register(ROnDiskPage::Key(5, 1), ROnDiskPage(&memory[1], 2));
   auto cluster = std::make_unique<RCluster>(0);
   cluster->Adopt(std::move(pageMap));
   cluster->SetColumnAvailable(5);

   EXPECT_EQ(nullptr, cluster->GetOnDiskPage(ROnDiskPage::Key(5, 2)));
   EXPECT_EQ(nullptr, cluster->GetOnDiskPage(ROnDiskPage::Key(4, 0)));
   auto onDiskPage = cluster->GetOnDiskPage(ROnDiskPage::Key(5, 0));
   EXPECT_EQ(&memory[0], onDiskPage->GetAddress());
   EXPECT_EQ(1U, onDiskPage->GetSize());
   onDiskPage = cluster->GetOnDiskPage(ROnDiskPage::Key(5, 1));
   EXPECT_EQ(&memory[1], onDiskPage->GetAddress());
   EXPECT_EQ(2U, onDiskPage->GetSize());
}

TEST(Cluster, AdoptPageMaps)
{
   auto mem1 = new unsigned char[3];
   auto pageMap1 =
      std::make_unique<ROOT::Experimental::Internal::ROnDiskPageMapHeap>(std::unique_ptr<unsigned char[]>(mem1));
   pageMap1->Register(ROnDiskPage::Key(5, 0), ROnDiskPage(&mem1[0], 1));
   pageMap1->Register(ROnDiskPage::Key(5, 1), ROnDiskPage(&mem1[1], 2));
   // Column 5 is in both mem1 and mem2 but that should not hurt
   auto mem2 = new unsigned char[4];
   auto pageMap2 =
      std::make_unique<ROOT::Experimental::Internal::ROnDiskPageMapHeap>(std::unique_ptr<unsigned char[]>(mem2));
   pageMap2->Register(ROnDiskPage::Key(5, 0), ROnDiskPage(&mem2[0], 1));
   pageMap2->Register(ROnDiskPage::Key(5, 1), ROnDiskPage(&mem2[1], 2));
   pageMap2->Register(ROnDiskPage::Key(6, 0), ROnDiskPage(&mem2[3], 1));

   auto cluster = std::make_unique<RCluster>(0);
   cluster->Adopt(std::move(pageMap1));
   cluster->Adopt(std::move(pageMap2));
   cluster->SetColumnAvailable(5);
   cluster->SetColumnAvailable(6);

   EXPECT_EQ(3U, cluster->GetNOnDiskPages());
   EXPECT_TRUE(cluster->ContainsColumn(5));
   EXPECT_TRUE(cluster->ContainsColumn(6));
   EXPECT_FALSE(cluster->ContainsColumn(7));

   auto onDiskPage = cluster->GetOnDiskPage(ROnDiskPage::Key(6, 0));
   EXPECT_EQ(&mem2[3], onDiskPage->GetAddress());
   EXPECT_EQ(1U, onDiskPage->GetSize());
   onDiskPage = cluster->GetOnDiskPage(ROnDiskPage::Key(5, 0));
   EXPECT_TRUE((onDiskPage->GetAddress() == &mem1[0]) || (onDiskPage->GetAddress() == &mem2[0]));
   EXPECT_EQ(1U, onDiskPage->GetSize());
   onDiskPage = cluster->GetOnDiskPage(ROnDiskPage::Key(5, 1));
   EXPECT_TRUE((onDiskPage->GetAddress() == &mem1[1]) || (onDiskPage->GetAddress() == &mem2[1]));
   EXPECT_EQ(2U, onDiskPage->GetSize());
}

TEST(Cluster, AdoptClusters)
{
   auto mem1 = new unsigned char[3];
   auto pageMap1 =
      std::make_unique<ROOT::Experimental::Internal::ROnDiskPageMapHeap>(std::unique_ptr<unsigned char[]>(mem1));
   pageMap1->Register(ROnDiskPage::Key(5, 0), ROnDiskPage(&mem1[0], 1));
   pageMap1->Register(ROnDiskPage::Key(5, 1), ROnDiskPage(&mem1[1], 2));
   auto cluster1 = std::make_unique<RCluster>(0);
   cluster1->Adopt(std::move(pageMap1));
   cluster1->SetColumnAvailable(5);

   // Column 5 is in both clusters but that should not hurt
   auto mem2 = new unsigned char[4];
   auto pageMap2 =
      std::make_unique<ROOT::Experimental::Internal::ROnDiskPageMapHeap>(std::unique_ptr<unsigned char[]>(mem2));
   pageMap2->Register(ROnDiskPage::Key(5, 0), ROnDiskPage(&mem2[0], 1));
   pageMap2->Register(ROnDiskPage::Key(5, 1), ROnDiskPage(&mem2[1], 2));
   pageMap2->Register(ROnDiskPage::Key(6, 0), ROnDiskPage(&mem2[3], 1));
   auto cluster2 = std::make_unique<RCluster>(0);
   cluster2->Adopt(std::move(pageMap2));
   cluster2->SetColumnAvailable(5);
   cluster2->SetColumnAvailable(6);

   cluster2->Adopt(std::move(*cluster1));

   EXPECT_EQ(3U, cluster2->GetNOnDiskPages());
   EXPECT_TRUE(cluster2->ContainsColumn(5));
   EXPECT_TRUE(cluster2->ContainsColumn(6));
   EXPECT_FALSE(cluster2->ContainsColumn(7));

   auto onDiskPage = cluster2->GetOnDiskPage(ROnDiskPage::Key(6, 0));
   EXPECT_EQ(&mem2[3], onDiskPage->GetAddress());
   EXPECT_EQ(1U, onDiskPage->GetSize());
   onDiskPage = cluster2->GetOnDiskPage(ROnDiskPage::Key(5, 0));
   EXPECT_TRUE((onDiskPage->GetAddress() == &mem1[0]) || (onDiskPage->GetAddress() == &mem2[0]));
   EXPECT_EQ(1U, onDiskPage->GetSize());
   onDiskPage = cluster2->GetOnDiskPage(ROnDiskPage::Key(5, 1));
   EXPECT_TRUE((onDiskPage->GetAddress() == &mem1[1]) || (onDiskPage->GetAddress() == &mem2[1]));
   EXPECT_EQ(2U, onDiskPage->GetSize());
}

TEST(ClusterPool, GetClusterBasics)
{
   RPageSourceMock p1;
   RClusterPool c1(p1, 1);
   c1.GetCluster(3, {0});
   c1.WaitForInFlightClusters();
   ASSERT_EQ(2U, p1.fReqsClusterIds.size());
   EXPECT_EQ(3U, p1.fReqsClusterIds[0]);
   EXPECT_EQ(4U, p1.fReqsClusterIds[1]);
   EXPECT_EQ(RCluster::ColumnSet_t({0}), p1.fReqsColumns[0]);
   EXPECT_EQ(RCluster::ColumnSet_t({0}), p1.fReqsColumns[1]);

   RPageSourceMock p2;
   {
      RClusterPool c2(p2, 2);
      c2.GetCluster(0, {0});
      c2.WaitForInFlightClusters();
   }
   ASSERT_EQ(4U, p2.fReqsClusterIds.size());
   EXPECT_EQ(0U, p2.fReqsClusterIds[0]);
   EXPECT_EQ(1U, p2.fReqsClusterIds[1]);
   EXPECT_EQ(2U, p2.fReqsClusterIds[2]);
   EXPECT_EQ(3U, p2.fReqsClusterIds[3]);
   EXPECT_EQ(RCluster::ColumnSet_t({0}), p2.fReqsColumns[0]);
   EXPECT_EQ(RCluster::ColumnSet_t({0}), p2.fReqsColumns[1]);
   EXPECT_EQ(RCluster::ColumnSet_t({0}), p2.fReqsColumns[2]);
   EXPECT_EQ(RCluster::ColumnSet_t({0}), p2.fReqsColumns[3]);

   RPageSourceMock p3;
   {
      RClusterPool c3(p3, 2);
      c3.GetCluster(0, {0});
      c3.GetCluster(1, {0});
      c3.WaitForInFlightClusters();
   }
   ASSERT_EQ(4U, p3.fReqsClusterIds.size());
   EXPECT_EQ(0U, p3.fReqsClusterIds[0]);
   EXPECT_EQ(1U, p3.fReqsClusterIds[1]);
   EXPECT_EQ(2U, p3.fReqsClusterIds[2]);
   EXPECT_EQ(3U, p3.fReqsClusterIds[3]);

   RPageSourceMock p4;
   {
      RClusterPool c4(p4, 3);
      c4.GetCluster(2, {0});
      c4.WaitForInFlightClusters();
   }
   ASSERT_EQ(4U, p4.fReqsClusterIds.size());
   EXPECT_EQ(2U, p4.fReqsClusterIds[0]);
   EXPECT_EQ(3U, p4.fReqsClusterIds[1]);
   EXPECT_EQ(4U, p4.fReqsClusterIds[2]);
   EXPECT_EQ(5U, p4.fReqsClusterIds[3]);
}

TEST(ClusterPool, SetEntryRange)
{
   RPageSourceMock p1;
   p1.SetEntryRange({0, 6});
   RClusterPool c1(p1, 1);
   c1.GetCluster(3, {0});
   c1.WaitForInFlightClusters();
   ASSERT_EQ(2U, p1.fReqsClusterIds.size());
   EXPECT_EQ(3U, p1.fReqsClusterIds[0]);
   EXPECT_EQ(4U, p1.fReqsClusterIds[1]);

   RPageSourceMock p2;
   p2.SetEntryRange({3, 1});
   RClusterPool c2(p2, 1);
   c2.GetCluster(3, {0});
   c2.WaitForInFlightClusters();
   ASSERT_EQ(1U, p2.fReqsClusterIds.size());
   EXPECT_EQ(3U, p2.fReqsClusterIds[0]);

   RPageSourceMock p3;
   p3.SetEntryRange({0, 1});
   RClusterPool c3(p3, 1);
   c3.GetCluster(3, {0});
   c3.WaitForInFlightClusters();
   ASSERT_EQ(1U, p3.fReqsClusterIds.size());
   EXPECT_EQ(3U, p3.fReqsClusterIds[0]);

   RPageSourceMock p4;
   p4.SetEntryRange({0, 3});
   RClusterPool c4(p4, 2);
   c4.GetCluster(0, {0});
   c4.WaitForInFlightClusters();
   ASSERT_EQ(3U, p4.fReqsClusterIds.size());
   EXPECT_EQ(0U, p4.fReqsClusterIds[0]);
   EXPECT_EQ(1U, p4.fReqsClusterIds[1]);
   EXPECT_EQ(2U, p4.fReqsClusterIds[2]);
}

TEST(ClusterPool, GetClusterIncrementally)
{
   RPageSourceMock p1;
   RClusterPool c1(p1, 1);
   c1.GetCluster(3, {0});
   c1.WaitForInFlightClusters();
   ASSERT_EQ(2U, p1.fReqsClusterIds.size());
   EXPECT_EQ(3U, p1.fReqsClusterIds[0]);
   EXPECT_EQ(RCluster::ColumnSet_t({0}), p1.fReqsColumns[0]);

   c1.GetCluster(3, {1});
   c1.WaitForInFlightClusters();
   ASSERT_EQ(4U, p1.fReqsClusterIds.size());
   EXPECT_EQ(3U, p1.fReqsClusterIds[2]);
   EXPECT_EQ(RCluster::ColumnSet_t({1}), p1.fReqsColumns[2]);
}

TEST(PageStorageFile, LoadClusters)
{
   FileRaii fileGuard("test_pagestoragefile_loadclusters.root");

   auto modelWrite = ROOT::Experimental::RNTupleModel::Create();
   auto wrPt = modelWrite->MakeField<float>("pt");
   auto wrTag = modelWrite->MakeField<std::int32_t>("tag");

   {
      auto writer = ROOT::Experimental::RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      *wrPt = 42.0;
      writer->Fill();
      writer->CommitCluster();
      *wrPt = 24.0;
      *wrTag = 1;
      writer->Fill();
   }

   ROOT::Experimental::Internal::RPageSourceFile source("myNTuple", fileGuard.GetPath(), RNTupleReadOptions());
   source.Attach();

   ROOT::DescriptorId_t ptId;
   ROOT::DescriptorId_t colId;
   {
      auto descriptorGuard = source.GetSharedDescriptorGuard();
      ptId = descriptorGuard->FindFieldId("pt");
      EXPECT_NE(ROOT::kInvalidDescriptorId, ptId);
      colId = descriptorGuard->FindPhysicalColumnId(ptId, 0, 0);
      EXPECT_NE(ROOT::kInvalidDescriptorId, colId);
   }

   std::vector<ROOT::Experimental::Internal::RCluster::RKey> clusterKeys;
   clusterKeys.push_back({0, {}});
   auto cluster = std::move(source.LoadClusters(clusterKeys)[0]);
   EXPECT_EQ(0U, cluster->GetId());
   EXPECT_EQ(0U, cluster->GetNOnDiskPages());

   auto column = ROOT::Experimental::Internal::RColumn::Create<float>(ROOT::ENTupleColumnType::kReal32, 0, 0);
   column->ConnectPageSource(ptId, source);
   clusterKeys[0].fClusterId = 1;
   clusterKeys[0].fPhysicalColumnSet.insert(colId);
   cluster = std::move(source.LoadClusters(clusterKeys)[0]);
   EXPECT_EQ(1U, cluster->GetId());
   EXPECT_EQ(1U, cluster->GetNOnDiskPages());

   ROnDiskPage::Key key(colId, 0);
   EXPECT_NE(nullptr, cluster->GetOnDiskPage(key));
   clusterKeys.push_back({1, {colId}});
   clusterKeys[0].fClusterId = 0;
   auto clusters = source.LoadClusters(clusterKeys);
   EXPECT_EQ(2U, clusters.size());
   EXPECT_EQ(0U, clusters[0]->GetId());
   EXPECT_EQ(1U, clusters[0]->GetNOnDiskPages());
   EXPECT_EQ(1U, clusters[1]->GetId());
   EXPECT_EQ(1U, clusters[1]->GetNOnDiskPages());
}

#ifdef R__USE_IMT
TEST(PageStorageFile, LoadClustersIMT)
{
   ROOT::EnableImplicitMT(2);

   FileRaii fileGuard("test_pagestoragefile_loadclustersimt.root");

   {
      auto model = ROOT::Experimental::RNTupleModel::Create();
      *model->MakeField<float>("pt") = 42.0;

      auto writer = ROOT::Experimental::RNTupleWriter::Recreate(std::move(model), "myNTuple", fileGuard.GetPath());
      writer->Fill();
   }

   ROOT::TThreadExecutor ex(2);
   ex.Foreach(
      [&]() {
         auto reader = ROOT::Experimental::RNTupleReader::Open("myNTuple", fileGuard.GetPath());
         reader->LoadEntry(0);
      },
      2);

   ROOT::DisableImplicitMT();
}
#endif
