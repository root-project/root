#include "gtest/gtest.h"

#include <ROOT/RCluster.hxx>
#include <ROOT/RClusterPool.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageStorage.hxx>
#include <ROOT/RStringView.hxx>

#include <memory>
#include <utility>
#include <vector>

using ClusterSize_t = ROOT::Experimental::ClusterSize_t;
using DescriptorId_t = ROOT::Experimental::DescriptorId_t;
using NTupleSize_t = ROOT::Experimental::NTupleSize_t;
using RCluster = ROOT::Experimental::Detail::RCluster;
using RClusterIndex = ROOT::Experimental::RClusterIndex;
using RClusterPool = ROOT::Experimental::Detail::RClusterPool;
using RHeapCluster = ROOT::Experimental::Detail::RHeapCluster;
using RNTupleDescriptor = ROOT::Experimental::RNTupleDescriptor;
using RNTupleDescriptorBuilder = ROOT::Experimental::RNTupleDescriptorBuilder;
using RNTupleMetrics = ROOT::Experimental::Detail::RNTupleMetrics;
using RNTupleReadOptions = ROOT::Experimental::RNTupleReadOptions;
using RNTupleVersion = ROOT::Experimental::RNTupleVersion;
using ROnDiskPage = ROOT::Experimental::Detail::ROnDiskPage;
using RPage = ROOT::Experimental::Detail::RPage;
using RPageSource = ROOT::Experimental::Detail::RPageSource;

namespace {

class RPageSourceMock : public RPageSource {
private:
   RNTupleMetrics fMetrics;

protected:
   RNTupleDescriptor AttachImpl() final { return RNTupleDescriptor(); }

public:
   /// Records the cluster IDs requests by LoadCluster() calls
   std::vector<DescriptorId_t> fLoadRequests;

   RPageSourceMock() : RPageSource("test", RNTupleReadOptions()), fMetrics("test") {
      RNTupleDescriptorBuilder descBuilder;
      descBuilder.AddCluster(0, RNTupleVersion(), 0, ClusterSize_t(1));
      descBuilder.AddCluster(1, RNTupleVersion(), 1, ClusterSize_t(1));
      descBuilder.AddCluster(2, RNTupleVersion(), 2, ClusterSize_t(1));
      descBuilder.AddCluster(3, RNTupleVersion(), 3, ClusterSize_t(1));
      descBuilder.AddCluster(4, RNTupleVersion(), 4, ClusterSize_t(1));
      fDescriptor = descBuilder.MoveDescriptor();
   }
   std::unique_ptr<RPageSource> Clone() const final { return nullptr; }
   RPage PopulatePage(ColumnHandle_t, NTupleSize_t) final { return RPage(); }
   RPage PopulatePage(ColumnHandle_t, const RClusterIndex &) final { return RPage(); }
   void ReleasePage(RPage &) final {}
   std::unique_ptr<RCluster> LoadCluster(DescriptorId_t clusterId) final {
      fLoadRequests.push_back(clusterId);
      return std::make_unique<RCluster>(nullptr, clusterId);
   }
   RNTupleMetrics &GetMetrics() final { return fMetrics; }
};

} // anonymous namespace


TEST(Cluster, Allocate)
{
   auto cluster = new RHeapCluster(nullptr, 0);
   delete cluster;

   auto memory = new char[1];
   cluster = new RHeapCluster(memory, 0);
   delete cluster;
}


TEST(Cluster, Basics)
{
   auto memory = new char[3];
   auto cluster = std::make_unique<RHeapCluster>(memory, 0);
   cluster->Insert(ROnDiskPage::Key(5, 0), ROnDiskPage(&memory[0], 1));
   cluster->Insert(ROnDiskPage::Key(5, 1), ROnDiskPage(&memory[1], 2));

   EXPECT_EQ(nullptr, cluster->GetOnDiskPage(ROnDiskPage::Key(5, 2)));
   EXPECT_EQ(nullptr, cluster->GetOnDiskPage(ROnDiskPage::Key(4, 0)));
   auto onDiskPage = cluster->GetOnDiskPage(ROnDiskPage::Key(5, 0));
   EXPECT_EQ(&memory[0], onDiskPage->GetAddress());
   EXPECT_EQ(1U, onDiskPage->GetSize());
   onDiskPage = cluster->GetOnDiskPage(ROnDiskPage::Key(5, 1));
   EXPECT_EQ(&memory[1], onDiskPage->GetAddress());
   EXPECT_EQ(2U, onDiskPage->GetSize());
}

TEST(ClusterPool, Windows) {
   EXPECT_DEATH(RClusterPool(nullptr, 0), ".*");
   RClusterPool c1(nullptr, 1);
   EXPECT_EQ(0U, c1.GetWindowPre());
   EXPECT_EQ(1U, c1.GetWindowPost());
   RClusterPool c2(nullptr, 2);
   EXPECT_EQ(0U, c2.GetWindowPre());
   EXPECT_EQ(2U, c2.GetWindowPost());
   RClusterPool c3(nullptr, 3);
   EXPECT_EQ(1U, c3.GetWindowPre());
   EXPECT_EQ(2U, c3.GetWindowPost());
   RClusterPool c5(nullptr, 5);
   EXPECT_EQ(1U, c5.GetWindowPre());
   EXPECT_EQ(4U, c5.GetWindowPost());
   RClusterPool c6(nullptr, 6);
   EXPECT_EQ(2U, c6.GetWindowPre());
   EXPECT_EQ(4U, c6.GetWindowPost());
   RClusterPool c9(nullptr, 9);
   EXPECT_EQ(2U, c9.GetWindowPre());
   EXPECT_EQ(7U, c9.GetWindowPost());
   RClusterPool c10(nullptr, 10);
   EXPECT_EQ(3U, c10.GetWindowPre());
   EXPECT_EQ(7U, c10.GetWindowPost());
   RClusterPool c15(nullptr, 15);
   EXPECT_EQ(3U,  c15.GetWindowPre());
   EXPECT_EQ(12U, c15.GetWindowPost());
   RClusterPool c16(nullptr, 16);
   EXPECT_EQ(4U,  c16.GetWindowPre());
   EXPECT_EQ(12U, c16.GetWindowPost());
}

TEST(ClusterPool, GetCluster) {
   RPageSourceMock p1;
   RClusterPool c1(&p1, 1);
   c1.GetCluster(3);
   ASSERT_EQ(1U, p1.fLoadRequests.size());
   EXPECT_EQ(3U, p1.fLoadRequests[0]);

   RPageSourceMock p2;
   {
      RClusterPool c2(&p2, 2);
      c2.GetCluster(0);
   }
   ASSERT_EQ(2U, p2.fLoadRequests.size());
   EXPECT_EQ(0U, p2.fLoadRequests[0]);
   EXPECT_EQ(1U, p2.fLoadRequests[1]);

   RPageSourceMock p3;
   {
      RClusterPool c3(&p3, 4);
      c3.GetCluster(2);
   }
   ASSERT_EQ(3U, p3.fLoadRequests.size());
   EXPECT_EQ(2U, p3.fLoadRequests[0]);
   EXPECT_EQ(3U, p3.fLoadRequests[1]);
   EXPECT_EQ(4U, p3.fLoadRequests[2]);
}
