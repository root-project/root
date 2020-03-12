#include "gtest/gtest.h"

#include <ROOT/RCluster.hxx>

#include <memory>
#include <utility>

using RHeapCluster = ROOT::Experimental::Detail::RHeapCluster;
using ROnDiskPage = ROOT::Experimental::Detail::ROnDiskPage;

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
