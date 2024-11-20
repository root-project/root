#include <TFile.h>
#include <TTree.h>

#include "libttreestltest/MySTLEvent.h"
#include "gtest/gtest.h"

void check_bitset(const std::bitset<16> &a, const std::bitset<16> &b)
{
   EXPECT_EQ(a, b);
}

void check_vector(const std::vector<int> &a, const std::vector<int> &b)
{
   EXPECT_EQ(a.size(), b.size());
   for (std::size_t i = 0; i < a.size(); i++) {
      EXPECT_EQ(a[i], b[i]);
   }
}

void check_unordered_set(const std::unordered_set<double> &a, const std::unordered_set<double> &b)
{
   EXPECT_EQ(a, b);
}

void check_unordered_map(const std::unordered_map<int, std::string> &a, const std::unordered_map<int, std::string> &b)
{
   EXPECT_EQ(a, b);
}

void check_unordered_multimap(const std::unordered_multimap<std::string, std::vector<float>> &a,
                              const std::unordered_multimap<std::string, std::vector<float>> &b)
{
   EXPECT_EQ(a, b);
}

TEST(RNTupleMakeProject, ReadBackTTree)
{
   std::unique_ptr<TFile> f{TFile::Open("ntuple_makeproject_stl_example_ttree.root")};
   std::unique_ptr<TTree> t{f->Get<TTree>("events")};

   MySTLEvent *event{nullptr};
   t->SetBranchAddress("test", &event);
   t->GetEntry(0);

   std::bitset<16> foo = 0xfa2;
   std::vector<int> bar = {1, 2};
   std::unordered_set<double> spam = {1, 3, 5, 6, 8};
   std::unordered_map<int, std::string> eggs = {{2, "two"}, {4, "four"}};
   std::unordered_multimap<std::string, std::vector<float>> strange = {
      {"one", {1, 2, 3}}, {"one", {4, 5, 6}}, {"two", {7, 8, 9}}};

   check_bitset(event->foo, foo);
   check_vector(event->bar, bar);
   check_unordered_set(event->spam, spam);
   check_unordered_map(event->eggs, eggs);
   check_unordered_multimap(event->strange, strange);
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
