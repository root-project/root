#include <TFile.h>
#include <TInterpreter.h>
#include <TString.h>
#include <TSystem.h>
#include <TTree.h>

#include "gtest/gtest.h"

#include <cstdint>
#include <bitset>
#include <unordered_map>
#include <unordered_set>

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

template <typename T>
T interpreter_get(const char *expr)
{
   T *ptr = reinterpret_cast<T *>(gInterpreter->Calc(expr));
   EXPECT_NE(ptr, nullptr);
   return *ptr;
}

TEST(RNTupleMakeProject, ReadBackTTree)
{
   ASSERT_EQ(gSystem->Load("libttreestltest/libttreestltest"), 0);

   std::unique_ptr<TFile> f{TFile::Open("ntuple_makeproject_stl_example_ttree.root")};
   std::unique_ptr<TTree> t{f->Get<TTree>("events")};

   // Expose the TTree pointer to the interpreter
   gInterpreter->ProcessLine(TString::Format("TTree *t = (TTree *)0x%zx;", reinterpret_cast<std::size_t>(t.get())));

   // Everything involving MySTLEvent lives in the interpreter
   gInterpreter->ProcessLine(R"(

      MySTLEvent *event{nullptr};
      t->SetBranchAddress("test", &event);
      t->GetEntry(0);
   )");

   // Get values out as plain STL types
   auto event_foo = interpreter_get<std::bitset<16>>("&event->foo");
   auto event_bar = interpreter_get<std::vector<int>>("&event->bar");
   auto event_spam = interpreter_get<std::unordered_set<double>>("&event->spam");
   auto event_eggs = interpreter_get<std::unordered_map<int, std::string>>("&event->eggs");
   auto event_strange = interpreter_get<std::unordered_multimap<std::string, std::vector<float>>>("&event->strange");

   std::bitset<16> foo = 0xfa2;
   std::vector<int> bar = {1, 2};
   std::unordered_set<double> spam = {1, 3, 5, 6, 8};
   std::unordered_map<int, std::string> eggs = {{2, "two"}, {4, "four"}};
   std::unordered_multimap<std::string, std::vector<float>> strange = {
      {"one", {1, 2, 3}}, {"one", {4, 5, 6}}, {"two", {7, 8, 9}}};

   check_bitset(event_foo, foo);
   check_vector(event_bar, bar);
   check_unordered_set(event_spam, spam);
   check_unordered_map(event_eggs, eggs);
   check_unordered_multimap(event_strange, strange);
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}
