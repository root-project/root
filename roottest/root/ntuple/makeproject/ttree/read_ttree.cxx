#include <TFile.h>
#include <TTree.h>

#include "libttreestltest/MySTLEvent.h"

#include <bitset>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

int gNFailures = 0;

void ReportFailure(const std::string &what, const std::string &details = "")
{
   ++gNFailures;
   std::cerr << "FAILURE: " << what;
   if (!details.empty())
      std::cerr << ": " << details;
   std::cerr << '\n';
}

std::string SizeMismatch(std::size_t got, std::size_t expected)
{
   return "got size " + std::to_string(got) + ", expected size " + std::to_string(expected);
}

void check_bitset(const std::bitset<16> &a, const std::bitset<16> &b)
{
   if (a != b)
      ReportFailure("bitset mismatch", "got " + a.to_string() + ", expected " + b.to_string());
}

void check_vector(const std::vector<int> &a, const std::vector<int> &b)
{
   if (a.size() != b.size()) {
      ReportFailure("vector size mismatch", SizeMismatch(a.size(), b.size()));
      return;
   }
   for (std::size_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) {
         ReportFailure("vector element mismatch", "index " + std::to_string(i) + ": got " + std::to_string(a[i]) +
                                                     ", expected " + std::to_string(b[i]));
      }
   }
}

void check_unordered_set(const std::unordered_set<double> &a, const std::unordered_set<double> &b)
{
   if (a != b)
      ReportFailure("unordered_set mismatch", SizeMismatch(a.size(), b.size()));
}

void check_unordered_map(const std::unordered_map<int, std::string> &a, const std::unordered_map<int, std::string> &b)
{
   if (a != b)
      ReportFailure("unordered_map mismatch", SizeMismatch(a.size(), b.size()));
}

void check_unordered_multimap(const std::unordered_multimap<std::string, std::vector<float>> &a,
                              const std::unordered_multimap<std::string, std::vector<float>> &b)
{
   if (a != b)
      ReportFailure("unordered_multimap mismatch", SizeMismatch(a.size(), b.size()));
}

void ReadBackTTree()
{
   std::unique_ptr<TFile> f{TFile::Open("ntuple_makeproject_stl_example_ttree.root")};
   if (!f || f->IsZombie()) {
      ReportFailure("cannot open file", "ntuple_makeproject_stl_example_ttree.root");
      return;
   }

   std::unique_ptr<TTree> t{f->Get<TTree>("events")};
   if (!t) {
      ReportFailure("cannot find tree", "events");
      return;
   }

   MySTLEvent *event{nullptr};
   t->SetBranchAddress("test", &event);
   if (t->GetEntry(0) <= 0 || !event) {
      ReportFailure("cannot read entry 0 of branch", "test");
      return;
   }

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

} // anonymous namespace

int main()
{
   try {
      ReadBackTTree();
   } catch (const std::exception &e) {
      std::cerr << "FAILURE: unexpected exception: " << e.what() << '\n';
      return EXIT_FAILURE;
   }

   if (gNFailures > 0) {
      std::cerr << gNFailures << " check(s) failed\n";
      return EXIT_FAILURE;
   }

   std::cout << "All checks passed\n";
   return EXIT_SUCCESS;
}
