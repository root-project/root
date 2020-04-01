#include <algorithm>
#include <atomic>
#include <chrono>
#include <random>
#include <string>
#include <thread>
#include <utility>

#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <ROOT/TTreeProcessorMT.hxx>

#include "gtest/gtest.h"

void WriteFiles(const std::vector<std::string> &treenames, const std::vector<std::string> &filenames)
{
   int v = 0;
   const auto nFiles = filenames.size();
   EXPECT_EQ(nFiles, treenames.size()) << "this should never happen, fix test logic";
   for (auto i = 0u; i < nFiles; ++i) {
      const auto &fname = filenames[i];
      const auto &treename = treenames[i];

      TFile file(fname.c_str(), "recreate");
      TTree t(treename.c_str(), treename.c_str());
      t.Branch("v", &v);
      for (auto e = 0; e < 10; ++e) {
         ++v;
         t.Fill();
      }
      t.Write();
   }
}

void WriteFileManyClusters(unsigned int nevents, const char *treename, const char *filename)
{
   int v = 0;
   TFile file(filename, "recreate");
   TTree t(treename, treename);
   t.Branch("v", &v);
   // t.SetAutoFlush(1);
   for (auto i = 0U; i < nevents; ++i) {
      t.Fill();
      t.FlushBaskets();
   }
   t.Write();
   file.Close();
}

void DeleteFiles(const std::vector<std::string> &filenames)
{
   for (const auto &f : filenames)
      gSystem->Unlink(f.c_str());
}

TEST(TreeProcessorMT, EmptyTChain)
{
   TChain c("mytree");
   auto exceptionFired(false);
   try {
      ROOT::TTreeProcessorMT proc(c);
   } catch (...) {
      exceptionFired = true;
   }
   EXPECT_TRUE(exceptionFired);
}

TEST(TreeProcessorMT, ManyFiles)
{
   const auto nFiles = 100u;
   const std::string treename = "t";
   std::vector<std::string> filenames;
   for (auto i = 0u; i < nFiles; ++i)
      filenames.emplace_back("treeprocmt_manyfiles" + std::to_string(i) + ".root");

   WriteFiles(std::vector<std::string>(nFiles, treename), filenames);

   std::atomic_int sum(0);
   std::atomic_int count(0);
   auto sumValues = [&sum, &count](TTreeReader &r) {
      TTreeReaderValue<int> v(r, "v");
      std::random_device seed;
      std::default_random_engine eng(seed());
      std::uniform_int_distribution<> rand(1, 100);
      while (r.Next()) {
         std::this_thread::sleep_for(std::chrono::nanoseconds(rand(eng)));
         sum += *v;
         ++count;
      }
   };

   // TTreeProcMT requires a vector<string_view>
   std::vector<std::string_view> fnames;
   for (const auto &f : filenames)
      fnames.emplace_back(f);

   ROOT::TTreeProcessorMT proc(fnames, treename);
   proc.Process(sumValues);

   EXPECT_EQ(count.load(), int(nFiles * 10)); // 10 entries per file
   EXPECT_EQ(sum.load(), 500500);             // sum of [1..nFiles*nEntriesPerFile] inclusive

   DeleteFiles(filenames);
}

TEST(TreeProcessorMT, TreesWithDifferentNamesChainCtor)
{
   const std::vector<std::string> treenames{"t0","t1","t2"};
   const std::vector<std::string> filenames{"treeprocmt_chainctor0.root", "treeprocmt_chainctor1.root",
                                            "treeprocmt_chainctor2.root"};

   WriteFiles(treenames, filenames);

   std::atomic_int sum(0);
   std::atomic_int count(0);
   auto sumValues = [&sum, &count](TTreeReader &r) {
      TTreeReaderValue<int> v(r, "v");
      while (r.Next()) {
         sum += *v;
         ++count;
      }
   };

   // TTreeProcMT requires a vector<string_view>
   TChain chain;
   const auto nFiles = filenames.size();
   for (auto i = 0u; i < nFiles; ++i) {
      const auto n = std::to_string(i);
      const auto full_fname = "treeprocmt_chainctor" + n + ".root/t" + n;
      chain.Add(full_fname.c_str());
   }

   // tree names are inferred from the files
   ROOT::TTreeProcessorMT proc(chain);
   proc.Process(sumValues);

   EXPECT_EQ(count.load(), int(nFiles * 10)); // 10 entries per file
   EXPECT_EQ(sum.load(), 465);                // sum of [1..nFiles*nEntriesPerFile] inclusive

   DeleteFiles(filenames);
}

TEST(TreeProcessorMT, TreesWithDifferentNamesVecCtor)
{
   const std::vector<std::string> treenames{"t0","t1","t2"};
   const std::vector<std::string> filenames{"treeprocmt_vecctor0.root", "treeprocmt_vecctor1.root",
                                            "treeprocmt_vecctor2.root"};

   WriteFiles(treenames, filenames);

   std::atomic_int sum(0);
   std::atomic_int count(0);
   auto sumValues = [&sum, &count](TTreeReader &r) {
      TTreeReaderValue<int> v(r, "v");
      while (r.Next()) {
         sum += *v;
         ++count;
      }
   };

   // TTreeProcMT requires a vector<string_view>
   std::vector<std::string_view> fnames;
   for (const auto &f : filenames)
      fnames.emplace_back(f);

   ROOT::TTreeProcessorMT proc(fnames);
   proc.Process(sumValues);

   EXPECT_EQ(count.load(), int(filenames.size() * 10)); // 10 entries per file
   EXPECT_EQ(sum.load(), 465); // sum of [1..nFiles*nEntriesPerFile] inclusive

   DeleteFiles(filenames);
}

TEST(TreeProcessorMT, TreeInSubDirectory)
{
   auto filename = "fileTreeInSubDirectory.root";
   auto procLambda = [](TTreeReader &r) {
      while (r.Next())
         ;
   };

   {
      TFile f(filename, "RECREATE");
      auto dir0 = f.mkdir("dir0");
      dir0->cd();
      auto dir1 = dir0->mkdir("dir1");
      dir1->cd();
      TTree t("tree", "tree");
      t.Write();
   }

   ROOT::EnableThreadSafety();

   auto fullPath = "dir0/dir1/tree";

   // With a TTree
   TFile f(filename);
   auto t = (TTree *)f.Get(fullPath);
   ROOT::TTreeProcessorMT tp(*t);
   tp.Process(procLambda);

   // With a TChain
   std::string chainElementName = filename;
   chainElementName += "/";
   chainElementName += fullPath;
   TChain chain;
   chain.Add(chainElementName.c_str());
   ROOT::TTreeProcessorMT tpc(chain);
   tpc.Process(procLambda);

   gSystem->Unlink(filename);
}

TEST(TreeProcessorMT, LimitNTasks_CheckEntries)
{
   const auto nEvents = 991;
   const auto filename = "TreeProcessorMT_LimitNTasks_CheckEntries.root";
   const auto treename = "t";
   WriteFileManyClusters(nEvents, treename, filename);
   auto nTasks = 0U;
   std::map<unsigned int, unsigned int> nEntriesCountsMap;
   std::mutex theMutex;
   auto f = [&](TTreeReader &t) {
      auto nentries = 0U;
      while (t.Next())
         nentries++;
      std::lock_guard<std::mutex> lg(theMutex);
      nTasks++;
      if (!nEntriesCountsMap.insert({nentries, 1U}).second) {
         nEntriesCountsMap[nentries]++;
      }
   };

   ROOT::DisableImplicitMT();
   ROOT::EnableImplicitMT(4);

   ROOT::TTreeProcessorMT p(filename, treename);
   p.Process(f);

   EXPECT_EQ(nTasks, 96U) << "Wrong number of tasks generated!\n";
   EXPECT_EQ(nEntriesCountsMap[10], 65U) << "Wrong number of tasks with 10 clusters each!\n";
   EXPECT_EQ(nEntriesCountsMap[11], 31U) << "Wrong number of tasks with 11 clusters each!\n";

   gSystem->Unlink(filename);
   ROOT::DisableImplicitMT();
}

void CheckClusters(std::vector<std::pair<Long64_t, Long64_t>> &clusters, Long64_t entries)
{
   using R = std::pair<Long64_t, Long64_t>;
   // sort them
   std::sort(clusters.begin(), clusters.end(), [](const R &p1, const R &p2) { return p1.first < p2.first; });
   // check each end corresponds to the next start
   const auto nClusters = clusters.size();
   for (auto i = 0u; i < nClusters - 1; ++i)
      EXPECT_EQ(clusters[i].second, clusters[i + 1].first);
   // check start and end correspond to true start and end
   EXPECT_EQ(clusters.front().first, 0LL);
   EXPECT_EQ(clusters.back().second, entries);
}

TEST(TreeProcessorMT, LimitNTasks_CheckClusters)
{
   const auto nEvents = 156;
   const auto filename = "TreeProcessorMT_LimitNTasks_CheckClusters.root";
   const auto treename = "t";
   WriteFileManyClusters(nEvents, treename, filename);

   std::mutex m;
   std::vector<std::pair<Long64_t, Long64_t>> clusters;
   auto get_clusters = [&m, &clusters](TTreeReader &t) {
      std::lock_guard<std::mutex> l(m);
      clusters.emplace_back(t.GetEntriesRange());
   };

   for (auto nThreads = 0; nThreads <= 4; ++nThreads) {
      ROOT::DisableImplicitMT();
      ROOT::EnableImplicitMT(nThreads);

      ROOT::TTreeProcessorMT p(filename, treename);
      p.Process(get_clusters);

      CheckClusters(clusters, nEvents);
      clusters.clear();
   }

   gSystem->Unlink(filename);
   ROOT::DisableImplicitMT();
}

#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
TEST(TreeProcessorMT, PathName)
{
   auto fname = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/ZZTo4mu.root";
   auto f = std::unique_ptr<TFile>(TFile::Open(fname));
   ASSERT_TRUE(f != nullptr) << "Could not open remote file\n";
   auto tree = f->Get<TTree>("Events");
   ROOT::TTreeProcessorMT p(*tree);
   std::atomic<unsigned int> n(0U);
   auto func = [&n](TTreeReader &t) {
      while (t.Next())
         n++;
   };
   p.Process(func);
   EXPECT_EQ(n.load(), 1499064U) << "Wrong number of events processed!\n";
}
#endif

TEST(TreeProcessorMT, TreeWithFriendTree)
{
   std::vector<std::string> fileNames = {"TreeWithFriendTree_Tree.root", "TreeWithFriendTree_Friend.root"};
   for (auto &name : fileNames) {
      TFile f(name.c_str(), "RECREATE");
      TTree t("treeName", "treeTitle");
      t.Write();
      f.Close();
   }

   ROOT::EnableImplicitMT(1);
   auto procLambda = [](TTreeReader &r) {
      while (r.Next())
         ;
   };

   auto f1 = TFile::Open(fileNames[0].c_str());
   auto t1 = (TTree *)f1->Get("treeName");

   auto f2 = TFile::Open(fileNames[1].c_str());
   auto t2 = (TTree *)f2->Get("treeName");

   t1->AddFriend(t2);

   ROOT::TTreeProcessorMT tp(*t1);
   tp.Process(procLambda);

   DeleteFiles(fileNames);
}

TEST(TreeProcessorMT, ChainWithFriendChain)
{
   std::vector<std::string> fileNames = {"ChainWithFriendChain_Tree1.root", "ChainWithFriendChain_Tree2.root", "ChainWithFriendChain_Friend1.root", "ChainWithFriendChain_Friend2.root"};
   for (auto &name : fileNames) {
      TFile f(name.c_str(), "RECREATE");
      TTree t("treeName", "treeTitle");
      t.Write();
      f.Close();
   }

   ROOT::EnableImplicitMT(1);
   auto procLambda = [](TTreeReader &r) {
      while (r.Next())
         ;
   };

   // Version 1: Use tree name in constructor
   TChain c1("treeName");
   c1.AddFile(fileNames[0].c_str());
   c1.AddFile(fileNames[1].c_str());

   TChain c2("treeName");
   c2.AddFile(fileNames[2].c_str());
   c2.AddFile(fileNames[3].c_str());

   c1.AddFriend(&c2);

   ROOT::TTreeProcessorMT tp1(c1);
   tp1.Process(procLambda);

   // Version 2: Use tree name in AddFile
   TChain c3;
   c3.AddFile((fileNames[0] + "/treeName").c_str());
   c3.AddFile((fileNames[1] + "/treeName").c_str());

   TChain c4;
   c4.AddFile((fileNames[2] + "/treeName").c_str());
   c4.AddFile((fileNames[3] + "/treeName").c_str());

   c3.AddFriend(&c4);

   ROOT::TTreeProcessorMT tp2(c3);
   tp2.Process(procLambda);

   // Clean-up
   DeleteFiles(fileNames);
}

TEST(TreeProcessorMT, SetNThreads)
{
   ROOT::DisableImplicitMT();
   EXPECT_EQ(ROOT::GetImplicitMTPoolSize(), 0u);
   {
      ROOT::TTreeProcessorMT p("somefile", "sometree", 1u);
      EXPECT_EQ(ROOT::GetImplicitMTPoolSize(), 1u);
   }
   EXPECT_EQ(ROOT::GetImplicitMTPoolSize(), 0u);

   {
      ROOT::TTreeProcessorMT p({"somefile", "some_other"}, "sometree", 1u);
      EXPECT_EQ(ROOT::GetImplicitMTPoolSize(), 1u);
   }

   {
      // we need a file because in-memory trees are not supported
      // (and are detected at TTreeProcessorMT construction time)
      TFile f("treeprocmt_setnthreads.root", "recreate");
      TTree t("t", "t");
      t.Write();
      TEntryList l;
      ROOT::TTreeProcessorMT p(t, l, 1u);
      EXPECT_EQ(ROOT::GetImplicitMTPoolSize(), 1u);
      f.Close();
      gSystem->Unlink("treeprocmt_setnthreads.root");
   }

   {
      // we need a file because in-memory trees are not supported
      // (and are detected at TTreeProcessorMT construction time)
      TFile f("treeprocmt_setnthreads.root", "recreate");
      TTree t("t", "t");
      t.Write();
      ROOT::TTreeProcessorMT p(t, 1u);
      EXPECT_EQ(ROOT::GetImplicitMTPoolSize(), 1u);
      gSystem->Unlink("treeprocmt_setnthreads.root");
   }
}
