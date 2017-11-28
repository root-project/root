#include "ROOT/TDataFrame.hxx"
#include "ROOT/TSeq.hxx"
#include "ROOT/TTrivialDS.hxx"
#include "TH1F.h"
#include "TRandom.h"
#include "TSystem.h"

#include "gtest/gtest.h"

#include <algorithm>

using namespace ROOT::Experimental;
using namespace ROOT::Experimental::TDF;

TEST(Cache, FundType)
{
   TDataFrame tdf(5);
   int i = 1;
   auto cached =
      tdf.Define("c0", [&i]() { return i++; }).Define("c1", []() { return 1.; }).Cache<int, double>({"c0", "c1"});

   auto c = cached.Count();
   auto m = cached.Min<int>("c0");
   auto v = *cached.Take<int>("c0");

   EXPECT_EQ(1, *m);
   EXPECT_EQ(5UL, *c);
   for (auto j : ROOT::TSeqI(5)) {
      EXPECT_EQ(j + 1, v[j]);
   }
}

TEST(Cache, Contiguity)
{
   TDataFrame tdf(2);
   auto f = 0.f;
   auto cached = tdf.Define("float", [&f]() { return f++; }).Cache<float>({"float"});
   int counter = 0;
   float *fPrec = nullptr;
   auto count = [&counter, &fPrec](float &ff) {
      if (1 == counter++) {
         EXPECT_EQ(1, std::distance(fPrec, &ff));
      }
      fPrec = &ff;
   };
   cached.Foreach(count, {"float"});
}

TEST(Cache, Class)
{
   TH1F h("", "h", 64, 0, 1);
   gRandom->SetSeed(1);
   h.FillRandom("gaus", 10);
   TDataFrame tdf(1);
   auto cached = tdf.Define("c0", [&h]() { return h; }).Cache<TH1F>({"c0"});

   auto c = cached.Count();
   auto d = cached.Define("Mean", [](TH1F &hh) { return hh.GetMean(); }, {"c0"})
               .Define("StdDev", [](TH1F &hh) { return hh.GetStdDev(); }, {"c0"});
   auto m = d.Max<double>("Mean");
   auto s = d.Max<double>("StdDev");

   EXPECT_EQ(h.GetMean(), *m);
   EXPECT_EQ(h.GetStdDev(), *s);
   EXPECT_EQ(1UL, *c);
}

TEST(Cache, RunTwiceOnCached)
{
   auto nevts = 10U;
   TDataFrame tdf(nevts);
   auto f = 0.f;
   auto nCalls = 0U;
   auto orig = tdf.Define("float", [&f, &nCalls]() {
      nCalls++;
      return f++;
   });

   auto cached = orig.Cache<float>({"float"});
   EXPECT_EQ(nevts, nCalls);
   auto m0 = cached.Mean<float>("float");
   EXPECT_EQ(nevts, nCalls);
   cached.Foreach([]() {});               // run the event loop
   auto m1 = cached.Mean<float>("float"); // re-run the event loop
   EXPECT_EQ(nevts, nCalls);
   EXPECT_EQ(*m0, *m1);
}

// Broken - caching a cached tdf destroys the cache of the cached.
TEST(Cache, CacheFromCache)
{
   auto nevts = 10U;
   TDataFrame tdf(nevts);
   auto f = 0.f;
   auto orig = tdf.Define("float", [&f]() { return f++; });

   auto cached = orig.Cache<float>({"float"});
   f = 0.f;
   auto recached = cached.Cache<float>({"float"});
   auto ofloat = *orig.Take<float>("float");
   auto cfloat = *cached.Take<float>("float");
   auto rcfloat = *recached.Take<float>("float");

   for (auto j : ROOT::TSeqU(nevts)) {
      EXPECT_EQ(ofloat[j], cfloat[j]);
      EXPECT_EQ(ofloat[j], rcfloat[j]);
   }
}

TEST(Cache, InternalColumnsSnapshot)
{
   TDataFrame tdf(2);
   auto f = 0.f;
   auto colName = "tdfMySecretcol_";
   auto orig = tdf.Define(colName, [&f]() { return f++; });
   auto cached = orig.Cache<float>({colName});
   auto snapshot = cached.Snapshot("t", "InternalColumnsSnapshot.root", "", {"RECREATE", ROOT::kZLIB, 0, 0, 99});
   int ret(1);
   try {
      testing::internal::CaptureStderr();
      snapshot.Mean<ULong64_t>(colName);
   } catch (const std::runtime_error &e) {
      ret = 0;
   }
   EXPECT_EQ(0, ret) << "Internal column " << colName << " has been snapshotted!";
}

TEST(Cache, CollectionColumns)
{
   TDataFrame tdf(3);
   int i = 0;
   auto d = tdf.Define("vector",
                       [&i]() {
                          std::vector<int> v(3);
                          v[0] = i;
                          v[1] = i + 1;
                          v[2] = i + 2;
                          return v;
                       })
               .Define("list",
                       [&i]() {
                          std::list<int> v;
                          for (auto j : {0, 1, 2})
                             v.emplace_back(j + i);
                          return v;
                       })
               .Define("deque",
                       [&i]() {
                          std::deque<int> v(3);
                          v[0] = i;
                          v[1] = i + 1;
                          v[2] = i + 2;
                          return v;
                       })
               .Define("blob", [&i]() { return ++i; });
   {
      auto c = d.Cache<std::vector<int>, std::list<int>, std::deque<int>>({"vector", "list", "deque"});
      auto hv = c.Histo1D<std::vector<int>>("vector");
      auto hl = c.Histo1D<std::list<int>>("list");
      auto hd = c.Histo1D<std::deque<int>>("deque");
      EXPECT_EQ(1, hv->GetMean());
      EXPECT_EQ(1, hl->GetMean());
      EXPECT_EQ(1, hd->GetMean());
   }

   // same but jitted
   auto c = d.Cache({"vector", "list", "deque"});
   auto hv = c.Histo1D<std::vector<int>>("vector");
   auto hl = c.Histo1D<std::list<int>>("list");
   auto hd = c.Histo1D<std::deque<int>>("deque");
   EXPECT_EQ(1, hv->GetMean());
   EXPECT_EQ(1, hl->GetMean());
   EXPECT_EQ(1, hd->GetMean());
}

TEST(Cache, evtCounter)
{
   TDataFrame tdf(4);
   auto c = tdf.Alias("entry", "tdfentry_")
               .Filter([](ULong64_t e) { return 0 == e % 2; }, {"entry"})
               .Cache<ULong64_t>({"entry"});
   std::vector<ULong64_t> evenE_ref{0, 2};
   auto evenE = c.Take<ULong64_t>("entry");
   for (auto i : {0, 1}) {
      EXPECT_EQ(evenE->at(i), evenE_ref[i]);
   }
   std::vector<ULong64_t> allE_ref{0, 1};
   auto allE = c.Alias("entry2", "tdfentry_").Take<ULong64_t>("entry2");
   for (auto i : {0, 1}) {
      EXPECT_EQ(allE->at(i), allE_ref[i]);
   }
}

#ifdef R__B64

TEST(Cache, Regex)
{

   TDataFrame tdf(1);
   auto d = tdf.Define("c0", []() { return 0; }).Define("c1", []() { return 1; }).Define("b0", []() { return 2; });

   auto cachedAll = d.Cache();
   auto cachedC = d.Cache("c[0,1].*");

   auto sumAll = [](int c0, int c1, int b0) { return c0 + c1 + b0; };
   auto mAll = cachedAll.Define("sum", sumAll, {"c0", "c1", "b0"}).Max<int>("sum");
   EXPECT_EQ(3, *mAll);
   auto sumC = [](int c0, int c1) { return c0 + c1; };
   auto mC = cachedC.Define("sum", sumC, {"c0", "c1"}).Max<int>("sum");
   EXPECT_EQ(1, *mC);

   // Now from source
   std::unique_ptr<TDataSource> tds(new TTrivialDS(4));
   TDataFrame tdfs(std::move(tds));
   auto cached = tdfs.Cache();
   auto m = cached.Max<ULong64_t>("col0");
   EXPECT_EQ(3UL, *m);
}

TEST(Cache, Carrays)
{
   auto treeName = "t";
   auto fileName = "CacheCarrays.root";

   {
      TFile f(fileName, "RECREATE");
      TTree t(treeName, treeName);
      float arr[4];
      t.Branch("arr", arr, "arr[4]/F");
      for (auto i : ROOT::TSeqU(4)) {
         for (auto j : ROOT::TSeqU(4)) {
            arr[j] = i + j;
         }
         t.Fill();
      }
      t.Write();
   }

   TDataFrame tdf(treeName, fileName);
   auto cache = tdf.Cache<TArrayBranch<float>>({"arr"});
   int i = 0;
   auto checkArr = [&i](std::vector<float> av) {
      auto ifloat = float(i);
      EXPECT_EQ(ifloat, av[0]);
      EXPECT_EQ(ifloat + 1, av[1]);
      EXPECT_EQ(ifloat + 2, av[2]);
      EXPECT_EQ(ifloat + 3, av[3]);
      i++;
   };
   cache.Foreach(checkArr, {"arr"});

   // now jitted
   auto cachej = tdf.Cache("arr");
   i = 0;
   cache.Foreach(checkArr, {"arr"});

   gSystem->Unlink(fileName);
}

#endif // R__B64
