#include "TRandom.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/TSeq.hxx"
#include "ROOT/RDF/RCutFlowReport.hxx"
#include "gtest/gtest.h"

#include <TFile.h>

TEST(RDataFrameReport, AnalyseCuts)
{
   // Full coverage :) ?
   ROOT::RDataFrame d(128);
   TRandom r(1);
   auto gen = [&r]() { return r.Gaus(0, 1); };
   auto cut0 = [](double x) { return x > 0; };
   auto cut1 = [](double x) { return x > 0.1; };
   auto cut2 = [](double x) { return x > 0.2; };
   auto colName = "col0";
   auto dd = d.Define(colName, gen)
                .Filter(cut0, {colName}, "cut0")
                .Filter(cut1, {colName}, "cut1")
                .Filter(cut2, {colName}, "cut2");

   auto repPr = dd.Report();
   testing::internal::CaptureStdout();
   repPr->Print();
   std::string output = testing::internal::GetCapturedStdout();
   auto expOut = "cut0      : pass=67         all=128        -- eff=52.34 % cumulative eff=52.34 %\n"
                 "cut1      : pass=59         all=67         -- eff=88.06 % cumulative eff=46.09 %\n"
                 "cut2      : pass=50         all=59         -- eff=84.75 % cumulative eff=39.06 %\n";

   EXPECT_STREQ(output.c_str(), expOut);

   std::vector<const char *> cutNames{"cut0", "cut1", "cut2"};
   std::vector<ULong64_t> allEvts{128, 67, 59};
   std::vector<ULong64_t> passEvts{67, 59, 50};
   std::vector<float> effs{52.34375f, 88.0597f, 84.745766f};
   unsigned int i = 0;

   for (auto &&cut : repPr) {
      EXPECT_STREQ(cut.GetName().c_str(), cutNames[i]);
      EXPECT_EQ(cut.GetAll(), allEvts[i]);
      EXPECT_EQ(cut.GetPass(), passEvts[i]);
      ASSERT_FLOAT_EQ(cut.GetEff(), effs[i]);
      i++;
   }

   auto rep = *repPr;
   std::vector<ROOT::RDF::TCutInfo> cutis{rep["cut0"], rep["cut1"], rep["cut2"]};

   for (auto j : ROOT::TSeqI(3)) {
      EXPECT_STREQ(cutis[j].GetName().c_str(), cutNames[j]);
      EXPECT_EQ(cutis[j].GetAll(), allEvts[j]);
      EXPECT_EQ(cutis[j].GetPass(), passEvts[j]);
      ASSERT_FLOAT_EQ(cutis[j].GetEff(), effs[j]);
   }

   EXPECT_ANY_THROW(rep["NonExisting"]) << "No exception thrown when trying to get a non-existing cut.\n";
   EXPECT_ANY_THROW(rep[""]) << "No exception thrown when trying to get an unnamed cut.\n";
}

TEST(RDataFrameReport, Printing)
{
   // Full coverage :) ?
   ROOT::RDataFrame d(8);
   TRandom r(1);
   r.SetSeed(1);
   auto gen = [&r]() { return r.Gaus(0, 1); };
   auto cut0 = [](double x) { return x > 0; };
   auto colName = "col0";
   auto dd = d.Define(colName, gen).Filter(cut0, {colName}, "cut0");

   auto rep0 = dd.Report();

   testing::internal::CaptureStdout();
   rep0->Print();
   std::string output0 = testing::internal::GetCapturedStdout();
   EXPECT_FALSE(output0.empty());

   auto rep1 = dd.Report();

   r.SetSeed(1); // reset the seed

   testing::internal::CaptureStdout();
   rep1->Print();
   auto output1 = testing::internal::GetCapturedStdout();
   EXPECT_STREQ(output1.c_str(), output0.c_str());
}

TEST(RDataFrameReport, ActionLazyness)
{
   ROOT::RDataFrame d(1);
   auto hasRun = false;
   auto rep = d.Define("a", []() { return 1; })
                 .Filter(
                    [&hasRun](int a) {
                       hasRun = true;
                       return a == 1;
                    },
                    {"a"})
                 .Report();
   EXPECT_FALSE(hasRun);
   *rep;
   EXPECT_TRUE(hasRun);
}

TEST(RDataFrameReport, ReadReportFromFile)
{
   class FileRAII {
   private:
      std::string fPath;

   public:
      explicit FileRAII(const std::string &path) : fPath(path) {}
      FileRAII(const FileRAII &) = delete;
      FileRAII &operator=(const FileRAII &) = delete;
      ~FileRAII() { std::remove(fPath.c_str()); }
      std::string GetPath() const { return fPath; }
   };
   std::string fileName{"RDataFrameReport_ReadReportFromFile.root"};
   FileRAII r{fileName};

   auto df = ROOT::RDataFrame(50)
                .Define("b1", [](ULong64_t entry) { return entry; }, {"rdfentry_"})
                .Define("b2", [](ULong64_t entry) { return entry * entry; }, {"rdfentry_"});

   auto cut1 = df.Filter([](ULong64_t entry) { return entry > 25; }, {"rdfentry_"}, "cut1");
   auto cut2 = df.Filter([](ULong64_t entry) { return (entry % 2) == 0; }, {"rdfentry_"}, "cut2");

   auto report = df.Report();

   {
      TFile f{fileName.c_str(), "recreate"};
      f.WriteObject(report.GetPtr(), "report");
   }

   {
      TFile f{fileName.c_str()};
      auto *repFromFile = f.Get<ROOT::RDF::RCutFlowReport>("report");

      auto compFunc = [&report](const ROOT::RDF::TCutInfo &cutInfo) {
         const auto &origCutInfo = (*report)[cutInfo.GetName()];
         EXPECT_EQ(cutInfo.GetAll(), origCutInfo.GetAll());
         EXPECT_EQ(cutInfo.GetPass(), origCutInfo.GetPass());
         EXPECT_FLOAT_EQ(cutInfo.GetEff(), origCutInfo.GetEff());
      };
      std::for_each(repFromFile->begin(), repFromFile->end(), compFunc);
   }
}
