#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RVec.hxx>
#include <TSystem.h>

#include <algorithm>
#include <deque>
#include <vector>

#include "gtest/gtest.h"
using namespace ROOT;
using namespace ROOT::RDF;
using namespace ROOT::VecOps;

struct TrueFunctor {
   bool operator()() const { return true; }
};

bool trueFunction()
{
   return true;
}

TEST(RDFHelpers, Not)
{
   // Not(lambda)
   auto l = []() { return true; };
   EXPECT_EQ(Not(l)(), !l());
   // Not(functor)
   TrueFunctor t;
   auto falseFunctor = Not(t);
   EXPECT_EQ(falseFunctor(), false);
   EXPECT_EQ(Not(TrueFunctor())(), false);
   // Not(freeFunction)
   EXPECT_EQ(Not(trueFunction)(), false);

   // Not+RDF
   EXPECT_EQ(1u, *RDataFrame(1).Filter(Not(Not(l))).Count());
}

TEST(RDFHelpers, PassAsVec)
{
   auto One = [] { return 1; };
   auto df = RDataFrame(1).Define("one", One).Define("_1", One);

   auto TwoOnes = [](const std::vector<int> &v) { return v.size() == 2 && v[0] == 1 && v[1] == 1; };
   EXPECT_EQ(1u, *df.Filter(PassAsVec<2, int>(TwoOnes), {"one", "_1"}).Count());
   auto TwoOnesRVec = [](const RVec<int> &v) { return v.size() == 2 && All(v == 1); };
   EXPECT_EQ(1u, *df.Filter(PassAsVec<2, int>(TwoOnesRVec), {"one", "_1"}).Count());
   auto TwoOnesDeque = [](const std::deque<int> &v) { return v.size() == 2 && v[0] == 1 && v[1] == 1; };
   EXPECT_EQ(1u, *df.Filter(PassAsVec<2, int>(TwoOnesDeque), {"one", "_1"}).Count());
}

class SaveGraphTestHelper {
private:
   RDataFrame rd1;

   bool hasLoopRun = false;

   RResultPtr<double> branch1_1;
   RResultPtr<unsigned long long> branch1_2;
   RResultPtr<double> branch2_1;
   RResultPtr<unsigned long long> branch2_2;

public:
   SaveGraphTestHelper() : rd1(8)
   {
      auto root = rd1.Define("Root_def1", []() { return 1; })
                     .Filter([](int b1) { return b1 < 2; }, {"Root_def1"})
                     .Define("Root_def2", []() { return 1; });

      auto branch1 = root.Define("Branch_1_def", []() { return 1; }); // hanging
      auto branch2 = root.Define("Branch_2_def", []() { return 1; }); // hanging

      branch1_1 = branch1.Filter([](int b1) { return b1 < 2; }, {"Branch_1_def"})
                     .Define("Branch_1_1_def", []() { return 1; })
                     .Filter("1 == Branch_1_1_def % 2")
                     .Mean("Branch_1_1_def"); // complete

      branch1_2 = branch1.Define("Branch_1_2_def", []() { return 1; })
                     .Filter([](int b1) { return b1 < 2; }, {"Branch_1_2_def"})
                     .Count(); // complete

      branch2_1 = branch2.Filter([](int b1) { return b1 < 2; }, {"Branch_2_def"})
                     .Define("Branch_2_1_def", []() { return 1; })
                     .Define("Branch_2_2_def", []() { return 1; })
                     .Filter("1 == Branch_2_1_def % 2")
                     .Max("Branch_2_1_def");

      branch2_2 = branch2.Count();
   }

   void RunLoop()
   {
      hasLoopRun = true;
      *branch2_2;
   }

   std::string GetRepresentationFromRoot()
   {
      return SaveGraph(rd1);
   }

   std::string GetRealRepresentationFromRoot()
   {
      return std::string("digraph {\n"
                         "\t9 [label=\"Mean\", style=\"filled\", fillcolor=\"") +
             (hasLoopRun ? "#baf1e5" : "#9cbbe5") +
             "\", shape=\"box\"];\n"
             "\t7 [label=\"Filter\", style=\"filled\", fillcolor=\"#c4cfd4\", shape=\"diamond\"];\n"
             "\t8 [label=\"Define\n"
             "Branch_1_1_def\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t4 [label=\"Filter\", style=\"filled\", fillcolor=\"#c4cfd4\", shape=\"diamond\"];\n"
             "\t5 [label=\"Define\n"
             "Branch_1_def\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t6 [label=\"Define\n"
             "Root_def2\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t2 [label=\"Filter\", style=\"filled\", fillcolor=\"#c4cfd4\", shape=\"diamond\"];\n"
             "\t3 [label=\"Define\n"
             "Root_def1\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t0 [label=\"8\", style=\"filled\", fillcolor=\"#e8f8fc\", shape=\"oval\"];\n"
             "\t13 [label=\"Count\", style=\"filled\", fillcolor=\"" +
             (hasLoopRun ? "#baf1e5" : "#9cbbe5") +
             "\", shape=\"box\"];\n"
             "\t11 [label=\"Filter\", style=\"filled\", fillcolor=\"#c4cfd4\", shape=\"diamond\"];\n"
             "\t12 [label=\"Define\n"
             "Branch_1_2_def\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t20 [label=\"Max\", style=\"filled\", fillcolor=\"" +
             (hasLoopRun ? "#baf1e5" : "#9cbbe5") +
             "\", shape=\"box\"];\n"
             "\t17 [label=\"Filter\", style=\"filled\", fillcolor=\"#c4cfd4\", shape=\"diamond\"];\n"
             "\t18 [label=\"Define\n"
             "Branch_2_1_def\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t19 [label=\"Define\n"
             "Branch_2_2_def\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t15 [label=\"Filter\", style=\"filled\", fillcolor=\"#c4cfd4\", shape=\"diamond\"];\n"
             "\t16 [label=\"Define\n"
             "Branch_2_def\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t22 [label=\"Count\", style=\"filled\", fillcolor=\"" +
             (hasLoopRun ? "#baf1e5" : "#9cbbe5") +
             "\", shape=\"box\"];\n"
             "\t7 -> 9;\n"
             "\t8 -> 7;\n"
             "\t4 -> 8;\n"
             "\t5 -> 4;\n"
             "\t6 -> 5;\n"
             "\t2 -> 6;\n"
             "\t3 -> 2;\n"
             "\t0 -> 3;\n"
             "\t11 -> 13;\n"
             "\t12 -> 11;\n"
             "\t5 -> 12;\n"
             "\t17 -> 20;\n"
             "\t18 -> 17;\n"
             "\t19 -> 18;\n"
             "\t15 -> 19;\n"
             "\t16 -> 15;\n"
             "\t6 -> 16;\n"
             "\t16 -> 22;\n"
             "}";
   }

   std::string GetRepresentationFromAction()
   {
      return SaveGraph(branch1_1);
   }

   std::string GetRealRepresentationFromAction()
   {
      return std::string("digraph {\n"
                         "\t9 [label=\"Mean\", style=\"filled\", fillcolor=\"") +
             (hasLoopRun ? "#baf1e5" : "#9cbbe5") +
             "\", shape=\"box\"];\n"
             "\t7 [label=\"Filter\", style=\"filled\", fillcolor=\"#c4cfd4\", shape=\"diamond\"];\n"
             "\t8 [label=\"Define\n"
             "Branch_1_1_def\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t4 [label=\"Filter\", style=\"filled\", fillcolor=\"#c4cfd4\", shape=\"diamond\"];\n"
             "\t5 [label=\"Define\n"
             "Branch_1_def\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t6 [label=\"Define\n"
             "Root_def2\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t2 [label=\"Filter\", style=\"filled\", fillcolor=\"#c4cfd4\", shape=\"diamond\"];\n"
             "\t3 [label=\"Define\n"
             "Root_def1\", style=\"filled\", fillcolor=\"#60aef3\", shape=\"oval\"];\n"
             "\t0 [label=\"8\", style=\"filled\", fillcolor=\"#e8f8fc\", shape=\"oval\"];\n"
             "\t7 -> 9;\n"
             "\t8 -> 7;\n"
             "\t4 -> 8;\n"
             "\t5 -> 4;\n"
             "\t6 -> 5;\n"
             "\t2 -> 6;\n"
             "\t3 -> 2;\n"
             "\t0 -> 3;\n"
             "}";
   }
};

TEST(RDFHelpers, SaveGraphFromRoot)
{
   SaveGraphTestHelper helper;
   EXPECT_EQ(helper.GetRepresentationFromRoot(), helper.GetRealRepresentationFromRoot());
}

TEST(RDFHelpers, SaveGraphFromAction)
{
   SaveGraphTestHelper helper;
   EXPECT_EQ(helper.GetRepresentationFromAction(), helper.GetRealRepresentationFromAction());
}

TEST(RDFHelpers, SaveGraphMultipleTimes)
{
   SaveGraphTestHelper helper;
   EXPECT_EQ(helper.GetRepresentationFromRoot(), helper.GetRealRepresentationFromRoot());
   EXPECT_EQ(helper.GetRepresentationFromAction(), helper.GetRealRepresentationFromAction());
   EXPECT_EQ(helper.GetRepresentationFromRoot(), helper.GetRealRepresentationFromRoot());
   EXPECT_EQ(helper.GetRepresentationFromAction(), helper.GetRealRepresentationFromAction());
}

TEST(RDFHelpers, SaveGraphAfterEventLoop)
{
   SaveGraphTestHelper helper;
   helper.RunLoop();
   EXPECT_EQ(helper.GetRepresentationFromRoot(), helper.GetRealRepresentationFromRoot());
   EXPECT_EQ(helper.GetRepresentationFromAction(), helper.GetRealRepresentationFromAction());
}

TEST(RDFHelpers, SaveGraphRootFromTree)
{
   TFile f("savegraphrootfromtree.root", "recreate");
   TTree t("t", "t");
   int a;
   t.Branch("a", &a);
   a = 42; // The answer to life the universe and everything
   t.Fill();
   t.Write();
   f.Close();

   static const std::string expectedGraph(
      "digraph {\n\t2 [label=\"Count\", style=\"filled\", fillcolor=\"#9cbbe5\", shape=\"box\"];\n\t0 [label=\"t\", "
      "style=\"filled\", fillcolor=\"#e8f8fc\", shape=\"oval\"];\n\t0 -> 2;\n}");

   ROOT::RDataFrame df("t", "savegraphrootfromtree.root");
   auto c = df.Count();

   auto strOut = SaveGraph(c);

   EXPECT_EQ(expectedGraph, strOut);
}

TEST(RDFHelpers, SaveGraphToFile)
{
   TFile f("savegraphtofile.root", "recreate");
   TTree t("t", "t");
   int a;
   t.Branch("a", &a);
   a = 42; // The answer to life the universe and everything
   t.Fill();
   t.Write();
   f.Close();

   static const std::string expectedGraph(
      "digraph {\n\t2 [label=\"Count\", style=\"filled\", fillcolor=\"#9cbbe5\", shape=\"box\"];\n\t0 [label=\"t\", "
      "style=\"filled\", fillcolor=\"#e8f8fc\", shape=\"oval\"];\n\t0 -> 2;\n}");

   ROOT::RDataFrame df("t", "savegraphtofile.root");
   auto c = df.Count();

   const auto outFileName = "savegraphout.root";
   SaveGraph(c, outFileName);

   std::ifstream outFile(outFileName);
   std::stringstream outString;
   outString << outFile.rdbuf();
   EXPECT_EQ(expectedGraph, outString.str());

   gSystem->Unlink(outFileName);
}
