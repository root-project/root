#include "ROOT/TestSupport.hxx"
#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RVec.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RResultHandle.hxx>
#include <TSystem.h>
#include <RConfigure.h>

#include <algorithm>
#include <deque>
#include <vector>
#include <string>

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

// this tests https://github.com/root-project/root/issues/8276
TEST(RDFHelpers, ReturnPassAsVec)
{
   auto returnPassAsVecLambda = [] {
      double f = 42;
      auto fn = [f](std::vector<int>) { return f; };
      return PassAsVec<1, int>(fn);
   };
   auto fn = returnPassAsVecLambda();
   EXPECT_EQ(fn(0), 42.);
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

   std::string GetRepresentationFromRoot() { return SaveGraph(rd1); }

   std::string GetRealRepresentationFromRoot()
   {
      return std::string("digraph {\n\t8 [label=<Mean") +
             (hasLoopRun ? "<BR/><FONT POINT-SIZE=\"10.0\">Already Run</FONT>" : "") +
             ">, style=\"filled\", fillcolor=\"" + (hasLoopRun ? "#e6e5e6" : "#e47c7e") +
             "\", shape=\"box\"];\n"
             "\t6 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
             "\t7 [label=<Define<BR/>Branch_1_1_def>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t3 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
             "\t4 [label=<Define<BR/>Branch_1_def>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t5 [label=<Define<BR/>Root_def2>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t1 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
             "\t2 [label=<Define<BR/>Root_def1>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t0 [label=<Empty source<BR/>Entries: 8>, style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n"
             "\t11 [label=<Count" +
             (hasLoopRun ? "<BR/><FONT POINT-SIZE=\"10.0\">Already Run</FONT>" : "") +
             ">, style=\"filled\", fillcolor=\"" + (hasLoopRun ? "#e6e5e6" : "#e47c7e") +
             "\", shape=\"box\"];\n"
             "\t9 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
             "\t10 [label=<Define<BR/>Branch_1_2_def>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t17 [label=<Max" +
             (hasLoopRun ? "<BR/><FONT POINT-SIZE=\"10.0\">Already Run</FONT>" : "") +
             ">, style=\"filled\", fillcolor=\"" + (hasLoopRun ? "#e6e5e6" : "#e47c7e") +
             "\", shape=\"box\"];\n"
             "\t14 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
             "\t15 [label=<Define<BR/>Branch_2_2_def>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t16 [label=<Define<BR/>Branch_2_1_def>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t12 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
             "\t13 [label=<Define<BR/>Branch_2_def>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t18 [label=<Count" +
             (hasLoopRun ? "<BR/><FONT POINT-SIZE=\"10.0\">Already Run</FONT>" : "") +
             ">, style=\"filled\", fillcolor=\"" + (hasLoopRun ? "#e6e5e6" : "#e47c7e") +
             "\", shape=\"box\"];\n"
             "\t6 -> 8;\n"
             "\t7 -> 6;\n"
             "\t3 -> 7;\n"
             "\t4 -> 3;\n"
             "\t5 -> 4;\n"
             "\t1 -> 5;\n"
             "\t2 -> 1;\n"
             "\t0 -> 2;\n"
             "\t9 -> 11;\n"
             "\t10 -> 9;\n"
             "\t4 -> 10;\n"
             "\t14 -> 17;\n"
             "\t15 -> 14;\n"
             "\t16 -> 15;\n"
             "\t12 -> 16;\n"
             "\t13 -> 12;\n"
             "\t5 -> 13;\n"
             "\t13 -> 18;\n}";
   }

   std::string GetRepresentationFromAction() { return SaveGraph(branch1_1); }

   std::string GetRealRepresentationFromAction()
   {
      return std::string("digraph {\n\t8 [label=<Mean") +
             (hasLoopRun ? "<BR/><FONT POINT-SIZE=\"10.0\">Already Run</FONT>" : "") +
             ">, style=\"filled\", fillcolor=\"" + (hasLoopRun ? "#e6e5e6" : "#e47c7e") +
             "\", shape=\"box\"];\n"
             "\t6 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", "
             "shape=\"hexagon\"];\n"
             "\t7 [label=<Define<BR/>Branch_1_1_def>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t3 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
             "\t4 [label=<Define<BR/>Branch_1_def>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t5 [label=<Define<BR/>Root_def2>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t1 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
             "\t2 [label=<Define<BR/>Root_def1>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
             "\t0 [label=<Empty source<BR/>Entries: 8>, style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n"
             "\t6 -> 8;\n"
             "\t7 -> 6;\n"
             "\t3 -> 7;\n"
             "\t4 -> 3;\n"
             "\t5 -> 4;\n"
             "\t1 -> 5;\n"
             "\t2 -> 1;\n"
             "\t0 -> 2;\n}";
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

TEST(RDFHelpers, SaveGraphHistograms)
{
   ROOT::RDataFrame df(10);
   int x = 0;
   auto d1 = df.Define("x", [&x]() { return x++; })
                .Filter([](int n) { return n >= 2; }, {"x"}, "MyFilt")
                .Range(3)
                .Filter([](int n) { return n >= 2; }, {"x"}, "");
   auto h0 = d1.Histo1D<int>({"h1", "h1", 10, 0, 10}, "x");
   auto d2 = df.Define("v0",
                       []() {
                          std::vector<float> v({1, 2, 3});
                          return v;
                       })
                .Define("v1",
                        []() {
                           std::vector<float> v({4, 5, 6});
                           return v;
                        })
                .Define("v2",
                        []() {
                           std::vector<float> v({7, 8, 9});
                           return v;
                        })
                .Define("w", []() { return 3; });
   auto h1 = d2.Histo1D<std::vector<float>, int>("v0", "w");
   auto h2 = d2.Histo2D<std::vector<float>, std::vector<float>, int>({"A", "B", 16, 0, 16, 16, 0, 16}, "v0", "v1", "w");
   auto h3 = d2.Histo3D<std::vector<float>, std::vector<float>, std::vector<float>, int>(
      {"C", "D", 16, 0, 16, 16, 0, 16, 16, 0, 16}, "v0", "v1", "v2", "w");

   auto strOut = SaveGraph(df);

   static const std::string expectedGraph(
      "digraph {\n"
      "\t5 [label=<TH1D<BR/>h1>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t4 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
      "\t3 [label=<Range>, style=\"filled\", fillcolor=\"#9574b4\", shape=\"diamond\"];\n"
      "\t1 [label=<MyFilt>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
      "\t2 [label=<Define<BR/>x>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
      "\t0 [label=<Empty source<BR/>Entries: 10>, style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n"
      "\t10 [label=<TH1D<BR/>v0_weighted_w>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t6 [label=<Define<BR/>w>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
      "\t7 [label=<Define<BR/>v2>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
      "\t8 [label=<Define<BR/>v1>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
      "\t9 [label=<Define<BR/>v0>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
      "\t11 [label=<TH2D<BR/>A>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t12 [label=<TH3D<BR/>C>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t4 -> 5;\n"
      "\t3 -> 4;\n"
      "\t1 -> 3;\n"
      "\t2 -> 1;\n"
      "\t0 -> 2;\n"
      "\t6 -> 10;\n"
      "\t7 -> 6;\n"
      "\t8 -> 7;\n"
      "\t9 -> 8;\n"
      "\t0 -> 9;\n"
      "\t6 -> 11;\n"
      "\t6 -> 12;\n}");

   EXPECT_EQ(expectedGraph, strOut);
}

struct Jet {
   double a, b;
};

struct CustomFiller {

   TH2D h{"", "", 10, 0, 10, 10, 0, 10};

   void Fill(const Jet &j) { h.Fill(j.a, j.b); }

   void Merge(const std::vector<CustomFiller *> &)
   {
      // unused, single-thread test
   }
};

TEST(RDFHelpers, CustomObjects)
{
   auto df = ROOT::RDataFrame(10);
   auto res = df.Define("Jet", [] { return Jet{1., 2.}; }).Fill<Jet>(CustomFiller{}, {"Jet"});

   int x = 0;
   auto d1 = df.Define("x", [&x]() { return x++; })
                .Filter([](int n) { return n >= 2; }, {"x"}, "MyFilt")
                .Range(3)
                .Filter([](int n) { return n >= 2; }, {"x"}, "");
   auto h0 = d1.Histo1D<int>({"h1", "h1", 10, 0, 10}, "x");

   auto d2 = df.Define("v0", []() { return 1; }).Define("w", []() { return 2; });
   auto h1 = d2.Histo1D<int, int>("v0", "w");

   auto strOut = SaveGraph(df);

   static const std::string expectedGraph(
      "digraph {\n"
      "\t2 [label=<Fill custom object>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t1 [label=<Define<BR/>Jet>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
      "\t0 [label=<Empty source<BR/>Entries: 10>, style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n"
      "\t7 [label=<TH1D<BR/>h1>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t6 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
      "\t5 [label=<Range>, style=\"filled\", fillcolor=\"#9574b4\", shape=\"diamond\"];\n"
      "\t3 [label=<MyFilt>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n"
      "\t4 [label=<Define<BR/>x>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
      "\t10 [label=<TH1D<BR/>v0_weighted_w>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n"
      "\t8 [label=<Define<BR/>w>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
      "\t9 [label=<Define<BR/>v0>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n"
      "\t1 -> 2;\n"
      "\t0 -> 1;\n"
      "\t6 -> 7;\n"
      "\t5 -> 6;\n"
      "\t3 -> 5;\n"
      "\t4 -> 3;\n"
      "\t0 -> 4;\n"
      "\t8 -> 10;\n"
      "\t9 -> 8;\n"
      "\t0 -> 9;\n}");

   EXPECT_EQ(expectedGraph, strOut);
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
      "digraph {\n\t1 [label=<Count>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n\t0 [label=<t>, "
      "style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n\t0 -> 1;\n}");

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
      "digraph {\n\t1 [label=<Count>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n\t0 [label=<t>, "
      "style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n\t0 -> 1;\n}");

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

// ROOT-9977
TEST(RDFHelpers, SaveGraphNoActions)
{
   auto df = ROOT::RDataFrame(1);
   auto df2 = df.Filter([] { return true; });
   const auto res = ROOT::RDF::SaveGraph(df);
   const std::string expected =
      "digraph {\n\t1 [label=<Filter>, style=\"filled\", fillcolor=\"#0f9d58\", shape=\"hexagon\"];\n\t0 [label=<Empty "
      "source<BR/>Entries: 1>, style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n\t0 -> 1;\n}";
   EXPECT_EQ(res, expected);
}

TEST(RDFHelpers, SaveGraphSharedDefines)
{
   auto One = [] { return 1; };
   ROOT::RDataFrame df(1);
   auto df2 = df.Define("shared", One);
   auto c1 = df2.Define("one", One).Count();
   auto c2 = df2.Define("two", One).Count();
   std::string graph = ROOT::RDF::SaveGraph(df);
   const std::string expected =
      "digraph {\n\t3 [label=<Count>, style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n\t1 "
      "[label=<Define<BR/>one>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n\t2 "
      "[label=<Define<BR/>shared>, style=\"filled\", fillcolor=\"#4285f4\", shape=\"ellipse\"];\n\t0 [label=<Empty "
      "source<BR/>Entries: 1>, style=\"filled\", fillcolor=\"#f4b400\", shape=\"ellipse\"];\n\t5 [label=<Count>, "
      "style=\"filled\", fillcolor=\"#e47c7e\", shape=\"box\"];\n\t4 [label=<Define<BR/>two>, style=\"filled\", "
      "fillcolor=\"#4285f4\", shape=\"ellipse\"];\n\t1 -> 3;\n\t2 -> 1;\n\t0 -> 2;\n\t4 -> 5;\n\t2 -> 4;\n}";
   EXPECT_EQ(graph, expected);
}

TEST(RDFHelpers, GraphContainers)
{
   const std::vector<double> xx = {-0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95};
   const std::vector<double> yy = {1., 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1};

   auto df = RDataFrame(1).Define("xx", [&] { return xx; }).Define("yy", [&] { return yy; });

   auto gr1 = df.Graph<std::vector<double>, std::vector<double>>("xx", "yy");

   for (size_t i = 0; i < xx.size(); ++i) {
      EXPECT_DOUBLE_EQ(gr1->GetX()[i], xx[i]);
      EXPECT_DOUBLE_EQ(gr1->GetY()[i], yy[i]);
   }
}

TEST(RDFHelpers, GraphScalars)
{
   auto df =
      RDataFrame(2).DefineSlotEntry("x", [](unsigned int, ULong64_t x) { return x; }).Define("y", [] { return .5; });

   auto gr1 = df.Graph<ULong64_t, double>("x", "y");

   for (size_t i = 0; i < 2; ++i) {
      EXPECT_EQ(gr1->GetX()[i], i);
      EXPECT_DOUBLE_EQ(gr1->GetY()[i], .5);
   }
}

TEST(RDFHelpers, GraphRunTimeErrors)
{
   const std::vector<double> xx = {-0.22}; // smaller size
   const std::vector<double> yy = {1., 2.9};

   auto df =
      RDataFrame(1).Define("xx", [&] { return xx; }).Define("yy", [&] { return yy; }).Define("x", [] { return .5; });

   auto gr1 = df.Graph<std::vector<double>, std::vector<double>>("xx", "yy"); // still no error since lazy action
   auto gr2 = df.Graph<double, std::vector<double>>("x", "yy");               // still no error since lazy action

   EXPECT_THROW(gr1.GetValue(), std::runtime_error);
   EXPECT_THROW(gr2.GetValue(), std::runtime_error);
}

TEST(RDFHelpers, GraphAsymmErrorsContainers)
{
   using Ds = std::vector<double>;
   const Ds xx = {-0.22, 0.05, 0.25, 0.35, 0.5, 0.61, 0.7, 0.85, 0.89, 0.95};
   const Ds yy = {1., 2.9, 5.6, 7.4, 9, 9.6, 8.7, 6.3, 4.5, 1};
   const Ds exl = {.05, .1, .07, .07, .04, .05, .06, .07, .08, .05};
   const Ds exh = {.02, .08, .05, .05, .03, .03, .04, .05, .06, .03};
   const Ds eyl = {.8, .7, .6, .5, .4, .4, .5, .6, .7, .8};
   const Ds eyh = {.6, .5, .4, .3, .2, .2, .3, .4, .5, .6};

   auto df = RDataFrame(1)
                .Define("xx", [&] { return xx; })
                .Define("yy", [&] { return yy; })
                .Define("exl", [&] { return exl; })
                .Define("exh", [&] { return exh; })
                .Define("eyl", [&] { return eyl; })
                .Define("eyh", [&] { return eyh; });

   auto gr1 = df.GraphAsymmErrors<Ds, Ds, Ds, Ds, Ds, Ds>("xx", "yy", "exl", "exh", "eyl", "eyh");

   for (size_t i = 0; i < xx.size(); ++i) {
      EXPECT_DOUBLE_EQ(gr1->GetX()[i], xx[i]);
      EXPECT_DOUBLE_EQ(gr1->GetY()[i], yy[i]);
      EXPECT_DOUBLE_EQ(gr1->GetEXlow()[i], exl[i]);
      EXPECT_DOUBLE_EQ(gr1->GetEXhigh()[i], exh[i]);
      EXPECT_DOUBLE_EQ(gr1->GetEYlow()[i], eyl[i]);
      EXPECT_DOUBLE_EQ(gr1->GetEYhigh()[i], eyh[i]);
   }
}

TEST(RDFHelpers, GraphAsymmErrorsScalars)
{
   auto df = RDataFrame(3)
                .DefineSlotEntry("x", [](unsigned int, ULong64_t e) { return e + 1.2; })
                .DefineSlotEntry("y", [](unsigned int, ULong64_t e) { return e + 3.4; })
                .DefineSlotEntry("exl", [](unsigned int, ULong64_t e) { return e + .5; })
                .DefineSlotEntry("exh", [](unsigned int, ULong64_t e) { return e + .2; })
                .DefineSlotEntry("eyl", [](unsigned int, ULong64_t e) { return e + .8; })
                .DefineSlotEntry("eyh", [](unsigned int, ULong64_t e) { return e + .3; });

   auto gr1 = df.GraphAsymmErrors<double, double, double, double, double, double>("x", "y", "exl", "exh", "eyl", "eyh");

   for (size_t i = 0; i < 3; ++i) {
      EXPECT_DOUBLE_EQ(gr1->GetX()[i], i + 1.2);
      EXPECT_DOUBLE_EQ(gr1->GetY()[i], i + 3.4);
      EXPECT_DOUBLE_EQ(gr1->GetEXlow()[i], i + .5);
      EXPECT_DOUBLE_EQ(gr1->GetEXhigh()[i], i + .2);
      EXPECT_DOUBLE_EQ(gr1->GetEYlow()[i], i + .8);
      EXPECT_DOUBLE_EQ(gr1->GetEYhigh()[i], i + .3);
   }
}

TEST(RDFHelpers, GraphAsymmErrorsJitted)
{
   auto df = RDataFrame(3)
                .DefineSlotEntry("x", [](unsigned int, ULong64_t e) { return e + 1.2; })
                .DefineSlotEntry("y", [](unsigned int, ULong64_t e) { return e + 3.4; })
                .DefineSlotEntry("exl", [](unsigned int, ULong64_t e) { return e + .5; })
                .DefineSlotEntry("exh", [](unsigned int, ULong64_t e) { return e + .2; })
                .DefineSlotEntry("eyl", [](unsigned int, ULong64_t e) { return e + .8; })
                .DefineSlotEntry("eyh", [](unsigned int, ULong64_t e) { return e + .3; });

   auto gr1 = df.GraphAsymmErrors("x", "y", "exl", "exh", "eyl", "eyh");

   for (size_t i = 0; i < 3; ++i) {
      EXPECT_DOUBLE_EQ(gr1->GetX()[i], i + 1.2);
      EXPECT_DOUBLE_EQ(gr1->GetY()[i], i + 3.4);
      EXPECT_DOUBLE_EQ(gr1->GetEXlow()[i], i + .5);
      EXPECT_DOUBLE_EQ(gr1->GetEXhigh()[i], i + .2);
      EXPECT_DOUBLE_EQ(gr1->GetEYlow()[i], i + .8);
      EXPECT_DOUBLE_EQ(gr1->GetEYhigh()[i], i + .3);
   }
}

TEST(RDFHelpers, GraphAsymmErrorsRunTimeErrors)
{
   using Ds = std::vector<double>;
   const Ds xx = {-0.22}; // smaller size
   const Ds yy = {1., 2.9};
   const Ds exl = {.05, .1};
   const Ds exh = {.02, .08};
   const Ds eyl = {.8, .7};
   const Ds eyh = {.6, .5};

   auto df = RDataFrame(1)
                .Define("xx", [&] { return xx; })
                .Define("yy", [&] { return yy; })
                .Define("exl", [&] { return exl; })
                .Define("exh", [&] { return exh; })
                .Define("eyl", [&] { return eyl; })
                .Define("eyh", [&] { return eyh; })
                .Define("x", [] { return 3.14; }); // scalar

   // still no error since lazy action
   auto gr1 = df.GraphAsymmErrors<Ds, Ds, Ds, Ds, Ds, Ds>("xx", "yy", "exl", "exh", "eyl", "eyh");
   auto gr2 = df.GraphAsymmErrors<Ds, Ds, Ds, Ds, Ds, Ds>("x", "yy", "exl", "exh", "eyl", "eyh");

   EXPECT_THROW(gr1.GetValue(), std::runtime_error);
   EXPECT_THROW(gr2.GetValue(), std::runtime_error);
}

TEST(RunGraphs, RunGraphs)
{
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif // R__USE_IMT

   ROOT::RDataFrame df1(3);
   auto df1a = df1.Define("x", [](ULong64_t x) { return (float)x; }, {"rdfentry_"});
   auto r1 = df1a.Sum<float>("x");
   auto r2 = df1a.Count();

   ROOT::RDataFrame df2(3);
   auto df2a = df2.Define("x", [](ULong64_t x) { return 2.f * x; }, {"rdfentry_"});
   auto r3 = df2a.Sum<float>("x");
   auto r4 = df2a.Count();

   std::vector<RResultHandle> v = {r1, r2, r3, r4};
   ROOT::RDF::RunGraphs(v);

   EXPECT_EQ(df1.GetNRuns(), 1u);
   EXPECT_EQ(df2.GetNRuns(), 1u);

   for (auto &h : v)
      EXPECT_TRUE(h.IsReady());
   EXPECT_EQ(r1.GetValue(), 3.f);
   EXPECT_EQ(r2.GetValue(), 3u);
   EXPECT_EQ(r3.GetValue(), 6.f);
   EXPECT_EQ(r4.GetValue(), 3u);
}

TEST(RunGraphs, RunGraphsWithJitting)
{
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif // R__USE_IMT

   ROOT::RDataFrame df1(3);
   auto r1 = df1.Sum("rdfentry_");
   auto r2 = df1.Count();

   ROOT::RDataFrame df2(3);
   auto df2a = df2.Define("x", "2.f * rdfentry_");
   auto r3 = df2a.Sum("x");
   auto r4 = df2a.Count();

   std::vector<RResultHandle> v = {r1, r2, r3, r4};
   ROOT::RDF::RunGraphs(v);

   EXPECT_EQ(df1.GetNRuns(), 1u);
   EXPECT_EQ(df2.GetNRuns(), 1u);

   for (auto &h : v)
      EXPECT_TRUE(h.IsReady());
   EXPECT_EQ(r1.GetValue(), 3.f);
   EXPECT_EQ(r2.GetValue(), 3u);
   EXPECT_EQ(r3.GetValue(), 6.f);
   EXPECT_EQ(r4.GetValue(), 3u);
}

TEST(RunGraphs, RunGraphsWithDisabledIMT)
{
#ifdef R__USE_IMT
   ROOT::DisableImplicitMT();
#endif // R__USE_IMT

   ROOT::RDataFrame df1(3);
   auto r1 = df1.Count();

   ROOT::RDataFrame df2(3);
   auto r2 = df2.Count();

   EXPECT_FALSE(r1.IsReady());
   EXPECT_FALSE(r2.IsReady());

   ROOT::RDF::RunGraphs({r1, r2});

   EXPECT_TRUE(r1.IsReady());
   EXPECT_TRUE(r2.IsReady());
   EXPECT_EQ(*r1, 3u);
   EXPECT_EQ(*r2, 3u);
}

TEST(RunGraphs, EmptyListOfHandles)
{
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif // R__USE_IMT

   ROOT_EXPECT_WARNING(ROOT::RDF::RunGraphs({}), "RunGraphs", "Got an empty list of handles");
}

TEST(RunGraphs, AlreadyRun)
{
#ifdef R__USE_IMT
   ROOT::EnableImplicitMT();
#endif // R__USE_IMT

   ROOT::RDataFrame df1(3);
   auto r1 = df1.Count();
   auto r2 = df1.Sum<ULong64_t>("rdfentry_");
   r1.GetValue();
   ROOT::RDataFrame df2(3);
   auto r3 = df2.Count();
   auto r4 = df2.Sum<ULong64_t>("rdfentry_");

   ROOT_EXPECT_WARNING(ROOT::RDF::RunGraphs({r1, r2, r3, r4}), "RunGraphs",
                       "Got 4 handles from which 2 link to results which are already ready.");
}
