#include "TInterpreter.h"
#include "ROOT/RDataFrame.hxx"
#include "gtest/gtest.h"

TEST(CompGraphTests, ExecOrderTwoDefines)
{
   ROOT::RDataFrame d{1};
   int i{42};
   auto graph = d.Define("x",
                         [&i]() {
                            auto ret = i;
                            i = 10;
                            return ret;
                         })
                   .Define("y", [&i]() { return i; })
                   .Graph<int, int>("x", "y");

   // Expected x=42,y=10
   EXPECT_EQ(graph->GetN(), 1);
   EXPECT_EQ(graph->GetPointX(0), 42);
   EXPECT_EQ(graph->GetPointY(0), 10);
}

TEST(CompGraphTests, ExecOrderTwoDefinesInterpreted)
{
   ROOT::RDataFrame d{1};
   gInterpreter->Declare(R"(
    namespace ROOT::Internal::RDF::Testing{
        int myInt{42};
    }
   )");
   auto graph =
      d.Define("x", "return [&i = ROOT::Internal::RDF::Testing::myInt]() { auto ret = i; i = 10; return ret; }();")
         .Define("y", "return [&i = ROOT::Internal::RDF::Testing::myInt]() { return i; }();")
         .Graph<int, int>("x", "y");

   // Expected x=42,y=10
   EXPECT_EQ(graph->GetN(), 1);
   EXPECT_EQ(graph->GetPointX(0), 42);
   EXPECT_EQ(graph->GetPointY(0), 10);
}

TEST(CompGraphTests, ExecOrderThreeDefines)
{
   struct Filler {
      int fA;
      float fB;
      std::string fC;
      void Fill(int a, float b, const std::string &c)
      {
         fA = a;
         fB = b;
         fC = c;
      }
      void Merge(const std::vector<Filler *> &)
      {
         // no-op to comply with interface for RInterface::Fill
      }
   };

   int i{};
   ROOT::RDataFrame d{1};
   auto obj = d.Define("x",
                       [&i]() {
                          auto ret{i == 0 ? 33 : 999};
                          i = 10;
                          return ret;
                       })
                 .Define("y",
                         [&i]() {
                            auto ret{i == 10 ? 42.f : 33.f};
                            i = 20;
                            return ret;
                         })
                 .Define("z",
                         [&i]() {
                            std::string ret{i == 20 ? "correct" : "wrong"};
                            i = 30;
                            return ret;
                         })
                 .Fill<int, float, std::string>(Filler{}, {"x", "y", "z"});

   EXPECT_EQ(obj->fA, 33);
   EXPECT_EQ(obj->fB, 42.f);
   EXPECT_EQ(obj->fC, "correct");
}
