#include <ROOT/RDataFrame.hxx>
#include <ROOT/RDFHelpers.hxx>
#include <ROOT/RIntegerSequence.hxx>
#include "gtest/gtest.h"
#include <algorithm>
#include <vector>
using namespace ROOT;
using namespace ROOT::RDF;

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

TEST(RDFHelpers, MakeVec)
{
   auto One = [] { return 1; };
   auto df = RDataFrame(1).Define("one", One).Define("_1", One);

   auto TwoOnes = [](const std::vector<int> &v) { return v.size() == 2 && v[0] == 1 && v[1] == 1; };
#if R__HAS_VARIABLE_TEMPLATES
   auto with_vec = df.Define("ones", MakeVec<2, int>::func, {"one", "_1"});
#else
   auto with_vec =
      df.Define("ones", ROOT::Internal::RDF::MakeVecHelper<std::make_index_sequence<2>, int>::func, {"one", "_1"});
#endif
   EXPECT_EQ(1u, *with_vec.Filter(TwoOnes, {"ones"}).Count());
}
