// TODO: add tests on TVec containers when TVec is available
#include <ROOT/TDFHelpers.hxx>
#include "gtest/gtest.h"
#include <vector>
using namespace ROOT::Experimental;

struct TrueFunctor {
   bool operator()() const { return true; }
};

bool trueFunction()
{
   return true;
}

TEST(TDFHelpers, Not)
{
   // Not(lambda)
   auto l = []() { return true; };
   EXPECT_EQ(TDF::Not(l)(), !l());
   // Not(functor)
   TrueFunctor t;
   auto falseFunctor = TDF::Not(t);
   EXPECT_EQ(falseFunctor(), false);
   EXPECT_EQ(TDF::Not(TrueFunctor())(), false);
   // Not(freeFunction)
   EXPECT_EQ(TDF::Not(trueFunction)(), false);
}

