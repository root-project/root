#include <ROOT/RDFHelpers.hxx>
#include "gtest/gtest.h"
#include <vector>
;

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
   EXPECT_EQ(ROOT::RDF::Not(l)(), !l());
   // Not(functor)
   TrueFunctor t;
   auto falseFunctor = ROOT::RDF::Not(t);
   EXPECT_EQ(falseFunctor(), false);
   EXPECT_EQ(ROOT::RDF::Not(TrueFunctor())(), false);
   // Not(freeFunction)
   EXPECT_EQ(ROOT::RDF::Not(trueFunction)(), false);
}

