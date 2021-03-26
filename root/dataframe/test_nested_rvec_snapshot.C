#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
using namespace ROOT::VecOps;

struct TwoInts {
   int a, b;
};

#ifdef __CLING__
#pragma link C++ class ROOT::VecOps::RVec<TwoInts>+;
#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<TwoInts>>+;
#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<int>>+;
#pragma link C++ class ROOT::VecOps::RVec<ROOT::VecOps::RVec<ROOT::VecOps::RVec<int>>>+;
#endif

void test_nested_rvec_snapshot()
{
   const auto fname = "snapshot_nestedrvecs.root";

   auto df = ROOT::RDataFrame(1)
                .Define("vv",
                        [] {
                           return RVec<RVec<int>>{{1, 2}, {3, 4}};
                        })
                .Define("vvv",
                        [] {
                           return RVec<RVec<RVec<int>>>{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
                        })
                .Define("vvti", [] {
                   return RVec<RVec<TwoInts>>{{{1, 2}, {3, 4}}};
                });

   auto check = [](ROOT::RDF::RNode d) {
      d.Foreach(
         [](const RVec<RVec<int>> &vv, const RVec<RVec<RVec<int>>> &vvv, const RVec<RVec<TwoInts>> &vvti) {
            R__ASSERT(All(vv[0] == RVec<int>{1, 2}));
            R__ASSERT(All(vv[1] == RVec<int>{3, 4}));
            R__ASSERT(All(vvv[0][0] == RVec<int>{1, 2}));
            R__ASSERT(All(vvv[0][1] == RVec<int>{3, 4}));
            R__ASSERT(All(vvv[1][0] == RVec<int>{5, 6}));
            R__ASSERT(All(vvv[1][1] == RVec<int>{7, 8}));
            R__ASSERT(vvti[0][0].a == 1 && vvti[0][0].b == 2 && vvti[0][1].a == 3 && vvti[0][1].b == 4);
         },
         {"vv", "vvv", "vvti"});
   };

   // compiled
   auto out_df1 =
      df.Snapshot<RVec<RVec<int>>, RVec<RVec<RVec<int>>>, RVec<RVec<TwoInts>>>("t", fname, {"vv", "vvv", "vvti"});
   check(*out_df1);

   // jitted
   auto out_df2 = df.Snapshot("t", fname, {"vv", "vvv", "vvti"});
   check(*out_df2);
}
