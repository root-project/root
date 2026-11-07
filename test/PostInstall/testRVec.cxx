/*
RVec can publicly depend on VDT, so here we ensure that the VDT headers are found correctly.
*/

#include <ROOT/RVec.hxx>

#include <iostream>

#define CHECK(ARG)                      \
   if (!(ARG)) {                        \
      success = false;                  \
      std::cerr << #ARG << " failed\n"; \
   }

int main()
{
   bool success = true;

   ROOT::RVec<ROOT::RVecD> rv{{1., 2., 3., 4., 5.}, {1., 2., 3., 4., 5.}};
   auto sum = rv[0] + rv[1];
   CHECK(sum.size() == rv[0].size())
   CHECK(sum[3] == 8.)

#ifdef R__HAS_VDT
   ROOT::RVecD rv2{1., 2., 3., 4., 5.};
   auto logs = fast_log(rv2);
   CHECK(std::fabs(logs[0]) < 1.E-15)
   CHECK(std::fabs(logs[1] - std::log(2.)) < 1.E-15);
   if (!success)
      std::cout << "logs=" << logs << "\n";
#endif

   return success ? 0 : 1;
}
