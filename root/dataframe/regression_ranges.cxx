#include "ROOT/TDataFrame.hxx"
#include <cassert>

int main() {
   auto fileName = "test_ranges.root";
   auto treeName = "rangeTree";

   // write file
   {
      ROOT::Experimental::TDataFrame d(100);
      d.Define("b", []() { static int b = 0; return b++; }).Snapshot<int>(treeName, fileName, {"b"});
   }

   // one child ending before the father -- only one stop signal must be propagated upstream
   ROOT::Experimental::TDataFrame d(treeName, fileName, {"b"});
   auto fromARange  = d.Range(10, 50).Range(10, 20).Min();                      // 20
   auto fromAFilter = d.Filter([](int b) { return b > 95; }).Range(10).Count(); // 4

   assert(*fromARange == 20);
   assert(*fromAFilter == 4);

   // child and father ending on the same entry -- only one stop signal must be propagated upstream
   ROOT::Experimental::TDataFrame d2(treeName, fileName, {"b"});
   auto two = d2.Range(2).Range(2).Count();
   auto ten = d2.Range(10).Count();

   assert(*two == 2);
   assert(*ten == 10);

   return 0;
}
