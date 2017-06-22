#include "ROOT/TDataFrame.hxx"

int main() {
   ROOT::Experimental::TDataFrame d(10);
   auto c = d.Define("b", "1").Filter("b > 0").Count();
   if (*c != 10)
      return 1;
   return 0;
}
