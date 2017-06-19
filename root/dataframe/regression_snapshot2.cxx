#include "ROOT/TDataFrame.hxx"

int main() {
   ROOT::Experimental::TDataFrame d(1);
   *(d.Define("b", []{ return 1; }).Filter("b > 0").Count());
   return 0;
}
